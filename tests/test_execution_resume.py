import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ecgcert.execution import DAGRunner, ExecutionError, ExperimentManifest
from ecgcert.execution.runner import validate_attempt_history


@pytest.fixture(autouse=True)
def _stable_lineage(monkeypatch):
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.environment_sha256", lambda: "a" * 64
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [], "cuda_runtime": "unavailable"},
    )

    def locked_environment(**kwargs):
        lock = Path(kwargs["repo"]) / "environments" / "cpu.lock.txt"
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        record = {
            "lock_name": "cpu",
            "lock_path": "environments/cpu.lock.txt",
            "lock_sha256": digest,
            "python_executable": str(Path(sys.executable).resolve()),
            "requirement_count": 1,
            "applicable_requirement_count": 1,
            "checked_requirement_count": 1,
            "mismatches": [],
            "ok": True,
        }
        return SimpleNamespace(
            lock_name="cpu",
            lock_path="environments/cpu.lock.txt",
            lock_sha256=digest,
            python_executable=str(Path(sys.executable).resolve()),
            as_dict=lambda: record,
        )

    monkeypatch.setattr(
        "ecgcert.execution.runner.require_locked_environment", locked_environment
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _manifest(*, train_seed: int = 7) -> ExperimentManifest:
    common = {
        "profile": ["icassp", "extended", "legacy"],
        "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
        "timeout": 10,
    }
    return ExperimentManifest.from_dict(
        {
            "schema_version": 1,
            "nodes": [
                {
                    **common,
                    "id": "train",
                    "command": ["{python}", "train.py"],
                    "deps": [],
                    "inputs": [],
                    "outputs": ["artifacts/train"],
                    "seed": train_seed,
                },
                {
                    **common,
                    "id": "stage15_review",
                    "command": ["{python}", "gate.py"],
                    "deps": ["train"],
                    "inputs": ["artifacts/train"],
                    "outputs": ["artifacts/control/stage15_review"],
                    "seed": 8,
                },
                {
                    **common,
                    "id": "finish",
                    "command": ["{python}", "finish.py"],
                    "deps": ["stage15_review"],
                    "inputs": ["artifacts/control/stage15_review"],
                    "outputs": ["artifacts/final.json"],
                    "seed": 9,
                },
            ],
        }
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "environments").mkdir()
    (repo / "environments" / "cpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8"
    )
    (repo / "train.py").write_text(
        "from pathlib import Path\n"
        "out = Path('artifacts/train')\n"
        "out.mkdir(parents=True)\n"
        "(out / 'model.ckpt').write_bytes(b'expensive-trained-checkpoint')\n"
        "(out / 'metrics.json').write_text('{\"trained\": true}\\n')\n",
        encoding="utf-8",
    )
    (repo / "gate.py").write_text(
        "from pathlib import Path\n"
        "approval = Path('artifacts/gates/stage15.approval.json')\n"
        "out = Path('artifacts/control/stage15_review')\n"
        "out.mkdir(parents=True)\n"
        "if not approval.is_file():\n"
        "    (out / 'partial-timeout.txt').write_text('not reviewed')\n"
        "    raise TimeoutError('Stage 15 review window expired')\n"
        "(out / 'decision.json').write_text(approval.read_text())\n",
        encoding="utf-8",
    )
    (repo / "finish.py").write_text(
        "from pathlib import Path\n"
        "decision = Path('artifacts/control/stage15_review/decision.json').read_text()\n"
        "Path('artifacts/final.json').write_text(decision)\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "resume fixture")
    return repo


def _failed_gate_run(tmp_path: Path) -> tuple[Path, Path, ExperimentManifest]:
    repo = _repo(tmp_path)
    manifest = _manifest()
    run_root = tmp_path / "runs"
    runner = DAGRunner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=run_root,
        run_id="review-run",
        environment_lock="cpu",
    )
    with pytest.raises(ExecutionError, match="command failed"):
        runner.run()
    return repo, runner.run_dir, manifest


def _resume(repo: Path, run_dir: Path, manifest: ExperimentManifest) -> Path:
    runner = DAGRunner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=run_dir.parent,
        run_id=run_dir.name,
        environment_lock="cpu",
        resume=True,
    )
    return runner.run()


def test_late_gate_inbox_resumes_without_retraining_and_keeps_attempt_evidence(tmp_path):
    repo, run_dir, manifest = _failed_gate_run(tmp_path)
    workspace = run_dir / "workspace"
    before_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    checkpoint = workspace / "artifacts" / "train" / "model.ckpt"
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    train_envelope = (run_dir / "envelopes" / "train.json").read_bytes()
    initial_train_log = run_dir / "logs" / "train.stdout.log"
    partial = (
        workspace
        / "artifacts"
        / "control"
        / "stage15_review"
        / "partial-timeout.txt"
    )
    assert partial.is_file()
    assert {path.stem for path in (run_dir / "envelopes").glob("*.json")} == {"train"}

    inbox = workspace / "artifacts" / "gates" / "stage15.approval.json"
    inbox.parent.mkdir(parents=True)
    inbox.write_text('{"decision": "PROCEED"}\n', encoding="utf-8")
    assert _resume(repo, run_dir, manifest) == run_dir

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    for field in (
        "commit",
        "manifest_sha256",
        "environment_lock",
        "environment_sha256",
        "python_executable",
        "source_snapshot_sha256",
        "mutable_workspace_roots",
        "run_identity_sha256",
    ):
        assert status[field] == before_status[field]
    assert status["python_executable"] == str(Path(sys.executable).resolve())
    for envelope_path in (run_dir / "envelopes").glob("*.json"):
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert envelope["argv"][0] == status["python_executable"]
        assert Path(envelope["argv"][0]).is_absolute()
    assert [attempt["budget_run_id"] for attempt in status["attempts"]] == [
        "review-run",
        "review-run.resume-1",
    ]
    assert status["attempts"][1]["selected_nodes"] == ["stage15_review", "finish"]
    assert "train" not in status["attempts"][1]["node_results"]
    assert not (run_dir / "attempts" / "resume-1" / "logs" / "train.stdout.log").exists()
    assert initial_train_log.is_file()
    assert (run_dir / "envelopes" / "train.json").read_bytes() == train_envelope
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha256
    assert not partial.exists()
    assert (workspace / "artifacts" / "final.json").is_file()
    assert (run_dir / "budget-settlement.v1.json").is_file()
    assert (
        run_dir / "attempts" / "resume-1" / "budget-settlement.v1.json"
    ).is_file()
    ledger = [
        json.loads(line)
        for line in (run_dir.parent / "budget-ledger.v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["run_id"] for event in ledger] == [
        "review-run",
        "review-run",
        "review-run.resume-1",
        "review-run.resume-1",
    ]
    validated = validate_attempt_history(
        run_dir=run_dir,
        status=status,
        limits={
            "cpu_core_hours": 4000.0,
            "gpu_hours": 500.0,
            "artifact_bytes": 100 * 1024**3,
        },
        reserved_gpu_hours=100.0,
        selected_node_ids=[node.id for node in manifest.select("icassp")],
    )
    assert validated["attempt_count"] == 2
    assert validated["used"] == status["budget"]["used"]
    assert status["budget"]["used"] == {
        "cpu_core_hours": round(
            sum(attempt["used"]["cpu_core_hours"] for attempt in status["attempts"]),
            6,
        ),
        "gpu_hours": round(
            sum(attempt["used"]["gpu_hours"] for attempt in status["attempts"]),
            6,
        ),
        "artifact_bytes": sum(
            attempt["used"]["artifact_bytes"] for attempt in status["attempts"]
        ),
    }
    without_initial = {**status, "attempts": status["attempts"][1:]}
    with pytest.raises(ExecutionError, match="attempt 0 identity"):
        validate_attempt_history(
            run_dir=run_dir,
            status=without_initial,
            limits={
                "cpu_core_hours": 4000.0,
                "gpu_hours": 500.0,
                "artifact_bytes": 100 * 1024**3,
            },
            reserved_gpu_hours=100.0,
            selected_node_ids=[node.id for node in manifest.select("icassp")],
        )


def test_resume_refuses_an_existing_active_or_stale_lease(tmp_path):
    repo, run_dir, manifest = _failed_gate_run(tmp_path)
    lease = run_dir.parent / ".ecgcert-execution.lease"
    lease.mkdir()
    (lease / "owner.json").write_text(
        '{"run_id": "possibly-active", "token": "unknown"}\n', encoding="utf-8"
    )
    ledger = run_dir.parent / "budget-ledger.v1.jsonl"
    ledger_before = ledger.read_bytes()

    with pytest.raises(ExecutionError, match="already held"):
        _resume(repo, run_dir, manifest)

    assert ledger.read_bytes() == ledger_before
    assert not (run_dir / "attempts").exists()
    assert lease.is_dir()


@pytest.mark.parametrize(
    "tamper",
    [
        "output",
        "checkpoint",
        "envelope",
        "extra_failed_envelope",
        "source",
        "environment",
        "manifest",
        "commit",
        "lock",
        "argv",
        "active_attempt",
    ],
)
def test_resume_tampering_fails_before_lease_or_execution(
    tmp_path, monkeypatch, tamper
):
    repo, run_dir, manifest = _failed_gate_run(tmp_path)
    workspace = run_dir / "workspace"
    partial = (
        workspace
        / "artifacts"
        / "control"
        / "stage15_review"
        / "partial-timeout.txt"
    )
    ledger = run_dir.parent / "budget-ledger.v1.jsonl"
    ledger_before = ledger.read_bytes()
    partial_before = partial.read_bytes()
    if tamper == "output":
        (workspace / "artifacts" / "train" / "metrics.json").write_text(
            '{"tampered": true}\n', encoding="utf-8"
        )
    elif tamper == "checkpoint":
        (workspace / "artifacts" / "train" / "model.ckpt").write_bytes(b"tampered")
    elif tamper in {"envelope", "argv"}:
        path = run_dir / "envelopes" / "train.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "envelope":
            value["config_sha256"] = "f" * 64
        else:
            value["argv"] = [str(Path(sys.executable).resolve()), "other.py"]
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "extra_failed_envelope":
        source = run_dir / "envelopes" / "train.json"
        target = run_dir / "envelopes" / "stage15_review.json"
        target.write_bytes(source.read_bytes())
    elif tamper == "source":
        (workspace / "train.py").write_text("print('tampered')\n", encoding="utf-8")
    elif tamper == "environment":
        monkeypatch.setattr(
            "ecgcert.execution.runner.lineage.environment_sha256", lambda: "b" * 64
        )
    elif tamper == "manifest":
        manifest = _manifest(train_seed=99)
    elif tamper == "commit":
        (repo / "new-source.txt").write_text("new commit\n", encoding="utf-8")
        _git(repo, "add", "new-source.txt")
        _git(repo, "commit", "-m", "different clean commit")
    elif tamper in {"lock", "active_attempt"}:
        path = run_dir / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "lock":
            value["environment_lock"]["lock_sha256"] = "f" * 64
        else:
            value["active_attempt"] = {"state": "running"}
        path.write_text(json.dumps(value), encoding="utf-8")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(tamper)

    with pytest.raises(ExecutionError):
        _resume(repo, run_dir, manifest)

    assert ledger.read_bytes() == ledger_before
    assert partial.read_bytes() == partial_before
    assert not (run_dir / "attempts").exists()
