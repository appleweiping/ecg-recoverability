import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from ecgcert.execution import DAGRunner, ExecutionError, ExperimentManifest, ResultEnvelope
from ecgcert.execution import runner as runner_module
from ecgcert.execution.runner import validate_attempt_history
from ecgcert.execution.late_inputs import validate_late_control_snapshot


@pytest.fixture(autouse=True)
def _fast_host_fingerprints(monkeypatch):
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.environment_sha256", lambda: "a" * 64,
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [], "cuda_runtime": "unavailable"},
    )
    def fake_locked_environment(**kwargs):
        lock = Path(kwargs["repo"]) / "environments" / "cpu.lock.txt"
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        return SimpleNamespace(
            lock_name="cpu",
            lock_path="environments/cpu.lock.txt",
            lock_sha256=digest,
            python_executable=str(Path(__file__).resolve()),
            as_dict=lambda: {
                "lock_name": "cpu",
                "lock_path": "environments/cpu.lock.txt",
                "lock_sha256": digest,
                "python_executable": str(Path(__file__).resolve()),
                "requirement_count": 1,
                "applicable_requirement_count": 1,
                "checked_requirement_count": 1,
                "mismatches": [],
                "ok": True,
            },
        )

    monkeypatch.setattr(
        "ecgcert.execution.runner.require_locked_environment",
        fake_locked_environment,
    )


def _runner(**kwargs):
    return DAGRunner(environment_lock="cpu", **kwargs)


def test_timeout_cleanup_kills_surviving_posix_process_group(monkeypatch):
    class ExitedLeader:
        pid = 43210

        @staticmethod
        def poll():
            return 0

    signals: list[tuple[int, int]] = []
    waits = iter((False, True))
    runner = object.__new__(DAGRunner)
    monkeypatch.setattr(runner_module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda process_group_id, sent_signal: signals.append(
            (process_group_id, sent_signal)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_posix_process_group_exists",
        lambda process_group_id: process_group_id == ExitedLeader.pid,
    )
    monkeypatch.setattr(
        runner,
        "_wait_for_posix_process_group",
        lambda process_group_id, timeout_seconds: next(waits),
    )

    runner._terminate_posix_job(ExitedLeader())

    assert signals == [
        (ExitedLeader.pid, runner_module.signal.SIGTERM),
        (ExitedLeader.pid, runner_module.signal.SIGKILL),
    ]


def _git(repo: Path, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _repo(tmp_path: Path, source: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "task.py").write_text(source, encoding="utf-8")
    (repo / "environments").mkdir()
    (repo / "environments" / "cpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8",
    )
    (repo / "environments" / "gpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "task.py", "environments")
    _git(repo, "commit", "-m", "test")
    return repo


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict({
        "schema_version": 1,
        "nodes": [{
            "id": "task",
            "profile": ["icassp", "extended", "legacy"],
            "command": ["{python}", "task.py"],
            "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
            "deps": [],
            "inputs": [],
            "outputs": ["artifacts/result.json"],
            "timeout": 10,
            "seed": 7,
        }],
    })


def test_runner_uses_isolated_workspace_and_writes_status(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('artifacts').mkdir()\n"
        "Path('artifacts/result.json').write_text('{\\\"ok\\\": true}')\n",
    )
    runner = _runner(
        repo=repo, manifest=_manifest(), profile="icassp",
        run_root=tmp_path / "runs", run_id="run-1",
    )
    run_dir = runner.run()
    assert not (repo / "artifacts").exists()
    assert (run_dir / "workspace" / "artifacts" / "result.json").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "succeeded"
    assert status["exit_code"] == 0
    assert status["nodes"]["task"]["exit_code"] == 0
    envelope = ResultEnvelope.read(run_dir / "envelopes" / "task.json")
    assert envelope.seed == 7 and envelope.outputs_sha256
    status = json.loads((run_dir / "status.json").read_text())
    assert status["budget"]["limits"]["gpu_hours"] == 500.0
    assert status["budget"]["limits"]["cpu_core_hours"] == 4000.0
    assert status["budget"]["used"]["artifact_bytes"] > 0
    assert status["schema_version"] == 3
    assert len(status["run_identity_sha256"]) == 64
    assert len(status["attempts"]) == 1
    assert len(status["attempts"][0]["settlement"]["reservation_sha256"]) == 64
    assert len(status["attempts"][0]["settlement"]["settlement_sha256"]) == 64
    assert status["control_root"] == str((tmp_path / "runs").resolve())
    assert status["budget"]["global_ledger"]["bound_event_count"] == 2
    assert status["attempts"][0]["settlement"]["ledger_event_ordinal"] == 1
    assert (run_dir / "budget-settlement.v1.json").is_file()
    assert (tmp_path / "runs" / "budget-ledger.v1.jsonl").is_file()
    assert not (tmp_path / "runs" / ".ecgcert-execution.lease").exists()


def test_runner_records_linear_npz_checkpoint_sha256(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "models = Path('artifacts/models')\n"
        "models.mkdir(parents=True)\n"
        "(models / 'low_rank.npz').write_bytes(b'linear-checkpoint')\n",
    )
    manifest = ExperimentManifest.from_dict(
        {
            "schema_version": 1,
            "nodes": [
                {
                    "id": "linear",
                    "profile": ["icassp", "extended", "legacy"],
                    "command": ["{python}", "task.py"],
                    "resource": {
                        "kind": "cpu",
                        "cpus": 1,
                        "memory_gb": 2,
                        "gpus": 0,
                    },
                    "deps": [],
                    "inputs": [],
                    "outputs": ["artifacts/models"],
                    "timeout": 10,
                    "seed": 7,
                }
            ],
        }
    )
    runner = _runner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="npz-checkpoint",
    )

    run_dir = runner.run()
    envelope = ResultEnvelope.read(run_dir / "envelopes" / "linear.json")
    checkpoint = run_dir / "workspace" / "artifacts" / "models" / "low_rank.npz"

    assert envelope.checkpoint_sha256 == {
        "artifacts/models/low_rank.npz": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
    }


def test_runner_atomically_binds_late_control_to_envelope_and_config(tmp_path):
    repo = _repo(
        tmp_path,
        "import time\n"
        "from pathlib import Path\n"
        "from ecgcert.execution.late_inputs import capture_late_control_input\n"
        "source = Path('artifacts/gates/approval.json')\n"
        "deadline = time.monotonic() + 5\n"
        "while not source.is_file():\n"
        "    if time.monotonic() >= deadline: raise RuntimeError('missing approval')\n"
        "    time.sleep(0.01)\n"
        "captured = capture_late_control_input(source)\n"
        "Path('artifacts/result.json').write_bytes(captured.read_bytes())\n",
    )
    payload = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "gate",
                "profile": ["icassp", "extended", "legacy"],
                "command": [
                    "{python}",
                    "task.py",
                    "artifacts/gates/approval.json",
                ],
                "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
                "deps": [],
                "inputs": [],
                "late_control_inputs": ["artifacts/gates/approval.json"],
                "outputs": ["artifacts/result.json"],
                "timeout": 10,
                "seed": 7,
            }
        ],
    }
    manifest = ExperimentManifest.from_dict(payload)
    runner = _runner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="late-control-run",
    )
    expected = b'{"decision":"PROCEED"}\n'

    def publish() -> None:
        target = runner.workspace / "artifacts" / "gates" / "approval.json"
        deadline = time.monotonic() + 5
        while not runner.workspace.is_dir():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(expected)
        temporary.replace(target)

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    run_dir = runner.run()
    publisher.join(timeout=1)

    node = manifest.nodes[0]
    envelope = ResultEnvelope.read(run_dir / "envelopes" / "gate.json")
    binding = validate_late_control_snapshot(
        snapshot_root=run_dir / "control-inputs" / "gate",
        expected_run_id="late-control-run",
        expected_node_id="gate",
        expected_inputs=node.late_control_inputs,
    )
    assert (run_dir / "workspace" / "artifacts" / "result.json").read_bytes() == expected
    assert envelope.late_control_inputs_sha256 == binding.inputs_sha256
    assert envelope.late_control_snapshot_sha256 == binding.snapshot_sha256
    assert envelope.config_sha256 == node.config_sha256(
        late_control_inputs_sha256=binding.inputs_sha256
    )


def test_attempt_history_requires_the_bound_external_budget_ledger(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('artifacts').mkdir()\n"
        "Path('artifacts/result.json').write_text('{}')\n",
    )
    run_root = tmp_path / "runs"
    runner = _runner(
        repo=repo,
        manifest=_manifest(),
        profile="icassp",
        run_root=run_root,
        run_id="ledger-required",
    )
    run_dir = runner.run()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    (run_root / "budget-ledger.v1.jsonl").unlink()

    with pytest.raises(ExecutionError, match="global budget ledger"):
        validate_attempt_history(
            run_dir=run_dir,
            status=status,
            limits={
                "cpu_core_hours": 4000.0,
                "gpu_hours": 500.0,
                "artifact_bytes": 100 * 1024**3,
            },
            reserved_gpu_hours=100.0,
            selected_node_ids=["task"],
            expected_control_root=run_root,
        )


def test_runner_records_nonzero_exit_without_global_sentinel(tmp_path):
    repo = _repo(tmp_path, "import sys\nsys.exit(7)\n")
    runner = _runner(
        repo=repo, manifest=_manifest(), profile="icassp",
        run_root=tmp_path / "runs", run_id="failed-run",
    )
    with pytest.raises(ExecutionError, match="exit code 7"):
        runner.run()
    status = json.loads((runner.run_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["exit_code"] == 1
    assert status["nodes"]["task"]["exit_code"] == 7
    assert not list(tmp_path.rglob("DONE*"))


def test_runner_refuses_dirty_source(tmp_path):
    repo = _repo(tmp_path, "print('ok')\n")
    (repo / "task.py").write_text("print('dirty')\n", encoding="utf-8")
    runner = _runner(
        repo=repo, manifest=_manifest(), profile="icassp",
        run_root=tmp_path / "runs", run_id="dirty-run",
    )
    with pytest.raises(ExecutionError, match="dirty source"):
        runner.run()


def test_runner_does_not_accept_stale_committed_output(tmp_path):
    repo = _repo(tmp_path, "print('does not produce an output')\n")
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "result.json").write_text('{"stale": true}', encoding="utf-8")
    _git(repo, "add", "artifacts/result.json")
    _git(repo, "commit", "-m", "stale artifact")
    runner = _runner(
        repo=repo, manifest=_manifest(), profile="icassp",
        run_root=tmp_path / "runs", run_id="stale-run",
    )
    with pytest.raises(ExecutionError, match="missing declared outputs"):
        runner.run()


def test_runner_requires_explicit_environment_lock_before_budget_reservation(tmp_path):
    repo = _repo(tmp_path, "print('ok')\n")
    run_root = tmp_path / "runs"
    runner = DAGRunner(
        repo=repo,
        manifest=_manifest(),
        profile="icassp",
        run_root=run_root,
        run_id="missing-lock",
    )

    with pytest.raises(ExecutionError, match="explicit run-level environment_lock"):
        runner.run()

    assert not run_root.exists()


def test_mixed_resource_nodes_share_one_verified_run_lock(tmp_path, monkeypatch):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('artifacts').mkdir(exist_ok=True)\n"
        "Path('artifacts/cpu.json').write_text('{}')\n",
    )
    (repo / "gpu.py").write_text(
        "from pathlib import Path\n"
        "Path('artifacts/gpu.json').write_text('{}')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "gpu.py")
    _git(repo, "commit", "-m", "add gpu fixture")
    manifest = ExperimentManifest.from_dict(
        {
            "schema_version": 1,
            "nodes": [
                {
                    "id": "cpu",
                    "profile": ["icassp", "extended", "legacy"],
                    "command": ["{python}", "task.py"],
                    "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
                    "deps": [],
                    "inputs": [],
                    "outputs": ["artifacts/cpu.json"],
                    "timeout": 10,
                    "seed": 1,
                },
                {
                    "id": "gpu",
                    "profile": ["icassp", "extended", "legacy"],
                    "command": ["{python}", "gpu.py"],
                    "resource": {"kind": "gpu", "cpus": 1, "memory_gb": 2, "gpus": 1},
                    "deps": ["cpu"],
                    "inputs": ["artifacts/cpu.json"],
                    "outputs": ["artifacts/gpu.json"],
                    "timeout": 10,
                    "seed": 2,
                },
            ],
        }
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [{"name": "fixture"}], "cuda_runtime": "12.8"},
    )
    runner = _runner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="mixed-lock",
    )

    run_dir = runner.run()
    cpu = ResultEnvelope.read(run_dir / "envelopes" / "cpu.json")
    gpu = ResultEnvelope.read(run_dir / "envelopes" / "gpu.json")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert cpu.environment_lock_sha256 == gpu.environment_lock_sha256
    assert cpu.env_sha256 == gpu.env_sha256 == status["environment_sha256"]
    assert status["environment_lock"]["lock_name"] == "cpu"


def _dependent_manifest(consumer_script: str = "consumer.py") -> ExperimentManifest:
    common = {
        "profile": ["icassp", "extended", "legacy"],
        "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
        "timeout": 10,
        "seed": 7,
    }
    return ExperimentManifest.from_dict({
        "schema_version": 1,
        "nodes": [
            {
                **common,
                "id": "producer",
                "command": ["{python}", "task.py"],
                "deps": [],
                "inputs": [],
                "outputs": ["artifacts/producer.json"],
            },
            {
                **common,
                "id": "consumer",
                "command": ["{python}", consumer_script],
                "deps": ["producer"],
                "inputs": ["artifacts/producer.json"],
                "outputs": ["artifacts/consumer.json"],
            },
        ],
    })


def test_runner_rehashes_inputs_and_rejects_downstream_in_place_mutation(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('artifacts').mkdir()\n"
        "Path('artifacts/producer.json').write_text('{\\\"producer\\\": true}')\n",
    )
    (repo / "consumer.py").write_text(
        "from pathlib import Path\n"
        "Path('artifacts/producer.json').write_text('{\\\"tampered\\\": true}')\n"
        "Path('artifacts/consumer.json').write_text('{\\\"consumer\\\": true}')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "consumer.py")
    _git(repo, "commit", "-m", "add consumer")
    runner = _runner(
        repo=repo,
        manifest=_dependent_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="input-mutation-run",
    )

    with pytest.raises(ExecutionError, match="mutated declared inputs"):
        runner.run()


def test_runner_requires_dependency_envelope_without_output_hash_fallback(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('artifacts').mkdir()\n"
        "Path('artifacts/producer.json').write_text('{\\\"producer\\\": true}')\n",
    )
    (repo / "consumer.py").write_text(
        "from pathlib import Path\n"
        "Path('../envelopes/producer.json').unlink()\n"
        "Path('artifacts/consumer.json').write_text('{\\\"consumer\\\": true}')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "consumer.py")
    _git(repo, "commit", "-m", "add consumer")
    runner = _runner(
        repo=repo,
        manifest=_dependent_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="missing-envelope-run",
    )

    with pytest.raises(ExecutionError, match="dependency envelope is missing"):
        runner.run()


def test_runner_rejects_node_source_mutation(tmp_path):
    repo = _repo(
        tmp_path,
        "from pathlib import Path\n"
        "Path('task.py').write_text(\"print('tampered')\\n\")\n"
        "Path('artifacts').mkdir()\n"
        "Path('artifacts/result.json').write_text('{}')\n",
    )
    runner = _runner(
        repo=repo,
        manifest=_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="source-mutation-run",
    )

    with pytest.raises(ExecutionError, match="immutable source snapshot changed"):
        runner.run()


def _external_input_manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict({
        "schema_version": 1,
        "nodes": [{
            "id": "task",
            "profile": ["icassp", "extended", "legacy"],
            "command": ["{python}", "task.py"],
            "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
            "deps": [],
            "inputs": ["data/ptbxl"],
            "outputs": ["artifacts/result.json"],
            "timeout": 10,
            "seed": 7,
        }],
    })


def test_stage_snapshot_links_external_directory_to_absolute_resolved_source(
    tmp_path, monkeypatch,
):
    repo = _repo(tmp_path, "print('ok')\n")
    source = repo / "data" / "ptbxl"
    source.mkdir(parents=True)
    (source / "record.dat").write_bytes(b"ecg")
    runner = _runner(
        repo=repo,
        manifest=_external_input_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="external-stage-run",
    )
    calls: list[tuple[Path, Path, bool]] = []

    def record_symlink(path, target, target_is_directory=False):
        calls.append((path, Path(target), target_is_directory))

    monkeypatch.setattr(Path, "symlink_to", record_symlink)
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    runner._stage_snapshot(commit, _external_input_manifest().nodes)

    assert calls == [
        (runner.workspace / "data" / "ptbxl", source.resolve(), True),
    ]
    assert "data/ptbxl" in runner._mutable_roots


def test_stage_snapshot_never_copies_external_directory_when_linking_fails(
    tmp_path, monkeypatch,
):
    repo = _repo(tmp_path, "print('ok')\n")
    source = repo / "data" / "ptbxl"
    source.mkdir(parents=True)
    (source / "record.dat").write_bytes(b"ecg")
    runner = _runner(
        repo=repo,
        manifest=_external_input_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="external-stage-failure",
    )

    def fail_symlink(*args, **kwargs):
        raise OSError("symlink unavailable")

    monkeypatch.setattr(Path, "symlink_to", fail_symlink)
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(ExecutionError, match="cannot stage external directory input"):
        runner._stage_snapshot(commit, _external_input_manifest().nodes)
    assert not (runner.workspace / "data" / "ptbxl" / "record.dat").exists()


def test_stage_snapshot_dereferences_repository_data_link(tmp_path):
    repo = _repo(tmp_path, "print('ok')\n")
    external = tmp_path / "persistent" / "ptbxl"
    external.mkdir(parents=True)
    (external / "record.dat").write_bytes(b"ecg")
    source = repo / "data" / "ptbxl"
    source.parent.mkdir()
    try:
        source.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable on this test host: {exc}")
    runner = _runner(
        repo=repo,
        manifest=_external_input_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="external-two-hop-stage",
    )
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    runner._stage_snapshot(commit, _external_input_manifest().nodes)

    staged = runner.workspace / "data" / "ptbxl"
    assert staged.is_symlink()
    assert staged.resolve(strict=True) == external.resolve(strict=True)
    assert Path(os.readlink(staged)) == external.resolve(strict=True)
