import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ecgcert import lineage
from ecgcert.execution import ExperimentManifest, ResultEnvelope
from ecgcert.execution.budget import BudgetLease, SETTLEMENT_SCHEMA
from ecgcert.execution.envelope import SCHEMA_VERSION
from ecgcert.execution.runner import collect_checkpoint_hashes, declared_path_hashes
from ecgcert.execution.runner import expected_mutable_roots, immutable_workspace_sha256
from scripts import release


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _manifest_payload() -> dict:
    common = {
        "profile": ["icassp", "extended", "legacy"],
        "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
        "timeout": 10,
        "seed": 11,
    }
    return {
        "schema_version": 1,
        "nodes": [
            {
                **common,
                "id": "producer",
                "command": ["{python}", "task.py"],
                "deps": [],
                "inputs": ["source.txt"],
                "outputs": ["artifacts/producer.json"],
            },
            {
                **common,
                "id": "consumer",
                "command": ["{python}", "consumer.py"],
                "deps": ["producer"],
                "inputs": ["artifacts/producer.json"],
                "outputs": ["artifacts/consumer.json"],
            },
        ],
    }


def _write_envelope(
    *,
    run_dir: Path,
    manifest: ExperimentManifest,
    node_id: str,
    commit: str,
) -> None:
    workspace = run_dir / "workspace"
    envelope_dir = run_dir / "envelopes"
    node = manifest.by_id()[node_id]
    argv = [sys.executable if token == "{python}" else token for token in node.command]
    input_hashes = declared_path_hashes(workspace, node.inputs)
    output_hashes = declared_path_hashes(workspace, node.outputs)
    checkpoints = collect_checkpoint_hashes(workspace, (*node.inputs, *node.outputs))
    data_inputs = {
        rel: digest for rel, digest in input_hashes.items() if rel not in checkpoints
    }
    upstream = {
        dep: lineage.artifact_sha256(envelope_dir / f"{dep}.json") for dep in node.deps
    }
    envelope = ResultEnvelope(
        schema_version=SCHEMA_VERSION,
        run_id=run_dir.name,
        node_id=node.id,
        status="succeeded",
        exit_code=0,
        started_at="2026-07-19T00:00:00Z",
        finished_at="2026-07-19T00:01:00Z",
        commit=commit,
        dirty=False,
        argv=argv,
        config_sha256=node.config_sha256(),
        data_sha256=lineage.canonical_sha256(data_inputs),
        split_sha256=lineage.canonical_sha256({
            "seed": node.seed,
            "argv": argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        }),
        env_sha256="a" * 64,
        environment_lock_sha256=lineage.artifact_sha256(
            workspace / "environments" / "cpu.lock.txt"
        ),
        source_sha256=immutable_workspace_sha256(
            workspace,
            expected_mutable_roots(release.ROOT, commit, manifest.select("icassp")),
        ),
        hardware={"cpu_count": 1},
        seed=node.seed,
        upstream_sha256=upstream,
        checkpoint_sha256=checkpoints,
        outputs_sha256=output_hashes,
    )
    envelope.write(envelope_dir / f"{node.id}.json")


def _release_run(tmp_path: Path, monkeypatch) -> tuple[Path, ExperimentManifest]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    payload = _manifest_payload()
    (root / "scripts" / "experiment_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
    )
    (root / "task.py").write_text("print('producer')\n", encoding="utf-8")
    (root / "consumer.py").write_text("print('consumer')\n", encoding="utf-8")
    (root / "environments").mkdir()
    (root / "environments" / "cpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8",
    )
    (root / "environments" / "gpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "release fixture")
    commit = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(release, "ROOT", root)

    manifest = ExperimentManifest.from_path(root / "scripts" / "experiment_manifest.yaml")
    run_dir = tmp_path / "runs" / "release-test"
    workspace = run_dir / "workspace"
    (workspace / "environments").mkdir(parents=True)
    for name in ("cpu.lock.txt", "gpu.lock.txt"):
        (workspace / "environments" / name).write_bytes(
            (root / "environments" / name).read_bytes()
        )
    (workspace / "task.py").write_bytes((root / "task.py").read_bytes())
    (workspace / "consumer.py").write_bytes((root / "consumer.py").read_bytes())
    (workspace / "scripts").mkdir()
    (workspace / "scripts" / "experiment_manifest.yaml").write_bytes(
        (root / "scripts" / "experiment_manifest.yaml").read_bytes()
    )
    (workspace / "artifacts").mkdir(parents=True)
    (workspace / "source.txt").write_text("frozen input\n", encoding="utf-8")
    (workspace / "artifacts" / "producer.json").write_text(
        '{"producer": true}\n', encoding="utf-8",
    )
    (workspace / "artifacts" / "consumer.json").write_text(
        '{"consumer": true}\n', encoding="utf-8",
    )
    (run_dir / "envelopes").mkdir()
    _write_envelope(
        run_dir=run_dir, manifest=manifest, node_id="producer", commit=commit,
    )
    _write_envelope(
        run_dir=run_dir, manifest=manifest, node_id="consumer", commit=commit,
    )
    report_path = workspace / "artifacts" / "paper" / "submission" / "build_report.v3.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({
            "pages": 5,
            "overfull_boxes": 0,
            "submission_ready": True,
        }),
        encoding="utf-8",
    )
    limits = {
        "cpu_core_hours": 4000.0,
        "gpu_hours": 500.0,
        "artifact_bytes": 100 * 1024**3,
    }
    budget_lease = BudgetLease(
        control_root=tmp_path / "control",
        run_id=run_dir.name,
        limits=limits,
        reserved_gpu_hours=100.0,
    )
    zero_usage = {
        "cpu_core_hours": 0.0,
        "gpu_hours": 0.0,
        "artifact_bytes": 0,
    }
    reservation = budget_lease.acquire(zero_usage)
    settlement = budget_lease.settle(zero_usage, run_state="succeeded")
    budget_lease.release()
    settlement_path = run_dir / "budget-settlement.v1.json"
    settlement_path.write_text(json.dumps({
        "schema_version": SETTLEMENT_SCHEMA,
        "reservation": reservation,
        "settlement": settlement,
    }), encoding="utf-8")
    status = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "profile": "icassp",
        "resource": None,
        "manifest_sha256": manifest.sha256(),
        "commit": commit,
        "state": "succeeded",
        "exit_code": 0,
        "nodes": {
            node.id: {"state": "succeeded", "exit_code": 0}
            for node in manifest.select("icassp")
        },
        "budget": {
            "limits": {**limits, "reserved_gpu_hours_for_rerun": 100.0},
            "used": zero_usage,
            "global_reservation_sha256": reservation["event_sha256"],
            "global_settlement_sha256": settlement["event_sha256"],
            "settlement_artifact_sha256": lineage.artifact_sha256(settlement_path),
        },
    }
    mutable_roots = expected_mutable_roots(root, commit, manifest.select("icassp"))
    status["mutable_workspace_roots"] = list(mutable_roots)
    status["source_snapshot_sha256"] = immutable_workspace_sha256(workspace, mutable_roots)
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return run_dir, manifest


def test_release_recomputes_workspace_and_accepts_consistent_lineage(tmp_path, monkeypatch):
    run_dir, _ = _release_run(tmp_path, monkeypatch)

    release._validate_run(run_dir, "icassp", allow_pending_gate=True)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("output", "workspace outputs"),
        ("input", "input-data hash"),
        ("config", "config hash"),
        ("envelope_commit", "identity mismatch"),
        ("status_commit", "current source commit"),
        ("upstream", "upstream envelope hashes"),
        ("source", "source snapshot does not match the commit"),
        ("lock_envelope", "environment lock hash"),
        ("budget", "invalid global budget settlement"),
    ],
)
def test_release_fails_closed_on_lineage_tampering(
    tmp_path,
    monkeypatch,
    tamper,
    message,
):
    run_dir, _ = _release_run(tmp_path, monkeypatch)
    workspace = run_dir / "workspace"
    envelope_dir = run_dir / "envelopes"
    if tamper == "output":
        (workspace / "artifacts" / "consumer.json").write_text(
            '{"tampered": true}\n', encoding="utf-8",
        )
    elif tamper == "input":
        (workspace / "source.txt").write_text("tampered input\n", encoding="utf-8")
    elif tamper in {"config", "envelope_commit", "upstream"}:
        node_id = "producer" if tamper != "upstream" else "consumer"
        path = envelope_dir / f"{node_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "config":
            value["config_sha256"] = "f" * 64
        elif tamper == "envelope_commit":
            value["commit"] = "e" * 40
        else:
            value["upstream_sha256"] = {"producer": "f" * 64}
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "status_commit":
        path = run_dir / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["commit"] = "e" * 40
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "source":
        (workspace / "task.py").write_text("print('source tamper')\n", encoding="utf-8")
    elif tamper == "lock_envelope":
        path = envelope_dir / "producer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["environment_lock_sha256"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "budget":
        path = run_dir / "budget-settlement.v1.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["settlement"]["used"]["cpu_core_hours"] = 10.0
        path.write_text(json.dumps(value), encoding="utf-8")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(tamper)

    with pytest.raises(SystemExit, match=message):
        release._validate_run(run_dir, "icassp", allow_pending_gate=True)
