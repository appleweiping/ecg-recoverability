import json
from pathlib import Path
import subprocess

import pytest

from ecgcert.execution import DAGRunner, ExecutionError, ExperimentManifest, ResultEnvelope


@pytest.fixture(autouse=True)
def _fast_host_fingerprints(monkeypatch):
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.environment_sha256", lambda: "a" * 64,
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [], "cuda_runtime": "unavailable"},
    )


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
    runner = DAGRunner(
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
    assert len(status["budget"]["global_reservation_sha256"]) == 64
    assert len(status["budget"]["global_settlement_sha256"]) == 64
    assert (run_dir / "budget-settlement.v1.json").is_file()
    assert (tmp_path / "runs" / "budget-ledger.v1.jsonl").is_file()
    assert not (tmp_path / "runs" / ".ecgcert-execution.lease").exists()


def test_runner_records_nonzero_exit_without_global_sentinel(tmp_path):
    repo = _repo(tmp_path, "import sys\nsys.exit(7)\n")
    runner = DAGRunner(
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
    runner = DAGRunner(
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
    runner = DAGRunner(
        repo=repo, manifest=_manifest(), profile="icassp",
        run_root=tmp_path / "runs", run_id="stale-run",
    )
    with pytest.raises(ExecutionError, match="missing declared outputs"):
        runner.run()


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
    runner = DAGRunner(
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
    runner = DAGRunner(
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
    runner = DAGRunner(
        repo=repo,
        manifest=_manifest(),
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="source-mutation-run",
    )

    with pytest.raises(ExecutionError, match="immutable source snapshot changed"):
        runner.run()
