import json

import pytest

from ecgcert import lineage
from ecgcert.execution.envelope import ResultEnvelope, SCHEMA_VERSION


def _valid_envelope():
    digest = "a" * 64
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
        "node_id": "node-1",
        "status": "succeeded",
        "exit_code": 0,
        "started_at": "2026-07-19T00:00:00Z",
        "finished_at": "2026-07-19T00:01:00Z",
        "commit": "b" * 40,
        "dirty": False,
        "argv": ["python", "task.py"],
        "config_sha256": digest,
        "data_sha256": digest,
        "split_sha256": digest,
        "env_sha256": digest,
        "environment_lock_sha256": digest,
        "source_sha256": digest,
        "hardware": {"cpu_count": 1},
        "seed": 0,
        "upstream_sha256": {},
        "late_control_inputs_sha256": {},
        "late_control_snapshot_sha256": digest,
        "checkpoint_sha256": {},
        "outputs_sha256": {"result.json": digest},
    }


def test_result_envelope_roundtrip(tmp_path):
    envelope = ResultEnvelope.from_dict(_valid_envelope())
    path = tmp_path / "envelope.json"
    envelope.write(path)
    assert ResultEnvelope.read(path) == envelope
    assert json.loads(path.read_text())["dirty"] is False


@pytest.mark.parametrize("field", [
    "commit", "dirty", "argv", "config_sha256", "data_sha256", "split_sha256",
    "env_sha256", "environment_lock_sha256", "source_sha256", "hardware", "seed",
    "upstream_sha256", "late_control_inputs_sha256",
    "late_control_snapshot_sha256", "checkpoint_sha256",
])
def test_result_envelope_missing_or_null_fails(field):
    value = _valid_envelope()
    value[field] = None
    with pytest.raises(ValueError, match="missing/null"):
        ResultEnvelope.from_dict(value)


def test_result_envelope_rejects_dirty_and_failed_results():
    value = _valid_envelope()
    value["dirty"] = True
    with pytest.raises(ValueError, match="dirty"):
        ResultEnvelope.from_dict(value)
    value = _valid_envelope()
    value.update(status="failed", exit_code=2)
    with pytest.raises(ValueError, match="succeeded"):
        ResultEnvelope.from_dict(value)


def test_strict_lineage_missing_null_and_checkpoint_fail():
    value = {
        "commit": "b" * 40,
        "git_dirty": False,
        "argv": ["python", "task.py"],
        "config_sha256": "a" * 64,
        "data_sha256": "a" * 64,
        "split_sha256": "a" * 64,
        "env_sha256": "a" * 64,
        "hardware": {"cpu_count": 1},
        "seed": 0,
        "upstream_sha256": {},
        "checkpoint_sha256": {},
    }
    lineage.validate_strict_lineage(value)
    with pytest.raises(ValueError, match="checkpoint"):
        lineage.validate_strict_lineage(value, require_checkpoint=True)
    value["hardware"] = None
    with pytest.raises(ValueError, match="missing/null"):
        lineage.validate_strict_lineage(value)
