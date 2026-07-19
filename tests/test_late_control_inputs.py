import json
from pathlib import Path

import pytest

from ecgcert.execution.late_inputs import (
    POLICY_ENV,
    LateControlInputError,
    capture_late_control_input,
    finalize_late_control_snapshot,
    validate_late_control_snapshot,
    write_late_control_policy,
)


SOURCE = "artifacts/gates/stage5.approval.v3.json"


def _policy(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    capture_root = tmp_path / "run" / "attempt" / "capture" / "gate"
    policy = capture_root / "policy.v1.json"
    write_late_control_policy(
        path=policy,
        run_id="run-1",
        node_id="stage5_review",
        workspace=workspace,
        capture_root=capture_root,
        inputs=(SOURCE,),
    )
    monkeypatch.setenv(POLICY_ENV, str(policy))
    return workspace, policy, tmp_path / "run" / "control-inputs" / "stage5_review"


def _write_source(workspace: Path, payload: bytes = b'{"signed":true}\n') -> Path:
    source = workspace / SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return source


def test_capture_consumes_run_owned_copy_and_seals_resume_binding(tmp_path: Path, monkeypatch):
    workspace, policy, final_root = _policy(tmp_path, monkeypatch)
    source = _write_source(workspace)

    captured = capture_late_control_input(Path(SOURCE))
    assert captured.read_bytes() == source.read_bytes()
    assert not captured.is_relative_to(workspace)

    binding = finalize_late_control_snapshot(
        policy_path=policy,
        final_root=final_root,
        expected_run_id="run-1",
        expected_node_id="stage5_review",
        expected_inputs=(SOURCE,),
    )
    replay = validate_late_control_snapshot(
        snapshot_root=final_root,
        expected_run_id="run-1",
        expected_node_id="stage5_review",
        expected_inputs=(SOURCE,),
    )
    assert replay == binding
    assert binding.inputs_sha256[SOURCE]
    assert binding.snapshot_sha256
    assert binding.artifact_bytes >= len(source.read_bytes())


def test_finalize_rejects_live_source_mutation_after_capture(tmp_path, monkeypatch):
    workspace, policy, final_root = _policy(tmp_path, monkeypatch)
    source = _write_source(workspace)
    capture_late_control_input(source)
    source.write_bytes(b'{"signed":false}\n')

    with pytest.raises(LateControlInputError, match="changed after capture"):
        finalize_late_control_snapshot(
            policy_path=policy,
            final_root=final_root,
            expected_run_id="run-1",
            expected_node_id="stage5_review",
            expected_inputs=(SOURCE,),
        )


def test_capture_rejects_symlink_and_undeclared_or_escaping_path(tmp_path, monkeypatch):
    workspace, _policy_path, _final_root = _policy(tmp_path, monkeypatch)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    source = workspace / SOURCE
    source.parent.mkdir(parents=True)
    try:
        source.symlink_to(external)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(LateControlInputError, match="symlink|link"):
        capture_late_control_input(source)
    with pytest.raises(LateControlInputError, match="undeclared"):
        capture_late_control_input("artifacts/gates/other.json")
    with pytest.raises(LateControlInputError, match="escapes"):
        capture_late_control_input(tmp_path / "outside.json")


def test_capture_rejects_directory_member_symlink(tmp_path, monkeypatch):
    workspace, _policy_path, _final_root = _policy(tmp_path, monkeypatch)
    bundle = workspace / SOURCE
    bundle.mkdir(parents=True)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    try:
        (bundle / "receipt.v1.json").symlink_to(external)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(LateControlInputError, match="symlink"):
        capture_late_control_input(bundle)


def test_resume_validation_rejects_snapshot_payload_or_manifest_tamper(tmp_path, monkeypatch):
    workspace, policy, final_root = _policy(tmp_path, monkeypatch)
    _write_source(workspace)
    capture_late_control_input(SOURCE)
    finalize_late_control_snapshot(
        policy_path=policy,
        final_root=final_root,
        expected_run_id="run-1",
        expected_node_id="stage5_review",
        expected_inputs=(SOURCE,),
    )
    payload = final_root / "payload" / "0000"
    payload.write_bytes(b"tampered")

    with pytest.raises(LateControlInputError, match="payload changed"):
        validate_late_control_snapshot(
            snapshot_root=final_root,
            expected_run_id="run-1",
            expected_node_id="stage5_review",
            expected_inputs=(SOURCE,),
        )


def test_policy_tamper_and_duplicate_capture_fail_closed(tmp_path, monkeypatch):
    workspace, policy, _final_root = _policy(tmp_path, monkeypatch)
    _write_source(workspace)
    capture_late_control_input(SOURCE)
    with pytest.raises(LateControlInputError, match="more than once"):
        capture_late_control_input(SOURCE)

    value = json.loads(policy.read_text(encoding="utf-8"))
    value["capture_root"] = str(tmp_path / "attacker")
    policy.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(LateControlInputError, match="outside its capture root"):
        capture_late_control_input(SOURCE)


def test_direct_non_dag_wait_utility_remains_usable(tmp_path, monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    source = tmp_path / "approval.json"
    source.write_text("{}", encoding="utf-8")
    assert capture_late_control_input(source) == source


def test_submission_wait_path_requires_runner_capture_policy(tmp_path, monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    source = tmp_path / "approval.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(LateControlInputError, match="capture policy is required"):
        capture_late_control_input(source, require_policy=True)
