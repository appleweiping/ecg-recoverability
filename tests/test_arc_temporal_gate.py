from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from ecgcert.arc_control import (
    validate_arc_waiting_bundle,
    validate_arc_waiting_report,
)
from ecgcert.arc_forward import (
    build_operator_response,
    validate_signed_review_response,
    waiting_control_evidence,
)
from ecgcert.execution.late_inputs import POLICY_ENV, write_late_control_policy
from ecgcert.stage_gates import (
    json_artifact_bytes,
    make_pending_gate,
    make_review,
    merge_review,
)
from scripts import run_arc_copilot_bridge as bridge
from scripts import wait_for_arc_control


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _waiting_stage5(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    output = tmp_path / "arc-run"
    receipt_root = tmp_path / "receipts"
    stage_dir = output / "stage-05"
    stage_dir.mkdir(parents=True)
    (stage_dir / "shortlist.jsonl").write_text(
        json.dumps({"title": "verified ECG paper", "verified": True}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        stage_dir / "decision.json",
        {
            "stage_id": "05-literature_screen",
            "run_id": "rc-temporal",
            "status": "blocked_approval",
            "decision": "block",
            "output_artifacts": ["shortlist.jsonl"],
            "error": None,
            "evidence_refs": ["stage-05/shortlist.jsonl"],
            "ts": "2026-07-19T01:00:00+00:00",
            "next_stage": 5,
        },
    )
    _write_json(
        stage_dir / "stage_health.json",
        {
            "stage_id": "05-literature_screen",
            "run_id": "rc-temporal",
            "status": "blocked_approval",
            "artifacts_count": 1,
            "error": None,
            "duration_sec": 1.0,
            "timestamp": "2026-07-19T01:00:01+00:00",
        },
    )
    waiting = {
        "stage": 5,
        "stage_name": "LITERATURE_SCREEN",
        "reason": "gate_approval",
        "since": "2026-07-19T01:00:02+00:00",
        "available_actions": ["approve", "reject", "abort"],
    }
    _write_json(output / "hitl" / "waiting.json", waiting)
    _write_json(
        output / "hitl" / "session.json",
        {
            "session_id": "session-temporal",
            "run_id": "rc-temporal",
            "state": "active",
            "mode": "co-pilot",
            "created_at": "2026-07-19T00:00:00+00:00",
            "last_activity": "2026-07-19T01:00:03+00:00",
            "waiting": waiting,
            "interventions_count": 0,
        },
    )
    _write_json(output / "checkpoint.json", {"last_completed_stage": 4})
    challenge = bridge._build_operator_challenge(
        output=output,
        waiting=waiting,
        waiting_sha256=bridge._sha256(output / "hitl" / "waiting.json"),
    )
    # Keep fixture time deterministic and inside the native 24-hour window.
    challenge["created_at"] = "2026-07-19T01:00:03+00:00"
    _write_json(output / "hitl" / "operator-challenge.v2.json", challenge)
    bundle = bridge._export_waiting_receipt(
        output=output,
        receipt_root=receipt_root,
        waiting=waiting,
        challenge=challenge,
    )
    report = validate_arc_waiting_bundle(bundle, 5)
    return output, receipt_root, waiting, report


def _signed_response(tmp_path: Path, reviewer_keys):
    output, receipt_root, waiting, report = _waiting_stage5(tmp_path)
    report_raw = json_artifact_bytes(report)
    report_sha = hashlib.sha256(report_raw).hexdigest()
    evidence = {
        "local_protocol_check": "passed",
        "official_arc_waiting": waiting_control_evidence(
            report, report_sha256=report_sha
        ),
    }
    gate = make_pending_gate(
        stage=5,
        evidence=evidence,
        eligible_for_proceed=True,
        automatic_reasons=[],
        created_at=datetime(2026, 7, 19, 1, 0, 4, tzinfo=timezone.utc),
    )
    gate_raw = json_artifact_bytes(gate)
    review = make_review(
        gate,
        gate_sha256=hashlib.sha256(gate_raw).hexdigest(),
        reviewer="author",
        decision="PROCEED",
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        repository_root=Path(__file__).resolve().parents[1],
        reviewed_at=datetime(2026, 7, 19, 1, 0, 5, tzinfo=timezone.utc),
    )
    reviewed = merge_review(
        gate,
        review,
        gate_sha256=hashlib.sha256(gate_raw).hexdigest(),
        approval_sha256=hashlib.sha256(json_artifact_bytes(review)).hexdigest(),
        public_key_path=reviewer_keys.public,
    )
    response = build_operator_response(
        waiting_report=report,
        waiting_report_sha256=report_sha,
        reviewed_gate=reviewed,
        reviewer_public_key=reviewer_keys.public,
    )
    challenge = json.loads(
        (output / "hitl" / "operator-challenge.v2.json").read_text(encoding="utf-8")
    )
    return output, receipt_root, waiting, challenge, report, reviewed, response


def test_waiting_receipt_does_not_masquerade_as_formal_handoff(tmp_path: Path) -> None:
    _output, _root, _waiting, report = _waiting_stage5(tmp_path)
    assert report["decision"] == "awaiting_signed_review"
    assert report["phase"] == "waiting"
    assert "human_approval" not in report
    validate_arc_waiting_report(report, 5)


def test_native_gate_accepts_only_the_single_signed_local_review(
    tmp_path: Path, reviewer_keys
) -> None:
    output, _root, waiting, challenge, _report, reviewed, response = _signed_response(
        tmp_path, reviewer_keys
    )
    bridge._validate_response(response, waiting, challenge)
    payload = validate_signed_review_response(
        response,
        expected_stage=5,
        expected_run_id=challenge["run_id"],
        expected_session_id=challenge["session_id"],
        expected_waiting_sha256=challenge["waiting_sha256"],
        expected_checkpoint_sha256=challenge["preapproval_checkpoint_sha256"],
        expected_nonce=challenge["nonce"],
        reviewer_public_key=reviewer_keys.public,
    )
    assert payload["reviewed_gate"]["approval_sha256"] == reviewed["approval_sha256"]
    assert payload["reviewed_gate"]["review_signature_ed25519"] == reviewed[
        "review_signature_ed25519"
    ]
    # No native intervention exists before this authenticated translation is
    # consumed, so ARC cannot cross the local scientific gate early.
    assert not (output / "hitl" / "interventions.jsonl").exists()


def test_unsigned_or_tampered_local_decision_cannot_advance_native_arc(
    tmp_path: Path, reviewer_keys
) -> None:
    _output, _root, _waiting, challenge, _report, _reviewed, response = _signed_response(
        tmp_path, reviewer_keys
    )
    unsigned = dict(response)
    unsigned["message"] = "manual approve"
    with pytest.raises(ValueError, match="payload"):
        validate_signed_review_response(
            unsigned,
            expected_stage=5,
            expected_run_id=challenge["run_id"],
            expected_session_id=challenge["session_id"],
            expected_waiting_sha256=challenge["waiting_sha256"],
            expected_checkpoint_sha256=challenge["preapproval_checkpoint_sha256"],
            expected_nonce=challenge["nonce"],
            reviewer_public_key=reviewer_keys.public,
        )


def test_restart_recovers_identical_response_and_consumption_is_one_time(
    tmp_path: Path, reviewer_keys
) -> None:
    output, _root, _waiting, challenge, report, reviewed, response = _signed_response(
        tmp_path, reviewer_keys
    )
    rebuilt = build_operator_response(
        waiting_report=report,
        waiting_report_sha256=json.loads(response["message"])[
            "waiting_report_sha256"
        ],
        reviewed_gate=reviewed,
        reviewer_public_key=reviewer_keys.public,
    )
    assert rebuilt == response
    source = tmp_path / "durable-response.json"
    source.write_bytes(json_artifact_bytes(response))
    response_sha = bridge._sha256(source)
    snapshot = bridge._snapshot_operator_response(
        output=output,
        response_path=source,
        stage=5,
        response_sha256=response_sha,
    )
    claim = bridge._claim_operator_response(
        output=output,
        response_path=snapshot,
        response=response,
        response_sha256=response_sha,
        challenge=challenge,
    )
    assert claim.is_file() and source.is_file()
    with pytest.raises(ValueError, match="replay"):
        bridge._claim_operator_response(
            output=output,
            response_path=snapshot,
            response=response,
            response_sha256=response_sha,
            challenge=challenge,
        )


def test_old_signed_response_cannot_replay_at_new_nonce(
    tmp_path: Path, reviewer_keys
) -> None:
    _output, _root, _waiting, challenge, _report, _reviewed, response = _signed_response(
        tmp_path, reviewer_keys
    )
    with pytest.raises(ValueError, match="nonce"):
        validate_signed_review_response(
            response,
            expected_stage=5,
            expected_run_id=challenge["run_id"],
            expected_session_id=challenge["session_id"],
            expected_waiting_sha256=challenge["waiting_sha256"],
            expected_checkpoint_sha256=challenge["preapproval_checkpoint_sha256"],
            expected_nonce="f" * 64,
            reviewer_public_key=reviewer_keys.public,
        )


def test_waiting_report_node_recovers_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _output, receipt_root, _waiting, expected = _waiting_stage5(tmp_path)
    report_path = tmp_path / "dag" / "stage5-waiting.json"
    arguments = [
        "wait_for_arc_control.py",
        "--phase",
        "waiting",
        "--bundle",
        str(receipt_root / "arc-stage5-waiting"),
        "--stage",
        "5",
        "--output",
        str(report_path),
        "--poll-seconds",
        "0.1",
        "--timeout-hours",
        "1",
    ]
    bundle = receipt_root / "arc-stage5-waiting"

    def install_capture_policy(attempt: str) -> None:
        capture_root = tmp_path / "captures" / attempt
        policy = capture_root / "policy.v1.json"
        write_late_control_policy(
            path=policy,
            run_id="run-1",
            node_id="arc_stage5_waiting",
            workspace=tmp_path,
            capture_root=capture_root,
            inputs=(bundle.relative_to(tmp_path).as_posix(),),
        )
        monkeypatch.setenv(POLICY_ENV, str(policy))

    install_capture_policy("initial")
    monkeypatch.setattr("sys.argv", arguments)
    wait_for_arc_control.main()
    first = report_path.read_bytes()
    assert json.loads(first) == expected
    install_capture_policy("restart")
    monkeypatch.setattr("sys.argv", arguments)
    wait_for_arc_control.main()
    assert report_path.read_bytes() == first
