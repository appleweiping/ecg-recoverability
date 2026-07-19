"""Authenticated forwarding of one signed scientific review to native ARC HITL."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from ecgcert.arc_control import validate_arc_waiting_report
from ecgcert.stage_gates import DEFAULT_REVIEWER_PUBLIC_KEY, validate_reviewed_gate


FORWARD_SCHEMA = "arc-signed-review-forward-v1"
OPERATOR_RESPONSE_SCHEMA = "arc-operator-response-v2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_message(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def waiting_control_evidence(
    waiting_report: Mapping[str, Any], *, report_sha256: str
) -> dict[str, Any]:
    """Return the exact pre-approval control fields frozen into a local gate."""

    stage = waiting_report.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError("waiting report stage is invalid")
    report = validate_arc_waiting_report(waiting_report, stage)
    if not isinstance(report_sha256, str) or not _HEX64.fullmatch(report_sha256):
        raise ValueError("waiting report SHA-256 must be a full lowercase digest")
    predecessor = report["waiting_lineage"]["predecessor"]
    return {
        "phase": "waiting",
        "report_sha256": report_sha256,
        "waiting_receipt_sha256": report["waiting_receipt_sha256"],
        "run_id": report["run_id"],
        "session_id": report["session_id"],
        "stage": report["stage"],
        "decision": report["decision"],
        "waiting_sha256": report["waiting"]["sha256"],
        "waiting_since": report["waiting"]["since"],
        "expires_at": report["waiting"]["expires_at"],
        "challenge_sha256": report["challenge"]["sha256"],
        "preapproval_checkpoint_sha256": report["challenge"][
            "preapproval_checkpoint_sha256"
        ],
        "nonce": report["challenge"]["nonce"],
        "waiting_chain_sha256": report["waiting_lineage"]["chain_sha256"],
        "predecessor": predecessor,
        "stage_output_sha256": report["stage_output_sha256"],
    }


def _review_status_allowed(stage: int, status: Any) -> bool:
    if stage == 15:
        return status in {"PROCEED", "PIVOT"}
    return status == "PROCEED"


def build_operator_response(
    *,
    waiting_report: Mapping[str, Any],
    waiting_report_sha256: str,
    reviewed_gate: Mapping[str, Any],
    reviewer_public_key: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> dict[str, Any]:
    """Translate the one signed author decision into ARC's native approve input."""

    stage = waiting_report.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError("waiting report stage is invalid")
    report = validate_arc_waiting_report(waiting_report, stage)
    validate_reviewed_gate(reviewed_gate, public_key_path=reviewer_public_key)
    if reviewed_gate.get("stage") != stage:
        raise ValueError("reviewed scientific gate does not match the ARC waiting stage")
    if not _review_status_allowed(stage, reviewed_gate.get("status")):
        raise ValueError(
            f"Stage-{stage} review {reviewed_gate.get('status')!r} cannot advance ARC"
        )
    evidence = reviewed_gate.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("reviewed scientific gate lacks evidence")
    expected_control = waiting_control_evidence(
        report, report_sha256=waiting_report_sha256
    )
    if evidence.get("official_arc_waiting") != expected_control:
        raise ValueError("reviewed gate is not bound to this exact ARC waiting receipt")
    forward = {
        "schema_version": FORWARD_SCHEMA,
        "waiting_report_sha256": waiting_report_sha256,
        "waiting_report": report,
        "reviewed_gate": dict(reviewed_gate),
    }
    return {
        "schema_version": OPERATOR_RESPONSE_SCHEMA,
        "stage": stage,
        "run_id": report["run_id"],
        "session_id": report["session_id"],
        "waiting_sha256": report["waiting"]["sha256"],
        "preapproval_checkpoint_sha256": report["challenge"][
            "preapproval_checkpoint_sha256"
        ],
        "nonce": report["challenge"]["nonce"],
        "action": "approve",
        "issued_at": reviewed_gate["reviewed_at"],
        "message": _canonical_message(forward),
    }


def validate_signed_review_response(
    response: Mapping[str, Any],
    *,
    expected_stage: int,
    expected_run_id: str,
    expected_session_id: str,
    expected_waiting_sha256: str,
    expected_checkpoint_sha256: str,
    expected_nonce: str,
    reviewer_public_key: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> dict[str, Any]:
    """Verify that a native response is a deterministic signed-review forward."""

    message = response.get("message")
    if not isinstance(message, str) or not message:
        raise ValueError("operator response lacks its signed-review forward payload")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as error:
        raise ValueError("operator response forward payload is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "waiting_report_sha256",
        "waiting_report",
        "reviewed_gate",
    }:
        raise ValueError("operator response forward payload fields are invalid")
    if payload.get("schema_version") != FORWARD_SCHEMA:
        raise ValueError(f"operator response forward schema must be {FORWARD_SCHEMA}")
    if message != _canonical_message(payload):
        raise ValueError("operator response forward payload is not canonical JSON")
    rebuilt = build_operator_response(
        waiting_report=payload["waiting_report"],
        waiting_report_sha256=payload["waiting_report_sha256"],
        reviewed_gate=payload["reviewed_gate"],
        reviewer_public_key=reviewer_public_key,
    )
    if dict(response) != rebuilt:
        raise ValueError("operator response differs from the signed review translation")
    expected = {
        "stage": expected_stage,
        "run_id": expected_run_id,
        "session_id": expected_session_id,
        "waiting_sha256": expected_waiting_sha256,
        "preapproval_checkpoint_sha256": expected_checkpoint_sha256,
        "nonce": expected_nonce,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(f"signed review forward {field} does not match active ARC")
    return payload


__all__ = [
    "FORWARD_SCHEMA",
    "build_operator_response",
    "validate_signed_review_response",
    "waiting_control_evidence",
]
