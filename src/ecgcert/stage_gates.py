"""Shared fail-closed contracts for ARC human evidence gates.

The automatic gate artifact freezes evidence and records whether the hard criteria
permit a human ``PROCEED`` decision.  It never approves itself.  Human review is a
separate, signed JSON artifact bound to the exact gate bytes.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ecgcert import lineage


SUPPORTED_STAGES = (5, 9, 15, 20)
REVIEW_SCHEMA = "arc-stage-review-ed25519-v1"
GATE_SCHEMAS = {stage: f"arc-stage{stage}-v3" for stage in SUPPORTED_STAGES}
DECISIONS = ("PROCEED", "REFINE", "PIVOT")
SIGNATURE_ALGORITHM = "Ed25519"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWER_PUBLIC_KEY = PROJECT_ROOT / "security" / "reviewer_ed25519.pub"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _full_sha256(value: Any, *, field: str) -> str:
    rendered = str(value)
    if not _HEX64.fullmatch(rendered):
        raise ValueError(f"{field} must be a full lowercase SHA-256")
    return rendered


def json_artifact_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the only accepted encoding for gate/review JSON artifacts."""
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def json_artifact_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json_artifact_bytes(value)).hexdigest()


def _signature_message(value: Mapping[str, Any]) -> bytes:
    """Canonical, unambiguous bytes covered by the Ed25519 signature."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_public_key(path: Path | str) -> Ed25519PublicKey:
    key_path = Path(path).expanduser().resolve(strict=True)
    raw = key_path.read_bytes()
    try:
        key = serialization.load_ssh_public_key(raw)
    except ValueError:
        key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("reviewer public key must be an Ed25519 key")
    return key


def _load_private_key(
    path: Path | str,
    *,
    repository_root: Path | str,
    password: bytes | None,
) -> Ed25519PrivateKey:
    key_path = Path(path).expanduser().resolve(strict=True)
    root = Path(repository_root).resolve(strict=True)
    if key_path == root or key_path.is_relative_to(root):
        raise ValueError("reviewer private key must be stored outside the repository")
    raw = key_path.read_bytes()
    try:
        key = serialization.load_ssh_private_key(raw, password=password)
    except ValueError:
        key = serialization.load_pem_private_key(raw, password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("reviewer private key must be an Ed25519 key")
    return key


def public_key_sha256(key: Ed25519PublicKey) -> str:
    """Fingerprint the raw RFC 8032 public key bytes with SHA-256."""
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def reviewer_public_key_sha256(
    public_key_path: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> str:
    return public_key_sha256(_load_public_key(public_key_path))


def make_pending_gate(
    *,
    stage: int,
    evidence: Mapping[str, Any],
    eligible_for_proceed: bool,
    automatic_reasons: list[str] | tuple[str, ...],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create an unreviewed gate; an eligible gate is still only pending."""
    if stage not in (5, 9, 20):
        raise ValueError("make_pending_gate creates Stage 5, 9, or 20 gates")
    raw_timestamp = created_at or utc_now()
    if raw_timestamp.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    timestamp = raw_timestamp.astimezone(timezone.utc)
    frozen_evidence = dict(evidence)
    return {
        "schema_version": GATE_SCHEMAS[stage],
        "stage": stage,
        "status": "PENDING_USER_REVIEW",
        "eligible_for_proceed": bool(eligible_for_proceed),
        "human_review_required": True,
        "review_deadline_hours": 24,
        "created_at": timestamp.isoformat(timespec="seconds"),
        "automatic_reasons": list(automatic_reasons),
        "evidence_sha256": lineage.canonical_sha256(frozen_evidence),
        "evidence": frozen_evidence,
    }


def validate_gate(gate: Mapping[str, Any]) -> int:
    """Validate a generated or reviewed gate and return its ARC stage."""
    if not isinstance(gate, Mapping):
        raise ValueError("gate must be an object")
    stage = gate.get("stage")
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"unsupported ARC gate stage: {stage!r}")
    if gate.get("schema_version") != GATE_SCHEMAS[stage]:
        raise ValueError(f"invalid Stage-{stage} gate schema")
    if gate.get("status") not in {
        "PENDING_USER_REVIEW", "PROCEED", "REFINE", "PIVOT"
    }:
        raise ValueError(f"invalid Stage-{stage} gate status")
    if not isinstance(gate.get("eligible_for_proceed"), bool):
        raise ValueError("eligible_for_proceed must be boolean")
    deadline_hours = gate.get("review_deadline_hours")
    if isinstance(deadline_hours, bool) or not isinstance(deadline_hours, (int, float)):
        raise ValueError("review_deadline_hours must be numeric")
    if not 0 < float(deadline_hours) <= 24:
        raise ValueError("review_deadline_hours must be in (0, 24]")
    _parse_time(gate.get("created_at"), field="created_at")
    if stage == 15:
        _full_sha256(gate.get("meta_analysis_sha256"), field="meta_analysis_sha256")
        digest = _full_sha256(gate.get("evidence_sha256"), field="evidence_sha256")
        if "evidence" not in gate or not isinstance(gate["evidence"], Mapping):
            raise ValueError("gate evidence must be an object")
        if digest != lineage.canonical_sha256(gate["evidence"]):
            raise ValueError("gate evidence_sha256 does not match the frozen evidence")
    else:
        digest = _full_sha256(gate.get("evidence_sha256"), field="evidence_sha256")
        if "evidence" not in gate or not isinstance(gate["evidence"], Mapping):
            raise ValueError("gate evidence must be an object")
        if digest != lineage.canonical_sha256(gate["evidence"]):
            raise ValueError("gate evidence_sha256 does not match the frozen evidence")
    return int(stage)


def review_deadline(gate: Mapping[str, Any]) -> datetime:
    validate_gate(gate)
    created = _parse_time(gate["created_at"], field="created_at")
    return created + timedelta(hours=float(gate["review_deadline_hours"]))


def review_binding(gate: Mapping[str, Any]) -> dict[str, str]:
    """Return the stage-specific immutable evidence field signed by a reviewer."""
    stage = validate_gate(gate)
    if stage == 15:
        return {
            "meta_analysis_sha256": str(gate["meta_analysis_sha256"]),
            "evidence_sha256": str(gate["evidence_sha256"]),
        }
    return {"evidence_sha256": str(gate["evidence_sha256"])}


def make_review(
    gate: Mapping[str, Any],
    *,
    gate_sha256: str,
    reviewer: str,
    decision: str,
    private_key_path: Path | str,
    public_key_path: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
    repository_root: Path | str = PROJECT_ROOT,
    private_key_password: bytes | None = None,
    reviewed_at: datetime | None = None,
) -> dict[str, Any]:
    """Create an Ed25519-signed review bound to the immutable gate artifact."""
    stage = validate_gate(gate)
    if gate.get("status") not in {"PENDING_USER_REVIEW", "PIVOT"}:
        raise ValueError(f"gate is not reviewable from status {gate.get('status')!r}")
    if decision not in DECISIONS:
        raise ValueError(f"unsupported review decision: {decision!r}")
    if decision == "PROCEED" and gate.get("eligible_for_proceed") is not True:
        raise ValueError("automatic hard criteria do not permit PROCEED")
    clean_reviewer = reviewer.strip()
    if not clean_reviewer:
        raise ValueError("reviewer cannot be empty")
    raw_when = reviewed_at or utc_now()
    if raw_when.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    when = raw_when.astimezone(timezone.utc)
    created = _parse_time(gate["created_at"], field="created_at")
    if when < created:
        raise ValueError("review timestamp precedes gate creation")
    if when > review_deadline(gate):
        raise TimeoutError(f"Stage-{stage} review deadline has expired")
    artifact_sha256 = _full_sha256(gate_sha256, field="gate_sha256")
    public_key = _load_public_key(public_key_path)
    private_key = _load_private_key(
        private_key_path,
        repository_root=repository_root,
        password=private_key_password,
    )
    derived_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pinned_public = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if derived_public != pinned_public:
        raise ValueError("reviewer private key does not match the pinned public key")
    payload: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "stage": stage,
        "decision": decision,
        "reviewer": clean_reviewer,
        "reviewed_at": when.isoformat(timespec="seconds"),
        "reviewed_from_status": str(gate["status"]),
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "reviewer_public_key_sha256": public_key_sha256(public_key),
        "gate_content_sha256": lineage.canonical_sha256(gate),
    }
    payload.update(review_binding(gate))
    payload["gate_sha256"] = artifact_sha256
    signature = private_key.sign(_signature_message(payload))
    payload["review_signature_ed25519"] = base64.b64encode(signature).decode("ascii")
    return payload


def validate_review(
    review: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    gate_sha256: str,
    public_key_path: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> None:
    """Reject reviews with an invalid Ed25519 signature or evidence binding."""
    stage = validate_gate(gate)
    binding = review_binding(gate)
    expected_keys = {
        "schema_version", "stage", "decision", "reviewer", "reviewed_at",
        "reviewed_from_status", "signature_algorithm",
        "reviewer_public_key_sha256", "gate_sha256",
        "gate_content_sha256",
        "review_signature_ed25519", *binding,
    }
    if set(review) != expected_keys:
        raise ValueError("review has missing or unexpected fields")
    if review.get("schema_version") != REVIEW_SCHEMA or review.get("stage") != stage:
        raise ValueError("review schema/stage does not match the gate")
    if review.get("decision") not in DECISIONS:
        raise ValueError("review decision is invalid")
    if review.get("decision") == "PROCEED" and gate.get("eligible_for_proceed") is not True:
        raise ValueError("review cannot override failed automatic hard criteria")
    if review.get("reviewed_from_status") != gate.get("status"):
        raise ValueError("review does not bind the gate's pre-review status")
    if review.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        raise ValueError("review signature algorithm is invalid")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer is missing")
    when = _parse_time(review.get("reviewed_at"), field="reviewed_at")
    created = _parse_time(gate["created_at"], field="created_at")
    if when < created or when > review_deadline(gate):
        raise ValueError("review timestamp lies outside the 24-hour review window")
    expected_gate_sha256 = _full_sha256(gate_sha256, field="gate_sha256")
    if review.get("gate_sha256") != expected_gate_sha256:
        raise ValueError("review is bound to a different gate artifact")
    if review.get("gate_content_sha256") != lineage.canonical_sha256(gate):
        raise ValueError("review is bound to different gate content")
    for field, expected in binding.items():
        if review.get(field) != expected:
            raise ValueError(f"review {field} does not match the gate evidence")
    public_key = _load_public_key(public_key_path)
    fingerprint = public_key_sha256(public_key)
    if review.get("reviewer_public_key_sha256") != fingerprint:
        raise ValueError("review is bound to a different reviewer public key")
    encoded_signature = review.get("review_signature_ed25519")
    if not isinstance(encoded_signature, str):
        raise ValueError("review Ed25519 signature is missing")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("review Ed25519 signature is not valid base64") from error
    if len(signature) != 64:
        raise ValueError("review Ed25519 signature has an invalid length")
    unsigned = dict(review)
    unsigned.pop("review_signature_ed25519")
    try:
        public_key.verify(signature, _signature_message(unsigned))
    except InvalidSignature as error:
        raise ValueError("review Ed25519 signature is invalid") from error


def merge_review(
    gate: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    gate_sha256: str,
    approval_sha256: str,
    public_key_path: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> dict[str, Any]:
    """Combine a gate and its mandatory human review."""
    validate_gate(gate)
    combined = dict(gate)
    validate_review(
        review,
        gate,
        gate_sha256=gate_sha256,
        public_key_path=public_key_path,
    )
    expected_approval_sha256 = json_artifact_sha256(review)
    if _full_sha256(approval_sha256, field="approval_sha256") != expected_approval_sha256:
        raise ValueError("approval_sha256 does not match the signed approval artifact")
    combined["status"] = review["decision"]
    combined.update(
        {
            "reviewed_by": review["reviewer"],
            "reviewed_at": review["reviewed_at"],
            "reviewed_from_status": review["reviewed_from_status"],
            "signature_algorithm": review["signature_algorithm"],
            "reviewer_public_key_sha256": review["reviewer_public_key_sha256"],
            "review_signature_ed25519": review["review_signature_ed25519"],
            "review_gate_sha256": review["gate_sha256"],
            "review_gate_content_sha256": review["gate_content_sha256"],
            "approval_sha256": expected_approval_sha256,
        }
    )
    return combined


def validate_reviewed_gate(
    gate: Mapping[str, Any],
    *,
    require_proceed: bool = False,
    public_key_path: Path | str = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> None:
    """Validate the signed fields copied into a combined decision artifact."""
    stage = validate_gate(gate)
    status = gate.get("status")
    if require_proceed and status != "PROCEED":
        raise ValueError(f"Stage-{stage} has not reviewed PROCEED")
    if status not in DECISIONS:
        raise ValueError(f"Stage-{stage} is not a reviewed decision")
    if status == "PROCEED" and gate.get("eligible_for_proceed") is not True:
        raise ValueError("reviewed PROCEED conflicts with automatic eligibility")
    required = (
        "reviewed_by", "reviewed_at", "reviewed_from_status",
        "signature_algorithm", "reviewer_public_key_sha256",
        "review_signature_ed25519", "review_gate_sha256", "approval_sha256",
        "review_gate_content_sha256",
    )
    if any(not gate.get(field) for field in required):
        raise ValueError("reviewed gate lacks its signed human-review fields")
    binding = review_binding(gate)
    unsigned: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "stage": stage,
        "decision": status,
        "reviewer": gate["reviewed_by"],
        "reviewed_at": gate["reviewed_at"],
        "reviewed_from_status": gate["reviewed_from_status"],
        "signature_algorithm": gate["signature_algorithm"],
        "reviewer_public_key_sha256": gate["reviewer_public_key_sha256"],
        "gate_content_sha256": gate["review_gate_content_sha256"],
    }
    unsigned.update(binding)
    unsigned["gate_sha256"] = _full_sha256(
        gate["review_gate_sha256"], field="review_gate_sha256"
    )
    review = dict(unsigned)
    review["review_signature_ed25519"] = gate["review_signature_ed25519"]
    approval_sha256 = _full_sha256(gate["approval_sha256"], field="approval_sha256")
    if approval_sha256 != json_artifact_sha256(review):
        raise ValueError("combined gate does not match its approval_sha256")
    original = dict(gate)
    for field in required:
        original.pop(field, None)
    original["status"] = gate["reviewed_from_status"]
    if lineage.canonical_sha256(original) != unsigned["gate_content_sha256"]:
        raise ValueError("combined gate no longer matches the reviewed gate content")
    validate_review(
        review,
        original,
        gate_sha256=unsigned["gate_sha256"],
        public_key_path=public_key_path,
    )
    when = _parse_time(gate["reviewed_at"], field="reviewed_at")
    created = _parse_time(gate["created_at"], field="created_at")
    if when < created or when > review_deadline(gate):
        raise ValueError("combined review lies outside the 24-hour review window")


__all__ = [
    "DECISIONS",
    "DEFAULT_REVIEWER_PUBLIC_KEY",
    "GATE_SCHEMAS",
    "REVIEW_SCHEMA",
    "SIGNATURE_ALGORITHM",
    "SUPPORTED_STAGES",
    "json_artifact_bytes",
    "json_artifact_sha256",
    "make_pending_gate",
    "make_review",
    "merge_review",
    "review_deadline",
    "reviewer_public_key_sha256",
    "validate_gate",
    "validate_review",
    "validate_reviewed_gate",
]
