"""Fail-closed validation for official AutoResearchClaw control receipts.

The validator does not infer success from an ARC checkout, configuration file, or
console log.  A bundle is accepted only when it contains ``receipt.v1.json`` and
the hash-bound official files produced by one successful ARC v0.5.0 stage::

    {
      "schema_version": "autoresearchclaw-control-receipt-v2",
      "autoresearchclaw": {
        "repository": "aiming-lab/AutoResearchClaw",
        "version": "0.5.0",
        "commit": "e2e23c...55357"
      },
      "acp": {
        "acpx_version": "0.12.0",
        "claude_adapter_version": "0.37.0",
        "codex_adapter_version": "0.0.44",
        "package_lock_sha256": "34ba9f...c39e8"
      },
      "invocation": {"mode": "co-pilot", "auto_approve": false},
      "run": {"run_id": "rc-...", "stage": 5},
      "artifacts": {
        "decision": {"path": "stage-05/decision.json", "sha256": "..."},
        "stage_health": {"path": "stage-05/stage_health.json", "sha256": "..."},
        "session": {"path": "hitl/session.json", "sha256": "..."},
        "interventions": {"path": "hitl/interventions.jsonl", "sha256": "..."},
        "gate_handoff": {
          "path": "control/gate-handoff-stage-05.v2.json", "sha256": "..."
        },
        "stage_outputs": [
          {"path": "stage-05/screened_papers.json", "sha256": "..."}
        ]
      }
    }

All descriptor paths are POSIX-style paths relative to the bundle root.  The
fixed descriptors must preserve ARC's native run-directory layout, and
``stage_outputs`` must exactly cover ``decision.json:output_artifacts``.  This
receipt is an integrity/provenance contract, not a replacement for the separate
author-signed scientific stage-gate decision.  Reports for Stages 9, 15, and 20
also hash-bind the prior formal report, receipt, bridge handoff, and chain head.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "autoresearchclaw-control-receipt-v2"
REPORT_SCHEMA = "autoresearchclaw-control-report-v2"
WAITING_RECEIPT_SCHEMA = "autoresearchclaw-control-waiting-receipt-v1"
WAITING_REPORT_SCHEMA = "autoresearchclaw-control-waiting-report-v1"
WAITING_LINEAGE_SCHEMA = "autoresearchclaw-waiting-lineage-v1"
GATE_HANDOFF_SCHEMA = "arc-gate-handoff-v2"
GATE_LINEAGE_SCHEMA = "autoresearchclaw-gate-lineage-v1"
OPERATOR_RESPONSE_SCHEMA = "arc-operator-response-v2"
OPERATOR_RESPONSE_CONSUMPTION_SCHEMA = "arc-operator-response-consumption-v1"
ARC_REPOSITORY = "aiming-lab/AutoResearchClaw"
ARC_VERSION = "0.5.0"
ARC_COMMIT = "e2e23c93b4943fd21cc531deb09850d8fda55357"
ACPX_VERSION = "0.12.0"
CLAUDE_ADAPTER_VERSION = "0.37.0"
CODEX_ADAPTER_VERSION = "0.0.44"
ACP_PACKAGE_LOCK_SHA256 = (
    "34ba9fc3bb03cb39861f689bb9da88ca037fa5596b27567f6fcf98ae879c39e8"
)
SUPPORTED_STAGES: Mapping[int, str] = {
    5: "literature_screen",
    9: "experiment_design",
    15: "research_decision",
    20: "quality_gate",
}
ORDERED_STAGES = tuple(SUPPORTED_STAGES)
_NATIVE_STAGE_NAMES: Mapping[int, str] = {
    stage: name.upper() for stage, name in SUPPORTED_STAGES.items()
}
_NEXT_STAGE_NAMES: Mapping[int, str] = {
    5: "KNOWLEDGE_EXTRACT",
    9: "CODE_GENERATION",
    15: "PAPER_OUTLINE",
    20: "KNOWLEDGE_ARCHIVE",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_ARTIFACTS = ("decision", "stage_health", "session", "interventions")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSONL_BYTES = 64 * 1024 * 1024


class ArcControlValidationError(ValueError):
    """Raised when an ARC control bundle is absent, ambiguous, or inconsistent."""


def _fail(message: str) -> ArcControlValidationError:
    return ArcControlValidationError(message)


def _require_object(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise _fail(
            f"{field} has missing or unexpected fields: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _reject_constant(value: str) -> None:
    raise _fail(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, *, field: str) -> Any:
    if len(raw) > _MAX_JSON_BYTES:
        raise _fail(f"{field} exceeds the {_MAX_JSON_BYTES}-byte JSON limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail(f"{field} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise _fail(f"{field} is not valid JSON: {error.msg}") from error


def _read_bytes(path: Path, *, field: str, maximum: int) -> bytes:
    try:
        if not path.is_file():
            raise _fail(f"{field} does not identify a regular file")
        size = path.stat().st_size
        if size > maximum:
            raise _fail(f"{field} exceeds the {maximum}-byte size limit")
        return path.read_bytes()
    except OSError as error:
        raise _fail(f"cannot read {field}: {error}") from error


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _fail(f"{field} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise _fail(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _full_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise _fail(f"{field} must be a full lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_record_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} must be a non-empty relative path")
    if "\\" in value or "\x00" in value or ":" in value:
        raise _fail(f"{field} must use a portable POSIX relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise _fail(f"{field} contains an empty, current, or parent path component")
    parsed = PurePosixPath(value)
    if parsed.is_absolute():
        raise _fail(f"{field} must be relative to the bundle root")
    return parsed


def _descriptor_path(
    bundle_root: Path,
    descriptor: Any,
    *,
    field: str,
) -> tuple[str, Path, str]:
    item = _require_object(descriptor, field=field)
    _require_exact_keys(item, {"path", "sha256"}, field=field)
    relative = _safe_relative_path(item["path"], field=f"{field}.path")
    expected_hash = _full_sha256(item["sha256"], field=f"{field}.sha256")
    candidate = bundle_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise _fail(f"{field}.path does not exist: {relative.as_posix()}") from error
    if not resolved.is_relative_to(bundle_root):
        raise _fail(f"{field}.path escapes the bundle root")
    if candidate.is_symlink() or not resolved.is_file():
        raise _fail(f"{field}.path must identify a non-symlink regular file")
    actual_hash = _file_sha256(resolved)
    if actual_hash != expected_hash:
        raise _fail(
            f"{field}.sha256 mismatch for {relative.as_posix()}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return relative.as_posix(), resolved, actual_hash


def _validate_pins(receipt: Mapping[str, Any]) -> None:
    arc = _require_object(receipt["autoresearchclaw"], field="autoresearchclaw")
    _require_exact_keys(
        arc, {"repository", "version", "commit"}, field="autoresearchclaw"
    )
    expected_arc = {
        "repository": ARC_REPOSITORY,
        "version": ARC_VERSION,
        "commit": ARC_COMMIT,
    }
    if dict(arc) != expected_arc:
        raise _fail("AutoResearchClaw repository/version/commit is not the pinned release")

    acp = _require_object(receipt["acp"], field="acp")
    _require_exact_keys(
        acp,
        {
            "acpx_version",
            "claude_adapter_version",
            "codex_adapter_version",
            "package_lock_sha256",
        },
        field="acp",
    )
    expected_acp = {
        "acpx_version": ACPX_VERSION,
        "claude_adapter_version": CLAUDE_ADAPTER_VERSION,
        "codex_adapter_version": CODEX_ADAPTER_VERSION,
        "package_lock_sha256": ACP_PACKAGE_LOCK_SHA256,
    }
    if dict(acp) != expected_acp:
        raise _fail("ACP tool versions or package-lock hash are not pinned")


def _validate_output_names(value: Any, *, stage_dir: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise _fail("decision.output_artifacts must be a non-empty list")
    names: list[str] = []
    for index, raw_name in enumerate(value):
        relative = _safe_relative_path(
            raw_name, field=f"decision.output_artifacts[{index}]"
        )
        name = relative.as_posix()
        if name in names:
            raise _fail("decision.output_artifacts contains duplicate paths")
        if name in {"decision.json", "stage_health.json"}:
            raise _fail("ARC control metadata cannot masquerade as a stage output")
        names.append(name)
    # The return form deliberately stays relative to the native ARC stage dir.
    if any(name.startswith(f"{stage_dir}/") for name in names):
        raise _fail("decision.output_artifacts must be relative to the ARC stage directory")
    return names


def _parse_interventions(raw: bytes) -> list[Mapping[str, Any]]:
    if len(raw) > _MAX_JSONL_BYTES:
        raise _fail(f"interventions exceeds the {_MAX_JSONL_BYTES}-byte JSONL limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail("interventions is not UTF-8") from error
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise _fail(f"interventions line {line_number} is empty")
        value = _parse_json_bytes(
            line.encode("utf-8"), field=f"interventions line {line_number}"
        )
        rows.append(_require_object(value, field=f"interventions line {line_number}"))
    if not rows:
        raise _fail("interventions contains no records")
    return rows


def _human_approval(
    interventions: Sequence[Mapping[str, Any]],
    *,
    expected_stage: int,
    stage_name: str,
    created_at: datetime,
    stage_ready_at: datetime,
    last_activity: datetime,
) -> Mapping[str, Any]:
    matches: list[tuple[datetime, Mapping[str, Any]]] = []
    for item in interventions:
        stage = item.get("stage")
        if isinstance(stage, bool) or stage != expected_stage:
            continue
        if item.get("type") != "approve" or item.get("accepted") is not True:
            continue
        if item.get("stage_name") != stage_name:
            continue
        human_input = item.get("human_input")
        if not isinstance(human_input, Mapping) or human_input.get("action") != "approve":
            continue
        if item.get("pause_reason") != "gate_approval":
            continue
        if item.get("outcome") != "Human chose: approve":
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        timestamp = _parse_timestamp(
            item.get("timestamp"), field="approval intervention timestamp"
        )
        human_timestamp = _parse_timestamp(
            human_input.get("timestamp"), field="approval human-input timestamp"
        )
        duration = item.get("duration_sec")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or float(duration) < 0
        ):
            continue
        if (
            created_at <= human_timestamp <= timestamp
            and stage_ready_at <= timestamp <= last_activity
        ):
            matches.append((timestamp, item))
    if not matches:
        raise _fail(
            f"no accepted human gate-approval intervention exists for ARC Stage {expected_stage}"
        )
    if len(matches) != 1:
        raise _fail(
            "exactly one accepted human gate-approval intervention must exist for "
            f"ARC Stage {expected_stage}; found {len(matches)}"
        )
    return matches[0][1]


def _validate_gate_handoff(
    *,
    bundle_root: Path,
    descriptor: Any,
    expected_stage: int,
    run_id: str,
    session_id: str,
    expected_stage_files: Sequence[Mapping[str, str]],
    intervention_index: int,
    intervention_sha256: str,
) -> dict[str, Any]:
    field = "artifacts.gate_handoff"
    relative, path, digest = _descriptor_path(bundle_root, descriptor, field=field)
    expected_path = f"control/gate-handoff-stage-{expected_stage:02d}.v2.json"
    if relative != expected_path:
        raise _fail(f"{field}.path must preserve bridge path {expected_path!r}")
    handoff = _require_object(
        _parse_json_bytes(
            _read_bytes(path, field=field, maximum=_MAX_JSON_BYTES), field=field
        ),
        field=field,
    )
    _require_exact_keys(
        handoff,
        {
            "schema_version",
            "created_at",
            "stage",
            "stage_name",
            "native_identity",
            "next_stage",
            "next_stage_name",
            "stage_files",
            "approval",
            "checkpoint",
            "lineage",
        },
        field=field,
    )
    if handoff["schema_version"] != GATE_HANDOFF_SCHEMA:
        raise _fail(f"gate handoff schema must be {GATE_HANDOFF_SCHEMA!r}")
    _parse_timestamp(handoff["created_at"], field="gate_handoff.created_at")
    if (
        handoff["stage"] != expected_stage
        or handoff["stage_name"] != _NATIVE_STAGE_NAMES[expected_stage]
        or handoff["next_stage"] != expected_stage + 1
        or handoff["next_stage_name"] != _NEXT_STAGE_NAMES[expected_stage]
    ):
        raise _fail("gate handoff stage or next-stage identity is invalid")
    identity = _require_object(
        handoff["native_identity"], field="gate_handoff.native_identity"
    )
    _require_exact_keys(
        identity, {"run_id", "session_id"}, field="gate_handoff.native_identity"
    )
    if identity.get("run_id") != run_id or identity.get("session_id") != session_id:
        raise _fail("gate handoff native run/session does not match the receipt")

    stage_files = handoff["stage_files"]
    if not isinstance(stage_files, list) or not stage_files:
        raise _fail("gate_handoff.stage_files must be a non-empty descriptor list")
    normalized_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(stage_files):
        item = _require_object(raw, field=f"gate_handoff.stage_files[{index}]")
        _require_exact_keys(
            item, {"path", "sha256"}, field=f"gate_handoff.stage_files[{index}]"
        )
        file_path, _resolved, observed_digest = _descriptor_path(
            bundle_root,
            item,
            field=f"gate_handoff.stage_files[{index}]",
        )
        if not file_path.startswith(f"stage-{expected_stage:02d}/"):
            raise _fail("gate_handoff.stage_files contains a file outside its stage")
        if file_path in seen_paths:
            raise _fail("gate_handoff.stage_files contains duplicate paths")
        seen_paths.add(file_path)
        normalized_files.append(
            {
                "path": file_path,
                "sha256": observed_digest,
            }
        )
    expected_files = sorted(
        (dict(item) for item in expected_stage_files), key=lambda item: item["path"]
    )
    if sorted(normalized_files, key=lambda item: item["path"]) != expected_files:
        raise _fail("gate handoff stage artifact set or hashes do not match the receipt")

    approval = _require_object(handoff["approval"], field="gate_handoff.approval")
    approval_keys = {
        "response_sha256",
        "waiting_sha256",
        "run_id",
        "session_id",
        "nonce",
        "consumption_receipt_path",
        "consumption_receipt_sha256",
        "native_intervention_ordinal",
        "native_intervention_sha256",
        "bridge_event_ordinal",
        "bridge_event_sha256",
    }
    _require_exact_keys(approval, approval_keys, field="gate_handoff.approval")
    if approval["run_id"] != run_id or approval["session_id"] != session_id:
        raise _fail("gate handoff approval run/session binding is invalid")
    for key in (
        "response_sha256",
        "waiting_sha256",
        "nonce",
        "consumption_receipt_sha256",
        "native_intervention_sha256",
        "bridge_event_sha256",
    ):
        _full_sha256(approval[key], field=f"gate_handoff.approval.{key}")
    _safe_relative_path(
        approval["consumption_receipt_path"],
        field="gate_handoff.approval.consumption_receipt_path",
    )
    for key in ("native_intervention_ordinal", "bridge_event_ordinal"):
        value = approval[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _fail(f"gate_handoff.approval.{key} must be non-negative")
    if (
        approval["native_intervention_ordinal"] != intervention_index
        or approval["native_intervention_sha256"] != intervention_sha256
    ):
        raise _fail("gate handoff does not bind the accepted native intervention")
    consumption_relative, _consumption_path, consumption_digest = _descriptor_path(
        bundle_root,
        {
            "path": approval["consumption_receipt_path"],
            "sha256": approval["consumption_receipt_sha256"],
        },
        field="gate_handoff.approval.consumption_receipt",
    )
    if (
        consumption_relative != approval["consumption_receipt_path"]
        or consumption_digest != approval["consumption_receipt_sha256"]
    ):
        raise _fail("gate handoff response consumption binding is invalid")
    consumption = _require_object(
        _parse_json_bytes(
            _read_bytes(
                _consumption_path,
                field="gate_handoff.approval.consumption_receipt",
                maximum=_MAX_JSON_BYTES,
            ),
            field="gate_handoff.approval.consumption_receipt",
        ),
        field="gate_handoff.approval.consumption_receipt",
    )
    _require_exact_keys(
        consumption,
        {
            "schema_version",
            "claimed_at",
            "response_path",
            "response_sha256",
            "stage",
            "run_id",
            "session_id",
            "waiting_sha256",
            "preapproval_checkpoint_sha256",
            "nonce",
            "response",
        },
        field="gate_handoff.approval.consumption_receipt",
    )
    if consumption["schema_version"] != OPERATOR_RESPONSE_CONSUMPTION_SCHEMA:
        raise _fail("operator response consumption receipt schema is invalid")
    _parse_timestamp(
        consumption["claimed_at"], field="operator_response_consumption.claimed_at"
    )
    if (
        consumption["stage"] != expected_stage
        or consumption["run_id"] != run_id
        or consumption["session_id"] != session_id
        or consumption["waiting_sha256"] != approval["waiting_sha256"]
        or consumption["nonce"] != approval["nonce"]
        or consumption["response_sha256"] != approval["response_sha256"]
    ):
        raise _fail("operator response consumption receipt identity is invalid")
    expected_response_path = (
        f"control/operator-response-snapshots/stage-{expected_stage:02d}-"
        f"{approval['response_sha256']}.v2.json"
    )
    if consumption["response_path"] != expected_response_path:
        raise _fail("operator response snapshot path is not stage/hash bound")
    _response_relative, response_path, _response_digest = _descriptor_path(
        bundle_root,
        {
            "path": consumption["response_path"],
            "sha256": consumption["response_sha256"],
        },
        field="gate_handoff.approval.operator_response_snapshot",
    )
    consumed_response = _require_object(
        consumption["response"], field="operator_response_consumption.response"
    )
    response_keys = {
        "schema_version",
        "stage",
        "run_id",
        "session_id",
        "waiting_sha256",
        "preapproval_checkpoint_sha256",
        "nonce",
        "action",
        "issued_at",
        "message",
    }
    _require_exact_keys(
        consumed_response,
        response_keys,
        field="operator_response_consumption.response",
    )
    if (
        consumed_response["schema_version"] != OPERATOR_RESPONSE_SCHEMA
        or consumed_response["stage"] != expected_stage
        or consumed_response["run_id"] != run_id
        or consumed_response["session_id"] != session_id
        or consumed_response["waiting_sha256"] != approval["waiting_sha256"]
        or consumed_response["nonce"] != approval["nonce"]
        or consumed_response["action"] != "approve"
        or not isinstance(consumed_response["message"], str)
    ):
        raise _fail("consumed operator response does not bind this approval")
    _parse_timestamp(
        consumed_response["issued_at"], field="operator_response.issued_at"
    )
    response_snapshot = _require_object(
        _parse_json_bytes(
            _read_bytes(
                response_path,
                field="gate_handoff.approval.operator_response_snapshot",
                maximum=_MAX_JSON_BYTES,
            ),
            field="gate_handoff.approval.operator_response_snapshot",
        ),
        field="gate_handoff.approval.operator_response_snapshot",
    )
    if response_snapshot != consumed_response:
        raise _fail("consumed operator response differs from its immutable snapshot")

    checkpoint = _require_object(handoff["checkpoint"], field="gate_handoff.checkpoint")
    _require_exact_keys(
        checkpoint,
        {"path", "sha256", "last_completed_stage"},
        field="gate_handoff.checkpoint",
    )
    if (
        checkpoint["path"] != "checkpoint.json"
        or checkpoint["last_completed_stage"] != expected_stage - 1
    ):
        raise _fail("gate handoff pre-approval checkpoint binding is invalid")
    _full_sha256(checkpoint["sha256"], field="gate_handoff.checkpoint.sha256")
    _checkpoint_relative, checkpoint_path, _checkpoint_digest = _descriptor_path(
        bundle_root,
        {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        field="gate_handoff.checkpoint.snapshot",
    )
    checkpoint_snapshot = _require_object(
        _parse_json_bytes(
            _read_bytes(
                checkpoint_path,
                field="gate_handoff.checkpoint.snapshot",
                maximum=_MAX_JSON_BYTES,
            ),
            field="gate_handoff.checkpoint.snapshot",
        ),
        field="gate_handoff.checkpoint.snapshot",
    )
    if checkpoint_snapshot.get("last_completed_stage") != expected_stage - 1:
        raise _fail("pre-approval checkpoint snapshot has the wrong completed stage")
    if (
        consumption["preapproval_checkpoint_sha256"] != checkpoint["sha256"]
        or consumed_response["preapproval_checkpoint_sha256"]
        != checkpoint["sha256"]
    ):
        raise _fail("operator response does not bind the exported checkpoint snapshot")

    lineage = _require_object(handoff["lineage"], field="gate_handoff.lineage")
    _require_exact_keys(
        lineage,
        {
            "source_config_snapshot_sha256",
            "effective_config_sha256",
            "project_state_sha256",
            "previous_handoff_path",
            "previous_handoff_sha256",
        },
        field="gate_handoff.lineage",
    )
    for key in (
        "source_config_snapshot_sha256",
        "effective_config_sha256",
        "project_state_sha256",
    ):
        _full_sha256(lineage[key], field=f"gate_handoff.lineage.{key}")
    stage_index = ORDERED_STAGES.index(expected_stage)
    previous_stage = ORDERED_STAGES[stage_index - 1] if stage_index else None
    if previous_stage is None:
        if (
            lineage["previous_handoff_path"] is not None
            or lineage["previous_handoff_sha256"] is not None
        ):
            raise _fail("Stage 5 gate handoff must not have a predecessor")
    else:
        expected_previous_path = (
            f"control/gate-handoff-stage-{previous_stage:02d}.v2.json"
        )
        if lineage["previous_handoff_path"] != expected_previous_path:
            raise _fail("gate handoff predecessor path is not the prior registered gate")
        _full_sha256(
            lineage["previous_handoff_sha256"],
            field="gate_handoff.lineage.previous_handoff_sha256",
        )
    return {
        "schema_version": GATE_HANDOFF_SCHEMA,
        "path": relative,
        "sha256": digest,
        "previous_stage": previous_stage,
        "previous_handoff_sha256": lineage["previous_handoff_sha256"],
        "response_sha256": approval["response_sha256"],
        "waiting_sha256": approval["waiting_sha256"],
        "consumption_receipt_sha256": approval["consumption_receipt_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
    }


def _gate_lineage_material(
    report: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    handoff = _require_object(report["gate_handoff"], field="gate_handoff")
    return {
        "stage": report["stage"],
        "run_id": report["run_id"],
        "session_id": report["session_id"],
        "receipt_sha256": report["receipt_sha256"],
        "handoff_sha256": handoff["sha256"],
        "previous_stage": lineage["previous_stage"],
        "previous_report_sha256": lineage["previous_report_sha256"],
        "previous_receipt_sha256": lineage["previous_receipt_sha256"],
        "previous_handoff_sha256": lineage["previous_handoff_sha256"],
        "previous_chain_sha256": lineage["previous_chain_sha256"],
    }


def _attach_gate_lineage(
    report: Mapping[str, Any], previous_report: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(report)
    stage = int(result["stage"])
    stage_index = ORDERED_STAGES.index(stage)
    expected_previous_stage = ORDERED_STAGES[stage_index - 1] if stage_index else None
    handoff = _require_object(result["gate_handoff"], field="gate_handoff")
    if expected_previous_stage is None:
        if previous_report is not None:
            raise _fail("Stage 5 ARC report must not be given a predecessor")
        predecessor = {
            "previous_stage": None,
            "previous_report_sha256": None,
            "previous_receipt_sha256": None,
            "previous_handoff_sha256": None,
            "previous_chain_sha256": None,
        }
    else:
        if previous_report is None:
            raise _fail(
                f"ARC Stage {stage} requires the Stage {expected_previous_stage} report"
            )
        previous = validate_arc_control_report(
            previous_report, expected_previous_stage
        )
        if (
            previous["run_id"] != result["run_id"]
            or previous["session_id"] != result["session_id"]
        ):
            raise _fail("ARC run_id/session_id changed between formal gates")
        previous_handoff = _require_object(
            previous["gate_handoff"], field="previous gate_handoff"
        )
        if handoff["previous_handoff_sha256"] != previous_handoff["sha256"]:
            raise _fail("formal gate handoff hash chain is broken")
        previous_lineage = _require_object(
            previous["gate_lineage"], field="previous gate_lineage"
        )
        predecessor = {
            "previous_stage": expected_previous_stage,
            "previous_report_sha256": _canonical_sha256(previous),
            "previous_receipt_sha256": previous["receipt_sha256"],
            "previous_handoff_sha256": previous_handoff["sha256"],
            "previous_chain_sha256": previous_lineage["chain_sha256"],
        }
    lineage = {
        "schema_version": GATE_LINEAGE_SCHEMA,
        **predecessor,
    }
    lineage["chain_sha256"] = _canonical_sha256(
        _gate_lineage_material(result, lineage)
    )
    result["gate_lineage"] = lineage
    return result


def _waiting_predecessor(
    *,
    expected_stage: int,
    previous_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the exact prior formal handoff bound by a waiting receipt."""

    stage_index = ORDERED_STAGES.index(expected_stage)
    previous_stage = ORDERED_STAGES[stage_index - 1] if stage_index else None
    if previous_stage is None:
        if previous_report is not None:
            raise _fail("Stage 5 waiting receipt must not have a predecessor")
        return None
    if previous_report is None:
        raise _fail(
            f"ARC Stage {expected_stage} waiting receipt requires the formal "
            f"Stage {previous_stage} report"
        )
    previous = validate_arc_control_report(previous_report, previous_stage)
    return {
        "stage": previous_stage,
        "report_sha256": _canonical_sha256(previous),
        "receipt_sha256": previous["receipt_sha256"],
        "handoff_sha256": previous["gate_handoff"]["sha256"],
        "chain_sha256": previous["gate_lineage"]["chain_sha256"],
    }


def _waiting_lineage_material(
    report: Mapping[str, Any], predecessor: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "run_id": report["run_id"],
        "session_id": report["session_id"],
        "waiting_receipt_sha256": report["waiting_receipt_sha256"],
        "waiting_sha256": report["waiting"]["sha256"],
        "challenge_sha256": report["challenge"]["sha256"],
        "checkpoint_sha256": report["challenge"][
            "preapproval_checkpoint_sha256"
        ],
        "nonce": report["challenge"]["nonce"],
        "predecessor": predecessor,
    }


def _attach_waiting_lineage(
    report: Mapping[str, Any], predecessor: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(report)
    result["waiting_lineage"] = {
        "schema_version": WAITING_LINEAGE_SCHEMA,
        "predecessor": dict(predecessor) if predecessor is not None else None,
        "chain_sha256": _canonical_sha256(
            _waiting_lineage_material(result, predecessor)
        ),
    }
    return result


def validate_arc_waiting_bundle(
    bundle_dir: Path | str,
    expected_stage: int,
    *,
    previous_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an immutable native ARC pause before author approval.

    This is deliberately a *waiting* receipt: it proves that the official
    process completed the registered stage and is blocked on the nonce-bound
    post-stage gate.  It contains no approval and cannot be used as a formal
    handoff receipt.  The native process may advance only after a separately
    generated scientific gate is reviewed once with the pinned Ed25519 key.
    """

    if expected_stage not in SUPPORTED_STAGES:
        raise _fail(f"expected_stage must be one of {sorted(SUPPORTED_STAGES)}")
    try:
        bundle_root = Path(bundle_dir).expanduser().resolve(strict=True)
    except OSError as error:
        raise _fail(f"ARC waiting bundle does not exist: {bundle_dir}") from error
    if not bundle_root.is_dir():
        raise _fail("ARC waiting bundle must be a directory")

    receipt_path = bundle_root / "waiting-receipt.v1.json"
    receipt_raw = _read_bytes(
        receipt_path, field="waiting-receipt.v1.json", maximum=_MAX_JSON_BYTES
    )
    receipt = _require_object(
        _parse_json_bytes(receipt_raw, field="waiting-receipt.v1.json"),
        field="waiting-receipt.v1.json",
    )
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "autoresearchclaw",
            "acp",
            "invocation",
            "run",
            "artifacts",
            "predecessor",
        },
        field="waiting-receipt.v1.json",
    )
    if receipt["schema_version"] != WAITING_RECEIPT_SCHEMA:
        raise _fail(f"waiting receipt schema must be {WAITING_RECEIPT_SCHEMA!r}")
    _validate_pins(receipt)
    invocation = _require_object(receipt["invocation"], field="invocation")
    _require_exact_keys(invocation, {"mode", "auto_approve"}, field="invocation")
    if invocation != {"mode": "co-pilot", "auto_approve": False}:
        raise _fail("ARC waiting receipt is not a manual co-pilot invocation")

    run = _require_object(receipt["run"], field="run")
    _require_exact_keys(run, {"run_id", "session_id", "stage"}, field="run")
    run_id = run["run_id"]
    session_id = run["session_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise _fail("run.run_id must be a non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        raise _fail("run.session_id must be a non-empty string")
    if isinstance(run["stage"], bool) or run["stage"] != expected_stage:
        raise _fail(f"waiting receipt does not bind ARC Stage {expected_stage}")

    artifacts = _require_object(receipt["artifacts"], field="artifacts")
    control_names = {
        "decision",
        "stage_health",
        "session",
        "waiting",
        "operator_challenge",
        "checkpoint",
    }
    _require_exact_keys(
        artifacts, {*control_names, "stage_outputs"}, field="artifacts"
    )
    stage_dir = f"stage-{expected_stage:02d}"
    fixed_paths = {
        "decision": f"{stage_dir}/decision.json",
        "stage_health": f"{stage_dir}/stage_health.json",
        "session": "hitl/session.json",
        "waiting": "hitl/waiting.json",
        "operator_challenge": "hitl/operator-challenge.v2.json",
        "checkpoint": "checkpoint.json",
    }
    resolved: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    all_paths: set[str] = set()
    for name in control_names:
        relative, path, digest = _descriptor_path(
            bundle_root, artifacts[name], field=f"artifacts.{name}"
        )
        if relative != fixed_paths[name]:
            raise _fail(
                f"artifacts.{name}.path must preserve ARC path {fixed_paths[name]!r}"
            )
        if relative in all_paths:
            raise _fail("waiting artifact descriptors must identify distinct files")
        all_paths.add(relative)
        resolved[name] = path
        artifact_hashes[name] = digest

    decision = _require_object(
        _parse_json_bytes(
            _read_bytes(resolved["decision"], field="decision", maximum=_MAX_JSON_BYTES),
            field="decision",
        ),
        field="decision",
    )
    expected_stage_id = f"{expected_stage:02d}-{SUPPORTED_STAGES[expected_stage]}"
    if (
        decision.get("stage_id") != expected_stage_id
        or decision.get("run_id") != run_id
        or decision.get("status") != "blocked_approval"
        or decision.get("decision") != "block"
        or decision.get("error") not in (None, "")
    ):
        raise _fail("native ARC decision is not a clean blocked_approval gate")
    decision_time = _parse_timestamp(decision.get("ts"), field="decision.ts")
    output_names = _validate_output_names(
        decision.get("output_artifacts"), stage_dir=stage_dir
    )

    health = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["stage_health"],
                field="stage_health",
                maximum=_MAX_JSON_BYTES,
            ),
            field="stage_health",
        ),
        field="stage_health",
    )
    if (
        health.get("stage_id") != expected_stage_id
        or health.get("run_id") != run_id
        or health.get("status") != "blocked_approval"
        or health.get("error") not in (None, "")
    ):
        raise _fail("native ARC stage_health is not a clean blocked_approval gate")
    count = health.get("artifacts_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(output_names):
        raise _fail("stage_health.artifacts_count does not match decision outputs")
    health_time = _parse_timestamp(
        health.get("timestamp"), field="stage_health.timestamp"
    )
    if health_time < decision_time:
        raise _fail("stage_health timestamp precedes decision metadata")

    raw_outputs = artifacts["stage_outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise _fail("artifacts.stage_outputs must be a non-empty descriptor list")
    stage_output_hashes: dict[str, str] = {}
    for index, descriptor in enumerate(raw_outputs):
        relative, _path, digest = _descriptor_path(
            bundle_root,
            descriptor,
            field=f"artifacts.stage_outputs[{index}]",
        )
        if relative in all_paths:
            raise _fail("waiting artifact descriptors must identify distinct files")
        all_paths.add(relative)
        prefix = f"{stage_dir}/"
        if not relative.startswith(prefix):
            raise _fail("stage output lies outside the native ARC stage directory")
        name = relative[len(prefix) :]
        if name in stage_output_hashes:
            raise _fail("stage output descriptors contain duplicate paths")
        stage_output_hashes[name] = digest
    if set(stage_output_hashes) != set(output_names):
        raise _fail("stage outputs do not exactly cover decision.output_artifacts")

    session = _require_object(
        _parse_json_bytes(
            _read_bytes(resolved["session"], field="session", maximum=_MAX_JSON_BYTES),
            field="session",
        ),
        field="session",
    )
    if (
        session.get("run_id") != run_id
        or session.get("session_id") != session_id
        or session.get("mode") != "co-pilot"
        or session.get("state") != "active"
        or session.get("waiting") is None
    ):
        raise _fail("HITL session is not the active waiting co-pilot session")
    created_at = _parse_timestamp(session.get("created_at"), field="session.created_at")
    last_activity = _parse_timestamp(
        session.get("last_activity"), field="session.last_activity"
    )
    if not (created_at <= decision_time <= health_time <= last_activity):
        raise _fail("native stage metadata lies outside the waiting HITL session")

    waiting = _require_object(
        _parse_json_bytes(
            _read_bytes(resolved["waiting"], field="waiting", maximum=_MAX_JSON_BYTES),
            field="waiting",
        ),
        field="waiting",
    )
    _require_exact_keys(
        waiting,
        {"stage", "stage_name", "reason", "since", "available_actions"},
        field="waiting",
    )
    waiting_since = _parse_timestamp(waiting["since"], field="waiting.since")
    actions = waiting["available_actions"]
    if (
        waiting["stage"] != expected_stage
        or waiting["stage_name"] != _NATIVE_STAGE_NAMES[expected_stage]
        or waiting["reason"] != "gate_approval"
        or not isinstance(actions, list)
        or len(actions) != 3
        or not all(isinstance(action, str) for action in actions)
        or set(actions) != {"approve", "reject", "abort"}
        or not (created_at <= waiting_since <= last_activity)
    ):
        raise _fail("native waiting record is not the registered post-stage gate")

    challenge = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["operator_challenge"],
                field="operator_challenge",
                maximum=_MAX_JSON_BYTES,
            ),
            field="operator_challenge",
        ),
        field="operator_challenge",
    )
    challenge_keys = {
        "schema_version",
        "created_at",
        "stage",
        "run_id",
        "session_id",
        "waiting_sha256",
        "waiting_since",
        "expires_at",
        "preapproval_checkpoint_sha256",
        "nonce",
        "available_actions",
    }
    _require_exact_keys(challenge, challenge_keys, field="operator_challenge")
    challenge_created = _parse_timestamp(
        challenge["created_at"], field="operator_challenge.created_at"
    )
    challenge_waiting = _parse_timestamp(
        challenge["waiting_since"], field="operator_challenge.waiting_since"
    )
    expires = _parse_timestamp(
        challenge["expires_at"], field="operator_challenge.expires_at"
    )
    if (
        challenge["schema_version"] != "arc-operator-challenge-v2"
        or challenge["stage"] != expected_stage
        or challenge["run_id"] != run_id
        or challenge["session_id"] != session_id
        or challenge["waiting_sha256"] != artifact_hashes["waiting"]
        or challenge_waiting != waiting_since
        or expires != waiting_since + timedelta(hours=24)
        or not (waiting_since <= challenge_created <= expires)
        or challenge["available_actions"] != actions
    ):
        raise _fail("operator challenge does not bind the active native waiting record")
    _full_sha256(challenge["waiting_sha256"], field="challenge.waiting_sha256")
    checkpoint_sha = _full_sha256(
        challenge["preapproval_checkpoint_sha256"],
        field="challenge.preapproval_checkpoint_sha256",
    )
    nonce = _full_sha256(challenge["nonce"], field="challenge.nonce")
    if checkpoint_sha != artifact_hashes["checkpoint"]:
        raise _fail("operator challenge does not bind the exported checkpoint")
    checkpoint = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["checkpoint"], field="checkpoint", maximum=_MAX_JSON_BYTES
            ),
            field="checkpoint",
        ),
        field="checkpoint",
    )
    if checkpoint.get("last_completed_stage") != expected_stage - 1:
        raise _fail("pre-approval checkpoint is at the wrong native stage")

    expected_predecessor = _waiting_predecessor(
        expected_stage=expected_stage, previous_report=previous_report
    )
    observed_predecessor = receipt["predecessor"]
    if observed_predecessor != expected_predecessor:
        raise _fail("waiting receipt does not bind the exact prior formal gate")
    if expected_predecessor is not None and (
        previous_report is None
        or previous_report.get("run_id") != run_id
        or previous_report.get("session_id") != session_id
    ):
        raise _fail("ARC run_id/session_id changed between formal and waiting gates")

    report = {
        "schema_version": WAITING_REPORT_SCHEMA,
        "validated": True,
        "official_control": True,
        "phase": "waiting",
        "stage": expected_stage,
        "stage_id": expected_stage_id,
        "stage_name": _NATIVE_STAGE_NAMES[expected_stage],
        "run_id": run_id,
        "session_id": session_id,
        "mode": "co-pilot",
        "auto_approve": False,
        "decision": "awaiting_signed_review",
        "autoresearchclaw": {
            "repository": ARC_REPOSITORY,
            "version": ARC_VERSION,
            "commit": ARC_COMMIT,
        },
        "acp": {
            "acpx_version": ACPX_VERSION,
            "claude_adapter_version": CLAUDE_ADAPTER_VERSION,
            "codex_adapter_version": CODEX_ADAPTER_VERSION,
            "package_lock_sha256": ACP_PACKAGE_LOCK_SHA256,
        },
        "waiting_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "control_artifact_sha256": dict(sorted(artifact_hashes.items())),
        "stage_output_sha256": dict(sorted(stage_output_hashes.items())),
        "waiting": {
            "sha256": artifact_hashes["waiting"],
            "since": waiting_since.isoformat(),
            "expires_at": expires.isoformat(),
        },
        "challenge": {
            "sha256": artifact_hashes["operator_challenge"],
            "preapproval_checkpoint_sha256": checkpoint_sha,
            "nonce": nonce,
        },
    }
    return _attach_waiting_lineage(report, expected_predecessor)


def validate_arc_waiting_report(
    value: Mapping[str, Any],
    expected_stage: int,
    *,
    previous_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a normalized pre-approval report without treating it as handoff."""

    if expected_stage not in SUPPORTED_STAGES:
        raise _fail(f"expected_stage must be one of {sorted(SUPPORTED_STAGES)}")
    report = _require_object(value, field="ARC waiting report")
    expected_keys = {
        "schema_version",
        "validated",
        "official_control",
        "phase",
        "stage",
        "stage_id",
        "stage_name",
        "run_id",
        "session_id",
        "mode",
        "auto_approve",
        "decision",
        "autoresearchclaw",
        "acp",
        "waiting_receipt_sha256",
        "control_artifact_sha256",
        "stage_output_sha256",
        "waiting",
        "challenge",
        "waiting_lineage",
    }
    _require_exact_keys(report, expected_keys, field="ARC waiting report")
    if (
        report["schema_version"] != WAITING_REPORT_SCHEMA
        or report["validated"] is not True
        or report["official_control"] is not True
        or report["phase"] != "waiting"
        or report["decision"] != "awaiting_signed_review"
    ):
        raise _fail("ARC waiting report is not a validated pre-approval receipt")
    stage_name = SUPPORTED_STAGES[expected_stage]
    if (
        report["stage"] != expected_stage
        or report["stage_id"] != f"{expected_stage:02d}-{stage_name}"
        or report["stage_name"] != stage_name.upper()
    ):
        raise _fail(f"ARC waiting report does not bind Stage {expected_stage}")
    if report["mode"] != "co-pilot" or report["auto_approve"] is not False:
        raise _fail("ARC waiting report is not a manual co-pilot pause")
    for field in ("run_id", "session_id"):
        if not isinstance(report[field], str) or not report[field].strip():
            raise _fail(f"ARC waiting report {field} must be non-empty")
    _validate_pins(
        {"autoresearchclaw": report["autoresearchclaw"], "acp": report["acp"]}
    )
    _full_sha256(report["waiting_receipt_sha256"], field="waiting_receipt_sha256")
    control_hashes = _require_object(
        report["control_artifact_sha256"], field="control_artifact_sha256"
    )
    _require_exact_keys(
        control_hashes,
        {
            "decision",
            "stage_health",
            "session",
            "waiting",
            "operator_challenge",
            "checkpoint",
        },
        field="control_artifact_sha256",
    )
    for key, digest in control_hashes.items():
        _full_sha256(digest, field=f"control_artifact_sha256.{key}")
    outputs = _require_object(
        report["stage_output_sha256"], field="stage_output_sha256"
    )
    if not outputs:
        raise _fail("stage_output_sha256 must not be empty")
    for name, digest in outputs.items():
        _safe_relative_path(name, field="stage_output_sha256 key")
        _full_sha256(digest, field=f"stage_output_sha256.{name}")
    waiting = _require_object(report["waiting"], field="waiting")
    _require_exact_keys(waiting, {"sha256", "since", "expires_at"}, field="waiting")
    _full_sha256(waiting["sha256"], field="waiting.sha256")
    since = _parse_timestamp(waiting["since"], field="waiting.since")
    expires = _parse_timestamp(waiting["expires_at"], field="waiting.expires_at")
    if expires != since + timedelta(hours=24):
        raise _fail("ARC waiting report has a non-24-hour review window")
    challenge = _require_object(report["challenge"], field="challenge")
    _require_exact_keys(
        challenge,
        {"sha256", "preapproval_checkpoint_sha256", "nonce"},
        field="challenge",
    )
    for field in ("sha256", "preapproval_checkpoint_sha256", "nonce"):
        _full_sha256(challenge[field], field=f"challenge.{field}")
    if challenge["sha256"] != control_hashes["operator_challenge"]:
        raise _fail("challenge hash differs from the control artifact inventory")
    if challenge["preapproval_checkpoint_sha256"] != control_hashes["checkpoint"]:
        raise _fail("checkpoint hash differs from the control artifact inventory")
    if waiting["sha256"] != control_hashes["waiting"]:
        raise _fail("waiting hash differs from the control artifact inventory")

    lineage = _require_object(report["waiting_lineage"], field="waiting_lineage")
    _require_exact_keys(
        lineage, {"schema_version", "predecessor", "chain_sha256"}, field="waiting_lineage"
    )
    if lineage["schema_version"] != WAITING_LINEAGE_SCHEMA:
        raise _fail(f"waiting lineage schema must be {WAITING_LINEAGE_SCHEMA!r}")
    predecessor = lineage["predecessor"]
    stage_index = ORDERED_STAGES.index(expected_stage)
    previous_stage = ORDERED_STAGES[stage_index - 1] if stage_index else None
    if previous_stage is None:
        if predecessor is not None:
            raise _fail("Stage 5 waiting lineage must not have a predecessor")
    else:
        predecessor = _require_object(predecessor, field="waiting_lineage.predecessor")
        _require_exact_keys(
            predecessor,
            {"stage", "report_sha256", "receipt_sha256", "handoff_sha256", "chain_sha256"},
            field="waiting_lineage.predecessor",
        )
        if predecessor["stage"] != previous_stage:
            raise _fail("waiting lineage skips or reorders a formal gate")
        for field in ("report_sha256", "receipt_sha256", "handoff_sha256", "chain_sha256"):
            _full_sha256(predecessor[field], field=f"waiting_lineage.predecessor.{field}")
    if lineage["chain_sha256"] != _canonical_sha256(
        _waiting_lineage_material(report, predecessor)
    ):
        raise _fail("ARC waiting lineage hash is invalid")
    if previous_report is not None or previous_stage is None:
        expected_predecessor = _waiting_predecessor(
            expected_stage=expected_stage, previous_report=previous_report
        )
        if predecessor != expected_predecessor:
            raise _fail("ARC waiting report does not bind the supplied predecessor")
        if previous_report is not None and (
            previous_report.get("run_id") != report["run_id"]
            or previous_report.get("session_id") != report["session_id"]
        ):
            raise _fail("ARC run_id/session_id changed before the waiting gate")
    return dict(report)


def validate_arc_control_bundle(
    bundle_dir: Path | str,
    expected_stage: int,
    previous_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an official ARC control bundle and return a normalized report.

    ``expected_stage`` is intentionally supplied by the caller rather than
    inferred from the receipt.  Only Stages 5, 9, 15, and 20 are supported.
    Every validation error raises :class:`ArcControlValidationError`.
    """
    if (
        isinstance(expected_stage, bool)
        or not isinstance(expected_stage, int)
        or expected_stage not in SUPPORTED_STAGES
    ):
        raise _fail(f"expected_stage must be one of {sorted(SUPPORTED_STAGES)}")
    try:
        bundle_root = Path(bundle_dir).expanduser().resolve(strict=True)
    except OSError as error:
        raise _fail(f"ARC control bundle does not exist: {bundle_dir}") from error
    if not bundle_root.is_dir():
        raise _fail("ARC control bundle must be a directory")

    receipt_path = bundle_root / "receipt.v1.json"
    receipt_raw = _read_bytes(
        receipt_path, field="receipt.v1.json", maximum=_MAX_JSON_BYTES
    )
    receipt = _require_object(
        _parse_json_bytes(receipt_raw, field="receipt.v1.json"),
        field="receipt.v1.json",
    )
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "autoresearchclaw",
            "acp",
            "invocation",
            "run",
            "artifacts",
        },
        field="receipt.v1.json",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise _fail(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    _validate_pins(receipt)

    invocation = _require_object(receipt["invocation"], field="invocation")
    _require_exact_keys(invocation, {"mode", "auto_approve"}, field="invocation")
    if invocation["mode"] != "co-pilot":
        raise _fail("ARC invocation mode must be 'co-pilot'")
    if invocation["auto_approve"] is not False:
        raise _fail("ARC auto_approve must be exactly false")

    run = _require_object(receipt["run"], field="run")
    _require_exact_keys(run, {"run_id", "stage"}, field="run")
    run_id = run["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise _fail("run.run_id must be a non-empty string")
    if isinstance(run["stage"], bool) or run["stage"] != expected_stage:
        raise _fail(f"receipt does not bind expected ARC Stage {expected_stage}")

    artifacts = _require_object(receipt["artifacts"], field="artifacts")
    _require_exact_keys(
        artifacts,
        {*_CONTROL_ARTIFACTS, "stage_outputs", "gate_handoff"},
        field="artifacts",
    )
    stage_dir = f"stage-{expected_stage:02d}"
    fixed_paths = {
        "decision": f"{stage_dir}/decision.json",
        "stage_health": f"{stage_dir}/stage_health.json",
        "session": "hitl/session.json",
        "interventions": "hitl/interventions.jsonl",
    }
    resolved: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    all_descriptor_paths: set[str] = set()
    for name in _CONTROL_ARTIFACTS:
        relative, path, digest = _descriptor_path(
            bundle_root, artifacts[name], field=f"artifacts.{name}"
        )
        if relative != fixed_paths[name]:
            raise _fail(
                f"artifacts.{name}.path must preserve ARC path {fixed_paths[name]!r}"
            )
        if relative in all_descriptor_paths:
            raise _fail("artifact descriptors must identify distinct files")
        all_descriptor_paths.add(relative)
        resolved[name] = path
        artifact_hashes[name] = digest

    decision = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["decision"], field="decision", maximum=_MAX_JSON_BYTES
            ),
            field="decision",
        ),
        field="decision",
    )
    expected_stage_id = f"{expected_stage:02d}-{SUPPORTED_STAGES[expected_stage]}"
    if decision.get("stage_id") != expected_stage_id:
        raise _fail(f"decision.stage_id must be {expected_stage_id!r}")
    if decision.get("run_id") != run_id:
        raise _fail("decision.run_id does not match the receipt")
    if (
        decision.get("status") != "blocked_approval"
        or decision.get("decision") != "block"
    ):
        raise _fail(
            "official ARC gate must preserve its pre-approval blocked_approval/block record"
        )
    if decision.get("error") not in (None, ""):
        raise _fail("official ARC decision records a stage error")
    decision_time = _parse_timestamp(decision.get("ts"), field="decision.ts")
    outputs = _validate_output_names(
        decision.get("output_artifacts"), stage_dir=stage_dir
    )

    health = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["stage_health"],
                field="stage_health",
                maximum=_MAX_JSON_BYTES,
            ),
            field="stage_health",
        ),
        field="stage_health",
    )
    if health.get("stage_id") != expected_stage_id or health.get("run_id") != run_id:
        raise _fail("stage_health stage/run identity does not match the receipt")
    if health.get("status") != "blocked_approval" or health.get("error") not in (
        None,
        "",
    ):
        raise _fail("official ARC gate stage_health is not blocked_approval with null error")
    count = health.get("artifacts_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(outputs):
        raise _fail("stage_health.artifacts_count does not match decision outputs")
    duration = health.get("duration_sec")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or float(duration) < 0
    ):
        raise _fail("stage_health.duration_sec must be a non-negative number")
    health_time = _parse_timestamp(
        health.get("timestamp"), field="stage_health.timestamp"
    )

    raw_stage_outputs = artifacts["stage_outputs"]
    if not isinstance(raw_stage_outputs, list) or not raw_stage_outputs:
        raise _fail("artifacts.stage_outputs must be a non-empty descriptor list")
    stage_output_hashes: dict[str, str] = {}
    for index, descriptor in enumerate(raw_stage_outputs):
        relative, _path, digest = _descriptor_path(
            bundle_root,
            descriptor,
            field=f"artifacts.stage_outputs[{index}]",
        )
        if relative in all_descriptor_paths:
            raise _fail("artifact descriptors must identify distinct files")
        all_descriptor_paths.add(relative)
        expected_prefix = f"{stage_dir}/"
        if not relative.startswith(expected_prefix):
            raise _fail("stage output descriptor lies outside the native ARC stage directory")
        local_name = relative[len(expected_prefix) :]
        if local_name in stage_output_hashes:
            raise _fail("stage output descriptors contain duplicate paths")
        stage_output_hashes[local_name] = digest
    if set(stage_output_hashes) != set(outputs):
        raise _fail(
            "stage output descriptors do not exactly cover decision.output_artifacts"
        )

    session = _require_object(
        _parse_json_bytes(
            _read_bytes(
                resolved["session"], field="session", maximum=_MAX_JSON_BYTES
            ),
            field="session",
        ),
        field="session",
    )
    if session.get("run_id") != run_id:
        raise _fail("HITL session.run_id does not match the receipt")
    if session.get("mode") != "co-pilot":
        raise _fail("official HITL session was not in co-pilot mode")
    if session.get("state") not in ("active", "completed"):
        raise _fail("official HITL session is not active or completed")
    future_waiting = session.get("waiting")
    if future_waiting is not None:
        future_waiting = _require_object(future_waiting, field="session.waiting")
        future_stage = future_waiting.get("stage")
        if (
            session.get("state") != "active"
            or isinstance(future_stage, bool)
            or not isinstance(future_stage, int)
            or future_stage not in SUPPORTED_STAGES
            or future_stage <= expected_stage
            or future_waiting.get("reason") != "gate_approval"
        ):
            raise _fail(
                "official HITL session waiting state is not a later registered gate"
            )
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise _fail("HITL session_id must be non-empty")
    created_at = _parse_timestamp(session.get("created_at"), field="session.created_at")
    last_activity = _parse_timestamp(
        session.get("last_activity"), field="session.last_activity"
    )
    if last_activity < created_at:
        raise _fail("HITL session activity precedes session creation")
    if health_time < decision_time:
        raise _fail("official stage_health timestamp precedes decision metadata")
    if (
        decision_time < created_at
        or health_time < created_at
        or decision_time > last_activity
        or health_time > last_activity
    ):
        raise _fail("official stage metadata lies outside the HITL session timeline")

    interventions_raw = _read_bytes(
        resolved["interventions"],
        field="interventions",
        maximum=_MAX_JSONL_BYTES,
    )
    interventions = _parse_interventions(interventions_raw)
    intervention_count = session.get("interventions_count")
    if (
        isinstance(intervention_count, bool)
        or not isinstance(intervention_count, int)
        or intervention_count != len(interventions)
    ):
        raise _fail("session.interventions_count does not match interventions.jsonl")
    approval = _human_approval(
        interventions,
        expected_stage=expected_stage,
        stage_name=SUPPORTED_STAGES[expected_stage].upper(),
        created_at=created_at,
        stage_ready_at=max(decision_time, health_time),
        last_activity=last_activity,
    )

    expected_stage_files = [
        {"path": fixed_paths["decision"], "sha256": artifact_hashes["decision"]},
        {
            "path": fixed_paths["stage_health"],
            "sha256": artifact_hashes["stage_health"],
        },
        *[
            {
                "path": f"{stage_dir}/{name}",
                "sha256": digest,
            }
            for name, digest in sorted(stage_output_hashes.items())
        ],
    ]
    handoff_binding = _validate_gate_handoff(
        bundle_root=bundle_root,
        descriptor=artifacts["gate_handoff"],
        expected_stage=expected_stage,
        run_id=run_id,
        session_id=session_id,
        expected_stage_files=expected_stage_files,
        intervention_index=next(
            index for index, item in enumerate(interventions) if item is approval
        ),
        intervention_sha256=_canonical_record_sha256(approval),
    )

    receipt_hash = hashlib.sha256(receipt_raw).hexdigest()
    report = {
        "schema_version": REPORT_SCHEMA,
        "validated": True,
        "official_control": True,
        "stage": expected_stage,
        "stage_id": expected_stage_id,
        "stage_name": SUPPORTED_STAGES[expected_stage].upper(),
        "run_id": run_id,
        "session_id": session_id,
        "mode": "co-pilot",
        "auto_approve": False,
        # ARC v0.5.0 deliberately leaves the native gate record blocked after
        # the post-stage HITL hook.  The accepted intervention and bridge
        # handoff prove the control decision; the separate signed scientific
        # gate carries any PROCEED/PIVOT research decision.
        "decision": "proceed",
        "autoresearchclaw": {
            "repository": ARC_REPOSITORY,
            "version": ARC_VERSION,
            "commit": ARC_COMMIT,
        },
        "acp": {
            "acpx_version": ACPX_VERSION,
            "claude_adapter_version": CLAUDE_ADAPTER_VERSION,
            "codex_adapter_version": CODEX_ADAPTER_VERSION,
            "package_lock_sha256": ACP_PACKAGE_LOCK_SHA256,
        },
        "receipt_sha256": receipt_hash,
        "gate_handoff": handoff_binding,
        "control_artifact_sha256": artifact_hashes,
        "stage_output_sha256": dict(sorted(stage_output_hashes.items())),
        "human_approval": {
            "intervention_id": approval["id"],
            "timestamp": approval["timestamp"],
            "pause_reason": approval["pause_reason"],
        },
    }
    return _attach_gate_lineage(report, previous_report)


def validate_arc_control_report(
    value: Mapping[str, Any],
    expected_stage: int,
    previous_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the normalized output of :func:`validate_arc_control_bundle`.

    The report remains meaningful only when its producing DAG envelope is also
    authenticated.  This function lets each scientific gate bind that exact
    report and rejects a wrong-stage or partially copied control record.
    """

    if expected_stage not in SUPPORTED_STAGES:
        raise _fail(f"expected_stage must be one of {sorted(SUPPORTED_STAGES)}")
    report = _require_object(value, field="ARC control report")
    expected_keys = {
        "schema_version", "validated", "official_control", "stage", "stage_id",
        "stage_name", "run_id", "session_id", "mode", "auto_approve", "decision",
        "autoresearchclaw", "acp", "receipt_sha256", "control_artifact_sha256",
        "stage_output_sha256", "human_approval", "gate_handoff", "gate_lineage",
    }
    _require_exact_keys(report, expected_keys, field="ARC control report")
    if report["schema_version"] != REPORT_SCHEMA:
        raise _fail(f"ARC control report schema must be {REPORT_SCHEMA!r}")
    if report["validated"] is not True or report["official_control"] is not True:
        raise _fail("ARC control report is not an official validated receipt")
    stage_name = SUPPORTED_STAGES[expected_stage]
    if (
        report["stage"] != expected_stage
        or report["stage_id"] != f"{expected_stage:02d}-{stage_name}"
        or report["stage_name"] != stage_name.upper()
    ):
        raise _fail(f"ARC control report does not bind Stage {expected_stage}")
    if report["mode"] != "co-pilot" or report["auto_approve"] is not False:
        raise _fail("ARC control report is not a manual co-pilot run")
    for field in ("run_id", "session_id"):
        if not isinstance(report[field], str) or not report[field].strip():
            raise _fail(f"ARC control report {field} must be non-empty")
    if report["decision"] != "proceed":
        raise _fail("ARC control report has an invalid successful decision")
    _validate_pins({
        "autoresearchclaw": report["autoresearchclaw"],
        "acp": report["acp"],
    })
    _full_sha256(report["receipt_sha256"], field="receipt_sha256")
    handoff = _require_object(report["gate_handoff"], field="gate_handoff")
    _require_exact_keys(
        handoff,
        {
            "schema_version",
            "path",
            "sha256",
            "previous_stage",
            "previous_handoff_sha256",
            "response_sha256",
            "waiting_sha256",
            "consumption_receipt_sha256",
            "checkpoint_sha256",
        },
        field="gate_handoff",
    )
    if handoff["schema_version"] != GATE_HANDOFF_SCHEMA:
        raise _fail(f"gate_handoff schema must be {GATE_HANDOFF_SCHEMA!r}")
    expected_handoff_path = (
        f"control/gate-handoff-stage-{expected_stage:02d}.v2.json"
    )
    if handoff["path"] != expected_handoff_path:
        raise _fail("ARC control report gate_handoff path is stage-inconsistent")
    for field in (
        "sha256",
        "response_sha256",
        "waiting_sha256",
        "consumption_receipt_sha256",
        "checkpoint_sha256",
    ):
        _full_sha256(handoff[field], field=f"gate_handoff.{field}")
    stage_index = ORDERED_STAGES.index(expected_stage)
    expected_previous_stage = ORDERED_STAGES[stage_index - 1] if stage_index else None
    if handoff["previous_stage"] != expected_previous_stage:
        raise _fail("ARC control report handoff predecessor is not the prior gate")
    if expected_previous_stage is None:
        if handoff["previous_handoff_sha256"] is not None:
            raise _fail("Stage 5 handoff must not bind a predecessor")
    else:
        _full_sha256(
            handoff["previous_handoff_sha256"],
            field="gate_handoff.previous_handoff_sha256",
        )

    gate_lineage = _require_object(report["gate_lineage"], field="gate_lineage")
    _require_exact_keys(
        gate_lineage,
        {
            "schema_version",
            "previous_stage",
            "previous_report_sha256",
            "previous_receipt_sha256",
            "previous_handoff_sha256",
            "previous_chain_sha256",
            "chain_sha256",
        },
        field="gate_lineage",
    )
    if gate_lineage["schema_version"] != GATE_LINEAGE_SCHEMA:
        raise _fail(f"gate_lineage schema must be {GATE_LINEAGE_SCHEMA!r}")
    if gate_lineage["previous_stage"] != expected_previous_stage:
        raise _fail("ARC control report lineage skips or reorders a formal gate")
    predecessor_fields = (
        "previous_report_sha256",
        "previous_receipt_sha256",
        "previous_handoff_sha256",
        "previous_chain_sha256",
    )
    if expected_previous_stage is None:
        if any(gate_lineage[field] is not None for field in predecessor_fields):
            raise _fail("Stage 5 formal lineage must not have a predecessor")
    else:
        for field in predecessor_fields:
            _full_sha256(gate_lineage[field], field=f"gate_lineage.{field}")
        if (
            gate_lineage["previous_handoff_sha256"]
            != handoff["previous_handoff_sha256"]
        ):
            raise _fail("report and handoff predecessor hashes disagree")
    expected_chain_sha256 = _canonical_sha256(
        _gate_lineage_material(report, gate_lineage)
    )
    if gate_lineage["chain_sha256"] != expected_chain_sha256:
        raise _fail("ARC control report gate lineage hash is invalid")
    control_hashes = _require_object(
        report["control_artifact_sha256"], field="control_artifact_sha256"
    )
    _require_exact_keys(
        control_hashes, set(_CONTROL_ARTIFACTS), field="control_artifact_sha256"
    )
    for key, digest in control_hashes.items():
        _full_sha256(digest, field=f"control_artifact_sha256.{key}")
    output_hashes = _require_object(
        report["stage_output_sha256"], field="stage_output_sha256"
    )
    if not output_hashes:
        raise _fail("stage_output_sha256 must not be empty")
    for key, digest in output_hashes.items():
        _safe_relative_path(key, field="stage_output_sha256 key")
        _full_sha256(digest, field=f"stage_output_sha256.{key}")
    approval = _require_object(report["human_approval"], field="human_approval")
    _require_exact_keys(
        approval, {"intervention_id", "timestamp", "pause_reason"},
        field="human_approval",
    )
    if (
        not isinstance(approval["intervention_id"], str)
        or not approval["intervention_id"].strip()
        or approval["pause_reason"] != "gate_approval"
    ):
        raise _fail("ARC report lacks a human gate approval")
    _parse_timestamp(approval["timestamp"], field="human_approval.timestamp")
    normalized = dict(report)
    if previous_report is not None or expected_previous_stage is None:
        expected = _attach_gate_lineage(normalized, previous_report)
        if expected["gate_lineage"] != normalized["gate_lineage"]:
            raise _fail("ARC control report does not bind the supplied predecessor")
    return normalized


def validate_arc_control_chain(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Independently replay the complete Stage 5/9/15/20 formal gate chain."""

    if len(reports) != len(ORDERED_STAGES):
        raise _fail(
            f"formal ARC chain must contain exactly stages {list(ORDERED_STAGES)}"
        )
    normalized: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for expected_stage, report in zip(ORDERED_STAGES, reports, strict=True):
        current = validate_arc_control_report(
            report,
            expected_stage,
            previous_report=previous,
        )
        normalized.append(current)
        previous = current
    identities = {(item["run_id"], item["session_id"]) for item in normalized}
    if len(identities) != 1:
        raise _fail("formal ARC gates do not share one run_id/session_id")
    return normalized


__all__ = [
    "ACP_PACKAGE_LOCK_SHA256",
    "ACPX_VERSION",
    "ARC_COMMIT",
    "ARC_REPOSITORY",
    "ARC_VERSION",
    "ArcControlValidationError",
    "CLAUDE_ADAPTER_VERSION",
    "CODEX_ADAPTER_VERSION",
    "GATE_HANDOFF_SCHEMA",
    "GATE_LINEAGE_SCHEMA",
    "ORDERED_STAGES",
    "RECEIPT_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPORTED_STAGES",
    "WAITING_LINEAGE_SCHEMA",
    "WAITING_RECEIPT_SCHEMA",
    "WAITING_REPORT_SCHEMA",
    "validate_arc_control_bundle",
    "validate_arc_control_chain",
    "validate_arc_control_report",
    "validate_arc_waiting_bundle",
    "validate_arc_waiting_report",
]
