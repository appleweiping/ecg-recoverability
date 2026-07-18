"""Fail-closed validation for official AutoResearchClaw control receipts.

The validator does not infer success from an ARC checkout, configuration file, or
console log.  A bundle is accepted only when it contains ``receipt.v1.json`` and
the hash-bound official files produced by one successful ARC v0.5.0 stage::

    {
      "schema_version": "autoresearchclaw-control-receipt-v1",
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
        "stage_outputs": [
          {"path": "stage-05/screened_papers.json", "sha256": "..."}
        ]
      }
    }

All descriptor paths are POSIX-style paths relative to the bundle root.  The
four fixed descriptors must preserve ARC's native run-directory layout, and
``stage_outputs`` must exactly cover ``decision.json:output_artifacts``.  This
receipt is an integrity/provenance contract, not a replacement for the separate
author-signed scientific stage-gate decision.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "autoresearchclaw-control-receipt-v1"
REPORT_SCHEMA = "autoresearchclaw-control-report-v1"
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
    return max(matches, key=lambda item: item[0])[1]


def validate_arc_control_bundle(
    bundle_dir: Path | str,
    expected_stage: int,
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
        artifacts, {*_CONTROL_ARTIFACTS, "stage_outputs"}, field="artifacts"
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
    if decision.get("status") != "done":
        raise _fail("official ARC stage did not finish with status 'done'")
    allowed_decisions = (
        ("proceed", "refine", "pivot") if expected_stage == 15 else ("proceed",)
    )
    if decision.get("decision") not in allowed_decisions:
        raise _fail("official ARC stage has no accepted successful decision")
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
    if health.get("status") != "done" or health.get("error") not in (None, ""):
        raise _fail("official ARC stage_health is not successful")
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
    if session.get("waiting") is not None:
        raise _fail("official HITL session is still waiting for human input")
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

    receipt_hash = hashlib.sha256(receipt_raw).hexdigest()
    return {
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
        "decision": decision["decision"],
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
        "control_artifact_sha256": artifact_hashes,
        "stage_output_sha256": dict(sorted(stage_output_hashes.items())),
        "human_approval": {
            "intervention_id": approval["id"],
            "timestamp": approval["timestamp"],
            "pause_reason": approval["pause_reason"],
        },
    }


def validate_arc_control_report(
    value: Mapping[str, Any],
    expected_stage: int,
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
        "stage_output_sha256", "human_approval",
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
    allowed_decisions = (
        {"proceed", "refine", "pivot"} if expected_stage == 15 else {"proceed"}
    )
    if report["decision"] not in allowed_decisions:
        raise _fail("ARC control report has an invalid successful decision")
    _validate_pins({
        "autoresearchclaw": report["autoresearchclaw"],
        "acp": report["acp"],
    })
    _full_sha256(report["receipt_sha256"], field="receipt_sha256")
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
    return dict(report)


__all__ = [
    "ACP_PACKAGE_LOCK_SHA256",
    "ACPX_VERSION",
    "ARC_COMMIT",
    "ARC_REPOSITORY",
    "ARC_VERSION",
    "ArcControlValidationError",
    "CLAUDE_ADAPTER_VERSION",
    "CODEX_ADAPTER_VERSION",
    "RECEIPT_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPORTED_STAGES",
    "validate_arc_control_bundle",
    "validate_arc_control_report",
]
