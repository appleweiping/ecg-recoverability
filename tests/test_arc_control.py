from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest

from ecgcert.arc_control import (
    ACP_PACKAGE_LOCK_SHA256,
    ACPX_VERSION,
    ARC_COMMIT,
    ARC_REPOSITORY,
    ARC_VERSION,
    CLAUDE_ADAPTER_VERSION,
    CODEX_ADAPTER_VERSION,
    RECEIPT_SCHEMA,
    REPORT_SCHEMA,
    ArcControlValidationError,
    validate_arc_control_bundle,
    validate_arc_control_report,
)


STAGE_NAMES = {
    5: "literature_screen",
    9: "experiment_design",
    15: "research_decision",
    20: "quality_gate",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(bundle: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(bundle / relative)}


def _make_bundle(root: Path, *, stage: int = 5, status: str = "done") -> Path:
    bundle = root / f"arc-stage-{stage}"
    stage_dir = f"stage-{stage:02d}"
    run_id = "rc-20260719-120000-control"
    output_name = "scientific_output.json"
    _write_json(bundle / stage_dir / output_name, {"stage": stage, "evidence": "frozen"})
    _write_json(
        bundle / stage_dir / "decision.json",
        {
            "stage_id": f"{stage:02d}-{STAGE_NAMES[stage]}",
            "run_id": run_id,
            "status": status,
            "decision": "proceed" if status == "done" else "retry",
            "output_artifacts": [output_name] if status == "done" else [],
            "evidence_refs": [f"{stage_dir}/{output_name}"] if status == "done" else [],
            "error": None if status == "done" else "Queue owner disconnected",
            "ts": "2026-07-19T04:01:00+00:00",
            "next_stage": stage + 1 if status == "done" else stage,
        },
    )
    _write_json(
        bundle / stage_dir / "stage_health.json",
        {
            "stage_id": f"{stage:02d}-{STAGE_NAMES[stage]}",
            "run_id": run_id,
            "duration_sec": 61.25,
            "status": status,
            "artifacts_count": 1 if status == "done" else 0,
            "error": None if status == "done" else "Queue owner disconnected",
            "timestamp": "2026-07-19T04:01:01+00:00",
        },
    )
    _write_json(
        bundle / "hitl" / "session.json",
        {
            "session_id": "session-control-1",
            "run_id": run_id,
            "state": "active",
            "mode": "co-pilot",
            "interventions_count": 1,
            "human_edits": [],
            "total_human_time_sec": 12.5,
            "created_at": "2026-07-19T04:00:00+00:00",
            "last_activity": "2026-07-19T04:03:00+00:00",
            "waiting": None,
        },
    )
    intervention = {
        "id": "human-review-1",
        "type": "approve",
        "stage": stage,
        "stage_name": STAGE_NAMES[stage].upper(),
        "timestamp": "2026-07-19T04:02:00+00:00",
        "human_input": {
            "action": "approve",
            "message": "Reviewed the frozen stage evidence.",
            "guidance": "",
            "edited_files": {},
            "config_changes": {},
            "resources": [],
            "rollback_to_stage": None,
            "timestamp": "2026-07-19T04:02:00+00:00",
        },
        "pause_reason": "gate_approval",
        "stage_output_summary": "Frozen output reviewed.",
        "quality_score": 9.0,
        "confidence_score": 0.9,
        "outcome": "Human chose: approve",
        "accepted": True,
        "duration_sec": 12.5,
    }
    interventions_path = bundle / "hitl" / "interventions.jsonl"
    interventions_path.parent.mkdir(parents=True, exist_ok=True)
    interventions_path.write_text(
        json.dumps(intervention, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
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
        "invocation": {"mode": "co-pilot", "auto_approve": False},
        "run": {"run_id": run_id, "stage": stage},
        "artifacts": {
            "decision": _descriptor(bundle, f"{stage_dir}/decision.json"),
            "stage_health": _descriptor(bundle, f"{stage_dir}/stage_health.json"),
            "session": _descriptor(bundle, "hitl/session.json"),
            "interventions": _descriptor(bundle, "hitl/interventions.jsonl"),
            "stage_outputs": [_descriptor(bundle, f"{stage_dir}/{output_name}")],
        },
    }
    _write_json(bundle / "receipt.v1.json", receipt)
    return bundle


def _mutate_receipt(bundle: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = bundle / "receipt.v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutation(receipt)
    _write_json(path, receipt)


def _rehash_control(bundle: Path, name: str) -> None:
    def update(receipt: dict[str, Any]) -> None:
        descriptor = receipt["artifacts"][name]
        descriptor["sha256"] = _sha256(bundle / descriptor["path"])

    _mutate_receipt(bundle, update)


@pytest.mark.parametrize("stage", [5, 9, 15, 20])
def test_validate_official_control_bundle(stage: int, tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, stage=stage)

    report = validate_arc_control_bundle(bundle, stage)

    assert report["schema_version"] == REPORT_SCHEMA
    assert report["validated"] is True
    assert report["official_control"] is True
    assert report["stage"] == stage
    assert report["stage_id"] == f"{stage:02d}-{STAGE_NAMES[stage]}"
    assert report["run_id"] == "rc-20260719-120000-control"
    assert report["mode"] == "co-pilot"
    assert report["auto_approve"] is False
    assert report["autoresearchclaw"]["commit"] == ARC_COMMIT
    assert report["human_approval"]["intervention_id"] == "human-review-1"
    assert set(report["control_artifact_sha256"]) == {
        "decision",
        "stage_health",
        "session",
        "interventions",
    }
    assert set(report["stage_output_sha256"]) == {"scientific_output.json"}
    assert report["receipt_sha256"] == _sha256(bundle / "receipt.v1.json")


def test_stage_15_accepts_successful_pivot_direction(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, stage=15)
    decision_path = bundle / "stage-15" / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "pivot"
    _write_json(decision_path, decision)
    _rehash_control(bundle, "decision")

    report = validate_arc_control_bundle(bundle, 15)

    assert report["decision"] == "pivot"


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    (bundle / "stage-05" / "scientific_output.json").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ArcControlValidationError, match="sha256 mismatch"):
        validate_arc_control_bundle(bundle, 5)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../decision.json", "parent path component"),
        ("C:/decision.json", "portable POSIX"),
        ("stage-05\\decision.json", "portable POSIX"),
        ("/stage-05/decision.json", "empty, current, or parent"),
    ],
)
def test_descriptor_paths_cannot_escape_or_change_path_dialect(
    path: str, message: str, tmp_path: Path
) -> None:
    bundle = _make_bundle(tmp_path)
    _mutate_receipt(
        bundle, lambda receipt: receipt["artifacts"]["decision"].update(path=path)
    )

    with pytest.raises(ArcControlValidationError, match=message):
        validate_arc_control_bundle(bundle, 5)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt["autoresearchclaw"].update(commit="0" * 40),
        lambda receipt: receipt["autoresearchclaw"].update(version="0.5.1"),
        lambda receipt: receipt["acp"].update(acpx_version="0.12.1"),
        lambda receipt: receipt["acp"].update(package_lock_sha256="0" * 64),
        lambda receipt: receipt["invocation"].update(mode="full-auto"),
        lambda receipt: receipt["invocation"].update(auto_approve=True),
    ],
)
def test_unpinned_or_automatic_control_fails_closed(
    mutation: Callable[[dict[str, Any]], None], tmp_path: Path
) -> None:
    bundle = _make_bundle(tmp_path)
    _mutate_receipt(bundle, mutation)

    with pytest.raises(ArcControlValidationError):
        validate_arc_control_bundle(bundle, 5)


def test_caller_expected_stage_cannot_be_overridden_by_receipt(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, stage=5)

    with pytest.raises(ArcControlValidationError, match="expected ARC Stage 9"):
        validate_arc_control_bundle(bundle, 9)


def test_failed_official_stage_fails_even_when_files_are_hash_bound(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, status="failed")
    # A failed probe has no legitimate stage output, so its receipt is adjusted
    # consistently; semantic success must still fail before it can become evidence.
    _mutate_receipt(
        bundle,
        lambda receipt: receipt["artifacts"].update(stage_outputs=[]),
    )

    with pytest.raises(ArcControlValidationError, match="status 'done'"):
        validate_arc_control_bundle(bundle, 5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "reject"),
        ("accepted", False),
        ("pause_reason", "post_stage"),
        ("outcome", "Human chose: abort"),
    ],
)
def test_human_gate_approval_is_mandatory(
    field: str, value: Any, tmp_path: Path
) -> None:
    bundle = _make_bundle(tmp_path)
    path = bundle / "hitl" / "interventions.jsonl"
    intervention = json.loads(path.read_text(encoding="utf-8"))
    intervention[field] = value
    path.write_text(
        json.dumps(intervention, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _rehash_control(bundle, "interventions")

    with pytest.raises(ArcControlValidationError, match="no accepted human"):
        validate_arc_control_bundle(bundle, 5)


def test_stage_outputs_must_exactly_cover_decision_outputs(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    extra = bundle / "stage-05" / "unclaimed.json"
    _write_json(extra, {"not": "declared by ARC"})
    _mutate_receipt(
        bundle,
        lambda receipt: receipt["artifacts"]["stage_outputs"].append(
            {"path": "stage-05/unclaimed.json", "sha256": _sha256(extra)}
        ),
    )

    with pytest.raises(ArcControlValidationError, match="exactly cover"):
        validate_arc_control_bundle(bundle, 5)


def test_current_failed_probe_directory_is_not_a_control_bundle() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ArcControlValidationError):
        validate_arc_control_bundle(repository_root / "arc_audit", 5)


def test_cli_emits_immutable_normalized_report(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    output = tmp_path / "normalized" / "stage5-control.json"
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "validate_arc_control.py"),
            "--bundle",
            str(bundle),
            "--stage",
            "5",
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validated"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["stage"] == 5
    repeated = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "validate_arc_control.py"),
            "--bundle",
            str(bundle),
            "--stage",
            "5",
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert repeated.returncode != 0
    assert "already exists and is immutable" in repeated.stderr


def test_normalized_report_is_stage_bound_and_fail_closed(tmp_path: Path) -> None:
    report = validate_arc_control_bundle(_make_bundle(tmp_path), 5)
    assert validate_arc_control_report(report, 5)["official_control"] is True
    report["stage"] = 9
    with pytest.raises(ArcControlValidationError, match="does not bind Stage 5"):
        validate_arc_control_report(report, 5)
