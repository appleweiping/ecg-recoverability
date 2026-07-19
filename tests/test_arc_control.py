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
    validate_arc_control_chain,
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


def _make_bundle(
    root: Path,
    *,
    stage: int = 5,
    status: str = "blocked_approval",
    previous_handoff_sha256: str | None = None,
    decision_value: str | None = None,
    run_id: str = "rc-20260719-120000-control",
    session_id: str = "session-control-1",
) -> Path:
    bundle = root / f"arc-stage-{stage}"
    stage_dir = f"stage-{stage:02d}"
    output_name = "scientific_output.json"
    _write_json(bundle / stage_dir / output_name, {"stage": stage, "evidence": "frozen"})
    _write_json(
        bundle / stage_dir / "decision.json",
        {
            "stage_id": f"{stage:02d}-{STAGE_NAMES[stage]}",
            "run_id": run_id,
            "status": status,
            "decision": decision_value or (
                "block" if status == "blocked_approval" else "retry"
            ),
            "output_artifacts": (
                [output_name] if status == "blocked_approval" else []
            ),
            "evidence_refs": (
                [f"{stage_dir}/{output_name}"] if status == "blocked_approval" else []
            ),
            "error": (
                None if status == "blocked_approval" else "Queue owner disconnected"
            ),
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
            "artifacts_count": 1 if status == "blocked_approval" else 0,
            "error": (
                None if status == "blocked_approval" else "Queue owner disconnected"
            ),
            "timestamp": "2026-07-19T04:01:01+00:00",
        },
    )
    _write_json(
        bundle / "hitl" / "session.json",
        {
            "session_id": session_id,
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
    intervention_sha256 = hashlib.sha256(
        json.dumps(
            intervention,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    prior_stages = [candidate for candidate in STAGE_NAMES if candidate < stage]
    prior_stage = prior_stages[-1] if prior_stages else None
    if prior_stage is not None and previous_handoff_sha256 is None:
        previous_handoff_sha256 = "d" * 64
    handoff_relative = f"control/gate-handoff-stage-{stage:02d}.v2.json"
    waiting_sha256 = "2" * 64
    nonce = "3" * 64
    _write_json(
        bundle / "checkpoint.json",
        {"last_completed_stage": stage - 1, "frozen": True},
    )
    checkpoint_sha256 = _sha256(bundle / "checkpoint.json")
    response = {
        "schema_version": "arc-operator-response-v2",
        "stage": stage,
        "run_id": run_id,
        "session_id": session_id,
        "waiting_sha256": waiting_sha256,
        "preapproval_checkpoint_sha256": checkpoint_sha256,
        "nonce": nonce,
        "action": "approve",
        "issued_at": "2026-07-19T04:02:00+00:00",
        "message": "Reviewed the frozen stage evidence.",
    }
    response_staging = (
        bundle
        / "control"
        / "operator-response-snapshots"
        / f"stage-{stage:02d}-pending.v2.json"
    )
    _write_json(response_staging, response)
    response_sha256 = _sha256(response_staging)
    response_relative = (
        f"control/operator-response-snapshots/stage-{stage:02d}-"
        f"{response_sha256}.v2.json"
    )
    response_staging.replace(bundle / response_relative)
    consumption_relative = (
        f"control/operator-response-consumption/stage-{stage:02d}-"
        + waiting_sha256
        + ".v1.json"
    )
    _write_json(
        bundle / consumption_relative,
        {
            "schema_version": "arc-operator-response-consumption-v1",
            "claimed_at": "2026-07-19T04:02:01+00:00",
            "response_path": response_relative,
            "response_sha256": response_sha256,
            "stage": stage,
            "run_id": run_id,
            "session_id": session_id,
            "waiting_sha256": waiting_sha256,
            "preapproval_checkpoint_sha256": checkpoint_sha256,
            "nonce": nonce,
            "response": response,
        },
    )
    consumption_sha256 = _sha256(bundle / consumption_relative)
    stage_files = [
        _descriptor(bundle, f"{stage_dir}/decision.json"),
        _descriptor(bundle, f"{stage_dir}/stage_health.json"),
    ]
    if status == "blocked_approval":
        stage_files.append(_descriptor(bundle, f"{stage_dir}/{output_name}"))
    next_stage_names = {
        5: "KNOWLEDGE_EXTRACT",
        9: "CODE_GENERATION",
        15: "PAPER_OUTLINE",
        20: "KNOWLEDGE_ARCHIVE",
    }
    _write_json(
        bundle / handoff_relative,
        {
            "schema_version": "arc-gate-handoff-v2",
            "created_at": "2026-07-19T04:02:30+00:00",
            "stage": stage,
            "stage_name": STAGE_NAMES[stage].upper(),
            "native_identity": {
                "run_id": run_id,
                "session_id": session_id,
            },
            "next_stage": stage + 1,
            "next_stage_name": next_stage_names[stage],
            "stage_files": stage_files,
            "approval": {
                "response_sha256": response_sha256,
                "waiting_sha256": waiting_sha256,
                "run_id": run_id,
                "session_id": session_id,
                "nonce": nonce,
                "consumption_receipt_path": consumption_relative,
                "consumption_receipt_sha256": consumption_sha256,
                "native_intervention_ordinal": 0,
                "native_intervention_sha256": intervention_sha256,
                "bridge_event_ordinal": 0,
                "bridge_event_sha256": "6" * 64,
            },
            "checkpoint": {
                "path": "checkpoint.json",
                "sha256": checkpoint_sha256,
                "last_completed_stage": stage - 1,
            },
            "lineage": {
                "source_config_snapshot_sha256": "8" * 64,
                "effective_config_sha256": "9" * 64,
                "project_state_sha256": "a" * 64,
                "previous_handoff_path": (
                    f"control/gate-handoff-stage-{prior_stage:02d}.v2.json"
                    if prior_stage is not None
                    else None
                ),
                "previous_handoff_sha256": previous_handoff_sha256,
            },
        },
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
            "gate_handoff": _descriptor(bundle, handoff_relative),
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


def _validated_through(
    root: Path, stage: int, *, final_decision: str | None = None
) -> tuple[Path, dict[str, Any]]:
    previous: dict[str, Any] | None = None
    bundle: Path | None = None
    for candidate in STAGE_NAMES:
        if candidate > stage:
            break
        bundle = _make_bundle(
            root,
            stage=candidate,
            previous_handoff_sha256=(
                previous["gate_handoff"]["sha256"] if previous is not None else None
            ),
            decision_value=final_decision if candidate == stage else None,
        )
        previous = validate_arc_control_bundle(
            bundle,
            candidate,
            previous_report=previous,
        )
    assert bundle is not None and previous is not None
    return bundle, previous


def _validated_reports(
    root: Path,
    *,
    run_id: str = "rc-20260719-120000-control",
    session_id: str = "session-control-1",
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for stage in STAGE_NAMES:
        bundle = _make_bundle(
            root,
            stage=stage,
            previous_handoff_sha256=(
                previous["gate_handoff"]["sha256"] if previous is not None else None
            ),
            run_id=run_id,
            session_id=session_id,
        )
        previous = validate_arc_control_bundle(
            bundle,
            stage,
            previous_report=previous,
        )
        reports.append(previous)
    return reports


@pytest.mark.parametrize("stage", [5, 9, 15, 20])
def test_validate_official_control_bundle(stage: int, tmp_path: Path) -> None:
    bundle, report = _validated_through(tmp_path, stage)

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


def test_stage_15_control_approval_is_separate_from_scientific_pivot(
    tmp_path: Path,
) -> None:
    _bundle, report = _validated_through(tmp_path, 15)

    assert report["decision"] == "proceed"


def test_complete_formal_gate_chain_rejects_cross_run_splice(tmp_path: Path) -> None:
    first = _validated_reports(
        tmp_path / "first", run_id="rc-first", session_id="session-first"
    )
    second = _validated_reports(
        tmp_path / "second", run_id="rc-second", session_id="session-second"
    )

    with pytest.raises(
        ArcControlValidationError,
        match="run_id/session_id changed|supplied predecessor",
    ):
        validate_arc_control_chain([first[0], first[1], second[2], second[3]])


def test_complete_formal_gate_chain_rejects_tampered_hash_link(tmp_path: Path) -> None:
    reports = _validated_reports(tmp_path)
    tampered = json.loads(json.dumps(reports))
    tampered[2]["gate_lineage"]["previous_handoff_sha256"] = "f" * 64

    with pytest.raises(
        ArcControlValidationError,
        match="predecessor hashes disagree|lineage hash is invalid|supplied predecessor",
    ):
        validate_arc_control_chain(tampered)


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    (bundle / "stage-05" / "scientific_output.json").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ArcControlValidationError, match="sha256 mismatch"):
        validate_arc_control_bundle(bundle, 5)


@pytest.mark.parametrize(
    "artifact",
    ["operator_response_snapshot", "consumption_receipt", "checkpoint_snapshot"],
)
def test_transitive_approval_artifact_tampering_fails_closed(
    artifact: str, tmp_path: Path
) -> None:
    bundle = _make_bundle(tmp_path)
    handoff = json.loads(
        (bundle / "control" / "gate-handoff-stage-05.v2.json").read_text(
            encoding="utf-8"
        )
    )
    if artifact == "operator_response_snapshot":
        consumption = json.loads(
            (bundle / handoff["approval"]["consumption_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        path = bundle / consumption["response_path"]
    elif artifact == "consumption_receipt":
        path = bundle / handoff["approval"]["consumption_receipt_path"]
    else:
        path = bundle / handoff["checkpoint"]["path"]
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArcControlValidationError, match="sha256 mismatch"):
        validate_arc_control_bundle(bundle, 5)


def test_gate_handoff_stage_files_must_exactly_match_receipt(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    extra = bundle / "stage-05" / "unclaimed.json"
    _write_json(extra, {"not": "declared by ARC"})
    handoff_path = bundle / "control" / "gate-handoff-stage-05.v2.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["stage_files"].append(
        {"path": "stage-05/unclaimed.json", "sha256": _sha256(extra)}
    )
    _write_json(handoff_path, handoff)
    _rehash_control(bundle, "gate_handoff")

    with pytest.raises(ArcControlValidationError, match="artifact set or hashes"):
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

    with pytest.raises(ArcControlValidationError, match="blocked_approval/block"):
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
