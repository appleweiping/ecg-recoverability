from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
from pathlib import Path

import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.arc_control import (
    ACP_PACKAGE_LOCK_SHA256,
    ACPX_VERSION,
    ARC_COMMIT,
    ARC_REPOSITORY,
    ARC_VERSION,
    CLAUDE_ADAPTER_VERSION,
    CODEX_ADAPTER_VERSION,
    REPORT_SCHEMA,
)
from ecgcert.evaluation import (
    BootstrapEffect,
    method_specific_delta_r2,
    prediction_delta_r2,
    stage15_decision,
)
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.stage_gates import json_artifact_bytes, make_review, merge_review
from experiments.meta_analysis_v3 import SCHEMA_VERSION, stage15
from scripts.claim_sync_v3 import MACROS, _review_is_valid, synchronize_claims


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_artifact_bytes(value))
    return path


def _arc_stage15_report(path: Path) -> Path:
    digest = "a" * 64
    return _write(path, {
        "schema_version": REPORT_SCHEMA,
        "validated": True,
        "official_control": True,
        "stage": 15,
        "stage_id": "15-research_decision",
        "stage_name": "RESEARCH_DECISION",
        "run_id": "rc-test-stage15",
        "session_id": "session-test-stage15",
        "mode": "co-pilot",
        "auto_approve": False,
        "decision": "pivot",
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
        "receipt_sha256": digest,
        "control_artifact_sha256": {
            "decision": digest,
            "stage_health": digest,
            "session": digest,
            "interventions": digest,
        },
        "stage_output_sha256": {"research_decision.json": digest},
        "human_approval": {
            "intervention_id": "approval-test",
            "timestamp": "2026-07-19T04:02:00+00:00",
            "pause_reason": "gate_approval",
        },
    })


def _reviewed_stage15(tmp_path: Path, status: str, reviewer_keys) -> Path:
    evidence = {
        "ptbxl": {"point": 0.031 if status == "PROCEED" else -0.012, "ci95": [0.004, 0.055]},
        "external": {
            "chapman": {"point": 0.021, "ci95": [0.002, 0.041]},
            "cpsc2018": {"point": -0.004, "ci95": [-0.021, 0.013]},
        },
        "method_delta_r2": {
            "lowrank": 0.02,
            "ridge": 0.01,
            "masked-unet": 0.03,
            "imputeecg": -0.01,
        },
        "bootstrap_replicates": 2_000,
    }
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    gate = {
        "schema_version": "arc-stage15-v3",
        "stage": 15,
        "status": "PENDING_USER_REVIEW",
        "eligible_for_proceed": status == "PROCEED",
        "human_review_required": True,
        "review_deadline_hours": 24,
        "created_at": now.isoformat(timespec="seconds"),
        "automatic_decision": status,
        "automatic_reasons": [] if status == "PROCEED" else ["PTB-XL criterion failed"],
        "meta_analysis_sha256": "a" * 64,
        "evidence_sha256": lineage.canonical_sha256(evidence),
        "evidence": evidence,
    }
    automatic = _write(tmp_path / f"{status}-automatic.json", gate)
    review = make_review(
        gate,
        gate_sha256=lineage.artifact_sha256(automatic),
        reviewer="author",
        decision=status,
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        reviewed_at=now + timedelta(seconds=10),
    )
    approval = _write(tmp_path / f"{status}-approval.json", review)
    combined = merge_review(
        gate,
        review,
        gate_sha256=lineage.artifact_sha256(automatic),
        approval_sha256=lineage.artifact_sha256(approval),
        public_key_path=reviewer_keys.public,
    )
    assert _review_is_valid(combined, reviewer_keys.public)
    return _write(tmp_path / status / "decision.v3.json", combined)


def _source_registry(path: Path) -> Path:
    return _write(
        path,
        {
            "schema_version": "verified-registry-v1",
            "citations": {
                "source": {"status": "verified_primary", "source": "https://example.test"}
            },
            "numeric_claims": {
                macro: {
                    "status": "blocked_until_stage15_review",
                    "artifact": "artifacts/primary/meta_analysis/summary.v3.json",
                }
                for macro in MACROS
            },
        },
    )


def _figure_bundle(path: Path, *, stage15_sha256: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(FIGURE_ARTIFACTS):
        (path / name).write_bytes(f"figure-evidence-{index}".encode())
    _write(
        path / "summary.v3.json",
        {
            "schema_version": "paper-figures-v3",
            "input_sha256": {"stage15": stage15_sha256},
            "artifacts_sha256": direct_artifact_hashes(path),
        },
    )
    return path


@pytest.mark.parametrize("status", ["PROCEED", "PIVOT"])
def test_claim_sync_releases_reviewed_positive_or_negative_evidence(
    tmp_path: Path, status: str, reviewer_keys
) -> None:
    gate_path = _reviewed_stage15(tmp_path, status, reviewer_keys)
    figures = _figure_bundle(
        tmp_path / f"{status}-figures",
        stage15_sha256=lineage.artifact_sha256(gate_path),
    )
    output = tmp_path / f"{status}-claims"
    summary = synchronize_claims(
        stage15=gate_path.parent,
        figures=figures,
        output=output,
        source_registry=_source_registry(tmp_path / f"{status}-registry.json"),
        reviewer_public_key=reviewer_keys.public,
    )
    assert summary["status"] == status
    assert summary["submission_ready"] is True
    assert "PENDING" not in json.dumps(summary)
    assert "Delta" in summary["values"]["ResultIncrementalValue"]
    assert ("negative result" in summary["values"]["ResultPrimaryAssociation"]) == (
        status == "PIVOT"
    )
    macros = (output / "robust_map_placeholders.tex").read_text(encoding="utf-8")
    assert r"\ResultHeadline" in macros and r"\ResultConclusion" in macros
    assert r"95\%" in macros and r"95\\%" not in macros
    assert summary["claim_macros_sha256"] == lineage.artifact_sha256(
        output / "robust_map_placeholders.tex"
    )
    assert summary["figures_summary_sha256"] == lineage.artifact_sha256(
        figures / "summary.v3.json"
    )
    assert summary["figure_artifacts_sha256"] == direct_artifact_hashes(figures)
    registry = json.loads((output / "verified_registry.v1.json").read_text(encoding="utf-8"))
    for macro in MACROS:
        entry = registry["numeric_claims"][macro]
        assert entry["status"] == "verified_artifact"
        assert entry["stage15_sha256"] == lineage.artifact_sha256(gate_path)
        assert entry["stage15_status"] == status
        assert entry["value_sha256"] == lineage.canonical_sha256(summary["values"][macro])
        assert entry["claim_macros_sha256"] == summary["claim_macros_sha256"]


def test_claim_sync_rejects_unsigned_pivot_and_unbound_figures(
    tmp_path: Path, reviewer_keys
) -> None:
    gate_path = _reviewed_stage15(tmp_path, "PIVOT", reviewer_keys)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.pop("review_signature_ed25519")
    assert not _review_is_valid(gate, reviewer_keys.public)

    figures = _figure_bundle(
        tmp_path / "figures", stage15_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="figures.*Stage-15"):
        synchronize_claims(
            stage15=gate_path.parent,
            figures=figures,
            output=tmp_path / "claims",
            source_registry=_source_registry(tmp_path / "registry.json"),
            reviewer_public_key=reviewer_keys.public,
        )


@pytest.mark.parametrize("name", FIGURE_ARTIFACTS)
def test_claim_sync_rejects_tampered_figure_pdf_or_source_table(
    tmp_path: Path, name: str, reviewer_keys
) -> None:
    gate_path = _reviewed_stage15(tmp_path, "PROCEED", reviewer_keys)
    figures = _figure_bundle(
        tmp_path / "figures",
        stage15_sha256=lineage.artifact_sha256(gate_path),
    )
    with (figures / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="figure artifact hash mismatch"):
        synchronize_claims(
            stage15=gate_path.parent,
            figures=figures,
            output=tmp_path / "claims",
            source_registry=_source_registry(tmp_path / "registry.json"),
            reviewer_public_key=reviewer_keys.public,
        )


def test_automatic_pivot_still_opens_a_mandatory_24_hour_review(
    tmp_path: Path, reviewer_keys
) -> None:
    meta = tmp_path / "meta"
    artifact_paths = {
        "alpha_tuning": meta / "alpha_tuning.parquet",
        "ptbxl_predictions": meta / "ptbxl_predictions.parquet",
        "ptbxl_seed_predictions": meta / "ptbxl_seed_predictions.parquet",
        "effects": meta / "effects.parquet",
    }

    truth = [0.0, 1.0, 2.0, 3.0]
    prediction_rows = []
    augmented_by_method = {
        "lowrank": [3.0, 2.0, 1.0, 0.0],
        "ridge": [0.1, 0.9, 2.1, 2.9],
        "masked-unet": [3.0, 2.0, 1.0, 0.0],
        "imputeecg": [1.5, 1.5, 1.5, 1.5],
    }
    for method, augmented in augmented_by_method.items():
        for index, outcome in enumerate(truth):
            prediction_rows.append(
                {
                    "patient_id": f"patient-{index}",
                    "method": method,
                    "outcome_log_rmse": outcome,
                    "prediction_simple": 1.5,
                    "prediction_augmented": augmented[index],
                }
            )
    ptb_predictions = pd.DataFrame(prediction_rows)
    external_predictions = {
        "chapman": ptb_predictions.copy(),
        "cpsc2018": ptb_predictions.copy(),
    }

    artifact_paths["alpha_tuning"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"alpha": 0.1, "mean_mse": 1.0}]).to_parquet(
        artifact_paths["alpha_tuning"], index=False
    )
    ptb_predictions.to_parquet(artifact_paths["ptbxl_predictions"], index=False)
    ptb_predictions.to_parquet(artifact_paths["ptbxl_seed_predictions"], index=False)

    external_artifacts = {}
    for cohort, predictions in external_predictions.items():
        point = meta / f"{cohort}_predictions.parquet"
        seeds = meta / f"{cohort}_seed_predictions.parquet"
        predictions.to_parquet(point, index=False)
        predictions.to_parquet(seeds, index=False)
        external_artifacts[cohort] = {
            "point_predictions": {
                "path": point.name,
                "sha256": lineage.artifact_sha256(point),
            },
            "seed_predictions": {
                "path": seeds.name,
                "sha256": lineage.artifact_sha256(seeds),
            },
        }
    def effect(point: float, interval: list[float], seed: int) -> dict:
        return {"point": point, "ci95": interval, "replicates": 2_000, "seed": seed}

    ptb_effect = BootstrapEffect(
        prediction_delta_r2(ptb_predictions), (-0.3, 0.1), 2_000, 1
    )
    external_effects = {
        cohort: BootstrapEffect(
            prediction_delta_r2(predictions), (-0.3, 0.1), 2_000, seed
        )
        for seed, (cohort, predictions) in enumerate(external_predictions.items(), start=2)
    }
    method_deltas = method_specific_delta_r2(ptb_predictions)
    automatic = stage15_decision(
        ptbxl=ptb_effect,
        external=external_effects,
        method_deltas=method_deltas,
    )
    pd.DataFrame(
        [
            {
                "cohort": "PTB-XL",
                "point": ptb_effect.point,
                "ci95": ptb_effect.ci95,
                "replicates": ptb_effect.replicates,
                "seed": ptb_effect.seed,
            },
            *(
                {
                    "cohort": cohort,
                    "point": value.point,
                    "ci95": value.ci95,
                    "replicates": value.replicates,
                    "seed": value.seed,
                }
                for cohort, value in external_effects.items()
            ),
        ]
    ).to_parquet(artifact_paths["effects"], index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "automatic_stage15_status": automatic.status,
        "automatic_stage15_reasons": list(automatic.reasons),
        "ptbxl": effect(ptb_effect.point, list(ptb_effect.ci95), ptb_effect.seed),
        "external": {
            cohort: effect(value.point, list(value.ci95), value.seed)
            for cohort, value in external_effects.items()
        },
        "method_delta_r2": method_deltas,
        "common_panel_methods": ["lowrank", "ridge", "masked-unet", "imputeecg"],
        "bootstrap_replicates": 2_000,
        "release_contract_verified": True,
        "release_lineage": {"validated": True},
        "artifacts": {
            **{
                key: {"path": path.name, "sha256": lineage.artifact_sha256(path)}
                for key, path in artifact_paths.items()
            },
            "external": external_artifacts,
        },
    }
    _write(meta / "summary.v3.json", summary)
    output = tmp_path / "gate"
    stage15(SimpleNamespace(
        meta_analysis=meta,
        arc_control=_arc_stage15_report(tmp_path / "arc-stage15.json"),
        output_dir=output,
    ))
    gate = json.loads((output / "decision.v3.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PENDING_USER_REVIEW"
    assert gate["automatic_decision"] == "PIVOT"
    assert gate["eligible_for_proceed"] is False
    assert gate["human_review_required"] is True
    assert gate["review_deadline_hours"] == 24
    assert gate["evidence_sha256"] == lineage.canonical_sha256(gate["evidence"])
    with pytest.raises(ValueError, match="do not permit PROCEED"):
        make_review(
            gate,
            gate_sha256=lineage.artifact_sha256(output / "decision.v3.json"),
            reviewer="author",
            decision="PROCEED",
            private_key_path=reviewer_keys.private,
            public_key_path=reviewer_keys.public,
        )
    pivot_review = make_review(
        gate,
        gate_sha256=lineage.artifact_sha256(output / "decision.v3.json"),
        reviewer="author",
        decision="PIVOT",
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
    )
    assert pivot_review["decision"] == "PIVOT"

    summary["ptbxl"]["point"] += 0.1
    _write(meta / "summary.v3.json", summary)
    with pytest.raises(ValueError, match="authenticated evidence"):
        stage15(
            SimpleNamespace(
                meta_analysis=meta,
                output_dir=tmp_path / "tampered-gate",
            )
        )
