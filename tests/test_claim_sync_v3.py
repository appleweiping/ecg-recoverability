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
    WAITING_LINEAGE_SCHEMA,
    WAITING_REPORT_SCHEMA,
)
from ecgcert.evaluation import META_RIDGE_ALPHA_GRID, stage15_decision
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.stage_gates import json_artifact_bytes, make_review, merge_review
from experiments.meta_analysis_v3 import (
    META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
    META_BOOTSTRAP_SEED,
    META_BOOTSTRAP_SEED_OFFSETS,
    META_METRIC_COLUMNS,
    META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
    META_SEED_PREDICTION_SCHEMA_VERSION,
    META_SUFFICIENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _bootstrap_effect_and_draws_from_sufficient,
    _expected_common_seeds,
    _method_deltas_from_sufficient,
    _write_seed_evidence_and_sufficient,
    stage15,
)
from scripts.claim_sync_v3 import MACROS, _review_is_valid, synchronize_claims


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_artifact_bytes(value))
    return path


def _arc_stage15_report(path: Path) -> Path:
    digest = "a" * 64
    predecessor = {
        "stage": 9,
        "report_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "handoff_sha256": "b" * 64,
        "chain_sha256": "3" * 64,
    }
    report = {
        "schema_version": WAITING_REPORT_SCHEMA,
        "validated": True,
        "official_control": True,
        "phase": "waiting",
        "stage": 15,
        "stage_id": "15-research_decision",
        "stage_name": "RESEARCH_DECISION",
        "run_id": "rc-test-stage15",
        "session_id": "session-test-stage15",
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
        "waiting_receipt_sha256": digest,
        "control_artifact_sha256": {
            "decision": digest,
            "stage_health": digest,
            "session": digest,
            "waiting": "d" * 64,
            "operator_challenge": "e" * 64,
            "checkpoint": "f" * 64,
        },
        "stage_output_sha256": {"research_decision.json": digest},
        "waiting": {
            "sha256": "d" * 64,
            "since": "2026-07-19T04:00:00+00:00",
            "expires_at": "2026-07-20T04:00:00+00:00",
        },
        "challenge": {
            "sha256": "e" * 64,
            "preapproval_checkpoint_sha256": "f" * 64,
            "nonce": "9" * 64,
        },
    }
    waiting_lineage = {
        "schema_version": WAITING_LINEAGE_SCHEMA,
        "predecessor": predecessor,
    }
    waiting_lineage["chain_sha256"] = lineage.canonical_sha256(
        {
            "stage": 15,
            "run_id": "rc-test-stage15",
            "session_id": "session-test-stage15",
            "waiting_receipt_sha256": digest,
            "waiting_sha256": "d" * 64,
            "challenge_sha256": "e" * 64,
            "checkpoint_sha256": "f" * 64,
            "nonce": "9" * 64,
            "predecessor": predecessor,
        }
    )
    report["waiting_lineage"] = waiting_lineage
    return _write(path, report)


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
        "ptbxl_sufficient_stats": meta / "ptbxl_sufficient_stats.parquet",
        "ptbxl_paired_seed_sufficient": (
            meta / "ptbxl_paired_seed_sufficient.parquet"
        ),
        "bootstrap_draws": meta / "bootstrap_draws.parquet",
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
                    "cohort": "PTB-XL",
                    "partition": "test",
                    "patient_id": f"patient-{index}",
                    "method": method,
                    "segment": "QRS",
                    "configuration": "I",
                    "target": ("II", "III", "V1", "V2")[index],
                    "outcome_log_rmse": outcome,
                    "prediction_simple": 1.5,
                    "prediction_augmented": augmented[index],
                }
            )
    ptb_predictions = pd.DataFrame(prediction_rows)
    external_predictions = {}
    for cohort in ("chapman", "cpsc2018"):
        frame = ptb_predictions.copy()
        frame["cohort"] = cohort
        external_predictions[cohort] = frame
    artifact_paths["alpha_tuning"].parent.mkdir(parents=True, exist_ok=True)
    alpha_rows = []
    for alpha in META_RIDGE_ALPHA_GRID:
        mse = 1.0 if alpha == 0.1 else 2.0 + float(alpha)
        alpha_rows.append(
            {
                "alpha": alpha,
                "mse_simple": mse,
                "mse_augmented": mse,
                "mean_mse": mse,
            }
        )
    pd.DataFrame(alpha_rows).to_parquet(artifact_paths["alpha_tuning"], index=False)
    ptb_predictions.to_parquet(artifact_paths["ptbxl_predictions"], index=False)

    def seed_evidence(
        cohort: str,
        predictions: pd.DataFrame,
        seed_path: Path,
        sufficient_path: Path,
        paired_path: Path,
    ):
        sources = {}
        for method in augmented_by_method:
            rows = []
            method_points = predictions[predictions["method"] == method]
            for model_seed in _expected_common_seeds(method):
                for point in method_points.itertuples(index=False):
                    rows.append(
                        {
                            "schema_version": "reconstruction-benchmark-v3",
                            "cohort": cohort,
                            "partition": "test",
                            "patient_id": point.patient_id,
                            "method": method,
                            "model_seed": model_seed,
                            "segment": point.segment,
                            "configuration": point.configuration,
                            "target": point.target,
                            "n_observed": 1,
                            "n_records": 1,
                            "n_samples": 10,
                            "target_rms": 1.0,
                            "max_target_observed_correlation": 0.5,
                            "outcome_log_rmse": point.outcome_log_rmse,
                        }
                    )
            source = meta / f"{cohort}-{method}-raw.parquet"
            pd.DataFrame(rows, columns=META_METRIC_COLUMNS).to_parquet(
                source, index=False, row_group_size=3
            )
            sources[method] = source
        sufficient_frame = _write_seed_evidence_and_sufficient(
            sources,
            predictions,
            cohort=cohort,
            seed_path=seed_path,
            sufficient_path=sufficient_path,
            paired_sufficient_path=paired_path,
        )[0]
        return sufficient_frame, pd.read_parquet(paired_path)

    ptb_sufficient, ptb_paired_sufficient = seed_evidence(
        "PTB-XL",
        ptb_predictions,
        artifact_paths["ptbxl_seed_predictions"],
        artifact_paths["ptbxl_sufficient_stats"],
        artifact_paths["ptbxl_paired_seed_sufficient"],
    )

    external_artifacts = {}
    for cohort, predictions in external_predictions.items():
        point = meta / f"{cohort}_predictions.parquet"
        seeds = meta / f"{cohort}_seed_predictions.parquet"
        sufficient = meta / f"{cohort}_sufficient_stats.parquet"
        paired = meta / f"{cohort}_paired_seed_sufficient.parquet"
        predictions.to_parquet(point, index=False)
        cohort_sufficient, cohort_paired = seed_evidence(
            cohort, predictions, seeds, sufficient, paired
        )
        external_artifacts[cohort] = {
            "point_predictions": {
                "path": point.name,
                "sha256": lineage.artifact_sha256(point),
            },
            "seed_predictions": {
                "path": seeds.name,
                "sha256": lineage.artifact_sha256(seeds),
            },
            "sufficient_stats": {
                "path": sufficient.name,
                "sha256": lineage.artifact_sha256(sufficient),
            },
            "paired_seed_sufficient": {
                "path": paired.name,
                "sha256": lineage.artifact_sha256(paired),
            },
        }
        external_predictions[cohort].attrs["sufficient"] = cohort_sufficient
        external_predictions[cohort].attrs["paired_sufficient"] = cohort_paired
    def effect(point: float, interval: list[float], seed: int) -> dict:
        return {"point": point, "ci95": interval, "replicates": 2_000, "seed": seed}

    ptb_effect, ptb_draws = _bootstrap_effect_and_draws_from_sufficient(
        ptb_sufficient,
        paired_sufficient=ptb_paired_sufficient,
        cohort="PTB-XL",
        replicates=2_000,
        seed=META_BOOTSTRAP_SEED + META_BOOTSTRAP_SEED_OFFSETS["PTB-XL"],
    )
    external_effects = {}
    draw_frames = [ptb_draws]
    for cohort, predictions in external_predictions.items():
        external_effects[cohort], draws = _bootstrap_effect_and_draws_from_sufficient(
            predictions.attrs["sufficient"],
            paired_sufficient=predictions.attrs["paired_sufficient"],
            cohort=cohort,
            replicates=2_000,
            seed=META_BOOTSTRAP_SEED + META_BOOTSTRAP_SEED_OFFSETS[cohort],
        )
        draw_frames.append(draws)
    pd.concat(draw_frames, ignore_index=True).to_parquet(
        artifact_paths["bootstrap_draws"], index=False
    )
    method_deltas = _method_deltas_from_sufficient(ptb_sufficient)
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
        "external_stage15_gate": {
            "eligible_cohorts": list(automatic.gate_eligible_external_cohorts),
            "qualifying_cohorts": list(automatic.qualifying_external_cohorts),
            "reported_but_gate_ineligible": {
                "cpsc2018": (
                    "record-name pseudopatients; no public cross-record patient key"
                )
            },
            "rule": (
                "only external cohorts with a documented patient-level identity key may "
                "satisfy the Stage-15 external lower-confidence-bound criterion"
            ),
        },
        "ptbxl": effect(ptb_effect.point, list(ptb_effect.ci95), ptb_effect.seed),
        "external": {
            cohort: effect(value.point, list(value.ci95), value.seed)
            for cohort, value in external_effects.items()
        },
        "method_delta_r2": method_deltas,
        "common_panel_methods": ["lowrank", "ridge", "masked-unet", "imputeecg"],
        "meta_alpha": 0.1,
        "meta_alpha_grid": list(META_RIDGE_ALPHA_GRID),
        "bootstrap_replicates": 2_000,
        "bootstrap_draw_schema_version": META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
        "seed_prediction_schema_version": META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_stat_schema_version": META_SUFFICIENT_SCHEMA_VERSION,
        "paired_seed_sufficient_schema_version": (
            META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
        ),
        "exact_model_seed_contract": {
            method: list(_expected_common_seeds(method))
            for method in augmented_by_method
        },
        "seed": META_BOOTSTRAP_SEED,
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
    assert set(gate["rule"]) == {
        "ptbxl_ci_lower_gt_zero",
        "chapman_ci_lower_gt_zero",
        "positive_common_panel_methods_at_least",
    }
    assert gate["policy"] == {"post_test_retuning_forbidden": True}
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

    changed_point = ptb_predictions.copy()
    changed_point.loc[0, "prediction_augmented"] += 1.0
    changed_point.to_parquet(artifact_paths["ptbxl_predictions"], index=False)
    summary["artifacts"]["ptbxl_predictions"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["ptbxl_predictions"]
    )
    _write(meta / "summary.v3.json", summary)
    with pytest.raises(ValueError, match="change authenticated predictions"):
        stage15(
            SimpleNamespace(
                meta_analysis=meta,
                output_dir=tmp_path / "tampered-point-gate",
            )
        )
    ptb_predictions.to_parquet(artifact_paths["ptbxl_predictions"], index=False)
    summary["artifacts"]["ptbxl_predictions"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["ptbxl_predictions"]
    )

    changed_alpha = pd.DataFrame(alpha_rows)
    changed_alpha.loc[
        changed_alpha["alpha"] == 10.0,
        ["mse_simple", "mse_augmented", "mean_mse"],
    ] = 0.0
    changed_alpha.to_parquet(artifact_paths["alpha_tuning"], index=False)
    summary["artifacts"]["alpha_tuning"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["alpha_tuning"]
    )
    _write(meta / "summary.v3.json", summary)
    with pytest.raises(ValueError, match="frozen fold-8 LOCO selection"):
        stage15(
            SimpleNamespace(
                meta_analysis=meta,
                output_dir=tmp_path / "post-test-retuned-alpha-gate",
            )
        )
    pd.DataFrame(alpha_rows).to_parquet(artifact_paths["alpha_tuning"], index=False)
    summary["artifacts"]["alpha_tuning"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["alpha_tuning"]
    )

    bootstrap_draws = pd.read_parquet(artifact_paths["bootstrap_draws"])
    bootstrap_draws.loc[0, "delta_r2"] = 999.0
    bootstrap_draws.to_parquet(artifact_paths["bootstrap_draws"], index=False)
    summary["artifacts"]["bootstrap_draws"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["bootstrap_draws"]
    )
    _write(meta / "summary.v3.json", summary)
    with pytest.raises(ValueError, match="delta_r2 disagrees"):
        stage15(
            SimpleNamespace(
                meta_analysis=meta,
                output_dir=tmp_path / "tampered-draw-gate",
            )
        )

    pd.concat(draw_frames, ignore_index=True).to_parquet(
        artifact_paths["bootstrap_draws"], index=False
    )
    summary["artifacts"]["bootstrap_draws"]["sha256"] = lineage.artifact_sha256(
        artifact_paths["bootstrap_draws"]
    )

    summary["ptbxl"]["point"] += 0.1
    _write(meta / "summary.v3.json", summary)
    with pytest.raises(
        ValueError,
        match="authenticated evidence|outside its confidence interval",
    ):
        stage15(
            SimpleNamespace(
                meta_analysis=meta,
                output_dir=tmp_path / "tampered-gate",
            )
        )
