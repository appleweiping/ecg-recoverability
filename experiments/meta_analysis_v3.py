"""Locked fold-8/9/10 meta-analysis and fail-closed ARC Stage-15 gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.arc_control import validate_arc_control_report
from ecgcert.benchmarking import (
    EXPECTED_METHODS,
    PRIMARY_METHODS,
    RELEASE_NEURAL_SEEDS,
    load_benchmark_bundles,
    path_sha256,
)
from ecgcert.evaluation import (
    META_RIDGE_ALPHA_GRID,
    BootstrapEffect,
    aggregate_model_seed_outcomes,
    attach_seed_bootstrap_predictions,
    cluster_bootstrap_delta_r2,
    loco_meta_predictions,
    method_specific_delta_r2,
    prediction_delta_r2,
    stage15_decision,
    tune_meta_ridge_alpha,
)
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.reconstruction import SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION
from ecgcert.data.common import CANONICAL_LEADS


SCHEMA_VERSION = "meta-analysis-v3"
COMMON_PANEL_METHODS = ("lowrank", "ridge", "masked-unet", "imputeecg")
NEURAL_METHODS = ("masked-unet", "imputeecg")
RANK_MAP_SCHEMA_VERSION = "robust-recoverability-map-v3"
EXTERNAL_VALIDATION_SCHEMA_VERSION = "external-validation-v3"
EFFECT_COMPARISON_ATOL = 1e-12


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _authenticated_artifact(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_name: str,
) -> Path:
    relative = Path(str(descriptor.get("path", "")))
    if (
        relative.as_posix() != expected_name
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"unsafe or unexpected artifact path {relative!s} in {root}")
    artifact = (root / relative).resolve()
    if root.resolve() not in artifact.parents or not artifact.is_file():
        raise FileNotFoundError(artifact)
    if lineage.artifact_sha256(artifact) != descriptor.get("sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {artifact}")
    return artifact


def _load_metrics(bundle: Path, *, authenticate: bool = False) -> pd.DataFrame:
    path = bundle / "patient_metrics.parquet"
    if authenticate:
        summary = _load_json(bundle / "summary.v3.json")
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(
            artifacts.get("patient_metrics"), Mapping
        ):
            raise ValueError(f"benchmark summary lacks patient-metric descriptor: {bundle}")
        path = _authenticated_artifact(
            bundle.resolve(),
            artifacts["patient_metrics"],
            expected_name="patient_metrics.parquet",
        )
    if not path.is_file():
        raise FileNotFoundError(f"benchmark bundle lacks {path.name}: {bundle}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"empty patient metrics: {path}")
    return frame


def _rank_map_release_contract(rank_maps: Path) -> tuple[dict[str, Any], str]:
    root = rank_maps.resolve()
    summary = _load_json(root / "summary.v3.json")
    required = {
        "schema_version": RANK_MAP_SCHEMA_VERSION,
        "status": "complete",
        "analysis_mode": "primary",
        "population": "all",
        "basis_variant": "independent8_lifted",
        "rate_hz": PRIMARY_RATE_HZ,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "patient",
        "n_configurations": 255,
        "deep_panel_sha256": configuration_panel_sha256(),
    }
    mismatches = {
        key: (summary.get(key), expected)
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if tuple(summary.get("segments", ())) != PRIMARY_SEGMENTS:
        mismatches["segments"] = (summary.get("segments"), list(PRIMARY_SEGMENTS))
    if tuple(summary.get("ranks", ())) != RANK_GRID:
        mismatches["ranks"] = (summary.get("ranks"), list(RANK_GRID))
    if mismatches:
        raise ValueError(f"rank-map release contract mismatch: {mismatches}")
    manifest_sha256 = str(summary.get("data_manifest_sha256", ""))
    if len(manifest_sha256) != 64:
        raise ValueError("rank-map summary lacks its authenticated PTB-XL manifest hash")
    artifacts = summary.get("artifacts")
    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifacts, Mapping) or not isinstance(artifact_sha256, Mapping):
        raise ValueError("rank-map summary lacks authenticated artifacts")
    for key, filename in (
        ("rank_path", "rank_path.parquet"),
        ("map_cells", "map_cells.parquet"),
        ("regularization_tuning", "regularization_tuning.parquet"),
        ("patient_audit", "patient_audit.json"),
    ):
        relative = artifacts.get(key)
        if relative != filename or artifact_sha256.get(key) is None:
            raise ValueError(f"rank-map summary lacks authenticated {key}")
        _authenticated_artifact(
            root,
            {"path": relative, "sha256": artifact_sha256[key]},
            expected_name=filename,
        )
    return summary, path_sha256(root)


def _validate_primary_rank_map_cells(
    frame: pd.DataFrame, summary: Mapping[str, Any]
) -> None:
    """Require the exact primary segment/configuration/target map without duplicates."""

    keys = ["segment", "configuration", "target"]
    required = {"schema_version", *keys}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"rank-map cells lack columns: {sorted(missing)}")
    if set(frame["schema_version"].astype(str)) != {RANK_MAP_SCHEMA_VERSION}:
        raise ValueError("rank-map cells use an unsupported schema")
    if frame.duplicated(keys).any():
        raise ValueError("rank-map cells contain duplicate segment/configuration/target keys")
    expected_configurations = {
        "+".join(configuration) for configuration in all_independent_configurations()
    }
    if set(frame["configuration"].astype(str)) != expected_configurations:
        raise ValueError("rank-map cells do not contain the exact 255-configuration universe")
    if set(frame["segment"].astype(str)) != set(PRIMARY_SEGMENTS):
        raise ValueError("rank-map cells do not contain exactly QRS/ST/T")
    if set(frame["target"].astype(str)) != set(CANONICAL_LEADS):
        raise ValueError("rank-map cells do not contain exactly the canonical 12 targets")
    expected_rows = len(PRIMARY_SEGMENTS) * len(expected_configurations) * len(CANONICAL_LEADS)
    if len(frame) != expected_rows or int(summary.get("n_map_cells", -1)) != expected_rows:
        raise ValueError(
            f"rank-map cell count must be {expected_rows}, got frame={len(frame)} "
            f"summary={summary.get('n_map_cells')}"
        )


def _validate_common_panel_metrics(
    frame: pd.DataFrame,
    *,
    cohort: str,
    partitions: set[str],
) -> pd.DataFrame:
    """Require equal patient/configuration cells for the four primary methods."""

    required_columns = {
        "schema_version",
        "cohort",
        "partition",
        "patient_id",
        "method",
        "model_seed",
        "segment",
        "configuration",
        "target",
        "outcome_log_rmse",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark metrics lack columns: {sorted(missing)}")
    if set(frame["schema_version"].astype(str)) != {BENCHMARK_SCHEMA_VERSION}:
        raise ValueError("benchmark metrics use an unsupported schema")
    if set(frame["cohort"].astype(str)) != {cohort}:
        raise ValueError(f"benchmark metrics do not contain exactly cohort {cohort}")
    if set(frame["partition"].astype(str)) != partitions:
        raise ValueError("benchmark metrics contain an incomplete or unexpected partition set")
    methods = set(frame["method"].astype(str))
    if methods != set(EXPECTED_METHODS):
        raise ValueError(f"benchmark metrics require exactly {EXPECTED_METHODS}, got {methods}")

    primary = frame[frame["method"].isin(COMMON_PANEL_METHODS)].copy()
    expected_configurations = {
        "+".join(configuration) for configuration in deep_configuration_panel()
    }
    key_columns = [
        "cohort",
        "partition",
        "patient_id",
        "segment",
        "configuration",
        "target",
    ]
    reference: pd.MultiIndex | None = None
    for method in COMMON_PANEL_METHODS:
        method_rows = primary[primary["method"] == method]
        seeds = {int(seed) for seed in method_rows["model_seed"].unique()}
        expected_seeds = (
            set(RELEASE_NEURAL_SEEDS) if method in NEURAL_METHODS else {0}
        )
        if seeds != expected_seeds:
            raise ValueError(f"{method} seeds {sorted(seeds)} != {sorted(expected_seeds)}")
        for seed in sorted(seeds):
            seed_rows = method_rows[method_rows["model_seed"].astype(int) == seed]
            for partition in sorted(partitions):
                partition_rows = seed_rows[
                    seed_rows["partition"].astype(str) == partition
                ]
                if set(partition_rows["configuration"].astype(str)) != expected_configurations:
                    raise ValueError(
                        f"{method} seed {seed} partition {partition} does not cover "
                        "the frozen 64-configuration panel"
                    )
                if set(partition_rows["segment"].astype(str)) != set(PRIMARY_SEGMENTS):
                    raise ValueError(
                        f"{method} seed {seed} partition {partition} does not contain "
                        "exactly QRS/ST/T"
                    )
            if seed_rows.duplicated(key_columns).any():
                raise ValueError(f"{method} seed {seed} has duplicate scientific cells")
            cells = pd.MultiIndex.from_frame(seed_rows[key_columns]).sort_values()
            if reference is None:
                reference = cells
            elif not cells.equals(reference):
                raise ValueError(
                    f"{method} seed {seed} does not share the complete patient/configuration panel"
                )
    return primary


def _validate_external_release_bundle(
    bundle: Path,
    *,
    source_manifest_sha256: str,
    rank_maps_sha256: str,
    benchmark_bundles: Mapping[str, Any],
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    root = bundle.resolve()
    summary_path = root / "summary.v3.json"
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("mode") != "zero-transfer"
        or summary.get("partition") != "test"
        or summary.get("external_training_or_adaptation")
        != "forbidden_and_not_performed"
        or summary.get("source_manifest_sha256") != source_manifest_sha256
        or summary.get("rank_maps_sha256") != rank_maps_sha256
        or summary.get("configuration_panel_sha256") != configuration_panel_sha256()
        or tuple(summary.get("common_panel_methods", ())) != COMMON_PANEL_METHODS
    ):
        raise ValueError(f"external bundle violates the zero-transfer release contract: {root}")
    cohort = str(summary.get("cohort", ""))
    if cohort not in {"chapman", "cpsc2018"}:
        raise ValueError(f"unexpected external cohort {cohort!r}")
    if int(summary.get("n_test_records_included", 0)) <= 0:
        raise ValueError(f"external bundle {cohort} contains no included test records")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"external summary lacks authenticated artifacts: {root}")
    metrics_descriptor = artifacts.get("patient_metrics")
    audit_descriptor = artifacts.get("evaluation_audit")
    if not isinstance(metrics_descriptor, Mapping) or not isinstance(
        audit_descriptor, Mapping
    ):
        raise ValueError(f"external summary lacks metrics/audit descriptors: {root}")
    metrics_path = _authenticated_artifact(
        root, metrics_descriptor, expected_name="patient_metrics.parquet"
    )
    audit_path = _authenticated_artifact(
        root, audit_descriptor, expected_name="evaluation_audit.json"
    )
    audit = _load_json(audit_path)
    if (
        audit.get("schema_version") != EXTERNAL_VALIDATION_SCHEMA_VERSION
        or audit.get("mode") != "zero-transfer"
        or audit.get("cohort") != cohort
        or audit.get("partition") != "test"
        or audit.get("no_external_fit") is not True
        or audit.get("source_manifest_sha256") != source_manifest_sha256
        or audit.get("rank_maps_sha256") != rank_maps_sha256
        or audit.get("target_manifest_sha256") != summary.get("target_manifest_sha256")
        or audit.get("target_split_sha256") != summary.get("target_split_sha256")
    ):
        raise ValueError(f"external audit does not prove zero-transfer evaluation: {root}")
    bundle_inventory = summary.get("benchmark_bundles")
    if not isinstance(bundle_inventory, Mapping) or set(bundle_inventory) != set(
        EXPECTED_METHODS
    ):
        raise ValueError(f"external bundle {cohort} has an incomplete benchmark inventory")
    for method, validated in benchmark_bundles.items():
        entry = bundle_inventory.get(method)
        if not isinstance(entry, Mapping):
            raise ValueError(f"external bundle {cohort} lacks {method} lineage")
        if (
            entry.get("bundle_sha256")
            != lineage.artifact_sha256(validated.root / "bundle.v3.json")
            or tuple(entry.get("seeds", ())) != validated.seeds
            or int(entry.get("n_configurations", 0)) != len(validated.configurations)
            or entry.get("training_predictors_sha256")
            != validated.training_predictors_sha256
        ):
            raise ValueError(f"external bundle {cohort} disagrees with {method} checkpoint bank")
    metrics = pd.read_parquet(metrics_path)
    if len(metrics) != int(summary.get("n_patient_metric_rows", -1)):
        raise ValueError(f"external metric row count disagrees with summary: {cohort}")
    report = {
        "summary_sha256": lineage.artifact_sha256(summary_path),
        "audit_sha256": lineage.artifact_sha256(audit_path),
        "metrics_sha256": lineage.artifact_sha256(metrics_path),
        "target_manifest_sha256": summary["target_manifest_sha256"],
        "target_split_sha256": summary["target_split_sha256"],
        "no_external_fit": True,
    }
    return metrics, cohort, report


def _attach_robust_map(frame: pd.DataFrame, map_cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["segment", "configuration", "target"]
    map_columns = keys + [
        "ambiguity_robust_mv",
        "configuration_rank_max",
        "log10_condition_max",
        "target_rms",
        "max_target_observed_correlation",
    ]
    missing = set(map_columns) - set(map_cells.columns)
    if missing:
        raise ValueError(f"rank map is missing columns: {sorted(missing)}")
    if map_cells.duplicated(keys).any():
        raise ValueError("rank map contains duplicate segment/configuration/target cells")
    robust = map_cells[map_columns]
    frame = frame.drop(
        columns=[
            "ambiguity_robust_mv",
            "target_rms",
            "max_target_observed_correlation",
            "configuration_rank",
            "log10_condition",
        ],
        errors="ignore",
    )
    merged = frame.merge(robust, on=keys, how="left", validate="many_to_one")
    if merged["ambiguity_robust_mv"].isna().any():
        raise ValueError("patient metrics contain cells absent from the frozen rank map")
    merged["configuration_rank"] = merged["configuration_rank_max"]
    merged["log10_condition"] = merged["log10_condition_max"]
    return merged.drop(columns=["configuration_rank_max", "log10_condition_max"])


def _combine_bundles(paths: Iterable[Path], *, authenticate: bool = False) -> pd.DataFrame:
    frames = [_load_metrics(path, authenticate=authenticate) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    identity = [
        column for column in (
            "cohort", "partition", "patient_id", "method", "model_seed",
            "segment", "configuration", "target",
        ) if column in combined
    ]
    if combined.duplicated(identity).any():
        raise ValueError("duplicate patient-level benchmark cells across bundles")
    return combined


def _effect_dict(effect) -> dict[str, Any]:
    value = asdict(effect)
    value["ci95"] = list(effect.ci95)
    return value


def _ecgrecover_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame[frame["method"].astype(str) == "ecgrecover"]
    if rows.empty:
        return []
    return (
        rows.groupby(["cohort", "partition", "segment"], sort=True)
        .agg(
            mean_log_rmse=("outcome_log_rmse", "mean"),
            patient_metric_rows=("outcome_log_rmse", "size"),
            patients=("patient_id", "nunique"),
        )
        .reset_index()
        .to_dict(orient="records")
    )


def analyze(arguments: argparse.Namespace) -> None:
    rank_root = arguments.rank_maps.resolve()
    rank_map_path = rank_root / "map_cells.parquet"
    if not rank_map_path.is_file():
        raise FileNotFoundError(rank_map_path)
    release_lineage: dict[str, Any] = {}
    if arguments.release:
        rank_summary, rank_maps_sha256 = _rank_map_release_contract(rank_root)
        source_manifest_sha256 = str(rank_summary["data_manifest_sha256"])
        validated_bundles = load_benchmark_bundles(
            arguments.benchmark,
            source_manifest_sha256=source_manifest_sha256,
            rank_maps_sha256=rank_maps_sha256,
            release=True,
        )
        if tuple(PRIMARY_METHODS) != COMMON_PANEL_METHODS:
            raise RuntimeError("primary method definitions disagree across evaluation modules")
        release_lineage = {
            "rank_map_summary_sha256": lineage.artifact_sha256(
                rank_root / "summary.v3.json"
            ),
            "rank_maps_sha256": rank_maps_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "benchmarks": {
                method: {
                    "bundle_sha256": lineage.artifact_sha256(
                        bundle.root / "bundle.v3.json"
                    ),
                    "summary_sha256": lineage.artifact_sha256(
                        bundle.root / "summary.v3.json"
                    ),
                    "metrics_sha256": lineage.artifact_sha256(
                        bundle.root / "patient_metrics.parquet"
                    ),
                    "seeds": list(bundle.seeds),
                    "n_configurations": len(bundle.configurations),
                }
                for method, bundle in validated_bundles.items()
            },
        }
    else:
        rank_maps_sha256 = path_sha256(rank_root)
        source_manifest_sha256 = "development-unverified"
        validated_bundles = {}

    rank_map = pd.read_parquet(rank_map_path)
    if arguments.release:
        _validate_primary_rank_map_cells(rank_map, rank_summary)
    ptbxl_raw = _combine_bundles(arguments.benchmark, authenticate=arguments.release)
    partitions = set(ptbxl_raw["partition"].astype(str))
    required_partitions = {"tune", "calibration", "test"}
    if partitions != required_partitions:
        raise ValueError(f"PTB-XL metrics lack partitions {sorted(required_partitions - partitions)}")
    if arguments.release:
        ptbxl_primary = _validate_common_panel_metrics(
            ptbxl_raw, cohort="PTB-XL", partitions=required_partitions
        )
    else:
        ptbxl_primary = ptbxl_raw[
            ptbxl_raw["method"].astype(str).isin(COMMON_PANEL_METHODS)
        ].copy()
    ptbxl_seed_rows = _attach_robust_map(ptbxl_primary, rank_map)
    ptbxl = aggregate_model_seed_outcomes(ptbxl_seed_rows)

    selection = tune_meta_ridge_alpha(
        ptbxl[ptbxl["partition"] == "tune"].copy(), grid=META_RIDGE_ALPHA_GRID
    )
    calibration = ptbxl[ptbxl["partition"] == "calibration"].copy()
    test = ptbxl[ptbxl["partition"] == "test"].copy()
    ptb_comparison = loco_meta_predictions(calibration, test, alpha=selection.alpha)
    ptb_seed_predictions = attach_seed_bootstrap_predictions(
        ptbxl_seed_rows[ptbxl_seed_rows["partition"] == "test"].copy(),
        ptb_comparison.predictions,
    )
    ptb_effect = cluster_bootstrap_delta_r2(
        ptb_comparison.predictions,
        replicates=arguments.bootstrap_replicates,
        seed=arguments.seed,
        bootstrap_predictions=ptb_seed_predictions,
    )

    external_effects = {}
    external_predictions = {}
    external_seed_predictions = {}
    external_hashes = {}
    external_release_reports = {}
    ecgrecover_supplement = _ecgrecover_summary(ptbxl_raw)
    for bundle in arguments.external:
        if arguments.release:
            external_raw, cohort, release_report = _validate_external_release_bundle(
                bundle,
                source_manifest_sha256=source_manifest_sha256,
                rank_maps_sha256=rank_maps_sha256,
                benchmark_bundles=validated_bundles,
            )
            external_release_reports[cohort] = release_report
            external_primary = _validate_common_panel_metrics(
                external_raw, cohort=cohort, partitions={"test"}
            )
        else:
            external_raw = _load_metrics(bundle)
            cohorts = external_raw["cohort"].astype(str).unique()
            if len(cohorts) != 1:
                raise ValueError(f"external bundle must contain exactly one cohort: {bundle}")
            cohort = str(cohorts[0])
            external_primary = external_raw[
                external_raw["method"].astype(str).isin(COMMON_PANEL_METHODS)
            ].copy()
        if cohort in external_effects:
            raise ValueError(f"duplicate external cohort {cohort}")
        if set(external_primary["partition"].astype(str)) != {"test"}:
            raise ValueError(f"external zero-transfer bundle {cohort} must contain only test rows")
        external_seed_rows = _attach_robust_map(external_primary, rank_map)
        external = aggregate_model_seed_outcomes(external_seed_rows)
        comparison = loco_meta_predictions(calibration, external, alpha=selection.alpha)
        seed_predictions = attach_seed_bootstrap_predictions(
            external_seed_rows, comparison.predictions
        )
        effect = cluster_bootstrap_delta_r2(
            comparison.predictions,
            replicates=arguments.bootstrap_replicates,
            seed=arguments.seed + 100 + len(external_effects),
            bootstrap_predictions=seed_predictions,
        )
        external_effects[cohort] = effect
        external_predictions[cohort] = comparison.predictions
        external_seed_predictions[cohort] = seed_predictions
        external_hashes[cohort] = lineage.artifact_sha256(bundle / "patient_metrics.parquet")
        ecgrecover_supplement.extend(_ecgrecover_summary(external_raw))

    if not arguments.allow_partial_external and set(external_effects) != {"chapman", "cpsc2018"}:
        raise ValueError("release analysis requires both Chapman and CPSC2018")
    all_method_deltas = method_specific_delta_r2(ptb_comparison.predictions)
    method_deltas = {
        method: all_method_deltas[method]
        for method in COMMON_PANEL_METHODS if method in all_method_deltas
    }
    if not arguments.allow_partial_external and set(method_deltas) != set(COMMON_PANEL_METHODS):
        raise ValueError("common-panel evidence must contain all four primary reconstructors")
    decision = stage15_decision(
        ptbxl=ptb_effect, external=external_effects, method_deltas=method_deltas
    )

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    alpha_path = output / "alpha_tuning.parquet"
    ptb_path = output / "ptbxl_predictions.parquet"
    ptb_seed_path = output / "ptbxl_seed_predictions.parquet"
    selection.table.to_parquet(alpha_path, index=False, compression="zstd")
    ptb_comparison.predictions.to_parquet(ptb_path, index=False, compression="zstd")
    ptb_seed_predictions.to_parquet(ptb_seed_path, index=False, compression="zstd")
    external_artifacts = {}
    for cohort, predictions in external_predictions.items():
        point_path = output / f"{cohort}_predictions.parquet"
        seed_path = output / f"{cohort}_seed_predictions.parquet"
        predictions.to_parquet(point_path, index=False, compression="zstd")
        external_seed_predictions[cohort].to_parquet(
            seed_path, index=False, compression="zstd"
        )
        external_artifacts[cohort] = {
            "point_predictions": {
                "path": point_path.name,
                "sha256": lineage.artifact_sha256(point_path),
            },
            "seed_predictions": {
                "path": seed_path.name,
                "sha256": lineage.artifact_sha256(seed_path),
            },
        }
    effects = [
        {"cohort": "PTB-XL", **_effect_dict(ptb_effect)},
        *(
            {"cohort": cohort, **_effect_dict(effect)}
            for cohort, effect in sorted(external_effects.items())
        ),
    ]
    effects_path = output / "effects.parquet"
    pd.DataFrame(effects).to_parquet(effects_path, index=False, compression="zstd")
    if arguments.release:
        release_lineage["external"] = external_release_reports
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "outcome": "patient-level log(RMSE) on missing target samples",
        "meta_alpha": selection.alpha,
        "meta_alpha_grid": list(META_RIDGE_ALPHA_GRID),
        "ptbxl": _effect_dict(ptb_effect),
        "external": {
            cohort: _effect_dict(effect) for cohort, effect in sorted(external_effects.items())
        },
        "method_delta_r2": method_deltas,
        "common_panel_methods": list(COMMON_PANEL_METHODS),
        "meta_estimand": (
            "one equally weighted patient-method-segment-configuration-target cell; "
            "neural outcomes averaged over five seeds for the point estimate"
        ),
        "nested_seed_bootstrap": (
            "one model seed selected globally per neural method within each patient bootstrap"
        ),
        "excluded_from_common_panel_meta_model": ["ecgrecover"],
        "supplementary_ecgrecover": ecgrecover_supplement,
        "automatic_stage15_status": decision.status,
        "automatic_stage15_reasons": list(decision.reasons),
        "bootstrap_unit": "patient with nested neural-seed resampling",
        "bootstrap_replicates": arguments.bootstrap_replicates,
        "seed": arguments.seed,
        "release_contract_verified": bool(arguments.release),
        "release_lineage": release_lineage,
        "input_sha256": {
            "rank_map": lineage.artifact_sha256(rank_map_path),
            "benchmarks": {
                str(path): lineage.artifact_sha256(path / "patient_metrics.parquet")
                for path in arguments.benchmark
            },
            "external": external_hashes,
        },
        "artifacts": {
            "alpha_tuning": {
                "path": alpha_path.name,
                "sha256": lineage.artifact_sha256(alpha_path),
            },
            "ptbxl_predictions": {
                "path": ptb_path.name,
                "sha256": lineage.artifact_sha256(ptb_path),
            },
            "ptbxl_seed_predictions": {
                "path": ptb_seed_path.name,
                "sha256": lineage.artifact_sha256(ptb_seed_path),
            },
            "effects": {
                "path": effects_path.name,
                "sha256": lineage.artifact_sha256(effects_path),
            },
            "external": external_artifacts,
        },
    }
    _write_json(output / "summary.v3.json", summary)


def stage15(arguments: argparse.Namespace) -> None:
    summary_path = arguments.meta_analysis / "summary.v3.json"
    summary = _load_json(summary_path)
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "complete":
        raise ValueError("Stage 15 requires a complete v3 meta-analysis")
    if summary.get("release_contract_verified") is not True:
        raise ValueError("Stage 15 requires a release-validated meta-analysis")
    if summary.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError(f"Stage 15 requires exactly {BOOTSTRAP_REPLICATES} bootstraps")
    if tuple(summary.get("common_panel_methods", ())) != COMMON_PANEL_METHODS:
        raise ValueError("Stage 15 requires the frozen four-method common panel")
    if set(summary.get("external", {})) != {"chapman", "cpsc2018"}:
        raise ValueError("Stage 15 requires both external zero-transfer cohorts")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage 15 meta-analysis lacks authenticated artifacts")
    authenticated: dict[str, Path] = {}
    for key, filename in (
        ("alpha_tuning", "alpha_tuning.parquet"),
        ("ptbxl_predictions", "ptbxl_predictions.parquet"),
        ("ptbxl_seed_predictions", "ptbxl_seed_predictions.parquet"),
        ("effects", "effects.parquet"),
    ):
        descriptor = artifacts.get(key)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Stage 15 meta-analysis lacks {key} descriptor")
        authenticated[key] = _authenticated_artifact(
            arguments.meta_analysis.resolve(), descriptor, expected_name=filename
        )
    external_artifacts = artifacts.get("external")
    if not isinstance(external_artifacts, Mapping) or set(external_artifacts) != {
        "chapman",
        "cpsc2018",
    }:
        raise ValueError("Stage 15 meta-analysis lacks external prediction artifacts")
    authenticated_external: dict[str, dict[str, Path]] = {}
    for cohort, inventory in external_artifacts.items():
        if not isinstance(inventory, Mapping):
            raise ValueError(f"invalid external artifact inventory for {cohort}")
        authenticated_external[cohort] = {}
        for key, filename in (
            ("point_predictions", f"{cohort}_predictions.parquet"),
            ("seed_predictions", f"{cohort}_seed_predictions.parquet"),
        ):
            descriptor = inventory.get(key)
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"Stage 15 lacks {cohort} {key}")
            authenticated_external[cohort][key] = _authenticated_artifact(
                arguments.meta_analysis.resolve(), descriptor, expected_name=filename
            )

    def parse_effect(value: Mapping[str, Any], *, label: str) -> BootstrapEffect:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} effect must be an object")
        interval = value.get("ci95")
        if not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2:
            raise ValueError(f"{label} effect lacks a two-sided confidence interval")
        effect = BootstrapEffect(
            point=float(value["point"]),
            ci95=(float(interval[0]), float(interval[1])),
            replicates=int(value["replicates"]),
            seed=int(value["seed"]),
        )
        numbers = (effect.point, *effect.ci95)
        if effect.replicates != BOOTSTRAP_REPLICATES or not np.isfinite(numbers).all():
            raise ValueError(f"{label} effect is not a finite 2,000-bootstrap estimate")
        if effect.ci95[0] > effect.ci95[1]:
            raise ValueError(f"{label} confidence interval endpoints are reversed")
        return effect

    ptb_effect = parse_effect(summary["ptbxl"], label="PTB-XL")
    external_effects = {
        cohort: parse_effect(value, label=cohort)
        for cohort, value in summary["external"].items()
    }
    method_deltas = summary.get("method_delta_r2")
    if not isinstance(method_deltas, Mapping) or set(method_deltas) != set(
        COMMON_PANEL_METHODS
    ):
        raise ValueError("Stage 15 requires one point estimate for each common-panel method")
    method_deltas = {key: float(value) for key, value in method_deltas.items()}
    if not np.isfinite(list(method_deltas.values())).all():
        raise ValueError("Stage 15 method effects must be finite")

    def require_close(actual: float, expected: float, *, label: str) -> None:
        if not np.isclose(
            float(actual), float(expected), rtol=0.0, atol=EFFECT_COMPARISON_ATOL
        ):
            raise ValueError(
                f"{label} disagrees with authenticated evidence: {actual} != {expected}"
            )

    # The compact effects table is the authenticated source for intervals and
    # bootstrap metadata.  Prediction tables independently determine all point
    # estimates, so editing summary numbers cannot change the hard decision.
    effects_frame = pd.read_parquet(authenticated["effects"])
    required_effect_columns = {"cohort", "point", "ci95", "replicates", "seed"}
    missing_effect_columns = required_effect_columns - set(effects_frame.columns)
    if missing_effect_columns:
        raise ValueError(
            f"effects artifact lacks columns: {sorted(missing_effect_columns)}"
        )
    expected_effects = {"PTB-XL": ptb_effect, **external_effects}
    if (
        effects_frame["cohort"].astype(str).duplicated().any()
        or set(effects_frame["cohort"].astype(str)) != set(expected_effects)
    ):
        raise ValueError("effects artifact has incomplete or duplicate cohorts")
    for cohort, expected in expected_effects.items():
        row = effects_frame[
            effects_frame["cohort"].astype(str) == cohort
        ].iloc[0].to_dict()
        artifact_effect = parse_effect(row, label=f"{cohort} artifact")
        require_close(artifact_effect.point, expected.point, label=f"{cohort} point")
        require_close(artifact_effect.ci95[0], expected.ci95[0], label=f"{cohort} CI lower")
        require_close(artifact_effect.ci95[1], expected.ci95[1], label=f"{cohort} CI upper")
        if (
            artifact_effect.replicates != expected.replicates
            or artifact_effect.seed != expected.seed
        ):
            raise ValueError(f"{cohort} bootstrap metadata disagrees with effects artifact")

    ptb_predictions = pd.read_parquet(authenticated["ptbxl_predictions"])
    require_close(
        prediction_delta_r2(ptb_predictions),
        ptb_effect.point,
        label="PTB-XL point estimate",
    )
    artifact_method_deltas = method_specific_delta_r2(ptb_predictions)
    if set(artifact_method_deltas) != set(COMMON_PANEL_METHODS):
        raise ValueError("PTB-XL prediction artifact lacks the frozen four-method panel")
    for method in COMMON_PANEL_METHODS:
        require_close(
            artifact_method_deltas[method],
            method_deltas[method],
            label=f"{method} point estimate",
        )
    for cohort, expected in external_effects.items():
        point_predictions = pd.read_parquet(
            authenticated_external[cohort]["point_predictions"]
        )
        require_close(
            prediction_delta_r2(point_predictions),
            expected.point,
            label=f"{cohort} point estimate",
        )

    recomputed = stage15_decision(
        ptbxl=ptb_effect,
        external=external_effects,
        method_deltas=method_deltas,
    )
    automatic = summary.get("automatic_stage15_status")
    if (
        automatic != recomputed.status
        or list(recomputed.reasons) != summary.get("automatic_stage15_reasons")
    ):
        raise ValueError("stored Stage-15 decision does not match the frozen hard rule")
    # The automatic rule determines which human decision is admissible, but it
    # never substitutes for the required review.  In particular, a failed hard
    # rule is still held for a 24-hour, hash-bound author review before a
    # transparent negative-result PIVOT may enter the manuscript.
    eligible = automatic == "PROCEED"
    evidence = {
        "ptbxl": _effect_dict(ptb_effect),
        "external": {
            cohort: _effect_dict(effect)
            for cohort, effect in sorted(external_effects.items())
        },
        "method_delta_r2": method_deltas,
        "bootstrap_replicates": summary["bootstrap_replicates"],
        "release_lineage_sha256": lineage.canonical_sha256(
            summary.get("release_lineage")
        ),
    }
    arc_report = _load_json(arguments.arc_control)
    arc_control = validate_arc_control_report(arc_report, 15)
    evidence["official_arc_control"] = {
        "report_sha256": lineage.artifact_sha256(arguments.arc_control),
        "receipt_sha256": arc_control["receipt_sha256"],
        "run_id": arc_control["run_id"],
        "session_id": arc_control["session_id"],
        "decision": arc_control["decision"],
        "stage_output_sha256": arc_control["stage_output_sha256"],
    }
    gate = {
        "schema_version": "arc-stage15-v3",
        "stage": 15,
        "status": "PENDING_USER_REVIEW",
        "eligible_for_proceed": eligible,
        "human_review_required": True,
        "review_deadline_hours": 24,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "automatic_decision": automatic,
        "automatic_reasons": summary["automatic_stage15_reasons"],
        "meta_analysis_sha256": lineage.artifact_sha256(summary_path),
        "evidence_sha256": lineage.canonical_sha256(evidence),
        "evidence": evidence,
        "rule": {
            "ptbxl_ci_lower_gt_zero": True,
            "one_external_ci_lower_gt_zero": True,
            "positive_common_panel_methods_at_least": 3,
            "post_test_retuning_forbidden": True,
        },
    }
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "decision.v3.json", gate)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze", "stage15"), required=True)
    parser.add_argument("--rank-maps", type=Path)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--external", type=Path, action="append", default=[])
    parser.add_argument("--meta-analysis", type=Path)
    parser.add_argument("--arc-control", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--allow-partial-external", action="store_true")
    parser.add_argument("--release", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.mode == "analyze":
        if arguments.rank_maps is None or not arguments.benchmark or not arguments.external:
            raise SystemExit("analyze requires --rank-maps, --benchmark and --external")
        if arguments.release:
            if arguments.bootstrap_replicates != BOOTSTRAP_REPLICATES:
                raise SystemExit(
                    f"release analysis requires exactly {BOOTSTRAP_REPLICATES} bootstraps"
                )
            if arguments.allow_partial_external:
                raise SystemExit("release analysis forbids --allow-partial-external")
        elif arguments.bootstrap_replicates < 100:
            raise SystemExit("--bootstrap-replicates must be at least 100")
        analyze(arguments)
    else:
        if arguments.meta_analysis is None or arguments.arc_control is None:
            raise SystemExit("stage15 requires --meta-analysis and --arc-control")
        stage15(arguments)


if __name__ == "__main__":
    main()
