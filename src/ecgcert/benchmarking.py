"""Fail-closed helpers shared by external reconstruction validation.

This module deliberately contains no training entry point.  It validates fitted
PTB-XL benchmark bundles, reloads them through the frozen reconstruction API, and
computes ranking agreement for secondary external-cohort maps.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ecgcert import lineage
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.official import ECG_RECOVER, IMPUTE_ECG
from ecgcert.protocol import (
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.reconstruction import (
    BUNDLE_FILENAME,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    TRAINING_PREDICTORS_FILENAME,
    EvaluationRecord,
    ModelBundleError,
    evaluate_reconstructor,
    load_fitted_reconstructor,
    load_training_predictors,
    training_predictor_lookup,
)


PRIMARY_METHODS = ("lowrank", "ridge", "masked-unet", "imputeecg")
SUPPLEMENTARY_METHOD = "ecgrecover"
EXPECTED_METHODS = (*PRIMARY_METHODS, SUPPLEMENTARY_METHOD)
RELEASE_NEURAL_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class FittedBenchmarkBundle:
    root: Path
    method: str
    seeds: tuple[int, ...]
    configurations: tuple[tuple[str, ...], ...]
    training_predictors_sha256: str
    metadata: Mapping[str, Any]
    summary: Mapping[str, Any]


def path_sha256(path: str | Path) -> str:
    """Hash a file or a directory's sorted relative-path/hash inventory."""

    root = Path(path).resolve()
    if root.is_file():
        return lineage.artifact_sha256(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = [
        (item.relative_to(root).as_posix(), lineage.artifact_sha256(item))
        for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]
    if not entries:
        raise ValueError(f"artifact directory is empty: {root}")
    return lineage.canonical_sha256(entries)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelBundleError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelBundleError(f"{label} must be a JSON object")
    return value


def _checkpoint_path(root: Path, entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ModelBundleError(f"unsafe checkpoint path: {relative}")
    checkpoint = (root / relative).resolve()
    if root != checkpoint and root not in checkpoint.parents:
        raise ModelBundleError(f"checkpoint escapes bundle: {relative}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = str(entry.get("sha256", ""))
    if lineage.artifact_sha256(checkpoint) != expected:
        raise ModelBundleError(f"checkpoint SHA-256 mismatch: {relative}")
    return checkpoint


def _authenticated_artifact(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_name: str | None = None,
) -> Path:
    relative = Path(str(descriptor.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ModelBundleError(f"unsafe benchmark artifact path: {relative}")
    if expected_name is not None and relative.as_posix() != expected_name:
        raise ModelBundleError(
            f"benchmark artifact must be named {expected_name}, got {relative.as_posix()}"
        )
    artifact = (root / relative).resolve()
    if root != artifact and root not in artifact.parents:
        raise ModelBundleError(f"benchmark artifact escapes bundle: {relative}")
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if lineage.artifact_sha256(artifact) != descriptor.get("sha256"):
        raise ModelBundleError(f"benchmark artifact SHA-256 mismatch: {relative}")
    return artifact


def _official_bridge(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(token, str) and token for token in value
    ):
        raise ModelBundleError(f"{label} must be a non-empty argv list")
    joined = "\n".join(value)
    if "{input}" not in joined or "{output}" not in joined:
        raise ModelBundleError(f"{label} lacks input/output placeholders")
    return tuple(value)


def _canonical_entry_configuration(entry: Mapping[str, Any]) -> tuple[str, ...]:
    configuration = tuple(str(lead) for lead in entry.get("configuration", ()))
    if not configuration:
        raise ModelBundleError("linear checkpoint lacks configuration metadata")
    if len(configuration) != len(set(configuration)):
        raise ModelBundleError("linear checkpoint configuration contains duplicates")
    if set(configuration) - set(CANONICAL_LEADS):
        raise ModelBundleError("linear checkpoint configuration contains unknown leads")
    return configuration


def load_benchmark_bundles(
    bundle_paths: Sequence[str | Path],
    *,
    source_manifest_sha256: str,
    rank_maps_sha256: str,
    release: bool,
) -> dict[str, FittedBenchmarkBundle]:
    """Validate the complete five-bundle PTB-XL zero-transfer input contract."""

    bundles: dict[str, FittedBenchmarkBundle] = {}
    frozen_panel = deep_configuration_panel()
    frozen_panel_hash = configuration_panel_sha256(frozen_panel)
    for raw_path in bundle_paths:
        root = Path(raw_path).resolve()
        metadata = _read_json(root / BUNDLE_FILENAME, "benchmark bundle")
        summary = _read_json(root / SUMMARY_FILENAME, "benchmark summary")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ModelBundleError(f"unsupported benchmark schema in {root}")
        if summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "complete":
            raise ModelBundleError(f"benchmark summary is not complete: {root}")
        method = str(metadata.get("method", ""))
        if method not in EXPECTED_METHODS or summary.get("method") != method:
            raise ModelBundleError(f"unknown or inconsistent benchmark method {method!r}")
        if method in bundles:
            raise ModelBundleError(f"duplicate benchmark bundle for {method}")
        if metadata.get("load_helper") != "ecgcert.reconstruction.load_fitted_reconstructor":
            raise ModelBundleError(f"{method} does not declare the frozen load helper")
        if not str(metadata.get("adapter_class", "")):
            raise ModelBundleError(f"{method} lacks adapter_class")
        for field in (
            "adapter_class",
            "load_helper",
            "training_config",
            "tuning_config",
            "official",
        ):
            if summary.get(field) != metadata.get(field):
                raise ModelBundleError(f"{method} summary/bundle disagree on {field}")
        if summary.get("tuning_source") != metadata.get("tuning_source"):
            raise ModelBundleError(f"{method} summary/bundle disagree on tuning_source")

        manifest = summary.get("manifest", {})
        if not isinstance(manifest, Mapping):
            raise ModelBundleError(f"{method} summary has invalid manifest metadata")
        if manifest.get("sha256") != source_manifest_sha256:
            raise ModelBundleError(f"{method} was not fitted from the requested PTB manifest")
        if not str(manifest.get("split_sha256", "")):
            raise ModelBundleError(f"{method} summary lacks its PTB split hash")
        if summary.get("rank_maps_sha256") != rank_maps_sha256:
            raise ModelBundleError(f"{method} used a different frozen rank-map artifact")
        artifacts = summary.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise ModelBundleError(f"{method} summary lacks authenticated artifacts")
        for artifact_name, expected_name in (
            ("bundle", BUNDLE_FILENAME),
            ("patient_metrics", "patient_metrics.parquet"),
            ("evaluation_audit", "evaluation_audit.json"),
            ("training_predictors", TRAINING_PREDICTORS_FILENAME),
        ):
            descriptor = artifacts.get(artifact_name)
            if not isinstance(descriptor, Mapping):
                raise ModelBundleError(
                    f"{method} summary lacks authenticated {artifact_name} artifact"
                )
            _authenticated_artifact(root, descriptor, expected_name=expected_name)
        predictor_descriptor = metadata.get("training_predictors")
        if not isinstance(predictor_descriptor, Mapping):
            raise ModelBundleError(f"{method} bundle lacks folds1-7 training predictors")
        if predictor_descriptor.get("source_partition") != "PTB-XL/folds1-7/train":
            raise ModelBundleError(f"{method} predictors do not come from PTB folds1-7")
        predictor_path = _authenticated_artifact(
            root,
            predictor_descriptor,
            expected_name=TRAINING_PREDICTORS_FILENAME,
        )
        if (
            artifacts["training_predictors"].get("sha256")
            != predictor_descriptor.get("sha256")
        ):
            raise ModelBundleError(f"{method} predictor hashes disagree across metadata")

        training = metadata.get("training_config", {})
        if not isinstance(training, Mapping):
            raise ModelBundleError(f"{method} has invalid training_config metadata")
        simple_predictors = training.get("simple_predictors", {})
        if not isinstance(simple_predictors, Mapping):
            raise ModelBundleError(f"{method} has invalid simple predictor metadata")
        if (
            training.get("cohort") != "PTB-XL"
            or training.get("train_role") != "folds1-7"
            or training.get("evaluation_roles")
            != {
                "tune": "fold8/tune",
                "calibration": "fold9/calibration",
                "test": "fold10/test",
            }
            or training.get("rate_hz") != PRIMARY_RATE_HZ
            or tuple(training.get("segments", ())) != PRIMARY_SEGMENTS
            or training.get("delineator") != "dwt"
            or training.get("signal_unit") != "raw_mV"
            or training.get("mask") != "whole-lead; identical across methods"
            or simple_predictors.get("heldout_target_statistics_used") is not False
            or int(training.get("n_train_records", 0)) <= 0
            or len(str(training.get("train_signals_sha256", ""))) != 64
            or bool(training.get("subsampled", True))
            or (release and training.get("release") is not True)
        ):
            raise ModelBundleError(f"{method} is not a full frozen PTB-XL training bundle")

        entries = metadata.get("models", ())
        if not isinstance(entries, list) or not entries:
            raise ModelBundleError(f"{method} bundle contains no fitted checkpoints")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("seed"), int):
                raise ModelBundleError(f"{method} checkpoint metadata is incomplete")
            _checkpoint_path(root, entry)
        seeds = tuple(sorted({int(entry["seed"]) for entry in entries}))
        if training.get("model_seeds") != list(seeds):
            raise ModelBundleError(f"{method} training seed metadata is inconsistent")

        if method in PRIMARY_METHODS:
            if (
                training.get("n_configurations") != len(frozen_panel)
                or training.get("configuration_panel_sha256") != frozen_panel_hash
            ):
                raise ModelBundleError(f"{method} does not cover the frozen 64 configurations")
            configurations = frozen_panel
            if method in {"lowrank", "ridge"}:
                if seeds != (0,):
                    raise ModelBundleError(f"{method} must contain deterministic seed 0 only")
                if len(entries) != len(frozen_panel):
                    raise ModelBundleError(
                        f"{method} requires exactly one checkpoint per panel configuration"
                    )
                by_seed = {
                    seed: {
                        _canonical_entry_configuration(entry)
                        for entry in entries
                        if entry["seed"] == seed
                    }
                    for seed in seeds
                }
                if any(
                    configurations_set != set(frozen_panel)
                    for configurations_set in by_seed.values()
                ):
                    raise ModelBundleError(
                        f"{method} checkpoint bank is missing panel configurations"
                    )
            else:
                if any(sum(entry["seed"] == seed for entry in entries) != 1 for seed in seeds):
                    raise ModelBundleError(f"{method} requires exactly one checkpoint per seed")
                if release and seeds != RELEASE_NEURAL_SEEDS:
                    raise ModelBundleError(f"release {method} requires seeds 0,1,2,3,4")
            if method == "imputeecg":
                official = metadata.get("official", {})
                if not isinstance(official, Mapping):
                    raise ModelBundleError("ImputeECG official metadata must be an object")
                if (
                    not official.get("source_dir")
                    or official.get("repository") != IMPUTE_ECG.repository
                    or official.get("commit") != IMPUTE_ECG.commit
                    or len(str(official.get("integration_config_sha256", ""))) != 64
                ):
                    raise ModelBundleError("ImputeECG bundle lacks its pinned official adapter")
        else:
            official = metadata.get("official", {})
            if not isinstance(official, Mapping):
                raise ModelBundleError("ECGrecover official metadata must be an object")
            input_lead = str(official.get("input_lead", ""))
            if input_lead not in CANONICAL_LEADS:
                raise ModelBundleError("ECGrecover bundle lacks its official canonical input lead")
            if (
                not official.get("source_dir")
                or official.get("repository") != ECG_RECOVER.repository
                or official.get("commit") != ECG_RECOVER.commit
                or len(str(official.get("integration_config_sha256", ""))) != 64
            ):
                raise ModelBundleError("ECGrecover bundle lacks its pinned official adapter")
            _official_bridge(
                official.get("inference_bridge"),
                label="ECGrecover official inference bridge",
            )
            if any(sum(entry["seed"] == seed for entry in entries) != 1 for seed in seeds):
                raise ModelBundleError("ECGrecover requires exactly one checkpoint per seed")
            for entry in entries:
                if tuple(entry.get("configuration", ())) != (input_lead,):
                    raise ModelBundleError(
                        "ECGrecover checkpoint is not restricted to the official input lead"
                    )
                _official_bridge(
                    entry.get("inference_bridge"),
                    label=f"ECGrecover seed {entry['seed']} inference bridge",
                )
            if release and seeds != RELEASE_NEURAL_SEEDS:
                raise ModelBundleError("release ECGrecover requires seeds 0,1,2,3,4")
            if (
                training.get("n_configurations") != 1
                or training.get("configuration_panel_sha256")
                != configuration_panel_sha256(((input_lead,),))
            ):
                raise ModelBundleError("ECGrecover training scope is not its single-input task")
            configurations = ((input_lead,),)

        bundles[method] = FittedBenchmarkBundle(
            root=root,
            method=method,
            seeds=seeds,
            configurations=tuple(configurations),
            training_predictors_sha256=lineage.artifact_sha256(predictor_path),
            metadata=metadata,
            summary=summary,
        )
    missing = set(EXPECTED_METHODS) - set(bundles)
    extra = set(bundles) - set(EXPECTED_METHODS)
    if missing or extra:
        raise ModelBundleError(
            f"zero-transfer requires exactly {EXPECTED_METHODS}; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return bundles


def _predictor_content(
    predictors: Any,
) -> tuple[dict[tuple[str, str, str], tuple[float, float]], str]:
    lookup = training_predictor_lookup(predictors)
    expected_keys = {
        (segment, "+".join(configuration), target)
        for segment in PRIMARY_SEGMENTS
        for configuration in deep_configuration_panel()
        for target in CANONICAL_LEADS
    }
    if set(lookup) != expected_keys:
        raise ModelBundleError(
            "folds1-7 predictors do not cover the frozen 64-configuration panel"
        )
    payload = [
        {
            "segment": segment,
            "configuration": configuration,
            "target": target,
            "target_rms": values[0],
            "max_target_observed_correlation": values[1],
        }
        for (segment, configuration, target), values in sorted(lookup.items())
    ]
    return lookup, lineage.canonical_sha256(payload)


def evaluate_zero_transfer_bundles(
    bundles: Mapping[str, FittedBenchmarkBundle],
    records: Sequence[EvaluationRecord],
    *,
    cohort: str,
    device: str,
    model_loader: Callable[..., Any] = load_fitted_reconstructor,
    predictor_loader: Callable[[str | Path], Any] = load_training_predictors,
) -> pd.DataFrame:
    """Reload fitted PTB models and score external records without any fit call."""

    if tuple(bundles) != EXPECTED_METHODS:
        # Dict insertion order is not scientific, but requiring the exact method set is.
        if set(bundles) != set(EXPECTED_METHODS):
            raise ModelBundleError("incomplete zero-transfer method set")
    records = tuple(records)
    if not records:
        raise ValueError("zero-transfer evaluation requires at least one external record")
    for record in records:
        record.validate()
    predictors_by_method = {}
    predictor_content_sha256: str | None = None
    for method in EXPECTED_METHODS:
        bundle = bundles[method]
        training_predictors, content_sha256 = _predictor_content(
            predictor_loader(bundle.root)
        )
        if predictor_content_sha256 is None:
            predictor_content_sha256 = content_sha256
        elif content_sha256 != predictor_content_sha256:
            raise ModelBundleError(
                "zero-transfer bundles disagree on folds1-7 predictor content"
            )
        predictors_by_method[method] = training_predictors

    frames = []
    for method in EXPECTED_METHODS:
        bundle = bundles[method]
        training_predictors = predictors_by_method[method]
        for seed in bundle.seeds:
            reconstructor = model_loader(bundle.root, method, seed, device=device)
            if not callable(getattr(reconstructor, "reconstruct", None)):
                raise ModelBundleError(f"{method} loader did not return a reconstruction adapter")
            for configuration in bundle.configurations:
                frames.append(
                    evaluate_reconstructor(
                        reconstructor,
                        records,
                        configuration=configuration,
                        method=method,
                        model_seed=seed,
                        segments=PRIMARY_SEGMENTS,
                        training_predictors=training_predictors,
                        cohort=cohort,
                        partition="test",
                    )
                )
    output = pd.concat(frames, ignore_index=True)
    output.attrs["training_predictors_content_sha256"] = predictor_content_sha256
    return output


def compare_recoverability_rankings(
    external_cells: pd.DataFrame,
    ptb_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman agreement of missing-target recoverability rankings."""

    required = {
        "segment",
        "configuration",
        "target",
        "target_observed",
        "recoverability_lower",
    }
    for label, frame in (("external", external_cells), ("PTB", ptb_cells)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} map lacks columns: {sorted(missing)}")
    keys = ["segment", "configuration", "target"]
    external = external_cells.loc[~external_cells["target_observed"].astype(bool)].copy()
    ptb = ptb_cells.loc[~ptb_cells["target_observed"].astype(bool)].copy()
    if external.duplicated(keys).any() or ptb.duplicated(keys).any():
        raise ValueError("rank-map cells must be unique by segment/configuration/target")
    ptb = ptb[ptb["segment"].isin(external["segment"].unique())]
    merged = external[keys + ["recoverability_lower"]].merge(
        ptb[keys + ["recoverability_lower"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_external", "_ptb"),
    )
    if merged["recoverability_lower_ptb"].isna().any() or len(merged) != len(external):
        raise ValueError("external map contains cells absent from the PTB rank map")
    if not np.isfinite(
        merged[["recoverability_lower_external", "recoverability_lower_ptb"]].to_numpy()
    ).all():
        raise ValueError("recoverability ranking inputs must be finite")

    rows: list[dict[str, Any]] = []
    groups = [("__all__", "__all__", merged)]
    groups.extend(
        (str(segment), str(target), group)
        for (segment, target), group in merged.groupby(["segment", "target"], sort=True)
    )
    for segment, target, group in groups:
        external_score = group["recoverability_lower_external"].to_numpy(dtype=float)
        ptb_score = group["recoverability_lower_ptb"].to_numpy(dtype=float)
        if len(group) < 2 or np.ptp(external_score) == 0 or np.ptp(ptb_score) == 0:
            rho, pvalue = np.nan, np.nan
        else:
            result = spearmanr(external_score, ptb_score)
            rho, pvalue = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "segment": segment,
                "target": target,
                "n_cells": int(len(group)),
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "ranking_metric": "recoverability_lower; missing targets only",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EXPECTED_METHODS",
    "FittedBenchmarkBundle",
    "PRIMARY_METHODS",
    "RELEASE_NEURAL_SEEDS",
    "SUPPLEMENTARY_METHOD",
    "compare_recoverability_rankings",
    "evaluate_zero_transfer_bundles",
    "load_benchmark_bundles",
    "path_sha256",
]
