"""Frozen fold-10 diagnostic-superclass sensitivity for meta-model prediction gain.

This extended-only analysis never fits a score, reconstructor, or meta-model.  It
subsets the authenticated PTB-XL fold-10 sufficient statistics emitted by
``meta_analysis_v3.py`` and applies that analysis' exact patient-cluster plus
five-seed-mean bootstrap estimator to each prespecified multilabel diagnostic
superclass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ecgcert import lineage
from ecgcert.data.ptbxl import SUPERCLASSES

try:
    from experiments.meta_analysis_v3 import (
        COMMON_PANEL_METHODS,
        META_BOOTSTRAP_SEED,
        META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
        META_SEED_PREDICTION_SCHEMA_VERSION,
        META_SUFFICIENT_SCHEMA_VERSION,
        SCHEMA_VERSION as META_ANALYSIS_SCHEMA_VERSION,
        _bootstrap_effect_and_draws_from_sufficient,
        _expected_common_seeds,
        _read_small_parquet,
        _validate_paired_sufficient_contract,
        _validate_sufficient_contract,
    )
    from experiments.reconstruction_benchmark_v3 import load_ptbxl_manifest
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from meta_analysis_v3 import (
        COMMON_PANEL_METHODS,
        META_BOOTSTRAP_SEED,
        META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
        META_SEED_PREDICTION_SCHEMA_VERSION,
        META_SUFFICIENT_SCHEMA_VERSION,
        SCHEMA_VERSION as META_ANALYSIS_SCHEMA_VERSION,
        _bootstrap_effect_and_draws_from_sufficient,
        _expected_common_seeds,
        _read_small_parquet,
        _validate_paired_sufficient_contract,
        _validate_sufficient_contract,
    )
    from reconstruction_benchmark_v3 import load_ptbxl_manifest


SCHEMA_VERSION = "diagnostic-subgroup-prediction-v3"
STATUS_SCHEMA_VERSION = "diagnostic-subgroup-status-v3"
EFFECT_SCHEMA_VERSION = "diagnostic-subgroup-effect-v3"
MEMBERSHIP_SCHEMA_VERSION = "diagnostic-subgroup-membership-v3"
DRAW_SCHEMA_VERSION = "diagnostic-subgroup-bootstrap-draw-v3"
BOOTSTRAP_SEED_OFFSET = 200
NOT_ESTIMABLE_MIN_PATIENTS = 2

_META_ARTIFACTS = {
    "ptbxl_predictions": "ptbxl_predictions.parquet",
    "ptbxl_seed_predictions": "ptbxl_seed_predictions.parquet",
    "ptbxl_sufficient_stats": "ptbxl_sufficient_stats.parquet",
    "ptbxl_paired_seed_sufficient": "ptbxl_paired_seed_sufficient.parquet",
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _authenticated_meta_artifact(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_name: str,
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"meta-analysis lacks descriptor for {expected_name}")
    relative = descriptor.get("path")
    expected_sha256 = descriptor.get("sha256")
    if relative != expected_name or not isinstance(expected_sha256, str):
        raise ValueError(f"meta-analysis descriptor for {expected_name} is invalid")
    candidate = (root / expected_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # defensive; expected_name is constant
        raise ValueError("meta-analysis artifact escapes its bundle") from exc
    if lineage.artifact_sha256(candidate) != expected_sha256:
        raise ValueError(f"meta-analysis artifact SHA-256 mismatch: {expected_name}")
    return candidate


def _load_frozen_meta_evidence(
    root: Path,
    *,
    release: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    pd.DataFrame,
    pd.DataFrame,
]:
    root = root.resolve()
    summary_path = root / "summary.v3.json"
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version") != META_ANALYSIS_SCHEMA_VERSION
        or summary.get("status") != "complete"
    ):
        raise ValueError("diagnostic sensitivity requires a complete meta-analysis-v3")
    if summary.get("sufficient_stat_schema_version") != META_SUFFICIENT_SCHEMA_VERSION:
        raise ValueError("meta-analysis uses the wrong sufficient-stat schema")
    if (
        summary.get("seed_prediction_schema_version")
        != META_SEED_PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("meta-analysis uses the wrong seed-prediction schema")
    if (
        summary.get("paired_seed_sufficient_schema_version")
        != META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
    ):
        raise ValueError("meta-analysis uses the wrong paired-seed sufficient schema")
    if tuple(summary.get("common_panel_methods", ())) != COMMON_PANEL_METHODS:
        raise ValueError("meta-analysis does not use the frozen four-method common panel")
    expected_seed_contract = {
        method: list(_expected_common_seeds(method)) for method in COMMON_PANEL_METHODS
    }
    if summary.get("exact_model_seed_contract") != expected_seed_contract:
        raise ValueError("meta-analysis violates the exact fitted-seed contract")
    if release:
        if summary.get("release_contract_verified") is not True:
            raise ValueError(
                "release diagnostic sensitivity requires release-verified meta evidence"
            )
        if (
            summary.get("bootstrap_replicates") != 2_000
            or summary.get("seed") != META_BOOTSTRAP_SEED
        ):
            raise ValueError("release meta-analysis does not use the frozen bootstrap contract")

    inventory = summary.get("artifacts")
    if not isinstance(inventory, Mapping):
        raise ValueError("meta-analysis lacks an authenticated artifact inventory")
    paths = {
        key: _authenticated_meta_artifact(
            root,
            inventory.get(key),
            expected_name=filename,
        )
        for key, filename in _META_ARTIFACTS.items()
    }
    sufficient = _read_small_parquet(paths["ptbxl_sufficient_stats"])
    paired = _read_small_parquet(paths["ptbxl_paired_seed_sufficient"])
    _validate_sufficient_contract(sufficient, cohort="PTB-XL")
    _validate_paired_sufficient_contract(paired, cohort="PTB-XL")
    sufficient_patients = set(sufficient["patient_id"].astype(str))
    paired_patients = set(paired["patient_id"].astype(str))
    if sufficient_patients != paired_patients:
        raise ValueError("ordinary and paired sufficient evidence cover different patients")
    return summary, paths, sufficient, paired


def _manifest_patient_membership(
    path: Path,
    *,
    release: bool,
) -> tuple[
    dict[str, Any],
    dict[str, frozenset[str]],
    dict[str, tuple[str, ...]],
]:
    path = path.resolve()
    payload = _load_json(path)
    embedded_sha256 = str(payload.get("manifest_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if lineage.canonical_sha256(unhashed) != embedded_sha256:
        raise ValueError("PTB-XL manifest SHA-256 does not match its content")
    contract = load_ptbxl_manifest(path, release=release)

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("PTB-XL manifest lacks record-level diagnostic labels")
    by_record: dict[str, Mapping[str, Any]] = {}
    record_counts = {superclass: 0 for superclass in SUPERCLASSES}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("PTB-XL manifest contains a non-object record")
        record_id = str(raw.get("record_id", ""))
        labels = raw.get("diagnostic_superclasses")
        if not isinstance(labels, list) or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(
                f"PTB-XL record {record_id!r} lacks explicit diagnostic_superclasses"
            )
        if labels != sorted(set(labels)):
            raise ValueError(
                f"PTB-XL record {record_id!r} diagnostic superclasses are not canonical"
            )
        unknown = set(labels) - set(SUPERCLASSES)
        if unknown:
            raise ValueError(
                f"PTB-XL record {record_id!r} has unknown diagnostic superclasses: "
                f"{sorted(unknown)}"
            )
        if record_id in by_record:
            raise ValueError(f"duplicate PTB-XL record {record_id!r}")
        by_record[record_id] = raw
        for label in labels:
            record_counts[label] += 1

    structure = payload.get("structure")
    if not isinstance(structure, Mapping):
        raise ValueError("PTB-XL manifest lacks diagnostic structure audit")
    if structure.get("diagnostic_superclass_record_counts_multilabel") != record_counts:
        raise ValueError("PTB-XL diagnostic structure counts disagree with its records")
    if structure.get("n_records_with_multiple_diagnostic_superclasses") != sum(
        len(raw["diagnostic_superclasses"]) > 1 for raw in by_record.values()
    ):
        raise ValueError("PTB-XL multilabel record count disagrees with its records")
    if structure.get("n_records_without_diagnostic_superclass") != sum(
        not raw["diagnostic_superclasses"] for raw in by_record.values()
    ):
        raise ValueError("PTB-XL unlabelled record count disagrees with its records")

    patient_labels: dict[str, set[str]] = {}
    patient_records: dict[str, list[str]] = {}
    for record_id in contract.split["test"]:
        raw = by_record[record_id]
        patient_id = str(raw["patient_id"])
        patient_labels.setdefault(patient_id, set()).update(
            raw["diagnostic_superclasses"]
        )
        patient_records.setdefault(patient_id, []).append(record_id)
    return (
        payload,
        {
            patient: frozenset(labels)
            for patient, labels in patient_labels.items()
        },
        {
            patient: tuple(sorted(records))
            for patient, records in patient_records.items()
        },
    )


def _membership_frame(
    patient_labels: Mapping[str, frozenset[str]],
    patient_records: Mapping[str, Sequence[str]],
    *,
    analyzable_patients: set[str],
) -> pd.DataFrame:
    rows = []
    for patient_id in sorted(patient_labels):
        labels = sorted(patient_labels[patient_id])
        records = tuple(patient_records[patient_id])
        rows.append(
            {
                "schema_version": MEMBERSHIP_SCHEMA_VERSION,
                "patient_id": patient_id,
                "diagnostic_superclasses_json": json.dumps(
                    labels, separators=(",", ":")
                ),
                "n_test_records": len(records),
                "test_record_ids_sha256": lineage.canonical_sha256(list(records)),
                "included_in_frozen_meta_evidence": patient_id in analyzable_patients,
            }
        )
    return pd.DataFrame(rows)


def _empty_draw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": pd.Series(dtype="string"),
            "superclass": pd.Series(dtype="string"),
            "source_bootstrap_schema_version": pd.Series(dtype="string"),
            "cohort": pd.Series(dtype="string"),
            "bootstrap_index": pd.Series(dtype="int64"),
            "attempt_index": pd.Series(dtype="int64"),
            "base_seed": pd.Series(dtype="int64"),
            "n_patients": pd.Series(dtype="int64"),
            "patient_draw_sha256": pd.Series(dtype="string"),
            "selected_model_seeds_json": pd.Series(dtype="string"),
            "delta_r2": pd.Series(dtype="float64"),
        }
    )


def _not_estimable_reason(exc: Exception) -> str | None:
    message = str(exc)
    if "R2 is undefined" in message:
        return "nonpositive_subgroup_outcome_variance"
    if "could not obtain the requested number of valid patient bootstrap" in message:
        return "insufficient_valid_patient_bootstrap_draws"
    return None


def analyze(arguments: argparse.Namespace) -> None:
    if arguments.bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if arguments.release:
        if arguments.bootstrap_replicates != 2_000:
            raise ValueError("release sensitivity requires exactly 2,000 bootstraps")
        if arguments.seed != META_BOOTSTRAP_SEED:
            raise ValueError(
                f"release sensitivity requires frozen seed {META_BOOTSTRAP_SEED}"
            )
        try:
            arguments.output_dir.resolve().relative_to(
                (Path.cwd() / "artifacts").resolve()
            )
        except ValueError as exc:
            raise ValueError("release output must be under artifacts/") from exc

    meta_root = arguments.meta_analysis.resolve()
    meta_summary, meta_paths, sufficient, paired = _load_frozen_meta_evidence(
        meta_root,
        release=arguments.release,
    )
    manifest_payload, patient_labels, patient_records = _manifest_patient_membership(
        arguments.ptbxl_manifest,
        release=arguments.release,
    )
    manifest_sha256 = str(manifest_payload["manifest_sha256"])
    manifest_artifact_sha256 = lineage.artifact_sha256(arguments.ptbxl_manifest)
    release_lineage = meta_summary.get("release_lineage")
    if not isinstance(release_lineage, Mapping):
        raise ValueError("meta-analysis lacks source-manifest lineage")
    if (
        release_lineage.get("source_manifest_sha256") != manifest_sha256
        or release_lineage.get("source_manifest_artifact_sha256")
        != manifest_artifact_sha256
    ):
        raise ValueError("meta-analysis and diagnostic manifest lineage disagree")

    analyzable_patients = set(sufficient["patient_id"].astype(str))
    unknown_patients = analyzable_patients - set(patient_labels)
    if unknown_patients:
        raise ValueError(
            "frozen fold-10 meta evidence contains patients outside the manifest test split: "
            f"{sorted(unknown_patients)[:5]}"
        )
    membership = _membership_frame(
        patient_labels,
        patient_records,
        analyzable_patients=analyzable_patients,
    )
    membership_semantic_sha256 = lineage.canonical_sha256(
        {
            patient: sorted(patient_labels[patient])
            for patient in sorted(patient_labels)
        }
    )

    status_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    draw_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for class_index, superclass in enumerate(SUPERCLASSES):
        patients = sorted(
            patient
            for patient in analyzable_patients
            if superclass in patient_labels[patient]
        )
        patient_sha256 = lineage.canonical_sha256(patients)
        bootstrap_seed = arguments.seed + BOOTSTRAP_SEED_OFFSET + class_index
        base_status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "superclass": superclass,
            "n_patients": len(patients),
            "patient_ids_sha256": patient_sha256,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates_requested": arguments.bootstrap_replicates,
        }
        if len(patients) < NOT_ESTIMABLE_MIN_PATIENTS:
            reason = "fewer_than_two_analyzable_patients"
            status_rows.append(
                {
                    **base_status,
                    "status": "not_estimable",
                    "reason_code": reason,
                }
            )
            summary_rows.append(
                {
                    "superclass": superclass,
                    "status": "not_estimable",
                    "n_patients": len(patients),
                    "patient_ids_sha256": patient_sha256,
                    "bootstrap_seed": bootstrap_seed,
                    "not_estimable_reason_code": reason,
                }
            )
            continue

        patient_set = set(patients)
        subgroup_sufficient = sufficient[
            sufficient["patient_id"].astype(str).isin(patient_set)
        ].reset_index(drop=True)
        subgroup_paired = paired[
            paired["patient_id"].astype(str).isin(patient_set)
        ].reset_index(drop=True)
        try:
            effect, draws = _bootstrap_effect_and_draws_from_sufficient(
                subgroup_sufficient,
                paired_sufficient=subgroup_paired,
                cohort="PTB-XL",
                replicates=arguments.bootstrap_replicates,
                seed=bootstrap_seed,
            )
        except (ValueError, RuntimeError) as exc:
            reason = _not_estimable_reason(exc)
            if reason is None:
                raise
            status_rows.append(
                {
                    **base_status,
                    "status": "not_estimable",
                    "reason_code": reason,
                }
            )
            summary_rows.append(
                {
                    "superclass": superclass,
                    "status": "not_estimable",
                    "n_patients": len(patients),
                    "patient_ids_sha256": patient_sha256,
                    "bootstrap_seed": bootstrap_seed,
                    "not_estimable_reason_code": reason,
                }
            )
            continue

        status_rows.append(
            {
                **base_status,
                "status": "estimated",
                "reason_code": "not_applicable",
            }
        )
        effect_row = {
            "schema_version": EFFECT_SCHEMA_VERSION,
            "superclass": superclass,
            "point_delta_r2": effect.point,
            "ci95_lower": effect.ci95[0],
            "ci95_upper": effect.ci95[1],
            "n_patients": len(patients),
            "bootstrap_replicates": effect.replicates,
            "bootstrap_seed": effect.seed,
            "patient_ids_sha256": patient_sha256,
        }
        effect_rows.append(effect_row)
        summary_rows.append(
            {
                "superclass": superclass,
                "status": "estimated",
                "n_patients": len(patients),
                "patient_ids_sha256": patient_sha256,
                "effect": asdict(effect),
            }
        )
        source_schema = str(draws["schema_version"].iloc[0])
        draws = draws.assign(
            source_bootstrap_schema_version=source_schema,
            schema_version=DRAW_SCHEMA_VERSION,
            superclass=superclass,
        )
        draw_frames.append(
            draws[
                [
                    "schema_version",
                    "superclass",
                    "source_bootstrap_schema_version",
                    "cohort",
                    "bootstrap_index",
                    "attempt_index",
                    "base_seed",
                    "n_patients",
                    "patient_draw_sha256",
                    "selected_model_seeds_json",
                    "delta_r2",
                ]
            ]
        )

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    membership_path = output / "patient_membership.parquet"
    status_path = output / "subgroup_status.parquet"
    effects_path = output / "subgroup_effects.parquet"
    draws_path = output / "bootstrap_draws.parquet"
    membership.to_parquet(membership_path, index=False, compression="zstd")
    pd.DataFrame(status_rows).to_parquet(status_path, index=False, compression="zstd")
    pd.DataFrame(
        effect_rows,
        columns=(
            "schema_version",
            "superclass",
            "point_delta_r2",
            "ci95_lower",
            "ci95_upper",
            "n_patients",
            "bootstrap_replicates",
            "bootstrap_seed",
            "patient_ids_sha256",
        ),
    ).to_parquet(effects_path, index=False, compression="zstd")
    (
        pd.concat(draw_frames, ignore_index=True)
        if draw_frames
        else _empty_draw_frame()
    ).to_parquet(draws_path, index=False, compression="zstd")

    meta_summary_path = meta_root / "summary.v3.json"
    input_hashes = {
        "meta_summary": lineage.artifact_sha256(meta_summary_path),
        "ptbxl_manifest": manifest_artifact_sha256,
        **{
            key: lineage.artifact_sha256(path)
            for key, path in sorted(meta_paths.items())
        },
    }
    strict_lineage = lineage.make(
        seed=arguments.seed,
        targets=SUPERCLASSES,
        normalization=(
            "none; consumes frozen fold-10 patient-level log(RMSE) prediction moments"
        ),
        train_ids=(),
        test_ids=sorted(analyzable_patients),
        protocol=SCHEMA_VERSION,
        script=__file__,
        upstream=input_hashes,
        extra={
            "data_sha256": lineage.canonical_sha256(input_hashes),
            "split_sha256": str(manifest_payload["split_sha256"]),
            "checkpoint_sha256": {},
            "analysis_role": "extended_only_frozen_prediction_sensitivity",
        },
    )
    if arguments.release:
        lineage.validate_strict_lineage(strict_lineage)

    artifact_paths = {
        "patient_membership": membership_path,
        "subgroup_status": status_path,
        "subgroup_effects": effects_path,
        "bootstrap_draws": draws_path,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_role": "extended_only",
        "stage15_eligible": False,
        "superclasses": list(SUPERCLASSES),
        "multilabel_membership_rule": (
            "patient belongs when any fold-10 manifest record contains the superclass"
        ),
        "estimand": (
            "Delta R2 of the frozen augmented versus simple fold-9 meta-model on "
            "fold-10 patients carrying the given diagnostic superclass"
        ),
        "model_or_score_refit": {
            "recoverability_map": False,
            "reconstructor": False,
            "meta_model": False,
            "hyperparameter": False,
        },
        "bootstrap": {
            "unit": "shared patient cluster",
            "neural_seed_estimand": (
                "five fitted seeds resampled five times with replacement and re-averaged"
            ),
            "replicates": arguments.bootstrap_replicates,
            "base_seed": arguments.seed,
            "superclass_seed_offset": BOOTSTRAP_SEED_OFFSET,
            "estimator_implementation": (
                "experiments.meta_analysis_v3."
                "_bootstrap_effect_and_draws_from_sufficient"
            ),
        },
        "n_manifest_test_patients": len(patient_labels),
        "n_analyzable_fold10_patients": len(analyzable_patients),
        "n_manifest_test_patients_without_meta_evidence": (
            len(patient_labels) - len(analyzable_patients)
        ),
        "n_analyzable_patients_without_prespecified_superclass": sum(
            not patient_labels[patient] for patient in analyzable_patients
        ),
        "patient_membership_semantic_sha256": membership_semantic_sha256,
        "subgroups": summary_rows,
        "input_sha256": input_hashes,
        "source_manifest": {
            "content_sha256": manifest_sha256,
            "artifact_sha256": manifest_artifact_sha256,
            "split_sha256": manifest_payload["split_sha256"],
        },
        "source_meta_analysis": {
            "schema_version": meta_summary["schema_version"],
            "summary_sha256": input_hashes["meta_summary"],
            "primary_effect": meta_summary.get("ptbxl"),
        },
        "artifacts": {
            key: {
                "path": path.name,
                "sha256": lineage.artifact_sha256(path),
            }
            for key, path in artifact_paths.items()
        },
        "lineage": strict_lineage,
    }
    _write_json(output / "summary.v3.json", summary)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-analysis", type=Path, required=True)
    parser.add_argument("--ptbxl-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=META_BOOTSTRAP_SEED)
    parser.add_argument("--release", action="store_true")
    return parser.parse_args()


def main() -> None:
    analyze(_arguments())


if __name__ == "__main__":
    main()
