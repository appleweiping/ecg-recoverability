"""Build rank-robust, target-specific recoverability maps for ICASSP 2027.

This is the submission-path replacement for ``recoverability_maps.py``.  It fits
the spatial model on PTB-XL folds 1--7, freezes the only Gaussian observation
regularizer on fold 8, and then evaluates all 255 independent-lead subsets under
patient-cluster bootstrap uncertainty.  It never reads or writes ``results/``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.data.ptbxl import SUPERCLASSES
from ecgcert.physics import LEADS, LEAD_INDEX, fit_spatial_subspace, observed_dipole
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.recoverability import (
    REGULARIZATION_GRID_MV2,
    bootstrap_spatial_model_bank,
    gaussian_prior_ambiguity_per_lead,
    tune_gaussian_regularization,
)
try:  # ``python experiments/robust_maps_v3.py`` puts experiments/ on sys.path.
    from experiments.reconstruction_benchmark_v3 import (
        PTBXLManifestV3,
        _file_sha256,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the DAG subprocess.
    from reconstruction_benchmark_v3 import (  # type: ignore[no-redef]
        PTBXLManifestV3,
        _file_sha256,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )


SCHEMA_VERSION = "robust-recoverability-map-v3"
SENSITIVITY_CHOICES = ("p-wave", "100hz", "delineator", "raw12", "diagnosis")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False, compression="zstd")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet is mandatory for ICASSP artifacts; install the locked pyarrow dependency"
        ) from exc


def _configuration_id(configuration: Sequence[str]) -> str:
    return "+".join(configuration)


def _safe_quantile(values: np.ndarray, probability: float, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanquantile(values, probability, axis=axis)


def _model_metrics(model, configuration, observation_variance_mv2: float):
    """Compute ambiguity and decomposition diagnostics from one observed SVD."""

    ambiguity = gaussian_prior_ambiguity_per_lead(
        model,
        configuration,
        observation_variance_mv2=observation_variance_mv2,
    )
    decomposition = observed_dipole(model.M, configuration, rcond=None)
    q = model.M.shape[1]
    unobserved = model.M @ (np.eye(q) - decomposition.P_obs)
    denominator = np.linalg.norm(model.M, axis=1)
    eta_normalized = np.divide(
        np.linalg.norm(unobserved, axis=1),
        denominator,
        out=np.full(len(LEADS), np.nan),
        where=denominator > 1e-12,
    )
    reconstruction_gain = model.M @ decomposition.pinv
    kappa_per_target = np.linalg.norm(reconstruction_gain, axis=1)
    kappa_global = float(np.linalg.norm(reconstruction_gain, ord=2))
    positive = decomposition.sv[decomposition.sv > decomposition.threshold]
    condition = float(positive[0] / positive[-1]) if positive.size else 1.0
    return {
        "ambiguity": ambiguity,
        "eta_normalized": eta_normalized,
        "kappa_per_target": kappa_per_target,
        "kappa_global": kappa_global,
        "configuration_rank": int(decomposition.rank),
        "condition_number": condition,
    }


def _training_predictors(bank, configuration: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Training-only target RMS and maximum target--observed correlation."""

    statistics = bank.statistics
    count = float(statistics.n_samples)
    centred_sum = statistics.sums.sum(axis=0)
    centred_cross = statistics.crossproducts.sum(axis=0)
    raw_sum = centred_sum + count * statistics.origin
    raw_cross = (
        centred_cross
        + np.outer(statistics.origin, centred_sum)
        + np.outer(centred_sum, statistics.origin)
        + count * np.outer(statistics.origin, statistics.origin)
    )
    target_rms = np.sqrt(np.clip(np.diag(raw_cross) / count, 0.0, None))
    covariance = raw_cross / count - np.outer(raw_sum / count, raw_sum / count)
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    correlation = np.divide(
        covariance,
        np.outer(scale, scale),
        out=np.zeros_like(covariance),
        where=np.outer(scale, scale) > 1e-15,
    )
    observed_indices = [LEAD_INDEX[lead] for lead in configuration]
    max_correlation = np.max(np.abs(correlation[:, observed_indices]), axis=1)
    return target_rms, max_correlation


def summarize_model_bank(
    bank,
    configurations: Sequence[Sequence[str]],
    *,
    segment: str,
    observation_variance_mv2: float,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rank-level and cross-rank map cells without treating cells as iid."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    alpha = (1.0 - confidence) / 2.0
    rank_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    point_by_rank = {model.rank: model for model in bank.point_models}
    if set(point_by_rank) != set(bank.ranks):
        raise ValueError("summarize_model_bank expects one primary basis variant")

    for configuration_value in configurations:
        configuration = tuple(configuration_value)
        config_id = _configuration_id(configuration)
        training_target_rms, training_max_correlation = _training_predictors(bank, configuration)
        point_metrics = {
            rank: _model_metrics(model, configuration, observation_variance_mv2)
            for rank, model in point_by_rank.items()
        }
        boot_metrics = {
            rank: {
                "ambiguity": np.empty((bank.n_boot, len(LEADS))),
                "eta_normalized": np.empty((bank.n_boot, len(LEADS))),
                "kappa_per_target": np.empty((bank.n_boot, len(LEADS))),
                "kappa_global": np.empty(bank.n_boot),
                "configuration_rank": np.empty(bank.n_boot, dtype=np.int16),
                "condition_number": np.empty(bank.n_boot),
            }
            for rank in bank.ranks
        }
        for bootstrap_index, group in enumerate(bank.bootstrap_models):
            for model in group:
                metrics = _model_metrics(model, configuration, observation_variance_mv2)
                for name, values in boot_metrics[model.rank].items():
                    values[bootstrap_index] = metrics[name]

        per_rank_summary: dict[int, dict[str, np.ndarray | float | int]] = {}
        for rank in bank.ranks:
            point = point_metrics[rank]
            bootstrap = boot_metrics[rank]
            summary = {
                "ambiguity_lower": _safe_quantile(bootstrap["ambiguity"], alpha),
                "ambiguity_upper": _safe_quantile(bootstrap["ambiguity"], 1.0 - alpha),
                "eta_upper": _safe_quantile(bootstrap["eta_normalized"], 1.0 - alpha),
                "kappa_target_upper": _safe_quantile(
                    bootstrap["kappa_per_target"], 1.0 - alpha
                ),
                "kappa_global_upper": float(
                    _safe_quantile(bootstrap["kappa_global"], 1.0 - alpha)
                ),
                "configuration_rank": int(point["configuration_rank"]),
                "condition_number": float(point["condition_number"]),
            }
            per_rank_summary[rank] = summary
            for target_index, target in enumerate(LEADS):
                rank_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "segment": segment,
                        "configuration": config_id,
                        "observed_leads": ",".join(configuration),
                        "n_observed": len(configuration),
                        "target": target,
                        "target_observed": target in configuration,
                        "rank": rank,
                        "ambiguity_point_mv": float(point["ambiguity"][target_index]),
                        "ambiguity_q025_mv": float(summary["ambiguity_lower"][target_index]),
                        "ambiguity_q975_mv": float(summary["ambiguity_upper"][target_index]),
                        "eta_normalized_point": float(point["eta_normalized"][target_index]),
                        "eta_normalized_q975": float(summary["eta_upper"][target_index]),
                        "kappa_target_point": float(point["kappa_per_target"][target_index]),
                        "kappa_target_q975": float(
                            summary["kappa_target_upper"][target_index]
                        ),
                        "kappa_global_point": float(point["kappa_global"]),
                        "kappa_global_q975": float(summary["kappa_global_upper"]),
                        "configuration_rank": int(summary["configuration_rank"]),
                        "condition_number": float(summary["condition_number"]),
                        "observation_variance_mv2": observation_variance_mv2,
                        "target_rms": float(training_target_rms[target_index]),
                        "max_target_observed_correlation": float(
                            training_max_correlation[target_index]
                        ),
                        "bootstrap_replicates": bank.n_boot,
                        "bootstrap_rejected_draws": bank.rejected_draws,
                        "bootstrap_rejection_fraction": bank.rejection_fraction,
                    }
                )

        for target_index, target in enumerate(LEADS):
            ambiguity_point = np.asarray(
                [point_metrics[rank]["ambiguity"][target_index] for rank in bank.ranks]
            )
            ambiguity_upper = np.asarray(
                [per_rank_summary[rank]["ambiguity_upper"][target_index] for rank in bank.ranks]
            )
            eta_upper = np.asarray(
                [per_rank_summary[rank]["eta_upper"][target_index] for rank in bank.ranks]
            )
            kappa_target_upper = np.asarray(
                [per_rank_summary[rank]["kappa_target_upper"][target_index] for rank in bank.ranks]
            )
            kappa_global_upper = np.asarray(
                [per_rank_summary[rank]["kappa_global_upper"] for rank in bank.ranks]
            )
            condition_by_rank = np.asarray(
                [per_rank_summary[rank]["condition_number"] for rank in bank.ranks]
            )
            rank_by_model = np.asarray(
                [per_rank_summary[rank]["configuration_rank"] for rank in bank.ranks]
            )
            map_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "segment": segment,
                    "configuration": config_id,
                    "observed_leads": ",".join(configuration),
                    "n_observed": len(configuration),
                    "target": target,
                    "target_observed": target in configuration,
                    "ambiguity_robust_mv": float(np.nanmax(ambiguity_upper)),
                    "recoverability_lower": float(np.clip(1.0 - np.nanmax(eta_upper), 0, 1)),
                    "ambiguity_rank_span_mv": float(
                        np.nanmax(ambiguity_point) - np.nanmin(ambiguity_point)
                    ),
                    "log10_kappa_target_upper": float(
                        np.log10(max(float(np.nanmax(kappa_target_upper)), np.finfo(float).tiny))
                    ),
                    "log10_kappa_global_upper": float(
                        np.log10(max(float(np.nanmax(kappa_global_upper)), np.finfo(float).tiny))
                    ),
                    "configuration_rank_min": int(rank_by_model.min()),
                    "configuration_rank_max": int(rank_by_model.max()),
                    "log10_condition_max": float(
                        np.log10(max(float(np.nanmax(condition_by_rank)), 1.0))
                    ),
                    "observation_variance_mv2": observation_variance_mv2,
                    "target_rms": float(training_target_rms[target_index]),
                    "max_target_observed_correlation": float(
                        training_max_correlation[target_index]
                    ),
                    "bootstrap_replicates": bank.n_boot,
                    "bootstrap_rejected_draws": bank.rejected_draws,
                    "bootstrap_rejection_fraction": bank.rejection_fraction,
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(map_rows)


def _parse_segments(value: str) -> tuple[str, ...]:
    segments = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    unknown = set(segments) - {"P", "QRS", "ST", "T"}
    if not segments or unknown:
        raise argparse.ArgumentTypeError(f"segments must be selected from P,QRS,ST,T; got {value}")
    return segments


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "sensitivity"), default="primary")
    parser.add_argument("--sensitivity", choices=SENSITIVITY_CHOICES)
    parser.add_argument("--diagnosis-class", choices=SUPERCLASSES)
    parser.add_argument("--primary-rank-maps", type=Path)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/robust_maps"))
    parser.add_argument("--segments", type=_parse_segments, default=PRIMARY_SEGMENTS)
    parser.add_argument("--rate", type=int, default=PRIMARY_RATE_HZ)
    parser.add_argument("--population", choices=("all", "norm"), default="all")
    parser.add_argument("--delineator", choices=("dwt", "peak"), default="dwt")
    parser.add_argument(
        "--basis-variant",
        choices=("independent8_lifted", "raw12_pca"),
        default="independent8_lifted",
    )
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-per-record", type=int, default=40)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--observation-variance-mv2", type=float)
    parser.add_argument("--release", action="store_true")
    return parser.parse_args()


def _resolve_sensitivity(
    arguments: argparse.Namespace, primary_summary: Mapping[str, Any]
) -> None:
    """Apply exactly one preregistered sensitivity change in place.

    Every sensitivity reuses the fold-8 regularizer selected by the primary map;
    no extended analysis is allowed to tune a second value.
    """

    arguments.observation_variance_mv2 = float(
        primary_summary["observation_variance_mv2"]
    )
    if arguments.sensitivity == "p-wave":
        arguments.segments = ("P",)
    elif arguments.sensitivity == "100hz":
        arguments.rate = 100
    elif arguments.sensitivity == "delineator":
        arguments.delineator = "peak"
    elif arguments.sensitivity == "raw12":
        arguments.basis_variant = "raw12_pca"
    elif arguments.sensitivity == "diagnosis":
        if arguments.diagnosis_class is None:
            raise SystemExit("diagnosis sensitivity requires --diagnosis-class")
    else:  # argparse prevents this, but fail closed for programmatic callers.
        raise SystemExit(f"unsupported sensitivity {arguments.sensitivity!r}")


def _expected_release_fields(arguments: argparse.Namespace) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "segments": PRIMARY_SEGMENTS,
        "rate": PRIMARY_RATE_HZ,
        "population": "all",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
    }
    if arguments.mode == "sensitivity":
        if arguments.sensitivity == "p-wave":
            expected["segments"] = ("P",)
        elif arguments.sensitivity == "100hz":
            expected["rate"] = 100
        elif arguments.sensitivity == "delineator":
            expected["delineator"] = "peak"
        elif arguments.sensitivity == "raw12":
            expected["basis_variant"] = "raw12_pca"
    return expected


def validate_release_arguments(arguments: argparse.Namespace) -> None:
    """Enforce the primary protocol or one isolated preregistered sensitivity."""

    if not arguments.release:
        return
    StudyProtocol().validate()
    violations: list[str] = []
    if arguments.max_records is not None:
        violations.append("--max-records is forbidden")
    if arguments.n_bootstrap != BOOTSTRAP_REPLICATES:
        violations.append(f"--n-bootstrap must equal {BOOTSTRAP_REPLICATES}")
    if arguments.max_per_record != 40:
        violations.append("--max-per-record must equal the locked value 40")
    if arguments.manifest is None:
        violations.append("--manifest is required for a release run")
    if arguments.mode == "primary":
        if arguments.sensitivity is not None or arguments.primary_rank_maps is not None:
            violations.append("primary release cannot declare a sensitivity")
        if arguments.diagnosis_class is not None:
            violations.append("primary release cannot filter a diagnosis")
        if arguments.observation_variance_mv2 is not None:
            violations.append("primary release must select regularization on fold 8")
    else:
        if arguments.sensitivity is None:
            violations.append("sensitivity release requires --sensitivity")
        if arguments.primary_rank_maps is None:
            violations.append("sensitivity release requires --primary-rank-maps")
        if arguments.observation_variance_mv2 is None:
            violations.append("sensitivity release must reuse the primary fold-8 regularizer")
        if arguments.sensitivity == "diagnosis" and arguments.diagnosis_class is None:
            violations.append("diagnosis sensitivity requires --diagnosis-class")
        if arguments.sensitivity != "diagnosis" and arguments.diagnosis_class is not None:
            violations.append("--diagnosis-class is valid only for diagnosis sensitivity")
    for field, expected in _expected_release_fields(arguments).items():
        actual = getattr(arguments, field)
        if field == "segments":
            actual = tuple(actual)
        if actual != expected:
            violations.append(f"--{field.replace('_', '-')} must equal {expected!r}")
    if violations:
        raise SystemExit("release protocol violation: " + "; ".join(violations))


def _load_primary_summary(primary_rank_maps: Path) -> dict[str, Any]:
    """Load a complete primary map and verify the bytes named by its summary."""

    root = primary_rank_maps.resolve()
    path = root / "summary.v3.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_mode": "primary",
        "population": "all",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
        "segments": list(PRIMARY_SEGMENTS),
        "rate_hz": PRIMARY_RATE_HZ,
        "ranks": list(RANK_GRID),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "patient",
    }
    mismatches = [key for key, expected in frozen.items() if value.get(key) != expected]
    if mismatches:
        raise RuntimeError(f"sensitivity requires a frozen primary map: {mismatches}")
    paths = value.get("artifacts")
    hashes = value.get("artifact_sha256")
    required = {"rank_path", "map_cells", "regularization_tuning", "patient_audit"}
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError("primary map summary lacks artifact paths or direct SHA-256 values")
    if not required <= set(paths) or not required <= set(hashes):
        raise RuntimeError("primary map summary has incomplete artifact lineage")
    for name in sorted(required):
        artifact = (root / str(paths[name])).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"primary artifact escapes its bundle: {name}") from exc
        if not artifact.is_file() or lineage.artifact_sha256(artifact) != hashes[name]:
            raise RuntimeError(f"primary artifact SHA-256 mismatch: {name}")
    return value


def _verify_manifest_identity(
    contract: PTBXLManifestV3,
    db: PTBXL,
    *,
    rate: int,
    release: bool,
) -> None:
    """Verify the frozen files and their exact PTB-XL database identities."""

    payload = json.loads(contract.path.read_text(encoding="utf-8"))
    for name, field in (
        ("ptbxl_database.csv", "metadata_sha256"),
        ("scp_statements.csv", "scp_statements_sha256"),
    ):
        expected = payload.get(field)
        candidate = contract.root / name
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"PTB-XL manifest lacks {field}")
        if not candidate.is_file() or _file_sha256(candidate) != expected:
            raise ValueError(f"PTB-XL {name} hash mismatch")

    verification_ids = (
        tuple(contract.records)
        if release
        else contract.record_ids("train") + contract.record_ids("tune")
    )
    _verify_manifest_files(contract, verification_ids, rate=rate)
    _validate_database_identity(db, contract, verification_ids)
    if release and {str(value) for value in db.meta.index} != set(contract.records):
        raise ValueError("release manifest must cover the complete PTB-XL database")
    columns = {100: "filename_lr", 500: "filename_hr"}
    for record_id in verification_ids:
        record = contract.records[record_id]
        row = db.meta.loc[int(record_id)]
        if int(row["strat_fold"]) != int(record["strat_fold"]):
            raise ValueError(f"manifest fold mismatch for record {record_id}")
        entry = record["files"].get(str(rate), {})
        if str(entry.get("record")) != str(row[columns[rate]]).replace("\\", "/"):
            raise ValueError(f"manifest record path mismatch for record {record_id}")


def _role_ids(
    contract: PTBXLManifestV3,
    db: PTBXL,
    diagnosis_class: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select from frozen manifest roles, optionally with multi-label diagnosis membership."""

    train = contract.record_ids("train")
    tune = contract.record_ids("tune")
    if diagnosis_class is None:
        return train, tune
    if diagnosis_class not in SUPERCLASSES:
        raise ValueError(f"unsupported diagnostic superclass {diagnosis_class!r}")

    def included(record_id: str) -> bool:
        labels = db.meta.loc[int(record_id), "superclass"]
        return diagnosis_class in labels

    return tuple(filter(included, train)), tuple(filter(included, tune))


def _artifact_hashes(output_dir: Path, paths: Mapping[str, str]) -> dict[str, str]:
    """Return direct SHA-256 values for every emitted, non-summary artifact."""

    return {
        name: lineage.artifact_sha256(output_dir / relative_path)
        for name, relative_path in paths.items()
    }


def main() -> None:
    arguments = _arguments()
    primary_summary = None
    if arguments.mode == "sensitivity":
        if arguments.sensitivity is None or arguments.primary_rank_maps is None:
            raise SystemExit("sensitivity mode requires --sensitivity and --primary-rank-maps")
        primary_summary = _load_primary_summary(arguments.primary_rank_maps)
        _resolve_sensitivity(arguments, primary_summary)
    elif arguments.sensitivity is not None or arguments.primary_rank_maps is not None:
        raise SystemExit("--sensitivity/--primary-rank-maps are valid only in sensitivity mode")
    elif arguments.diagnosis_class is not None:
        raise SystemExit("--diagnosis-class is valid only in sensitivity mode")
    validate_release_arguments(arguments)
    if arguments.n_bootstrap < 1:
        raise SystemExit("--n-bootstrap must be positive")
    if arguments.max_per_record < 1:
        raise SystemExit("--max-per-record must be positive")
    if arguments.rate not in {100, 500}:
        raise SystemExit("--rate must be one of the PTB-XL source rates: 100 or 500")
    os.environ["ECG_DELINEATOR"] = arguments.delineator

    if arguments.manifest is None:
        raise RuntimeError("--manifest is required; ad-hoc database splitting is forbidden")
    contract = load_ptbxl_manifest(arguments.manifest)
    requested_root = arguments.ptbxl_root.resolve()
    default_root = Path("data/ptbxl").resolve()
    if requested_root != default_root and requested_root != contract.root:
        raise RuntimeError("--ptbxl-root disagrees with --manifest")
    arguments.ptbxl_root = contract.root
    db = PTBXL(contract.root)
    _verify_manifest_identity(
        contract,
        db,
        rate=arguments.rate,
        release=arguments.release,
    )
    role_diagnosis = arguments.diagnosis_class
    if role_diagnosis is None and arguments.population == "norm":
        role_diagnosis = "NORM"
    train_ids, tune_ids = _role_ids(contract, db, role_diagnosis)
    if primary_summary is not None:
        if primary_summary.get("data_manifest_sha256") != contract.manifest_sha256:
            raise RuntimeError("sensitivity and primary map use different PTB-XL manifests")
        if primary_summary.get("split_sha256") != contract.split_sha256:
            raise RuntimeError("sensitivity and primary map use different PTB-XL role assignments")
    train_data, train_audit = db.collect_all_segments_audited(
        train_ids,
        rate=arguments.rate,
        max_per_record=arguments.max_per_record,
        max_records=arguments.max_records,
        seed=arguments.seed,
    )
    tune_data, tune_audit = db.collect_all_segments_audited(
        tune_ids,
        rate=arguments.rate,
        max_per_record=arguments.max_per_record,
        max_records=arguments.max_records,
        seed=arguments.seed + 1,
    )
    for segment in arguments.segments:
        if train_data[segment][0].shape[0] < 2 or len(set(train_data[segment][2])) < 2:
            raise RuntimeError(f"insufficient training patients for {segment}")
        if tune_data[segment][0].shape[0] < 2:
            raise RuntimeError(f"insufficient fold-8 tuning samples for {segment}")

    point_models = {
        segment: tuple(
            fit_spatial_subspace(
                train_data[segment][0],
                rank=rank,
                basis_variant=arguments.basis_variant,
                fit_cohort=f"PTB-XL/folds1-7/{segment}",
            )
            for rank in RANK_GRID
        )
        for segment in arguments.segments
    }
    if arguments.observation_variance_mv2 is None:
        selection = tune_gaussian_regularization(
            point_models,
            {
                segment: (tune_data[segment][0], tune_data[segment][2])
                for segment in arguments.segments
            },
            deep_configuration_panel(),
            grid_mv2=REGULARIZATION_GRID_MV2,
        )
        observation_variance = selection.selected_variance_mv2
        tuning_table = selection.table
        tuning_source = "PTB-XL fold 8"
    else:
        observation_variance = float(arguments.observation_variance_mv2)
        if not np.isfinite(observation_variance) or observation_variance <= 0:
            raise SystemExit("--observation-variance-mv2 must be finite and positive")
        tuning_table = pd.DataFrame(
            [{"observation_variance_mv2": observation_variance, "externally_frozen": True}]
        )
        tuning_source = "command-line frozen sensitivity value"

    rank_frames = []
    map_frames = []
    bootstrap_rejections = {}
    configurations = all_independent_configurations()
    for segment_index, segment in enumerate(arguments.segments):
        X, _, patient_ids = train_data[segment]
        bank = bootstrap_spatial_model_bank(
            X,
            patient_ids,
            ranks=RANK_GRID,
            basis_variants=(arguments.basis_variant,),
            fit_cohort=f"PTB-XL/folds1-7/{segment}",
            n_boot=arguments.n_bootstrap,
            seed=arguments.seed + 1000 * segment_index,
        )
        rank_frame, map_frame = summarize_model_bank(
            bank,
            configurations,
            segment=segment,
            observation_variance_mv2=observation_variance,
        )
        rank_frames.append(rank_frame)
        map_frames.append(map_frame)
        bootstrap_rejections[segment] = {
            "rejected_draws": bank.rejected_draws,
            "rejection_fraction": bank.rejection_fraction,
        }

    rank_path = pd.concat(rank_frames, ignore_index=True)
    map_cells = pd.concat(map_frames, ignore_index=True)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "rank_path": "rank_path.parquet",
        "map_cells": "map_cells.parquet",
        "regularization_tuning": "regularization_tuning.parquet",
        "patient_audit": "patient_audit.json",
    }
    _write_parquet(rank_path, output_dir / artifact_paths["rank_path"])
    _write_parquet(map_cells, output_dir / artifact_paths["map_cells"])
    _write_parquet(tuning_table, output_dir / artifact_paths["regularization_tuning"])
    _write_json(
        output_dir / artifact_paths["patient_audit"],
        {
            "schema_version": SCHEMA_VERSION,
            "train": train_audit.to_dict(),
            "tune": tune_audit.to_dict(),
        },
    )
    protocol = StudyProtocol()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_scope": "model-conditional predictive score; not a certificate or safety claim",
        "cohort": "PTB-XL",
        "population": (
            arguments.population
            if arguments.diagnosis_class is None
            else f"multilabel:{arguments.diagnosis_class}"
        ),
        "analysis_mode": arguments.mode,
        "delineator": arguments.delineator,
        "basis_variant": arguments.basis_variant,
        "segments": list(arguments.segments),
        "rate_hz": arguments.rate,
        "ranks": list(RANK_GRID),
        "n_configurations": len(configurations),
        "n_rank_rows": len(rank_path),
        "n_map_cells": len(map_cells),
        "bootstrap_replicates": arguments.n_bootstrap,
        "bootstrap_unit": "patient",
        "bootstrap_rank_deficient_draws": bootstrap_rejections,
        "seed": arguments.seed,
        "observation_variance_mv2": observation_variance,
        "regularization_grid_mv2": list(REGULARIZATION_GRID_MV2),
        "regularization_tuning_source": tuning_source,
        "deep_panel_sha256": configuration_panel_sha256(),
        "split_sha256": contract.split_sha256,
        "data_manifest_sha256": contract.manifest_sha256,
        "train_role_ids_sha256": lineage.canonical_sha256(list(train_ids)),
        "tune_role_ids_sha256": lineage.canonical_sha256(list(tune_ids)),
        "train_audit_sha256": train_audit.sha256(),
        "tune_audit_sha256": tune_audit.sha256(),
        "protocol": asdict(protocol),
        "protocol_sha256": lineage.canonical_sha256(asdict(protocol)),
        "artifacts": artifact_paths,
        "artifact_sha256": _artifact_hashes(output_dir, artifact_paths),
    }
    if arguments.sensitivity is not None:
        summary["sensitivity"] = arguments.sensitivity
    if arguments.diagnosis_class is not None:
        summary["diagnosis_class"] = arguments.diagnosis_class
        summary["diagnosis_membership"] = "multi-label contains superclass"
    _write_json(output_dir / "summary.v3.json", summary)


if __name__ == "__main__":
    main()
