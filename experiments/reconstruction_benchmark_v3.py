"""Run the frozen PTB-XL reconstruction benchmark and emit v3 artifacts.

Each invocation evaluates exactly one method.  All methods consume the same
patient split, 500 Hz records, delineated windows, whole-lead masks, target-RMS
normalization, and missing-target scorer.  Public methods are their pinned
official implementations; missing source, data, commands, or checkpoints are
hard failures and are never replaced by a local surrogate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators import (
    ECG_RECOVER,
    IMPUTE_ECG,
    ECGrecoverReconstructor,
    ImputeECGReconstructor,
    LowRankConditionalMeanReconstructor,
    MaskedUNetReconstructor,
    ReconstructorConfig,
    RidgeLeadReconstructor,
    TrainManifest,
)
from ecgcert.estimators.api import sha256_file
from ecgcert.protocol import (
    PatientSplit,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.physics.dipolar_subspace import INDEPENDENT_LEADS
from ecgcert.reconstruction import (
    TRAINING_PREDICTORS_FILENAME,
    OfficialCommandBridgeReconstructor,
    SCHEMA_VERSION,
    checkpoint_descriptor,
    evaluate_reconstructor,
    evaluation_records_sha256,
    write_benchmark_artifacts,
    write_bundle_metadata,
    EvaluationRecord,
    TrainingPredictorAccumulator,
    training_predictor_lookup,
)


METHODS = ("lowrank", "ridge", "masked-unet", "imputeecg", "ecgrecover")
NEURAL_METHODS = frozenset({"masked-unet", "imputeecg", "ecgrecover"})
OFFICIAL_METHODS = frozenset({"imputeecg", "ecgrecover"})
RELEASE_NEURAL_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TUNING = {
    "lowrank": {"rank": 3, "noise_variance": 1e-6},
    "ridge": {"ridge_lambda": 1e-3},
    "masked-unet": {
        "epochs": 60,
        "batch_size": 16,
        "width": 48,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "deterministic": True,
    },
    "imputeecg": {"epochs": 100, "batch_size": 128, "num_workers": 8},
    "ecgrecover": {},
}


@dataclass(frozen=True)
class PTBXLManifestV3:
    path: Path
    root: Path
    records: Mapping[str, Mapping[str, Any]]
    split: Mapping[str, tuple[str, ...]]
    manifest_sha256: str
    split_sha256: str

    def record_ids(self, role: str, limit: int | None = None) -> tuple[str, ...]:
        values = self.split[role]
        return values if limit is None else values[:limit]


def _parse_csv(value: str) -> tuple[str, ...]:
    out = tuple(item.strip() for item in value.split(",") if item.strip())
    if not out:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return out


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in _parse_csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank-maps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstreams", type=Path)
    parser.add_argument("--tuning-config", type=Path)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=int, default=PRIMARY_RATE_HZ)
    parser.add_argument("--segments", type=_parse_csv, default=PRIMARY_SEGMENTS)
    parser.add_argument("--delineator", choices=("dwt", "peak"), default="dwt")
    parser.add_argument("--seeds", type=_parse_seeds)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--release", action="store_true")
    return parser


def resolve_model_seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    neural_release = arguments.release and arguments.method in NEURAL_METHODS
    default = RELEASE_NEURAL_SEEDS if neural_release else (0,)
    seeds = default if arguments.seeds is None else tuple(arguments.seeds)
    if arguments.method not in NEURAL_METHODS and seeds != (0,):
        raise ValueError("lowrank and ridge have no stochastic model seed; use only seed 0")
    if arguments.release and arguments.method in NEURAL_METHODS and seeds != RELEASE_NEURAL_SEEDS:
        raise ValueError("release neural methods require exactly model seeds 0,1,2,3,4")
    return seeds


def validate_release_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_records is not None and arguments.max_records < 1:
        raise ValueError("--max-records must be positive")
    if arguments.max_configurations is not None and arguments.max_configurations < 1:
        raise ValueError("--max-configurations must be positive")
    resolve_model_seeds(arguments)
    if not arguments.release:
        return
    violations = []
    if arguments.max_records is not None:
        violations.append("--max-records is forbidden")
    if arguments.max_configurations is not None:
        violations.append("--max-configurations is forbidden")
    if arguments.rate != PRIMARY_RATE_HZ:
        violations.append(f"--rate must equal {PRIMARY_RATE_HZ}")
    if tuple(arguments.segments) != PRIMARY_SEGMENTS:
        violations.append(f"--segments must equal {PRIMARY_SEGMENTS}")
    if arguments.delineator != "dwt":
        violations.append("--delineator must be dwt")
    if arguments.tuning_config is None:
        violations.append("--tuning-config is required (frozen fold-8 choices)")
    try:
        arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
    except ValueError:
        violations.append("--output-dir must be under artifacts/ for release")
    if violations:
        raise ValueError("release protocol violation: " + "; ".join(violations))


def load_ptbxl_manifest(path: str | Path) -> PTBXLManifestV3:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PTB-XL manifest {manifest_path}: {exc}") from exc
    required = {
        "schema_version",
        "cohort",
        "root",
        "records",
        "split",
        "split_sha256",
        "manifest_sha256",
    }
    missing = required - set(payload) if isinstance(payload, Mapping) else required
    if missing:
        raise ValueError(f"PTB-XL manifest is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "ptbxl-manifest-v3" or payload["cohort"] != "PTB-XL":
        raise ValueError("benchmark requires the fold-aware ptbxl-manifest-v3 schema")
    expected_manifest_hash = payload["manifest_sha256"]
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256")
    if lineage.canonical_sha256(unhashed) != expected_manifest_hash:
        raise ValueError("PTB-XL manifest SHA-256 does not match its content")

    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("PTB-XL manifest records must be a non-empty list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise ValueError("PTB-XL manifest record must be an object")
        record_required = {"record_id", "patient_id", "strat_fold", "files"}
        if record_required - set(record):
            raise ValueError("PTB-XL manifest record is incomplete")
        record_id = str(record["record_id"])
        if record_id in records:
            raise ValueError(f"duplicate PTB-XL record_id {record_id}")
        if not str(record["patient_id"]):
            raise ValueError(f"record {record_id} has no patient_id")
        records[record_id] = record

    raw_split = payload["split"]
    roles = ("train", "tune", "calibration", "test")
    if not isinstance(raw_split, Mapping) or not set(roles) <= set(raw_split):
        raise ValueError(f"PTB-XL split must contain {roles}")
    split = {role: tuple(str(value) for value in raw_split[role]) for role in roles}
    flat = [record_id for role in roles for record_id in split[role]]
    if len(flat) != len(set(flat)):
        raise ValueError("PTB-XL split contains duplicate/cross-role records")
    if set(flat) != set(records):
        raise ValueError("PTB-XL split must assign every manifest record exactly once")
    computed_split_sha256 = PatientSplit(
        train=split["train"],
        tune=split["tune"],
        calibration=split["calibration"],
        test=split["test"],
    ).sha256()
    if computed_split_sha256 != payload["split_sha256"]:
        raise ValueError("PTB-XL split SHA-256 does not match its role assignments")
    expected_folds = {
        "train": set(range(1, 8)),
        "tune": {8},
        "calibration": {9},
        "test": {10},
    }
    patient_roles: dict[str, set[str]] = {}
    for role in roles:
        for record_id in split[role]:
            record = records[record_id]
            if int(record["strat_fold"]) not in expected_folds[role]:
                raise ValueError(f"record {record_id} has wrong fold for role {role}")
            patient_roles.setdefault(str(record["patient_id"]), set()).add(role)
    leaking = sorted(patient for patient, values in patient_roles.items() if len(values) != 1)
    if leaking:
        raise ValueError(f"patient leakage across PTB-XL roles: {leaking[:5]}")
    root = Path(str(payload["root"])).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return PTBXLManifestV3(
        path=manifest_path,
        root=root,
        records=records,
        split=split,
        manifest_sha256=str(expected_manifest_hash),
        split_sha256=str(payload["split_sha256"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest_files(
    manifest: PTBXLManifestV3, record_ids: Sequence[str], *, rate: int
) -> None:
    for record_id in record_ids:
        record = manifest.records[record_id]
        files = record.get("files", {})
        entry = files.get(str(rate)) if isinstance(files, Mapping) else None
        if not isinstance(entry, Mapping):
            raise ValueError(f"record {record_id} has no frozen {rate} Hz files")
        required = {"record", "header_sha256", "signal_sha256", "signal_size_bytes"}
        if required - set(entry):
            raise ValueError(f"record {record_id} has incomplete {rate} Hz provenance")
        stem = manifest.root / str(entry["record"])
        header = stem.with_suffix(".hea")
        signal = stem.with_suffix(".dat")
        if not header.is_file() or _file_sha256(header) != entry["header_sha256"]:
            raise ValueError(f"record {record_id} header hash mismatch")
        if (
            not signal.is_file()
            or signal.stat().st_size != int(entry["signal_size_bytes"])
            or _file_sha256(signal) != entry["signal_sha256"]
        ):
            raise ValueError(f"record {record_id} signal hash mismatch")


def _validate_database_identity(
    db: PTBXL, manifest: PTBXLManifestV3, record_ids: Sequence[str]
) -> None:
    for record_id in record_ids:
        numeric_id = int(record_id)
        if numeric_id not in db.meta.index:
            raise ValueError(f"manifest record {record_id} is absent from PTB-XL metadata")
        expected_patient = str(manifest.records[record_id]["patient_id"])
        if db.patient_id(numeric_id) != expected_patient:
            raise ValueError(f"manifest patient mismatch for record {record_id}")


def _materialize_training_manifest(
    db: PTBXL,
    contract: PTBXLManifestV3,
    record_ids: Sequence[str],
    *,
    rate: int,
    work_dir: Path,
    segments: Sequence[str],
    delineator: str,
    configurations: Sequence[Sequence[str]],
) -> tuple[TrainManifest, pd.DataFrame, dict[str, Any]]:
    if not record_ids:
        raise ValueError("training split is empty")
    full_path = work_dir / "train_signals.full.npy"
    path = work_dir / "train_signals.npy"
    signals = None
    common_shape: tuple[int, int] | None = None
    included_records: list[str] = []
    trail = AuditTrail()
    predictor_accumulator = TrainingPredictorAccumulator(segments)
    for record_id in record_ids:
        patient_id = str(contract.records[record_id]["patient_id"])
        base_audit = None
        try:
            signal_value, base_audit = db.signal_with_audit(int(record_id), rate=rate)
            signal = np.asarray(signal_value, dtype=np.float32)
            if signal.ndim != 2 or signal.shape[1] != len(CANONICAL_LEADS):
                raise ValueError(f"unexpected PTB-XL signal shape {signal.shape}")
            if not np.isfinite(signal).all():
                raise ValueError("signal contains non-finite samples")
            if common_shape is None:
                common_shape = signal.shape
                signals = np.lib.format.open_memmap(
                    full_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(record_ids), len(CANONICAL_LEADS), signal.shape[0]),
                )
            if signal.shape != common_shape:
                raise ValueError(
                    f"signal shape {signal.shape} differs from common shape {common_shape}"
                )
            segment_indices = PTBXL.segment_indices(signal, fs=rate, method=delineator)
            selected = {
                segment: np.asarray(segment_indices.get(segment, ()), dtype=np.int64)
                for segment in segments
            }
            counts = {segment: int(indices.size) for segment, indices in selected.items()}
            assert signals is not None
            signals[len(included_records)] = signal.T
            included_records.append(record_id)
            trail.append(
                SignalAudit(**{**base_audit.__dict__, "segment_counts": counts})
            )
        except Exception as exc:
            values = {
                "cohort": "PTB-XL",
                "record_id": str(record_id),
                "patient_id": patient_id,
                "requested_rate_hz": rate,
            }
            if base_audit is not None:
                values.update(
                    {
                        "source_rate_hz": base_audit.source_rate_hz,
                        "n_samples": base_audit.n_samples,
                        "input_leads": base_audit.input_leads,
                        "input_units": base_audit.input_units,
                        "unit_scales_to_mv": base_audit.unit_scales_to_mv,
                    }
                )
            trail.append(
                SignalAudit(
                    **values,
                    status="excluded",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        predictor_accumulator.update(
            EvaluationRecord(
                patient_id=patient_id,
                record_id=record_id,
                signal=signal.T,
                segment_indices=selected,
            )
        )
    if signals is None or not included_records:
        raise RuntimeError("no PTB-XL training records passed the signal contract")
    signals.flush()
    del signals
    if len(included_records) == len(record_ids):
        full_path.replace(path)
    else:
        source = np.load(full_path, mmap_mode="r")
        compact = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(included_records), *source.shape[1:]),
        )
        compact[:] = source[: len(included_records)]
        compact.flush()
        del compact, source
        full_path.unlink()
    patient_ids = sorted(
        {str(contract.records[value]["patient_id"]) for value in included_records}
    )
    train_manifest = TrainManifest(
        dataset="PTB-XL",
        split="folds1-7/train",
        signals_path=str(path),
        signals_sha256=sha256_file(path),
        split_sha256=contract.split_sha256,
        patient_ids_sha256=lineage.canonical_sha256(patient_ids),
        rate_hz=rate,
        normalization="raw_mV",
    )
    return train_manifest, predictor_accumulator.finalize(configurations), trail.to_dict()


def _load_evaluation_records(
    db: PTBXL,
    contract: PTBXLManifestV3,
    record_ids: Sequence[str],
    *,
    rate: int,
    segments: Sequence[str],
    delineator: str,
    partition: str,
) -> tuple[list[EvaluationRecord], dict[str, Any]]:
    records = []
    trail = AuditTrail()
    for record_id in record_ids:
        patient_id = str(contract.records[record_id]["patient_id"])
        base_audit = None
        try:
            signal_value, base_audit = db.signal_with_audit(int(record_id), rate=rate)
            signal = np.asarray(signal_value, dtype=np.float32).T
            if signal.ndim != 2 or signal.shape[0] != len(CANONICAL_LEADS):
                raise ValueError(f"unexpected canonical signal shape {signal.shape}")
            if not np.isfinite(signal).all():
                raise ValueError("signal contains non-finite samples")
            indices = PTBXL.segment_indices(signal.T, fs=rate, method=delineator)
            selected = {
                segment: np.asarray(indices.get(segment, ()), dtype=np.int64)
                for segment in segments
            }
            counts = {segment: int(value.size) for segment, value in selected.items()}
            if not any(counts.values()):
                raise ValueError("no requested delineated windows")
            record = EvaluationRecord(
                patient_id=patient_id,
                record_id=record_id,
                signal=signal,
                segment_indices=selected,
            )
            record.validate()
            records.append(record)
            trail.append(
                SignalAudit(**{**base_audit.__dict__, "segment_counts": counts})
            )
        except Exception as exc:
            values = {
                "cohort": "PTB-XL",
                "record_id": str(record_id),
                "patient_id": patient_id,
                "requested_rate_hz": rate,
            }
            if base_audit is not None:
                values.update(
                    {
                        "source_rate_hz": base_audit.source_rate_hz,
                        "n_samples": base_audit.n_samples,
                        "input_leads": base_audit.input_leads,
                        "input_units": base_audit.input_units,
                        "unit_scales_to_mv": base_audit.unit_scales_to_mv,
                    }
                )
            trail.append(
                SignalAudit(
                    **values,
                    status="excluded",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
    if not records:
        raise RuntimeError(f"no {partition} records have evaluable requested windows")
    summary = trail.summary_without_hash()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "role": partition,
        "n_requested": len(record_ids),
        "n_included": summary["n_included"],
        "n_excluded": summary["n_excluded"],
        "exclusion_reasons": summary["exclusion_reasons"],
        "audit_sha256": trail.sha256(),
        "records": trail.to_dict()["records"],
    }
    return records, audit


def _score_partitions(
    model: Any,
    evaluation_records: Mapping[str, Sequence[EvaluationRecord]],
    *,
    configuration: Sequence[str],
    method: str,
    model_seed: int,
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> list[pd.DataFrame]:
    required = ("tune", "calibration", "test")
    missing = set(required) - set(evaluation_records)
    if missing:
        raise ValueError(f"evaluation partitions are missing: {sorted(missing)}")
    return [
        evaluate_reconstructor(
            model,
            evaluation_records[partition],
            configuration=configuration,
            method=method,
            model_seed=model_seed,
            segments=segments,
            training_predictors=training_predictors,
            cohort="PTB-XL",
            partition=partition,
        )
        for partition in required
    ]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return lineage.artifact_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = [
        (item.relative_to(path).as_posix(), lineage.artifact_sha256(item))
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]
    if not entries:
        raise ValueError(f"artifact directory is empty: {path}")
    return lineage.canonical_sha256(entries)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_tuning(arguments: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if arguments.tuning_config is None:
        return dict(DEFAULT_TUNING[arguments.method]), "nonrelease_defaults"
    path = arguments.tuning_config.resolve()
    payload = _load_json_object(path, "tuning config")
    if arguments.release:
        if payload.get("schema_version") != "reconstruction-tuning-v3":
            raise ValueError("release tuning config must use reconstruction-tuning-v3")
        if payload.get("source_role") != "fold8/tune":
            raise ValueError("release tuning config must declare source_role=fold8/tune")
    methods = payload.get("methods", payload)
    value = methods.get(arguments.method) if isinstance(methods, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"tuning config has no object for method {arguments.method}")
    if arguments.release and "max_records" in value:
        raise ValueError("release tuning config cannot subsample with max_records")
    return dict(value), lineage.artifact_sha256(path)


def _resolve_official_source(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if arguments.method not in OFFICIAL_METHODS:
        raise ValueError("official source requested for a native method")
    if arguments.upstreams is None or not arguments.upstreams.is_dir():
        raise FileNotFoundError(
            "official method requires --upstreams with pinned checkouts; no surrogate is allowed"
        )
    if arguments.official_config is None or not arguments.official_config.is_file():
        raise FileNotFoundError(
            "official method requires --official-config with data/train/inference integration"
        )
    config = _load_json_object(arguments.official_config.resolve(), "official config")
    spec = IMPUTE_ECG if arguments.method == "imputeecg" else ECG_RECOVER
    default_source = arguments.upstreams.resolve() / f"{spec.name}-{spec.commit[:12]}"
    source = Path(str(config.get("source_dir", default_source))).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    return source, config


def _workspace_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _streaming_training_moments(
    train_manifest: TrainManifest,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute stable full-data mean/scatter with one manifest hash verification."""

    train_manifest.validate()
    signals = np.load(train_manifest.signals_path, mmap_mode="r")
    if signals.ndim != 3 or 12 not in signals.shape[1:]:
        raise ValueError(f"training signals must have one twelve-lead axis: {signals.shape}")
    mean = np.zeros(12, dtype=np.float64)
    scatter = np.zeros((12, 12), dtype=np.float64)
    count = 0
    for raw_record in signals:
        record = np.asarray(raw_record, dtype=np.float64)
        samples = record.T if record.shape[0] == 12 else record
        if samples.ndim != 2 or samples.shape[1] != 12 or not np.isfinite(samples).all():
            raise ValueError("training record violates the finite twelve-lead contract")
        block_count = samples.shape[0]
        block_mean = samples.mean(axis=0)
        centered = samples - block_mean
        block_scatter = centered.T @ centered
        if count == 0:
            mean = block_mean
            scatter = block_scatter
            count = block_count
            continue
        combined = count + block_count
        delta = block_mean - mean
        scatter += block_scatter + np.outer(delta, delta) * count * block_count / combined
        mean += delta * block_count / combined
        count = combined
    if count < 2:
        raise ValueError("at least two training time samples are required")
    scatter = (scatter + scatter.T) / 2.0
    return mean, scatter, count


def _replace_tokens(values: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    out = []
    for raw in values:
        token = str(raw)
        for marker, replacement in replacements.items():
            token = token.replace("{" + marker + "}", replacement)
        out.append(token)
    return out


def _fit_and_score_native_linear(
    method: str,
    train_manifest: TrainManifest,
    evaluation_records: Mapping[str, Sequence[EvaluationRecord]],
    configurations: Sequence[Sequence[str]],
    *,
    tuning: Mapping[str, Any],
    output_dir: Path,
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], str]:
    mean, scatter, sample_count = _streaming_training_moments(train_manifest)
    covariance = None
    ridge_lambda = None
    if method == "lowrank":
        rank = int(tuning.get("rank", 3))
        noise_variance = float(tuning.get("noise_variance", 1e-6))
        if not 1 <= rank <= 12 or not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("lowrank tuning requires rank in [1,12] and positive noise_variance")
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        order = np.argsort(eigenvalues)[::-1][:rank]
        basis = eigenvectors[:, order]
        coordinate_variance = np.maximum(eigenvalues[order], 0.0) / (sample_count - 1)
        covariance = (basis * coordinate_variance[None, :]) @ basis.T
        covariance += noise_variance * np.eye(12)
    else:
        ridge_lambda = float(tuning.get("ridge_lambda", 1e-3))
        if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
            raise ValueError("ridge_lambda must be finite and non-negative")
    frames = []
    checkpoints = []
    for config_index, configuration in enumerate(configurations):
        model_dir = output_dir / "models" / "seed-0" / f"config-{config_index:03d}"
        model_dir.mkdir(parents=True, exist_ok=True)
        observed = np.asarray(
            [CANONICAL_LEADS.index(lead) for lead in configuration], dtype=np.int64
        )
        if method == "lowrank":
            assert covariance is not None
            model = LowRankConditionalMeanReconstructor()
            model.mean = mean
            model.covariance = covariance
            model.observed = observed
            checkpoint = model_dir / "low_rank.npz"
            np.savez(
                checkpoint,
                mean=mean,
                covariance=covariance,
                observed=observed,
                rank=np.asarray([rank]),
                noise_variance=np.asarray([noise_variance]),
            )
        else:
            assert ridge_lambda is not None
            gram = scatter[np.ix_(observed, observed)] + ridge_lambda * np.eye(
                observed.size
            )
            weights = np.linalg.solve(gram, scatter[observed]).T
            model = RidgeLeadReconstructor()
            model.x_mean = mean[observed]
            model.y_mean = mean
            model.weights = weights
            model.observed = observed
            checkpoint = model_dir / "ridge.npz"
            np.savez(
                checkpoint,
                weights=weights,
                x_mean=model.x_mean,
                y_mean=mean,
                observed=observed,
                ridge_lambda=np.asarray([ridge_lambda]),
            )
        model._checkpoint_path = checkpoint
        model._fitted = True
        checkpoints.append(
            checkpoint_descriptor(
                checkpoint,
                output_dir,
                seed=0,
                configuration=list(configuration),
            )
        )
        frames.extend(
            _score_partitions(
                model,
                evaluation_records,
                configuration=configuration,
                method=method,
                model_seed=0,
                segments=segments,
                training_predictors=training_predictors,
            )
        )
    adapter = (
        "ecgcert.estimators.LowRankConditionalMeanReconstructor"
        if method == "lowrank"
        else "ecgcert.estimators.RidgeLeadReconstructor"
    )
    return frames, checkpoints, adapter


def _fit_and_score_masked_unet(
    train_manifest: TrainManifest,
    evaluation_records: Mapping[str, Sequence[EvaluationRecord]],
    configurations: Sequence[Sequence[str]],
    *,
    seeds: Sequence[int],
    tuning: Mapping[str, Any],
    output_dir: Path,
    segments: Sequence[str],
    device: str,
    release: bool,
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], str]:
    frames = []
    checkpoints = []
    parameters = dict(tuning)
    n_train_records = int(np.load(train_manifest.signals_path, mmap_mode="r").shape[0])
    if release:
        normalization_records = int(parameters.get("normalization_records", n_train_records))
        if normalization_records != n_train_records:
            raise ValueError("release masked-unet normalization must use every training record")
        parameters["max_records"] = n_train_records
        parameters["normalization_records"] = n_train_records
    for seed in seeds:
        model_dir = output_dir / "models" / f"seed-{seed}"
        model = MaskedUNetReconstructor().fit(
            train_manifest,
            ReconstructorConfig(
                observed_leads=tuple(configurations[0]),
                seed=int(seed),
                output_dir=str(model_dir),
                device=device,
                parameters=parameters,
            ),
        )
        checkpoint = model_dir / "masked_unet.pt"
        checkpoints.append(checkpoint_descriptor(checkpoint, output_dir, seed=int(seed)))
        for configuration in configurations:
            frames.extend(
                _score_partitions(
                    model,
                    evaluation_records,
                    configuration=configuration,
                    method="masked-unet",
                    model_seed=int(seed),
                    segments=segments,
                    training_predictors=training_predictors,
                )
            )
    return frames, checkpoints, "ecgcert.estimators.MaskedUNetReconstructor"


def _fit_and_score_imputeecg(
    train_manifest: TrainManifest,
    evaluation_records: Mapping[str, Sequence[EvaluationRecord]],
    configurations: Sequence[Sequence[str]],
    *,
    seeds: Sequence[int],
    tuning: Mapping[str, Any],
    official_config: Mapping[str, Any],
    source_dir: Path,
    output_dir: Path,
    segments: Sequence[str],
    device: str,
    release: bool,
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], str, dict[str, Any]]:
    official_data = official_config.get("official_data_path")
    if not official_data:
        raise ValueError("ImputeECG official config requires official_data_path")
    frames = []
    checkpoints = []
    if release and official_config.get("split_sha256") != train_manifest.split_sha256:
        raise ValueError("ImputeECG official data split_sha256 does not match PTB-XL manifest")
    expected_records = int(np.load(train_manifest.signals_path, mmap_mode="r").shape[0])
    for seed in seeds:
        model_dir = output_dir / "models" / f"seed-{seed}"
        parameters = {
            **dict(tuning),
            **dict(official_config.get("parameters", {})),
            "official_data_path": str(official_data).replace("{seed}", str(seed)),
        }
        official_root = Path(parameters["official_data_path"]).resolve()
        ground_truth = official_root / "train_data_gt.npy"
        masks = official_root / "train_data_mask.npy"
        if not ground_truth.is_file() or not masks.is_file():
            raise FileNotFoundError(
                f"ImputeECG official arrays are incomplete under {official_root}"
            )
        gt_shape = np.load(ground_truth, mmap_mode="r").shape
        mask_shape = np.load(masks, mmap_mode="r").shape
        if (
            gt_shape != mask_shape
            or len(gt_shape) != 3
            or gt_shape[0] != expected_records
            or gt_shape[1:] not in {(5000, 12), (12, 5000)}
        ):
            raise ValueError(
                "ImputeECG official arrays must align exactly with the selected training records"
            )
        config = ReconstructorConfig(
            observed_leads=tuple(configurations[0]),
            seed=int(seed),
            output_dir=str(model_dir),
            device=device,
            parameters=parameters,
        )
        model = ImputeECGReconstructor(source_dir)
        train_command = model.build_train_command(train_manifest, config)
        model.fit(train_manifest, config)
        checkpoint = model._checkpoint_path
        if checkpoint is None:
            raise RuntimeError("official ImputeECG produced no checkpoint path")
        checkpoints.append(
            checkpoint_descriptor(
                checkpoint,
                output_dir,
                seed=int(seed),
                train_command=train_command,
            )
        )
        for configuration in configurations:
            frames.extend(
                _score_partitions(
                    model,
                    evaluation_records,
                    configuration=configuration,
                    method="imputeecg",
                    model_seed=int(seed),
                    segments=segments,
                    training_predictors=training_predictors,
                )
            )
    official = {
        "name": IMPUTE_ECG.name,
        "repository": IMPUTE_ECG.repository,
        "commit": IMPUTE_ECG.commit,
        "source_dir": _workspace_relative(source_dir),
        "inference": "in-process pinned ImputeECGReconstructor.load",
    }
    return frames, checkpoints, "ecgcert.estimators.ImputeECGReconstructor", official


def _fit_and_score_ecgrecover(
    train_manifest: TrainManifest,
    evaluation_records: Mapping[str, Sequence[EvaluationRecord]],
    *,
    seeds: Sequence[int],
    official_config: Mapping[str, Any],
    source_dir: Path,
    output_dir: Path,
    segments: Sequence[str],
    release: bool,
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], str, dict[str, Any], tuple[str, ...]]:
    input_lead = str(official_config.get("input_lead", ""))
    if input_lead not in INDEPENDENT_LEADS:
        raise ValueError("ECGrecover input_lead must be one independent canonical lead")
    train_template = official_config.get("official_train_command")
    bridge_template = official_config.get("official_inference_bridge")
    checkpoint_template = official_config.get("checkpoint")
    if not isinstance(train_template, list) or not train_template:
        raise ValueError("ECGrecover requires official_train_command argv")
    if not isinstance(bridge_template, list) or not bridge_template:
        raise ValueError("ECGrecover requires official_inference_bridge argv")
    if not checkpoint_template:
        raise ValueError("ECGrecover requires the official checkpoint output path")
    if release and official_config.get("split_sha256") != train_manifest.split_sha256:
        raise ValueError("ECGrecover official config split_sha256 does not match PTB-XL manifest")
    frames = []
    checkpoints = []
    for seed in seeds:
        model_dir = (output_dir / "models" / f"seed-{seed}").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        replacements = {
            "seed": str(seed),
            "output_dir": str(model_dir),
            "source_dir": str(source_dir),
        }
        train_command = _replace_tokens(train_template, replacements)
        bridge = [str(token) for token in bridge_template]
        if any(
            marker in "\n".join(bridge)
            for marker in ("{seed}", "{output_dir}", "{source_dir}")
        ):
            raise ValueError(
                "ECGrecover inference bridge must be portable and use only "
                "{input}/{output}/{checkpoint} placeholders"
            )
        checkpoint = Path(
            _replace_tokens([str(checkpoint_template)], replacements)[0]
        ).resolve()
        adapter = ECGrecoverReconstructor(source_dir)
        adapter.fit(
            train_manifest,
            ReconstructorConfig(
                observed_leads=(input_lead,),
                seed=int(seed),
                output_dir=str(model_dir),
                parameters={
                    "official_train_command": train_command,
                    "official_inference_bridge": bridge,
                    "checkpoint": str(checkpoint),
                },
            ),
        )
        checkpoints.append(
            checkpoint_descriptor(
                checkpoint,
                output_dir,
                seed=int(seed),
                configuration=[input_lead],
                inference_bridge=bridge,
                train_command=train_command,
            )
        )
        bridge_model = OfficialCommandBridgeReconstructor(
            command=bridge,
            checkpoint=checkpoint,
            source_dir=source_dir,
            single_input_only=True,
        )
        frames.extend(
            _score_partitions(
                bridge_model,
                evaluation_records,
                configuration=(input_lead,),
                method="ecgrecover",
                model_seed=int(seed),
                segments=segments,
                training_predictors=training_predictors,
            )
        )
    official = {
        "name": ECG_RECOVER.name,
        "repository": ECG_RECOVER.repository,
        "commit": ECG_RECOVER.commit,
        "source_dir": _workspace_relative(source_dir),
        "input_lead": input_lead,
        "inference_bridge": list(bridge_template),
        "expected_input_schema": OfficialCommandBridgeReconstructor.expected_input_schema,
        "expected_output_schema": OfficialCommandBridgeReconstructor.expected_output_schema,
        "scope": "official single-input task only; not part of multi-input parity claims",
    }
    return (
        frames,
        checkpoints,
        "ecgcert.reconstruction.OfficialCommandBridgeReconstructor",
        official,
        (input_lead,),
    )


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "reconstruction benchmark requires locked pyarrow; CSV fallback is forbidden"
        ) from exc


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    validate_release_arguments(arguments)
    seeds = resolve_model_seeds(arguments)
    manifest = load_ptbxl_manifest(arguments.manifest)
    rank_maps = arguments.rank_maps.resolve()
    rank_maps_sha256 = _path_sha256(rank_maps)
    tuning, tuning_source = _load_tuning(arguments)
    source_dir = None
    official_config: dict[str, Any] = {}
    if arguments.method in OFFICIAL_METHODS:
        source_dir, official_config = _resolve_official_source(arguments)
    _require_parquet_engine()

    train_ids = manifest.record_ids("train", arguments.max_records)
    partition_ids = {
        role: manifest.record_ids(role, arguments.max_records)
        for role in ("tune", "calibration", "test")
    }
    verification_ids = (
        tuple(manifest.records)
        if arguments.release
        else (train_ids + tuple(value for ids in partition_ids.values() for value in ids))
    )
    _verify_manifest_files(manifest, verification_ids, rate=arguments.rate)
    db = PTBXL(manifest.root)
    _validate_database_identity(db, manifest, verification_ids)

    configurations: tuple[tuple[str, ...], ...] = deep_configuration_panel()
    if arguments.max_configurations is not None:
        configurations = configurations[: arguments.max_configurations]
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_records = {}
    evaluation_audit = {
        "schema_version": SCHEMA_VERSION,
        "partitions": {},
    }
    case_sha256 = {}
    partition_labels = {
        "tune": "fold8/tune",
        "calibration": "fold9/calibration",
        "test": "fold10/test",
    }
    for partition, record_ids in partition_ids.items():
        records, audit = _load_evaluation_records(
            db,
            manifest,
            record_ids,
            rate=arguments.rate,
            segments=arguments.segments,
            delineator=arguments.delineator,
            partition=partition_labels[partition],
        )
        evaluation_records[partition] = records
        evaluation_audit["partitions"][partition] = audit
        case_sha256[partition] = evaluation_records_sha256(
            records, segments=arguments.segments
        )
    with TemporaryDirectory(prefix=".ecgcert-train-", dir=output_dir) as temporary:
        train_manifest, training_predictors, training_audit = _materialize_training_manifest(
            db,
            manifest,
            train_ids,
            rate=arguments.rate,
            work_dir=Path(temporary),
            segments=arguments.segments,
            delineator=arguments.delineator,
            configurations=configurations,
        )
        predictor_lookup = training_predictor_lookup(training_predictors)
        training_predictor_path = output_dir / TRAINING_PREDICTORS_FILENAME
        training_predictors.to_parquet(
            training_predictor_path, index=False, compression="zstd"
        )
        official_metadata: dict[str, Any] | None = None
        if arguments.method in {"lowrank", "ridge"}:
            frames, checkpoints, adapter_class = _fit_and_score_native_linear(
                arguments.method,
                train_manifest,
                evaluation_records,
                configurations,
                tuning=tuning,
                output_dir=output_dir,
                segments=arguments.segments,
                training_predictors=predictor_lookup,
            )
        elif arguments.method == "masked-unet":
            frames, checkpoints, adapter_class = _fit_and_score_masked_unet(
                train_manifest,
                evaluation_records,
                configurations,
                seeds=seeds,
                tuning=tuning,
                output_dir=output_dir,
                segments=arguments.segments,
                device=arguments.device,
                release=arguments.release,
                training_predictors=predictor_lookup,
            )
        elif arguments.method == "imputeecg":
            assert source_dir is not None
            frames, checkpoints, adapter_class, official_metadata = _fit_and_score_imputeecg(
                train_manifest,
                evaluation_records,
                configurations,
                seeds=seeds,
                tuning=tuning,
                official_config=official_config,
                source_dir=source_dir,
                output_dir=output_dir,
                segments=arguments.segments,
                device=arguments.device,
                release=arguments.release,
                training_predictors=predictor_lookup,
            )
        else:
            assert source_dir is not None
            (
                frames,
                checkpoints,
                adapter_class,
                official_metadata,
                ecgrecover_configuration,
            ) = _fit_and_score_ecgrecover(
                train_manifest,
                evaluation_records,
                seeds=seeds,
                official_config=official_config,
                source_dir=source_dir,
                output_dir=output_dir,
                segments=arguments.segments,
                release=arguments.release,
                training_predictors=predictor_lookup,
            )
            configurations = (ecgrecover_configuration,)
        training_signal_sha256 = train_manifest.signals_sha256

    evaluation_audit["training"] = training_audit
    audit_path = output_dir / "evaluation_audit.json"
    _atomic_json(audit_path, evaluation_audit)

    training_config = {
        "cohort": "PTB-XL",
        "train_role": "folds1-7",
        "evaluation_roles": partition_labels,
        "rate_hz": arguments.rate,
        "segments": list(arguments.segments),
        "delineator": arguments.delineator,
        "signal_unit": "raw_mV",
        "mask": "whole-lead; identical across methods",
        "simple_predictors": {
            "source": "folds1-7/train only",
            "target_rms": "sqrt pooled segment-target mean square in mV",
            "max_target_observed_correlation": (
                "maximum absolute pooled segment correlation over observed leads"
            ),
            "heldout_target_statistics_used": False,
        },
        "configuration_panel_sha256": configuration_panel_sha256(configurations),
        "n_configurations": len(configurations),
        "n_train_records": len(train_ids),
        "n_train_records_included": training_audit["summary"]["n_included"],
        "n_train_records_excluded": training_audit["summary"]["n_excluded"],
        "n_evaluation_records_requested": {
            partition: len(record_ids) for partition, record_ids in partition_ids.items()
        },
        "train_signals_sha256": training_signal_sha256,
        "model_seeds": list(seeds),
        "release": bool(arguments.release),
        "subsampled": arguments.max_records is not None
        or arguments.max_configurations is not None,
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "method": arguments.method,
        "adapter_class": adapter_class,
        "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
        "models": checkpoints,
        "training_config": training_config,
        "tuning_config": dict(tuning),
        "tuning_source": tuning_source,
        "training_predictors": {
            "path": TRAINING_PREDICTORS_FILENAME,
            "sha256": lineage.artifact_sha256(training_predictor_path),
            "source_partition": "PTB-XL/folds1-7/train",
        },
    }
    if official_metadata is not None:
        official_metadata["integration_config_sha256"] = lineage.artifact_sha256(
            arguments.official_config.resolve()
        )
        bundle["official"] = official_metadata
    bundle_path = write_bundle_metadata(output_dir, bundle)

    summary = {
        "method": arguments.method,
        "method_scope": (
            "official single-input independent baseline"
            if arguments.method == "ecgrecover"
            else "shared frozen configuration panel"
        ),
        "adapter_class": adapter_class,
        "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
        "checkpoints": checkpoints,
        "training_config": training_config,
        "tuning_config": dict(tuning),
        "tuning_source": tuning_source,
        "manifest": {
            "path": str(arguments.manifest),
            "sha256": manifest.manifest_sha256,
            "split_sha256": manifest.split_sha256,
        },
        "rank_maps_sha256": rank_maps_sha256,
        "evaluation_records_sha256": case_sha256,
        "evaluation_contract_sha256": lineage.canonical_sha256(case_sha256),
        "official": official_metadata,
        "artifacts": {
            "bundle": {
                "path": bundle_path.relative_to(output_dir).as_posix(),
                "sha256": lineage.artifact_sha256(bundle_path),
            },
            "evaluation_audit": {
                "path": audit_path.relative_to(output_dir).as_posix(),
                "sha256": lineage.artifact_sha256(audit_path),
            },
            "training_predictors": {
                "path": TRAINING_PREDICTORS_FILENAME,
                "sha256": lineage.artifact_sha256(training_predictor_path),
            },
        },
    }
    metrics = pd.concat(frames, ignore_index=True)
    return write_benchmark_artifacts(metrics, output_dir, summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = run(arguments)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"reconstruction benchmark failed closed: {exc}") from exc
    print(
        f"[{summary['method']}] {summary['n_patient_metric_rows']} patient metric rows -> "
        f"{arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
