"""Shared reconstruction evaluation and reloadable model-bundle contracts.

The benchmark entry point and external zero-transfer validation both use this
module.  Ground-truth missing leads never cross an official command bridge, and
every prediction is rejected unless the observed samples are preserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.api import normalize_observed_mask


SCHEMA_VERSION = "reconstruction-benchmark-v3"
METRIC_FILENAME = "patient_metrics.parquet"
SUMMARY_FILENAME = "summary.v3.json"
BUNDLE_FILENAME = "bundle.v3.json"
TRAINING_PREDICTORS_FILENAME = "training_predictors.parquet"
LOG_FLOOR = 1e-12

DIMENSION_COLUMNS = (
    "cohort",
    "partition",
    "patient_id",
    "segment",
    "configuration",
    "target",
    "method",
    "model_seed",
)
REQUIRED_METRIC_COLUMNS = (
    "schema_version",
    *DIMENSION_COLUMNS,
    "observed_leads",
    "n_observed",
    "n_records",
    "n_samples",
    "rmse_mv",
    "log_rmse_mv",
    "target_rms",
    "max_target_observed_correlation",
    "target_rms_mv",
    "normalized_rmse",
    "outcome_log_rmse",
)


class ReconstructionContractError(RuntimeError):
    """A prediction or persisted model violates the frozen benchmark contract."""


class ObservedSampleViolation(ReconstructionContractError):
    """A model changed a sample declared as observed."""


class ModelBundleError(ReconstructionContractError):
    """A persisted model bundle is missing, inconsistent, or has a bad hash."""


@dataclass(frozen=True)
class EvaluationRecord:
    """One canonical 12-lead record and its frozen evaluation windows."""

    patient_id: str
    record_id: str
    signal: np.ndarray
    segment_indices: Mapping[str, np.ndarray]

    def validate(self) -> None:
        signal = np.asarray(self.signal)
        if not self.patient_id or not self.record_id:
            raise ValueError("patient_id and record_id must be non-empty")
        if signal.ndim != 2 or signal.shape[0] != len(CANONICAL_LEADS):
            raise ValueError(f"signal must have shape (12,T), got {signal.shape}")
        if signal.shape[1] < 1 or not np.isfinite(signal).all():
            raise ValueError("signal must contain finite samples")
        for segment, raw_indices in self.segment_indices.items():
            if not isinstance(segment, str) or not segment:
                raise ValueError("segment names must be non-empty strings")
            indices = np.asarray(raw_indices)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(f"{segment} indices must be a one-dimensional integer array")
            if indices.size and (indices.min() < 0 or indices.max() >= signal.shape[1]):
                raise ValueError(f"{segment} indices leave record bounds")
            if indices.size != np.unique(indices).size:
                raise ValueError(f"{segment} indices contain duplicates")


class TrainingPredictorAccumulator:
    """Accumulate folds-1--7-only segment moments without retaining signals."""

    def __init__(self, segments: Sequence[str]):
        if not segments or len(segments) != len(set(segments)):
            raise ValueError("training predictor segments must be non-empty and unique")
        self.segments = tuple(segments)
        self._moments = {
            segment: {
                "count": 0,
                "sum": np.zeros(12, dtype=np.float64),
                "cross": np.zeros((12, 12), dtype=np.float64),
            }
            for segment in self.segments
        }

    def update(self, record: EvaluationRecord) -> None:
        record.validate()
        signal = np.asarray(record.signal, dtype=np.float64)
        for segment in self.segments:
            indices = np.asarray(record.segment_indices.get(segment, ()), dtype=np.int64)
            if not indices.size:
                continue
            samples = signal[:, indices].T
            moment = self._moments[segment]
            moment["count"] += int(samples.shape[0])
            moment["sum"] += samples.sum(axis=0)
            moment["cross"] += samples.T @ samples

    def finalize(self, configurations: Sequence[Sequence[str]]) -> pd.DataFrame:
        rows = []
        for segment in self.segments:
            moment = self._moments[segment]
            count = int(moment["count"])
            if count < 2:
                raise ReconstructionContractError(
                    f"fewer than two folds-1--7 samples for segment {segment}"
                )
            mean = moment["sum"] / count
            second = moment["cross"] / count
            covariance = (second - np.outer(mean, mean))
            covariance = (covariance + covariance.T) / 2.0
            variance = np.diag(covariance)
            target_rms = np.sqrt(np.maximum(np.diag(second), 0.0))
            if np.any(variance <= 0) or np.any(target_rms <= 0):
                raise ReconstructionContractError(
                    f"non-positive folds-1--7 scale/variance for segment {segment}"
                )
            correlation = covariance / np.sqrt(np.outer(variance, variance))
            correlation = np.clip(correlation, -1.0, 1.0)
            for configuration in configurations:
                observed = tuple(configuration)
                observed_indices = [CANONICAL_LEADS.index(lead) for lead in observed]
                configuration_id = "+".join(observed)
                for target_index, target in enumerate(CANONICAL_LEADS):
                    max_correlation = float(
                        np.max(np.abs(correlation[target_index, observed_indices]))
                    )
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "source_partition": "PTB-XL/folds1-7/train",
                            "segment": segment,
                            "configuration": configuration_id,
                            "target": target,
                            "target_rms": float(target_rms[target_index]),
                            "max_target_observed_correlation": max_correlation,
                            "n_training_samples": count,
                        }
                    )
        return pd.DataFrame(rows)


def training_predictor_lookup(
    predictors: pd.DataFrame | Mapping[tuple[str, str, str], tuple[float, float]],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Validate and index immutable folds-1--7 simple predictors."""

    if not isinstance(predictors, pd.DataFrame):
        lookup = dict(predictors)
    else:
        required = {
            "source_partition",
            "segment",
            "configuration",
            "target",
            "target_rms",
            "max_target_observed_correlation",
        }
        missing = required - set(predictors.columns)
        if missing:
            raise ValueError(f"training predictor table lacks columns: {sorted(missing)}")
        if set(predictors["source_partition"]) != {"PTB-XL/folds1-7/train"}:
            raise ValueError("simple predictors must come only from PTB-XL folds 1--7")
        if predictors.duplicated(["segment", "configuration", "target"]).any():
            raise ValueError("training predictor table contains duplicate cells")
        lookup = {
            (str(row.segment), str(row.configuration), str(row.target)): (
                float(row.target_rms),
                float(row.max_target_observed_correlation),
            )
            for row in predictors.itertuples(index=False)
        }
    if not lookup:
        raise ValueError("training predictor lookup is empty")
    values = np.asarray(list(lookup.values()), dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 0] <= 0):
        raise ValueError("training predictors contain invalid values")
    if np.any((values[:, 1] < 0) | (values[:, 1] > 1)):
        raise ValueError("training correlations must lie in [0,1]")
    return lookup


class ConfigurationReconstructorBank:
    """Dispatch configuration-specific linear models through one reloadable API."""

    def __init__(self, models: Mapping[tuple[str, ...], Any], *, method: str):
        if not models:
            raise ValueError("a configuration bank requires at least one model")
        self.models = dict(models)
        self.method = method

    def reconstruct(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        mask = normalize_observed_mask(observed_mask, np.asarray(signal).shape)
        if not np.all(mask == mask[:, :1]):
            raise ValueError("configuration banks require whole-lead masks")
        configuration = tuple(
            lead for index, lead in enumerate(CANONICAL_LEADS) if bool(mask[index, 0])
        )
        try:
            model = self.models[configuration]
        except KeyError as exc:
            raise ModelBundleError(
                f"{self.method} bundle has no model for configuration {configuration}"
            ) from exc
        return model.reconstruct(signal, mask)


class OfficialCommandBridgeReconstructor:
    """Run a pinned official inference bridge without exposing missing truth.

    Input ``.npz`` schema:

    - ``observed_signal``: float64 ``(12,T)``, zero at every missing position;
    - ``observed_mask``: bool ``(12,T)``;
    - ``lead_order``: Unicode canonical lead names.

    Output ``.npz`` must contain only a finite ``reconstruction`` array with
    shape ``(12,T)``.  Commands may use ``{input}``, ``{output}``, and
    ``{checkpoint}`` placeholders.
    """

    expected_input_schema = {
        "format": "npz",
        "fields": {
            "observed_signal": "float64[12,T]; missing positions are zero",
            "observed_mask": "bool[12,T]",
            "lead_order": "unicode[12]",
        },
    }
    expected_output_schema = {
        "format": "npz",
        "fields": {"reconstruction": "finite float[12,T]"},
    }

    def __init__(
        self,
        *,
        command: Sequence[str],
        checkpoint: str | Path,
        source_dir: str | Path,
        single_input_only: bool,
    ) -> None:
        if not command or not all(isinstance(token, str) and token for token in command):
            raise ModelBundleError("official inference bridge must be a non-empty argv list")
        joined = "\n".join(command)
        if "{input}" not in joined or "{output}" not in joined:
            raise ModelBundleError("official bridge must declare {input} and {output} placeholders")
        self.command = tuple(command)
        self.checkpoint = Path(checkpoint).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.single_input_only = bool(single_input_only)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.source_dir.is_dir():
            raise FileNotFoundError(self.source_dir)

    def reconstruct(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        source = np.asarray(signal, dtype=float)
        if source.ndim != 2 or source.shape[0] != len(CANONICAL_LEADS):
            raise ValueError(f"signal must have shape (12,T), got {source.shape}")
        mask = normalize_observed_mask(observed_mask, source.shape)
        if not np.all(mask == mask[:, :1]):
            raise ValueError("official bridge benchmark uses whole-lead masks")
        if self.single_input_only and int(mask[:, 0].sum()) != 1:
            raise ValueError("ECGrecover is restricted to its official single-input task")
        with TemporaryDirectory(prefix="ecgcert-official-") as temporary:
            root = Path(temporary)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            np.savez_compressed(
                input_path,
                observed_signal=np.where(mask, source, 0.0),
                observed_mask=mask,
                lead_order=np.asarray(CANONICAL_LEADS),
            )
            replacements = {
                "{input}": str(input_path),
                "{output}": str(output_path),
                "{checkpoint}": str(self.checkpoint),
            }
            argv = []
            for token in self.command:
                for marker, value in replacements.items():
                    token = token.replace(marker, value)
                argv.append(token)
            completed = subprocess.run(argv, cwd=self.source_dir, check=False)
            if completed.returncode:
                raise ReconstructionContractError(
                    f"official inference bridge failed with exit code {completed.returncode}"
                )
            if not output_path.is_file():
                raise ReconstructionContractError("official bridge produced no output npz")
            with np.load(output_path, allow_pickle=False) as payload:
                if set(payload.files) != {"reconstruction"}:
                    raise ReconstructionContractError(
                        "official bridge output must contain only 'reconstruction'"
                    )
                prediction = np.asarray(payload["reconstruction"], dtype=float)
        if prediction.shape != source.shape or not np.isfinite(prediction).all():
            raise ReconstructionContractError(
                f"official bridge returned invalid shape/values: {prediction.shape}"
            )
        if not np.array_equal(prediction[mask], source[mask]):
            raise ObservedSampleViolation("official bridge changed an observed sample")
        return prediction


def _configuration_mask(configuration: Sequence[str], length: int) -> np.ndarray:
    requested = tuple(configuration)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("configuration must contain unique observed leads")
    unknown = set(requested) - set(CANONICAL_LEADS)
    if unknown:
        raise ValueError(f"unknown observed leads: {sorted(unknown)}")
    mask = np.zeros((len(CANONICAL_LEADS), length), dtype=bool)
    for lead in requested:
        mask[CANONICAL_LEADS.index(lead)] = True
    return mask


def evaluate_reconstructor(
    reconstructor: Any,
    records: Iterable[EvaluationRecord],
    *,
    configuration: Sequence[str],
    method: str,
    model_seed: int,
    segments: Sequence[str],
    training_predictors: (
        pd.DataFrame | Mapping[tuple[str, str, str], tuple[float, float]]
    ),
    cohort: str = "PTB-XL",
    partition: str = "test",
) -> pd.DataFrame:
    """Return patient-level missing-target errors for one model/configuration.

    Squared errors and target energy are pooled only within a patient.  The
    primary meta-analysis outcome is natural-log RMSE in mV.  Normalized RMSE
    remains a secondary diagnostic because target RMS is a separate covariate.
    """

    configuration = tuple(configuration)
    if not method or not cohort or not partition:
        raise ValueError("method, cohort, and partition must be non-empty")
    if not segments or len(set(segments)) != len(tuple(segments)):
        raise ValueError("segments must be non-empty and unique")
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        record.validate()
        signal = np.asarray(record.signal, dtype=float)
        mask = _configuration_mask(configuration, signal.shape[1])
        prediction = np.asarray(reconstructor.reconstruct(signal, mask), dtype=float)
        if prediction.shape != signal.shape or not np.isfinite(prediction).all():
            raise ReconstructionContractError(
                f"{method} returned invalid shape/values for record {record.record_id}"
            )
        if not np.array_equal(prediction[mask], signal[mask]):
            raise ObservedSampleViolation(
                f"{method} changed observed samples for record {record.record_id}"
            )
        for segment in segments:
            indices = np.asarray(record.segment_indices.get(segment, ()), dtype=np.int64)
            if not indices.size:
                continue
            for target_index, target in enumerate(CANONICAL_LEADS):
                if target in configuration:
                    continue
                truth = signal[target_index, indices]
                residual = prediction[target_index, indices] - truth
                key = (str(record.patient_id), str(segment), target)
                aggregate = aggregates.setdefault(
                    key,
                    {
                        "squared_error": 0.0,
                        "target_energy": 0.0,
                        "n_samples": 0,
                        "record_ids": set(),
                    },
                )
                aggregate["squared_error"] += float(residual @ residual)
                aggregate["target_energy"] += float(truth @ truth)
                aggregate["n_samples"] += int(indices.size)
                aggregate["record_ids"].add(str(record.record_id))
    if not aggregates:
        raise ValueError("no evaluable missing-target segment samples")

    observed_text = ",".join(configuration)
    configuration_id = "+".join(configuration)
    predictor_lookup = training_predictor_lookup(training_predictors)
    rows = []
    for (patient_id, segment, target), aggregate in sorted(aggregates.items()):
        count = int(aggregate["n_samples"])
        rmse = float(np.sqrt(aggregate["squared_error"] / count))
        target_rms = float(np.sqrt(aggregate["target_energy"] / count))
        if not np.isfinite(target_rms) or target_rms <= 0:
            raise ReconstructionContractError(
                f"target RMS is not positive for patient={patient_id}, segment={segment}, "
                f"target={target}"
            )
        normalized = rmse / target_rms
        predictor_key = (segment, configuration_id, target)
        try:
            fixed_target_rms, max_correlation = predictor_lookup[predictor_key]
        except KeyError as exc:
            raise ReconstructionContractError(
                f"missing folds-1--7 predictor cell {predictor_key}"
            ) from exc
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cohort": cohort,
                "partition": partition,
                "patient_id": patient_id,
                "segment": segment,
                "configuration": configuration_id,
                "target": target,
                "method": method,
                "model_seed": int(model_seed),
                "observed_leads": observed_text,
                "n_observed": len(configuration),
                "n_records": len(aggregate["record_ids"]),
                "n_samples": count,
                "rmse_mv": rmse,
                "log_rmse_mv": float(np.log(max(rmse, LOG_FLOOR))),
                "target_rms": fixed_target_rms,
                "max_target_observed_correlation": max_correlation,
                "target_rms_mv": target_rms,
                "normalized_rmse": normalized,
                "outcome_log_rmse": float(np.log(max(rmse, LOG_FLOOR))),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_METRIC_COLUMNS)


def evaluation_records_sha256(
    records: Iterable[EvaluationRecord], *, segments: Sequence[str]
) -> str:
    """Fingerprint patient membership and exact delineated indices, not signal truth."""

    payload = []
    for record in records:
        record.validate()
        payload.append(
            {
                "patient_id": str(record.patient_id),
                "record_id": str(record.record_id),
                "segments": {
                    segment: np.asarray(record.segment_indices.get(segment, ()), dtype=int).tolist()
                    for segment in segments
                },
            }
        )
    return lineage.canonical_sha256(payload)


def checkpoint_descriptor(checkpoint: str | Path, bundle_dir: str | Path, **extra: Any) -> dict:
    """Describe a bundle-owned checkpoint by safe relative path and full SHA-256."""

    checkpoint_path = Path(checkpoint).resolve()
    bundle_path = Path(bundle_dir).resolve()
    try:
        relative = checkpoint_path.relative_to(bundle_path)
    except ValueError as exc:
        raise ModelBundleError(f"checkpoint is outside its bundle: {checkpoint_path}") from exc
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return {
        "path": relative.as_posix(),
        "sha256": lineage.artifact_sha256(checkpoint_path),
        **extra,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_bundle_metadata(bundle_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    destination = Path(bundle_dir).resolve() / BUNDLE_FILENAME
    value = dict(metadata)
    value.setdefault("schema_version", SCHEMA_VERSION)
    if value["schema_version"] != SCHEMA_VERSION:
        raise ModelBundleError("bundle metadata has the wrong schema_version")
    _atomic_json(destination, value)
    return destination


def write_benchmark_artifacts(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the mandatory Parquet and summary JSON, with no CSV fallback."""

    missing = set(REQUIRED_METRIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"metric frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("metric frame is empty")
    if frame.duplicated(list(DIMENSION_COLUMNS)).any():
        raise ValueError(
            "metric frame has duplicate patient/segment/config/target/method/seed rows"
        )
    numeric = [
        "n_observed",
        "n_records",
        "n_samples",
        "rmse_mv",
        "log_rmse_mv",
        "target_rms",
        "max_target_observed_correlation",
        "target_rms_mv",
        "normalized_rmse",
        "outcome_log_rmse",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("metric frame contains non-finite values")
    if not np.array_equal(
        frame["outcome_log_rmse"].to_numpy(dtype=float),
        frame["log_rmse_mv"].to_numpy(dtype=float),
    ):
        raise ValueError("primary outcome_log_rmse must equal patient-level log(RMSE_mV)")
    if (frame["target_rms"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("folds-1--7 target_rms must be positive")
    correlation = frame["max_target_observed_correlation"].to_numpy(dtype=float)
    if ((correlation < 0) | (correlation > 1)).any():
        raise ValueError("training-only max correlation must lie in [0,1]")
    predictor_cells = frame.groupby(["segment", "configuration", "target"], sort=False)
    if (
        predictor_cells["target_rms"].nunique().max() != 1
        or predictor_cells["max_target_observed_correlation"].nunique().max() != 1
    ):
        raise ValueError("simple predictors must be fixed across patients and partitions")
    for row in frame.itertuples(index=False):
        if row.target in str(row.observed_leads).split(","):
            raise ValueError("observed targets must not enter the missing-target metric table")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / METRIC_FILENAME
    try:
        frame.loc[:, list(REQUIRED_METRIC_COLUMNS)].to_parquet(
            metrics_path, index=False, compression="zstd"
        )
    except ImportError as exc:
        raise RuntimeError(
            "Parquet is mandatory for reconstruction artifacts; install locked pyarrow"
        ) from exc
    value = dict(summary)
    value.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "n_patient_metric_rows": int(len(frame)),
            "metric_dimensions": list(DIMENSION_COLUMNS),
            "metric_columns": list(REQUIRED_METRIC_COLUMNS),
            "observed_sample_integrity": "passed_exact_pointwise",
            "missing_targets_only": True,
            "normalization": {
                "raw_unit": "mV",
                "normalized_rmse": "RMSE_mV / patient-segment-target RMS_mV",
                "outcome_log_rmse": "natural_log(max(RMSE_mV,1e-12)); primary outcome",
                "log_rmse_mv": "natural_log(max(RMSE_mV,1e-12))",
            },
            "artifacts": {
                **dict(value.get("artifacts", {})),
                "patient_metrics": {
                    "path": METRIC_FILENAME,
                    "sha256": lineage.artifact_sha256(metrics_path),
                },
            },
        }
    )
    _atomic_json(root / SUMMARY_FILENAME, value)
    return value


def _read_bundle(bundle_dir: Path, method: str, seed: int) -> tuple[dict, list[dict]]:
    metadata_path = bundle_dir / BUNDLE_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ModelBundleError("unsupported reconstruction bundle schema")
    if metadata.get("method") != method:
        raise ModelBundleError(
            f"bundle method {metadata.get('method')!r} does not match requested {method!r}"
        )
    entries = [entry for entry in metadata.get("models", ()) if entry.get("seed") == seed]
    if not entries:
        raise ModelBundleError(f"bundle has no {method} model for seed {seed}")
    for entry in entries:
        checkpoint = _bundle_checkpoint(bundle_dir, entry)
        if lineage.artifact_sha256(checkpoint) != entry.get("sha256"):
            raise ModelBundleError(f"checkpoint SHA-256 mismatch: {entry.get('path')}")
    return metadata, entries


def _bundle_checkpoint(bundle_dir: Path, entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ModelBundleError(f"unsafe checkpoint path: {relative}")
    checkpoint = (bundle_dir / relative).resolve()
    if bundle_dir != checkpoint and bundle_dir not in checkpoint.parents:
        raise ModelBundleError(f"checkpoint escapes bundle: {relative}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _load_linear(checkpoint: Path, method: str):
    from ecgcert.estimators.baselines_v3 import (
        LowRankConditionalMeanReconstructor,
        RidgeLeadReconstructor,
    )

    with np.load(checkpoint, allow_pickle=False) as payload:
        if method == "ridge":
            required = {"weights", "x_mean", "y_mean", "observed"}
            if not required <= set(payload.files):
                missing = sorted(required - set(payload.files))
                raise ModelBundleError(f"ridge checkpoint lacks {missing}")
            model = RidgeLeadReconstructor()
            model.weights = np.asarray(payload["weights"], dtype=float)
            model.x_mean = np.asarray(payload["x_mean"], dtype=float)
            model.y_mean = np.asarray(payload["y_mean"], dtype=float)
            model.observed = np.asarray(payload["observed"], dtype=int)
            if model.weights.shape != (12, model.observed.size):
                raise ModelBundleError("ridge checkpoint has invalid weight shape")
        else:
            required = {"mean", "covariance", "observed"}
            if not required <= set(payload.files):
                raise ModelBundleError(
                    f"lowrank checkpoint lacks {sorted(required - set(payload.files))}"
                )
            model = LowRankConditionalMeanReconstructor()
            model.mean = np.asarray(payload["mean"], dtype=float)
            model.covariance = np.asarray(payload["covariance"], dtype=float)
            model.observed = np.asarray(payload["observed"], dtype=int)
            if model.mean.shape != (12,) or model.covariance.shape != (12, 12):
                raise ModelBundleError("lowrank checkpoint has invalid shape")
    arrays = [value for value in model.__dict__.values() if isinstance(value, np.ndarray)]
    if not all(np.isfinite(value).all() for value in arrays):
        raise ModelBundleError(f"{method} checkpoint contains non-finite values")
    model._checkpoint_path = checkpoint
    model._fitted = True
    return model


def _workspace_root(bundle_dir: Path) -> Path:
    for candidate in (bundle_dir, *bundle_dir.parents):
        if candidate.name == "artifacts":
            return candidate.parent
    return Path.cwd().resolve()


def _resolve_workspace_path(bundle_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_workspace_root(bundle_dir) / path).resolve()


def load_fitted_reconstructor(
    bundle_dir: str | Path,
    method: str,
    seed: int,
    *,
    device: str = "cpu",
):
    """Reload a fitted PTB-XL model for true external zero-transfer inference."""

    aliases = {"masked_unet": "masked-unet", "low_rank": "lowrank"}
    method = aliases.get(method, method)
    root = Path(bundle_dir).resolve()
    metadata, entries = _read_bundle(root, method, int(seed))
    if method in {"lowrank", "ridge"}:
        models = {}
        for entry in entries:
            configuration = tuple(entry.get("configuration", ()))
            if not configuration:
                raise ModelBundleError("linear checkpoint lacks its observed configuration")
            model = _load_linear(_bundle_checkpoint(root, entry), method)
            checkpoint_configuration = tuple(CANONICAL_LEADS[index] for index in model.observed)
            if checkpoint_configuration != configuration:
                raise ModelBundleError(
                    f"checkpoint observed leads {checkpoint_configuration} do not match "
                    f"metadata {configuration}"
                )
            models[configuration] = model
        return ConfigurationReconstructorBank(models, method=method)

    if method == "masked-unet":
        if len(entries) != 1:
            raise ModelBundleError("masked-unet requires exactly one checkpoint per seed")
        import torch
        from ecgcert.estimators.masked_unet import MaskedUNetReconstructor, _build_unet

        checkpoint = _bundle_checkpoint(root, entries[0])
        try:
            # This project-owned checkpoint contains a NumPy scale array in addition
            # to tensors.  Its full SHA-256 was verified above before pickle loading.
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except Exception as exc:
            raise ModelBundleError(f"cannot safely load masked-unet checkpoint: {exc}") from exc
        required = {"model", "scale", "width"}
        if not isinstance(payload, Mapping) or not required <= set(payload):
            raise ModelBundleError("masked-unet checkpoint lacks model/scale/width")
        model = MaskedUNetReconstructor()
        model.model = _build_unet(int(payload["width"])).to(torch.device(device))
        model.model.load_state_dict(payload["model"], strict=True)
        model.model.eval()
        model.scale = np.asarray(payload["scale"], dtype=np.float32)
        if model.scale.shape != (12,) or not np.isfinite(model.scale).all():
            raise ModelBundleError("masked-unet checkpoint has invalid normalization scale")
        model.device = torch.device(device)
        model._checkpoint_path = checkpoint
        model._fitted = True
        return model

    if method == "imputeecg":
        if len(entries) != 1:
            raise ModelBundleError("ImputeECG requires exactly one checkpoint per seed")
        from ecgcert.estimators.official import ImputeECGReconstructor

        source_dir = metadata.get("official", {}).get("source_dir")
        if not source_dir:
            raise ModelBundleError("ImputeECG bundle lacks its pinned source_dir")
        adapter = ImputeECGReconstructor(
            _resolve_workspace_path(root, source_dir), _bundle_checkpoint(root, entries[0])
        )
        return adapter.load(device)

    if method == "ecgrecover":
        if len(entries) != 1:
            raise ModelBundleError("ECGrecover requires exactly one checkpoint per seed")
        official = metadata.get("official", {})
        source_dir = official.get("source_dir")
        bridge = entries[0].get("inference_bridge") or official.get("inference_bridge")
        if not source_dir or not bridge:
            raise ModelBundleError(
                "ECGrecover bundle lacks official source_dir or inference command bridge"
            )
        resolved_source = _resolve_workspace_path(root, source_dir)
        from ecgcert.estimators.official import ECG_RECOVER, validate_pinned_checkout

        validate_pinned_checkout(resolved_source, ECG_RECOVER)
        input_lead = official.get("input_lead")
        configuration = tuple(entries[0].get("configuration", ()))
        if input_lead not in CANONICAL_LEADS or configuration != (input_lead,):
            raise ModelBundleError("ECGrecover bundle has an invalid official single-input lead")
        return OfficialCommandBridgeReconstructor(
            command=bridge,
            checkpoint=_bundle_checkpoint(root, entries[0]),
            source_dir=resolved_source,
            single_input_only=True,
        )
    raise ValueError(f"unknown reconstruction method: {method}")


def load_training_predictors(bundle_dir: str | Path) -> pd.DataFrame:
    """Load and hash-check the folds-1--7 simple-predictor table from a bundle."""

    root = Path(bundle_dir).resolve()
    metadata_path = root / BUNDLE_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    descriptor = metadata.get("training_predictors")
    if not isinstance(descriptor, Mapping):
        raise ModelBundleError("bundle lacks training_predictors metadata")
    path = _bundle_checkpoint(root, descriptor)
    if lineage.artifact_sha256(path) != descriptor.get("sha256"):
        raise ModelBundleError("training predictor SHA-256 mismatch")
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("loading training predictors requires locked pyarrow") from exc
    training_predictor_lookup(frame)
    return frame


__all__ = [
    "BUNDLE_FILENAME",
    "DIMENSION_COLUMNS",
    "EvaluationRecord",
    "LOG_FLOOR",
    "METRIC_FILENAME",
    "ModelBundleError",
    "ObservedSampleViolation",
    "OfficialCommandBridgeReconstructor",
    "REQUIRED_METRIC_COLUMNS",
    "ReconstructionContractError",
    "SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "TRAINING_PREDICTORS_FILENAME",
    "TrainingPredictorAccumulator",
    "checkpoint_descriptor",
    "evaluate_reconstructor",
    "evaluation_records_sha256",
    "load_fitted_reconstructor",
    "load_training_predictors",
    "training_predictor_lookup",
    "write_benchmark_artifacts",
    "write_bundle_metadata",
]
