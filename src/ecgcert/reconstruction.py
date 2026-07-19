"""Shared reconstruction evaluation and reloadable model-bundle contracts.

The benchmark entry point and external zero-transfer validation both use this
module.  Ground-truth missing leads never cross an official command bridge, and
every prediction is rejected unless the observed samples are preserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import islice
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
METRIC_INVENTORY_FILENAME = "patient_metrics.inventory.v1.json"
METRIC_STAGING_DIRNAME = ".patient-metric-shards"
METRIC_DATASET_SCHEMA_VERSION = "reconstruction-metric-dataset-v1"
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


@dataclass(frozen=True, order=True)
class MetricShardKey:
    """One independently recoverable reconstruction-evaluation unit."""

    cohort: str
    partition: str
    method: str
    model_seed: int
    configuration: str

    def __post_init__(self) -> None:
        for label in ("cohort", "partition", "method", "configuration"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"metric shard {label} must be a non-empty string")
        if (
            isinstance(self.model_seed, bool)
            or not isinstance(self.model_seed, (int, np.integer))
            or int(self.model_seed) < 0
        ):
            raise ValueError("metric shard model_seed must be a non-negative integer")
        observed = tuple(self.configuration.split("+"))
        if (
            not observed
            or any(not lead for lead in observed)
            or len(observed) != len(set(observed))
            or set(observed) - set(CANONICAL_LEADS)
        ):
            raise ValueError(
                "metric shard configuration must contain unique canonical leads"
            )

    @classmethod
    def from_values(
        cls,
        *,
        cohort: str,
        partition: str,
        method: str,
        model_seed: int,
        configuration: Sequence[str] | str,
    ) -> "MetricShardKey":
        configuration_id = (
            configuration
            if isinstance(configuration, str)
            else "+".join(str(lead) for lead in configuration)
        )
        return cls(
            cohort=str(cohort),
            partition=str(partition),
            method=str(method),
            model_seed=int(model_seed),
            configuration=configuration_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "partition": self.partition,
            "method": self.method,
            "model_seed": int(self.model_seed),
            "configuration": self.configuration,
        }

    @property
    def shard_id(self) -> str:
        return lineage.canonical_sha256(self.to_dict())[:24]


@dataclass(frozen=True)
class MetricCoverageContract:
    """Expected patient/segment/target rows for one metric shard.

    The contract is derived from the evaluation records before reconstruction,
    so a model crash or a publication bug cannot silently drop rows while still
    producing a self-consistent metric artifact.
    """

    n_rows: int
    dimensions_sha256: str
    patient_id_range: tuple[str, str]
    segments: tuple[str, ...]
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.n_rows, bool) or not isinstance(self.n_rows, int) or self.n_rows < 1:
            raise ValueError("metric coverage row count must be a positive integer")
        if not _is_sha256(self.dimensions_sha256):
            raise ValueError("metric coverage dimensions SHA-256 is invalid")
        if (
            len(self.patient_id_range) != 2
            or not all(isinstance(value, str) and value for value in self.patient_id_range)
            or self.patient_id_range[0] > self.patient_id_range[1]
        ):
            raise ValueError("metric coverage patient ID range is invalid")
        if (
            not self.segments
            or tuple(sorted(set(self.segments))) != self.segments
            or not all(isinstance(value, str) and value for value in self.segments)
        ):
            raise ValueError("metric coverage segments must be sorted and non-empty")
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or set(self.targets) - set(CANONICAL_LEADS)
        ):
            raise ValueError("metric coverage targets must be sorted canonical leads")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "dimensions_sha256": self.dimensions_sha256,
            "patient_id_range": {
                "min": self.patient_id_range[0],
                "max": self.patient_id_range[1],
            },
            "segments": list(self.segments),
            "targets": list(self.targets),
        }


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

    - ``observed_signal``: float64 ``(N,12,T)``, zero at every missing position;
    - ``observed_mask``: bool ``(N,12,T)``;
    - ``lead_order``: Unicode canonical lead names.

    Output ``.npz`` must contain only a finite ``reconstruction`` array with
    shape ``(N,12,T)``.  Commands may use ``{input}``, ``{output}``, and
    ``{checkpoint}`` placeholders.
    """

    expected_input_schema = {
        "format": "npz",
        "fields": {
            "observed_signal": "float64[N,12,T]; missing positions are zero",
            "observed_mask": "bool[N,12,T]",
            "lead_order": "unicode[12]",
        },
    }
    expected_output_schema = {
        "format": "npz",
        "fields": {"reconstruction": "finite float[N,12,T]"},
    }

    def __init__(
        self,
        *,
        command: Sequence[str],
        checkpoint: str | Path,
        source_dir: str | Path,
        single_input_only: bool,
        records_per_process: int = 128,
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
        self.preferred_batch_size = int(records_per_process)
        if self.preferred_batch_size < 1:
            raise ModelBundleError("official bridge records_per_process must be positive")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.source_dir.is_dir():
            raise FileNotFoundError(self.source_dir)

    def reconstruct(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        source = np.asarray(signal, dtype=float)
        return self.reconstruct_batch(source[None, ...], np.asarray(observed_mask)[None, ...])[0]

    def reconstruct_batch(
        self,
        signals: np.ndarray | Sequence[np.ndarray],
        observed_masks: np.ndarray | Sequence[np.ndarray],
    ) -> np.ndarray:
        source = np.asarray(signals, dtype=float)
        if source.ndim != 3 or source.shape[0] < 1 or source.shape[1] != len(CANONICAL_LEADS):
            raise ValueError(f"signals must have shape (N,12,T), got {source.shape}")
        raw_masks = np.asarray(observed_masks, dtype=bool)
        if raw_masks.shape == source.shape[:2]:
            raw_masks = np.repeat(raw_masks[:, :, None], source.shape[2], axis=2)
        if raw_masks.shape != source.shape:
            raise ValueError(
                f"observed_masks shape {raw_masks.shape} does not match {source.shape}"
            )
        mask = np.stack(
            [
                normalize_observed_mask(record_mask, record.shape)
                for record, record_mask in zip(source, raw_masks, strict=True)
            ]
        )
        if not np.all(mask == mask[:, :, :1]):
            raise ValueError("official bridge benchmark uses whole-lead masks")
        if self.single_input_only and not np.all(mask[:, :, 0].sum(axis=1) == 1):
            raise ValueError("ECGrecover is restricted to its official single-input task")
        with TemporaryDirectory(prefix="ecgcert-official-") as temporary:
            root = Path(temporary)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            np.savez(
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


def _record_batches(
    records: Iterable[EvaluationRecord], batch_size: int
) -> Iterable[tuple[EvaluationRecord, ...]]:
    iterator = iter(records)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


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
    batch_method = getattr(reconstructor, "reconstruct_batch", None)
    if callable(batch_method):
        batch_size = int(getattr(reconstructor, "preferred_batch_size", 1))
        if batch_size < 1:
            raise ReconstructionContractError(
                f"{method} declares an invalid preferred_batch_size={batch_size}"
            )
    else:
        batch_size = 1
    for record_batch in _record_batches(records, batch_size):
        references = []
        expected_masks = []
        for record in record_batch:
            record.validate()
            # Keep truth and the contract mask outside the model's object graph.  A
            # malicious or simply in-place adapter must not be able to mutate its
            # input and then pass the observed-copy comparison against that mutation.
            reference = np.array(record.signal, dtype=float, copy=True)
            references.append(reference)
            expected_masks.append(_configuration_mask(configuration, reference.shape[1]))
        if callable(batch_method):
            try:
                batch_signals = np.stack(references)
                batch_masks = np.stack(expected_masks)
            except ValueError as exc:
                raise ReconstructionContractError(
                    f"{method} batch contains heterogeneous record lengths"
                ) from exc
            model_signals = batch_signals.copy()
            model_masks = batch_masks.copy()
            batch_predictions = np.asarray(
                batch_method(model_signals, model_masks), dtype=float
            )
            if batch_predictions.shape != batch_signals.shape:
                raise ReconstructionContractError(
                    f"{method} returned invalid batch shape {batch_predictions.shape}; "
                    f"expected {batch_signals.shape}"
                )
            predictions = tuple(batch_predictions)
        else:
            predictions = tuple(
                np.asarray(
                    reconstructor.reconstruct(reference.copy(), expected_mask.copy()),
                    dtype=float,
                )
                for reference, expected_mask in zip(
                    references, expected_masks, strict=True
                )
            )
        for record, reference, expected_mask, prediction in zip(
            record_batch, references, expected_masks, predictions, strict=True
        ):
            if prediction.shape != reference.shape or not np.isfinite(prediction).all():
                raise ReconstructionContractError(
                    f"{method} returned invalid shape/values for record {record.record_id}"
                )
            if not np.array_equal(
                prediction[expected_mask], reference[expected_mask]
            ):
                raise ObservedSampleViolation(
                    f"{method} changed observed samples for record {record.record_id}"
                )
            for segment in segments:
                indices = np.asarray(
                    record.segment_indices.get(segment, ()), dtype=np.int64
                )
                if not indices.size:
                    continue
                for target_index, target in enumerate(CANONICAL_LEADS):
                    if target in configuration:
                        continue
                    truth = reference[target_index, indices]
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


def _coverage_from_dimensions(
    dimensions: Iterable[tuple[str, str, str]],
) -> MetricCoverageContract:
    values = sorted(set(dimensions))
    if not values:
        raise ValueError("metric coverage is empty")
    patients = [value[0] for value in values]
    return MetricCoverageContract(
        n_rows=len(values),
        dimensions_sha256=lineage.canonical_sha256(values),
        patient_id_range=(min(patients), max(patients)),
        segments=tuple(sorted({value[1] for value in values})),
        targets=tuple(sorted({value[2] for value in values})),
    )


def metric_coverage_contract(
    records: Iterable[EvaluationRecord],
    *,
    configuration: Sequence[str],
    segments: Sequence[str],
) -> MetricCoverageContract:
    """Derive independent expected metric rows from frozen evaluation records."""

    configuration = tuple(configuration)
    if not configuration or len(configuration) != len(set(configuration)):
        raise ValueError("coverage configuration must be non-empty and unique")
    if set(configuration) - set(CANONICAL_LEADS):
        raise ValueError("coverage configuration contains unknown leads")
    segments = tuple(str(value) for value in segments)
    if not segments or len(segments) != len(set(segments)) or any(not value for value in segments):
        raise ValueError("coverage segments must be non-empty and unique")
    patient_segments: set[tuple[str, str]] = set()
    for record in records:
        record.validate()
        for segment in segments:
            if np.asarray(record.segment_indices.get(segment, ()), dtype=np.int64).size:
                patient_segments.add((str(record.patient_id), segment))
    missing_targets = tuple(lead for lead in CANONICAL_LEADS if lead not in configuration)
    return _coverage_from_dimensions(
        (patient_id, segment, target)
        for patient_id, segment in patient_segments
        for target in missing_targets
    )


def metric_frame_coverage_contract(frame: pd.DataFrame) -> MetricCoverageContract:
    """Fingerprint metric dimensions (compatibility/tests, not an independent source)."""

    missing = {"patient_id", "segment", "target"} - set(frame.columns)
    if missing:
        raise ValueError(f"metric coverage frame lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("metric coverage frame is empty")
    dimensions = []
    for row in frame.loc[:, ["patient_id", "segment", "target"]].itertuples(index=False):
        values = (row.patient_id, row.segment, row.target)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("metric coverage dimensions must be non-empty strings")
        dimensions.append(values)
    if len(dimensions) != len(set(dimensions)):
        raise ValueError("metric coverage dimensions contain duplicate rows")
    return _coverage_from_dimensions(dimensions)


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


_NUMERIC_METRIC_COLUMNS = (
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
_INTEGER_METRIC_COLUMNS = ("model_seed", "n_observed", "n_records", "n_samples")
_FLOAT_METRIC_COLUMNS = tuple(
    column
    for column in _NUMERIC_METRIC_COLUMNS
    if column not in {"n_observed", "n_records", "n_samples"}
)
_SHARD_UNIT_COLUMNS = (
    "cohort",
    "partition",
    "method",
    "model_seed",
    "configuration",
)


def _parquet_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - locked release environments provide it
        raise RuntimeError(
            "streaming reconstruction artifacts require locked pyarrow"
        ) from exc
    return pa, pq


_STAGING_METADATA = {
    "schema_version": b"ecgcert.metric_dataset_schema_version",
    "dataset_identity": b"ecgcert.dataset_identity_sha256",
    "shard_id": b"ecgcert.metric_shard_id",
    "observed_integrity": b"ecgcert.observed_sample_integrity",
}


def _metric_table(frame: pd.DataFrame, *, metadata: Mapping[bytes, bytes] | None = None):
    pa, _ = _parquet_modules()
    table = pa.Table.from_pandas(
        frame.loc[:, list(REQUIRED_METRIC_COLUMNS)], preserve_index=False
    )
    return table.replace_schema_metadata(dict(metadata or {}))


def _metric_content_sha256(frame: pd.DataFrame) -> str:
    """Hash logical metric values independently of their Parquet container."""

    pa, _ = _parquet_modules()
    table = _metric_table(frame).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _staging_metadata(
    *, dataset_identity_sha256: str, key: MetricShardKey
) -> dict[bytes, bytes]:
    return {
        _STAGING_METADATA["schema_version"]: METRIC_DATASET_SCHEMA_VERSION.encode(),
        _STAGING_METADATA["dataset_identity"]: dataset_identity_sha256.encode(),
        _STAGING_METADATA["shard_id"]: key.shard_id.encode(),
        _STAGING_METADATA["observed_integrity"]: b"passed_exact_pointwise",
    }


def _read_staged_metric_frame(
    path: Path,
    *,
    dataset_identity_sha256: str,
    key: MetricShardKey,
) -> pd.DataFrame:
    _, pq = _parquet_modules()
    parquet = pq.ParquetFile(path)
    expected_metadata = _staging_metadata(
        dataset_identity_sha256=dataset_identity_sha256, key=key
    )
    actual_metadata = dict(parquet.schema_arrow.metadata or {})
    if any(actual_metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ReconstructionContractError(
            f"staged metric shard identity metadata changed: {key.shard_id}"
        )
    if (
        parquet.schema_arrow.names != list(REQUIRED_METRIC_COLUMNS)
        or parquet.metadata.num_row_groups != 1
    ):
        raise ReconstructionContractError(
            f"staged metric shard has an invalid Parquet schema: {key.shard_id}"
        )
    return parquet.read().to_pandas()


def _safe_metric_artifact(root: Path, relative_value: Any) -> Path:
    relative = Path(str(relative_value))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ReconstructionContractError(f"unsafe metric artifact path: {relative}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ReconstructionContractError(f"metric artifact must not be a symlink: {relative}")
    path = candidate.resolve()
    if root != path and root not in path.parents:
        raise ReconstructionContractError(f"metric artifact escapes output: {relative}")
    return path


class _IncrementalMetricValidator:
    """Validate bounded metric shards while retaining only small global state."""

    def __init__(self) -> None:
        self.units: set[MetricShardKey] = set()
        self.predictors: dict[tuple[str, str, str], tuple[float, float]] = {}

    def validate(
        self,
        frame: pd.DataFrame,
        key: MetricShardKey,
        expected_coverage: MetricCoverageContract,
    ) -> None:
        if tuple(frame.columns) != REQUIRED_METRIC_COLUMNS:
            raise ValueError("metric shard columns do not match the frozen schema and order")
        if frame.empty:
            raise ValueError(f"metric shard {key.shard_id} is empty")
        if key in self.units:
            raise ValueError(f"duplicate metric shard unit: {key.to_dict()}")
        for column, expected in key.to_dict().items():
            values = frame[column]
            if values.nunique(dropna=False) != 1 or values.iloc[0] != expected:
                raise ValueError(
                    f"metric shard {key.shard_id} does not match {column}={expected!r}"
                )
        if frame.duplicated(list(DIMENSION_COLUMNS)).any():
            raise ValueError(
                "metric shard has duplicate patient/segment/config/target/method/seed rows"
            )
        for column in ("patient_id", "segment"):
            if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
                raise ValueError(f"metric shard {column} must contain non-empty strings")
        schema_values = frame["schema_version"]
        if schema_values.nunique(dropna=False) != 1 or schema_values.iloc[0] != SCHEMA_VERSION:
            raise ValueError("metric shard has the wrong schema_version")
        numeric = frame.loc[:, list(_NUMERIC_METRIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("metric shard contains non-finite values")
        if any(
            not pd.api.types.is_integer_dtype(frame[column].dtype)
            for column in _INTEGER_METRIC_COLUMNS
        ):
            raise ValueError("metric shard integer columns must use integer dtypes")
        if any(
            not pd.api.types.is_float_dtype(frame[column].dtype)
            for column in _FLOAT_METRIC_COLUMNS
        ):
            raise ValueError("metric shard error/predictor columns must use floating dtypes")
        for column in ("n_observed", "n_records", "n_samples"):
            values = frame[column].to_numpy(dtype=float)
            if (values <= 0).any() or not np.array_equal(values, np.floor(values)):
                raise ValueError(f"metric shard {column} must contain positive integers")
        rmse = frame["rmse_mv"].to_numpy(dtype=float)
        target_rms_mv = frame["target_rms_mv"].to_numpy(dtype=float)
        normalized = frame["normalized_rmse"].to_numpy(dtype=float)
        if (rmse < 0).any() or (target_rms_mv <= 0).any() or (normalized < 0).any():
            raise ValueError("metric shard contains invalid patient-level error scales")
        if not np.array_equal(
            frame["outcome_log_rmse"].to_numpy(dtype=float),
            frame["log_rmse_mv"].to_numpy(dtype=float),
        ):
            raise ValueError("primary outcome_log_rmse must equal patient-level log(RMSE_mV)")
        expected_log = np.log(np.maximum(rmse, LOG_FLOOR))
        if not np.allclose(
            frame["log_rmse_mv"].to_numpy(dtype=float),
            expected_log,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("metric shard log_rmse_mv is inconsistent with rmse_mv")
        if not np.allclose(
            normalized,
            rmse / target_rms_mv,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("metric shard normalized_rmse is inconsistent with RMSE/RMS")
        if (frame["target_rms"].to_numpy(dtype=float) <= 0).any():
            raise ValueError("folds-1--7 target_rms must be positive")
        correlation = frame["max_target_observed_correlation"].to_numpy(dtype=float)
        if ((correlation < 0) | (correlation > 1)).any():
            raise ValueError("training-only max correlation must lie in [0,1]")
        observed = tuple(key.configuration.split("+"))
        if (
            not observed
            or len(observed) != len(set(observed))
            or any(lead not in CANONICAL_LEADS for lead in observed)
        ):
            raise ValueError(f"invalid metric configuration: {key.configuration}")
        if not frame["target"].isin(CANONICAL_LEADS).all():
            raise ValueError("metric shard targets must use canonical lead names")
        if frame["target"].isin(observed).any():
            raise ValueError("observed targets must not enter the missing-target metric table")
        observed_text = ",".join(observed)
        if (
            frame["observed_leads"].nunique(dropna=False) != 1
            or frame["observed_leads"].iloc[0] != observed_text
            or frame["n_observed"].nunique(dropna=False) != 1
            or int(frame["n_observed"].iloc[0]) != len(observed)
        ):
            raise ValueError("metric shard observed-lead metadata is inconsistent")
        actual_coverage = metric_frame_coverage_contract(frame)
        if actual_coverage != expected_coverage:
            raise ReconstructionContractError(
                f"metric shard coverage disagrees with evaluation records: {key.shard_id}"
            )

        predictor_columns = [
            "segment",
            "configuration",
            "target",
            "target_rms",
            "max_target_observed_correlation",
        ]
        local = frame.loc[:, predictor_columns].drop_duplicates()
        predictor_keys = ["segment", "configuration", "target"]
        if local.duplicated(predictor_keys).any():
            raise ValueError("simple predictors vary within a metric shard")
        for row in local.itertuples(index=False):
            predictor_key = (str(row.segment), str(row.configuration), str(row.target))
            predictor_value = (
                float(row.target_rms),
                float(row.max_target_observed_correlation),
            )
            previous = self.predictors.setdefault(predictor_key, predictor_value)
            if previous != predictor_value:
                raise ValueError(
                    "simple predictors must be fixed across patients, partitions, methods, "
                    "and seeds"
                )
        self.units.add(key)


def _metric_shard_descriptor(
    frame: pd.DataFrame,
    key: MetricShardKey,
    path: Path,
    root: Path,
) -> dict[str, Any]:
    facts = _metric_frame_facts(frame)
    return {
        "id": key.shard_id,
        **key.to_dict(),
        "status": "staged",
        "path": path.relative_to(root).as_posix(),
        "parquet_sha256": lineage.artifact_sha256(path),
        "content_sha256": _metric_content_sha256(frame),
        "dimensions_sha256": facts.pop("dimensions_sha256"),
        **facts,
        "observed_sample_integrity": "passed_exact_pointwise",
        "row_group_start": None,
        "row_group_count": None,
        "staging_retained": True,
    }


def _metric_frame_facts(frame: pd.DataFrame) -> dict[str, Any]:
    return metric_frame_coverage_contract(frame).to_dict()


_SHARD_DESCRIPTOR_KEYS = {
    "id",
    *_SHARD_UNIT_COLUMNS,
    "status",
    "path",
    "parquet_sha256",
    "content_sha256",
    "dimensions_sha256",
    "n_rows",
    "patient_id_range",
    "segments",
    "targets",
    "observed_sample_integrity",
    "row_group_start",
    "row_group_count",
    "staging_retained",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class MetricDatasetWriter:
    """Crash-recoverable shard journal with bounded-memory Parquet publication.

    Each method/seed/configuration/partition frame is first written atomically as
    an authenticated staging shard. Finalization reads one shard at a time into a
    single backward-compatible Parquet file, mapping every shard to one row group.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        expected_shards: Sequence[MetricShardKey],
        expected_coverage: Mapping[MetricShardKey, MetricCoverageContract],
        dataset_identity: Mapping[str, Any],
        resume: bool = False,
        allow_resume: bool = True,
    ) -> None:
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.inventory_path = self.root / METRIC_INVENTORY_FILENAME
        self.metrics_path = self.root / METRIC_FILENAME
        self.staging_dir = self.root / METRIC_STAGING_DIRNAME
        self.expected = tuple(expected_shards)
        if not self.expected or len(self.expected) != len(set(self.expected)):
            raise ValueError("expected metric shards must be non-empty and unique")
        self._expected_set = set(self.expected)
        expected_ids = [key.shard_id for key in self.expected]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("metric shard identifiers collide")
        self._by_id = {key.shard_id: key for key in self.expected}
        self._order = {key.shard_id: index for index, key in enumerate(self.expected)}
        self.coverage = dict(expected_coverage)
        if set(self.coverage) != self._expected_set or not all(
            isinstance(value, MetricCoverageContract) for value in self.coverage.values()
        ):
            raise ValueError(
                "expected coverage must define one MetricCoverageContract per shard"
            )
        self.identity = dict(dataset_identity)
        if not self.identity:
            raise ValueError("metric dataset identity must not be empty")
        self.identity_sha256 = lineage.canonical_sha256(self.identity)
        self.allow_resume = bool(allow_resume)
        if resume and not self.allow_resume:
            raise ValueError("metric dataset resume is disabled for this producer")
        self._validator = _IncrementalMetricValidator()
        self._descriptors: dict[str, dict[str, Any]] = {}
        self.status = "writing"

        if self.inventory_path.exists():
            if not resume:
                raise FileExistsError(
                    f"metric inventory already exists; explicit resume is required: "
                    f"{self.inventory_path}"
                )
            self._load_inventory()
        else:
            if self.metrics_path.exists():
                raise FileExistsError(
                    f"untracked metric artifact exists without inventory: {self.metrics_path}"
                )
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            self._recover_or_reject_staging_files()
            self._write_inventory()

    def _expected_payload(self) -> list[dict[str, Any]]:
        return [{"id": key.shard_id, **key.to_dict()} for key in self.expected]

    def _coverage_payload(self) -> list[dict[str, Any]]:
        return [
            {"id": key.shard_id, **self.coverage[key].to_dict()}
            for key in self.expected
        ]

    def _inventory_value(self) -> dict[str, Any]:
        descriptors = sorted(
            self._descriptors.values(), key=lambda value: self._order[str(value["id"])]
        )
        total_rows = sum(int(value["n_rows"]) for value in descriptors)
        return {
            "schema_version": METRIC_DATASET_SCHEMA_VERSION,
            "status": self.status,
            "dataset_identity": self.identity,
            "dataset_identity_sha256": self.identity_sha256,
            "metric_columns": list(REQUIRED_METRIC_COLUMNS),
            "metric_dimensions": list(DIMENSION_COLUMNS),
            "expected_shards": self._expected_payload(),
            "expected_coverage": self._coverage_payload(),
            "expected_coverage_sha256": lineage.canonical_sha256(
                self._coverage_payload()
            ),
            "n_expected_shards": len(self.expected),
            "n_completed_shards": len(descriptors),
            "total_rows": total_rows,
            "shards": descriptors,
            "recovery": {
                "resume_supported_while_writing": self.allow_resume,
                "remaining_shard_ids": [
                    key.shard_id for key in self.expected if key.shard_id not in self._descriptors
                ],
            },
            "patient_metrics": (
                {
                    "path": METRIC_FILENAME,
                    "sha256": lineage.artifact_sha256(self.metrics_path),
                }
                if self.status == "complete" and self.metrics_path.is_file()
                else None
            ),
        }

    def _write_inventory(self) -> None:
        _atomic_json(self.inventory_path, self._inventory_value())

    def _validate_descriptor_shape(
        self, descriptor: Mapping[str, Any], key: MetricShardKey
    ) -> None:
        if set(descriptor) != _SHARD_DESCRIPTOR_KEYS:
            raise ReconstructionContractError(
                f"metric shard descriptor has the wrong schema: {key.shard_id}"
            )
        if descriptor.get("id") != key.shard_id:
            raise ReconstructionContractError("metric shard descriptor ID changed")
        for column, expected in key.to_dict().items():
            if descriptor.get(column) != expected:
                raise ReconstructionContractError(
                    f"metric shard descriptor disagrees on {column}"
                )
        if (
            not _is_sha256(descriptor.get("parquet_sha256"))
            or not _is_sha256(descriptor.get("content_sha256"))
            or not _is_sha256(descriptor.get("dimensions_sha256"))
        ):
            raise ReconstructionContractError("metric shard descriptor has an invalid hash")
        n_rows = descriptor.get("n_rows")
        if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
            raise ReconstructionContractError("metric shard descriptor has invalid row count")
        patient_range = descriptor.get("patient_id_range")
        if (
            not isinstance(patient_range, dict)
            or set(patient_range) != {"min", "max"}
            or not all(isinstance(value, str) and value for value in patient_range.values())
        ):
            raise ReconstructionContractError("metric shard patient range is invalid")
        segments = descriptor.get("segments")
        targets = descriptor.get("targets")
        if (
            not isinstance(segments, list)
            or segments != sorted(set(segments))
            or not segments
            or not all(isinstance(value, str) and value for value in segments)
            or not isinstance(targets, list)
            or targets != sorted(set(targets))
            or not targets
            or set(targets) - set(CANONICAL_LEADS)
        ):
            raise ReconstructionContractError("metric shard dimensions are invalid")
        if descriptor.get("observed_sample_integrity") != "passed_exact_pointwise":
            raise ObservedSampleViolation(
                "metric shard lacks exact pointwise observed-sample attestation"
            )
        expected_coverage = self.coverage[key].to_dict()
        if any(
            descriptor.get(field) != expected_coverage[field]
            for field in expected_coverage
        ):
            raise ReconstructionContractError(
                f"metric shard descriptor violates expected coverage: {key.shard_id}"
            )

        expected_path = f"{METRIC_STAGING_DIRNAME}/{key.shard_id}.parquet"
        if self.status == "writing":
            valid_publication = (
                descriptor.get("status") == "staged"
                and descriptor.get("path") == expected_path
                and descriptor.get("row_group_start") is None
                and descriptor.get("row_group_count") is None
                and descriptor.get("staging_retained") is True
            )
        else:
            retained = descriptor.get("staging_retained")
            valid_path = (
                descriptor.get("path") == expected_path
                if retained is True
                else descriptor.get("path") is None
            )
            valid_publication = (
                descriptor.get("status") == "published"
                and isinstance(retained, bool)
                and valid_path
                and descriptor.get("row_group_start") == self._order[key.shard_id]
                and descriptor.get("row_group_count") == 1
            )
        if not valid_publication:
            raise ReconstructionContractError(
                f"metric shard publication metadata is invalid: {key.shard_id}"
            )

    @staticmethod
    def _validate_descriptor_frame(
        descriptor: Mapping[str, Any], frame: pd.DataFrame
    ) -> None:
        facts = _metric_frame_facts(frame)
        if any(descriptor.get(field) != value for field, value in facts.items()):
            raise ReconstructionContractError("metric shard facts changed during recovery")
        if descriptor.get("content_sha256") != _metric_content_sha256(frame):
            raise ReconstructionContractError("metric shard logical content SHA-256 mismatch")

    def _validate_inventory_document(self, value: Mapping[str, Any]) -> None:
        if value != self._inventory_value():
            raise ReconstructionContractError(
                "metric inventory counters, recovery state, or descriptors are inconsistent"
            )

    def _load_inventory(self) -> None:
        try:
            value = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReconstructionContractError("cannot read metric dataset inventory") from exc
        if not isinstance(value, dict):
            raise ReconstructionContractError("metric dataset inventory must be an object")
        if (
            value.get("schema_version") != METRIC_DATASET_SCHEMA_VERSION
            or value.get("dataset_identity") != self.identity
            or value.get("dataset_identity_sha256") != self.identity_sha256
            or value.get("expected_shards") != self._expected_payload()
            or value.get("expected_coverage") != self._coverage_payload()
            or value.get("expected_coverage_sha256")
            != lineage.canonical_sha256(self._coverage_payload())
        ):
            raise ReconstructionContractError(
                "metric resume inventory does not match the frozen dataset identity"
            )
        self.status = str(value.get("status"))
        if self.status not in {"writing", "complete"}:
            raise ReconstructionContractError("metric inventory has an invalid status")
        raw_descriptors = value.get("shards")
        if not isinstance(raw_descriptors, list):
            raise ReconstructionContractError("metric inventory shards must be a list")
        for descriptor in raw_descriptors:
            if not isinstance(descriptor, dict):
                raise ReconstructionContractError("invalid metric shard descriptor")
            shard_id = str(descriptor.get("id", ""))
            key = self._by_id.get(shard_id)
            if key is None or shard_id in self._descriptors:
                raise ReconstructionContractError("metric inventory contains an unknown shard")
            self._validate_descriptor_shape(descriptor, key)
            self._descriptors[shard_id] = descriptor
        if self.status == "complete":
            self._verify_complete_inventory(value)
            self._validate_inventory_document(value)
            self._cleanup_staging_after_publish()
            return
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._validator = _IncrementalMetricValidator()
        for key in self.expected:
            descriptor = self._descriptors.get(key.shard_id)
            if descriptor is None:
                continue
            path = _safe_metric_artifact(self.root, descriptor.get("path"))
            if not path.is_file() or lineage.artifact_sha256(path) != descriptor.get(
                "parquet_sha256"
            ):
                raise ReconstructionContractError(
                    f"staged metric shard is missing or changed: {key.shard_id}"
                )
            frame = _read_staged_metric_frame(
                path,
                dataset_identity_sha256=self.identity_sha256,
                key=key,
            )
            self._validator.validate(frame, key, self.coverage[key])
            self._validate_descriptor_frame(descriptor, frame)
        self._validate_inventory_document(value)
        if self._recover_or_reject_staging_files():
            self._write_inventory()

    def _recover_or_reject_staging_files(self) -> bool:
        recovered = False
        if not self.staging_dir.exists():
            return recovered
        if self.staging_dir.is_symlink() or not self.staging_dir.is_dir():
            raise ReconstructionContractError("metric staging area must be a real directory")
        for path in sorted(self.staging_dir.iterdir()):
            if path.is_symlink() or path.is_dir():
                raise ReconstructionContractError(
                    f"unsafe entry in metric staging area: {path.name}"
                )
            if path.name.endswith(".tmp"):
                path.unlink()
                continue
            if path.suffix != ".parquet":
                raise ReconstructionContractError(
                    f"unknown file in metric staging area: {path.name}"
                )
            shard_id = path.stem
            key = self._by_id.get(shard_id)
            if key is None:
                raise ReconstructionContractError(f"unknown staged metric shard: {path.name}")
            if shard_id in self._descriptors:
                continue
            frame = _read_staged_metric_frame(
                path,
                dataset_identity_sha256=self.identity_sha256,
                key=key,
            )
            self._validator.validate(frame, key, self.coverage[key])
            self._descriptors[shard_id] = _metric_shard_descriptor(
                frame, key, path, self.root
            )
            recovered = True
        return recovered

    def _verify_complete_inventory(self, value: Mapping[str, Any]) -> None:
        if set(self._descriptors) != set(self._by_id):
            raise ReconstructionContractError("complete metric inventory is missing shards")
        metrics = value.get("patient_metrics")
        if not isinstance(metrics, Mapping) or metrics.get("path") != METRIC_FILENAME:
            raise ReconstructionContractError("complete inventory lacks patient metrics")
        if not self.metrics_path.is_file() or lineage.artifact_sha256(
            self.metrics_path
        ) != metrics.get("sha256"):
            raise ReconstructionContractError("published metric dataset SHA-256 mismatch")
        _, pq = _parquet_modules()
        parquet = pq.ParquetFile(self.metrics_path)
        metadata = parquet.metadata
        total_rows = sum(int(value["n_rows"]) for value in self._descriptors.values())
        if (
            metadata.num_rows != total_rows
            or int(value.get("total_rows", -1)) != total_rows
            or metadata.num_row_groups != len(self.expected)
            or parquet.schema_arrow.names != list(REQUIRED_METRIC_COLUMNS)
        ):
            raise ReconstructionContractError("published metric dataset shape is inconsistent")
        validator = _IncrementalMetricValidator()
        for row_group_index, key in enumerate(self.expected):
            descriptor = self._descriptors[key.shard_id]
            self._validate_descriptor_shape(descriptor, key)
            frame = parquet.read_row_group(row_group_index).to_pandas()
            validator.validate(frame, key, self.coverage[key])
            self._validate_descriptor_frame(descriptor, frame)
            if descriptor.get("staging_retained") is True:
                staging_path = _safe_metric_artifact(self.root, descriptor.get("path"))
                if not staging_path.is_file() or lineage.artifact_sha256(
                    staging_path
                ) != descriptor.get("parquet_sha256"):
                    raise ReconstructionContractError(
                        f"retained metric shard is missing or changed: {key.shard_id}"
                    )
                staged = _read_staged_metric_frame(
                    staging_path,
                    dataset_identity_sha256=self.identity_sha256,
                    key=key,
                )
                self._validate_descriptor_frame(descriptor, staged)
        self._validator = validator

    def is_complete(self, key: MetricShardKey) -> bool:
        if key not in self._expected_set:
            raise ValueError(f"unexpected metric shard: {key.to_dict()}")
        return key.shard_id in self._descriptors

    @property
    def n_rows(self) -> int:
        return sum(int(value["n_rows"]) for value in self._descriptors.values())

    def write_shard(
        self,
        frame: pd.DataFrame,
        key: MetricShardKey,
        *,
        observed_sample_integrity: bool,
    ) -> dict[str, Any]:
        if not observed_sample_integrity:
            raise ObservedSampleViolation(
                "metric shards require exact pointwise observed-sample validation"
            )
        if self.status != "writing":
            if self.is_complete(key):
                return self._descriptors[key.shard_id]
            raise ReconstructionContractError("cannot append to a published metric dataset")
        if key not in self._expected_set:
            raise ValueError(f"unexpected metric shard: {key.to_dict()}")
        if self.is_complete(key):
            return self._descriptors[key.shard_id]
        self._validator.validate(frame, key, self.coverage[key])
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        path = self.staging_dir / f"{key.shard_id}.parquet"
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            _, pq = _parquet_modules()
            table = _metric_table(
                frame,
                metadata=_staging_metadata(
                    dataset_identity_sha256=self.identity_sha256,
                    key=key,
                ),
            )
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                row_group_size=table.num_rows,
            )
        except ImportError as exc:  # pragma: no cover - locked environments provide pyarrow
            raise RuntimeError("metric shard publication requires locked pyarrow") from exc
        temporary.replace(path)
        descriptor = _metric_shard_descriptor(frame, key, path, self.root)
        self._descriptors[key.shard_id] = descriptor
        self._write_inventory()
        return descriptor

    def _cleanup_staging_after_publish(self) -> None:
        if not self.staging_dir.exists():
            return
        known_paths = {
            (self.root / str(descriptor["path"])).resolve()
            for descriptor in self._descriptors.values()
            if descriptor.get("staging_retained") is True
            and descriptor.get("path") is not None
        }
        for path in self.staging_dir.glob("*"):
            if path.is_dir():
                raise ReconstructionContractError(
                    f"unexpected directory in metric staging area: {path.name}"
                )
            if path.resolve() not in known_paths and not path.name.endswith(".tmp"):
                raise ReconstructionContractError(
                    f"unexpected file in metric staging area: {path.name}"
                )
            path.unlink()
        changed = False
        for descriptor in self._descriptors.values():
            if descriptor.get("staging_retained") is not False:
                descriptor["staging_retained"] = False
                descriptor["path"] = None
                changed = True
        try:
            self.staging_dir.rmdir()
        except OSError:
            pass
        if changed and self.status == "complete":
            self._write_inventory()

    def finalize(self, *, summary: Mapping[str, Any]) -> dict[str, Any]:
        if set(self._descriptors) != set(self._by_id):
            missing = [
                key.shard_id for key in self.expected if key.shard_id not in self._descriptors
            ]
            raise ReconstructionContractError(
                f"cannot publish incomplete metric dataset; missing={missing[:8]}"
            )
        if self.status != "complete":
            _, pq = _parquet_modules()
            temporary = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            parquet_writer = None
            fresh_validator = _IncrementalMetricValidator()
            try:
                for row_group_index, key in enumerate(self.expected):
                    descriptor = self._descriptors[key.shard_id]
                    self._validate_descriptor_shape(descriptor, key)
                    path = _safe_metric_artifact(self.root, descriptor.get("path"))
                    if not path.is_file() or lineage.artifact_sha256(path) != descriptor.get(
                        "parquet_sha256"
                    ):
                        raise ReconstructionContractError(
                            f"staged metric shard changed before publish: {key.shard_id}"
                        )
                    frame = _read_staged_metric_frame(
                        path,
                        dataset_identity_sha256=self.identity_sha256,
                        key=key,
                    )
                    fresh_validator.validate(frame, key, self.coverage[key])
                    self._validate_descriptor_frame(descriptor, frame)
                    table = _metric_table(frame)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(
                            temporary, table.schema, compression="zstd"
                        )
                    elif table.schema != parquet_writer.schema:
                        raise ReconstructionContractError(
                            "metric shards disagree on the frozen Parquet column types"
                        )
                    parquet_writer.write_table(table, row_group_size=table.num_rows)
                    descriptor["status"] = "published"
                    descriptor["row_group_start"] = row_group_index
                    descriptor["row_group_count"] = 1
            finally:
                if parquet_writer is not None:
                    parquet_writer.close()
            if parquet_writer is None:
                raise ReconstructionContractError("no metric row groups were published")
            temporary.replace(self.metrics_path)
            self.status = "complete"
            self._write_inventory()
            self._cleanup_staging_after_publish()
            complete_value = json.loads(self.inventory_path.read_text(encoding="utf-8"))
            self._verify_complete_inventory(complete_value)
            self._validate_inventory_document(complete_value)

        value = dict(summary)
        value.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "n_patient_metric_rows": self.n_rows,
                "metric_dimensions": list(DIMENSION_COLUMNS),
                "metric_columns": list(REQUIRED_METRIC_COLUMNS),
                "observed_sample_integrity": "passed_exact_pointwise",
                "missing_targets_only": True,
                "normalization": {
                    "raw_unit": "mV",
                    "normalized_rmse": "RMSE_mV / patient-segment-target RMS_mV",
                    "outcome_log_rmse": (
                        "natural_log(max(RMSE_mV,1e-12)); primary outcome"
                    ),
                    "log_rmse_mv": "natural_log(max(RMSE_mV,1e-12))",
                },
                "metric_dataset": {
                    "schema_version": METRIC_DATASET_SCHEMA_VERSION,
                    "dataset_identity_sha256": self.identity_sha256,
                    "expected_coverage_sha256": lineage.canonical_sha256(
                        self._coverage_payload()
                    ),
                    "n_shards": len(self.expected),
                    "publication": "atomic staging shards -> one Parquet row group per shard",
                    "resume_supported": self.allow_resume,
                },
                "artifacts": {
                    **dict(value.get("artifacts", {})),
                    "patient_metrics": {
                        "path": METRIC_FILENAME,
                        "sha256": lineage.artifact_sha256(self.metrics_path),
                    },
                    "patient_metrics_inventory": {
                        "path": METRIC_INVENTORY_FILENAME,
                        "sha256": lineage.artifact_sha256(self.inventory_path),
                    },
                },
            }
        )
        _atomic_json(self.root / SUMMARY_FILENAME, value)
        return value


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
    """Compatibility entry point backed by the bounded-memory shard publisher."""

    missing = set(_SHARD_UNIT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"metric frame is missing shard columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("metric frame is empty")
    unit_frame = frame.loc[:, list(_SHARD_UNIT_COLUMNS)].drop_duplicates()
    keys = tuple(
        MetricShardKey.from_values(
            cohort=str(row.cohort),
            partition=str(row.partition),
            method=str(row.method),
            model_seed=int(row.model_seed),
            configuration=str(row.configuration),
        )
        for row in unit_frame.itertuples(index=False)
    )
    grouped = frame.groupby(list(_SHARD_UNIT_COLUMNS), sort=False, dropna=False)
    grouped_frames = {}
    expected_coverage = {}
    for key in keys:
        group_key = (
            key.cohort,
            key.partition,
            key.method,
            key.model_seed,
            key.configuration,
        )
        grouped_frames[key] = grouped.get_group(group_key)
        expected_coverage[key] = metric_frame_coverage_contract(grouped_frames[key])
    writer = MetricDatasetWriter(
        output_dir,
        expected_shards=keys,
        expected_coverage=expected_coverage,
        dataset_identity={
            "schema_version": SCHEMA_VERSION,
            "mode": "bounded-dataframe-compatibility",
            "shards": [key.to_dict() for key in keys],
        },
        allow_resume=False,
    )
    for key in keys:
        writer.write_shard(
            grouped_frames[key],
            key,
            observed_sample_integrity=True,
        )
    return writer.finalize(summary=summary)


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
        records_per_process = int(official.get("inference_records_per_process", 0))
        configuration = tuple(entries[0].get("configuration", ()))
        if input_lead not in CANONICAL_LEADS or configuration != (input_lead,):
            raise ModelBundleError("ECGrecover bundle has an invalid official single-input lead")
        if records_per_process < 1:
            raise ModelBundleError("ECGrecover bundle has no valid inference batch size")
        return OfficialCommandBridgeReconstructor(
            command=bridge,
            checkpoint=_bundle_checkpoint(root, entries[0]),
            source_dir=resolved_source,
            single_input_only=True,
            records_per_process=records_per_process,
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
    "METRIC_DATASET_SCHEMA_VERSION",
    "METRIC_FILENAME",
    "METRIC_INVENTORY_FILENAME",
    "MetricCoverageContract",
    "MetricDatasetWriter",
    "MetricShardKey",
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
    "metric_coverage_contract",
    "metric_frame_coverage_contract",
    "training_predictor_lookup",
    "write_benchmark_artifacts",
    "write_bundle_metadata",
]
