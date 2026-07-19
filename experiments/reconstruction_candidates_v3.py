"""Fit the preregistered reconstruction grid and score it on PTB-XL fold 8.

This is the executable producer consumed by :mod:`tune_reconstructors_v3`.
Every candidate is fitted only on folds 1--7.  Low-rank and ridge candidates
are evaluated directly on the complete frozen fold-8 panel.  Masked U-Net
candidates use fold 8 only as a validation set: each seed is stopped with the
frozen patience rule, its best state is restored, and that state is evaluated
once on the complete patient/configuration/segment/target panel.  The compact
best-state representation avoids materialising the same multi-million-row
patient table at every neural training epoch.

No fold-9 calibration or fold-10 test record is loaded by this entry point.
Official methods are deliberately absent: their hyperparameters are pinned by
their official protocols and are never selected by held-out outcomes.

Formal-scale metrics are committed as authenticated candidate/seed/configuration
shards.  A rerun resumes completed units (and completed U-Net training seeds),
then atomically publishes one row-grouped Parquet artifact without a full-table
``DataFrame`` concatenation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators import (
    LowRankConditionalMeanReconstructor,
    MaskedUNetReconstructor,
    RidgeLeadReconstructor,
    TrainManifest,
)
from ecgcert.estimators.masked_unet import _build_unet
from ecgcert.protocol import (
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.reconstruction import (
    EvaluationRecord,
    TRAINING_PREDICTORS_FILENAME,
    evaluate_reconstructor,
    evaluation_records_sha256,
    training_predictor_lookup,
)
from ecgcert.training_inclusion import TrainingInclusion, load_training_inclusion

try:  # package import in tests; sibling import under ``python experiments/...``
    from .reconstruction_benchmark_v3 import (
        _atomic_json,
        _load_evaluation_records,
        _materialize_training_manifest,
        _require_parquet_engine,
        _streaming_training_moments,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
    from .tune_reconstructors_v3 import (
        CANDIDATE_BUNDLE_SCHEMA_VERSION,
        CANDIDATE_SCHEMA_VERSION,
        TRACE_SCHEMA_VERSION,
        TUNING_SEEDS,
        Candidate,
        _validate_early_stop_trace,
        _verify_candidate_checkpoints,
        candidate_grid,
        scan_candidate_metrics_parquet,
    )
except ImportError:  # pragma: no cover - direct CLI import path
    from reconstruction_benchmark_v3 import (
        _atomic_json,
        _load_evaluation_records,
        _materialize_training_manifest,
        _require_parquet_engine,
        _streaming_training_moments,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
    from tune_reconstructors_v3 import (
        CANDIDATE_BUNDLE_SCHEMA_VERSION,
        CANDIDATE_SCHEMA_VERSION,
        TRACE_SCHEMA_VERSION,
        TUNING_SEEDS,
        Candidate,
        _validate_early_stop_trace,
        _verify_candidate_checkpoints,
        candidate_grid,
        scan_candidate_metrics_parquet,
    )


CANDIDATE_METRICS_FILENAME = "candidate_metrics.parquet"
CANDIDATE_METRIC_INVENTORY_FILENAME = "candidate_metrics.inventory.v1.json"
CANDIDATE_METRIC_INVENTORY_SCHEMA_VERSION = "candidate-metric-inventory-v1"
EARLY_STOP_TRACE_FILENAME = "unet_early_stop_trace.parquet"
CANDIDATE_BUNDLE_FILENAME = "candidate_bundle.json"
CANDIDATE_COLUMNS = (
    "schema_version",
    "cohort",
    "train_partition",
    "partition",
    "manifest_sha256",
    "split_sha256",
    "method",
    "candidate_id",
    "patient_id",
    "segment",
    "configuration",
    "target",
    "model_seed",
    "epoch",
    "rmse_mv",
    "log_rmse_mv",
    "checkpoint_path",
    "checkpoint_sha256",
    "observed_integrity",
)
TRACE_COLUMNS = (
    "schema_version",
    "cohort",
    "train_partition",
    "partition",
    "manifest_sha256",
    "split_sha256",
    "fold8_records_sha256",
    "configuration_panel_sha256",
    "candidate_id",
    "model_seed",
    "epoch",
    "monitor_log_rmse_mv",
    "best_so_far_log_rmse_mv",
    "stale_epochs",
    "is_strict_improvement",
    "best_epoch",
    "stopped_epoch",
    "checkpoint_path",
    "checkpoint_sha256",
)


def _candidate_unit(
    method: str,
    candidate_id: str,
    model_seed: int,
    configuration: Sequence[str] | str,
) -> tuple[str, str, int, str]:
    configuration_id = (
        str(configuration)
        if isinstance(configuration, str)
        else "+".join(str(lead) for lead in configuration)
    )
    return str(method), str(candidate_id), int(model_seed), configuration_id


class CandidateMetricStore:
    """Crash-recoverable candidate/configuration shards with atomic publication."""

    def __init__(
        self,
        output_dir: Path,
        *,
        identity: Mapping[str, Any],
        configurations: Sequence[Sequence[str]],
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - locked release env contains pyarrow
            raise RuntimeError("candidate metric storage requires locked pyarrow") from exc
        self._pq = pq
        self.output_dir = output_dir.resolve()
        self.inventory_path = self.output_dir / CANDIDATE_METRIC_INVENTORY_FILENAME
        self.metrics_path = self.output_dir / CANDIDATE_METRICS_FILENAME
        self.staging_dir = self.output_dir / ".candidate_metrics.staging"
        self.configurations = tuple(tuple(value) for value in configurations)
        self.identity = dict(identity)
        self.identity_sha256 = lineage.canonical_sha256(self.identity)
        self.expected = self._expected_units()
        self._expected_set = set(self.expected)
        self._unit_index = {unit: index for index, unit in enumerate(self.expected)}
        self._descriptors: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        self.status = "writing"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.inventory_path.exists():
            self._load_inventory()
        elif any(self.output_dir.iterdir()):
            raise FileExistsError(
                "candidate output is non-empty but has no authenticated progress inventory: "
                f"{self.output_dir}"
            )
        else:
            self.staging_dir.mkdir()
            self._write_inventory()
        if self.status == "writing":
            self.staging_dir.mkdir(exist_ok=True)
            self._adopt_orphan_shards()

    def _expected_units(self) -> tuple[tuple[str, str, int, str], ...]:
        units = []
        grid = candidate_grid()
        for method in ("lowrank", "ridge", "masked-unet"):
            seeds = TUNING_SEEDS if method == "masked-unet" else (0,)
            for candidate in grid[method]:
                for seed in seeds:
                    for configuration in self.configurations:
                        units.append(
                            _candidate_unit(
                                method, candidate.candidate_id, seed, configuration
                            )
                        )
        return tuple(units)

    @staticmethod
    def _unit_dict(unit: tuple[str, str, int, str]) -> dict[str, Any]:
        return {
            "method": unit[0],
            "candidate_id": unit[1],
            "model_seed": unit[2],
            "configuration": unit[3],
        }

    @staticmethod
    def _unit_from_dict(value: Mapping[str, Any]) -> tuple[str, str, int, str]:
        return _candidate_unit(
            str(value.get("method", "")),
            str(value.get("candidate_id", "")),
            int(value.get("model_seed", -1)),
            str(value.get("configuration", "")),
        )

    def _shard_path(self, unit: tuple[str, str, int, str]) -> Path:
        ordinal = self._unit_index[unit]
        digest = lineage.canonical_sha256(self._unit_dict(unit))[:20]
        return self.staging_dir / f"{ordinal:05d}-{digest}.parquet"

    def _inventory_value(self) -> dict[str, Any]:
        descriptors = []
        for unit in self.expected:
            if unit not in self._descriptors:
                continue
            descriptors.append(
                {"unit": self._unit_dict(unit), **self._descriptors[unit]}
            )
        value = {
            "schema_version": CANDIDATE_METRIC_INVENTORY_SCHEMA_VERSION,
            "status": self.status,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "expected_units": [self._unit_dict(unit) for unit in self.expected],
            "expected_units_sha256": lineage.canonical_sha256(
                [self._unit_dict(unit) for unit in self.expected]
            ),
            "completed_units": descriptors,
            "n_expected_units": len(self.expected),
            "n_completed_units": len(descriptors),
            "n_rows": sum(int(item["n_rows"]) for item in self._descriptors.values()),
        }
        if self.status == "complete":
            value["candidate_metrics"] = {
                "path": CANDIDATE_METRICS_FILENAME,
                "sha256": lineage.artifact_sha256(self.metrics_path),
            }
        return value

    def _write_inventory(self) -> None:
        _atomic_json(self.inventory_path, self._inventory_value())

    def _load_inventory(self) -> None:
        value = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != CANDIDATE_METRIC_INVENTORY_SCHEMA_VERSION:
            raise ValueError("candidate metric inventory schema is invalid")
        if value.get("identity") != self.identity or value.get(
            "identity_sha256"
        ) != self.identity_sha256:
            raise ValueError("candidate metric resume identity changed")
        expected_values = [self._unit_dict(unit) for unit in self.expected]
        if value.get("expected_units") != expected_values or value.get(
            "expected_units_sha256"
        ) != lineage.canonical_sha256(expected_values):
            raise ValueError("candidate metric expected-unit inventory changed")
        status = value.get("status")
        if status not in {"writing", "complete"}:
            raise ValueError("candidate metric inventory status is invalid")
        for raw in value.get("completed_units", []):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("unit"), Mapping):
                raise ValueError("candidate metric descriptor is invalid")
            unit = self._unit_from_dict(raw["unit"])
            if unit not in self._expected_set or unit in self._descriptors:
                raise ValueError("candidate metric descriptor unit is unexpected or duplicated")
            descriptor = {key: raw[key] for key in raw if key != "unit"}
            self._validate_descriptor(unit, descriptor, require_staging=status == "writing")
            self._descriptors[unit] = descriptor
        if int(value.get("n_completed_units", -1)) != len(self._descriptors):
            raise ValueError("candidate metric inventory completed count is inconsistent")
        if int(value.get("n_rows", -1)) != sum(
            int(item["n_rows"]) for item in self._descriptors.values()
        ):
            raise ValueError("candidate metric inventory row count is inconsistent")
        self.status = str(status)
        if self.status == "complete":
            metric = value.get("candidate_metrics")
            if (
                set(self._descriptors) != self._expected_set
                or not isinstance(metric, Mapping)
                or metric.get("path") != CANDIDATE_METRICS_FILENAME
                or not self.metrics_path.is_file()
                or lineage.artifact_sha256(self.metrics_path) != metric.get("sha256")
            ):
                raise ValueError("published candidate metric artifact is incomplete or changed")

    def _validate_descriptor(
        self,
        unit: tuple[str, str, int, str],
        descriptor: Mapping[str, Any],
        *,
        require_staging: bool,
    ) -> None:
        if not isinstance(descriptor.get("n_rows"), int) or descriptor["n_rows"] < 1:
            raise ValueError("candidate metric descriptor row count is invalid")
        if not isinstance(descriptor.get("epoch"), int):
            raise ValueError("candidate metric descriptor epoch is invalid")
        if not isinstance(descriptor.get("sha256"), str) or len(descriptor["sha256"]) != 64:
            raise ValueError("candidate metric descriptor SHA-256 is invalid")
        if require_staging:
            relative = descriptor.get("path")
            if relative != self._shard_path(unit).relative_to(self.output_dir).as_posix():
                raise ValueError("candidate metric descriptor path is invalid")
            path = self.output_dir / str(relative)
            if not path.is_file() or lineage.artifact_sha256(path) != descriptor["sha256"]:
                raise ValueError("candidate metric staged shard is missing or changed")

    def _read_shard_unit(self, path: Path) -> tuple[tuple[str, str, int, str], int]:
        parquet = self._pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"ecgcert_candidate_identity_sha256", b"").decode() != (
            self.identity_sha256
        ):
            raise ValueError("candidate metric shard identity is invalid")
        try:
            raw_unit = json.loads(metadata[b"ecgcert_candidate_unit_json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("candidate metric shard unit metadata is invalid") from exc
        unit = self._unit_from_dict(raw_unit)
        if unit not in self._expected_set:
            raise ValueError("candidate metric shard contains an unexpected unit")
        if parquet.metadata.num_row_groups != 1 or parquet.metadata.num_rows < 1:
            raise ValueError("candidate metric shard must contain one non-empty row group")
        table = parquet.read()
        frame = table.to_pandas()
        actual = {
            _candidate_unit(method, candidate, seed, configuration)
            for method, candidate, seed, configuration in frame[
                ["method", "candidate_id", "model_seed", "configuration"]
            ].itertuples(index=False, name=None)
        }
        epochs = set(frame["epoch"].astype(int))
        if actual != {unit} or len(epochs) != 1:
            raise ValueError("candidate metric shard rows disagree with shard metadata")
        return unit, int(next(iter(epochs)))

    def _adopt_orphan_shards(self) -> None:
        changed = False
        expected_paths = {self._shard_path(unit) for unit in self.expected}
        for path in sorted(self.staging_dir.iterdir()):
            if path.is_dir():
                raise ValueError("unexpected directory in candidate metric staging")
            if path.name.endswith(".parquet.tmp"):
                final_path = path.with_suffix("")
                if final_path not in expected_paths:
                    raise ValueError("unexpected temporary candidate metric shard")
                path.unlink()
            elif path.suffix != ".parquet" or path not in expected_paths:
                raise ValueError("unexpected file in candidate metric staging")
        for path in sorted(self.staging_dir.glob("*.parquet")):
            unit, epoch = self._read_shard_unit(path)
            if path != self._shard_path(unit):
                raise ValueError("candidate metric orphan shard filename is invalid")
            digest = lineage.artifact_sha256(path)
            if unit in self._descriptors:
                descriptor = self._descriptors[unit]
                row_count = int(self._pq.ParquetFile(path).metadata.num_rows)
                if (
                    descriptor["sha256"] != digest
                    or int(descriptor["epoch"]) != epoch
                    or int(descriptor["n_rows"]) != row_count
                ):
                    raise ValueError("candidate metric recorded shard changed")
                continue
            self._descriptors[unit] = {
                "path": path.relative_to(self.output_dir).as_posix(),
                "sha256": digest,
                "n_rows": int(self._pq.ParquetFile(path).metadata.num_rows),
                "epoch": epoch,
                "staging_retained": True,
            }
            changed = True
        if changed:
            self._write_inventory()

    def is_complete(
        self,
        method: str,
        candidate_id: str,
        model_seed: int,
        configuration: Sequence[str] | str,
    ) -> bool:
        return _candidate_unit(
            method, candidate_id, model_seed, configuration
        ) in self._descriptors

    def seed_complete(self, method: str, candidate_id: str, model_seed: int) -> bool:
        return all(
            self.is_complete(method, candidate_id, model_seed, configuration)
            for configuration in self.configurations
        )

    def write_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        if self.status != "writing":
            raise ValueError("cannot append to a published candidate metric store")
        if frame.empty:
            raise ValueError("candidate metric shard is empty")
        units = {
            _candidate_unit(method, candidate, seed, configuration)
            for method, candidate, seed, configuration in frame[
                ["method", "candidate_id", "model_seed", "configuration"]
            ].itertuples(index=False, name=None)
        }
        epochs = set(frame["epoch"].astype(int))
        if len(units) != 1 or len(epochs) != 1:
            raise ValueError("candidate metric writes must contain one candidate/configuration unit")
        unit = next(iter(units))
        if unit not in self._expected_set:
            raise ValueError(f"unexpected candidate metric unit: {unit}")
        if unit in self._descriptors:
            raise ValueError(f"candidate metric unit was evaluated twice: {unit}")
        import pyarrow as pa

        path = self._shard_path(unit)
        temporary = path.with_suffix(path.suffix + ".tmp")
        metadata = {
            b"ecgcert_candidate_identity_sha256": self.identity_sha256.encode(),
            b"ecgcert_candidate_unit_json": json.dumps(
                self._unit_dict(unit), sort_keys=True, separators=(",", ":")
            ).encode(),
        }
        table = pa.Table.from_pandas(
            frame.loc[:, list(CANDIDATE_COLUMNS)], preserve_index=False
        ).replace_schema_metadata(metadata)
        self._pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            row_group_size=len(table),
        )
        temporary.replace(path)
        descriptor = {
            "path": path.relative_to(self.output_dir).as_posix(),
            "sha256": lineage.artifact_sha256(path),
            "n_rows": len(frame),
            "epoch": int(next(iter(epochs))),
            "staging_retained": True,
        }
        self._descriptors[unit] = descriptor
        self._write_inventory()
        return descriptor

    @property
    def n_rows(self) -> int:
        return sum(int(item["n_rows"]) for item in self._descriptors.values())

    def finalize(self) -> Path:
        if set(self._descriptors) != self._expected_set:
            missing = [unit for unit in self.expected if unit not in self._descriptors]
            raise ValueError(f"candidate metric store is incomplete: {missing[:3]}")
        if self.status == "complete":
            return self.metrics_path
        temporary = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        writer = None
        schema = None
        try:
            for unit in self.expected:
                descriptor = self._descriptors[unit]
                path = self.output_dir / str(descriptor["path"])
                if lineage.artifact_sha256(path) != descriptor["sha256"]:
                    raise ValueError("candidate metric shard changed before publication")
                table = self._pq.read_table(path).replace_schema_metadata(None)
                if writer is None:
                    schema = table.schema
                    writer = self._pq.ParquetWriter(
                        temporary,
                        schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                elif table.schema != schema:
                    raise ValueError("candidate metric shards disagree on Parquet schema")
                writer.write_table(table, row_group_size=len(table))
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("candidate metric store published no rows")
        temporary.replace(self.metrics_path)
        self.status = "complete"
        self._write_inventory()
        return self.metrics_path

    def cleanup_staging(self) -> None:
        if self.status != "complete":
            raise ValueError("cannot clean an incomplete candidate metric store")
        for unit, descriptor in self._descriptors.items():
            path = self._shard_path(unit)
            if path.exists():
                if lineage.artifact_sha256(path) != descriptor["sha256"]:
                    raise ValueError("candidate metric shard changed before cleanup")
                path.unlink()
            descriptor["path"] = None
            descriptor["staging_retained"] = False
        if self.staging_dir.exists():
            self.staging_dir.rmdir()
        self._write_inventory()


def _parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-inclusion", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=int, default=PRIMARY_RATE_HZ)
    parser.add_argument("--segments", type=_parse_csv, default=PRIMARY_SEGMENTS)
    parser.add_argument("--delineator", choices=("dwt", "peak"), default="dwt")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--release", action="store_true")
    return parser


def validate_release_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_records is not None and arguments.max_records < 1:
        raise ValueError("--max-records must be positive")
    if arguments.max_configurations is not None and arguments.max_configurations < 1:
        raise ValueError("--max-configurations must be positive")
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
    if getattr(arguments, "training_inclusion", None) is None:
        violations.append("--training-inclusion is required (shared folds1-7 cohort)")
    try:
        arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
    except ValueError:
        violations.append("--output-dir must be under artifacts/ for release")
    if violations:
        raise ValueError("release candidate-grid protocol violation: " + "; ".join(violations))


def _checkpoint_fields(checkpoint: Path, output_dir: Path) -> tuple[str, str]:
    relative = checkpoint.resolve().relative_to(output_dir.resolve()).as_posix()
    return relative, lineage.artifact_sha256(checkpoint)


def candidate_metric_rows(
    metrics: pd.DataFrame,
    *,
    candidate: Candidate,
    model_seed: int,
    epoch: int,
    checkpoint: Path,
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
) -> pd.DataFrame:
    """Convert shared benchmark metrics into the strict fold-8 tuning schema."""

    if metrics.empty:
        raise ValueError(f"{candidate.candidate_id} produced no fold-8 metrics")
    if set(metrics["partition"].astype(str)) != {"fold8/tune"}:
        raise ValueError("candidate metrics may only contain fold8/tune rows")
    if set(metrics["method"].astype(str)) != {candidate.method}:
        raise ValueError("candidate metric method does not match its grid declaration")
    if set(metrics["model_seed"].astype(int)) != {int(model_seed)}:
        raise ValueError("candidate metric seed does not match its fitted model")
    if not np.array_equal(
        metrics["outcome_log_rmse"].to_numpy(dtype=float),
        metrics["log_rmse_mv"].to_numpy(dtype=float),
    ):
        raise ValueError("candidate primary outcome must be raw log(RMSE_mV)")
    checkpoint_path, checkpoint_sha256 = _checkpoint_fields(checkpoint, output_dir)
    output = metrics.loc[
        :, ["patient_id", "segment", "configuration", "target", "rmse_mv", "log_rmse_mv"]
    ].copy()
    output.insert(0, "candidate_id", candidate.candidate_id)
    output.insert(0, "method", candidate.method)
    output.insert(0, "split_sha256", split_sha256)
    output.insert(0, "manifest_sha256", manifest_sha256)
    output.insert(0, "partition", "fold8/tune")
    output.insert(0, "train_partition", "folds1-7/train")
    output.insert(0, "cohort", "PTB-XL")
    output.insert(0, "schema_version", CANDIDATE_SCHEMA_VERSION)
    output["model_seed"] = int(model_seed)
    output["epoch"] = int(epoch)
    output["checkpoint_path"] = checkpoint_path
    output["checkpoint_sha256"] = checkpoint_sha256
    # ``evaluate_reconstructor`` already fails closed if a method changes any
    # observed sample.  This explicit flag makes that successful check auditable.
    output["observed_integrity"] = True
    return output.loc[:, list(CANDIDATE_COLUMNS)]


def _linear_candidate_checkpoint(
    method: str,
    candidate: Candidate,
    output_dir: Path,
    *,
    mean: np.ndarray,
    scatter: np.ndarray,
    sample_count: int,
    covariance: np.ndarray | None = None,
) -> Path:
    checkpoint = output_dir / "checkpoints" / method / candidate.candidate_id / "seed-0.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "mean": np.asarray(mean, dtype=np.float64),
        "scatter": np.asarray(scatter, dtype=np.float64),
        "sample_count": np.asarray([sample_count], dtype=np.int64),
        "candidate_parameters_json": np.asarray(
            [json.dumps(dict(candidate.parameters), sort_keys=True, allow_nan=False)]
        ),
    }
    if covariance is not None:
        payload["covariance"] = np.asarray(covariance, dtype=np.float64)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as stored:
            if set(stored.files) != set(payload) or any(
                not np.array_equal(stored[key], expected)
                for key, expected in payload.items()
            ):
                raise ValueError(
                    f"existing linear candidate checkpoint changed: {checkpoint}"
                )
        return checkpoint
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(checkpoint)
    return checkpoint


def evaluate_linear_candidate_grid(
    method: str,
    *,
    mean: np.ndarray,
    scatter: np.ndarray,
    sample_count: int,
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
    metric_sink: Callable[[pd.DataFrame], Any] | None = None,
    is_complete: Callable[[str, str, int, Sequence[str]], bool] | None = None,
) -> list[pd.DataFrame]:
    """Fit and evaluate every real low-rank or ridge candidate."""

    if method not in {"lowrank", "ridge"}:
        raise ValueError(f"not a native linear candidate method: {method}")
    frames: list[pd.DataFrame] = []
    for candidate in candidate_grid()[method]:
        if is_complete is not None and all(
            is_complete(method, candidate.candidate_id, 0, configuration)
            for configuration in configurations
        ):
            continue
        covariance = None
        ridge_lambda = None
        if method == "lowrank":
            rank = int(candidate.parameters["rank"])
            noise_variance = float(candidate.parameters["noise_variance"])
            eigenvalues, eigenvectors = np.linalg.eigh(scatter)
            order = np.argsort(eigenvalues)[::-1][:rank]
            basis = eigenvectors[:, order]
            coordinate_variance = np.maximum(eigenvalues[order], 0.0) / (
                sample_count - 1
            )
            covariance = (basis * coordinate_variance[None, :]) @ basis.T
            covariance += noise_variance * np.eye(len(CANONICAL_LEADS))
        else:
            ridge_lambda = float(candidate.parameters["ridge_lambda"])
        checkpoint = _linear_candidate_checkpoint(
            method,
            candidate,
            output_dir,
            mean=mean,
            scatter=scatter,
            sample_count=sample_count,
            covariance=covariance,
        )
        for configuration in configurations:
            if is_complete is not None and is_complete(
                method, candidate.candidate_id, 0, configuration
            ):
                continue
            observed = np.asarray(
                [CANONICAL_LEADS.index(lead) for lead in configuration], dtype=np.int64
            )
            if method == "lowrank":
                assert covariance is not None
                model = LowRankConditionalMeanReconstructor()
                model.mean = np.asarray(mean, dtype=np.float64)
                model.covariance = covariance
                model.observed = observed
            else:
                assert ridge_lambda is not None
                gram = scatter[np.ix_(observed, observed)] + ridge_lambda * np.eye(
                    observed.size
                )
                model = RidgeLeadReconstructor()
                model.x_mean = np.asarray(mean[observed], dtype=np.float64)
                model.y_mean = np.asarray(mean, dtype=np.float64)
                model.weights = np.linalg.solve(gram, scatter[observed]).T
                model.observed = observed
            model._checkpoint_path = checkpoint
            model._fitted = True
            metrics = evaluate_reconstructor(
                model,
                evaluation_records,
                configuration=configuration,
                method=method,
                model_seed=0,
                segments=segments,
                training_predictors=training_predictors,
                cohort="PTB-XL",
                partition="fold8/tune",
            )
            candidate_rows = candidate_metric_rows(
                metrics,
                candidate=candidate,
                model_seed=0,
                epoch=0,
                checkpoint=checkpoint,
                output_dir=output_dir,
                manifest_sha256=manifest_sha256,
                split_sha256=split_sha256,
            )
            if metric_sink is None:
                frames.append(candidate_rows)
            else:
                metric_sink(candidate_rows)
    return frames


def _patient_balanced_score(frame: pd.DataFrame) -> float:
    patients = frame.groupby("patient_id", sort=True)["log_rmse_mv"].mean()
    if patients.empty or not np.isfinite(patients.to_numpy(dtype=float)).all():
        raise ValueError("fold-8 early-stopping monitor is empty or non-finite")
    return float(patients.mean())


def _early_stop_fold8_score(
    model: MaskedUNetReconstructor,
    records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    *,
    seed: int,
    epoch: int,
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> float:
    """Patient-balanced fold-8 monitor with a frozen rotating mask assignment.

    A record is evaluated under one deterministically assigned configuration.
    The assignment is fixed across epochs and distributed across the complete
    panel, so early-stopping scores are comparable while requiring one, rather
    than 64, neural forward passes per record.  The restored best model is
    subsequently scored on every configuration.
    """

    frames = []
    del epoch  # the validation mask assignment is intentionally frozen across epochs
    offset = (int(seed) * 17) % len(configurations)
    grouped_records: dict[tuple[str, ...], list[EvaluationRecord]] = {}
    for index, record in enumerate(records):
        configuration = tuple(configurations[(index + offset) % len(configurations)])
        grouped_records.setdefault(configuration, []).append(record)
    for configuration, configuration_records in grouped_records.items():
        frames.append(
            evaluate_reconstructor(
                model,
                configuration_records,
                configuration=configuration,
                method="masked-unet",
                model_seed=int(seed),
                segments=segments,
                training_predictors=training_predictors,
                cohort="PTB-XL",
                partition="fold8/tune",
            )
        )
    return _patient_balanced_score(pd.concat(frames, ignore_index=True))


def _masked_unet_runtime(model: Any, scale: np.ndarray, device: Any) -> MaskedUNetReconstructor:
    runtime = MaskedUNetReconstructor()
    runtime.model = model
    runtime.scale = np.asarray(scale, dtype=np.float32)
    runtime.device = device
    runtime._fitted = True
    return runtime


def _evaluate_fitted_masked_unet_candidate(
    runtime: MaskedUNetReconstructor,
    candidate: Candidate,
    *,
    seed: int,
    best_epoch: int,
    checkpoint: Path,
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
    metric_sink: Callable[[pd.DataFrame], Any] | None = None,
    is_complete: Callable[[str, str, int, Sequence[str]], bool] | None = None,
) -> list[pd.DataFrame]:
    frames = []
    for configuration in configurations:
        if is_complete is not None and is_complete(
            "masked-unet", candidate.candidate_id, int(seed), configuration
        ):
            continue
        metrics = evaluate_reconstructor(
            runtime,
            evaluation_records,
            configuration=configuration,
            method="masked-unet",
            model_seed=int(seed),
            segments=segments,
            training_predictors=training_predictors,
            cohort="PTB-XL",
            partition="fold8/tune",
        )
        candidate_rows = candidate_metric_rows(
            metrics,
            candidate=candidate,
            model_seed=int(seed),
            epoch=best_epoch,
            checkpoint=checkpoint,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            split_sha256=split_sha256,
        )
        if metric_sink is None:
            frames.append(candidate_rows)
        else:
            metric_sink(candidate_rows)
    return frames


def _load_masked_unet_candidate_checkpoint(
    candidate: Candidate,
    *,
    seed: int,
    audit: Mapping[str, Any],
    train_manifest: TrainManifest,
    output_dir: Path,
    device_name: str,
) -> tuple[MaskedUNetReconstructor, Path]:
    """Restore one completed training seed without consulting fold-8 outcomes again."""

    import torch

    if (
        audit.get("candidate_id") != candidate.candidate_id
        or int(audit.get("model_seed", -1)) != int(seed)
    ):
        raise ValueError("masked U-Net resume audit identity changed")
    relative = Path(str(audit.get("checkpoint_path", "")))
    expected = (
        Path("checkpoints")
        / "masked-unet"
        / candidate.candidate_id
        / f"seed-{seed}.pt"
    )
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("masked U-Net resume checkpoint path is invalid")
    checkpoint = (output_dir / relative).resolve()
    if output_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise ValueError("masked U-Net resume checkpoint is missing")
    if lineage.artifact_sha256(checkpoint) != audit.get("checkpoint_sha256"):
        raise ValueError("masked U-Net resume checkpoint SHA-256 changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(candidate.parameters)
    if (
        payload.get("candidate_id") != candidate.candidate_id
        or int(payload.get("seed", -1)) != int(seed)
        or payload.get("candidate_parameters") != parameters
        or payload.get("train_manifest_sha256") != train_manifest.signals_sha256
        or payload.get("training_record_ids_sha256")
        != train_manifest.record_ids_sha256
        or payload.get("training_patient_ids_sha256")
        != train_manifest.patient_ids_sha256
        or payload.get("training_inclusion_sha256")
        != train_manifest.training_inclusion_sha256
        or int(payload.get("best_epoch", -1)) != int(audit.get("best_epoch", -2))
        or int(payload.get("stopped_epoch", -1)) != int(audit.get("stopped_epoch", -2))
        or payload.get("fold8_monitor_history") != audit.get("history")
    ):
        raise ValueError("masked U-Net resume checkpoint metadata changed")
    scale = np.asarray(payload.get("scale"), dtype=np.float32)
    if scale.shape != (len(CANONICAL_LEADS),) or not np.isfinite(scale).all():
        raise ValueError("masked U-Net resume normalization scale is invalid")
    device = torch.device(device_name)
    network = _build_unet(width=int(parameters["width"])).to(device)
    network.load_state_dict(payload["model"])
    network.eval()
    return _masked_unet_runtime(network, scale, device), checkpoint


def fit_masked_unet_candidate(
    candidate: Candidate,
    *,
    seed: int,
    train_manifest: TrainManifest,
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
    device_name: str,
    normalization_scale: np.ndarray | None = None,
    metric_sink: Callable[[pd.DataFrame], Any] | None = None,
    audit_sink: Callable[[Mapping[str, Any]], Any] | None = None,
    is_complete: Callable[[str, str, int, Sequence[str]], bool] | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Train one U-Net seed, restore its patience-selected state, and score it."""

    import torch

    if candidate.method != "masked-unet":
        raise ValueError("fit_masked_unet_candidate requires a masked-unet candidate")
    # ``run`` performs the full multi-gigabyte file hash once before dispatching
    # candidates.  Structural checks remain local without hashing it nine more
    # times (three settings x three seeds).
    train_manifest.validate(verify_file=False)
    signals = np.load(train_manifest.signals_path, mmap_mode="r")
    if signals.ndim != 3 or signals.shape[1] != len(CANONICAL_LEADS):
        raise ValueError(f"masked-unet training signals must have shape (N,12,T): {signals.shape}")
    parameters = dict(candidate.parameters)
    max_records = int(signals.shape[0])
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(parameters["deterministic"]))
    device = torch.device(device_name)

    # Release normalization is fitted once on every folds-1--7 training record
    # and then shared by every candidate/seed.
    if normalization_scale is None:
        subset = np.asarray(signals[:max_records], dtype=np.float32)
        scale = np.percentile(np.abs(subset), 95, axis=(0, 2)).astype(np.float32)
        scale = np.clip(scale, 0.05, None)
        del subset
    else:
        scale = np.asarray(normalization_scale, dtype=np.float32)
        if scale.shape != (len(CANONICAL_LEADS),) or not np.isfinite(scale).all():
            raise ValueError("normalization_scale must be finite with shape (12,)")
        if np.any(scale < 0.05):
            raise ValueError("normalization_scale must be clipped at 0.05 mV")
    width = int(parameters["width"])
    network = _build_unet(width=width).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    panel = tuple(tuple(value) for value in deep_configuration_panel())
    lead_index = {lead: index for index, lead in enumerate(CANONICAL_LEADS)}

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return max_records

        def __getitem__(self, index: int):
            return torch.as_tensor(np.array(signals[index], dtype=np.float32, copy=True))

    loader = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(parameters["num_workers"]),
        drop_last=False,
    )
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)[None, :, None]
    mask_rng = np.random.default_rng(int(seed))
    patience = int(parameters["early_stopping_patience"])
    max_epochs = int(parameters["max_epochs"])
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history = []
    stopped_epoch = max_epochs
    runtime = _masked_unet_runtime(network, scale, device)
    try:
        for epoch in range(1, max_epochs + 1):
            network.train()
            training_loss_sum = 0.0
            training_batches = 0
            for target in loader:
                target = target.to(device=device, dtype=torch.float32) / scale_tensor
                batch_masks = np.zeros((target.shape[0], 12), dtype=np.float32)
                for row in range(target.shape[0]):
                    observed = panel[int(mask_rng.integers(0, len(panel)))]
                    batch_masks[row, [lead_index[lead] for lead in observed]] = 1.0
                observed_mask = torch.as_tensor(batch_masks, device=device)[:, :, None]
                observed_mask = observed_mask.expand(-1, -1, target.shape[-1])
                network_input = torch.cat([target * observed_mask, observed_mask], dim=1)
                prediction = network(network_input)
                missing = 1.0 - observed_mask
                loss = (((prediction - target) ** 2) * missing).sum() / missing.sum().clamp_min(1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                training_loss_sum += float(loss.detach().cpu())
                training_batches += 1
            monitor = _early_stop_fold8_score(
                runtime,
                evaluation_records,
                configurations,
                seed=int(seed),
                epoch=epoch,
                segments=segments,
                training_predictors=training_predictors,
            )
            improved = bool(monitor < best_score)
            if improved:
                best_score = monitor
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_masked_mse": training_loss_sum / max(training_batches, 1),
                    "fold8_patient_balanced_log_rmse_mv": monitor,
                    "best_so_far_log_rmse_mv": best_score,
                    "stale_epochs": stale_epochs,
                    "is_strict_improvement": improved,
                }
            )
            if stale_epochs >= patience:
                stopped_epoch = epoch
                break
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"{candidate.candidate_id}/seed-{seed} produced no best state")
    network.load_state_dict(best_state)
    checkpoint = (
        output_dir
        / "checkpoints"
        / "masked-unet"
        / candidate.candidate_id
        / f"seed-{seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "model": network.state_dict(),
            "scale": scale,
            "seed": int(seed),
            "width": width,
            "train_manifest_sha256": train_manifest.signals_sha256,
            "training_record_ids_sha256": train_manifest.record_ids_sha256,
            "training_patient_ids_sha256": train_manifest.patient_ids_sha256,
            "training_inclusion_sha256": train_manifest.training_inclusion_sha256,
            "panel_size": len(panel),
            "candidate_id": candidate.candidate_id,
            "candidate_parameters": parameters,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "fold8_monitor_history": history,
        },
        checkpoint_temporary,
    )
    checkpoint_temporary.replace(checkpoint)
    audit = {
        "candidate_id": candidate.candidate_id,
        "model_seed": int(seed),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_monitor_log_rmse_mv": best_score,
        "monitor": (
            "fold8 patient-balanced log(RMSE_mV), one frozen distributed panel mask per record"
        ),
        "history": history,
        "checkpoint_path": checkpoint.resolve().relative_to(output_dir.resolve()).as_posix(),
        "checkpoint_sha256": lineage.artifact_sha256(checkpoint),
    }
    if audit_sink is not None:
        audit_sink(audit)
    frames = _evaluate_fitted_masked_unet_candidate(
        runtime,
        candidate,
        seed=int(seed),
        best_epoch=best_epoch,
        checkpoint=checkpoint,
        evaluation_records=evaluation_records,
        configurations=configurations,
        segments=segments,
        training_predictors=training_predictors,
        output_dir=output_dir,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        metric_sink=metric_sink,
        is_complete=is_complete,
    )
    return frames, audit


def early_stop_trace_frame(
    audits: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    split_sha256: str,
    fold8_records_sha256: str,
    configuration_panel_sha256: str,
) -> pd.DataFrame:
    """Materialise the independently verifiable patience-8 training trace."""

    rows = []
    for audit in audits:
        for item in audit["history"]:
            rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "cohort": "PTB-XL",
                    "train_partition": "folds1-7/train",
                    "partition": "fold8/tune",
                    "manifest_sha256": manifest_sha256,
                    "split_sha256": split_sha256,
                    "fold8_records_sha256": fold8_records_sha256,
                    "configuration_panel_sha256": configuration_panel_sha256,
                    "candidate_id": str(audit["candidate_id"]),
                    "model_seed": int(audit["model_seed"]),
                    "epoch": int(item["epoch"]),
                    "monitor_log_rmse_mv": float(
                        item["fold8_patient_balanced_log_rmse_mv"]
                    ),
                    "best_so_far_log_rmse_mv": float(
                        item["best_so_far_log_rmse_mv"]
                    ),
                    "stale_epochs": int(item["stale_epochs"]),
                    "is_strict_improvement": bool(item["is_strict_improvement"]),
                    "best_epoch": int(audit["best_epoch"]),
                    "stopped_epoch": int(audit["stopped_epoch"]),
                    "checkpoint_path": str(audit["checkpoint_path"]),
                    "checkpoint_sha256": str(audit["checkpoint_sha256"]),
                }
            )
    frame = pd.DataFrame(rows, columns=list(TRACE_COLUMNS))
    if frame.empty:
        raise ValueError("masked-unet early-stop trace is empty")
    return frame


def _normalization_scale_from_memmap(signals: np.ndarray) -> np.ndarray:
    """Compute the exact release percentile with one lead resident at a time."""

    if signals.ndim != 3 or signals.shape[1] != len(CANONICAL_LEADS):
        raise ValueError("normalization source must have shape (N,12,T)")
    values = []
    for lead_index in range(len(CANONICAL_LEADS)):
        lead = np.abs(
            np.asarray(signals[:, lead_index, :], dtype=np.float32)
        ).reshape(-1)
        values.append(float(np.percentile(lead, 95)))
        del lead
    scale = np.asarray(values, dtype=np.float32)
    return np.clip(scale, 0.05, None)


def _unet_audit_path(output_dir: Path, candidate_id: str, seed: int) -> Path:
    return output_dir / "training-audits" / "masked-unet" / candidate_id / f"seed-{seed}.json"


def _remove_orphan_training_workdirs(output_dir: Path) -> None:
    """Remove only process-private memmap workdirs left by an interrupted run."""

    root = output_dir.resolve()
    for path in sorted(root.glob(".ecgcert-candidates-*")):
        resolved = path.resolve()
        if resolved.parent != root or not resolved.name.startswith(".ecgcert-candidates-"):
            raise ValueError("unsafe candidate training workdir cleanup target")
        if not resolved.is_dir():
            raise ValueError("candidate training workdir residue is not a directory")
        shutil.rmtree(resolved)


def _write_unet_audit(path: Path, audit: Mapping[str, Any]) -> None:
    _atomic_json(path, dict(audit))


def _load_unet_audit(path: Path, candidate: Candidate, seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("candidate_id") != candidate.candidate_id
        or int(value.get("model_seed", -1)) != int(seed)
        or not isinstance(value.get("history"), list)
    ):
        raise ValueError("masked U-Net training audit identity is invalid")
    return dict(value)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    validate_release_arguments(arguments)
    _require_parquet_engine()
    manifest = load_ptbxl_manifest(arguments.manifest, release=arguments.release)
    requested_train_ids = manifest.record_ids("train", arguments.max_records)
    training_inclusion: TrainingInclusion | None = None
    if arguments.training_inclusion is not None:
        training_inclusion = load_training_inclusion(
            arguments.training_inclusion,
            source_manifest_path=arguments.manifest,
            source_manifest_sha256=manifest.manifest_sha256,
            split_sha256=manifest.split_sha256,
            expected_record_ids=requested_train_ids,
            expected_records=manifest.records,
            rate_hz=arguments.rate,
            segments=arguments.segments,
            delineator=arguments.delineator,
            configuration_panel_sha256=configuration_panel_sha256(),
        )
        train_ids = training_inclusion.included_record_ids
    else:
        train_ids = requested_train_ids
    tune_ids = manifest.record_ids("tune", arguments.max_records)
    # Intentionally verify and materialise only training and fold-8 records.
    verification_ids = requested_train_ids + tune_ids
    _verify_manifest_files(manifest, verification_ids, rate=arguments.rate)
    db = PTBXL(manifest.root)
    _validate_database_identity(db, manifest, verification_ids)
    configurations: tuple[tuple[str, ...], ...] = deep_configuration_panel()
    if arguments.max_configurations is not None:
        configurations = configurations[: arguments.max_configurations]
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_records, evaluation_audit = _load_evaluation_records(
        db,
        manifest,
        tune_ids,
        rate=arguments.rate,
        segments=arguments.segments,
        delineator=arguments.delineator,
        partition="fold8/tune",
    )
    fold8_case_sha256 = evaluation_records_sha256(
        evaluation_records, segments=arguments.segments
    )
    panel_sha256 = configuration_panel_sha256(configurations)
    store = CandidateMetricStore(
        output_dir,
        identity={
            "schema_version": CANDIDATE_METRIC_INVENTORY_SCHEMA_VERSION,
            "manifest_sha256": manifest.manifest_sha256,
            "split_sha256": manifest.split_sha256,
            "train_role_ids_sha256": lineage.canonical_sha256(
                list(requested_train_ids)
            ),
            "training_inclusion_sha256": (
                training_inclusion.inclusion_sha256
                if training_inclusion is not None
                else "not_applicable"
            ),
            "training_record_ids_sha256": (
                training_inclusion.record_ids_sha256
                if training_inclusion is not None
                else "not_applicable"
            ),
            "training_patient_ids_sha256": (
                training_inclusion.patient_ids_sha256
                if training_inclusion is not None
                else "not_applicable"
            ),
            "tune_role_ids_sha256": lineage.canonical_sha256(list(tune_ids)),
            "fold8_records_sha256": fold8_case_sha256,
            "configuration_panel_sha256": panel_sha256,
            "rate_hz": int(arguments.rate),
            "segments": list(arguments.segments),
            "delineator": arguments.delineator,
            "candidate_grid": {
                method: [asdict(candidate) for candidate in candidates]
                for method, candidates in candidate_grid().items()
            },
            "tuning_seeds": list(TUNING_SEEDS),
        },
        configurations=configurations,
    )
    _remove_orphan_training_workdirs(output_dir)
    unet_audit = []
    with TemporaryDirectory(prefix=".ecgcert-candidates-", dir=output_dir) as temporary:
        predictor_path = output_dir / TRAINING_PREDICTORS_FILENAME
        if training_inclusion is not None:
            train_manifest = training_inclusion.materialize_signals(
                db,
                manifest.records,
                Path(temporary) / "train_signals.npy",
            )
            if predictor_path.exists():
                if lineage.artifact_sha256(predictor_path) != training_inclusion.predictors_sha256:
                    raise ValueError("resumed candidate predictors disagree with training inclusion")
            else:
                training_inclusion.copy_predictors(predictor_path)
            training_predictors = pd.read_parquet(predictor_path)
            training_audit = dict(training_inclusion.audit)
        else:
            train_manifest, training_predictors, training_audit = (
                _materialize_training_manifest(
                    db,
                    manifest,
                    train_ids,
                    rate=arguments.rate,
                    work_dir=Path(temporary),
                    segments=arguments.segments,
                    delineator=arguments.delineator,
                    configurations=configurations,
                )
            )
            training_predictors.to_parquet(
                predictor_path, index=False, compression="zstd"
            )
        predictor_lookup = training_predictor_lookup(training_predictors)
        mean, scatter, sample_count = _streaming_training_moments(train_manifest)
        training_signals = np.load(train_manifest.signals_path, mmap_mode="r")
        normalization_scale = _normalization_scale_from_memmap(training_signals)
        del training_signals
        for method in ("lowrank", "ridge"):
            evaluate_linear_candidate_grid(
                method,
                mean=mean,
                scatter=scatter,
                sample_count=sample_count,
                evaluation_records=evaluation_records,
                configurations=configurations,
                segments=arguments.segments,
                training_predictors=predictor_lookup,
                output_dir=output_dir,
                manifest_sha256=manifest.manifest_sha256,
                split_sha256=manifest.split_sha256,
                metric_sink=store.write_frame,
                is_complete=store.is_complete,
            )
        for candidate in candidate_grid()["masked-unet"]:
            for seed in TUNING_SEEDS:
                audit_path = _unet_audit_path(output_dir, candidate.candidate_id, seed)
                if audit_path.exists():
                    audit = _load_unet_audit(audit_path, candidate, seed)
                    if not store.seed_complete("masked-unet", candidate.candidate_id, seed):
                        runtime, checkpoint = _load_masked_unet_candidate_checkpoint(
                            candidate,
                            seed=seed,
                            audit=audit,
                            train_manifest=train_manifest,
                            output_dir=output_dir,
                            device_name=arguments.device,
                        )
                        _evaluate_fitted_masked_unet_candidate(
                            runtime,
                            candidate,
                            seed=seed,
                            best_epoch=int(audit["best_epoch"]),
                            checkpoint=checkpoint,
                            evaluation_records=evaluation_records,
                            configurations=configurations,
                            segments=arguments.segments,
                            training_predictors=predictor_lookup,
                            output_dir=output_dir,
                            manifest_sha256=manifest.manifest_sha256,
                            split_sha256=manifest.split_sha256,
                            metric_sink=store.write_frame,
                            is_complete=store.is_complete,
                        )
                else:
                    if any(
                        store.is_complete(
                            "masked-unet", candidate.candidate_id, seed, configuration
                        )
                        for configuration in configurations
                    ):
                        raise ValueError(
                            "masked U-Net metric shards exist without their training audit"
                        )
                    _, audit = fit_masked_unet_candidate(
                        candidate,
                        seed=seed,
                        train_manifest=train_manifest,
                        evaluation_records=evaluation_records,
                        configurations=configurations,
                        segments=arguments.segments,
                        training_predictors=predictor_lookup,
                        output_dir=output_dir,
                        manifest_sha256=manifest.manifest_sha256,
                        split_sha256=manifest.split_sha256,
                        device_name=arguments.device,
                        normalization_scale=normalization_scale,
                        metric_sink=store.write_frame,
                        audit_sink=lambda value, path=audit_path: _write_unet_audit(
                            path, value
                        ),
                        is_complete=store.is_complete,
                    )
                unet_audit.append(audit)
        training_signal_sha256 = train_manifest.signals_sha256

    metrics_path = store.finalize()
    scan = scan_candidate_metrics_parquet(
        metrics_path,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
        expected_n_configurations=len(configurations),
    )
    metrics = scan.patient_rows
    trace = early_stop_trace_frame(
        unet_audit,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
        fold8_records_sha256=fold8_case_sha256,
        configuration_panel_sha256=panel_sha256,
    )
    _verify_candidate_checkpoints(metrics, metrics_path)
    _validate_early_stop_trace(
        trace,
        metrics,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
        fold8_records_sha256=fold8_case_sha256,
        configuration_panel_sha256=panel_sha256,
    )
    trace_path = output_dir / EARLY_STOP_TRACE_FILENAME
    trace.to_parquet(trace_path, index=False, compression="zstd")
    store.cleanup_staging()
    summary = {
        "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "early_stop_trace_schema_version": TRACE_SCHEMA_VERSION,
        "status": "complete",
        "train_partition": "folds1-7/train",
        "evaluation_partition": "fold8/tune",
        "holdout_partitions_loaded": [],
        "manifest": {
            "path": str(arguments.manifest.resolve()),
            "sha256": manifest.manifest_sha256,
            "split_sha256": manifest.split_sha256,
        },
        "rate_hz": int(arguments.rate),
        "segments": list(arguments.segments),
        "delineator": arguments.delineator,
        "configuration_panel_sha256": panel_sha256,
        "n_configurations": len(configurations),
        "n_train_records": len(requested_train_ids),
        "n_train_records_included": training_audit["summary"]["n_included"],
        "n_train_records_excluded": training_audit["summary"]["n_excluded"],
        "n_tune_records_requested": len(tune_ids),
        "train_signals_sha256": training_signal_sha256,
        "training_inclusion_sha256": (
            training_inclusion.inclusion_sha256
            if training_inclusion is not None
            else "not_applicable"
        ),
        "training_inclusion_file_sha256": (
            lineage.artifact_sha256(training_inclusion.path)
            if training_inclusion is not None
            else "not_applicable"
        ),
        "training_record_ids_sha256": (
            training_inclusion.record_ids_sha256
            if training_inclusion is not None
            else "not_applicable"
        ),
        "training_patient_ids_sha256": (
            training_inclusion.patient_ids_sha256
            if training_inclusion is not None
            else "not_applicable"
        ),
        "fold8_records_sha256": fold8_case_sha256,
        "evaluation_audit": evaluation_audit,
        "training_audit": training_audit,
        "candidate_grid": {
            method: [asdict(candidate) for candidate in candidates]
            for method, candidates in candidate_grid().items()
        },
        "masked_unet_early_stopping": unet_audit,
        "candidate_metric_scan": {
            "mode": "bounded_parquet_row_groups",
            "n_rows": scan.n_rows,
            "n_row_groups": scan.n_row_groups,
            "n_configurations": len(scan.configurations),
            "cell_reference_sha256": scan.cell_reference_sha256,
            "resume_supported": True,
        },
        "release": bool(arguments.release),
        "subsampled": arguments.max_records is not None
        or arguments.max_configurations is not None,
        "artifacts": {
            "candidate_metrics": {
                "path": CANDIDATE_METRICS_FILENAME,
                "sha256": lineage.artifact_sha256(metrics_path),
                "n_rows": int(scan.n_rows),
            },
            "candidate_metrics_inventory": {
                "path": CANDIDATE_METRIC_INVENTORY_FILENAME,
                "sha256": lineage.artifact_sha256(store.inventory_path),
            },
            "early_stop_trace": {
                "path": EARLY_STOP_TRACE_FILENAME,
                "sha256": lineage.artifact_sha256(trace_path),
                "n_rows": int(len(trace)),
            },
            "training_predictors": {
                "path": TRAINING_PREDICTORS_FILENAME,
                "sha256": lineage.artifact_sha256(predictor_path),
            },
        },
    }
    summary_path = output_dir / CANDIDATE_BUNDLE_FILENAME
    _atomic_json(summary_path, summary)
    summary["summary_sha256"] = lineage.artifact_sha256(summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = run(arguments)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"reconstruction candidate grid failed closed: {exc}") from exc
    print(
        f"[candidate-grid] {summary['artifacts']['candidate_metrics']['n_rows']} "
        f"real fold-8 rows -> {arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
