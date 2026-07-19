"""Build rank-robust, target-specific recoverability maps for ICASSP 2027.

This is the submission-path replacement for ``recoverability_maps.py``.  It fits
the spatial model on PTB-XL folds 1--7, freezes the only Gaussian observation
regularizer on fold 8, and then evaluates all 255 independent-lead subsets under
patient-cluster bootstrap uncertainty.  It never reads or writes ``results/``.
Each completed wave segment is hash-committed before the next begins, allowing
QRS/ST/T execution to resume without replaying earlier 2,000-draw segments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.data.ptbxl import SUPERCLASSES
from ecgcert.physics import LEADS, LEAD_INDEX, fit_spatial_subspace, observed_dipole
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    SEGMENT_SAMPLING_ALGORITHM,
    SEGMENT_SAMPLING_SEED,
    SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.recoverability import (
    BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
    BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
    BOOTSTRAP_REPLAY_SCHEMA_VERSION,
    REGULARIZATION_GRID_MV2,
    bootstrap_attempts_table,
    bootstrap_moments_table,
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
BOOTSTRAP_DRAW_SCHEMA_VERSION = "robust-recoverability-bootstrap-draw-v3"
BOOTSTRAP_AUDIT_SCHEMA_VERSION = "robust-recoverability-bootstrap-audit-v3"
SENSITIVITY_CHOICES = (
    "p-wave",
    "100hz",
    "delineator",
    "raw12",
    "sample-cap",
    "diagnosis",
)
MAX_METRIC_WORKERS = 10
SEGMENT_SAMPLING_SCHEMA_VERSION = "robust-map-segment-timepoint-sampling-v1"
TRAIN_SAMPLING_NAMESPACE = "PTB-XL/folds1-7/spatial-map-fit"
TUNE_SAMPLING_NAMESPACE = "PTB-XL/fold8/regularization-tuning"


def _segment_sampling_config(*, cap_per_record: int, seed: int) -> dict[str, Any]:
    """Return the complete, hashable timepoint-sampling contract."""

    return {
        "schema_version": SEGMENT_SAMPLING_SCHEMA_VERSION,
        "active_cap_per_record_per_segment": int(cap_per_record),
        "preregistered_primary_cap_per_record_per_segment": (
            PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD
        ),
        "preregistered_sensitivity_cap_per_record_per_segment": (
            SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
        ),
        "algorithm": SEGMENT_SAMPLING_ALGORITHM,
        "base_seed": int(seed),
        "role_namespaces": {
            "train": TRAIN_SAMPLING_NAMESPACE,
            "tune": TUNE_SAMPLING_NAMESPACE,
        },
        "sampling_unit": "record_x_segment_timepoint",
        "replacement": False,
        "selection": "first_cap_of_full_keyed_pcg64_permutation",
        "returned_index_order": "ascending_time",
        "larger_cap_contains_primary_selection": True,
    }


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet is mandatory for ICASSP artifacts; install the locked pyarrow dependency"
        ) from exc
    temporary.replace(path)


class _AtomicParquetWriter:
    """Append bounded Arrow row groups and publish one atomic Zstandard artifact."""

    def __init__(self, path: Path) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - locked release env contains pyarrow.
            raise RuntimeError(
                "Parquet is mandatory for ICASSP artifacts; install the locked pyarrow dependency"
            ) from exc
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.temporary.unlink(missing_ok=True)
        self._pq = pq
        self._writer = None
        self._schema = None

    def write_table(self, table) -> None:
        table = table.replace_schema_metadata(None)
        if self._writer is None:
            self._schema = table.schema
            self._writer = self._pq.ParquetWriter(
                self.temporary,
                self._schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        self._writer.write_table(table, row_group_size=len(table))

    def write_frame(self, frame: pd.DataFrame) -> None:
        import pyarrow as pa

        self.write_table(pa.Table.from_pandas(frame, preserve_index=False))

    def close(self, *, publish: bool) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if publish:
            if not self.temporary.is_file():
                raise RuntimeError(f"no Parquet rows were written for {self.path.name}")
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)


ROBUST_MAP_INVENTORY_SCHEMA_VERSION = "robust-map-segment-inventory-v1"
ROBUST_MAP_INVENTORY_FILENAME = "robust_map.inventory.v1.json"
SEGMENT_ARTIFACT_FILENAMES = {
    "rank_path": "rank_path.parquet",
    "map_cells": "map_cells.parquet",
    "bootstrap_draws": "bootstrap_draws.parquet",
    "bootstrap_patients": "bootstrap_patients.parquet",
    "bootstrap_multiplicities": "bootstrap_multiplicities.parquet",
    "bootstrap_audit": "bootstrap_audit.parquet",
    "bootstrap_moments": "bootstrap_moments.parquet",
    "bootstrap_attempts": "bootstrap_attempts.parquet",
}


class RobustMapSegmentStore:
    """Persist completed wave segments so an eight-hour map run can resume."""

    def __init__(
        self,
        output_dir: Path,
        *,
        identity: Mapping[str, Any],
        segments: Sequence[str],
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.inventory_path = self.output_dir / ROBUST_MAP_INVENTORY_FILENAME
        self.staging_dir = self.output_dir / ".robust_map.staging"
        self.identity = dict(identity)
        self.identity_sha256 = lineage.canonical_sha256(self.identity)
        self.segments = tuple(str(segment) for segment in segments)
        if not self.segments or len(set(self.segments)) != len(self.segments):
            raise ValueError("robust-map segments must be non-empty and unique")
        self._descriptors: dict[str, dict[str, Any]] = {}
        self.status = "writing"
        self.final_artifacts: dict[str, dict[str, Any]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.inventory_path.exists():
            self._load_inventory()
        elif any(self.output_dir.iterdir()):
            raise FileExistsError(
                "robust-map output is non-empty but has no authenticated segment inventory: "
                f"{self.output_dir}"
            )
        else:
            self.staging_dir.mkdir()
            self._write_inventory()
        if self.status == "writing":
            self.staging_dir.mkdir(exist_ok=True)
            self._adopt_segment_summaries()

    def _segment_dir(self, segment: str) -> Path:
        if segment not in self.segments:
            raise ValueError(f"unexpected robust-map segment {segment!r}")
        return self.staging_dir / f"{self.segments.index(segment):02d}-{segment.lower()}"

    def segment_artifact(self, segment: str, name: str) -> Path:
        if name not in SEGMENT_ARTIFACT_FILENAMES:
            raise ValueError(f"unexpected robust-map segment artifact {name!r}")
        directory = self._segment_dir(segment)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / SEGMENT_ARTIFACT_FILENAMES[name]

    def _inventory_value(self) -> dict[str, Any]:
        return {
            "schema_version": ROBUST_MAP_INVENTORY_SCHEMA_VERSION,
            "status": self.status,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "expected_segments": list(self.segments),
            "completed_segments": [
                {"segment": segment, **self._descriptors[segment]}
                for segment in self.segments
                if segment in self._descriptors
            ],
            "final_artifacts": self.final_artifacts,
        }

    def _write_inventory(self) -> None:
        _write_json(self.inventory_path, self._inventory_value())

    def _validate_artifacts(
        self, artifacts: Mapping[str, Any], *, require_all: bool = True
    ) -> None:
        if require_all and set(artifacts) != set(SEGMENT_ARTIFACT_FILENAMES):
            raise ValueError("robust-map segment artifact inventory is incomplete")
        for name, descriptor in artifacts.items():
            if name not in SEGMENT_ARTIFACT_FILENAMES or not isinstance(
                descriptor, Mapping
            ):
                raise ValueError("robust-map segment artifact descriptor is invalid")
            relative = Path(str(descriptor.get("path", "")))
            digest = descriptor.get("sha256")
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError("robust-map segment artifact path/hash is invalid")
            path = (self.output_dir / relative).resolve()
            if self.output_dir not in path.parents or not path.is_file():
                raise ValueError("robust-map segment artifact is missing or escapes its bundle")
            if lineage.artifact_sha256(path) != digest:
                raise ValueError("robust-map segment artifact SHA-256 changed")

    def _validate_final_artifacts(self, artifacts: Mapping[str, Any]) -> None:
        if not artifacts:
            raise ValueError("robust-map final artifact inventory is empty")
        for name, descriptor in artifacts.items():
            if not isinstance(name, str) or not isinstance(descriptor, Mapping):
                raise ValueError("robust-map final artifact descriptor is invalid")
            relative = Path(str(descriptor.get("path", "")))
            digest = descriptor.get("sha256")
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError("robust-map final artifact path/hash is invalid")
            path = (self.output_dir / relative).resolve()
            if self.output_dir not in path.parents or not path.is_file():
                raise ValueError("robust-map final artifact is missing or escapes its bundle")
            if lineage.artifact_sha256(path) != digest:
                raise ValueError("robust-map final artifact SHA-256 changed")

    def _load_inventory(self) -> None:
        value = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != ROBUST_MAP_INVENTORY_SCHEMA_VERSION:
            raise ValueError("robust-map segment inventory schema is invalid")
        if value.get("identity") != self.identity or value.get(
            "identity_sha256"
        ) != self.identity_sha256:
            raise ValueError("robust-map resume identity changed")
        if value.get("expected_segments") != list(self.segments):
            raise ValueError("robust-map resume segment order changed")
        status = value.get("status")
        if status not in {"writing", "complete"}:
            raise ValueError("robust-map segment inventory status is invalid")
        for raw in value.get("completed_segments", []):
            if not isinstance(raw, Mapping):
                raise ValueError("robust-map segment descriptor is invalid")
            segment = str(raw.get("segment", ""))
            if segment not in self.segments or segment in self._descriptors:
                raise ValueError("robust-map segment descriptor is duplicated or unexpected")
            descriptor = {key: raw[key] for key in raw if key != "segment"}
            artifacts = descriptor.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError("robust-map segment descriptor lacks artifacts")
            if status == "writing" and descriptor.get("staging_retained") is not True:
                raise ValueError("writing robust-map inventory must retain segment staging")
            if descriptor.get("staging_retained") is True:
                self._validate_artifacts(artifacts)
            self._descriptors[segment] = descriptor
        self.status = str(status)
        finals = value.get("final_artifacts", {})
        if not isinstance(finals, Mapping):
            raise ValueError("robust-map final artifact inventory is invalid")
        self.final_artifacts = {str(key): dict(item) for key, item in finals.items()}
        if self.status == "complete":
            if set(self._descriptors) != set(self.segments) or not self.final_artifacts:
                raise ValueError("complete robust-map inventory is missing segments or outputs")
            self._validate_final_artifacts(self.final_artifacts)

    def _segment_summary_path(self, segment: str) -> Path:
        return self._segment_dir(segment) / "segment-summary.v1.json"

    def _descriptor_from_summary(self, segment: str) -> dict[str, Any]:
        path = self._segment_summary_path(segment)
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != ROBUST_MAP_INVENTORY_SCHEMA_VERSION
            or value.get("identity_sha256") != self.identity_sha256
            or value.get("segment") != segment
            or not isinstance(value.get("artifacts"), Mapping)
            or not isinstance(value.get("metadata"), Mapping)
        ):
            raise ValueError("robust-map segment summary identity is invalid")
        self._validate_artifacts(value["artifacts"])
        return {
            "artifacts": dict(value["artifacts"]),
            "metadata": dict(value["metadata"]),
            "segment_summary": {
                "path": path.relative_to(self.output_dir).as_posix(),
                "sha256": lineage.artifact_sha256(path),
            },
            "staging_retained": True,
        }

    def _adopt_segment_summaries(self) -> None:
        changed = False
        for segment in self.segments:
            path = self._segment_summary_path(segment)
            if not path.is_file():
                continue
            descriptor = self._descriptor_from_summary(segment)
            if segment in self._descriptors:
                if descriptor != self._descriptors[segment]:
                    raise ValueError("robust-map recorded segment summary changed")
                continue
            self._descriptors[segment] = descriptor
            changed = True
        if changed:
            self._write_inventory()

    def is_complete(self, segment: str) -> bool:
        return segment in self._descriptors

    def descriptor(self, segment: str) -> Mapping[str, Any]:
        if segment not in self._descriptors:
            raise ValueError(f"robust-map segment is incomplete: {segment}")
        return self._descriptors[segment]

    def commit_segment(self, segment: str, *, metadata: Mapping[str, Any]) -> None:
        if self.status != "writing" or segment in self._descriptors:
            raise ValueError("robust-map segment cannot be committed twice")
        artifacts = {
            name: {
                "path": self.segment_artifact(segment, name)
                .relative_to(self.output_dir)
                .as_posix(),
                "sha256": lineage.artifact_sha256(
                    self.segment_artifact(segment, name)
                ),
            }
            for name in SEGMENT_ARTIFACT_FILENAMES
        }
        self._validate_artifacts(artifacts)
        summary_path = self._segment_summary_path(segment)
        _write_json(
            summary_path,
            {
                "schema_version": ROBUST_MAP_INVENTORY_SCHEMA_VERSION,
                "identity_sha256": self.identity_sha256,
                "segment": segment,
                "artifacts": artifacts,
                "metadata": dict(metadata),
            },
        )
        self._descriptors[segment] = self._descriptor_from_summary(segment)
        self._write_inventory()

    def merge_parquet(self, name: str, destination: Path) -> None:
        if set(self._descriptors) != set(self.segments):
            raise ValueError("cannot merge incomplete robust-map segments")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("robust-map merge requires locked pyarrow") from exc
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        writer = None
        schema = None
        try:
            for segment in self.segments:
                descriptor = self._descriptors[segment]["artifacts"][name]
                source = self.output_dir / str(descriptor["path"])
                parquet = pq.ParquetFile(source)
                for row_group in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(row_group).replace_schema_metadata(None)
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            temporary,
                            schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    elif table.schema != schema:
                        table = table.cast(schema)
                    writer.write_table(table, row_group_size=len(table))
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"robust-map merge wrote no rows for {name}")
        temporary.replace(destination)

    def mark_complete(self, artifact_paths: Mapping[str, str]) -> None:
        if set(self._descriptors) != set(self.segments):
            raise ValueError("cannot complete robust-map inventory with missing segments")
        self.final_artifacts = {
            name: {
                "path": relative,
                "sha256": lineage.artifact_sha256(self.output_dir / relative),
            }
            for name, relative in artifact_paths.items()
            if name != "segment_inventory"
        }
        self._validate_final_artifacts(self.final_artifacts)
        self.status = "complete"
        self._write_inventory()

    def cleanup_staging(self) -> None:
        if self.status != "complete":
            raise ValueError("cannot clean incomplete robust-map staging")
        for segment in self.segments:
            descriptor = self._descriptors[segment]
            directory = self._segment_dir(segment)
            if descriptor.get("staging_retained") is True:
                expected = {
                    self.output_dir / str(item["path"])
                    for item in descriptor["artifacts"].values()
                }
                expected.add(
                    self.output_dir / str(descriptor["segment_summary"]["path"])
                )
                if directory.exists():
                    actual = set(directory.iterdir())
                    if actual != expected:
                        raise ValueError("unexpected file in robust-map segment staging")
                    for path in sorted(actual):
                        path.unlink()
                    directory.rmdir()
                descriptor["artifacts"] = {
                    name: {**dict(item), "path": None}
                    for name, item in descriptor["artifacts"].items()
                }
                descriptor["segment_summary"] = {
                    **dict(descriptor["segment_summary"]),
                    "path": None,
                }
                descriptor["staging_retained"] = False
        if self.staging_dir.exists():
            self.staging_dir.rmdir()
        self._write_inventory()


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


@dataclass(frozen=True)
class _StackedSpatialModels:
    """Read-only model arrays for one rank in bootstrap-index order."""

    models: tuple[Any, ...]
    matrices: np.ndarray
    covariances: np.ndarray
    covariance_norms: np.ndarray
    lead_norms: np.ndarray
    rank: int


def _stack_spatial_models(models: Sequence[Any]) -> _StackedSpatialModels:
    models = tuple(models)
    if not models:
        raise ValueError("cannot stack an empty spatial-model batch")
    rank = int(models[0].rank)
    keys = {(int(model.rank), str(model.basis_variant)) for model in models}
    if len(keys) != 1:
        raise ValueError("a vectorized metric batch must share one rank/basis variant")
    matrices = np.stack([np.asarray(model.M, dtype=float) for model in models])
    covariances = np.stack(
        [np.asarray(model.covariance, dtype=float) for model in models]
    )
    if matrices.shape != (len(models), len(LEADS), rank) or covariances.shape != (
        len(models),
        rank,
        rank,
    ):
        raise ValueError("stacked spatial models have inconsistent matrix shapes")
    covariance_norms = np.asarray(
        [np.linalg.norm(covariance, ord=2) for covariance in covariances],
        dtype=float,
    )
    lead_norms = np.linalg.norm(matrices, axis=2)
    for array in (matrices, covariances, covariance_norms, lead_norms):
        array.setflags(write=False)
    return _StackedSpatialModels(
        models=models,
        matrices=matrices,
        covariances=covariances,
        covariance_norms=covariance_norms,
        lead_norms=lead_norms,
        rank=rank,
    )


def _batched_model_metrics(
    batch: _StackedSpatialModels,
    configuration: Sequence[str],
    observation_variance_mv2: float,
) -> dict[str, np.ndarray]:
    """Vectorized equivalent of :func:`_model_metrics` for one rank.

    The first axis is bootstrap index.  NumPy performs the small SVD, solve and
    eigendecomposition stacks in native code, replacing one Python call per
    model/configuration while retaining the scalar truncation and PSD rules.
    """

    variance = float(observation_variance_mv2)
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("observation_variance_mv2 must be finite and non-negative")
    observed_indices = np.asarray([LEAD_INDEX[lead] for lead in configuration], dtype=int)
    if (
        not observed_indices.size
        or len(np.unique(observed_indices)) != len(observed_indices)
    ):
        raise ValueError("configuration must contain unique observed leads")

    matrices = batch.matrices
    covariances = batch.covariances
    n_models, _n_leads, rank = matrices.shape
    observed = matrices[:, observed_indices, :]

    # This is the same full-precision observed-dipole SVD and threshold used by
    # observed_dipole(..., rcond=None), applied to every bootstrap model at once.
    left, singular_values, right_transpose = np.linalg.svd(
        observed,
        full_matrices=False,
    )
    thresholds = (
        np.finfo(float).eps
        * max(observed.shape[1:])
        * singular_values[:, 0]
    )
    retained = singular_values > thresholds[:, None]
    configuration_rank = retained.sum(axis=1).astype(np.int16)
    inverse_singular_values = np.zeros_like(singular_values)
    np.divide(
        1.0,
        singular_values,
        out=inverse_singular_values,
        where=retained,
    )
    right = np.swapaxes(right_transpose, 1, 2)
    pseudo_inverse = (right * inverse_singular_values[:, None, :]) @ np.swapaxes(
        left, 1, 2
    )
    observed_projector = (right * retained[:, None, :]) @ right_transpose

    unobserved = matrices @ (np.eye(rank)[None, :, :] - observed_projector)
    eta_normalized = np.divide(
        np.linalg.norm(unobserved, axis=2),
        batch.lead_norms,
        out=np.full((n_models, len(LEADS)), np.nan),
        where=batch.lead_norms > 1e-12,
    )
    reconstruction_gain = matrices @ pseudo_inverse
    kappa_per_target = np.linalg.norm(reconstruction_gain, axis=2)

    # SpatialSubspaceModel enforces orthonormal M columns.  Therefore the
    # non-zero singular values of M @ M_S^+ are exactly those of M_S^+, and the
    # spectral norm is the reciprocal of the smallest retained singular value.
    # This removes a second SVD without changing the kappa definition.
    kappa_global = np.zeros(n_models, dtype=float)
    positive_rank = configuration_rank > 0
    last_indices = np.maximum(configuration_rank.astype(np.int64) - 1, 0)
    kappa_global[positive_rank] = 1.0 / singular_values[
        np.arange(n_models)[positive_rank], last_indices[positive_rank]
    ]
    condition_number = np.ones(n_models, dtype=float)
    condition_number[positive_rank] = (
        singular_values[positive_rank, 0]
        / singular_values[
            np.arange(n_models)[positive_rank], last_indices[positive_rank]
        ]
    )

    # Gaussian posterior covariance, including the same PSD eigenvalue check and
    # clip as gaussian_posterior_covariance, is likewise evaluated as a stack.
    observed_covariance = observed @ covariances
    innovation = observed_covariance @ np.swapaxes(observed, 1, 2)
    innovation = 0.5 * (innovation + np.swapaxes(innovation, 1, 2))
    if variance:
        innovation = innovation + variance * np.eye(len(observed_indices))[None, :, :]
        gain_right = np.linalg.solve(innovation, observed_covariance)
    else:
        gain_right = np.linalg.pinv(
            innovation,
            rcond=1e-12,
            hermitian=True,
        ) @ observed_covariance
    posterior = covariances - (
        covariances @ np.swapaxes(observed, 1, 2) @ gain_right
    )
    posterior = 0.5 * (posterior + np.swapaxes(posterior, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(posterior)
    tolerances = 1e-10 * np.maximum(1.0, batch.covariance_norms)
    invalid = eigenvalues[:, 0] < -tolerances
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        raise FloatingPointError(
            "Gaussian posterior covariance is not positive semidefinite "
            f"for model batch index {first}"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    posterior = (eigenvectors * eigenvalues[:, None, :]) @ np.swapaxes(
        eigenvectors, 1, 2
    )
    lead_covariance = matrices @ posterior @ np.swapaxes(matrices, 1, 2)
    ambiguity = np.sqrt(
        np.clip(np.diagonal(lead_covariance, axis1=1, axis2=2), 0.0, None)
    )
    return {
        "ambiguity": ambiguity,
        "eta_normalized": eta_normalized,
        "kappa_per_target": kappa_per_target,
        "kappa_global": kappa_global,
        "configuration_rank": configuration_rank,
        "condition_number": condition_number,
    }


def _metric_worker_count(requested: int | None) -> int:
    """Resolve the DAG CPU allocation, bounded by the project-wide 10-worker cap."""

    raw: Any = os.environ.get("ECGCERT_NUM_WORKERS", "1") if requested is None else requested
    if isinstance(raw, bool):
        raise ValueError("metric workers must be an integer, not a boolean")
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric workers must be an integer") from exc
    if not 1 <= workers <= MAX_METRIC_WORKERS:
        raise ValueError(f"metric workers must be in [1, {MAX_METRIC_WORKERS}]")
    return workers


def _draw_rows_per_configuration(n_boot: int, n_ranks: int) -> int:
    """Exact raw-draw row-group size for one configuration."""

    return int(n_boot) * int(n_ranks) * len(LEADS)


def _metric_workspace_proxy_bytes(
    *,
    n_boot: int,
    ranks: Sequence[int],
    max_observed: int = 8,
) -> int:
    """Conservative per-worker ndarray workspace proxy for release planning.

    The proxy deliberately excludes immutable model stacks shared by all worker
    threads and the one configuration draw table owned by the writer thread.
    It upper-bounds retained result arrays plus the largest-rank SVD/posterior
    intermediates and is used as a regression guard against configuration-wide
    tensor materialization.
    """

    ranks = tuple(int(rank) for rank in ranks)
    if n_boot < 1 or not ranks or max_observed < 1:
        raise ValueError("workspace dimensions must be positive")
    q = max(ranks)
    b = int(n_boot)
    m = int(max_observed)
    # Retained outputs for every rank: three (B,12), plus three (B,) arrays.
    retained = len(ranks) * b * (3 * len(LEADS) + 3) * 8
    # Conservative largest-rank temporaries: M_S, U, s, Vt, pinv, projector,
    # unobserved, gain, covariance/innovation/gain-right/posterior/eigenvectors,
    # and a 12x12 lead covariance used for the ambiguity diagonal.
    largest = b * (
        (m * q)
        + (m * min(m, q))
        + min(m, q)
        + (min(m, q) * q)
        + (q * m)
        + (q * q)
        + (len(LEADS) * q)
        + (len(LEADS) * m)
        + (q * q)
        + (m * m)
        + (m * q)
        + (q * q)
        + (q * q)
        + (len(LEADS) * len(LEADS))
    ) * 8
    return int(retained + largest)


def _training_predictor_statistics(bank) -> tuple[np.ndarray, np.ndarray]:
    """Training-only target RMS and absolute correlation, computed once per bank."""

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
    return target_rms, np.abs(correlation)


def _training_predictors(bank, configuration: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Training-only target RMS and maximum target--observed correlation."""

    target_rms, absolute_correlation = _training_predictor_statistics(bank)
    observed_indices = [LEAD_INDEX[lead] for lead in configuration]
    return target_rms, np.max(absolute_correlation[:, observed_indices], axis=1)


def _ordered_parallel_map(
    function: Callable[[tuple[str, ...]], Any],
    values: Sequence[tuple[str, ...]],
    *,
    workers: int,
):
    """Yield a bounded parallel map in exact input order."""

    # Each Python worker owns one configuration.  Keep native BLAS at one thread
    # so the declared DAG CPU allocation is not multiplied by hidden BLAS teams.
    with threadpool_limits(limits=1, user_api="blas"):
        if workers == 1:
            for value in values:
                yield function(value)
            return
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="robust-map-metrics",
        ) as executor:
            # One worker-width chunk prevents completed later configurations from
            # accumulating behind a slower earlier configuration.  Raw-draw row
            # groups therefore remain bounded and are still published in input order.
            for start in range(0, len(values), workers):
                chunk = values[start : start + workers]
                yield from executor.map(function, chunk)


def summarize_model_bank(
    bank,
    configurations: Sequence[Sequence[str]],
    *,
    segment: str,
    observation_variance_mv2: float,
    confidence: float = 0.95,
    draw_sink: Callable[[pd.DataFrame], None] | None = None,
    metric_workers: int | None = None,
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
    if len(bank.basis_variants) != 1:
        raise ValueError("summarize_model_bank expects exactly one basis variant")
    configuration_values = tuple(tuple(value) for value in configurations)
    if not configuration_values:
        raise ValueError("summarize_model_bank requires at least one configuration")
    workers = _metric_worker_count(metric_workers)
    point_batches = {
        rank: _stack_spatial_models((point_by_rank[rank],)) for rank in bank.ranks
    }
    bootstrap_batches: dict[int, _StackedSpatialModels] = {}
    for rank in bank.ranks:
        models = []
        for bootstrap_index, group in enumerate(bank.bootstrap_models):
            matches = [model for model in group if model.rank == rank]
            if len(matches) != 1:
                raise ValueError(
                    f"bootstrap draw {bootstrap_index} does not contain exactly one rank-{rank} model"
                )
            models.append(matches[0])
        bootstrap_batches[rank] = _stack_spatial_models(models)
    training_target_rms, training_absolute_correlation = (
        _training_predictor_statistics(bank)
    )

    def evaluate_configuration(configuration: tuple[str, ...]):
        observed_indices = [LEAD_INDEX[lead] for lead in configuration]
        training_max_correlation = np.max(
            training_absolute_correlation[:, observed_indices], axis=1
        )
        point_metrics = {}
        boot_metrics = {}
        for rank in bank.ranks:
            point_batch = _batched_model_metrics(
                point_batches[rank],
                configuration,
                observation_variance_mv2,
            )
            point_metrics[rank] = {
                name: values[0] for name, values in point_batch.items()
            }
            boot_metrics[rank] = _batched_model_metrics(
                bootstrap_batches[rank],
                configuration,
                observation_variance_mv2,
            )
        return configuration, training_max_correlation, point_metrics, boot_metrics

    for (
        configuration,
        training_max_correlation,
        point_metrics,
        boot_metrics,
    ) in _ordered_parallel_map(
        evaluate_configuration,
        configuration_values,
        workers=workers,
    ):
        config_id = _configuration_id(configuration)

        if draw_sink is not None:
            target_values = np.repeat(np.asarray(LEADS, dtype=object), bank.n_boot)
            bootstrap_values = np.tile(np.arange(bank.n_boot, dtype=np.int32), len(LEADS))
            draw_frames = []
            for rank in bank.ranks:
                metrics = boot_metrics[rank]
                draw_frames.append(
                    pd.DataFrame(
                        {
                            "schema_version": BOOTSTRAP_DRAW_SCHEMA_VERSION,
                            "segment": segment,
                            "configuration": config_id,
                            "target": target_values,
                            "rank": np.full(len(target_values), rank, dtype=np.int8),
                            "basis_variant": bank.basis_variants[0],
                            "bootstrap_index": bootstrap_values,
                            "a_r_mv": metrics["ambiguity"].T.reshape(-1),
                            "eta_normalized": metrics["eta_normalized"].T.reshape(-1),
                            "kappa_target": metrics["kappa_per_target"].T.reshape(-1),
                            "kappa_global": np.tile(metrics["kappa_global"], len(LEADS)),
                            "configuration_rank": np.tile(
                                metrics["configuration_rank"], len(LEADS)
                            ),
                            "condition_number": np.tile(
                                metrics["condition_number"], len(LEADS)
                            ),
                        }
                    )
                )
            draw_sink(pd.concat(draw_frames, ignore_index=True))

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
    parser.add_argument(
        "--max-per-record",
        type=int,
        default=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    )
    parser.add_argument("--sampling-seed", type=int, default=SEGMENT_SAMPLING_SEED)
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
    elif arguments.sensitivity == "sample-cap":
        arguments.max_per_record = SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
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
        "max_per_record": PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        "sampling_seed": SEGMENT_SAMPLING_SEED,
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
        elif arguments.sensitivity == "sample-cap":
            expected["max_per_record"] = (
                SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
            )
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
    expected_sampling = _segment_sampling_config(
        cap_per_record=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        seed=SEGMENT_SAMPLING_SEED,
    )
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
        "bootstrap_evidence_schema_version": BOOTSTRAP_AUDIT_SCHEMA_VERSION,
        "bootstrap_draw_schema_version": BOOTSTRAP_DRAW_SCHEMA_VERSION,
        "bootstrap_replay_schema_version": BOOTSTRAP_REPLAY_SCHEMA_VERSION,
        "bootstrap_moments_schema_version": BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
        "bootstrap_attempt_schema_version": BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
        "segment_sampling": expected_sampling,
        "segment_sampling_sha256": lineage.canonical_sha256(expected_sampling),
    }
    mismatches = [key for key, expected in frozen.items() if value.get(key) != expected]
    if mismatches:
        raise RuntimeError(f"sensitivity requires a frozen primary map: {mismatches}")
    paths = value.get("artifacts")
    hashes = value.get("artifact_sha256")
    required = {
        "rank_path",
        "map_cells",
        "regularization_tuning",
        "patient_audit",
        "bootstrap_draws",
        "bootstrap_patients",
        "bootstrap_multiplicities",
        "bootstrap_audit",
        "bootstrap_moments",
        "bootstrap_attempts",
    }
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


def _write_bootstrap_design(
    *,
    bank,
    segment: str,
    patient_writer: _AtomicParquetWriter,
    multiplicity_writer: _AtomicParquetWriter,
    moments_writer: _AtomicParquetWriter | None = None,
    attempt_writer: _AtomicParquetWriter | None = None,
) -> None:
    """Persist compatible accepted draws plus complete replay evidence."""

    import pyarrow as pa

    n_patients = len(bank.patient_ids)
    patient_writer.write_table(
        pa.table(
            {
                "schema_version": pa.array(
                    [BOOTSTRAP_AUDIT_SCHEMA_VERSION] * n_patients, type=pa.string()
                ),
                "segment": pa.array([segment] * n_patients, type=pa.string()),
                "basis_variant": pa.array(
                    [bank.basis_variants[0]] * n_patients, type=pa.string()
                ),
                "patient_index": pa.array(np.arange(n_patients), type=pa.int32()),
                "patient_id": pa.array(
                    [str(patient_id) for patient_id in bank.patient_ids], type=pa.string()
                ),
            }
        )
    )

    accepted_multiplicities = np.asarray(bank.bootstrap_multiplicities)
    if (
        accepted_multiplicities.dtype.itemsize <= np.dtype(np.uint16).itemsize
        and n_patients <= np.iinfo(np.uint16).max
    ):
        accepted_multiplicities = accepted_multiplicities.astype(np.uint16, copy=False)
        multiplicity_type = pa.uint16()
    else:
        accepted_multiplicities = accepted_multiplicities.astype(np.uint32, copy=False)
        multiplicity_type = pa.uint32()
    values = pa.array(
        accepted_multiplicities.reshape(-1),
        type=multiplicity_type,
    )
    offsets = pa.array(
        np.arange(bank.n_boot + 1, dtype=np.int64) * n_patients,
        type=pa.int64(),
    )
    multiplicities = pa.LargeListArray.from_arrays(offsets, values)
    multiplicity_writer.write_table(
        pa.table(
            {
                "schema_version": pa.array(
                    [BOOTSTRAP_AUDIT_SCHEMA_VERSION] * bank.n_boot, type=pa.string()
                ),
                "segment": pa.array([segment] * bank.n_boot, type=pa.string()),
                "basis_variant": pa.array(
                    [bank.basis_variants[0]] * bank.n_boot, type=pa.string()
                ),
                "bootstrap_index": pa.array(np.arange(bank.n_boot), type=pa.int32()),
                "accepted": pa.array([True] * bank.n_boot, type=pa.bool_()),
                "multiplicities": multiplicities,
            }
        )
    )
    if (moments_writer is None) != (attempt_writer is None):
        raise ValueError("moments_writer and attempt_writer must be supplied together")
    if moments_writer is not None and attempt_writer is not None:
        moments_writer.write_table(bootstrap_moments_table(bank, segment=segment))
        attempt_writer.write_table(bootstrap_attempts_table(bank, segment=segment))


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
    if isinstance(arguments.sampling_seed, bool):
        raise SystemExit("--sampling-seed must be an integer")
    if arguments.rate not in {100, 500}:
        raise SystemExit("--rate must be one of the PTB-XL source rates: 100 or 500")
    os.environ["ECG_DELINEATOR"] = arguments.delineator

    if arguments.manifest is None:
        raise RuntimeError("--manifest is required; ad-hoc database splitting is forbidden")
    contract = load_ptbxl_manifest(arguments.manifest, release=arguments.release)
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
    sampling_config = _segment_sampling_config(
        cap_per_record=arguments.max_per_record,
        seed=arguments.sampling_seed,
    )
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
        seed=arguments.sampling_seed,
        sampling_namespace=TRAIN_SAMPLING_NAMESPACE,
    )
    tune_data, tune_audit = db.collect_all_segments_audited(
        tune_ids,
        rate=arguments.rate,
        max_per_record=arguments.max_per_record,
        max_records=arguments.max_records,
        seed=arguments.sampling_seed,
        sampling_namespace=TUNE_SAMPLING_NAMESPACE,
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

    output_dir = arguments.output_dir.resolve()
    configurations = all_independent_configurations()
    metric_workers = _metric_worker_count(None)
    metric_workspace_bytes = _metric_workspace_proxy_bytes(
        n_boot=arguments.n_bootstrap,
        ranks=RANK_GRID,
        max_observed=max(len(configuration) for configuration in configurations),
    )
    artifact_paths = {
        "rank_path": "rank_path.parquet",
        "map_cells": "map_cells.parquet",
        "regularization_tuning": "regularization_tuning.parquet",
        "patient_audit": "patient_audit.json",
        "bootstrap_draws": "bootstrap_draws.parquet",
        "bootstrap_patients": "bootstrap_patients.parquet",
        "bootstrap_multiplicities": "bootstrap_multiplicities.parquet",
        "bootstrap_audit": "bootstrap_audit.parquet",
        "bootstrap_moments": "bootstrap_moments.parquet",
        "bootstrap_attempts": "bootstrap_attempts.parquet",
        "segment_inventory": ROBUST_MAP_INVENTORY_FILENAME,
    }
    run_configuration = {
        "analysis_mode": arguments.mode,
        "sensitivity": arguments.sensitivity,
        "diagnosis_class": arguments.diagnosis_class,
        "population": arguments.population,
        "segments": list(arguments.segments),
        "rate_hz": arguments.rate,
        "delineator": arguments.delineator,
        "basis_variant": arguments.basis_variant,
        "ranks": list(RANK_GRID),
        "n_bootstrap": arguments.n_bootstrap,
        "bootstrap_seed": arguments.seed,
        "max_records": arguments.max_records,
        "segment_sampling": sampling_config,
        "observation_variance_mv2": observation_variance,
        "configuration_sha256": lineage.canonical_sha256(
            [list(configuration) for configuration in configurations]
        ),
    }
    run_configuration_sha256 = lineage.canonical_sha256(run_configuration)
    store = RobustMapSegmentStore(
        output_dir,
        identity={
            "schema_version": ROBUST_MAP_INVENTORY_SCHEMA_VERSION,
            "analysis_mode": arguments.mode,
            "sensitivity": arguments.sensitivity,
            "diagnosis_class": arguments.diagnosis_class,
            "population": arguments.population,
            "manifest_sha256": contract.manifest_sha256,
            "split_sha256": contract.split_sha256,
            "train_role_ids_sha256": lineage.canonical_sha256(list(train_ids)),
            "tune_role_ids_sha256": lineage.canonical_sha256(list(tune_ids)),
            "train_audit_sha256": train_audit.sha256(),
            "tune_audit_sha256": tune_audit.sha256(),
            "rate_hz": arguments.rate,
            "delineator": arguments.delineator,
            "basis_variant": arguments.basis_variant,
            "ranks": list(RANK_GRID),
            "n_bootstrap": arguments.n_bootstrap,
            "seed": arguments.seed,
            "segment_sampling": sampling_config,
            "segment_sampling_sha256": lineage.canonical_sha256(sampling_config),
            "run_configuration": run_configuration,
            "run_configuration_sha256": run_configuration_sha256,
            "observation_variance_mv2": observation_variance,
            "configuration_sha256": lineage.canonical_sha256(
                [list(configuration) for configuration in configurations]
            ),
        },
        segments=arguments.segments,
    )
    if store.status == "writing":
        for segment_index, segment in enumerate(arguments.segments):
            if store.is_complete(segment):
                continue
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
            draw_writer = _AtomicParquetWriter(
                store.segment_artifact(segment, "bootstrap_draws")
            )
            patient_writer = _AtomicParquetWriter(
                store.segment_artifact(segment, "bootstrap_patients")
            )
            multiplicity_writer = _AtomicParquetWriter(
                store.segment_artifact(segment, "bootstrap_multiplicities")
            )
            moments_writer = _AtomicParquetWriter(
                store.segment_artifact(segment, "bootstrap_moments")
            )
            attempt_writer = _AtomicParquetWriter(
                store.segment_artifact(segment, "bootstrap_attempts")
            )
            published = False
            try:
                _write_bootstrap_design(
                    bank=bank,
                    segment=segment,
                    patient_writer=patient_writer,
                    multiplicity_writer=multiplicity_writer,
                    moments_writer=moments_writer,
                    attempt_writer=attempt_writer,
                )
                rank_frame, map_frame = summarize_model_bank(
                    bank,
                    configurations,
                    segment=segment,
                    observation_variance_mv2=observation_variance,
                    draw_sink=draw_writer.write_frame,
                    metric_workers=metric_workers,
                )
                published = True
            finally:
                draw_writer.close(publish=published)
                patient_writer.close(publish=published)
                multiplicity_writer.close(publish=published)
                moments_writer.close(publish=published)
                attempt_writer.close(publish=published)
            rejection = {
                "rejected_draws": bank.rejected_draws,
                "rejection_fraction": bank.rejection_fraction,
            }
            common_audit = {
                "schema_version": BOOTSTRAP_AUDIT_SCHEMA_VERSION,
                "segment": segment,
                "basis_variant": arguments.basis_variant,
                "seed": bank.seed,
                "n_patients": len(bank.patient_ids),
                "requested_draws": bank.n_boot,
                "attempted_draws": bank.n_boot + bank.rejected_draws,
            }
            audit_rows = [
                {**common_audit, "status": "accepted", "draw_count": bank.n_boot},
                {
                    **common_audit,
                    "status": "rejected_rank_deficient",
                    "draw_count": bank.rejected_draws,
                },
            ]
            _write_parquet(
                rank_frame, store.segment_artifact(segment, "rank_path")
            )
            _write_parquet(
                map_frame, store.segment_artifact(segment, "map_cells")
            )
            _write_parquet(
                pd.DataFrame(audit_rows),
                store.segment_artifact(segment, "bootstrap_audit"),
            )
            store.commit_segment(
                segment,
                metadata={
                    "n_rank_rows": len(rank_frame),
                    "n_map_cells": len(map_frame),
                    "n_bootstrap_draw_rows": (
                        bank.n_boot
                        * len(configurations)
                        * len(LEADS)
                        * len(bank.ranks)
                    ),
                    "n_bootstrap_attempt_rows": bank.attempt_ledger.n_attempts,
                    "bootstrap_rejection": rejection,
                },
            )
        for name in SEGMENT_ARTIFACT_FILENAMES:
            store.merge_parquet(name, output_dir / artifact_paths[name])
        _write_parquet(
            tuning_table, output_dir / artifact_paths["regularization_tuning"]
        )
        _write_json(
            output_dir / artifact_paths["patient_audit"],
            {
                "schema_version": SCHEMA_VERSION,
                "segment_sampling": sampling_config,
                "segment_sampling_sha256": lineage.canonical_sha256(
                    sampling_config
                ),
                "train": train_audit.to_dict(),
                "tune": tune_audit.to_dict(),
            },
        )
        store.mark_complete(artifact_paths)
        store.cleanup_staging()

    rank_path = pd.read_parquet(output_dir / artifact_paths["rank_path"])
    map_cells = pd.read_parquet(output_dir / artifact_paths["map_cells"])
    import pyarrow.parquet as pq

    draw_parquet = pq.ParquetFile(output_dir / artifact_paths["bootstrap_draws"])
    n_bootstrap_draw_rows = int(draw_parquet.metadata.num_rows)
    expected_draw_rows = (
        len(arguments.segments)
        * arguments.n_bootstrap
        * len(configurations)
        * len(LEADS)
        * len(RANK_GRID)
    )
    if n_bootstrap_draw_rows != expected_draw_rows:
        raise ValueError("published robust-map draw count disagrees with the frozen grid")
    attempt_frame = pd.read_parquet(
        output_dir / artifact_paths["bootstrap_attempts"]
    )
    audit_frame = pd.read_parquet(output_dir / artifact_paths["bootstrap_audit"])
    n_bootstrap_attempt_rows = len(attempt_frame)
    bootstrap_rejections = {}
    for segment in arguments.segments:
        rows = audit_frame[audit_frame["segment"].astype(str) == segment]
        accepted = rows[rows["status"].astype(str) == "accepted"]
        rejected = rows[
            rows["status"].astype(str) == "rejected_rank_deficient"
        ]
        if len(accepted) != 1 or len(rejected) != 1:
            raise ValueError("published robust-map bootstrap audit is incomplete")
        accepted_count = int(accepted["draw_count"].iloc[0])
        rejected_count = int(rejected["draw_count"].iloc[0])
        attempted_count = int(accepted["attempted_draws"].iloc[0])
        segment_attempt_rows = int(
            (attempt_frame["segment"].astype(str) == segment).sum()
        )
        if (
            accepted_count != arguments.n_bootstrap
            or attempted_count != accepted_count + rejected_count
            or segment_attempt_rows != attempted_count
            or int(rejected["attempted_draws"].iloc[0]) != attempted_count
        ):
            raise ValueError("published robust-map attempt counts are inconsistent")
        bootstrap_rejections[segment] = {
            "rejected_draws": rejected_count,
            "rejection_fraction": rejected_count / attempted_count,
        }
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
        "metric_engine": {
            "implementation": "rank-batched numpy linear algebra",
            "workers": metric_workers,
            "worker_source": "ECGCERT_NUM_WORKERS_or_serial_default",
            "per_worker_workspace_proxy_bytes": metric_workspace_bytes,
            "draw_rows_per_configuration": _draw_rows_per_configuration(
                arguments.n_bootstrap, len(RANK_GRID)
            ),
            "configuration_order": "frozen input order",
        },
        "segment_execution": {
            "inventory_schema_version": ROBUST_MAP_INVENTORY_SCHEMA_VERSION,
            "resume_unit": "wave_segment",
            "resume_supported": True,
            "completed_segments": list(arguments.segments),
        },
        "bootstrap_evidence_schema_version": BOOTSTRAP_AUDIT_SCHEMA_VERSION,
        "bootstrap_draw_schema_version": BOOTSTRAP_DRAW_SCHEMA_VERSION,
        "bootstrap_replay_schema_version": BOOTSTRAP_REPLAY_SCHEMA_VERSION,
        "bootstrap_moments_schema_version": BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
        "bootstrap_attempt_schema_version": BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
        "n_bootstrap_draw_rows": n_bootstrap_draw_rows,
        "n_bootstrap_attempt_rows": n_bootstrap_attempt_rows,
        "bootstrap_replicates": arguments.n_bootstrap,
        "bootstrap_unit": "patient",
        "bootstrap_rank_deficient_draws": bootstrap_rejections,
        "seed": arguments.seed,
        "segment_sampling": sampling_config,
        "segment_sampling_sha256": lineage.canonical_sha256(sampling_config),
        "run_configuration": run_configuration,
        "run_configuration_sha256": run_configuration_sha256,
        "lineage_identity_sha256": store.identity_sha256,
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
        if arguments.sensitivity == "sample-cap":
            summary["sensitivity_contrast"] = {
                "factor": "per_record_per_segment_timepoint_cap",
                "primary": PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
                "sensitivity": SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
                "primary_selection_is_nested_subset": True,
            }
    if arguments.diagnosis_class is not None:
        summary["diagnosis_class"] = arguments.diagnosis_class
        summary["diagnosis_membership"] = "multi-label contains superclass"
    _write_json(output_dir / "summary.v3.json", summary)


if __name__ == "__main__":
    main()
