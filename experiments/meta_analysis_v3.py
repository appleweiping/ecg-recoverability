"""Locked fold-8/9/10 meta-analysis and fail-closed ARC Stage-15 gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.arc_control import validate_arc_waiting_report
from ecgcert.arc_forward import waiting_control_evidence
from ecgcert.benchmarking import (
    EXPECTED_METHODS,
    PRIMARY_METHODS,
    RELEASE_NEURAL_SEEDS,
    load_benchmark_bundles,
    path_sha256,
)
from ecgcert.evaluation import (
    CATEGORICAL_PREDICTORS,
    META_RIDGE_ALPHA_GRID,
    STAGE15_GATE_ELIGIBLE_EXTERNAL_COHORTS,
    BootstrapEffect,
    LocoMetaModelBank,
    accumulate_meta_sufficient_statistics,
    fit_loco_meta_model_bank,
    fixed_meta_encoding,
    _bootstrap_rows_with_audit,
    prediction_delta_r2,
    predict_with_loco_meta_bank,
    stage15_decision,
    tune_meta_ridge_alpha_from_sufficient,
)
from ecgcert.panel_completeness import (
    PartitionCoverage,
    PredictorContract,
    load_common_predictor_contract,
    load_external_test_source,
    load_ptbxl_source_partitions,
    require_identical_coverages,
    validate_metric_panel,
    validate_partition_audit,
)
from ecgcert.data.manifest import DatasetManifest
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
from ecgcert.recoverability import (
    BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
    BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
    BOOTSTRAP_REPLAY_SCHEMA_VERSION,
    rebuild_model_bank_from_artifacts,
)


SCHEMA_VERSION = "meta-analysis-v3"
COMMON_PANEL_METHODS = ("lowrank", "ridge", "masked-unet", "imputeecg")
NEURAL_METHODS = ("masked-unet", "imputeecg")
RANK_MAP_SCHEMA_VERSION = "robust-recoverability-map-v3"
RANK_MAP_BOOTSTRAP_SCHEMA_VERSION = "robust-recoverability-bootstrap-audit-v3"
RANK_MAP_DRAW_SCHEMA_VERSION = "robust-recoverability-bootstrap-draw-v3"
EXTERNAL_VALIDATION_SCHEMA_VERSION = "external-validation-v3"
META_BOOTSTRAP_DRAW_SCHEMA_VERSION = "meta-analysis-bootstrap-draw-v3"
META_SEED_PREDICTION_SCHEMA_VERSION = "meta-analysis-seed-prediction-v3"
META_SUFFICIENT_SCHEMA_VERSION = "meta-analysis-sufficient-stat-v3"
META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION = (
    "meta-analysis-paired-seed-sufficient-v3"
)
META_BOOTSTRAP_SEED = 20260719
META_BOOTSTRAP_SEED_OFFSETS = {"PTB-XL": 0, "chapman": 100, "cpsc2018": 101}
EFFECT_COMPARISON_ATOL = 1e-12
STREAM_BATCH_ROWS = 96_000
MULTIPLICITY_VALIDATION_BATCH_ROWS = 16
POINT_SEED_SENTINEL = -1
PREDICTION_EVIDENCE_IDENTIFIERS = (
    "cohort",
    "partition",
    "patient_id",
    "method",
    "segment",
    "configuration",
    "target",
)
PREDICTION_EVIDENCE_COLUMNS = (
    *PREDICTION_EVIDENCE_IDENTIFIERS,
    "outcome_log_rmse",
    "prediction_simple",
    "prediction_augmented",
)
META_METRIC_COLUMNS = (
    "schema_version",
    "cohort",
    "partition",
    "patient_id",
    "method",
    "model_seed",
    "segment",
    "configuration",
    "target",
    "n_observed",
    "n_records",
    "n_samples",
    "target_rms",
    "max_target_observed_correlation",
    "outcome_log_rmse",
)
SUFFICIENT_COLUMNS = (
    "schema_version",
    "cohort",
    "patient_id",
    "method",
    "model_seed",
    "estimand",
    "row_count",
    "truth_sum",
    "truth_square_sum",
    "simple_square_error",
    "augmented_square_error",
)
PAIRED_SEED_SUFFICIENT_COLUMNS = (
    "schema_version",
    "cohort",
    "patient_id",
    "method",
    "model_seeds_json",
    "row_count",
    "truth_sums_json",
    "truth_crossproducts_json",
    "simple_truth_products_json",
    "augmented_truth_products_json",
    "simple_prediction_square_sum",
    "augmented_prediction_square_sum",
)


def _expected_common_seeds(method: str) -> tuple[int, ...]:
    if method in NEURAL_METHODS:
        return tuple(RELEASE_NEURAL_SEEDS)
    if method in {"lowrank", "ridge"}:
        return (0,)
    raise ValueError(f"{method!r} is not a common-panel method")


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


def _pyarrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - locked environments provide pyarrow
        raise RuntimeError("streaming meta-analysis requires locked pyarrow") from exc
    return pa, ds, pq


def _parquet_schema_names(path: Path) -> tuple[str, ...]:
    _pa, _ds, pq = _pyarrow_modules()
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _dataset_filter(ds: Any, filters: Mapping[str, Sequence[Any]] | None) -> Any:
    expression = None
    for column, raw_values in (filters or {}).items():
        values = tuple(raw_values)
        if not values:
            raise ValueError(f"empty Parquet filter for {column}")
        predicate = (
            ds.field(column) == values[0]
            if len(values) == 1
            else ds.field(column).isin(list(values))
        )
        expression = predicate if expression is None else expression & predicate
    return expression


def _iter_parquet_batches(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    filters: Mapping[str, Sequence[Any]] | None = None,
    batch_rows: int = STREAM_BATCH_ROWS,
) -> Iterator[pd.DataFrame]:
    """Yield bounded Arrow batches without materialising a complete Parquet file."""

    if batch_rows < 1:
        raise ValueError("Parquet batch size must be positive")
    _pa, ds, _pq = _pyarrow_modules()
    for path in paths:
        names = set(_parquet_schema_names(path))
        missing = set(columns) - names
        if missing:
            raise ValueError(f"{path} lacks streaming columns: {sorted(missing)}")
        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(
            columns=list(columns),
            filter=_dataset_filter(ds, filters),
            batch_size=batch_rows,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            if batch.num_rows:
                yield batch.to_pandas()


def _count_parquet_rows(
    paths: Sequence[Path],
    *,
    filters: Mapping[str, Sequence[Any]] | None = None,
) -> int:
    first_path = paths[0]
    names = _parquet_schema_names(first_path)
    probe = "method" if "method" in names else names[0]
    return sum(
        len(batch)
        for batch in _iter_parquet_batches(
            paths, columns=(probe,), filters=filters
        )
    )


def _validate_streamed_bootstrap_multiplicities(
    path: Path,
    *,
    segments: Sequence[str],
    basis_variant: str,
    replicates: int,
    replayed_banks: Mapping[str, Any],
    batch_rows: int = MULTIPLICITY_VALIDATION_BATCH_ROWS,
) -> dict[str, int]:
    """Validate every accepted patient draw with memory bounded by one Arrow batch.

    The multiplicity artifact is release-scale (one patient-length vector for
    every accepted draw), so converting it to one pandas frame is not safe.  We
    retain the exact replay check while materialising only one vector at a time.
    """

    if batch_rows < 1:
        raise ValueError("multiplicity validation batch size must be positive")
    if replicates < 1:
        raise ValueError("multiplicity validation requires accepted replicates")
    pa, _ds, pq = _pyarrow_modules()
    import pyarrow.compute as pc

    parquet = pq.ParquetFile(path)
    required = {
        "schema_version",
        "segment",
        "basis_variant",
        "bootstrap_index",
        "accepted",
        "multiplicities",
    }
    schema_names = tuple(parquet.schema_arrow.names)
    if len(schema_names) != len(required) or set(schema_names) != required:
        raise ValueError("bootstrap multiplicity table does not use the exact schema")
    segment_order = tuple(str(segment) for segment in segments)
    if not segment_order or len(segment_order) != len(set(segment_order)):
        raise ValueError("multiplicity validation requires unique segments")
    if set(replayed_banks) != set(segment_order):
        raise ValueError("multiplicity replay banks do not match the frozen segments")
    seen = {
        segment: np.zeros(replicates, dtype=np.bool_) for segment in segment_order
    }
    total_rows = 0
    maximum_batch_rows = 0
    maximum_batch_bytes = 0
    maximum_materialized_batch_bytes = 0
    maximum_materialized_vector_bytes = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
        maximum_batch_rows = max(maximum_batch_rows, batch.num_rows)
        maximum_batch_bytes = max(maximum_batch_bytes, int(batch.nbytes))
        columns = {
            name: batch.column(batch.schema.get_field_index(name)) for name in required
        }
        multiplicity_column = columns["multiplicities"]
        if multiplicity_column.null_count or not (
            pa.types.is_list(multiplicity_column.type)
            or pa.types.is_large_list(multiplicity_column.type)
            or pa.types.is_fixed_size_list(multiplicity_column.type)
        ):
            raise ValueError(
                "bootstrap multiplicities cannot reconstruct a patient draw"
            )
        lengths = pc.list_value_length(multiplicity_column).to_numpy(
            zero_copy_only=False
        )
        flattened = pc.list_flatten(multiplicity_column)
        if flattened.null_count or not pa.types.is_integer(flattened.type):
            raise ValueError(
                "bootstrap multiplicities cannot reconstruct a patient draw"
            )
        flat_values = flattened.to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        maximum_materialized_batch_bytes = max(
            maximum_materialized_batch_bytes, int(flat_values.nbytes)
        )
        offsets = np.empty(batch.num_rows + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
        if int(offsets[-1]) != len(flat_values):
            raise ValueError(
                "bootstrap multiplicities cannot reconstruct a patient draw"
            )
        for row_index in range(batch.num_rows):
            total_rows += 1
            segment = str(columns["segment"][row_index].as_py())
            if segment not in seen:
                raise ValueError(
                    "bootstrap multiplicity table has an unexpected segment"
                )
            if (
                str(columns["schema_version"][row_index].as_py())
                != RANK_MAP_BOOTSTRAP_SCHEMA_VERSION
                or str(columns["basis_variant"][row_index].as_py())
                != basis_variant
            ):
                raise ValueError(
                    "bootstrap multiplicity table uses the wrong schema or basis"
                )
            if columns["accepted"][row_index].as_py() is not True:
                raise ValueError(
                    "bootstrap multiplicity table contains rejected draws"
                )
            raw_bootstrap_index = columns["bootstrap_index"][row_index].as_py()
            if isinstance(raw_bootstrap_index, bool) or not isinstance(
                raw_bootstrap_index, (int, np.integer)
            ):
                raise ValueError("bootstrap multiplicity index must be an integer")
            bootstrap_index = int(raw_bootstrap_index)
            if (
                bootstrap_index < 0
                or bootstrap_index >= replicates
                or seen[segment][bootstrap_index]
            ):
                raise ValueError(
                    "bootstrap multiplicities contain an out-of-range or duplicate draw"
                )
            replayed = replayed_banks[segment]
            n_patients = len(replayed.patient_ids)
            values = flat_values[offsets[row_index] : offsets[row_index + 1]]
            if (
                n_patients < 1
                or int(lengths[row_index]) != n_patients
                or np.any(values < 0)
                or int(values.sum(dtype=np.int64)) != n_patients
            ):
                raise ValueError(
                    "bootstrap multiplicities cannot reconstruct a patient draw"
                )
            expected = np.asarray(
                replayed.bootstrap_multiplicities[bootstrap_index], dtype=np.int64
            )
            maximum_materialized_vector_bytes = max(
                maximum_materialized_vector_bytes, int(values.nbytes)
            )
            if not np.array_equal(values, expected):
                raise ValueError(
                    "accepted bootstrap multiplicities disagree with the replayed "
                    "attempt ledger"
                )
            seen[segment][bootstrap_index] = True
    expected_rows = len(segment_order) * replicates
    if total_rows != expected_rows or any(not flags.all() for flags in seen.values()):
        raise ValueError("bootstrap multiplicities do not cover accepted draws")
    return {
        "rows": total_rows,
        "max_batch_rows": maximum_batch_rows,
        "max_arrow_batch_bytes": maximum_batch_bytes,
        "max_materialized_batch_bytes": maximum_materialized_batch_bytes,
        "max_materialized_vector_bytes": maximum_materialized_vector_bytes,
    }


class _SequentialILoc:
    def __init__(self, owner: "_StreamingMetricFrame") -> None:
        self.owner = owner

    def __getitem__(self, key: slice) -> pd.DataFrame:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("streaming metric frame supports sequential unit slices only")
        start = 0 if key.start is None else int(key.start)
        stop = len(self.owner) if key.stop is None else int(key.stop)
        return self.owner._take(start, stop)


class _StreamingMetricFrame:
    """Minimal sequential DataFrame facade for the exact panel validator.

    ``validate_metric_panel`` already processes ``iloc`` chunks.  This facade
    supplies those chunks from Arrow and deliberately rejects random access, so
    the validator retains its exact compact bit-mask logic without ever holding
    the seed-level table in RAM.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[pd.DataFrame]],
        *,
        columns: Sequence[str],
        n_rows: int,
    ) -> None:
        self._factory = factory
        self._iterator: Iterator[pd.DataFrame] | None = None
        self._pending = pd.DataFrame(columns=list(columns))
        self._consumed = 0
        self.columns = pd.Index(columns)
        self._n_rows = int(n_rows)
        self.iloc = _SequentialILoc(self)

    @property
    def empty(self) -> bool:
        return self._n_rows == 0

    def __len__(self) -> int:
        return self._n_rows

    def _take(self, start: int, stop: int) -> pd.DataFrame:
        if start != self._consumed or stop < start:
            raise RuntimeError("streaming metric validation attempted non-sequential access")
        wanted = min(stop, self._n_rows) - start
        if self._iterator is None:
            self._iterator = iter(self._factory())
        chunks: list[pd.DataFrame] = []
        available = 0
        if not self._pending.empty:
            chunks.append(self._pending)
            available += len(self._pending)
            self._pending = self._pending.iloc[0:0]
        while available < wanted:
            try:
                chunk = next(self._iterator)
            except StopIteration as exc:
                raise ValueError("Parquet stream ended before its counted row total") from exc
            chunks.append(chunk)
            available += len(chunk)
        combined = chunks[0] if len(chunks) == 1 else pd.concat(chunks, ignore_index=True)
        result = combined.iloc[:wanted].reset_index(drop=True)
        self._pending = combined.iloc[wanted:].reset_index(drop=True)
        self._consumed += len(result)
        return result


def _streaming_panel_report(
    paths: Sequence[Path],
    *,
    cohort: str,
    coverages: Mapping[str, PartitionCoverage],
    method_seeds: Mapping[str, Sequence[int]],
    configurations: Sequence[str],
    predictor: PredictorContract | None,
) -> dict[str, Any]:
    filters = {"method": tuple(method_seeds)}
    columns = tuple(
        dict.fromkeys(
            (
                *META_METRIC_COLUMNS,
                "target_rms",
                "max_target_observed_correlation",
            )
        )
    )
    n_rows = _count_parquet_rows(paths, filters=filters)
    frame = _StreamingMetricFrame(
        lambda: _iter_parquet_batches(paths, columns=columns, filters=filters),
        columns=columns,
        n_rows=n_rows,
    )
    return validate_metric_panel(
        frame,  # type: ignore[arg-type]
        cohort=cohort,
        coverages=coverages,
        method_seeds=method_seeds,
        configurations=configurations,
        predictor=predictor,
    )


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
        "bootstrap_evidence_schema_version": RANK_MAP_BOOTSTRAP_SCHEMA_VERSION,
        "bootstrap_draw_schema_version": RANK_MAP_DRAW_SCHEMA_VERSION,
        "bootstrap_replay_schema_version": BOOTSTRAP_REPLAY_SCHEMA_VERSION,
        "bootstrap_moments_schema_version": BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
        "bootstrap_attempt_schema_version": BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
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
        ("bootstrap_draws", "bootstrap_draws.parquet"),
        ("bootstrap_patients", "bootstrap_patients.parquet"),
        ("bootstrap_multiplicities", "bootstrap_multiplicities.parquet"),
        ("bootstrap_audit", "bootstrap_audit.parquet"),
        ("bootstrap_moments", "bootstrap_moments.parquet"),
        ("bootstrap_attempts", "bootstrap_attempts.parquet"),
    ):
        relative = artifacts.get(key)
        if relative != filename or artifact_sha256.get(key) is None:
            raise ValueError(f"rank-map summary lacks authenticated {key}")
        _authenticated_artifact(
            root,
            {"path": relative, "sha256": artifact_sha256[key]},
            expected_name=filename,
        )
    _validate_rank_map_bootstrap_evidence(root, summary)
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


def _validate_rank_map_bootstrap_evidence(
    root: Path,
    summary: Mapping[str, Any],
    *,
    expected_configurations: set[str] | None = None,
    expected_targets: set[str] | None = None,
) -> None:
    """Rebuild every rank quantile and robust envelope from authenticated raw draws."""

    import pyarrow.parquet as pq

    root = root.resolve()
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("rank-map summary lacks bootstrap artifact paths")
    rank_path = root / str(artifacts.get("rank_path", ""))
    map_path = root / str(artifacts.get("map_cells", ""))
    draws_path = root / str(artifacts.get("bootstrap_draws", ""))
    patients_path = root / str(artifacts.get("bootstrap_patients", ""))
    multiplicities_path = root / str(artifacts.get("bootstrap_multiplicities", ""))
    audit_path = root / str(artifacts.get("bootstrap_audit", ""))
    moments_path = root / str(artifacts.get("bootstrap_moments", ""))
    attempts_path = root / str(artifacts.get("bootstrap_attempts", ""))
    rank_path_frame = _read_small_parquet(rank_path)
    map_frame = _read_small_parquet(map_path)
    patients = _read_small_parquet(patients_path)
    audit = _read_small_parquet(audit_path)

    segments = tuple(str(value) for value in summary.get("segments", ()))
    ranks = tuple(int(value) for value in summary.get("ranks", ()))
    replicates = int(summary.get("bootstrap_replicates", 0))
    basis_variant = str(summary.get("basis_variant", ""))
    configurations = (
        expected_configurations
        if expected_configurations is not None
        else {"+".join(value) for value in all_independent_configurations()}
    )
    targets = expected_targets if expected_targets is not None else set(CANONICAL_LEADS)
    if not segments or not ranks or replicates < 1 or not configurations or not targets:
        raise ValueError("rank-map summary has an empty bootstrap evidence grid")
    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping):
        raise ValueError("rank-map summary lacks replay artifact hashes")
    try:
        from experiments.robust_maps_v3 import (
            _batched_model_metrics,
            _stack_spatial_models,
        )
    except ModuleNotFoundError as exc:  # direct ``python experiments/...`` execution
        if exc.name != "experiments":
            raise
        from robust_maps_v3 import _batched_model_metrics, _stack_spatial_models

    replayed_banks: dict[str, Any] = {}
    replayed_metric_batches: dict[str, dict[int, Any]] = {}
    for segment_index, segment in enumerate(segments):
        replayed = rebuild_model_bank_from_artifacts(
            moments_path,
            attempts_path,
            artifact_sha256=artifact_sha256,
            segment=segment,
            ranks=ranks,
            basis_variants=(basis_variant,),
            fit_cohort=f"PTB-XL/folds1-7/{segment}",
            seed=int(summary.get("seed", -1)) + 1000 * segment_index,
        )
        if replayed.n_boot != replicates:
            raise ValueError("replayed model bank has the wrong accepted draw count")
        replayed_banks[segment] = replayed
        replayed_metric_batches[segment] = {}
        for rank in ranks:
            models = []
            for bootstrap_index, group in enumerate(replayed.bootstrap_models):
                matches = [model for model in group if model.rank == rank]
                if len(matches) != 1:
                    raise ValueError(
                        f"replayed draw {bootstrap_index} lacks exactly one rank-{rank} model"
                    )
                models.append(matches[0])
            replayed_metric_batches[segment][rank] = _stack_spatial_models(models)
    observation_variance = float(summary.get("observation_variance_mv2", np.nan))
    if not np.isfinite(observation_variance) or observation_variance <= 0:
        raise ValueError("rank-map summary lacks positive observation variance")

    rank_keys = ["segment", "configuration", "target", "rank"]
    rank_required = {
        "schema_version",
        *rank_keys,
        "ambiguity_q975_mv",
        "eta_normalized_q975",
        "kappa_target_q975",
        "kappa_global_q975",
        "bootstrap_replicates",
    }
    map_keys = ["segment", "configuration", "target"]
    map_required = {
        "schema_version",
        *map_keys,
        "ambiguity_robust_mv",
        "recoverability_lower",
        "log10_kappa_target_upper",
        "log10_kappa_global_upper",
    }
    if rank_required - set(rank_path_frame.columns):
        raise ValueError("rank path lacks bootstrap-derived release columns")
    if map_required - set(map_frame.columns):
        raise ValueError("map cells lack bootstrap-derived release columns")
    if rank_path_frame.duplicated(rank_keys).any() or map_frame.duplicated(map_keys).any():
        raise ValueError("rank-map bootstrap evidence contains duplicate summary cells")
    expected_rank_rows = len(segments) * len(configurations) * len(targets) * len(ranks)
    expected_map_rows = len(segments) * len(configurations) * len(targets)
    if len(rank_path_frame) != expected_rank_rows or len(map_frame) != expected_map_rows:
        raise ValueError("rank-map bootstrap summaries do not cover the frozen grid")
    rank_path_frame = rank_path_frame.copy()
    map_frame = map_frame.copy()
    for frame in (rank_path_frame, map_frame):
        frame["segment"] = frame["segment"].astype(str)
        frame["configuration"] = frame["configuration"].astype(str)
        frame["target"] = frame["target"].astype(str)
    rank_path_frame["rank"] = rank_path_frame["rank"].astype(int)
    rank_lookup = rank_path_frame.set_index(rank_keys, verify_integrity=True)
    map_lookup = map_frame.set_index(map_keys, verify_integrity=True)

    def require_close(actual: float, expected: float, label: str) -> None:
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"{label} disagrees with raw patient-bootstrap draws: {actual} != {expected}"
            )

    draw_required = {
        "schema_version",
        "segment",
        "configuration",
        "target",
        "rank",
        "basis_variant",
        "bootstrap_index",
        "a_r_mv",
        "eta_normalized",
        "kappa_target",
        "kappa_global",
        "configuration_rank",
        "condition_number",
    }
    parquet = pq.ParquetFile(draws_path)
    if set(parquet.schema_arrow.names) != draw_required:
        raise ValueError("raw rank-map bootstrap draws do not use the exact schema")
    seen_cells: set[tuple[str, str]] = set()
    total_draw_rows = 0
    for row_group_index in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group_index).to_pandas()
        total_draw_rows += len(frame)
        cell_values = frame[["segment", "configuration"]].drop_duplicates()
        if len(cell_values) != 1:
            raise ValueError("each raw-draw row group must contain one segment/configuration cell")
        segment = str(cell_values.iloc[0]["segment"])
        configuration = str(cell_values.iloc[0]["configuration"])
        cell = (segment, configuration)
        if cell in seen_cells:
            raise ValueError("raw rank-map draw cells occur in multiple row groups")
        seen_cells.add(cell)
        if segment not in segments or configuration not in configurations:
            raise ValueError("raw rank-map draws contain an unexpected segment/configuration")
        if set(frame["schema_version"].astype(str)) != {RANK_MAP_DRAW_SCHEMA_VERSION}:
            raise ValueError("raw rank-map draws use an unsupported schema")
        if set(frame["basis_variant"].astype(str)) != {basis_variant}:
            raise ValueError("raw rank-map draws use the wrong basis variant")
        if set(frame["target"].astype(str)) != targets or set(
            frame["rank"].astype(int)
        ) != set(ranks):
            raise ValueError("raw rank-map draws do not cover the frozen rank/target grid")
        draw_keys = ["target", "rank", "bootstrap_index"]
        if frame.duplicated(draw_keys).any():
            raise ValueError("raw rank-map draws contain duplicate bootstrap cells")
        numeric_columns = [
            "a_r_mv",
            "eta_normalized",
            "kappa_target",
            "kappa_global",
            "configuration_rank",
            "condition_number",
        ]
        if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("raw rank-map draws contain non-finite diagnostics")
        if (
            (frame[["a_r_mv", "kappa_target", "kappa_global", "condition_number"]] < 0)
            .any()
            .any()
            or (frame["eta_normalized"] < -1e-12).any()
            or (frame["eta_normalized"] > 1.0 + 1e-12).any()
        ):
            raise ValueError("raw rank-map draws contain invalid diagnostic ranges")

        configuration_tuple = tuple(configuration.split("+"))
        target_order = tuple(lead for lead in CANONICAL_LEADS if lead in targets)
        target_indices = np.asarray(
            [CANONICAL_LEADS.index(lead) for lead in target_order], dtype=np.int64
        )
        for rank in ranks:
            metrics = _batched_model_metrics(
                replayed_metric_batches[segment][rank],
                configuration_tuple,
                observation_variance,
            )
            observed = frame[frame["rank"].astype(int) == rank].copy()
            observed["_target_order"] = pd.Categorical(
                observed["target"].astype(str), categories=target_order, ordered=True
            ).codes
            observed = observed.sort_values(
                ["_target_order", "bootstrap_index"]
            ).reset_index(drop=True)
            if (observed["_target_order"] < 0).any():
                raise ValueError("raw rank-map draws contain an unexpected target")
            expected_rows = len(target_order) * replicates
            if len(observed) != expected_rows:
                raise ValueError("raw rank-map draws cannot align with the replayed model bank")
            exact_expected = {
                "bootstrap_index": np.tile(
                    np.arange(replicates, dtype=np.int64), len(target_order)
                ),
                "configuration_rank": np.tile(
                    metrics["configuration_rank"], len(target_order)
                ),
            }
            for column, expected in exact_expected.items():
                if not np.array_equal(
                    observed[column].to_numpy(dtype=np.int64),
                    np.asarray(expected, dtype=np.int64),
                ):
                    raise ValueError(
                        f"raw rank-map {column} disagrees with replayed patient moments"
                    )
            numeric_expected = {
                "a_r_mv": metrics["ambiguity"][:, target_indices].T.reshape(-1),
                "eta_normalized": metrics["eta_normalized"][:, target_indices]
                .T.reshape(-1),
                "kappa_target": metrics["kappa_per_target"][:, target_indices]
                .T.reshape(-1),
                "kappa_global": np.tile(metrics["kappa_global"], len(target_order)),
                "condition_number": np.tile(
                    metrics["condition_number"], len(target_order)
                ),
            }
            for column, expected in numeric_expected.items():
                if not np.allclose(
                    observed[column].to_numpy(dtype=float),
                    np.asarray(expected, dtype=float),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"raw rank-map {column} disagrees with replayed patient moments"
                    )

        ambiguity_by_target: dict[str, list[float]] = {target: [] for target in targets}
        eta_by_target: dict[str, list[float]] = {target: [] for target in targets}
        kappa_target_by_target: dict[str, list[float]] = {
            target: [] for target in targets
        }
        kappa_global_by_target: dict[str, list[float]] = {
            target: [] for target in targets
        }
        for (target_value, rank_value), group in frame.groupby(
            ["target", "rank"], sort=False
        ):
            target = str(target_value)
            rank = int(rank_value)
            if len(group) != replicates or set(group["bootstrap_index"].astype(int)) != set(
                range(replicates)
            ):
                raise ValueError("raw rank-map draw group lacks accepted bootstrap indices")
            quantiles = {
                "ambiguity_q975_mv": float(np.nanquantile(group["a_r_mv"], 0.975)),
                "eta_normalized_q975": float(
                    np.nanquantile(group["eta_normalized"], 0.975)
                ),
                "kappa_target_q975": float(np.nanquantile(group["kappa_target"], 0.975)),
                "kappa_global_q975": float(np.nanquantile(group["kappa_global"], 0.975)),
            }
            try:
                rank_row = rank_lookup.loc[(segment, configuration, target, rank)]
            except KeyError as exc:
                raise ValueError("rank path lacks one row for a raw-draw group") from exc
            if int(rank_row["bootstrap_replicates"]) != replicates:
                raise ValueError("rank path lacks one row for a raw-draw group")
            for column, expected in quantiles.items():
                require_close(float(rank_row[column]), expected, column)
            ambiguity_by_target[target].append(quantiles["ambiguity_q975_mv"])
            eta_by_target[target].append(quantiles["eta_normalized_q975"])
            kappa_target_by_target[target].append(quantiles["kappa_target_q975"])
            kappa_global_by_target[target].append(quantiles["kappa_global_q975"])

        for target in targets:
            try:
                map_row = map_lookup.loc[(segment, configuration, target)]
            except KeyError as exc:
                raise ValueError("map cells lack one row for a raw-draw envelope") from exc
            rebuilt = {
                "ambiguity_robust_mv": max(ambiguity_by_target[target]),
                "recoverability_lower": float(
                    np.clip(1.0 - max(eta_by_target[target]), 0.0, 1.0)
                ),
                "log10_kappa_target_upper": float(
                    np.log10(max(max(kappa_target_by_target[target]), np.finfo(float).tiny))
                ),
                "log10_kappa_global_upper": float(
                    np.log10(max(max(kappa_global_by_target[target]), np.finfo(float).tiny))
                ),
            }
            for column, expected in rebuilt.items():
                require_close(float(map_row[column]), expected, column)

    expected_cells = {(segment, configuration) for segment in segments for configuration in configurations}
    if seen_cells != expected_cells:
        raise ValueError("raw rank-map draw row groups do not cover the frozen grid")
    expected_draw_rows = expected_rank_rows * replicates
    if total_draw_rows != expected_draw_rows or int(
        summary.get("n_bootstrap_draw_rows", -1)
    ) != expected_draw_rows:
        raise ValueError("raw rank-map draw count disagrees with the frozen grid")

    patient_required = {
        "schema_version",
        "segment",
        "basis_variant",
        "patient_index",
        "patient_id",
    }
    audit_required = {
        "schema_version",
        "segment",
        "basis_variant",
        "seed",
        "n_patients",
        "requested_draws",
        "attempted_draws",
        "status",
        "draw_count",
    }
    if set(patients.columns) != patient_required:
        raise ValueError("bootstrap patient index does not use the exact schema")
    if set(audit.columns) != audit_required:
        raise ValueError("bootstrap accepted/rejected audit does not use the exact schema")
    for label, frame in (
        ("patient index", patients),
        ("accepted/rejected audit", audit),
    ):
        if set(frame["segment"].astype(str)) != set(segments):
            raise ValueError(f"bootstrap {label} has incomplete or unexpected segments")
    expected_rejections = summary.get("bootstrap_rank_deficient_draws")
    if not isinstance(expected_rejections, Mapping):
        raise ValueError("rank-map summary lacks rank-deficient draw audit")
    _validate_streamed_bootstrap_multiplicities(
        multiplicities_path,
        segments=segments,
        basis_variant=basis_variant,
        replicates=replicates,
        replayed_banks=replayed_banks,
    )
    replayed_attempt_rows = 0
    for segment_index, segment in enumerate(segments):
        replayed_bank = replayed_banks[segment]
        patient_rows = patients[patients["segment"].astype(str) == segment]
        audit_rows = audit[audit["segment"].astype(str) == segment]
        if patient_rows.empty or patient_rows["patient_id"].astype(str).duplicated().any():
            raise ValueError("bootstrap patient index is empty or duplicated")
        n_patients = len(patient_rows)
        if set(patient_rows["patient_index"].astype(int)) != set(range(n_patients)):
            raise ValueError("bootstrap patient indices are not contiguous")
        patient_rows = patient_rows.sort_values("patient_index")
        if tuple(patient_rows["patient_id"].astype(str)) != tuple(
            str(patient_id) for patient_id in replayed_bank.patient_ids
        ):
            raise ValueError("bootstrap patient index disagrees with replayed patient moments")
        if set(audit_rows["status"].astype(str)) != {
            "accepted",
            "rejected_rank_deficient",
        } or len(audit_rows) != 2:
            raise ValueError("bootstrap audit lacks accepted/rejected status rows")
        rejected_value = expected_rejections.get(segment)
        if not isinstance(rejected_value, Mapping):
            raise ValueError("rank-map summary lacks per-segment rejection audit")
        rejected = int(rejected_value.get("rejected_draws", -1))
        expected_rejection_fraction = rejected / (replicates + rejected)
        if not np.isclose(
            float(rejected_value.get("rejection_fraction", np.nan)),
            expected_rejection_fraction,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("rank-map rejection fraction disagrees with accepted/rejected audit")
        counts = {
            str(row.status): int(row.draw_count)
            for row in audit_rows[["status", "draw_count"]].itertuples(index=False)
        }
        if counts != {"accepted": replicates, "rejected_rank_deficient": rejected}:
            raise ValueError("bootstrap accepted/rejected counts disagree with summary")
        if (
            replayed_bank.rejected_draws != rejected
            or replayed_bank.attempt_ledger.n_attempts != replicates + rejected
        ):
            raise ValueError("bootstrap audit disagrees with the replayed attempt ledger")
        replayed_attempt_rows += replayed_bank.attempt_ledger.n_attempts
        expected_seed = int(summary.get("seed", -1)) + 1000 * segment_index
        if (
            set(audit_rows["seed"].astype(int)) != {expected_seed}
            or set(audit_rows["n_patients"].astype(int)) != {n_patients}
            or set(audit_rows["requested_draws"].astype(int)) != {replicates}
            or set(audit_rows["attempted_draws"].astype(int)) != {replicates + rejected}
        ):
            raise ValueError("bootstrap accepted/rejected audit metadata is inconsistent")
        for frame in (patient_rows, audit_rows):
            if set(frame["schema_version"].astype(str)) != {
                RANK_MAP_BOOTSTRAP_SCHEMA_VERSION
            } or set(frame["basis_variant"].astype(str)) != {basis_variant}:
                raise ValueError("bootstrap grouping evidence uses the wrong schema or basis")
    if int(summary.get("n_bootstrap_attempt_rows", -1)) != replayed_attempt_rows:
        raise ValueError("bootstrap attempt row count disagrees with replayed ledgers")


def _validate_common_panel_metrics(
    frame: pd.DataFrame,
    *,
    cohort: str,
    partitions: set[str],
    exact_cells: bool = True,
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
    if not exact_cells:
        return primary
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


def _benchmark_release_coverages(
    benchmark_bundles: Mapping[str, Any],
    *,
    sources: Mapping[str, Any],
) -> tuple[dict[str, PartitionCoverage], dict[str, Any]]:
    """Authenticate each PTB bundle audit against the same source partition."""

    required_partitions = {"tune", "calibration", "test"}
    if set(sources) != required_partitions:
        raise ValueError("PTB-XL source manifest lacks a release evaluation partition")
    by_partition: dict[str, list[PartitionCoverage]] = {
        partition: [] for partition in required_partitions
    }
    method_reports: dict[str, Any] = {}
    reference_case_hashes: Mapping[str, Any] | None = None
    for method in EXPECTED_METHODS:
        bundle = benchmark_bundles[method]
        summary = bundle.summary
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(
            artifacts.get("evaluation_audit"), Mapping
        ):
            raise ValueError(f"{method} benchmark lacks an authenticated evaluation audit")
        audit_path = _authenticated_artifact(
            bundle.root,
            artifacts["evaluation_audit"],
            expected_name="evaluation_audit.json",
        )
        audit = _load_json(audit_path)
        if audit.get("schema_version") != BENCHMARK_SCHEMA_VERSION or set(
            audit.get("partitions", {})
        ) != required_partitions:
            raise ValueError(f"{method} benchmark audit has the wrong schema/partitions")
        manifest = summary.get("manifest")
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("sha256") != sources["test"].manifest_sha256
            or manifest.get("split_sha256") != sources["test"].split_sha256
        ):
            raise ValueError(f"{method} benchmark summary disagrees with the PTB manifest")
        case_hashes = summary.get("evaluation_records_sha256")
        if (
            not isinstance(case_hashes, Mapping)
            or set(case_hashes) != required_partitions
            or any(len(str(value)) != 64 for value in case_hashes.values())
            or summary.get("evaluation_contract_sha256")
            != lineage.canonical_sha256(case_hashes)
        ):
            raise ValueError(f"{method} benchmark lacks its evaluation-case hashes")
        if reference_case_hashes is None:
            reference_case_hashes = dict(case_hashes)
        elif dict(case_hashes) != dict(reference_case_hashes):
            raise ValueError("benchmark methods disagree on exact evaluation records/windows")
        for partition in sorted(required_partitions):
            coverage = validate_partition_audit(
                audit["partitions"][partition],
                source=sources[partition],
                audit_artifact_sha256=lineage.artifact_sha256(audit_path),
            )
            by_partition[partition].append(coverage)
        method_reports[method] = {
            "audit_sha256": lineage.artifact_sha256(audit_path),
            "evaluation_records_sha256": dict(case_hashes),
        }
    coverages = {
        partition: require_identical_coverages(values)
        for partition, values in by_partition.items()
    }
    return coverages, {
        "methods": method_reports,
        "partitions": {
            partition: {
                "coverage_sha256": coverage.scientific_sha256,
                "manifest_sha256": coverage.manifest_sha256,
                "split_sha256": coverage.split_sha256,
                "n_attempted_records": len(coverage.attempted_record_ids),
                "n_included_records": len(coverage.included_record_ids),
                "n_excluded_records": len(coverage.excluded_record_ids),
                "excluded_reasons_sha256": lineage.canonical_sha256(
                    coverage.excluded_reasons
                ),
            }
            for partition, coverage in sorted(coverages.items())
        },
    }


def _external_manifest_index(paths: Iterable[Path]) -> dict[str, Path]:
    """Map the two authenticated external manifests to unique cohort names."""

    indexed: dict[str, Path] = {}
    for manifest_path in paths:
        manifest = DatasetManifest.from_path(manifest_path)
        cohort = str(manifest.cohort)
        if cohort not in {"chapman", "cpsc2018"} or cohort in indexed:
            raise ValueError(
                "external manifests must map uniquely to Chapman and CPSC2018 cohorts"
            )
        indexed[cohort] = Path(manifest_path).resolve()
    if set(indexed) != {"chapman", "cpsc2018"}:
        raise ValueError("release meta-analysis requires unique Chapman/CPSC2018 manifests")
    return indexed


def _validate_external_release_bundle(
    bundle: Path,
    *,
    target_manifest: Path,
    source_manifest_sha256: str,
    rank_maps_sha256: str,
    predictor_content_sha256: str,
    benchmark_bundles: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any], PartitionCoverage]:
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
        or audit.get("training_predictors_content_sha256")
        != predictor_content_sha256
        or summary.get("training_predictors_content_sha256")
        != predictor_content_sha256
    ):
        raise ValueError(f"external audit does not prove zero-transfer evaluation: {root}")
    source = load_external_test_source(
        target_manifest,
        cohort=cohort,
        expected_manifest_sha256=str(summary.get("target_manifest_sha256", "")),
        expected_split_sha256=str(summary.get("target_split_sha256", "")),
    )
    data_audit = audit.get("data_audit")
    if not isinstance(data_audit, Mapping):
        raise ValueError(f"external audit lacks record-level data coverage: {root}")
    coverage = validate_partition_audit(
        data_audit,
        source=source,
        audit_artifact_sha256=lineage.artifact_sha256(audit_path),
    )
    if (
        audit.get("requested_record_ids_sha256")
        != lineage.canonical_sha256(sorted(coverage.attempted_record_ids))
        or summary.get("n_test_records_requested") != len(coverage.attempted_record_ids)
        or summary.get("n_test_records_included") != len(coverage.included_record_ids)
        or summary.get("n_test_records_excluded") != len(coverage.excluded_record_ids)
    ):
        raise ValueError(f"external summary silently truncates {cohort} test records")
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
    _pa, _ds, pq = _pyarrow_modules()
    metric_rows = int(pq.ParquetFile(metrics_path).metadata.num_rows)
    if metric_rows != int(summary.get("n_patient_metric_rows", -1)):
        raise ValueError(f"external metric row count disagrees with summary: {cohort}")
    report = {
        "summary_sha256": lineage.artifact_sha256(summary_path),
        "audit_sha256": lineage.artifact_sha256(audit_path),
        "metrics_sha256": lineage.artifact_sha256(metrics_path),
        "target_manifest_sha256": summary["target_manifest_sha256"],
        "target_split_sha256": summary["target_split_sha256"],
        "target_manifest_artifact_sha256": lineage.artifact_sha256(target_manifest),
        "training_predictors_content_sha256": predictor_content_sha256,
        "coverage_sha256": coverage.scientific_sha256,
        "n_attempted_records": len(coverage.attempted_record_ids),
        "n_included_records": len(coverage.included_record_ids),
        "n_excluded_records": len(coverage.excluded_record_ids),
        "excluded_reasons_sha256": lineage.canonical_sha256(
            coverage.excluded_reasons
        ),
        "no_external_fit": True,
    }
    return metrics_path, cohort, report, coverage


def _attach_robust_map(frame: pd.DataFrame, map_cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["segment", "configuration", "target"]
    map_columns = keys + [
        "ambiguity_robust_mv",
        "configuration_rank_max",
        "log10_condition_max",
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


class _BatchCursor:
    """Take equal-sized slices from a bounded batch iterator."""

    def __init__(self, iterator: Iterator[pd.DataFrame], columns: Sequence[str]) -> None:
        self._iterator = iterator
        self._pending = pd.DataFrame(columns=list(columns))
        self._finished = False

    def take(self, rows: int) -> pd.DataFrame:
        if rows < 1:
            raise ValueError("cursor read size must be positive")
        pieces: list[pd.DataFrame] = []
        available = 0
        if not self._pending.empty:
            pieces.append(self._pending)
            available = len(self._pending)
            self._pending = self._pending.iloc[0:0]
        while available < rows and not self._finished:
            try:
                piece = next(self._iterator)
            except StopIteration:
                self._finished = True
                break
            pieces.append(piece)
            available += len(piece)
        if not pieces:
            return self._pending.copy()
        combined = pieces[0] if len(pieces) == 1 else pd.concat(pieces, ignore_index=True)
        result = combined.iloc[:rows].reset_index(drop=True)
        self._pending = combined.iloc[rows:].reset_index(drop=True)
        return result


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
    return left.astype(str).reset_index(drop=True).equals(
        right.astype(str).reset_index(drop=True)
    )


def _iter_aligned_seed_batches(
    path: Path,
    *,
    method: str,
    seeds: Sequence[int],
    partitions: Sequence[str] | None = None,
    batch_rows: int = STREAM_BATCH_ROWS,
) -> Iterator[tuple[pd.DataFrame, tuple[pd.DataFrame, ...]]]:
    """Align preregistered seed rows and emit one seed-mean scientific-cell batch."""

    seed_order = tuple(int(seed) for seed in seeds)
    if not seed_order or len(seed_order) != len(set(seed_order)):
        raise ValueError(f"{method} has an empty or duplicated seed contract")
    filters_base: dict[str, Sequence[Any]] = {"method": (method,)}
    if partitions is not None:
        filters_base["partition"] = tuple(partitions)
    cursors = []
    for seed in seed_order:
        filters = {**filters_base, "model_seed": (seed,)}
        cursors.append(
            _BatchCursor(
                _iter_parquet_batches(
                    (path,),
                    columns=META_METRIC_COLUMNS,
                    filters=filters,
                    batch_rows=batch_rows,
                ),
                META_METRIC_COLUMNS,
            )
        )
    compared_columns = tuple(
        column
        for column in META_METRIC_COLUMNS
        if column not in {"model_seed", "outcome_log_rmse"}
    )
    emitted = False
    while True:
        first = cursors[0].take(batch_rows)
        if first.empty:
            for cursor in cursors[1:]:
                if not cursor.take(1).empty:
                    raise ValueError(f"{method} seed streams have unequal scientific cells")
            break
        raw = [first]
        for cursor in cursors[1:]:
            candidate = cursor.take(len(first))
            if len(candidate) != len(first):
                raise ValueError(f"{method} seed streams have unequal scientific cells")
            raw.append(candidate)
        reference = raw[0]
        for seed, candidate in zip(seed_order[1:], raw[1:], strict=True):
            for column in compared_columns:
                if not _series_equal(reference[column], candidate[column]):
                    raise ValueError(
                        f"{method} seed {seed} does not align on {column!r} scientific cells"
                    )
        mean = reference.copy()
        outcomes = np.vstack(
            [frame["outcome_log_rmse"].to_numpy(dtype=float) for frame in raw]
        )
        if not np.isfinite(outcomes).all():
            raise ValueError(f"{method} seed outcomes contain non-finite values")
        mean["outcome_log_rmse"] = outcomes.mean(axis=0)
        mean["model_seed"] = POINT_SEED_SENTINEL
        emitted = True
        yield mean, tuple(raw)
    if not emitted:
        raise ValueError(f"{method} seed streams contain no metric rows")


def _metric_source_partitions(path: Path) -> set[str]:
    """Read the distinct partition labels without materializing patient rows."""

    partitions: set[str] = set()
    for chunk in _iter_parquet_batches((path,), columns=("partition",)):
        partitions.update(chunk["partition"].astype(str).unique())
    if not partitions:
        raise ValueError(f"metric source is empty: {path}")
    return partitions


def _validate_metric_source_partitions(
    sources: Mapping[str, Path], *, expected: set[str]
) -> None:
    for method, path in sources.items():
        actual = _metric_source_partitions(path)
        if actual != expected:
            raise ValueError(
                f"{method} metric source partitions {sorted(actual)} do not equal "
                f"{sorted(expected)}"
            )


def _iter_enriched_seed_mean_batches(
    sources: Mapping[str, Path],
    rank_map: pd.DataFrame,
    *,
    partitions: Sequence[str],
) -> Iterator[pd.DataFrame]:
    for method in COMMON_PANEL_METHODS:
        for mean, _raw in _iter_aligned_seed_batches(
            sources[method],
            method=method,
            seeds=_expected_common_seeds(method),
            partitions=partitions,
        ):
            yield _attach_robust_map(mean, rank_map)


def _fixed_release_meta_encoding():
    """Return the preregistered encoding, independent of observed row order."""

    return fixed_meta_encoding(
        categorical_levels={
            "method": COMMON_PANEL_METHODS,
            "segment": PRIMARY_SEGMENTS,
            "target": CANONICAL_LEADS,
        }
    )


def _meta_sufficient_for_partition(
    sources: Mapping[str, Path],
    rank_map: pd.DataFrame,
    *,
    partition: str,
):
    configurations = tuple(
        "+".join(configuration) for configuration in deep_configuration_panel()
    )
    return accumulate_meta_sufficient_statistics(
        _iter_enriched_seed_mean_batches(
            sources, rank_map, partitions=(partition,)
        ),
        encoding=_fixed_release_meta_encoding(),
        configurations=configurations,
    )


def _benchmark_metric_sources(
    paths: Sequence[Path],
    *,
    validated_bundles: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    if validated_bundles:
        return {
            method: Path(bundle.root / "patient_metrics.parquet").resolve()
            for method, bundle in validated_bundles.items()
        }
    sources: dict[str, Path] = {}
    for root in paths:
        summary = _load_json(root / "summary.v3.json")
        method = str(summary.get("method", ""))
        if method not in EXPECTED_METHODS or method in sources:
            raise ValueError(f"benchmark paths do not identify one bundle per method: {root}")
        path = (root / "patient_metrics.parquet").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[method] = path
    if set(sources) != set(EXPECTED_METHODS):
        raise ValueError("meta-analysis requires exactly the five benchmark metric sources")
    return sources


def _effect_dict(effect) -> dict[str, Any]:
    value = asdict(effect)
    value["ci95"] = list(effect.ci95)
    return value


class _AtomicParquetStreamWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self._writer: Any | None = None
        self.rows = 0
        self.row_groups = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        pa, _ds, pq = _pyarrow_modules()
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self._writer.schema:
            raise ValueError("streamed seed-prediction batches changed Arrow schema")
        self._writer.write_table(table, row_group_size=table.num_rows)
        self.rows += table.num_rows
        self.row_groups += 1

    def close(self) -> None:
        if self._writer is None:
            raise ValueError(f"refusing to write empty streamed artifact {self.path.name}")
        self._writer.close()
        self._writer = None
        self.temporary.replace(self.path)

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.temporary.exists():
            self.temporary.unlink()


class _SufficientAccumulator:
    def __init__(self, cohort: str) -> None:
        self.cohort = cohort
        self._values: dict[tuple[str, str, int, str], np.ndarray] = {}

    def add(self, frame: pd.DataFrame, *, estimand: str) -> None:
        required = {
            "patient_id",
            "method",
            "model_seed",
            "outcome_log_rmse",
            "prediction_simple",
            "prediction_augmented",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"sufficient-stat input lacks columns: {sorted(missing)}")
        truth = frame["outcome_log_rmse"].to_numpy(dtype=float)
        simple = frame["prediction_simple"].to_numpy(dtype=float)
        augmented = frame["prediction_augmented"].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack((truth, simple, augmented))).all():
            raise ValueError("sufficient-stat input contains non-finite values")
        work = pd.DataFrame(
            {
                "patient_id": frame["patient_id"].astype(str).to_numpy(),
                "method": frame["method"].astype(str).to_numpy(),
                "model_seed": pd.to_numeric(frame["model_seed"], errors="raise")
                .astype(np.int64)
                .to_numpy(),
                "row_count": np.ones(len(frame), dtype=np.int64),
                "truth_sum": truth,
                "truth_square_sum": truth**2,
                "simple_square_error": (truth - simple) ** 2,
                "augmented_square_error": (truth - augmented) ** 2,
            }
        )
        columns = [
            "row_count",
            "truth_sum",
            "truth_square_sum",
            "simple_square_error",
            "augmented_square_error",
        ]
        grouped = work.groupby(
            ["patient_id", "method", "model_seed"], sort=False, observed=True
        )[columns].sum()
        for (patient_id, method, model_seed), values in grouped.iterrows():
            key = (str(patient_id), str(method), int(model_seed), estimand)
            numeric = values.to_numpy(dtype=float)
            if key in self._values:
                self._values[key] += numeric
            else:
                self._values[key] = numeric

    def frame(self) -> pd.DataFrame:
        rows = []
        for (patient_id, method, model_seed, estimand), values in sorted(
            self._values.items()
        ):
            rows.append(
                {
                    "schema_version": META_SUFFICIENT_SCHEMA_VERSION,
                    "cohort": self.cohort,
                    "patient_id": patient_id,
                    "method": method,
                    "model_seed": model_seed,
                    "estimand": estimand,
                    "row_count": int(round(values[0])),
                    "truth_sum": float(values[1]),
                    "truth_square_sum": float(values[2]),
                    "simple_square_error": float(values[3]),
                    "augmented_square_error": float(values[4]),
                }
            )
        frame = pd.DataFrame(rows, columns=SUFFICIENT_COLUMNS)
        if frame.empty:
            raise ValueError("no sufficient statistics were accumulated")
        return frame


class _PairedSeedSufficientAccumulator:
    """Accumulate cross-seed moments needed to resample a seed mean exactly."""

    def __init__(self, cohort: str) -> None:
        self.cohort = cohort
        self._values: dict[tuple[str, str], tuple[tuple[int, ...], np.ndarray]] = {}

    def add(
        self,
        seed_frames: Sequence[pd.DataFrame],
        *,
        seeds: Sequence[int],
    ) -> None:
        seed_order = tuple(int(seed) for seed in seeds)
        if not seed_order or len(seed_frames) != len(seed_order):
            raise ValueError("paired sufficient input disagrees with its seed contract")
        reference = seed_frames[0]
        required = {
            "patient_id",
            "method",
            "outcome_log_rmse",
            "prediction_simple",
            "prediction_augmented",
        }
        if any(required - set(frame.columns) for frame in seed_frames):
            raise ValueError("paired sufficient input lacks prediction columns")
        if any(len(frame) != len(reference) for frame in seed_frames):
            raise ValueError("paired sufficient seed frames have unequal cell counts")
        methods = tuple(reference["method"].astype(str).unique())
        if len(methods) != 1:
            raise ValueError("paired sufficient batches must contain exactly one method")
        method = methods[0]
        if seed_order != _expected_common_seeds(method):
            raise ValueError(f"{method} paired sufficient input uses the wrong seed order")
        identifiers = [
            "cohort",
            "partition",
            "patient_id",
            "method",
            "segment",
            "configuration",
            "target",
        ]
        for frame in seed_frames[1:]:
            for column in identifiers:
                if not _series_equal(reference[column], frame[column]):
                    raise ValueError(
                        f"paired sufficient seed frames disagree on {column!r}"
                    )
            for column in ("prediction_simple", "prediction_augmented"):
                if not np.allclose(
                    reference[column].to_numpy(dtype=float),
                    frame[column].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=EFFECT_COMPARISON_ATOL,
                ):
                    raise ValueError(
                        f"paired sufficient seed frames disagree on {column!r}"
                    )
        outcomes = np.column_stack(
            [frame["outcome_log_rmse"].to_numpy(dtype=float) for frame in seed_frames]
        )
        simple = reference["prediction_simple"].to_numpy(dtype=float)
        augmented = reference["prediction_augmented"].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack((outcomes, simple, augmented))).all():
            raise ValueError("paired sufficient input contains non-finite values")
        patient_values = reference["patient_id"].astype(str).to_numpy()
        for patient in dict.fromkeys(patient_values):
            selected = patient_values == patient
            y = outcomes[selected]
            simple_patient = simple[selected]
            augmented_patient = augmented[selected]
            numeric = np.concatenate(
                (
                    np.asarray([len(y)], dtype=float),
                    y.sum(axis=0),
                    (y.T @ y).reshape(-1),
                    y.T @ simple_patient,
                    y.T @ augmented_patient,
                    np.asarray(
                        [
                            simple_patient @ simple_patient,
                            augmented_patient @ augmented_patient,
                        ],
                        dtype=float,
                    ),
                )
            )
            key = (str(patient), method)
            existing = self._values.get(key)
            if existing is None:
                self._values[key] = (seed_order, numeric)
            else:
                existing_seeds, existing_numeric = existing
                if existing_seeds != seed_order:
                    raise ValueError("paired sufficient seed order changed across batches")
                existing_numeric += numeric

    def frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (patient_id, method), (seeds, numeric) in sorted(self._values.items()):
            k = len(seeds)
            cursor = 0
            row_count = int(round(float(numeric[cursor])))
            cursor += 1
            truth_sums = numeric[cursor : cursor + k]
            cursor += k
            truth_crossproducts = numeric[cursor : cursor + k * k]
            cursor += k * k
            simple_truth_products = numeric[cursor : cursor + k]
            cursor += k
            augmented_truth_products = numeric[cursor : cursor + k]
            cursor += k
            prediction_squares = numeric[cursor : cursor + 2]
            rows.append(
                {
                    "schema_version": META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
                    "cohort": self.cohort,
                    "patient_id": patient_id,
                    "method": method,
                    "model_seeds_json": json.dumps(
                        list(seeds), separators=(",", ":")
                    ),
                    "row_count": row_count,
                    "truth_sums_json": json.dumps(
                        truth_sums.tolist(), separators=(",", ":"), allow_nan=False
                    ),
                    "truth_crossproducts_json": json.dumps(
                        truth_crossproducts.tolist(),
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "simple_truth_products_json": json.dumps(
                        simple_truth_products.tolist(),
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "augmented_truth_products_json": json.dumps(
                        augmented_truth_products.tolist(),
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "simple_prediction_square_sum": float(prediction_squares[0]),
                    "augmented_prediction_square_sum": float(prediction_squares[1]),
                }
            )
        frame = pd.DataFrame(rows, columns=PAIRED_SEED_SUFFICIENT_COLUMNS)
        if frame.empty:
            raise ValueError("no paired seed sufficient statistics were accumulated")
        return frame


def _prediction_lookup(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "segment", "configuration", "target"]
    columns = [*keys, "prediction_simple", "prediction_augmented"]
    missing = set(columns) - set(predictions.columns)
    if missing:
        raise ValueError(f"point predictions lack lookup columns: {sorted(missing)}")
    grouped = predictions.groupby(keys, sort=False, dropna=False)
    for column in ("prediction_simple", "prediction_augmented"):
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"meta prediction {column} changes across patients")
    return grouped[["prediction_simple", "prediction_augmented"]].first().reset_index()


def _attach_prediction_lookup(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "segment", "configuration", "target"]
    attached = frame.merge(lookup, on=keys, how="left", validate="many_to_one", sort=False)
    if attached[["prediction_simple", "prediction_augmented"]].isna().any().any():
        raise ValueError("seed outcomes contain a cell absent from fitted point predictions")
    return attached


def _interleave_seed_rows(
    raw: Sequence[pd.DataFrame],
    *,
    seeds: Sequence[int],
    lookup: pd.DataFrame,
    first_cell_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, ...]]:
    if len(raw) != len(seeds) or not raw:
        raise ValueError("seed interleave input disagrees with its seed contract")
    attached = tuple(_attach_prediction_lookup(frame, lookup) for frame in raw)
    mean = attached[0].copy()
    mean["outcome_log_rmse"] = np.vstack(
        [frame["outcome_log_rmse"].to_numpy(dtype=float) for frame in attached]
    ).mean(axis=0)
    mean["model_seed"] = POINT_SEED_SENTINEL
    count = len(mean)
    seed_count = len(seeds)
    data: dict[str, Any] = {
        "evidence_schema_version": np.repeat(
            META_SEED_PREDICTION_SCHEMA_VERSION, count * seed_count
        ),
        "cell_index": np.repeat(
            np.arange(first_cell_index, first_cell_index + count, dtype=np.int64),
            seed_count,
        ),
    }
    for column in PREDICTION_EVIDENCE_IDENTIFIERS:
        data[column] = np.repeat(mean[column].to_numpy(), seed_count)
    data["model_seed"] = np.tile(np.asarray(seeds, dtype=np.int64), count)
    data["outcome_log_rmse"] = np.column_stack(
        [frame["outcome_log_rmse"].to_numpy(dtype=float) for frame in attached]
    ).reshape(-1)
    data["prediction_simple"] = np.repeat(
        mean["prediction_simple"].to_numpy(dtype=float), seed_count
    )
    data["prediction_augmented"] = np.repeat(
        mean["prediction_augmented"].to_numpy(dtype=float), seed_count
    )
    return pd.DataFrame(data), mean, attached


def _write_seed_evidence_and_sufficient(
    sources: Mapping[str, Path],
    predictions: pd.DataFrame,
    *,
    cohort: str,
    seed_path: Path,
    sufficient_path: Path,
    paired_sufficient_path: Path | None = None,
    batch_rows: int = STREAM_BATCH_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream authenticated seed outcomes and compact patient sufficient stats."""

    lookup = _prediction_lookup(predictions)
    writer = _AtomicParquetStreamWriter(seed_path)
    accumulator = _SufficientAccumulator(cohort)
    paired_accumulator = _PairedSeedSufficientAccumulator(cohort)
    cell_index = 0
    method_cells: dict[str, int] = {}
    try:
        for method in COMMON_PANEL_METHODS:
            seeds = _expected_common_seeds(method)
            method_start = cell_index
            for _mean_source, raw in _iter_aligned_seed_batches(
                sources[method],
                method=method,
                seeds=seeds,
                partitions=("test",),
                batch_rows=batch_rows,
            ):
                evidence, mean, attached = _interleave_seed_rows(
                    raw,
                    seeds=seeds,
                    lookup=lookup,
                    first_cell_index=cell_index,
                )
                writer.write(evidence)
                accumulator.add(mean, estimand="point_seed_mean")
                for seed_frame in attached:
                    accumulator.add(seed_frame, estimand="seed_specific")
                paired_accumulator.add(attached, seeds=seeds)
                cell_index += len(mean)
            method_cells[method] = cell_index - method_start
        writer.close()
    except Exception:
        writer.abort()
        raise
    sufficient = accumulator.frame()
    _validate_sufficient_contract(sufficient, cohort=cohort)
    sufficient.to_parquet(sufficient_path, index=False, compression="zstd")
    paired_sufficient = paired_accumulator.frame()
    _validate_paired_sufficient_contract(paired_sufficient, cohort=cohort)
    if paired_sufficient_path is not None:
        paired_sufficient.to_parquet(
            paired_sufficient_path, index=False, compression="zstd"
        )
    return sufficient, {
        "seed_prediction_schema_version": META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_schema_version": META_SUFFICIENT_SCHEMA_VERSION,
        "paired_sufficient_schema_version": META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
        "seed_prediction_rows": writer.rows,
        "seed_prediction_row_groups": writer.row_groups,
        "scientific_cells": cell_index,
        "method_scientific_cells": method_cells,
        "stream_batch_rows": batch_rows,
    }


def _stream_loco_prediction_evidence(
    sources: Mapping[str, Path],
    rank_map: pd.DataFrame,
    bank: LocoMetaModelBank,
    *,
    cohort: str,
    point_path: Path,
    seed_path: Path,
    sufficient_path: Path,
    paired_sufficient_path: Path | None = None,
    batch_rows: int = STREAM_BATCH_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one frozen fold-9 LOCO bank and stream all patient evidence."""

    point_writer = _AtomicParquetStreamWriter(point_path)
    seed_writer = _AtomicParquetStreamWriter(seed_path)
    accumulator = _SufficientAccumulator(cohort)
    paired_accumulator = _PairedSeedSufficientAccumulator(cohort)
    cell_index = 0
    method_cells: dict[str, int] = {}
    try:
        for method in COMMON_PANEL_METHODS:
            method_start = cell_index
            seeds = _expected_common_seeds(method)
            for mean, raw in _iter_aligned_seed_batches(
                sources[method],
                method=method,
                seeds=seeds,
                partitions=("test",),
                batch_rows=batch_rows,
            ):
                enriched = _attach_robust_map(mean, rank_map)
                predicted = predict_with_loco_meta_bank(bank, enriched)
                point_writer.write(predicted)
                evidence, _seed_mean, attached = _interleave_seed_rows(
                    raw,
                    seeds=seeds,
                    lookup=_prediction_lookup(predicted),
                    first_cell_index=cell_index,
                )
                seed_writer.write(evidence)
                accumulator.add(predicted, estimand="point_seed_mean")
                for seed_frame in attached:
                    accumulator.add(seed_frame, estimand="seed_specific")
                paired_accumulator.add(attached, seeds=seeds)
                cell_index += len(predicted)
            method_cells[method] = cell_index - method_start
        point_writer.close()
        seed_writer.close()
    except Exception:
        point_writer.abort()
        seed_writer.abort()
        raise
    sufficient = accumulator.frame()
    _validate_sufficient_contract(sufficient, cohort=cohort)
    sufficient.to_parquet(sufficient_path, index=False, compression="zstd")
    paired_sufficient = paired_accumulator.frame()
    _validate_paired_sufficient_contract(paired_sufficient, cohort=cohort)
    if paired_sufficient_path is not None:
        paired_sufficient.to_parquet(
            paired_sufficient_path, index=False, compression="zstd"
        )
    return sufficient, {
        "seed_prediction_schema_version": META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_schema_version": META_SUFFICIENT_SCHEMA_VERSION,
        "paired_sufficient_schema_version": META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
        "point_prediction_rows": point_writer.rows,
        "point_prediction_row_groups": point_writer.row_groups,
        "seed_prediction_rows": seed_writer.rows,
        "seed_prediction_row_groups": seed_writer.row_groups,
        "scientific_cells": cell_index,
        "method_scientific_cells": method_cells,
        "stream_batch_rows": batch_rows,
        "fold9_held_configurations": len(bank.simple),
        "fold9_simple_augmented_pairs": len(bank.simple),
        "fold9_loco_models": 2 * len(bank.simple),
        "fold9_model_bank_reused": True,
    }


def _validate_sufficient_contract(frame: pd.DataFrame, *, cohort: str) -> None:
    if set(frame.columns) != set(SUFFICIENT_COLUMNS):
        raise ValueError("sufficient-stat artifact does not use the exact schema")
    if frame.empty:
        raise ValueError("sufficient-stat artifact is empty")
    if set(frame["schema_version"].astype(str)) != {META_SUFFICIENT_SCHEMA_VERSION}:
        raise ValueError("sufficient-stat artifact uses the wrong schema")
    if set(frame["cohort"].astype(str)) != {cohort}:
        raise ValueError("sufficient-stat artifact uses the wrong cohort")
    identity = ["patient_id", "method", "model_seed", "estimand"]
    if frame.duplicated(identity).any():
        raise ValueError("sufficient-stat artifact duplicates patient/method/seed rows")
    numeric_columns = [
        "row_count",
        "truth_sum",
        "truth_square_sum",
        "simple_square_error",
        "augmented_square_error",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame["row_count"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("sufficient-stat artifact contains invalid numeric values")
    if not np.equal(
        frame["row_count"].to_numpy(dtype=float),
        np.floor(frame["row_count"].to_numpy(dtype=float)),
    ).all():
        raise ValueError("sufficient-stat row counts must be integers")
    if set(frame["method"].astype(str)) != set(COMMON_PANEL_METHODS):
        raise ValueError("sufficient-stat artifact lacks the frozen four-method panel")
    if set(frame["estimand"].astype(str)) != {
        "point_seed_mean",
        "seed_specific",
    }:
        raise ValueError("sufficient-stat artifact contains an unknown estimand")
    for (patient_id, method), rows in frame.groupby(
        ["patient_id", "method"], sort=False
    ):
        point = rows[rows["estimand"].astype(str) == "point_seed_mean"]
        seed_rows = rows[rows["estimand"].astype(str) == "seed_specific"]
        expected = _expected_common_seeds(str(method))
        if (
            len(point) != 1
            or int(point.iloc[0]["model_seed"]) != POINT_SEED_SENTINEL
            or set(seed_rows["model_seed"].astype(int)) != set(expected)
            or len(seed_rows) != len(expected)
        ):
            raise ValueError(
                f"{cohort} patient {patient_id} method {method} violates the exact seed contract"
            )
        counts = seed_rows["row_count"].to_numpy(dtype=np.int64)
        if not np.equal(counts, int(point.iloc[0]["row_count"])).all():
            raise ValueError("point and seed sufficient statistics cover different cells")
        if not np.isclose(
            float(point.iloc[0]["truth_sum"]),
            float(seed_rows["truth_sum"].mean()),
            rtol=0.0,
            atol=EFFECT_COMPARISON_ATOL,
        ):
            raise ValueError("point sufficient truth sum is not the seed-mean estimand")


def _json_numeric_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"paired sufficient {label} is not valid JSON") from exc
    if not isinstance(decoded, list) or len(decoded) != length:
        raise ValueError(f"paired sufficient {label} has the wrong length")
    try:
        vector = np.asarray(decoded, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"paired sufficient {label} is not numeric") from exc
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"paired sufficient {label} is not finite")
    return vector


def _paired_row_values(row: Mapping[str, Any]) -> dict[str, Any]:
    method = str(row["method"])
    expected_seeds = _expected_common_seeds(method)
    try:
        decoded_seeds = json.loads(str(row["model_seeds_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("paired sufficient model seeds are not valid JSON") from exc
    if not isinstance(decoded_seeds, list) or tuple(decoded_seeds) != expected_seeds:
        raise ValueError(f"{method} paired sufficient row uses the wrong seed contract")
    k = len(expected_seeds)
    row_count_value = row["row_count"]
    if isinstance(row_count_value, bool):
        raise ValueError("paired sufficient row count must be a positive integer")
    try:
        row_count_float = float(row_count_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired sufficient row count must be numeric") from exc
    if (
        not np.isfinite(row_count_float)
        or row_count_float <= 0
        or row_count_float != np.floor(row_count_float)
    ):
        raise ValueError("paired sufficient row count must be a positive integer")
    truth_sums = _json_numeric_vector(
        row["truth_sums_json"], length=k, label="truth sums"
    )
    truth_crossproducts = _json_numeric_vector(
        row["truth_crossproducts_json"],
        length=k * k,
        label="truth crossproducts",
    ).reshape(k, k)
    simple_truth_products = _json_numeric_vector(
        row["simple_truth_products_json"],
        length=k,
        label="simple truth products",
    )
    augmented_truth_products = _json_numeric_vector(
        row["augmented_truth_products_json"],
        length=k,
        label="augmented truth products",
    )
    if not np.allclose(
        truth_crossproducts,
        truth_crossproducts.T,
        rtol=0.0,
        atol=EFFECT_COMPARISON_ATOL,
    ):
        raise ValueError("paired sufficient truth crossproducts are not symmetric")
    prediction_squares = np.asarray(
        [
            row["simple_prediction_square_sum"],
            row["augmented_prediction_square_sum"],
        ],
        dtype=float,
    )
    if (
        not np.isfinite(prediction_squares).all()
        or (prediction_squares < -EFFECT_COMPARISON_ATOL).any()
        or (np.diag(truth_crossproducts) < -EFFECT_COMPARISON_ATOL).any()
    ):
        raise ValueError("paired sufficient square sums are invalid")
    return {
        "seeds": expected_seeds,
        "row_count": int(row_count_float),
        "truth_sums": truth_sums,
        "truth_crossproducts": truth_crossproducts,
        "simple_truth_products": simple_truth_products,
        "augmented_truth_products": augmented_truth_products,
        "prediction_squares": prediction_squares,
    }


def _validate_paired_sufficient_contract(
    frame: pd.DataFrame, *, cohort: str
) -> None:
    if set(frame.columns) != set(PAIRED_SEED_SUFFICIENT_COLUMNS):
        raise ValueError("paired sufficient artifact does not use the exact schema")
    if frame.empty:
        raise ValueError("paired sufficient artifact is empty")
    if set(frame["schema_version"].astype(str)) != {
        META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
    }:
        raise ValueError("paired sufficient artifact uses the wrong schema")
    if set(frame["cohort"].astype(str)) != {cohort}:
        raise ValueError("paired sufficient artifact uses the wrong cohort")
    if frame.duplicated(["patient_id", "method"]).any():
        raise ValueError("paired sufficient artifact duplicates patient/method rows")
    if set(frame["method"].astype(str)) != set(COMMON_PANEL_METHODS):
        raise ValueError("paired sufficient artifact lacks the frozen four-method panel")
    patient_sets: list[set[str]] = []
    row_counts: dict[tuple[str, str], int] = {}
    for method, rows in frame.groupby("method", sort=False):
        if str(method) not in COMMON_PANEL_METHODS:
            raise ValueError("paired sufficient artifact contains an unknown method")
        patient_sets.append(set(rows["patient_id"].astype(str)))
        for row in rows.to_dict(orient="records"):
            values = _paired_row_values(row)
            row_counts[(str(row["patient_id"]), str(method))] = values["row_count"]
    if not patient_sets or any(patients != patient_sets[0] for patients in patient_sets[1:]):
        raise ValueError("paired sufficient methods do not cover the same patients")
    for patient in patient_sets[0]:
        counts = {row_counts[(patient, method)] for method in COMMON_PANEL_METHODS}
        if len(counts) != 1:
            raise ValueError(
                "paired sufficient methods do not cover the same cells for each patient"
            )


def _prepare_paired_method_arrays(
    frame: pd.DataFrame, *, cohort: str
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    _validate_paired_sufficient_contract(frame, cohort=cohort)
    patients = tuple(sorted(frame["patient_id"].astype(str).unique()))
    arrays: dict[str, dict[str, Any]] = {}
    for method in COMMON_PANEL_METHODS:
        method_rows = (
            frame[frame["method"].astype(str) == method]
            .assign(patient_id=lambda value: value["patient_id"].astype(str))
            .set_index("patient_id")
            .reindex(patients)
        )
        decoded = [_paired_row_values(row) for row in method_rows.to_dict(orient="records")]
        arrays[method] = {
            "seeds": _expected_common_seeds(method),
            "row_count": np.asarray([value["row_count"] for value in decoded], dtype=float),
            "truth_sums": np.stack([value["truth_sums"] for value in decoded]),
            "truth_crossproducts": np.stack(
                [value["truth_crossproducts"] for value in decoded]
            ),
            "simple_truth_products": np.stack(
                [value["simple_truth_products"] for value in decoded]
            ),
            "augmented_truth_products": np.stack(
                [value["augmented_truth_products"] for value in decoded]
            ),
            "prediction_squares": np.stack(
                [value["prediction_squares"] for value in decoded]
            ),
        }
    return patients, arrays


def _patient_moments_for_seed_multiplicities(
    values: Mapping[str, Any], multiplicities: Sequence[int]
) -> np.ndarray:
    counts = np.asarray(multiplicities, dtype=float)
    seeds = tuple(values["seeds"])
    if (
        counts.shape != (len(seeds),)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.equal(counts, np.floor(counts)).all()
        or int(counts.sum()) != len(seeds)
    ):
        raise ValueError("seed multiplicities must be non-negative integers summing to seed count")
    weights = counts / len(seeds)
    truth_sum = values["truth_sums"] @ weights
    truth_square_sum = np.einsum(
        "i,nij,j->n",
        weights,
        values["truth_crossproducts"],
        weights,
        optimize=True,
    )
    simple_square_error = (
        truth_square_sum
        - 2.0 * (values["simple_truth_products"] @ weights)
        + values["prediction_squares"][:, 0]
    )
    augmented_square_error = (
        truth_square_sum
        - 2.0 * (values["augmented_truth_products"] @ weights)
        + values["prediction_squares"][:, 1]
    )
    moments = np.column_stack(
        (
            truth_sum,
            truth_square_sum,
            simple_square_error,
            augmented_square_error,
            values["row_count"],
        )
    )
    if not np.isfinite(moments).all():
        raise ValueError("paired sufficient reconstruction produced non-finite moments")
    return moments


def _delta_from_sufficient_rows(frame: pd.DataFrame) -> float:
    totals = frame[
        [
            "truth_sum",
            "truth_square_sum",
            "simple_square_error",
            "augmented_square_error",
            "row_count",
        ]
    ].sum().to_numpy(dtype=float)
    truth_sum, truth_square_sum, simple_error, augmented_error, row_count = totals
    denominator = float(truth_square_sum - truth_sum**2 / row_count)
    if row_count <= 0 or denominator <= 0:
        raise ValueError("R2 is undefined for the sufficient-stat artifact")
    return float(simple_error / denominator - augmented_error / denominator)


def _method_deltas_from_sufficient(frame: pd.DataFrame) -> dict[str, float]:
    point = frame[frame["estimand"].astype(str) == "point_seed_mean"]
    return {
        str(method): _delta_from_sufficient_rows(rows)
        for method, rows in point.groupby("method", sort=False)
    }


def _bootstrap_effect_and_draws_from_sufficient(
    sufficient: pd.DataFrame,
    *,
    paired_sufficient: pd.DataFrame,
    cohort: str,
    replicates: int,
    seed: int,
) -> tuple[BootstrapEffect, pd.DataFrame]:
    """Bootstrap the five-run seed-mean estimand from paired patient moments."""

    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    _validate_sufficient_contract(sufficient, cohort=cohort)
    point_rows = sufficient[
        sufficient["estimand"].astype(str) == "point_seed_mean"
    ]
    point = _delta_from_sufficient_rows(point_rows)
    patients, method_arrays = _prepare_paired_method_arrays(
        paired_sufficient, cohort=cohort
    )
    if len(patients) < 2:
        raise ValueError("patient bootstrap requires at least two patients")
    if set(point_rows["patient_id"].astype(str)) != set(patients):
        raise ValueError("point and paired sufficient artifacts cover different patients")

    cached_method_moments: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

    def method_moments(method: str, counts: tuple[int, ...]) -> np.ndarray:
        key = (method, counts)
        cached = cached_method_moments.get(key)
        if cached is None:
            cached = _patient_moments_for_seed_multiplicities(
                method_arrays[method], counts
            )
            cached_method_moments[key] = cached
        return cached

    point_patient_moments = np.zeros((len(patients), 5), dtype=float)
    for method in COMMON_PANEL_METHODS:
        point_patient_moments += method_moments(
            method, tuple(1 for _seed in _expected_common_seeds(method))
        )
    paired_point_totals = point_patient_moments.sum(axis=0)
    paired_point_denominator = float(
        paired_point_totals[1]
        - paired_point_totals[0] ** 2 / paired_point_totals[4]
    )
    if paired_point_totals[4] <= 0 or paired_point_denominator <= 0:
        raise ValueError("R2 is undefined for paired point sufficient statistics")
    paired_point = float(
        paired_point_totals[2] / paired_point_denominator
        - paired_point_totals[3] / paired_point_denominator
    )
    if not np.isclose(point, paired_point, rtol=0.0, atol=EFFECT_COMPARISON_ATOL):
        raise ValueError(
            "paired seed moments do not reconstruct the authenticated five-seed point estimand"
        )

    rng = np.random.default_rng(seed)
    rows_out: list[dict[str, Any]] = []
    attempts = 0
    maximum_attempts = max(replicates * 10, replicates + 1_000)
    while len(rows_out) < replicates and attempts < maximum_attempts:
        attempt_index = attempts
        attempts += 1
        selected_seeds: dict[str, list[int]] = {}
        selected_counts: dict[str, tuple[int, ...]] = {}
        for method in COMMON_PANEL_METHODS:
            seeds = _expected_common_seeds(method)
            if len(seeds) > 1:
                selected_indices = rng.choice(
                    np.arange(len(seeds), dtype=np.int64),
                    size=len(seeds),
                    replace=True,
                )
                selected_seeds[method] = [
                    int(seeds[int(index)]) for index in selected_indices
                ]
                counts = tuple(
                    int(value)
                    for value in np.bincount(
                        selected_indices, minlength=len(seeds)
                    )
                )
            else:
                counts = (1,)
            selected_counts[method] = counts
        sampled_indices = rng.choice(
            np.arange(len(patients), dtype=np.int64),
            size=len(patients),
            replace=True,
        )
        multiplicities = np.bincount(
            sampled_indices,
            minlength=len(patients),
        ).astype(float)
        totals = np.zeros(5, dtype=float)
        for method in COMMON_PANEL_METHODS:
            totals += multiplicities @ method_moments(
                method, selected_counts[method]
            )
        truth_sum, truth_square_sum, simple_error, augmented_error, row_count = totals
        denominator = float(truth_square_sum - truth_sum**2 / row_count)
        if row_count <= 0 or denominator <= 0:
            continue
        rows_out.append(
            {
                "schema_version": META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
                "cohort": cohort,
                "bootstrap_index": len(rows_out),
                "attempt_index": attempt_index,
                "base_seed": seed,
                "n_patients": len(patients),
                "patient_draw_sha256": lineage.canonical_sha256(
                    [patients[int(index)] for index in sampled_indices]
                ),
                "selected_model_seeds_json": json.dumps(
                    selected_seeds, sort_keys=True, separators=(",", ":")
                ),
                "delta_r2": float(
                    simple_error / denominator - augmented_error / denominator
                ),
            }
        )
    if len(rows_out) != replicates:
        raise RuntimeError(
            "could not obtain the requested number of valid patient bootstrap "
            f"replicates ({len(rows_out)}/{replicates} after {attempts} attempts)"
        )
    draws = pd.DataFrame(rows_out)
    lower, upper = np.percentile(draws["delta_r2"].to_numpy(dtype=float), [2.5, 97.5])
    return BootstrapEffect(point, (float(lower), float(upper)), replicates, seed), draws


def _bootstrap_effect_and_draws(
    predictions: pd.DataFrame,
    *,
    bootstrap_predictions: pd.DataFrame,
    cohort: str,
    replicates: int,
    seed: int,
) -> tuple[BootstrapEffect, pd.DataFrame]:
    """Return the effect and the complete deterministic patient/seed draw audit."""

    required = {
        "patient_id",
        "outcome_log_rmse",
        "prediction_simple",
        "prediction_augmented",
    }
    for label, frame in (
        ("point", predictions), ("seed-specific bootstrap", bootstrap_predictions)
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions lack columns: {sorted(missing)}")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")

    point = prediction_delta_r2(predictions)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    maximum_attempts = max(replicates * 10, replicates + 1_000)
    while len(rows) < replicates and attempts < maximum_attempts:
        attempt_index = attempts
        attempts += 1
        sample, sampled, selected_seeds = _bootstrap_rows_with_audit(
            bootstrap_predictions, rng
        )
        try:
            delta = prediction_delta_r2(sample)
        except ValueError:
            continue
        rows.append(
            {
                "schema_version": META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
                "cohort": cohort,
                "bootstrap_index": len(rows),
                "attempt_index": attempt_index,
                "base_seed": seed,
                "n_patients": len(sampled),
                "patient_draw_sha256": lineage.canonical_sha256(
                    list(sampled)
                ),
                "selected_model_seeds_json": json.dumps(
                    {method: list(values) for method, values in selected_seeds.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "delta_r2": delta,
            }
        )
    if len(rows) != replicates:
        raise RuntimeError(
            "could not obtain the requested number of valid patient bootstrap "
            f"replicates ({len(rows)}/{replicates} after {attempts} attempts)"
        )
    draws = pd.DataFrame(rows)
    lower, upper = np.percentile(draws["delta_r2"].to_numpy(dtype=float), [2.5, 97.5])
    return (
        BootstrapEffect(float(point), (float(lower), float(upper)), replicates, seed),
        draws,
    )


def _validate_seed_prediction_binding(
    point_predictions: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Prove seed outcomes use exactly the authenticated seed-mean predictions."""

    identifiers = list(PREDICTION_EVIDENCE_IDENTIFIERS)
    required_point = {
        *identifiers,
        "outcome_log_rmse",
        "prediction_simple",
        "prediction_augmented",
    }
    required_seed = {*required_point, "model_seed"}
    if required_point - set(point_predictions.columns) or required_seed - set(
        seed_predictions.columns
    ):
        raise ValueError(f"{label} point/seed prediction binding columns are incomplete")
    if point_predictions.duplicated(identifiers).any() or seed_predictions.duplicated(
        [*identifiers, "model_seed"]
    ).any():
        raise ValueError(f"{label} point/seed predictions contain duplicate scientific cells")
    attached = seed_predictions.merge(
        point_predictions[
            [
                *identifiers,
                "outcome_log_rmse",
                "prediction_simple",
                "prediction_augmented",
            ]
        ],
        on=identifiers,
        how="left",
        validate="many_to_one",
        suffixes=("_seed", "_point"),
    )
    numeric_pairs = (
        ("prediction_simple_seed", "prediction_simple_point"),
        ("prediction_augmented_seed", "prediction_augmented_point"),
    )
    for seed_column, point_column in numeric_pairs:
        if attached[point_column].isna().any() or not np.allclose(
            attached[seed_column].to_numpy(dtype=float),
            attached[point_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=EFFECT_COMPARISON_ATOL,
        ):
            raise ValueError(f"{label} seed-specific rows change authenticated predictions")
    seed_means = (
        seed_predictions.groupby(identifiers, sort=False, dropna=False)["outcome_log_rmse"]
        .mean()
        .reset_index()
    )
    point_outcomes = point_predictions[[*identifiers, "outcome_log_rmse"]]
    outcomes = seed_means.merge(
        point_outcomes,
        on=identifiers,
        how="outer",
        validate="one_to_one",
        suffixes=("_seed_mean", "_point"),
    )
    if outcomes.isna().any().any() or not np.allclose(
        outcomes["outcome_log_rmse_seed_mean"].to_numpy(dtype=float),
        outcomes["outcome_log_rmse_point"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EFFECT_COMPARISON_ATOL,
    ):
        raise ValueError(f"{label} point outcomes are not the seed-specific mean estimand")


def _validate_streamed_point_seed_binding(
    point_path: Path,
    seed_path: Path,
    *,
    cohort: str,
    batch_rows: int = STREAM_BATCH_ROWS,
) -> dict[str, int]:
    """Bind both prediction artifacts while materialising only one bounded block.

    The seed writer never splits one scientific cell across row groups.  Point
    rows may use different row-group boundaries, so a cursor takes exactly the
    number of cells represented by each seed block.  The existing outer-join
    check then proves both directions: every point cell has seed outcomes and
    every seed cell has one authenticated point prediction.
    """

    _pa, _ds, pq = _pyarrow_modules()
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("point/seed binding batch size must be a positive integer")
    point_columns = tuple(PREDICTION_EVIDENCE_COLUMNS)
    seed_columns = (
        "evidence_schema_version",
        "cell_index",
        *PREDICTION_EVIDENCE_COLUMNS,
        "model_seed",
    )
    point_names = set(_parquet_schema_names(point_path))
    seed_names = set(_parquet_schema_names(seed_path))
    if set(point_columns) - point_names or set(seed_columns) - seed_names:
        raise ValueError(f"{cohort} point/seed prediction binding columns are incomplete")
    point_cursor = _BatchCursor(
        _iter_parquet_batches(
            (point_path,), columns=point_columns, batch_rows=batch_rows
        ),
        point_columns,
    )
    seed_parquet = pq.ParquetFile(seed_path)
    point_rows = 0
    seed_rows = 0
    maximum_point_block_rows = 0
    maximum_seed_block_rows = 0
    for row_group_index in range(seed_parquet.metadata.num_row_groups):
        seed_frame = seed_parquet.read_row_group(
            row_group_index, columns=list(seed_columns), use_threads=False
        ).to_pandas()
        if seed_frame.empty:
            raise ValueError(f"{cohort} seed-prediction evidence has an empty row group")
        cell_ids = pd.to_numeric(seed_frame["cell_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        starts = np.r_[0, np.flatnonzero(cell_ids[1:] != cell_ids[:-1]) + 1]
        point_frame = point_cursor.take(len(starts))
        if len(point_frame) != len(starts):
            raise ValueError(f"{cohort} point predictions omit seed-evidence cells")
        for label, frame in (("point", point_frame), ("seed", seed_frame)):
            if (
                set(frame["cohort"].astype(str)) != {cohort}
                or set(frame["partition"].astype(str)) != {"test"}
            ):
                raise ValueError(
                    f"{cohort} {label} predictions use the wrong cohort/partition"
                )
        _validate_seed_prediction_binding(
            point_frame,
            seed_frame,
            label=cohort,
        )
        point_rows += len(point_frame)
        seed_rows += len(seed_frame)
        maximum_point_block_rows = max(maximum_point_block_rows, len(point_frame))
        maximum_seed_block_rows = max(maximum_seed_block_rows, len(seed_frame))
    if not point_cursor.take(1).empty:
        raise ValueError(f"{cohort} point predictions contain cells absent from seed evidence")
    if point_rows < 1 or seed_rows < 1:
        raise ValueError(f"{cohort} point/seed prediction evidence is empty")
    return {
        "point_rows": point_rows,
        "seed_rows": seed_rows,
        "seed_row_groups": seed_parquet.metadata.num_row_groups,
        "max_point_block_rows": maximum_point_block_rows,
        "max_seed_block_rows": maximum_seed_block_rows,
    }


def _read_small_parquet(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a contractually compact artifact through Arrow, never pandas' bulk path."""

    _pa, _ds, pq = _pyarrow_modules()
    return pq.read_table(path, columns=None if columns is None else list(columns)).to_pandas()


def _rebuild_statistics_from_seed_evidence(
    path: Path,
    *,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Independently stream every seed row and reconstruct Stage-15 statistics."""

    _pa, _ds, pq = _pyarrow_modules()
    parquet = pq.ParquetFile(path)
    required = {
        "evidence_schema_version",
        "cell_index",
        *PREDICTION_EVIDENCE_COLUMNS,
        "model_seed",
    }
    if set(parquet.schema_arrow.names) != required:
        raise ValueError("seed-prediction evidence does not use the exact schema")
    accumulator = _SufficientAccumulator(cohort)
    paired_accumulator = _PairedSeedSufficientAccumulator(cohort)
    expected_cell_index = 0
    seed_rows_seen = 0
    methods_finished: set[str] = set()
    current_method: str | None = None
    method_cells = {method: 0 for method in COMMON_PANEL_METHODS}
    method_cell_hashers = {
        method: hashlib.sha256() for method in COMMON_PANEL_METHODS
    }
    last_order_key: dict[str, tuple[int, str, str, str]] = {}
    configuration_index = {
        "+".join(configuration): index
        for index, configuration in enumerate(deep_configuration_panel())
    }
    evidence_columns = [
        "evidence_schema_version",
        "cell_index",
        *PREDICTION_EVIDENCE_COLUMNS,
        "model_seed",
    ]
    for row_group_index in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(
            row_group_index, columns=evidence_columns, use_threads=False
        ).to_pandas()
        if frame.empty:
            raise ValueError("seed-prediction evidence contains an empty row group")
        if (
            set(frame["evidence_schema_version"].astype(str))
            != {META_SEED_PREDICTION_SCHEMA_VERSION}
            or set(frame["cohort"].astype(str)) != {cohort}
            or set(frame["partition"].astype(str)) != {"test"}
        ):
            raise ValueError("seed-prediction evidence uses the wrong schema/cohort/partition")
        numeric = frame[
            [
                "outcome_log_rmse",
                "prediction_simple",
                "prediction_augmented",
            ]
        ].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("seed-prediction evidence contains non-finite values")
        cell_ids = pd.to_numeric(frame["cell_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        starts = np.r_[0, np.flatnonzero(cell_ids[1:] != cell_ids[:-1]) + 1]
        lengths = np.diff(np.r_[starts, len(frame)])
        unique_ids = cell_ids[starts]
        expected_ids = np.arange(
            expected_cell_index, expected_cell_index + len(unique_ids), dtype=np.int64
        )
        if not np.array_equal(unique_ids, expected_ids):
            raise ValueError(
                "seed-prediction scientific cells are duplicated, reordered, or split across row groups"
            )
        repeated_starts = np.repeat(starts, lengths)
        for column in PREDICTION_EVIDENCE_IDENTIFIERS:
            values = frame[column].astype(str).to_numpy()
            if not np.array_equal(values, values[repeated_starts]):
                raise ValueError(f"seed rows change scientific identifier {column!r} within a cell")
        for column in ("prediction_simple", "prediction_augmented"):
            values = frame[column].to_numpy(dtype=float)
            if not np.allclose(
                values,
                values[repeated_starts],
                rtol=0.0,
                atol=EFFECT_COMPARISON_ATOL,
            ):
                raise ValueError(f"seed rows change {column} within a scientific cell")

        cell_methods = frame["method"].astype(str).to_numpy()[starts]
        if any(method not in COMMON_PANEL_METHODS for method in cell_methods):
            raise ValueError("seed-prediction evidence contains a non-common-panel method")
        seeds = pd.to_numeric(frame["model_seed"], errors="raise").to_numpy(dtype=np.int64)
        offsets = np.arange(len(frame), dtype=np.int64) - repeated_starts
        expected_seed_values = np.empty(len(frame), dtype=np.int64)
        for method in COMMON_PANEL_METHODS:
            cells = cell_methods == method
            if not cells.any():
                continue
            contract = np.asarray(_expected_common_seeds(method), dtype=np.int64)
            if not np.equal(lengths[cells], len(contract)).all():
                raise ValueError(f"{method} does not have the exact preregistered seeds per cell")
            row_mask = np.repeat(cells, lengths)
            expected_seed_values[row_mask] = contract[offsets[row_mask]]
        if not np.array_equal(seeds, expected_seed_values):
            raise ValueError("seed rows are missing, duplicated, or outside the preregistered set")

        cell_frame = frame.iloc[starts].copy().reset_index(drop=True)
        cell_frame["outcome_log_rmse"] = np.add.reduceat(
            frame["outcome_log_rmse"].to_numpy(dtype=float), starts
        ) / lengths
        cell_frame["model_seed"] = POINT_SEED_SENTINEL
        row_group_methods = np.unique(cell_methods)
        if len(row_group_methods) != 1:
            raise ValueError("seed-prediction row groups must not mix method blocks")
        method = str(row_group_methods[0])
        contract = _expected_common_seeds(method)
        seed_frames = tuple(
            frame.iloc[offset :: len(contract)].reset_index(drop=True)
            for offset in range(len(contract))
        )
        paired_accumulator.add(seed_frames, seeds=contract)
        if current_method is None:
            current_method = method
        elif method != current_method:
            methods_finished.add(current_method)
            if method in methods_finished:
                raise ValueError("seed-prediction methods are not in contiguous blocks")
            current_method = method
        config_codes = cell_frame["configuration"].map(configuration_index)
        if config_codes.isna().any():
            raise ValueError("seed-prediction evidence contains an unknown configuration")
        order = pd.MultiIndex.from_arrays(
            [
                config_codes.to_numpy(dtype=np.int64),
                cell_frame["patient_id"].astype(str).to_numpy(),
                cell_frame["segment"].astype(str).to_numpy(),
                cell_frame["target"].astype(str).to_numpy(),
            ]
        )
        first_order = tuple(order[0])
        previous = last_order_key.get(method)
        if (
            not order.is_monotonic_increasing
            or order.has_duplicates
            or (previous is not None and first_order <= previous)
        ):
            raise ValueError(
                f"{method} scientific cells are duplicated or not in frozen source order"
            )
        last_order_key[method] = tuple(order[-1])
        method_cells[method] += len(cell_frame)
        digest_columns = tuple(
            column
            for column in PREDICTION_EVIDENCE_IDENTIFIERS
            if column != "method"
        )
        hasher = method_cell_hashers[method]
        for values in cell_frame.loc[:, digest_columns].itertuples(
            index=False, name=None
        ):
            for raw_value in values:
                encoded = str(raw_value).encode("utf-8")
                hasher.update(len(encoded).to_bytes(8, byteorder="big"))
                hasher.update(encoded)
        accumulator.add(cell_frame, estimand="point_seed_mean")
        accumulator.add(frame, estimand="seed_specific")
        expected_cell_index += len(unique_ids)
        seed_rows_seen += len(frame)

    sufficient = accumulator.frame()
    _validate_sufficient_contract(sufficient, cohort=cohort)
    paired_sufficient = paired_accumulator.frame()
    _validate_paired_sufficient_contract(paired_sufficient, cohort=cohort)
    method_cell_sha256 = {
        method: method_cell_hashers[method].hexdigest()
        for method in COMMON_PANEL_METHODS
    }
    if (
        set(last_order_key) != set(COMMON_PANEL_METHODS)
        or len(set(method_cells.values())) != 1
        or len(set(method_cell_sha256.values())) != 1
    ):
        raise ValueError("seed-prediction evidence lacks an equal four-method common panel")
    return sufficient, paired_sufficient, {
        "seed_prediction_schema_version": META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_schema_version": META_SUFFICIENT_SCHEMA_VERSION,
        "paired_sufficient_schema_version": META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION,
        "seed_prediction_rows": seed_rows_seen,
        "seed_prediction_row_groups": parquet.metadata.num_row_groups,
        "scientific_cells": expected_cell_index,
        "method_scientific_cells": method_cells,
        "common_scientific_cells_sha256": next(iter(method_cell_sha256.values())),
        "exact_seed_contract": {
            method: list(_expected_common_seeds(method)) for method in COMMON_PANEL_METHODS
        },
    }


def _rebuild_sufficient_from_seed_evidence(
    path: Path,
    *,
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sufficient, _paired, report = _rebuild_statistics_from_seed_evidence(
        path, cohort=cohort
    )
    return sufficient, report


def _rebuild_paired_sufficient_from_seed_evidence(
    path: Path,
    *,
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _sufficient, paired, report = _rebuild_statistics_from_seed_evidence(
        path, cohort=cohort
    )
    return paired, report


def _require_sufficient_equal(
    stored: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    label: str,
) -> None:
    keys = ["cohort", "patient_id", "method", "model_seed", "estimand"]
    for frame in (stored, rebuilt):
        _validate_sufficient_contract(frame, cohort=label)
    left = stored.sort_values(keys).reset_index(drop=True)
    right = rebuilt.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].astype(str).equals(
        right[keys].astype(str)
    ):
        raise ValueError(f"{label} sufficient-stat identities disagree with seed evidence")
    if not left["schema_version"].astype(str).equals(right["schema_version"].astype(str)):
        raise ValueError(f"{label} sufficient-stat schema disagrees with seed evidence")
    if not np.array_equal(
        left["row_count"].to_numpy(dtype=np.int64),
        right["row_count"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"{label} sufficient-stat row counts disagree with seed evidence")
    numeric = [
        "truth_sum",
        "truth_square_sum",
        "simple_square_error",
        "augmented_square_error",
    ]
    if not np.allclose(
        left[numeric].to_numpy(dtype=float),
        right[numeric].to_numpy(dtype=float),
        rtol=0.0,
        atol=EFFECT_COMPARISON_ATOL,
    ):
        raise ValueError(f"{label} sufficient statistics disagree with seed evidence")


def _require_paired_sufficient_equal(
    stored: pd.DataFrame,
    rebuilt: pd.DataFrame,
    *,
    label: str,
) -> None:
    keys = ["cohort", "patient_id", "method"]
    for frame in (stored, rebuilt):
        _validate_paired_sufficient_contract(frame, cohort=label)
    left = stored.sort_values(keys).reset_index(drop=True)
    right = rebuilt.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].astype(str).equals(
        right[keys].astype(str)
    ):
        raise ValueError(
            f"{label} paired sufficient identities disagree with seed evidence"
        )
    for left_row, right_row in zip(
        left.to_dict(orient="records"), right.to_dict(orient="records"), strict=True
    ):
        left_values = _paired_row_values(left_row)
        right_values = _paired_row_values(right_row)
        if (
            left_values["seeds"] != right_values["seeds"]
            or left_values["row_count"] != right_values["row_count"]
        ):
            raise ValueError(
                f"{label} paired sufficient contract disagrees with seed evidence"
            )
        for field in (
            "truth_sums",
            "truth_crossproducts",
            "simple_truth_products",
            "augmented_truth_products",
            "prediction_squares",
        ):
            if not np.allclose(
                left_values[field],
                right_values[field],
                rtol=0.0,
                atol=EFFECT_COMPARISON_ATOL,
            ):
                raise ValueError(
                    f"{label} paired sufficient statistics disagree with seed evidence"
                )


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


def _ecgrecover_summary_stream(path: Path) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], list[Any]] = {}
    for chunk in _iter_parquet_batches(
        (path,),
        columns=(
            "cohort",
            "partition",
            "patient_id",
            "method",
            "segment",
            "outcome_log_rmse",
        ),
        filters={"method": ("ecgrecover",)},
    ):
        for key, rows in chunk.groupby(["cohort", "partition", "segment"], sort=False):
            state = totals.setdefault(tuple(str(value) for value in key), [0.0, 0, set()])
            state[0] += float(rows["outcome_log_rmse"].sum())
            state[1] += int(len(rows))
            state[2].update(rows["patient_id"].astype(str))
    return [
        {
            "cohort": cohort,
            "partition": partition,
            "segment": segment,
            "mean_log_rmse": total / count,
            "patient_metric_rows": count,
            "patients": len(patients),
        }
        for (cohort, partition, segment), (total, count, patients) in sorted(totals.items())
    ]


def analyze(arguments: argparse.Namespace) -> None:
    rank_root = arguments.rank_maps.resolve()
    rank_map_path = rank_root / "map_cells.parquet"
    if not rank_map_path.is_file():
        raise FileNotFoundError(rank_map_path)
    release_lineage: dict[str, Any] = {}
    predictor_contract: PredictorContract | None = None
    ptb_coverages: dict[str, PartitionCoverage] = {}
    external_manifest_by_cohort: dict[str, Path] = {}
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
        split_hashes = {
            str(bundle.summary.get("manifest", {}).get("split_sha256", ""))
            for bundle in validated_bundles.values()
        }
        if len(split_hashes) != 1 or len(next(iter(split_hashes))) != 64:
            raise ValueError("benchmark bundles disagree on the PTB-XL split hash")
        ptb_sources = load_ptbxl_source_partitions(
            arguments.ptbxl_manifest,
            expected_manifest_sha256=source_manifest_sha256,
            expected_split_sha256=next(iter(split_hashes)),
        )
        ptb_coverages, ptb_audit_report = _benchmark_release_coverages(
            validated_bundles,
            sources=ptb_sources,
        )
        common_configurations = tuple(
            "+".join(configuration) for configuration in deep_configuration_panel()
        )
        predictor_contract = load_common_predictor_contract(
            validated_bundles,
            methods=COMMON_PANEL_METHODS,
            configurations=common_configurations,
        )
        external_manifest_by_cohort = _external_manifest_index(
            arguments.external_manifest
        )
        release_lineage = {
            "rank_map_summary_sha256": lineage.artifact_sha256(
                rank_root / "summary.v3.json"
            ),
            "rank_maps_sha256": rank_maps_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_artifact_sha256": lineage.artifact_sha256(
                arguments.ptbxl_manifest
            ),
            "simple_predictors": predictor_contract.evidence(),
            "ptbxl_audits": ptb_audit_report,
            "rank_map_bootstrap_evidence_sha256": lineage.canonical_sha256(
                {
                    key: rank_summary["artifact_sha256"][key]
                    for key in (
                        "rank_path",
                        "map_cells",
                        "bootstrap_draws",
                        "bootstrap_patients",
                        "bootstrap_multiplicities",
                        "bootstrap_audit",
                        "bootstrap_moments",
                        "bootstrap_attempts",
                    )
                }
            ),
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
    ptb_metric_sources = _benchmark_metric_sources(
        arguments.benchmark,
        validated_bundles=validated_bundles if arguments.release else None,
    )
    common_ptb_sources = {
        method: ptb_metric_sources[method] for method in COMMON_PANEL_METHODS
    }
    required_partitions = {"tune", "calibration", "test"}
    _validate_metric_source_partitions(
        common_ptb_sources, expected=required_partitions
    )
    if arguments.release:
        assert predictor_contract is not None
        ptb_common_panel_report = _streaming_panel_report(
            tuple(ptb_metric_sources[method] for method in COMMON_PANEL_METHODS),
            cohort="PTB-XL",
            coverages=ptb_coverages,
            method_seeds={
                method: validated_bundles[method].seeds
                for method in COMMON_PANEL_METHODS
            },
            configurations=tuple(
                "+".join(configuration) for configuration in deep_configuration_panel()
            ),
            predictor=predictor_contract,
        )
        ptb_ecgrecover_panel_report = _streaming_panel_report(
            (ptb_metric_sources["ecgrecover"],),
            cohort="PTB-XL",
            coverages=ptb_coverages,
            method_seeds={"ecgrecover": validated_bundles["ecgrecover"].seeds},
            configurations=tuple(
                "+".join(configuration)
                for configuration in validated_bundles["ecgrecover"].configurations
            ),
            predictor=predictor_contract,
        )
        release_lineage["panel_completeness"] = {
            "PTB-XL": {
                "common_panel": ptb_common_panel_report,
                "ecgrecover": ptb_ecgrecover_panel_report,
            }
        }
    tune_statistics = _meta_sufficient_for_partition(
        common_ptb_sources, rank_map, partition="tune"
    )
    selection = tune_meta_ridge_alpha_from_sufficient(
        tune_statistics, grid=META_RIDGE_ALPHA_GRID
    )
    calibration_statistics = _meta_sufficient_for_partition(
        common_ptb_sources, rank_map, partition="calibration"
    )
    if tune_statistics.encoding != calibration_statistics.encoding:
        raise RuntimeError("fold-8 and fold-9 meta encodings are not identical")
    fold9_model_bank = fit_loco_meta_model_bank(
        calibration_statistics, alpha=selection.alpha
    )
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    alpha_path = output / "alpha_tuning.parquet"
    ptb_path = output / "ptbxl_predictions.parquet"
    ptb_seed_path = output / "ptbxl_seed_predictions.parquet"
    ptb_sufficient_path = output / "ptbxl_sufficient_stats.parquet"
    ptb_paired_sufficient_path = output / "ptbxl_paired_seed_sufficient.parquet"
    selection.table.to_parquet(alpha_path, index=False, compression="zstd")
    ptb_sufficient, ptb_stream_report = _stream_loco_prediction_evidence(
        common_ptb_sources,
        rank_map,
        fold9_model_bank,
        cohort="PTB-XL",
        point_path=ptb_path,
        seed_path=ptb_seed_path,
        sufficient_path=ptb_sufficient_path,
        paired_sufficient_path=ptb_paired_sufficient_path,
    )
    ptb_paired_sufficient = _read_small_parquet(ptb_paired_sufficient_path)
    ptb_effect, ptb_bootstrap_draws = _bootstrap_effect_and_draws_from_sufficient(
        ptb_sufficient,
        paired_sufficient=ptb_paired_sufficient,
        cohort="PTB-XL",
        replicates=arguments.bootstrap_replicates,
        seed=arguments.seed + META_BOOTSTRAP_SEED_OFFSETS["PTB-XL"],
    )

    external_effects = {}
    external_stream_reports: dict[str, dict[str, Any]] = {}
    external_bootstrap_draws = {}
    external_hashes = {}
    external_release_reports = {}
    external_artifacts = {}
    ecgrecover_supplement = _ecgrecover_summary_stream(
        ptb_metric_sources["ecgrecover"]
    )
    for bundle in arguments.external:
        if arguments.release:
            summary_hint = _load_json(bundle.resolve() / "summary.v3.json")
            cohort_hint = str(summary_hint.get("cohort", ""))
            if cohort_hint not in external_manifest_by_cohort:
                raise ValueError(f"no unique source manifest for external cohort {cohort_hint!r}")
            assert predictor_contract is not None
            external_metrics_path, cohort, release_report, external_coverage = (
                _validate_external_release_bundle(
                bundle,
                target_manifest=external_manifest_by_cohort[cohort_hint],
                source_manifest_sha256=source_manifest_sha256,
                rank_maps_sha256=rank_maps_sha256,
                predictor_content_sha256=predictor_contract.content_sha256,
                benchmark_bundles=validated_bundles,
                )
            )
            common_panel_report = _streaming_panel_report(
                (external_metrics_path,),
                cohort=cohort,
                coverages={"test": external_coverage},
                method_seeds={
                    method: validated_bundles[method].seeds
                    for method in COMMON_PANEL_METHODS
                },
                configurations=tuple(
                    "+".join(configuration)
                    for configuration in deep_configuration_panel()
                ),
                predictor=predictor_contract,
            )
            ecgrecover_panel_report = _streaming_panel_report(
                (external_metrics_path,),
                cohort=cohort,
                coverages={"test": external_coverage},
                method_seeds={"ecgrecover": validated_bundles["ecgrecover"].seeds},
                configurations=tuple(
                    "+".join(configuration)
                    for configuration in validated_bundles["ecgrecover"].configurations
                ),
                predictor=predictor_contract,
            )
            release_report["panel_completeness"] = {
                "common_panel": common_panel_report,
                "ecgrecover": ecgrecover_panel_report,
            }
            external_release_reports[cohort] = release_report
        else:
            external_metrics_path = (bundle / "patient_metrics.parquet").resolve()
            cohorts = {
                str(value)
                for chunk in _iter_parquet_batches(
                    (external_metrics_path,), columns=("cohort",)
                )
                for value in chunk["cohort"].unique()
            }
            if len(cohorts) != 1:
                raise ValueError(f"external bundle must contain exactly one cohort: {bundle}")
            cohort = next(iter(cohorts))
        if cohort in external_effects:
            raise ValueError(f"duplicate external cohort {cohort}")
        common_external_sources = {
            method: external_metrics_path for method in COMMON_PANEL_METHODS
        }
        _validate_metric_source_partitions(
            {cohort: external_metrics_path}, expected={"test"}
        )
        point_path = output / f"{cohort}_predictions.parquet"
        seed_path = output / f"{cohort}_seed_predictions.parquet"
        sufficient_path = output / f"{cohort}_sufficient_stats.parquet"
        paired_sufficient_path = output / f"{cohort}_paired_seed_sufficient.parquet"
        cohort_sufficient, stream_report = _stream_loco_prediction_evidence(
            common_external_sources,
            rank_map,
            fold9_model_bank,
            cohort=cohort,
            point_path=point_path,
            seed_path=seed_path,
            sufficient_path=sufficient_path,
            paired_sufficient_path=paired_sufficient_path,
        )
        cohort_paired_sufficient = _read_small_parquet(paired_sufficient_path)
        if cohort not in META_BOOTSTRAP_SEED_OFFSETS:
            raise ValueError(f"no frozen bootstrap seed offset for external cohort {cohort}")
        effect, cohort_bootstrap_draws = _bootstrap_effect_and_draws_from_sufficient(
            cohort_sufficient,
            paired_sufficient=cohort_paired_sufficient,
            cohort=cohort,
            replicates=arguments.bootstrap_replicates,
            seed=arguments.seed + META_BOOTSTRAP_SEED_OFFSETS[cohort],
        )
        external_effects[cohort] = effect
        external_stream_reports[cohort] = stream_report
        external_bootstrap_draws[cohort] = cohort_bootstrap_draws
        external_hashes[cohort] = lineage.artifact_sha256(external_metrics_path)
        external_artifacts[cohort] = {
            "point_predictions": {
                "path": point_path.name,
                "sha256": lineage.artifact_sha256(point_path),
            },
            "seed_predictions": {
                "path": seed_path.name,
                "sha256": lineage.artifact_sha256(seed_path),
            },
            "sufficient_stats": {
                "path": sufficient_path.name,
                "sha256": lineage.artifact_sha256(sufficient_path),
            },
            "paired_seed_sufficient": {
                "path": paired_sufficient_path.name,
                "sha256": lineage.artifact_sha256(paired_sufficient_path),
            },
        }
        ecgrecover_supplement.extend(_ecgrecover_summary_stream(external_metrics_path))

    if not arguments.allow_partial_external and set(external_effects) != {"chapman", "cpsc2018"}:
        raise ValueError("release analysis requires both Chapman and CPSC2018")
    all_method_deltas = _method_deltas_from_sufficient(ptb_sufficient)
    method_deltas = {
        method: all_method_deltas[method]
        for method in COMMON_PANEL_METHODS if method in all_method_deltas
    }
    if not arguments.allow_partial_external and set(method_deltas) != set(COMMON_PANEL_METHODS):
        raise ValueError("common-panel evidence must contain all four primary reconstructors")
    decision = stage15_decision(
        ptbxl=ptb_effect,
        external=external_effects,
        method_deltas=method_deltas,
        gate_eligible_external_cohorts=STAGE15_GATE_ELIGIBLE_EXTERNAL_COHORTS,
    )
    external_gate_contract = {
        "eligible_cohorts": list(decision.gate_eligible_external_cohorts),
        "qualifying_cohorts": list(decision.qualifying_external_cohorts),
        "reported_but_gate_ineligible": {
            "cpsc2018": (
                "record-name pseudopatients; no public cross-record patient key"
            )
        },
        "rule": (
            "only external cohorts with a documented patient-level identity key may "
            "satisfy the Stage-15 external lower-confidence-bound criterion"
        ),
    }

    bootstrap_draws_path = output / "bootstrap_draws.parquet"
    bootstrap_draw_frames = [ptb_bootstrap_draws]
    for cohort in sorted(external_effects):
        bootstrap_draw_frames.append(external_bootstrap_draws[cohort])
    pd.concat(bootstrap_draw_frames, ignore_index=True).to_parquet(
        bootstrap_draws_path, index=False, compression="zstd"
    )
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
        "meta_model_execution": {
            "fold8_tuning": "subtractable configuration-block sufficient statistics",
            "fold9_model_bank_builds": 1,
            "held_configurations": len(fold9_model_bank.simple),
            "fitted_ridge_models": (
                len(fold9_model_bank.simple) + len(fold9_model_bank.augmented)
            ),
            "model_bank_reused_for": [
                "PTB-XL",
                *sorted(external_effects),
            ],
            "categorical_levels": {
                name: list(levels)
                for name, levels in zip(
                    CATEGORICAL_PREDICTORS,
                    fold9_model_bank.encoding.categorical_levels,
                    strict=True,
                )
            },
            "numeric_columns": list(fold9_model_bank.encoding.numeric_columns),
        },
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
            "for each neural method, draw five of its five fitted seeds with replacement "
            "and reconstruct the cellwise five-run mean within each shared patient bootstrap"
        ),
        "excluded_from_common_panel_meta_model": ["ecgrecover"],
        "supplementary_ecgrecover": ecgrecover_supplement,
        "automatic_stage15_status": decision.status,
        "automatic_stage15_reasons": list(decision.reasons),
        "external_stage15_gate": external_gate_contract,
        "bootstrap_unit": (
            "shared patient cluster with nested five-draw neural seed-mean resampling"
        ),
        "bootstrap_replicates": arguments.bootstrap_replicates,
        "bootstrap_draw_schema_version": META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
        "seed_prediction_schema_version": META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_stat_schema_version": META_SUFFICIENT_SCHEMA_VERSION,
        "paired_seed_sufficient_schema_version": (
            META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
        ),
        "exact_model_seed_contract": {
            method: list(_expected_common_seeds(method))
            for method in COMMON_PANEL_METHODS
        },
        "streaming_evidence": {
            "PTB-XL": ptb_stream_report,
            **external_stream_reports,
        },
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
            "ptbxl_sufficient_stats": {
                "path": ptb_sufficient_path.name,
                "sha256": lineage.artifact_sha256(ptb_sufficient_path),
            },
            "ptbxl_paired_seed_sufficient": {
                "path": ptb_paired_sufficient_path.name,
                "sha256": lineage.artifact_sha256(ptb_paired_sufficient_path),
            },
            "bootstrap_draws": {
                "path": bootstrap_draws_path.name,
                "sha256": lineage.artifact_sha256(bootstrap_draws_path),
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
    if summary.get("seed") != META_BOOTSTRAP_SEED:
        raise ValueError(f"Stage 15 requires frozen bootstrap seed {META_BOOTSTRAP_SEED}")
    if summary.get("bootstrap_draw_schema_version") != META_BOOTSTRAP_DRAW_SCHEMA_VERSION:
        raise ValueError("Stage 15 requires raw v3 meta-bootstrap draws")
    if (
        summary.get("seed_prediction_schema_version")
        != META_SEED_PREDICTION_SCHEMA_VERSION
        or summary.get("sufficient_stat_schema_version")
        != META_SUFFICIENT_SCHEMA_VERSION
        or summary.get("paired_seed_sufficient_schema_version")
        != META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Stage 15 requires streamed seed predictions and paired sufficient stats"
        )
    if tuple(summary.get("common_panel_methods", ())) != COMMON_PANEL_METHODS:
        raise ValueError("Stage 15 requires the frozen four-method common panel")
    if summary.get("exact_model_seed_contract") != {
        method: list(_expected_common_seeds(method)) for method in COMMON_PANEL_METHODS
    }:
        raise ValueError("Stage 15 requires the exact preregistered model-seed contract")
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
        ("ptbxl_sufficient_stats", "ptbxl_sufficient_stats.parquet"),
        (
            "ptbxl_paired_seed_sufficient",
            "ptbxl_paired_seed_sufficient.parquet",
        ),
        ("bootstrap_draws", "bootstrap_draws.parquet"),
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
            ("sufficient_stats", f"{cohort}_sufficient_stats.parquet"),
            (
                "paired_seed_sufficient",
                f"{cohort}_paired_seed_sufficient.parquet",
            ),
        ):
            descriptor = inventory.get(key)
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"Stage 15 lacks {cohort} {key}")
            authenticated_external[cohort][key] = _authenticated_artifact(
                arguments.meta_analysis.resolve(), descriptor, expected_name=filename
            )

    point_seed_binding_reports = {
        "PTB-XL": _validate_streamed_point_seed_binding(
            authenticated["ptbxl_predictions"],
            authenticated["ptbxl_seed_predictions"],
            cohort="PTB-XL",
        )
    }
    for cohort in sorted(authenticated_external):
        point_seed_binding_reports[cohort] = _validate_streamed_point_seed_binding(
            authenticated_external[cohort]["point_predictions"],
            authenticated_external[cohort]["seed_predictions"],
            cohort=cohort,
        )

    alpha_tuning = _read_small_parquet(authenticated["alpha_tuning"])
    alpha_columns = {"alpha", "mse_simple", "mse_augmented", "mean_mse"}
    if set(alpha_tuning.columns) != alpha_columns or len(alpha_tuning) != len(
        META_RIDGE_ALPHA_GRID
    ):
        raise ValueError("Stage 15 alpha-tuning artifact violates its frozen schema/grid")
    alpha_tuning = alpha_tuning.sort_values("alpha").reset_index(drop=True)
    alpha_numeric = alpha_tuning[list(alpha_columns)].to_numpy(dtype=float)
    if (
        not np.isfinite(alpha_numeric).all()
        or not np.array_equal(
            alpha_tuning["alpha"].to_numpy(dtype=float),
            np.asarray(META_RIDGE_ALPHA_GRID, dtype=float),
        )
        or not np.allclose(
            alpha_tuning["mean_mse"].to_numpy(dtype=float),
            0.5
            * (
                alpha_tuning["mse_simple"].to_numpy(dtype=float)
                + alpha_tuning["mse_augmented"].to_numpy(dtype=float)
            ),
            rtol=0.0,
            atol=EFFECT_COMPARISON_ATOL,
        )
    ):
        raise ValueError("Stage 15 alpha-tuning artifact is numerically invalid")
    minimum_mse = float(alpha_tuning["mean_mse"].min())
    expected_alpha = float(
        alpha_tuning.loc[
            np.isclose(
                alpha_tuning["mean_mse"], minimum_mse, rtol=1e-12, atol=0.0
            ),
            "alpha",
        ].min()
    )
    try:
        stored_alpha = float(summary.get("meta_alpha", np.nan))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage 15 meta alpha is not numeric") from exc
    if summary.get("meta_alpha_grid") != list(META_RIDGE_ALPHA_GRID) or not np.isclose(
        stored_alpha,
        expected_alpha,
        rtol=0.0,
        atol=EFFECT_COMPARISON_ATOL,
    ):
        raise ValueError("Stage 15 meta alpha is not the frozen fold-8 LOCO selection")

    def parse_effect(value: Mapping[str, Any], *, label: str) -> BootstrapEffect:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} effect must be an object")
        interval = value.get("ci95")
        if not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2:
            raise ValueError(f"{label} effect lacks a two-sided confidence interval")
        if any(
            isinstance(raw, (bool, np.bool_))
            for raw in (
                value.get("point"),
                *interval,
                value.get("replicates"),
                value.get("seed"),
            )
        ):
            raise ValueError(f"{label} effect contains boolean numeric fields")
        if not isinstance(value.get("replicates"), (int, np.integer)) or not isinstance(
            value.get("seed"), (int, np.integer)
        ):
            raise ValueError(f"{label} effect bootstrap metadata must be integer-valued")
        try:
            effect = BootstrapEffect(
                point=float(value["point"]),
                ci95=(float(interval[0]), float(interval[1])),
                replicates=int(value["replicates"]),
                seed=int(value["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} effect contains invalid numeric fields") from exc
        numbers = (effect.point, *effect.ci95)
        if effect.replicates != BOOTSTRAP_REPLICATES or not np.isfinite(numbers).all():
            raise ValueError(f"{label} effect is not a finite 2,000-bootstrap estimate")
        if effect.ci95[0] > effect.ci95[1]:
            raise ValueError(f"{label} confidence interval endpoints are reversed")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, *(abs(value) for value in numbers)
        )
        if (
            effect.point < effect.ci95[0] - tolerance
            or effect.point > effect.ci95[1] + tolerance
        ):
            raise ValueError(f"{label} point estimate lies outside its confidence interval")
        return effect

    stored_ptb_effect = parse_effect(summary["ptbxl"], label="PTB-XL")
    stored_external_effects = {
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

    def require_effect(actual: BootstrapEffect, expected: BootstrapEffect, label: str) -> None:
        require_close(actual.point, expected.point, label=f"{label} point")
        require_close(actual.ci95[0], expected.ci95[0], label=f"{label} CI lower")
        require_close(actual.ci95[1], expected.ci95[1], label=f"{label} CI upper")
        if actual.replicates != expected.replicates or actual.seed != expected.seed:
            raise ValueError(f"{label} bootstrap metadata disagrees with authenticated evidence")

    # Recreate every Stage-15 bootstrap by streaming the authenticated underlying
    # seed predictions.  The compact sufficient table is authenticated and then
    # compared with this independent reconstruction; it is never trusted alone.
    (
        ptb_sufficient,
        ptb_paired_sufficient,
        ptb_stream_report,
    ) = _rebuild_statistics_from_seed_evidence(
        authenticated["ptbxl_seed_predictions"], cohort="PTB-XL"
    )
    stored_ptb_sufficient = _read_small_parquet(
        authenticated["ptbxl_sufficient_stats"]
    )
    _require_sufficient_equal(
        stored_ptb_sufficient, ptb_sufficient, label="PTB-XL"
    )
    stored_ptb_paired_sufficient = _read_small_parquet(
        authenticated["ptbxl_paired_seed_sufficient"]
    )
    _require_paired_sufficient_equal(
        stored_ptb_paired_sufficient,
        ptb_paired_sufficient,
        label="PTB-XL",
    )
    ptb_effect, ptb_draws = _bootstrap_effect_and_draws_from_sufficient(
        ptb_sufficient,
        paired_sufficient=ptb_paired_sufficient,
        cohort="PTB-XL",
        replicates=BOOTSTRAP_REPLICATES,
        seed=META_BOOTSTRAP_SEED + META_BOOTSTRAP_SEED_OFFSETS["PTB-XL"],
    )
    require_effect(stored_ptb_effect, ptb_effect, "PTB-XL summary")

    external_effects: dict[str, BootstrapEffect] = {}
    external_stream_reports: dict[str, dict[str, Any]] = {}
    rebuilt_draws = [ptb_draws]
    for cohort in sorted(stored_external_effects):
        (
            cohort_sufficient,
            cohort_paired_sufficient,
            stream_report,
        ) = _rebuild_statistics_from_seed_evidence(
            authenticated_external[cohort]["seed_predictions"], cohort=cohort
        )
        stored_sufficient = _read_small_parquet(
            authenticated_external[cohort]["sufficient_stats"]
        )
        _require_sufficient_equal(
            stored_sufficient, cohort_sufficient, label=cohort
        )
        stored_paired_sufficient = _read_small_parquet(
            authenticated_external[cohort]["paired_seed_sufficient"]
        )
        _require_paired_sufficient_equal(
            stored_paired_sufficient,
            cohort_paired_sufficient,
            label=cohort,
        )
        effect, cohort_draws = _bootstrap_effect_and_draws_from_sufficient(
            cohort_sufficient,
            paired_sufficient=cohort_paired_sufficient,
            cohort=cohort,
            replicates=BOOTSTRAP_REPLICATES,
            seed=META_BOOTSTRAP_SEED + META_BOOTSTRAP_SEED_OFFSETS[cohort],
        )
        require_effect(stored_external_effects[cohort], effect, f"{cohort} summary")
        external_effects[cohort] = effect
        external_stream_reports[cohort] = stream_report
        rebuilt_draws.append(cohort_draws)

    expected_draws = pd.concat(rebuilt_draws, ignore_index=True)
    stored_draws = _read_small_parquet(authenticated["bootstrap_draws"])
    expected_draw_columns = list(expected_draws.columns)
    if set(stored_draws.columns) != set(expected_draw_columns):
        raise ValueError("meta-bootstrap draw artifact has an unexpected schema")
    stored_draws = stored_draws[expected_draw_columns].sort_values(
        ["cohort", "bootstrap_index"]
    ).reset_index(drop=True)
    expected_draws = expected_draws.sort_values(
        ["cohort", "bootstrap_index"]
    ).reset_index(drop=True)
    if len(stored_draws) != len(expected_draws):
        raise ValueError("meta-bootstrap draw artifact has the wrong row count")
    exact_columns = [column for column in expected_draw_columns if column != "delta_r2"]
    for column in exact_columns:
        if not stored_draws[column].astype(str).equals(expected_draws[column].astype(str)):
            raise ValueError(
                f"meta-bootstrap draw {column} disagrees with authenticated seed predictions"
            )
    if not np.allclose(
        stored_draws["delta_r2"].to_numpy(dtype=float),
        expected_draws["delta_r2"].to_numpy(dtype=float),
        rtol=0.0,
        atol=EFFECT_COMPARISON_ATOL,
    ):
        raise ValueError(
            "meta-bootstrap draw delta_r2 disagrees with authenticated seed predictions"
        )

    # The compact table must be an exact synopsis of the independent recomputation.
    effects_frame = _read_small_parquet(authenticated["effects"])
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
        require_effect(artifact_effect, expected, f"{cohort} effects artifact")

    require_close(
        _delta_from_sufficient_rows(
            ptb_sufficient[
                ptb_sufficient["estimand"].astype(str) == "point_seed_mean"
            ]
        ),
        ptb_effect.point,
        label="PTB-XL point estimate",
    )
    artifact_method_deltas = _method_deltas_from_sufficient(ptb_sufficient)
    if set(artifact_method_deltas) != set(COMMON_PANEL_METHODS):
        raise ValueError("PTB-XL prediction artifact lacks the frozen four-method panel")
    for method in COMMON_PANEL_METHODS:
        require_close(
            artifact_method_deltas[method],
            method_deltas[method],
            label=f"{method} point estimate",
        )
    recomputed = stage15_decision(
        ptbxl=ptb_effect,
        external=external_effects,
        method_deltas=method_deltas,
        gate_eligible_external_cohorts=STAGE15_GATE_ELIGIBLE_EXTERNAL_COHORTS,
    )
    expected_external_gate = {
        "eligible_cohorts": list(recomputed.gate_eligible_external_cohorts),
        "qualifying_cohorts": list(recomputed.qualifying_external_cohorts),
        "reported_but_gate_ineligible": {
            "cpsc2018": (
                "record-name pseudopatients; no public cross-record patient key"
            )
        },
        "rule": (
            "only external cohorts with a documented patient-level identity key may "
            "satisfy the Stage-15 external lower-confidence-bound criterion"
        ),
    }
    if summary.get("external_stage15_gate") != expected_external_gate:
        raise ValueError("stored Stage-15 external gate eligibility is invalid")
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
        "external_stage15_gate": expected_external_gate,
        "bootstrap_replicates": summary["bootstrap_replicates"],
        "streaming_seed_reconstruction": {
            "PTB-XL": ptb_stream_report,
            **external_stream_reports,
        },
        "streaming_point_seed_binding": point_seed_binding_reports,
        "bootstrap_draws_sha256": lineage.artifact_sha256(
            authenticated["bootstrap_draws"]
        ),
        "release_lineage_sha256": lineage.canonical_sha256(
            summary.get("release_lineage")
        ),
    }
    arc_report = _load_json(arguments.arc_control)
    arc_control = validate_arc_waiting_report(arc_report, 15)
    evidence["official_arc_waiting"] = waiting_control_evidence(
        arc_control,
        report_sha256=lineage.artifact_sha256(arguments.arc_control),
    )
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
            "chapman_ci_lower_gt_zero": True,
            "positive_common_panel_methods_at_least": 3,
        },
        "policy": {
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
    parser.add_argument("--ptbxl-manifest", type=Path)
    parser.add_argument("--external-manifest", type=Path, action="append", default=[])
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
            if arguments.ptbxl_manifest is None:
                raise SystemExit("release analysis requires --ptbxl-manifest")
            if len(arguments.external_manifest) != 2:
                raise SystemExit(
                    "release analysis requires exactly two --external-manifest inputs"
                )
            if arguments.bootstrap_replicates != BOOTSTRAP_REPLICATES:
                raise SystemExit(
                    f"release analysis requires exactly {BOOTSTRAP_REPLICATES} bootstraps"
                )
            if arguments.seed != META_BOOTSTRAP_SEED:
                raise SystemExit(
                    f"release analysis requires frozen bootstrap seed {META_BOOTSTRAP_SEED}"
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
