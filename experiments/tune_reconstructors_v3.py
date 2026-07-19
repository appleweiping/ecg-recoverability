"""Select frozen reconstruction settings from real PTB-XL fold-8 candidate metrics.

This entry point does not invent tuning results and does not accept hand-written
winners.  It consumes the candidate bundle produced by folds-1--7 fits evaluated
only on fold 8, verifies the complete preregistered grid, checkpoints, and neural
early-stop trace, then applies a deterministic patient-balanced selection rule.
Official methods are recorded from a pinned fixed configuration and never
selected by outcome inspection.

The release CLI scans raw candidate metrics one Parquet row group at a time and
retains only exact patient sufficient statistics, so the roughly hundred-million
cell formal panel never becomes one in-memory pandas table.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage

try:  # package import in tests; direct sibling import under ``python experiments/...``
    from .reconstruction_benchmark_v3 import load_ptbxl_manifest
except ImportError:  # pragma: no cover - exercised by the CLI smoke check
    from reconstruction_benchmark_v3 import load_ptbxl_manifest


SCHEMA_VERSION = "reconstruction-tuning-v3"
CANDIDATE_SCHEMA_VERSION = "reconstruction-tuning-candidate-v3"
TRACE_SCHEMA_VERSION = "reconstruction-tuning-early-stop-trace-v3"
CANDIDATE_BUNDLE_SCHEMA_VERSION = "reconstruction-tuning-candidate-bundle-v3"
TUNING_SEEDS = (0, 1, 2)
UNET_PATIENCE = 8


@dataclass(frozen=True)
class Candidate:
    method: str
    candidate_id: str
    parameters: Mapping[str, Any]


def _candidate(method: str, parameters: Mapping[str, Any]) -> Candidate:
    digest = lineage.canonical_sha256({"method": method, "parameters": parameters})[:12]
    return Candidate(method, f"{method}-{digest}", dict(parameters))


def candidate_grid() -> dict[str, tuple[Candidate, ...]]:
    lowrank = tuple(
        _candidate("lowrank", {"rank": rank, "noise_variance": variance})
        for rank in (2, 3, 4, 5)
        for variance in (1e-8, 1e-6, 1e-4)
    )
    ridge = tuple(
        _candidate("ridge", {"ridge_lambda": ridge_lambda})
        for ridge_lambda in (1e-6, 1e-4, 1e-2, 1.0, 100.0)
    )
    masked_unet = tuple(
        _candidate(
            "masked-unet",
            {
                "width": width,
                "learning_rate": learning_rate,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "max_epochs": 60,
                "early_stopping_patience": UNET_PATIENCE,
                "num_workers": 0,
                "deterministic": True,
            },
        )
        for width, learning_rate in ((32, 1e-3), (48, 1e-3), (48, 3e-4))
    )
    return {"lowrank": lowrank, "ridge": ridge, "masked-unet": masked_unet}


REQUIRED_COLUMNS = {
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
}
CELL_COLUMNS = ("patient_id", "segment", "configuration", "target")
TRACE_REQUIRED_COLUMNS = {
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
}


def _cell_set(rows: pd.DataFrame) -> set[tuple[str, ...]]:
    return set(
        rows.loc[:, list(CELL_COLUMNS)].astype(str).itertuples(index=False, name=None)
    )


def _validate_candidate_metrics(
    frame: pd.DataFrame,
    *,
    manifest_sha256: str,
    split_sha256: str,
) -> dict[str, tuple[Candidate, ...]]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"candidate metrics are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("candidate metrics are empty")
    exact_values = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "cohort": "PTB-XL",
        "train_partition": "folds1-7/train",
        "partition": "fold8/tune",
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }
    for column, expected in exact_values.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"candidate {column} must be exactly {expected!r}")
    if set(frame["observed_integrity"].unique()) != {True}:
        raise ValueError("every tuning prediction must pass exact observed-sample integrity")
    numeric = frame[["model_seed", "epoch", "rmse_mv", "log_rmse_mv"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all() or (frame["rmse_mv"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("candidate metrics contain invalid numeric values")
    if not np.allclose(
        frame["log_rmse_mv"].to_numpy(dtype=float),
        np.log(frame["rmse_mv"].to_numpy(dtype=float)),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("candidate log_rmse_mv is not the log of RMSE_mV")
    if not frame["checkpoint_sha256"].map(
        lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))
    ).all():
        raise ValueError("candidate checkpoint_sha256 must be a full lowercase SHA-256")
    checkpoint_groups = frame.groupby(["method", "candidate_id", "model_seed"], sort=False)
    if (
        checkpoint_groups["checkpoint_path"].nunique().max() != 1
        or checkpoint_groups["checkpoint_sha256"].nunique().max() != 1
    ):
        raise ValueError("each candidate/seed must reference exactly one checkpoint")
    if frame.duplicated(["method", "candidate_id", "model_seed", "epoch", *CELL_COLUMNS]).any():
        raise ValueError("candidate metrics contain duplicate evaluation cells")

    grid = candidate_grid()
    if set(frame["method"]) != set(grid):
        raise ValueError("candidate metrics must contain only lowrank, ridge, and masked-unet")
    reference_cells: set[tuple[str, ...]] | None = None
    for method, candidates in grid.items():
        method_rows = frame[frame["method"] == method]
        expected_ids = {candidate.candidate_id for candidate in candidates}
        if set(method_rows["candidate_id"]) != expected_ids:
            raise ValueError(f"{method} candidate IDs do not match the preregistered grid")
        expected_seeds = set(TUNING_SEEDS) if method == "masked-unet" else {0}
        for candidate in candidates:
            candidate_rows = method_rows[method_rows["candidate_id"] == candidate.candidate_id]
            if set(candidate_rows["model_seed"].astype(int)) != expected_seeds:
                raise ValueError(f"{candidate.candidate_id} has the wrong tuning seeds")
            if method == "masked-unet":
                epochs_by_seed = []
                max_epochs = int(candidate.parameters["max_epochs"])
                for seed in TUNING_SEEDS:
                    seed_rows = candidate_rows[candidate_rows["model_seed"] == seed]
                    epochs = tuple(sorted(seed_rows["epoch"].astype(int).unique()))
                    if not epochs or any(epoch < 1 for epoch in epochs):
                        raise ValueError(
                            f"{candidate.candidate_id}/seed-{seed} epochs must be positive"
                        )
                    # Candidate producers may either emit the complete per-epoch
                    # trajectory (the legacy/research-debug representation) or
                    # only the independently early-stopped best checkpoint for
                    # each seed (the release representation).  A singleton is
                    # important in release: duplicating every patient/config/
                    # target row for all 60 epochs would create a multi-billion
                    # row artifact without adding selection information.
                    if len(epochs) > 1 and epochs != tuple(range(1, epochs[-1] + 1)):
                        raise ValueError(
                            f"{candidate.candidate_id}/seed-{seed} epoch trajectories "
                            "must start at 1 and be consecutive"
                        )
                    if epochs[-1] > max_epochs:
                        raise ValueError(f"{candidate.candidate_id} exceeds max_epochs")
                    epochs_by_seed.append(epochs)
                singleton_release = all(len(epochs) == 1 for epochs in epochs_by_seed)
                trajectory_debug = all(len(epochs) > 1 for epochs in epochs_by_seed)
                if not (singleton_release or trajectory_debug):
                    raise ValueError(
                        f"{candidate.candidate_id} mixes early-stopped and trajectory rows"
                    )
                if trajectory_debug and len(set(epochs_by_seed)) != 1:
                    raise ValueError(
                        f"{candidate.candidate_id} trajectory seeds have different epoch coverage"
                    )
                # Compare cells seed by seed.  In release, independently early-
                # stopped seeds can legitimately have different best epochs.
                for seed, epochs in zip(TUNING_SEEDS, epochs_by_seed):
                    seed_rows = candidate_rows[candidate_rows["model_seed"] == seed]
                    for epoch in epochs:
                        epoch_cells = _cell_set(seed_rows[seed_rows["epoch"] == epoch])
                        if reference_cells is not None and epoch_cells != reference_cells:
                            raise ValueError(
                                "candidate methods do not share identical fold-8 cells"
                            )
                        reference_cells = (
                            epoch_cells if reference_cells is None else reference_cells
                        )
            else:
                if set(candidate_rows["epoch"].astype(int)) != {0}:
                    raise ValueError(f"{candidate.candidate_id} linear epoch must be zero")
                cells = _cell_set(candidate_rows)
                if reference_cells is not None and cells != reference_cells:
                    raise ValueError("candidate methods do not share identical fold-8 cells")
                reference_cells = cells if reference_cells is None else reference_cells
    return grid


def _verify_candidate_checkpoints(frame: pd.DataFrame, candidate_metrics: Path) -> None:
    root = candidate_metrics.resolve().parent
    descriptors = frame[["checkpoint_path", "checkpoint_sha256"]].drop_duplicates()
    for row in descriptors.itertuples(index=False):
        relative = Path(str(row.checkpoint_path))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe tuning checkpoint path: {relative}")
        checkpoint = (root / relative).resolve()
        if root not in checkpoint.parents:
            raise ValueError(f"tuning checkpoint escapes candidate artifact root: {relative}")
        if lineage.artifact_sha256(checkpoint) != str(row.checkpoint_sha256):
            raise ValueError(f"tuning checkpoint SHA-256 mismatch: {relative}")


@dataclass(frozen=True)
class CandidateMetricScan:
    """Bounded-memory validation result for a candidate-metric Parquet file.

    ``patient_rows`` contains one sufficient row per candidate/seed/epoch/patient.
    It is selection-equivalent to the raw cell table because every candidate unit
    is required to contain the same cells and its value is the exact cell-weighted
    patient mean.  The raw Parquet remains the auditable source artifact.
    """

    patient_rows: pd.DataFrame
    grid: dict[str, tuple[Candidate, ...]]
    n_rows: int
    n_row_groups: int
    configurations: tuple[str, ...]
    cell_reference_sha256: str


def _validate_candidate_chunk(
    frame: pd.DataFrame,
    *,
    manifest_sha256: str,
    split_sha256: str,
) -> None:
    """Validate row-local contracts without retaining prior raw rows."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"candidate metrics are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("candidate metric row group is empty")
    exact_values = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "cohort": "PTB-XL",
        "train_partition": "folds1-7/train",
        "partition": "fold8/tune",
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }
    for column, expected in exact_values.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"candidate {column} must be exactly {expected!r}")
    if set(frame["observed_integrity"].unique()) != {True}:
        raise ValueError("every tuning prediction must pass exact observed-sample integrity")
    numeric = frame[["model_seed", "epoch", "rmse_mv", "log_rmse_mv"]].to_numpy(
        dtype=float
    )
    rmse = frame["rmse_mv"].to_numpy(dtype=float)
    log_rmse = frame["log_rmse_mv"].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (rmse <= 0).any():
        raise ValueError("candidate metrics contain invalid numeric values")
    if not np.allclose(log_rmse, np.log(rmse), rtol=1e-10, atol=1e-12):
        raise ValueError("candidate log_rmse_mv is not the log of RMSE_mV")
    if not frame["checkpoint_sha256"].map(
        lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))
    ).all():
        raise ValueError("candidate checkpoint_sha256 must be a full lowercase SHA-256")


def scan_candidate_metrics_parquet(
    candidate_metrics: Path,
    *,
    manifest_sha256: str,
    split_sha256: str,
    expected_n_configurations: int | None = None,
) -> CandidateMetricScan:
    """Validate and reduce the formal fold-8 table one Parquet row group at a time.

    The producer writes one candidate/seed/configuration unit per row group.  A
    fixture may place several complete units in one row group, but a unit may
    never be split across row groups.  This constraint makes duplicate and cell
    equality checks fail closed without an in-memory set containing roughly one
    hundred million raw rows.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - locked environments provide pyarrow
        raise RuntimeError("tuning requires locked pyarrow; CSV fallback is forbidden") from exc
    path = candidate_metrics.resolve()
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < 1 or parquet.metadata.num_row_groups < 1:
        raise ValueError("candidate metrics are empty")
    missing = REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"candidate metrics are missing columns: {sorted(missing)}")

    unit_columns = ("method", "candidate_id", "model_seed", "epoch", "configuration")
    seen_units: set[tuple[str, str, int, int, str]] = set()
    unit_configurations: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    epochs_by_seed: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    seeds_by_candidate: dict[tuple[str, str], set[int]] = defaultdict(set)
    checkpoint_by_seed: dict[tuple[str, str, int], tuple[str, str]] = {}
    reference_cells: dict[str, tuple[str, int]] = {}
    # (method, candidate, seed, epoch, patient) -> [sum(log_rmse), count]
    patient_sufficient: dict[tuple[str, str, int, int, str], list[float | int]] = {}
    total_rows = 0

    for row_group_index in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(row_group_index).to_pandas()
        _validate_candidate_chunk(
            frame,
            manifest_sha256=manifest_sha256,
            split_sha256=split_sha256,
        )
        total_rows += len(frame)
        grouped = frame.groupby(list(unit_columns), sort=False, dropna=False)
        for raw_unit, rows in grouped:
            method, candidate_id, raw_seed, raw_epoch, configuration = raw_unit
            unit = (
                str(method),
                str(candidate_id),
                int(raw_seed),
                int(raw_epoch),
                str(configuration),
            )
            if unit in seen_units:
                raise ValueError(
                    "candidate metric unit is split across Parquet row groups or duplicated: "
                    f"{unit}"
                )
            seen_units.add(unit)
            if rows.duplicated(list(CELL_COLUMNS)).any():
                raise ValueError("candidate metrics contain duplicate evaluation cells")
            checkpoint_values = rows[["checkpoint_path", "checkpoint_sha256"]].drop_duplicates()
            if len(checkpoint_values) != 1:
                raise ValueError("each candidate/seed must reference exactly one checkpoint")
            checkpoint = tuple(
                str(value) for value in checkpoint_values.iloc[0].to_list()
            )
            seed_key = unit[:3]
            if seed_key in checkpoint_by_seed and checkpoint_by_seed[seed_key] != checkpoint:
                raise ValueError("each candidate/seed must reference exactly one checkpoint")
            checkpoint_by_seed[seed_key] = checkpoint

            cell_frame = rows.loc[:, list(CELL_COLUMNS)].astype(str)
            cell_table = pa.Table.from_pandas(cell_frame, preserve_index=False)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, cell_table.schema) as writer:
                writer.write_table(cell_table)
            digest = hashlib.sha256(sink.getvalue()).hexdigest()
            n_cells = len(rows)
            reference = reference_cells.get(unit[4])
            if reference is not None and reference != (digest, n_cells):
                raise ValueError("candidate methods do not share identical fold-8 cells")
            reference_cells.setdefault(unit[4], (digest, n_cells))

            candidate_epoch = unit[:4]
            unit_configurations[candidate_epoch].add(unit[4])
            epochs_by_seed[seed_key].add(unit[3])
            seeds_by_candidate[unit[:2]].add(unit[2])
            patient_groups = rows.groupby("patient_id", sort=False)["log_rmse_mv"].agg(
                ["sum", "count"]
            )
            for patient_id, values in patient_groups.iterrows():
                patient_key = (*candidate_epoch, str(patient_id))
                current = patient_sufficient.setdefault(patient_key, [0.0, 0])
                current[0] = float(current[0]) + float(values["sum"])
                current[1] = int(current[1]) + int(values["count"])

    grid = candidate_grid()
    if {unit[0] for unit in seen_units} != set(grid):
        raise ValueError("candidate metrics must contain only lowrank, ridge, and masked-unet")
    configurations = tuple(sorted(reference_cells))
    if expected_n_configurations is not None and len(configurations) != int(
        expected_n_configurations
    ):
        raise ValueError(
            "candidate metric configuration count does not match the frozen panel: "
            f"{len(configurations)} != {expected_n_configurations}"
        )
    expected_configurations = set(configurations)
    for candidate_epoch, values in unit_configurations.items():
        if values != expected_configurations:
            raise ValueError("candidate methods do not share identical fold-8 cells")

    for method, candidates in grid.items():
        expected_ids = {candidate.candidate_id for candidate in candidates}
        actual_ids = {
            candidate_id
            for candidate_method, candidate_id in seeds_by_candidate
            if candidate_method == method
        }
        if actual_ids != expected_ids:
            raise ValueError(f"{method} candidate IDs do not match the preregistered grid")
        expected_seeds = set(TUNING_SEEDS) if method == "masked-unet" else {0}
        for candidate in candidates:
            candidate_key = (method, candidate.candidate_id)
            if seeds_by_candidate[candidate_key] != expected_seeds:
                raise ValueError(f"{candidate.candidate_id} has the wrong tuning seeds")
            candidate_epoch_sets = []
            for seed in sorted(expected_seeds):
                seed_key = (*candidate_key, seed)
                epochs = tuple(sorted(epochs_by_seed[seed_key]))
                if method != "masked-unet":
                    if epochs != (0,):
                        raise ValueError(f"{candidate.candidate_id} linear epoch must be zero")
                    continue
                max_epochs = int(candidate.parameters["max_epochs"])
                if not epochs or any(epoch < 1 for epoch in epochs):
                    raise ValueError(
                        f"{candidate.candidate_id}/seed-{seed} epochs must be positive"
                    )
                if len(epochs) > 1 and epochs != tuple(range(1, epochs[-1] + 1)):
                    raise ValueError(
                        f"{candidate.candidate_id}/seed-{seed} epoch trajectories "
                        "must start at 1 and be consecutive"
                    )
                if epochs[-1] > max_epochs:
                    raise ValueError(f"{candidate.candidate_id} exceeds max_epochs")
                candidate_epoch_sets.append(epochs)
            if method == "masked-unet":
                singleton_release = all(len(epochs) == 1 for epochs in candidate_epoch_sets)
                trajectory_debug = all(len(epochs) > 1 for epochs in candidate_epoch_sets)
                if not (singleton_release or trajectory_debug):
                    raise ValueError(
                        f"{candidate.candidate_id} mixes early-stopped and trajectory rows"
                    )
                if trajectory_debug and len(set(candidate_epoch_sets)) != 1:
                    raise ValueError(
                        f"{candidate.candidate_id} trajectory seeds have different epoch coverage"
                    )

    patient_rows = []
    for key, (log_sum, count) in sorted(patient_sufficient.items()):
        method, candidate_id, seed, epoch, patient_id = key
        if int(count) < 1:
            raise ValueError("candidate patient sufficient statistic has zero cells")
        checkpoint_path, checkpoint_sha256 = checkpoint_by_seed[
            (method, candidate_id, seed)
        ]
        patient_rows.append(
            {
                "method": method,
                "candidate_id": candidate_id,
                "model_seed": seed,
                "epoch": epoch,
                "patient_id": patient_id,
                "log_rmse_mv": float(log_sum) / int(count),
                "n_cells": int(count),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    compact = pd.DataFrame(patient_rows)
    if compact.empty or not np.isfinite(compact["log_rmse_mv"].to_numpy(dtype=float)).all():
        raise ValueError("candidate metrics contain no finite patient sufficient statistics")
    cell_reference_sha256 = lineage.canonical_sha256(
        [
            {
                "configuration": configuration,
                "cell_sha256": reference_cells[configuration][0],
                "n_cells": reference_cells[configuration][1],
            }
            for configuration in configurations
        ]
    )
    return CandidateMetricScan(
        patient_rows=compact,
        grid=grid,
        n_rows=total_rows,
        n_row_groups=parquet.metadata.num_row_groups,
        configurations=configurations,
        cell_reference_sha256=cell_reference_sha256,
    )


def _validate_early_stop_trace(
    trace: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    manifest_sha256: str,
    split_sha256: str,
    fold8_records_sha256: str | None = None,
    configuration_panel_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify release U-Net early stopping independently of the producer."""

    missing = TRACE_REQUIRED_COLUMNS - set(trace.columns)
    if missing:
        raise ValueError(f"early-stop trace is missing columns: {sorted(missing)}")
    if trace.empty:
        raise ValueError("early-stop trace is empty")
    exact_values = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "cohort": "PTB-XL",
        "train_partition": "folds1-7/train",
        "partition": "fold8/tune",
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }
    if fold8_records_sha256 is not None:
        exact_values["fold8_records_sha256"] = fold8_records_sha256
    if configuration_panel_sha256 is not None:
        exact_values["configuration_panel_sha256"] = configuration_panel_sha256
    for column, expected in exact_values.items():
        if set(trace[column].astype(str)) != {expected}:
            raise ValueError(f"early-stop trace {column} must be exactly {expected!r}")
    for column in ("fold8_records_sha256", "configuration_panel_sha256"):
        values = set(trace[column].astype(str))
        if len(values) != 1 or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            raise ValueError(f"early-stop trace {column} must be one full SHA-256")
    numeric = trace[
        [
            "model_seed",
            "epoch",
            "monitor_log_rmse_mv",
            "best_so_far_log_rmse_mv",
            "stale_epochs",
            "best_epoch",
            "stopped_epoch",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("early-stop trace contains non-finite numeric values")
    if trace.duplicated(["candidate_id", "model_seed", "epoch"]).any():
        raise ValueError("early-stop trace contains duplicate epochs")
    grid = candidate_grid()["masked-unet"]
    if set(trace["candidate_id"].astype(str)) != {
        candidate.candidate_id for candidate in grid
    }:
        raise ValueError("early-stop trace does not contain the complete U-Net grid")

    metric_rows = metrics[metrics["method"] == "masked-unet"]
    audit: dict[str, Any] = {}
    for candidate in grid:
        candidate_trace = trace[trace["candidate_id"] == candidate.candidate_id]
        if set(candidate_trace["model_seed"].astype(int)) != set(TUNING_SEEDS):
            raise ValueError(f"{candidate.candidate_id} trace has the wrong seeds")
        seed_audit = {}
        for seed in TUNING_SEEDS:
            rows = candidate_trace[candidate_trace["model_seed"] == seed].sort_values("epoch")
            epochs = rows["epoch"].to_numpy(dtype=int)
            stopped_values = rows["stopped_epoch"].astype(int).unique()
            best_values = rows["best_epoch"].astype(int).unique()
            if len(stopped_values) != 1 or len(best_values) != 1:
                raise ValueError(f"{candidate.candidate_id}/seed-{seed} trace summary varies")
            stopped_epoch = int(stopped_values[0])
            best_epoch = int(best_values[0])
            if not np.array_equal(epochs, np.arange(1, stopped_epoch + 1)):
                raise ValueError(
                    f"{candidate.candidate_id}/seed-{seed} trace epochs are not 1..stopped"
                )
            if stopped_epoch > int(candidate.parameters["max_epochs"]):
                raise ValueError(f"{candidate.candidate_id}/seed-{seed} exceeds max_epochs")
            scores = rows["monitor_log_rmse_mv"].to_numpy(dtype=float)
            expected_best = np.minimum.accumulate(scores)
            if not np.allclose(
                rows["best_so_far_log_rmse_mv"].to_numpy(dtype=float),
                expected_best,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(f"{candidate.candidate_id}/seed-{seed} best-so-far is invalid")
            expected_best_epoch = int(np.argmin(scores)) + 1
            if best_epoch != expected_best_epoch:
                raise ValueError(f"{candidate.candidate_id}/seed-{seed} best_epoch is invalid")
            stale = 0
            running_best = float("inf")
            expected_stale = []
            expected_improvement = []
            first_stop = None
            for epoch, score in zip(epochs, scores):
                improved = bool(score < running_best)
                if improved:
                    running_best = float(score)
                    stale = 0
                else:
                    stale += 1
                expected_stale.append(stale)
                expected_improvement.append(improved)
                if stale >= UNET_PATIENCE and first_stop is None:
                    first_stop = int(epoch)
            if not np.array_equal(
                rows["stale_epochs"].to_numpy(dtype=int), np.asarray(expected_stale)
            ):
                raise ValueError(f"{candidate.candidate_id}/seed-{seed} stale counts are invalid")
            if not np.array_equal(
                rows["is_strict_improvement"].to_numpy(dtype=bool),
                np.asarray(expected_improvement, dtype=bool),
            ):
                raise ValueError(
                    f"{candidate.candidate_id}/seed-{seed} strict-improvement flags are invalid"
                )
            expected_stop = (
                first_stop
                if first_stop is not None
                else int(candidate.parameters["max_epochs"])
            )
            if stopped_epoch != expected_stop:
                raise ValueError(
                    f"{candidate.candidate_id}/seed-{seed} did not stop at the first "
                    f"patience-{UNET_PATIENCE} epoch"
                )
            metric_seed = metric_rows[
                (metric_rows["candidate_id"] == candidate.candidate_id)
                & (metric_rows["model_seed"].astype(int) == seed)
            ]
            if set(metric_seed["epoch"].astype(int)) != {best_epoch}:
                raise ValueError(
                    f"{candidate.candidate_id}/seed-{seed} metrics are not from best_epoch"
                )
            trace_checkpoint = rows[["checkpoint_path", "checkpoint_sha256"]].drop_duplicates()
            metric_checkpoint = metric_seed[
                ["checkpoint_path", "checkpoint_sha256"]
            ].drop_duplicates()
            if len(trace_checkpoint) != 1 or len(metric_checkpoint) != 1 or not np.array_equal(
                trace_checkpoint.to_numpy(dtype=str), metric_checkpoint.to_numpy(dtype=str)
            ):
                raise ValueError(
                    f"{candidate.candidate_id}/seed-{seed} checkpoint/trace mismatch"
                )
            seed_audit[str(seed)] = {
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_monitor_log_rmse_mv": float(scores[best_epoch - 1]),
            }
        audit[candidate.candidate_id] = seed_audit
    return {
        "fold8_records_sha256": str(trace["fold8_records_sha256"].iloc[0]),
        "configuration_panel_sha256": str(
            trace["configuration_panel_sha256"].iloc[0]
        ),
        "candidates": audit,
    }


def _patient_balanced_score(rows: pd.DataFrame) -> float:
    patient_scores = rows.groupby("patient_id", sort=True)["log_rmse_mv"].mean()
    if patient_scores.empty or not np.isfinite(patient_scores.to_numpy(dtype=float)).all():
        raise ValueError("candidate has no finite patient-balanced score")
    return float(patient_scores.mean())


def _select_linear(rows: pd.DataFrame, candidates: Sequence[Candidate]) -> tuple[Candidate, dict]:
    scores = {
        candidate.candidate_id: _patient_balanced_score(
            rows[rows["candidate_id"] == candidate.candidate_id]
        )
        for candidate in candidates
    }
    selected = min(candidates, key=lambda item: (scores[item.candidate_id], item.candidate_id))
    return selected, {"score": scores[selected.candidate_id], "candidate_scores": scores}


def _select_unet(rows: pd.DataFrame, candidates: Sequence[Candidate]) -> tuple[Candidate, dict]:
    candidate_audit = {}
    for candidate in candidates:
        subset = rows[rows["candidate_id"] == candidate.candidate_id]
        epochs_by_seed = {
            int(seed): tuple(sorted(seed_rows["epoch"].astype(int).unique()))
            for seed, seed_rows in subset.groupby("model_seed", sort=True)
        }
        if epochs_by_seed and all(len(epochs) == 1 for epochs in epochs_by_seed.values()):
            seed_best_epochs = {
                seed: epochs[0] for seed, epochs in epochs_by_seed.items()
            }
            # The producer has already applied patience-based early stopping to
            # each independently seeded fit and evaluated the restored best
            # checkpoint on the full frozen fold-8 panel.  Outer selection is
            # therefore performed directly on those real patient-level rows.
            score = _patient_balanced_score(subset)
            best_epoch = int(np.median(list(seed_best_epochs.values())))
            candidate_audit[candidate.candidate_id] = {
                "score": score,
                "best_epoch": best_epoch,
                "stopped_epoch": None,
                "seed_best_epochs": seed_best_epochs,
                "early_stopping_evidence": "producer_restored_best_checkpoint",
            }
            continue
        epoch_scores = {
            int(epoch): _patient_balanced_score(epoch_rows)
            for epoch, epoch_rows in subset.groupby("epoch", sort=True)
        }
        patience = int(candidate.parameters["early_stopping_patience"])
        best_epoch = None
        best_score = float("inf")
        stale = 0
        stopped_epoch = max(epoch_scores)
        for epoch, score in sorted(epoch_scores.items()):
            if score < best_score:
                best_epoch, best_score, stale = epoch, score, 0
            else:
                stale += 1
            if stale >= patience:
                stopped_epoch = epoch
                break
        if best_epoch is None:
            raise ValueError(f"{candidate.candidate_id} has no valid epoch")
        candidate_audit[candidate.candidate_id] = {
            "score": best_score,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "epoch_scores": epoch_scores,
        }
    selected = min(
        candidates,
        key=lambda item: (candidate_audit[item.candidate_id]["score"], item.candidate_id),
    )
    return selected, {
        **candidate_audit[selected.candidate_id],
        "candidate_scores": {
            candidate_id: value["score"] for candidate_id, value in candidate_audit.items()
        },
    }


def _validate_official_config(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("official config must be a JSON object")
    if value.get("schema_version") != "official-reconstruction-config-v3":
        raise ValueError("official config must use official-reconstruction-config-v3")
    methods = value.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {"imputeecg", "ecgrecover"}:
        raise ValueError("official config must contain exactly ImputeECG and ECGrecover")
    output = {}
    for method in ("imputeecg", "ecgrecover"):
        config = methods[method]
        if not isinstance(config, Mapping) or config.get("selection") != "official_fixed":
            raise ValueError(f"{method} must declare selection=official_fixed")
        parameters = config.get("training_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{method} requires fixed training_parameters")
        output[method] = dict(parameters)
    return output


def select_tuning_configuration(
    frame: pd.DataFrame,
    *,
    manifest_sha256: str,
    split_sha256: str,
    official_config: Mapping[str, Any],
    early_stop_trace: pd.DataFrame | None = None,
    fold8_records_sha256: str | None = None,
    configuration_panel_sha256: str | None = None,
    validated_grid: Mapping[str, Sequence[Candidate]] | None = None,
) -> dict[str, Any]:
    if validated_grid is None:
        grid = _validate_candidate_metrics(
            frame,
            manifest_sha256=manifest_sha256,
            split_sha256=split_sha256,
        )
    else:
        frozen_grid = candidate_grid()
        grid = {
            method: tuple(candidates) for method, candidates in validated_grid.items()
        }
        if grid != frozen_grid:
            raise ValueError("prevalidated candidate grid does not match the frozen grid")
    official_parameters = _validate_official_config(official_config)
    unet_rows = frame[frame["method"] == "masked-unet"]
    release_singletons = all(
        group["epoch"].nunique() == 1
        for _, group in unet_rows.groupby(["candidate_id", "model_seed"], sort=False)
    )
    early_stop_audit = None
    if release_singletons:
        if early_stop_trace is None:
            raise ValueError(
                "early-stopped singleton U-Net metrics require a verifiable early-stop trace"
            )
        early_stop_audit = _validate_early_stop_trace(
            early_stop_trace,
            frame,
            manifest_sha256=manifest_sha256,
            split_sha256=split_sha256,
            fold8_records_sha256=fold8_records_sha256,
            configuration_panel_sha256=configuration_panel_sha256,
        )
    methods: dict[str, dict[str, Any]] = {}
    selection_audit = {}
    for method in ("lowrank", "ridge"):
        selected, audit = _select_linear(frame[frame["method"] == method], grid[method])
        methods[method] = dict(selected.parameters)
        selection_audit[method] = {
            "candidate_id": selected.candidate_id,
            **audit,
        }
    selected_unet, unet_audit = _select_unet(
        frame[frame["method"] == "masked-unet"], grid["masked-unet"]
    )
    unet_parameters = dict(selected_unet.parameters)
    unet_parameters["epochs"] = int(unet_audit["best_epoch"])
    unet_parameters.pop("max_epochs")
    methods["masked-unet"] = unet_parameters
    selection_audit["masked-unet"] = {
        "candidate_id": selected_unet.candidate_id,
        **unet_audit,
    }
    if early_stop_audit is not None:
        selection_audit["masked-unet"]["early_stop_trace"] = {
            "fold8_records_sha256": early_stop_audit["fold8_records_sha256"],
            "configuration_panel_sha256": early_stop_audit[
                "configuration_panel_sha256"
            ],
            "selected_candidate": early_stop_audit["candidates"][
                selected_unet.candidate_id
            ],
        }
    methods.update(official_parameters)
    selection_audit.update(
        {
            method: {"selection": "official_fixed_no_outcome_tuning"}
            for method in ("imputeecg", "ecgrecover")
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "train_role": "folds1-7/train",
        "source_role": "fold8/tune",
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "selection_rule": (
            "minimum patient-balanced mean log(RMSE_mV); lexical candidate-id tie break; "
            "masked-unet epoch selected with patience-8 early stopping"
        ),
        "methods": methods,
        "selection_audit": selection_audit,
        "candidate_grid": {
            method: [asdict(candidate) for candidate in candidates]
            for method, candidates in grid.items()
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_bundle_artifact(root: Path, descriptor: Any, label: str) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"candidate bundle {label} descriptor is missing")
    relative = Path(str(descriptor.get("path", "")))
    expected_sha256 = str(descriptor.get("sha256", ""))
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError(f"candidate bundle {label} descriptor is unsafe")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"candidate bundle {label} artifact is missing or escapes its root")
    if lineage.artifact_sha256(path) != expected_sha256:
        raise ValueError(f"candidate bundle {label} SHA-256 mismatch")
    return path


def _load_candidate_bundle(path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    bundle_path = path.resolve()
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("candidate bundle must be a JSON object")
    if value.get("schema_version") != CANDIDATE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"candidate bundle must use {CANDIDATE_BUNDLE_SCHEMA_VERSION}"
        )
    if value.get("status") != "complete":
        raise ValueError("candidate bundle status must be complete")
    if value.get("train_partition") != "folds1-7/train" or value.get(
        "evaluation_partition"
    ) != "fold8/tune":
        raise ValueError("candidate bundle must be train folds1-7 -> fold8/tune")
    if value.get("holdout_partitions_loaded") != []:
        raise ValueError("candidate bundle must not load fold9/fold10 holdouts")
    if value.get("release") is not True or value.get("subsampled") is not False:
        raise ValueError("candidate bundle selection requires an unsubsampled release run")
    if (
        value.get("rate_hz") != 500
        or value.get("segments") != ["QRS", "ST", "T"]
        or value.get("delineator") != "dwt"
        or value.get("n_configurations") != 64
    ):
        raise ValueError("candidate bundle does not use the frozen release evaluation panel")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("candidate bundle artifacts are missing")
    root = bundle_path.parent
    metrics = _resolve_bundle_artifact(root, artifacts.get("candidate_metrics"), "metrics")
    trace = _resolve_bundle_artifact(root, artifacts.get("early_stop_trace"), "trace")
    inventory = _resolve_bundle_artifact(
        root, artifacts.get("candidate_metrics_inventory"), "metric inventory"
    )
    _resolve_bundle_artifact(
        root, artifacts.get("training_predictors"), "training predictors"
    )
    hashes = {
        "fold8_records_sha256": str(value.get("fold8_records_sha256", "")),
        "configuration_panel_sha256": str(
            value.get("configuration_panel_sha256", "")
        ),
    }
    if not all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()):
        raise ValueError("candidate bundle case/panel hashes must be full SHA-256 values")
    inventory_value = json.loads(inventory.read_text(encoding="utf-8"))
    metric_descriptor = artifacts["candidate_metrics"]
    manifest_descriptor = value.get("manifest")
    if not isinstance(manifest_descriptor, Mapping):
        raise ValueError("candidate bundle manifest descriptor is missing")
    inventory_identity = inventory_value.get("identity")
    if not isinstance(inventory_identity, Mapping):
        raise ValueError("candidate metric inventory identity is missing")
    if (
        inventory_value.get("schema_version") != "candidate-metric-inventory-v1"
        or inventory_value.get("status") != "complete"
        or inventory_value.get("candidate_metrics")
        != {
            "path": metrics.name,
            "sha256": str(metric_descriptor.get("sha256")),
        }
        or inventory_identity.get("manifest_sha256")
        != manifest_descriptor.get("sha256")
        or inventory_identity.get("split_sha256")
        != manifest_descriptor.get("split_sha256")
        or inventory_identity.get("fold8_records_sha256")
        != hashes["fold8_records_sha256"]
        or inventory_identity.get("configuration_panel_sha256")
        != hashes["configuration_panel_sha256"]
        or int(inventory_value.get("n_completed_units", -1))
        != int(inventory_value.get("n_expected_units", -2))
        or int(inventory_value.get("n_rows", -1))
        != int(metric_descriptor.get("n_rows", -2))
    ):
        raise ValueError("candidate metric inventory is not bound to the complete bundle")
    completed = inventory_value.get("completed_units")
    if (
        not isinstance(completed, list)
        or len(completed) != int(inventory_value["n_expected_units"])
        or any(
            not isinstance(item, Mapping)
            or item.get("staging_retained") is not False
            or item.get("path") is not None
            for item in completed
        )
    ):
        raise ValueError("candidate metric inventory was not cleanly published")
    return metrics, trace, inventory, hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-bundle", type=Path)
    source.add_argument("--candidate-metrics", type=Path)
    parser.add_argument(
        "--candidate-trace",
        type=Path,
        help="required with singleton U-Net rows when --candidate-metrics is used",
    )
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        # Candidate selection only accepts an unsubsampled release bundle, so its
        # source manifest must satisfy the same complete official contract.
        manifest = load_ptbxl_manifest(arguments.manifest, release=True)
        if arguments.output.exists():
            raise FileExistsError(arguments.output)
        bundle_hashes: dict[str, str] = {}
        candidate_bundle = None
        candidate_inventory = None
        if arguments.candidate_bundle is not None:
            if arguments.candidate_trace is not None:
                raise ValueError("--candidate-trace cannot be combined with --candidate-bundle")
            candidate_bundle = arguments.candidate_bundle.resolve()
            (
                candidate_metrics,
                candidate_trace,
                candidate_inventory,
                bundle_hashes,
            ) = _load_candidate_bundle(candidate_bundle)
        else:
            candidate_metrics = arguments.candidate_metrics.resolve()
            candidate_trace = (
                None
                if arguments.candidate_trace is None
                else arguments.candidate_trace.resolve()
            )
        scan = scan_candidate_metrics_parquet(
            candidate_metrics,
            manifest_sha256=manifest.manifest_sha256,
            split_sha256=manifest.split_sha256,
            expected_n_configurations=64 if candidate_bundle is not None else None,
        )
        frame = scan.patient_rows
        try:
            trace = None if candidate_trace is None else pd.read_parquet(candidate_trace)
        except ImportError as exc:
            raise RuntimeError("tuning requires locked pyarrow; CSV fallback is forbidden") from exc
        _verify_candidate_checkpoints(frame, candidate_metrics)
        official_config = json.loads(arguments.official_config.read_text(encoding="utf-8"))
        selected = select_tuning_configuration(
            frame,
            manifest_sha256=manifest.manifest_sha256,
            split_sha256=manifest.split_sha256,
            official_config=official_config,
            early_stop_trace=trace,
            fold8_records_sha256=bundle_hashes.get("fold8_records_sha256"),
            configuration_panel_sha256=bundle_hashes.get(
                "configuration_panel_sha256"
            ),
            validated_grid=scan.grid,
        )
        selected["candidate_metric_scan"] = {
            "mode": "bounded_parquet_row_groups",
            "n_rows": scan.n_rows,
            "n_row_groups": scan.n_row_groups,
            "n_configurations": len(scan.configurations),
            "cell_reference_sha256": scan.cell_reference_sha256,
        }
        selected["candidate_metrics"] = {
            "path": str(candidate_metrics),
            "sha256": lineage.artifact_sha256(candidate_metrics),
        }
        if candidate_trace is not None:
            selected["early_stop_trace"] = {
                "path": str(candidate_trace),
                "sha256": lineage.artifact_sha256(candidate_trace),
            }
        if candidate_bundle is not None:
            selected["candidate_bundle"] = {
                "path": str(candidate_bundle),
                "sha256": lineage.artifact_sha256(candidate_bundle),
            }
        if candidate_inventory is not None:
            selected["candidate_metrics_inventory"] = {
                "path": str(candidate_inventory),
                "sha256": lineage.artifact_sha256(candidate_inventory),
            }
        selected["official_config_sha256"] = lineage.artifact_sha256(
            arguments.official_config
        )
        _atomic_json(arguments.output.resolve(), selected)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"reconstructor tuning failed closed: {exc}") from exc
    print(f"[tuning] fold-8 selection -> {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
