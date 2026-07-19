"""Formal-scale bounded-memory and resume contracts for fold-8 tuning metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.reconstruction_candidates_v3 import (
    CANDIDATE_COLUMNS,
    CANDIDATE_METRIC_INVENTORY_FILENAME,
    CandidateMetricStore,
    _remove_orphan_training_workdirs,
)
from experiments.tune_reconstructors_v3 import (
    CANDIDATE_SCHEMA_VERSION,
    scan_candidate_metrics_parquet,
)


MANIFEST_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def _identity(configurations: tuple[tuple[str, ...], ...]) -> dict:
    return {
        "schema_version": "fixture",
        "manifest_sha256": MANIFEST_SHA,
        "split_sha256": SPLIT_SHA,
        "configurations": [list(value) for value in configurations],
    }


def _frame(
    unit: tuple[str, str, int, str],
    *,
    n_patients: int = 1,
    cells_per_patient: int = 1,
) -> pd.DataFrame:
    method, candidate_id, seed, configuration = unit
    if cells_per_patient not in {1, 12}:
        raise ValueError("fixture supports one or twelve cells per patient")
    rows = []
    cell_values = [("QRS", "V1")]
    if cells_per_patient == 12:
        cell_values = [
            (segment, target)
            for segment in ("QRS", "ST", "T")
            for target in ("V1", "V2", "V3", "V4")
        ]
    epoch = 2 if method == "masked-unet" else 0
    checkpoint_sha256 = (candidate_id.encode().hex() + "0" * 64)[:64]
    for patient in range(n_patients):
        for cell_index, (segment, target) in enumerate(cell_values):
            log_rmse = -2.0 + patient * 1e-7 + cell_index * 1e-6
            rmse = float(np.exp(log_rmse))
            rows.append(
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "cohort": "PTB-XL",
                    "train_partition": "folds1-7/train",
                    "partition": "fold8/tune",
                    "manifest_sha256": MANIFEST_SHA,
                    "split_sha256": SPLIT_SHA,
                    "method": method,
                    "candidate_id": candidate_id,
                    "patient_id": f"patient-{patient:05d}",
                    "segment": segment,
                    "configuration": configuration,
                    "target": target,
                    "model_seed": seed,
                    "epoch": epoch,
                    "rmse_mv": rmse,
                    "log_rmse_mv": float(np.log(rmse)),
                    "checkpoint_path": f"checkpoints/{candidate_id}-seed-{seed}.ckpt",
                    "checkpoint_sha256": checkpoint_sha256,
                    "observed_integrity": True,
                }
            )
    return pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))


def test_candidate_metric_store_resumes_atomic_units_and_publishes_once(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    configurations = (("I",),)
    output = tmp_path / "candidates"
    first = CandidateMetricStore(
        output, identity=_identity(configurations), configurations=configurations
    )
    prefix = first.expected[:7]
    for unit in prefix:
        first.write_frame(_frame(unit))

    resumed = CandidateMetricStore(
        output, identity=_identity(configurations), configurations=configurations
    )
    assert all(resumed.is_complete(*unit) for unit in prefix)
    for unit in resumed.expected[7:]:
        resumed.write_frame(_frame(unit))
    metrics = resumed.finalize()
    scan = scan_candidate_metrics_parquet(
        metrics,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
        expected_n_configurations=1,
    )
    assert scan.n_rows == len(resumed.expected)
    assert scan.n_row_groups == len(resumed.expected)
    resumed.cleanup_staging()
    assert not resumed.staging_dir.exists()
    complete = CandidateMetricStore(
        output, identity=_identity(configurations), configurations=configurations
    )
    assert complete.status == "complete"
    assert complete.metrics_path == metrics
    assert (output / CANDIDATE_METRIC_INVENTORY_FILENAME).is_file()


def test_formal_scale_scan_never_uses_full_pandas_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    configurations = (("I",), ("II",))
    output = tmp_path / "large-candidates"
    store = CandidateMetricStore(
        output, identity=_identity(configurations), configurations=configurations
    )
    for unit in store.expected:
        store.write_frame(_frame(unit, n_patients=300, cells_per_patient=12))
    metrics = store.finalize()

    def forbid_full_read(*_args, **_kwargs):
        raise AssertionError("formal candidate metrics must not use pd.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", forbid_full_read)
    scan = scan_candidate_metrics_parquet(
        metrics,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
        expected_n_configurations=2,
    )
    assert scan.n_rows == len(store.expected) * 300 * 12
    # 17 deterministic linear fits plus 3 candidates x 3 U-Net seeds.
    assert len(scan.patient_rows) == 26 * 300
    assert scan.n_row_groups == len(store.expected)


def test_candidate_metric_resume_fails_closed_on_identity_change(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    configurations = (("I",),)
    output = tmp_path / "candidates"
    store = CandidateMetricStore(
        output, identity=_identity(configurations), configurations=configurations
    )
    store.write_frame(_frame(store.expected[0]))
    changed = _identity(configurations)
    changed["split_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="resume identity changed"):
        CandidateMetricStore(output, identity=changed, configurations=configurations)


def test_orphan_training_cleanup_is_limited_to_private_workdirs(tmp_path: Path) -> None:
    output = tmp_path / "candidates"
    orphan = output / ".ecgcert-candidates-dead-process"
    unrelated = output / "checkpoints"
    orphan.mkdir(parents=True)
    unrelated.mkdir()
    (orphan / "train_signals.npy").write_bytes(b"partial")
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    _remove_orphan_training_workdirs(output)
    assert not orphan.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
