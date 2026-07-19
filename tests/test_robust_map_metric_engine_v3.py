from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert.physics import LEADS
from ecgcert.recoverability import bootstrap_spatial_model_bank
import experiments.robust_maps_v3 as robust_maps


def _bank(*, n_boot: int = 12):
    rng = np.random.default_rng(20260719)
    latent = rng.normal(size=(120, 4))
    mixing, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    values = latent @ mixing.T + rng.normal(scale=0.02, size=(120, 12))
    patients = np.asarray(
        [f"patient-{index // 10:02d}" for index in range(len(values))],
        dtype=object,
    )
    return bootstrap_spatial_model_bank(
        values,
        patients,
        ranks=(2, 3),
        basis_variants=("independent8_lifted",),
        n_boot=n_boot,
        seed=31,
    )


def _scalar_batch_metrics(batch, configuration, observation_variance_mv2):
    rows = [
        robust_maps._model_metrics(
            model,
            configuration,
            observation_variance_mv2,
        )
        for model in batch.models
    ]
    output = {}
    for name in rows[0]:
        values = np.asarray([row[name] for row in rows])
        if name == "configuration_rank":
            values = values.astype(np.int16)
        output[name] = values
    return output


def _summary(bank, *, workers: int):
    draws = []
    rank_path, map_cells = robust_maps.summarize_model_bank(
        bank,
        (("I",), ("I", "II"), ("II", "V2", "V5"), ("I", "V1", "V3", "V6")),
        segment="QRS",
        observation_variance_mv2=1e-4,
        draw_sink=lambda frame: draws.append(frame.copy(deep=True)),
        metric_workers=workers,
    )
    return rank_path, map_cells, pd.concat(draws, ignore_index=True)


def test_vectorized_metric_engine_is_numerically_equivalent_to_scalar(monkeypatch) -> None:
    bank = _bank()
    vector_rank, vector_map, vector_draws = _summary(bank, workers=1)
    monkeypatch.setattr(robust_maps, "_batched_model_metrics", _scalar_batch_metrics)
    scalar_rank, scalar_map, scalar_draws = _summary(bank, workers=1)

    pd.testing.assert_frame_equal(
        vector_rank,
        scalar_rank,
        check_exact=False,
        rtol=2e-10,
        atol=2e-12,
    )
    pd.testing.assert_frame_equal(
        vector_map,
        scalar_map,
        check_exact=False,
        rtol=2e-10,
        atol=2e-12,
    )
    identity = [
        "schema_version",
        "segment",
        "configuration",
        "target",
        "rank",
        "basis_variant",
        "bootstrap_index",
    ]
    pd.testing.assert_frame_equal(vector_draws[identity], scalar_draws[identity])
    pd.testing.assert_frame_equal(
        vector_draws.drop(columns=identity),
        scalar_draws.drop(columns=identity),
        check_exact=False,
        rtol=2e-10,
        atol=2e-12,
    )


def test_metric_workers_are_bitwise_deterministic_and_preserve_draw_order() -> None:
    bank = _bank()
    serial = _summary(bank, workers=1)
    parallel = _summary(bank, workers=4)
    for serial_frame, parallel_frame in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(serial_frame, parallel_frame, check_exact=True)
    draws = parallel[2]
    expected_configurations = ["I", "I+II", "II+V2+V5", "I+V1+V3+V6"]
    assert list(dict.fromkeys(draws["configuration"])) == expected_configurations
    for configuration in expected_configurations:
        rows = draws[draws["configuration"] == configuration]
        assert list(dict.fromkeys(rows["rank"])) == [2, 3]
        for rank in (2, 3):
            rank_rows = rows[rows["rank"] == rank]
            assert list(dict.fromkeys(rank_rows["target"])) == list(LEADS)
            for target in LEADS:
                assert rank_rows.loc[
                    rank_rows["target"] == target, "bootstrap_index"
                ].tolist() == list(range(bank.n_boot))


def test_vectorization_batches_by_rank_not_by_individual_model(monkeypatch) -> None:
    bank = _bank(n_boot=9)
    original = robust_maps._batched_model_metrics
    batch_sizes = []

    def recording_batch(batch, configuration, observation_variance_mv2):
        batch_sizes.append(len(batch.models))
        return original(batch, configuration, observation_variance_mv2)

    monkeypatch.setattr(robust_maps, "_batched_model_metrics", recording_batch)
    robust_maps.summarize_model_bank(
        bank,
        (("I",), ("II", "V1"), ("I", "V2", "V6")),
        segment="ST",
        observation_variance_mv2=1e-3,
        metric_workers=3,
    )
    # Per configuration and rank: one point batch and one complete bootstrap batch.
    assert len(batch_sizes) == 3 * 2 * 2
    assert batch_sizes.count(1) == 3 * 2
    assert batch_sizes.count(bank.n_boot) == 3 * 2


def test_dag_cpu_allocation_drives_bounded_metric_workers(monkeypatch) -> None:
    monkeypatch.setenv("ECGCERT_NUM_WORKERS", "10")
    assert robust_maps._metric_worker_count(None) == 10
    monkeypatch.setenv("ECGCERT_NUM_WORKERS", "11")
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        robust_maps._metric_worker_count(None)


def test_release_workspace_and_parquet_row_groups_remain_configuration_bounded(
    tmp_path: Path,
) -> None:
    release_workspace = robust_maps._metric_workspace_proxy_bytes(
        n_boot=2_000,
        ranks=(2, 3, 4, 5),
        max_observed=8,
    )
    assert release_workspace < 16 * 2**20
    assert release_workspace * robust_maps.MAX_METRIC_WORKERS < 160 * 2**20
    assert robust_maps._draw_rows_per_configuration(2_000, 4) == 96_000

    bank = _bank(n_boot=5)
    configurations = (("I",), ("I", "II"), ("II", "V2", "V5"))
    output = tmp_path / "draws.parquet"
    writer = robust_maps._AtomicParquetWriter(output)
    try:
        robust_maps.summarize_model_bank(
            bank,
            configurations,
            segment="T",
            observation_variance_mv2=1e-4,
            draw_sink=writer.write_frame,
            metric_workers=2,
        )
    except Exception:
        writer.close(publish=False)
        raise
    writer.close(publish=True)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(output)
    expected_rows = robust_maps._draw_rows_per_configuration(
        bank.n_boot, len(bank.ranks)
    )
    assert parquet.num_row_groups == len(configurations)
    assert [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.num_row_groups)
    ] == [expected_rows] * len(configurations)
