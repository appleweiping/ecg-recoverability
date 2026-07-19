import json
from types import SimpleNamespace
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.evaluation import (
    _bootstrap_rows,
    _bootstrap_rows_with_audit,
    cluster_bootstrap_delta_r2,
    prediction_delta_r2,
)
from ecgcert.physics import LEADS
from ecgcert.recoverability import (
    bootstrap_attempts_table,
    bootstrap_moments_table,
    bootstrap_spatial_model_bank,
)
from experiments.meta_analysis_v3 import (
    META_BOOTSTRAP_DRAW_SCHEMA_VERSION,
    META_METRIC_COLUMNS,
    _bootstrap_effect_and_draws,
    _bootstrap_effect_and_draws_from_sufficient,
    _expected_common_seeds,
    _delta_from_sufficient_rows,
    _patient_moments_for_seed_multiplicities,
    _prepare_paired_method_arrays,
    _rebuild_paired_sufficient_from_seed_evidence,
    _rebuild_sufficient_from_seed_evidence,
    _require_sufficient_equal,
    _require_paired_sufficient_equal,
    _validate_rank_map_bootstrap_evidence,
    _validate_streamed_point_seed_binding,
    _validate_streamed_bootstrap_multiplicities,
    _write_seed_evidence_and_sufficient,
)
from experiments.robust_maps_v3 import (
    BOOTSTRAP_AUDIT_SCHEMA_VERSION,
    _AtomicParquetWriter,
    _write_bootstrap_design,
    summarize_model_bank,
)


def _streaming_seed_fixture(tmp_path):
    methods = ("lowrank", "ridge", "masked-unet", "imputeecg")
    sources = {}
    predictions = []
    for method_index, method in enumerate(methods):
        rows = []
        for model_seed in _expected_common_seeds(method):
            for patient_index in range(6):
                outcome = patient_index + method_index * 0.1 + model_seed * 0.001
                rows.append(
                    {
                        "schema_version": "reconstruction-benchmark-v3",
                        "cohort": "PTB-XL",
                        "partition": "test",
                        "patient_id": f"patient-{patient_index}",
                        "method": method,
                        "model_seed": model_seed,
                        "segment": "QRS",
                        "configuration": "I",
                        "target": "II",
                        "n_observed": 1,
                        "n_records": 1,
                        "n_samples": 10,
                        "target_rms": 1.0,
                        "max_target_observed_correlation": 0.5,
                        "outcome_log_rmse": outcome,
                    }
                )
        source = tmp_path / f"{method}.parquet"
        pd.DataFrame(rows, columns=META_METRIC_COLUMNS).to_parquet(
            source, index=False, row_group_size=2
        )
        sources[method] = source
        seed_mean = np.mean(_expected_common_seeds(method)) * 0.001
        for patient_index in range(6):
            predictions.append(
                {
                    "cohort": "PTB-XL",
                    "partition": "test",
                    "patient_id": f"patient-{patient_index}",
                    "method": method,
                    "segment": "QRS",
                    "configuration": "I",
                    "target": "II",
                    "outcome_log_rmse": patient_index + method_index * 0.1 + seed_mean,
                    "prediction_simple": 2.0,
                    "prediction_augmented": 2.2 + method_index * 0.01,
                }
            )
    seed_path = tmp_path / "seed_predictions.parquet"
    sufficient_path = tmp_path / "sufficient.parquet"
    paired_path = tmp_path / "paired-sufficient.parquet"
    sufficient, report = _write_seed_evidence_and_sufficient(
        sources,
        pd.DataFrame(predictions),
        cohort="PTB-XL",
        seed_path=seed_path,
        sufficient_path=sufficient_path,
        paired_sufficient_path=paired_path,
        batch_rows=2,
    )
    return seed_path, sufficient, pd.read_parquet(paired_path), report


def test_multigroup_seed_evidence_rebuild_is_streaming_and_exact(
    tmp_path, monkeypatch
) -> None:
    import pyarrow.parquet as pq

    seed_path, sufficient, paired, report = _streaming_seed_fixture(tmp_path)
    assert report["seed_prediction_row_groups"] > 4
    assert pq.ParquetFile(seed_path).metadata.num_row_groups > 4

    def forbid_full_pandas_read(*_args, **_kwargs):
        raise AssertionError("large seed evidence must not use pd.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", forbid_full_pandas_read)
    rebuilt, rebuilt_report = _rebuild_sufficient_from_seed_evidence(
        seed_path, cohort="PTB-XL"
    )
    _require_sufficient_equal(sufficient, rebuilt, label="PTB-XL")
    rebuilt_paired, _paired_report = _rebuild_paired_sufficient_from_seed_evidence(
        seed_path, cohort="PTB-XL"
    )
    _require_paired_sufficient_equal(paired, rebuilt_paired, label="PTB-XL")
    assert rebuilt_report["exact_seed_contract"]["masked-unet"] == [0, 1, 2, 3, 4]
    effect, draws = _bootstrap_effect_and_draws_from_sufficient(
        rebuilt,
        paired_sufficient=rebuilt_paired,
        cohort="PTB-XL",
        replicates=100,
        seed=20260719,
    )
    assert np.isfinite([effect.point, *effect.ci95]).all()
    assert len(draws) == 100
    selections = [json.loads(value) for value in draws["selected_model_seeds_json"]]
    for method in ("masked-unet", "imputeecg"):
        assert all(len(selection[method]) == 5 for selection in selections)
        assert any(len(set(selection[method])) < 5 for selection in selections)

    tampered = paired.copy()
    crossproducts = json.loads(tampered.loc[0, "truth_crossproducts_json"])
    crossproducts[0] += 0.5
    tampered.loc[0, "truth_crossproducts_json"] = json.dumps(crossproducts)
    with pytest.raises(ValueError, match="paired sufficient"):
        _require_paired_sufficient_equal(tampered, rebuilt_paired, label="PTB-XL")


def test_paired_moments_reconstruct_point_and_apply_repeated_seed_multiplicity(
    tmp_path,
) -> None:
    _seed_path, sufficient, paired, _report = _streaming_seed_fixture(tmp_path)
    patients, arrays = _prepare_paired_method_arrays(paired, cohort="PTB-XL")
    uniform = np.zeros((len(patients), 5), dtype=float)
    for method, values in arrays.items():
        uniform += _patient_moments_for_seed_multiplicities(
            values, tuple(1 for _seed in values["seeds"])
        )
    totals = uniform.sum(axis=0)
    denominator = totals[1] - totals[0] ** 2 / totals[4]
    paired_delta = totals[2] / denominator - totals[3] / denominator
    point_rows = sufficient[sufficient["estimand"] == "point_seed_mean"]
    assert np.isclose(
        paired_delta,
        _delta_from_sufficient_rows(point_rows),
        rtol=0.0,
        atol=1e-12,
    )

    neural = arrays["masked-unet"]
    seed_count = len(neural["seeds"])
    repeated = _patient_moments_for_seed_multiplicities(
        neural, (seed_count, *(0 for _ in range(seed_count - 1)))
    )
    seed_mean = _patient_moments_for_seed_multiplicities(
        neural, tuple(1 for _seed in neural["seeds"])
    )
    assert not np.allclose(repeated[:, :4], seed_mean[:, :4], rtol=0.0, atol=1e-15)


def test_streamed_seed_evidence_rejects_one_missing_neural_seed(tmp_path) -> None:
    import pyarrow.parquet as pq

    seed_path, _sufficient, _paired, _report = _streaming_seed_fixture(tmp_path)
    parquet = pq.ParquetFile(seed_path)
    broken = tmp_path / "broken-seed-predictions.parquet"
    writer = None
    removed = 0
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group)
        frame = table.to_pandas()
        masked_cells = frame.loc[frame["method"] == "masked-unet", "cell_index"]
        if not masked_cells.empty and removed == 0:
            victim = (
                (frame["method"] == "masked-unet")
                & (frame["cell_index"] == masked_cells.min())
                & (frame["model_seed"] == 4)
            )
            removed = int(victim.sum())
            table = type(table).from_pandas(frame.loc[~victim], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(broken, table.schema)
        writer.write_table(table, row_group_size=table.num_rows)
    assert writer is not None
    writer.close()
    assert removed == 1
    with pytest.raises(ValueError, match="exact preregistered seeds|missing, duplicated"):
        _rebuild_sufficient_from_seed_evidence(broken, cohort="PTB-XL")


def _point_predictions_from_seed_evidence(seed_path, point_path) -> pd.DataFrame:
    seed = pd.read_parquet(seed_path)
    identifiers = [
        "cohort",
        "partition",
        "patient_id",
        "method",
        "segment",
        "configuration",
        "target",
    ]
    point = (
        seed.groupby(identifiers, sort=False, as_index=False)
        .agg(
            outcome_log_rmse=("outcome_log_rmse", "mean"),
            prediction_simple=("prediction_simple", "first"),
            prediction_augmented=("prediction_augmented", "first"),
        )
    )
    point.to_parquet(point_path, index=False, row_group_size=3)
    return point


def test_point_and_seed_prediction_artifacts_are_streamed_and_bidirectionally_bound(
    tmp_path, monkeypatch
) -> None:
    seed_path, _sufficient, _paired, _report = _streaming_seed_fixture(tmp_path)
    point_path = tmp_path / "point-predictions.parquet"
    point = _point_predictions_from_seed_evidence(seed_path, point_path)

    def forbid_full_pandas_read(*_args, **_kwargs):
        raise AssertionError("point/seed binding must not bulk-read Parquet")

    monkeypatch.setattr(pd, "read_parquet", forbid_full_pandas_read)
    report = _validate_streamed_point_seed_binding(
        point_path, seed_path, cohort="PTB-XL", batch_rows=2
    )
    assert report["point_rows"] == len(point)
    assert report["max_point_block_rows"] <= 2

    changed = point.copy()
    changed.loc[0, "prediction_augmented"] += 1.0
    changed.to_parquet(point_path, index=False, row_group_size=3)
    with pytest.raises(ValueError, match="change authenticated predictions"):
        _validate_streamed_point_seed_binding(
            point_path, seed_path, cohort="PTB-XL", batch_rows=2
        )

    point.iloc[:-1].to_parquet(point_path, index=False, row_group_size=3)
    with pytest.raises(ValueError, match="omit seed-evidence cells"):
        _validate_streamed_point_seed_binding(
            point_path, seed_path, cohort="PTB-XL", batch_rows=2
        )

    extra = pd.concat(
        [
            point,
            point.iloc[[-1]].assign(patient_id="point-only-patient"),
        ],
        ignore_index=True,
    )
    extra.to_parquet(point_path, index=False, row_group_size=3)
    with pytest.raises(ValueError, match="absent from seed evidence"):
        _validate_streamed_point_seed_binding(
            point_path, seed_path, cohort="PTB-XL", batch_rows=2
        )


def test_seed_evidence_requires_identical_scientific_cells_across_four_methods(
    tmp_path,
) -> None:
    import pyarrow.parquet as pq

    seed_path, _sufficient, _paired, _report = _streaming_seed_fixture(tmp_path)
    parquet = pq.ParquetFile(seed_path)
    broken = tmp_path / "different-method-cells.parquet"
    writer = None
    changed = 0
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group)
        frame = table.to_pandas()
        selected = frame["method"].eq("lowrank")
        changed += int(selected.sum())
        frame.loc[selected, "target"] = "III"
        table = type(table).from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(broken, table.schema)
        writer.write_table(table, row_group_size=table.num_rows)
    assert writer is not None
    writer.close()
    assert changed > 0
    with pytest.raises(ValueError, match="equal four-method common panel"):
        _rebuild_sufficient_from_seed_evidence(broken, cohort="PTB-XL")


def test_rank_map_raw_draws_rebuild_quantiles_and_reject_999(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(818)
    X = rng.normal(size=(72, 12))
    patient_ids = np.asarray([f"patient-{index // 8}" for index in range(len(X))])
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(2, 3),
        basis_variants=("raw12_pca",),
        n_boot=5,
        seed=31,
        fit_cohort="PTB-XL/folds1-7/QRS",
    )

    draw_writer = _AtomicParquetWriter(tmp_path / "bootstrap_draws.parquet")
    rank_path, map_cells = summarize_model_bank(
        bank,
        (("I",),),
        segment="QRS",
        observation_variance_mv2=1e-4,
        draw_sink=draw_writer.write_frame,
    )
    draw_writer.close(publish=True)
    rank_path.to_parquet(tmp_path / "rank_path.parquet", index=False, compression="zstd")
    map_cells.to_parquet(tmp_path / "map_cells.parquet", index=False, compression="zstd")

    patient_writer = _AtomicParquetWriter(tmp_path / "bootstrap_patients.parquet")
    multiplicity_writer = _AtomicParquetWriter(
        tmp_path / "bootstrap_multiplicities.parquet"
    )
    _write_bootstrap_design(
        bank=bank,
        segment="QRS",
        patient_writer=patient_writer,
        multiplicity_writer=multiplicity_writer,
    )
    patient_writer.close(publish=True)
    multiplicity_writer.close(publish=True)
    pd.DataFrame(
        [
            {
                "schema_version": BOOTSTRAP_AUDIT_SCHEMA_VERSION,
                "segment": "QRS",
                "basis_variant": "raw12_pca",
                "seed": 31,
                "n_patients": len(bank.patient_ids),
                "requested_draws": bank.n_boot,
                "attempted_draws": bank.n_boot + bank.rejected_draws,
                "status": status,
                "draw_count": count,
            }
            for status, count in (
                ("accepted", bank.n_boot),
                ("rejected_rank_deficient", bank.rejected_draws),
            )
        ]
    ).to_parquet(tmp_path / "bootstrap_audit.parquet", index=False, compression="zstd")
    import pyarrow.parquet as pq

    pq.write_table(
        bootstrap_moments_table(bank, segment="QRS"),
        tmp_path / "bootstrap_moments.parquet",
        compression="zstd",
    )
    pq.write_table(
        bootstrap_attempts_table(bank, segment="QRS"),
        tmp_path / "bootstrap_attempts.parquet",
        compression="zstd",
    )
    artifacts = {
        "rank_path": "rank_path.parquet",
        "map_cells": "map_cells.parquet",
        "bootstrap_draws": "bootstrap_draws.parquet",
        "bootstrap_patients": "bootstrap_patients.parquet",
        "bootstrap_multiplicities": "bootstrap_multiplicities.parquet",
        "bootstrap_audit": "bootstrap_audit.parquet",
        "bootstrap_moments": "bootstrap_moments.parquet",
        "bootstrap_attempts": "bootstrap_attempts.parquet",
    }
    summary = {
        "segments": ["QRS"],
        "ranks": [2, 3],
        "basis_variant": "raw12_pca",
        "bootstrap_replicates": bank.n_boot,
        "seed": 31,
        "observation_variance_mv2": 1e-4,
        "n_bootstrap_draw_rows": bank.n_boot * 2 * len(LEADS),
        "n_bootstrap_attempt_rows": bank.attempt_ledger.n_attempts,
        "bootstrap_rank_deficient_draws": {
            "QRS": {
                "rejected_draws": bank.rejected_draws,
                "rejection_fraction": bank.rejection_fraction,
            }
        },
        "artifacts": artifacts,
        "artifact_sha256": {
            key: lineage.artifact_sha256(tmp_path / value)
            for key, value in artifacts.items()
        },
    }

    def forbid_full_pandas_read(*_args, **_kwargs):
        raise AssertionError("release bootstrap evidence must use Arrow readers")

    with monkeypatch.context() as patch:
        patch.setattr(pd, "read_parquet", forbid_full_pandas_read)
        _validate_rank_map_bootstrap_evidence(
            tmp_path,
            summary,
            expected_configurations={"I"},
            expected_targets=set(LEADS),
        )

    persisted = pd.read_parquet(tmp_path / "rank_path.parquet")
    tampered_rank_path = persisted.copy()
    tampered_rank_path.loc[0, "ambiguity_q975_mv"] = 999.0
    tampered_rank_path.to_parquet(
        tmp_path / "rank_path.parquet", index=False, compression="zstd"
    )
    with pytest.raises(ValueError, match="raw patient-bootstrap draws"):
        _validate_rank_map_bootstrap_evidence(
            tmp_path,
            summary,
            expected_configurations={"I"},
            expected_targets=set(LEADS),
        )

    persisted.to_parquet(tmp_path / "rank_path.parquet", index=False, compression="zstd")
    persisted_map = pd.read_parquet(tmp_path / "map_cells.parquet")
    persisted_map.loc[0, "ambiguity_robust_mv"] = 999.0
    persisted_map.to_parquet(
        tmp_path / "map_cells.parquet", index=False, compression="zstd"
    )
    with pytest.raises(ValueError, match="raw patient-bootstrap draws"):
        _validate_rank_map_bootstrap_evidence(
            tmp_path,
            summary,
            expected_configurations={"I"},
            expected_targets=set(LEADS),
        )


def test_multiplicity_replay_is_multigroup_streamed_and_memory_bounded(
    tmp_path, monkeypatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    replicates = 200
    n_patients = 4_096
    multiplicities = np.ones((replicates, n_patients), dtype=np.uint16)
    path = tmp_path / "many-multiplicities.parquet"

    def write(values: np.ndarray) -> None:
        table = pa.table(
            {
                "schema_version": pa.array(
                    [BOOTSTRAP_AUDIT_SCHEMA_VERSION] * replicates
                ),
                "segment": pa.array(["QRS"] * replicates),
                "basis_variant": pa.array(["raw12_pca"] * replicates),
                "bootstrap_index": pa.array(
                    np.arange(replicates, dtype=np.int64)
                ),
                "accepted": pa.array([True] * replicates),
                "multiplicities": pa.FixedSizeListArray.from_arrays(
                    pa.array(values.reshape(-1), type=pa.uint16()), n_patients
                ),
            }
        )
        pq.write_table(table, path, compression="zstd", row_group_size=17)

    write(multiplicities)
    assert pq.ParquetFile(path).num_row_groups > 4
    bank = SimpleNamespace(
        patient_ids=tuple(f"patient-{index}" for index in range(n_patients)),
        bootstrap_multiplicities=multiplicities,
    )

    def forbid_full_pandas_read(*_args, **_kwargs):
        raise AssertionError("release multiplicities must never use pd.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", forbid_full_pandas_read)
    tracemalloc.start()
    report = _validate_streamed_bootstrap_multiplicities(
        path,
        segments=("QRS",),
        basis_variant="raw12_pca",
        replicates=replicates,
        replayed_banks={"QRS": bank},
        batch_rows=4,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert report["rows"] == replicates
    assert report["max_batch_rows"] <= 4
    assert report["max_materialized_vector_bytes"] == n_patients * 8
    assert report["max_arrow_batch_bytes"] < 1 * 2**20
    assert peak < 16 * 2**20

    tampered = multiplicities.copy()
    tampered[0, 0] = 0
    tampered[0, 1] = 2
    write(tampered)
    with pytest.raises(ValueError, match="replayed attempt ledger"):
        _validate_streamed_bootstrap_multiplicities(
            path,
            segments=("QRS",),
            basis_variant="raw12_pca",
            replicates=replicates,
            replayed_banks={"QRS": bank},
            batch_rows=4,
        )


def test_meta_draws_persist_nested_seed_selection_and_are_reproducible() -> None:
    point_rows = []
    seed_rows = []
    for method in ("lowrank", "masked-unet"):
        seeds = (0,) if method == "lowrank" else (11, 22)
        for patient_index in range(5):
            common = {
                "cohort": "PTB-XL",
                "partition": "test",
                "patient_id": f"p{patient_index}",
                "method": method,
                "segment": "QRS",
                "configuration": "I",
                "target": "II",
                "prediction_simple": 0.0,
                "prediction_augmented": float(patient_index) * 0.9,
            }
            outcomes = []
            for model_seed in seeds:
                outcome = float(patient_index) + 0.01 * model_seed
                outcomes.append(outcome)
                seed_rows.append(
                    {**common, "model_seed": model_seed, "outcome_log_rmse": outcome}
                )
            point_rows.append({**common, "outcome_log_rmse": float(np.mean(outcomes))})
    point = pd.DataFrame(point_rows)
    seed_specific = pd.DataFrame(seed_rows)

    audited_sample, patient_draw, audited_seed_draws = _bootstrap_rows_with_audit(
        seed_specific, np.random.default_rng(991)
    )
    bootstrap_ids = {
        method: tuple(sorted(rows["_bootstrap_patient"].astype(str)))
        for method, rows in audited_sample.groupby("method", sort=False)
    }
    assert len(set(bootstrap_ids.values())) == 1
    assert len(patient_draw) == 5
    assert len(audited_seed_draws["masked-unet"]) == 2

    first_effect, first = _bootstrap_effect_and_draws(
        point,
        bootstrap_predictions=seed_specific,
        cohort="PTB-XL",
        replicates=100,
        seed=77,
    )
    second_effect, second = _bootstrap_effect_and_draws(
        point,
        bootstrap_predictions=seed_specific,
        cohort="PTB-XL",
        replicates=100,
        seed=77,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_effect == second_effect
    legacy_effect = cluster_bootstrap_delta_r2(
        point,
        bootstrap_predictions=seed_specific,
        replicates=100,
        seed=77,
    )
    assert np.isclose(first_effect.point, legacy_effect.point, rtol=0.0, atol=1e-12)
    assert np.allclose(first_effect.ci95, legacy_effect.ci95, rtol=0.0, atol=1e-12)
    legacy_rng = np.random.default_rng(77)
    legacy_draws = [
        prediction_delta_r2(_bootstrap_rows(seed_specific, legacy_rng))
        for _ in range(100)
    ]
    assert np.allclose(first["delta_r2"], legacy_draws, rtol=0.0, atol=2e-12)
    assert set(first["schema_version"]) == {META_BOOTSTRAP_DRAW_SCHEMA_VERSION}
    selections = [json.loads(value) for value in first["selected_model_seeds_json"]]
    assert all(len(selection["masked-unet"]) == 2 for selection in selections)
    assert all(set(selection["masked-unet"]) <= {11, 22} for selection in selections)
    assert any(len(set(selection["masked-unet"])) == 1 for selection in selections)
    assert list(first["bootstrap_index"]) == list(range(100))
    assert first["patient_draw_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
