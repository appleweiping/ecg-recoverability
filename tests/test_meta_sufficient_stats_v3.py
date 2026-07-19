import tracemalloc

import numpy as np
import pandas as pd
import pytest

import ecgcert.evaluation as evaluation
from ecgcert.benchmarking import RELEASE_NEURAL_SEEDS
from experiments import meta_analysis_v3 as meta


def test_preregistered_meta_alpha_grid_matches_manuscript() -> None:
    assert evaluation.META_RIDGE_ALPHA_GRID == (
        0.0,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1_000.0,
    )


def _meta_frame(
    seed: int,
    prefix: str,
    *,
    configurations: tuple[str, ...] = ("c0", "c1", "c2", "c3"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(6):
        for config_index, configuration in enumerate(configurations):
            for method_index, method in enumerate(("lowrank", "ridge")):
                for segment_index, segment in enumerate(("QRS", "ST")):
                    for target_index, target in enumerate(("I", "V2")):
                        robust = (
                            0.06
                            + 0.035 * config_index
                            + 0.008 * target_index
                            + 0.0007 * patient
                        )
                        log_condition = 0.15 * config_index + 0.004 * patient
                        target_rms = 0.25 + 0.02 * target_index + 0.001 * patient
                        correlation = (
                            0.45 + 0.025 * method_index + 0.0015 * patient
                        )
                        outcome = (
                            -1.8
                            + 1.7 * robust
                            + 0.11 * log_condition
                            + 0.04 * method_index
                            + 0.025 * segment_index
                            + rng.normal(scale=0.006)
                        )
                        rows.append(
                            {
                                "row_id": (
                                    f"{prefix}-{patient}-{configuration}-{method}-"
                                    f"{segment}-{target}"
                                ),
                                "patient_id": f"{prefix}-{patient}",
                                "configuration": configuration,
                                "method": method,
                                "segment": segment,
                                "target": target,
                                "n_observed": 1 + config_index % 3,
                                "configuration_rank": 2 + config_index % 2,
                                "log10_condition": log_condition,
                                "target_rms": target_rms,
                                "max_target_observed_correlation": correlation,
                                "ambiguity_robust_mv": robust,
                                "outcome_log_rmse": outcome,
                            }
                        )
    return pd.DataFrame(rows)


def _sklearn_reference_predictions(
    calibration: pd.DataFrame, test: pd.DataFrame, *, alpha: float
) -> pd.DataFrame:
    outputs = []
    for configuration in sorted(test["configuration"].astype(str).unique()):
        train = calibration[
            calibration["configuration"].astype(str) != configuration
        ]
        held = test[test["configuration"].astype(str) == configuration].copy()
        for label, numeric in (
            ("simple", evaluation.SIMPLE_NUMERIC_PREDICTORS),
            (
                "augmented",
                (*evaluation.SIMPLE_NUMERIC_PREDICTORS, evaluation.ROBUST_SCORE),
            ),
        ):
            columns = [*evaluation.CATEGORICAL_PREDICTORS, *numeric]
            model = evaluation._pipeline(numeric, alpha)
            model.fit(train[columns], train["outcome_log_rmse"])
            held[f"prediction_{label}"] = model.predict(held[columns])
        outputs.append(held)
    return pd.concat(outputs, ignore_index=True)


def test_sufficient_loco_matches_previous_sklearn_path_on_small_data() -> None:
    calibration = _meta_frame(11, "cal")
    test = _meta_frame(12, "test")
    for alpha in (1e-3, 0.1, 10.0):
        expected = _sklearn_reference_predictions(
            calibration, test, alpha=alpha
        ).sort_values("row_id")
        actual = evaluation.loco_meta_predictions(
            calibration, test, alpha=alpha
        ).predictions.sort_values("row_id")
        np.testing.assert_allclose(
            actual["prediction_simple"],
            expected["prediction_simple"],
            rtol=2e-8,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            actual["prediction_augmented"],
            expected["prediction_augmented"],
            rtol=2e-8,
            atol=2e-8,
        )


def test_sufficient_alpha_tuning_matches_previous_sklearn_loco_mse() -> None:
    tune = _meta_frame(21, "tune")
    grid = (1e-3, 0.1, 10.0)
    expected = []
    for alpha in grid:
        predictions = _sklearn_reference_predictions(tune, tune, alpha=alpha)
        truth = predictions["outcome_log_rmse"].to_numpy(dtype=float)
        simple = predictions["prediction_simple"].to_numpy(dtype=float)
        augmented = predictions["prediction_augmented"].to_numpy(dtype=float)
        mse_simple = float(np.mean((truth - simple) ** 2))
        mse_augmented = float(np.mean((truth - augmented) ** 2))
        expected.append((mse_simple, mse_augmented))

    selection = evaluation.tune_meta_ridge_alpha(tune, grid=grid)
    np.testing.assert_allclose(
        selection.table[["mse_simple", "mse_augmented"]],
        np.asarray(expected),
        rtol=2e-7,
        atol=2e-10,
    )


def test_sufficient_alpha_tuning_preserves_fail_closed_identifier_validation() -> None:
    tune = _meta_frame(25, "invalid")
    tune.loc[0, "patient_id"] = None
    with pytest.raises(ValueError, match="identifiers contain missing values"):
        evaluation.tune_meta_ridge_alpha(tune, grid=(0.1, 1.0))


def test_zero_alpha_compatibility_path_uses_deterministic_lstsq(monkeypatch) -> None:
    calibration = _meta_frame(22, "cal-zero")
    test = _meta_frame(23, "test-zero")
    original_lstsq = evaluation.np.linalg.lstsq
    calls = []

    def counted_lstsq(*args, **kwargs):
        calls.append(1)
        return original_lstsq(*args, **kwargs)

    def solve_forbidden(*_args, **_kwargs):
        raise AssertionError("alpha=0 must use the deterministic least-squares path")

    monkeypatch.setattr(evaluation.np.linalg, "lstsq", counted_lstsq)
    monkeypatch.setattr(evaluation.np.linalg, "solve", solve_forbidden)
    first = evaluation.loco_meta_predictions(
        calibration, test, alpha=0.0
    ).predictions
    second = evaluation.loco_meta_predictions(
        calibration, test, alpha=0.0
    ).predictions
    columns = ["prediction_simple", "prediction_augmented"]
    assert np.isfinite(first[columns].to_numpy(dtype=float)).all()
    np.testing.assert_array_equal(first[columns], second[columns])
    assert len(calls) == 4 * 2 * 2  # configurations * models * repeated fits


def test_release_encoding_is_frozen_independently_of_observed_rows() -> None:
    encoding = meta._fixed_release_meta_encoding()
    expected = {
        "method": tuple(sorted(meta.COMMON_PANEL_METHODS)),
        "segment": tuple(sorted(meta.PRIMARY_SEGMENTS)),
        "target": tuple(sorted(meta.CANONICAL_LEADS)),
    }
    assert {
        name: levels
        for name, levels in zip(
            evaluation.CATEGORICAL_PREDICTORS,
            encoding.categorical_levels,
            strict=True,
        )
    } == expected
    assert encoding.numeric_columns == (
        *evaluation.SIMPLE_NUMERIC_PREDICTORS,
        evaluation.ROBUST_SCORE,
    )

    unknown = _meta_frame(24, "unknown").iloc[:1].copy()
    unknown["method"] = "not-a-release-method"
    with pytest.raises(ValueError, match="unknown method"):
        encoding.encode(unknown)


def test_scale_proxy_accumulates_without_concat_or_row_resident_growth(
    monkeypatch,
) -> None:
    configurations = tuple(f"c{index:02d}" for index in range(64))
    base_rows = []
    for index, configuration in enumerate(configurations):
        base_rows.append(
            {
                "patient_id": f"p{index % 17}",
                "configuration": configuration,
                "method": ("lowrank", "ridge")[index % 2],
                "segment": ("QRS", "ST")[index % 2],
                "target": ("I", "V2")[index % 2],
                "n_observed": 1 + index % 8,
                "configuration_rank": 2 + index % 4,
                "log10_condition": 0.01 * index,
                "target_rms": 0.2 + 0.001 * index,
                "max_target_observed_correlation": 0.4 + 0.001 * index,
                "ambiguity_robust_mv": 0.05 + 0.002 * index,
                "outcome_log_rmse": -2.0 + 0.004 * index,
            }
        )
    base = pd.DataFrame(base_rows)
    template = pd.concat([base] * 32, ignore_index=True)
    encoding = evaluation.infer_meta_encoding(template)

    def concat_forbidden(*_args, **_kwargs):
        raise AssertionError("configuration sufficient-stat accumulation used concat")

    monkeypatch.setattr(evaluation.pd, "concat", concat_forbidden)
    tracemalloc.start()
    statistics = evaluation.accumulate_meta_sufficient_statistics(
        (template for _ in range(256)),
        encoding=encoding,
        configurations=configurations,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert statistics.total_rows == 256 * len(template)
    resident_state_bytes = sum(
        array.nbytes
        for array in (
            statistics.row_count,
            statistics.feature_sum,
            statistics.feature_crossproduct,
            statistics.outcome_sum,
            statistics.feature_outcome_sum,
            statistics.outcome_square_sum,
        )
    )
    assert resident_state_bytes < 1_000_000
    assert peak < 96 * 1024 * 1024
    forty_million_dense_rows = 40_000_000 * encoding.n_features * 8
    assert resident_state_bytes < forty_million_dense_rows / 1_000


def test_fold9_bank_is_fit_once_then_reused_without_sklearn(monkeypatch) -> None:
    calibration = _meta_frame(31, "cal")
    encoding = evaluation.infer_meta_encoding(calibration)
    configurations = tuple(sorted(calibration["configuration"].unique()))
    statistics = evaluation.accumulate_meta_sufficient_statistics(
        (calibration,), encoding=encoding, configurations=configurations
    )
    original_fit = evaluation.fit_loco_meta_model_bank
    calls = []

    def counted_fit(statistics, *, alpha):
        calls.append(float(alpha))
        return original_fit(statistics, alpha=alpha)

    def sklearn_forbidden(*_args, **_kwargs):
        raise AssertionError("streaming meta-analysis refit a sklearn pipeline")

    monkeypatch.setattr(evaluation, "fit_loco_meta_model_bank", counted_fit)
    monkeypatch.setattr(evaluation, "_pipeline", sklearn_forbidden)
    selection = evaluation.tune_meta_ridge_alpha_from_sufficient(
        statistics, grid=(0.01, 1.0)
    )
    bank = evaluation.fit_loco_meta_model_bank(
        statistics, alpha=selection.alpha
    )
    assert len(calls) == 3  # two fold-8 candidates, then one frozen fold-9 bank
    assert len(bank.simple) == len(configurations)
    assert len(bank.augmented) == len(configurations)

    for cohort, seed in (("PTB-XL", 41), ("chapman", 42), ("cpsc2018", 43)):
        frame = _meta_frame(seed, cohort)
        predicted = evaluation.predict_with_loco_meta_bank(bank, frame)
        assert len(predicted) == len(frame)
    assert len(calls) == 3


def test_patient_predictions_are_arrow_streamed_with_seed_contract(
    tmp_path, monkeypatch
) -> None:
    configurations = ("c0", "c1")
    rank_map = pd.DataFrame(
        [
            {
                "segment": "QRS",
                "configuration": configuration,
                "target": "V2",
                "ambiguity_robust_mv": 0.08 + 0.04 * index,
                "configuration_rank_max": 2 + index,
                "log10_condition_max": 0.2 + 0.1 * index,
            }
            for index, configuration in enumerate(configurations)
        ]
    )
    rows = []
    for method_index, method in enumerate(meta.COMMON_PANEL_METHODS):
        seeds = RELEASE_NEURAL_SEEDS if method in meta.NEURAL_METHODS else (0,)
        for model_seed in seeds:
            for patient in range(3):
                for config_index, configuration in enumerate(configurations):
                    rows.append(
                        {
                            "schema_version": "fixture-v3",
                            "cohort": "fixture",
                            "partition": "test",
                            "patient_id": f"p{patient}",
                            "method": method,
                            "model_seed": model_seed,
                            "segment": "QRS",
                            "configuration": configuration,
                            "target": "V2",
                            "n_observed": 1 + config_index,
                            "n_records": 1,
                            "n_samples": 100,
                            "target_rms": 0.3,
                            "max_target_observed_correlation": 0.5,
                            "outcome_log_rmse": (
                                -1.5
                                + 0.1 * config_index
                                + 0.03 * method_index
                                + 0.002 * model_seed
                                + 0.005 * patient
                            ),
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics_path = tmp_path / "patient_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    calibration = meta._attach_robust_map(
        metrics[metrics["model_seed"] == 0].copy(), rank_map
    )
    encoding = evaluation.infer_meta_encoding(calibration)
    statistics = evaluation.accumulate_meta_sufficient_statistics(
        (calibration,), encoding=encoding, configurations=configurations
    )
    bank = evaluation.fit_loco_meta_model_bank(statistics, alpha=1.0)
    point_path = tmp_path / "point.parquet"
    seed_path = tmp_path / "seed.parquet"
    sufficient_path = tmp_path / "sufficient.parquet"
    paired_path = tmp_path / "paired-sufficient.parquet"

    def full_pandas_read_forbidden(*_args, **_kwargs):
        raise AssertionError("patient prediction evidence must remain Arrow-streamed")

    monkeypatch.setattr(pd, "read_parquet", full_pandas_read_forbidden)
    sufficient, report = meta._stream_loco_prediction_evidence(
        {method: metrics_path for method in meta.COMMON_PANEL_METHODS},
        rank_map,
        bank,
        cohort="fixture",
        point_path=point_path,
        seed_path=seed_path,
        sufficient_path=sufficient_path,
        paired_sufficient_path=paired_path,
        batch_rows=3,
    )

    import pyarrow.parquet as pq

    point_file = pq.ParquetFile(point_path)
    seed_file = pq.ParquetFile(seed_path)
    assert point_file.metadata.num_rows == 24
    assert point_file.metadata.num_row_groups > 1
    assert seed_file.metadata.num_rows == 72
    assert seed_file.metadata.num_row_groups > 1
    assert report["point_prediction_rows"] == 24
    assert report["seed_prediction_rows"] == 72
    assert report["fold9_model_bank_reused"] is True
    assert report["fold9_loco_models"] == 4
    assert set(sufficient["estimand"]) == {"point_seed_mean", "seed_specific"}
    assert sufficient_path.is_file()
    assert paired_path.is_file()
    paired = pq.read_table(paired_path).to_pandas()
    meta._validate_paired_sufficient_contract(paired, cohort="fixture")

    tampered = sufficient.copy()
    extra = tampered.iloc[[0]].copy()
    extra["estimand"] = "unregistered"
    with pytest.raises(ValueError, match="unknown estimand"):
        meta._validate_sufficient_contract(
            pd.concat([tampered, extra], ignore_index=True), cohort="fixture"
        )
