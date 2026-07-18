import numpy as np
import pandas as pd

import ecgcert.evaluation as evaluation
from ecgcert.evaluation import (
    BootstrapEffect,
    aggregate_model_seed_outcomes,
    attach_seed_bootstrap_predictions,
    cluster_bootstrap_delta_r2,
    loco_meta_predictions,
    method_specific_delta_r2,
    stage15_decision,
    tune_meta_ridge_alpha,
)


def _frame(seed, patient_prefix):
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(20):
        patient_effect = rng.normal(scale=0.05)
        for config_index in range(6):
            ambiguity = 0.08 + 0.07 * config_index
            for method_index, method in enumerate(("low_rank", "ridge", "unet", "imputeecg")):
                outcome = -2.0 + 2.5 * ambiguity + 0.1 * method_index + patient_effect
                outcome += rng.normal(scale=0.02)
                rows.append(
                    {
                        "patient_id": f"{patient_prefix}{patient}",
                        "configuration": f"c{config_index}",
                        "method": method,
                        "segment": "QRS" if config_index % 2 else "ST",
                        "target": "V2" if config_index % 3 else "V5",
                        "n_observed": 1 + config_index % 3,
                        "configuration_rank": 1 + config_index % 3,
                        "log10_condition": 1.0 + 0.1 * (config_index % 2),
                        "target_rms": 0.4,
                        "max_target_observed_correlation": 0.5,
                        "ambiguity_robust_mv": ambiguity,
                        "outcome_log_rmse": outcome,
                        "model_seed": method_index if method in {"unet", "imputeecg"} else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def test_loco_score_adds_out_of_sample_information_and_bootstraps():
    calibration = _frame(1, "cal-")
    test = _frame(2, "test-")
    comparison = loco_meta_predictions(calibration, test, alpha=1e-3)
    assert comparison.delta_r2 > 0
    effect = cluster_bootstrap_delta_r2(comparison.predictions, replicates=100, seed=9)
    assert effect.point > 0
    assert effect.ci95[0] > 0
    assert all(value > 0 for value in method_specific_delta_r2(comparison.predictions).values())


def test_stage15_gate_is_fail_closed():
    good = BootstrapEffect(0.1, (0.02, 0.2), 2000, 1)
    weak = BootstrapEffect(0.01, (-0.02, 0.1), 2000, 1)
    proceed = stage15_decision(
        ptbxl=good,
        external={"Chapman": weak, "CPSC": good},
        method_deltas={"a": 0.1, "b": 0.1, "c": 0.1, "d": -0.1},
    )
    assert proceed.status == "PROCEED"
    pivot = stage15_decision(
        ptbxl=weak,
        external={"Chapman": weak},
        method_deltas={"a": 0.1, "b": -0.1, "c": -0.1, "d": -0.1},
    )
    assert pivot.status == "PIVOT"
    assert len(pivot.reasons) == 3


def test_meta_alpha_is_selected_using_loco_tune_rows():
    selection = tune_meta_ridge_alpha(_frame(4, "tune-"), grid=(0.0, 0.1, 10.0))
    assert selection.alpha in {0.0, 0.1, 10.0}
    assert list(selection.table.columns) == [
        "alpha", "mse_simple", "mse_augmented", "mean_mse"
    ]


def _five_seed_frame(seed: int, patient_prefix: str) -> pd.DataFrame:
    base = _frame(seed, patient_prefix)
    deterministic = base[~base["method"].isin({"unet", "imputeecg"})]
    neural = []
    for model_seed in range(5):
        rows = base[base["method"].isin({"unet", "imputeecg"})].copy()
        rows["model_seed"] = model_seed
        rows["outcome_log_rmse"] += 0.01 * (model_seed - 2)
        neural.append(rows)
    return pd.concat([deterministic, *neural], ignore_index=True)


def test_seed_mean_point_and_nested_seed_bootstrap_share_one_cell_estimand():
    calibration_raw = _five_seed_frame(7, "cal-seed-")
    test_raw = _five_seed_frame(8, "test-seed-")
    calibration = aggregate_model_seed_outcomes(calibration_raw)
    test = aggregate_model_seed_outcomes(test_raw)
    assert len(test) == len(_frame(8, "test-seed-"))
    comparison = loco_meta_predictions(calibration, test, alpha=1e-3)
    seed_predictions = attach_seed_bootstrap_predictions(
        test_raw, comparison.predictions
    )
    assert len(seed_predictions) > len(comparison.predictions)
    effect = cluster_bootstrap_delta_r2(
        comparison.predictions,
        bootstrap_predictions=seed_predictions,
        replicates=100,
        seed=19,
    )
    assert np.isclose(effect.point, comparison.delta_r2)


def test_patient_bootstrap_retries_until_exact_requested_valid_draw_count(monkeypatch):
    predictions = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "outcome_log_rmse": [0.0, 1.0, 2.0, 3.0],
            "prediction_simple": [1.5, 1.5, 1.5, 1.5],
            "prediction_augmented": [0.1, 0.9, 2.1, 2.9],
        }
    )
    invalid = predictions.copy()
    invalid["outcome_log_rmse"] = 1.0
    calls = 0

    def fake_bootstrap(_frame, _rng):
        nonlocal calls
        calls += 1
        return invalid if calls == 1 else predictions

    monkeypatch.setattr(evaluation, "_bootstrap_rows", fake_bootstrap)
    effect = cluster_bootstrap_delta_r2(predictions, replicates=100, seed=7)
    assert effect.replicates == 100
    assert calls == 101
