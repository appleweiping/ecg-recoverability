"""Locked fold-9 to fold-10 meta-model and Stage-15 evidence gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CATEGORICAL_PREDICTORS = ("method", "segment", "target")
SIMPLE_NUMERIC_PREDICTORS = (
    "n_observed",
    "configuration_rank",
    "log10_condition",
    "target_rms",
    "max_target_observed_correlation",
)
ROBUST_SCORE = "ambiguity_robust_mv"
META_RIDGE_ALPHA_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0)
SEED_CELL_COLUMNS = ("patient_id", "method", "segment", "configuration", "target")


@dataclass(frozen=True)
class PredictionComparison:
    predictions: pd.DataFrame
    r2_simple: float
    r2_augmented: float
    delta_r2: float


@dataclass(frozen=True)
class BootstrapEffect:
    point: float
    ci95: tuple[float, float]
    replicates: int
    seed: int


@dataclass(frozen=True)
class Stage15Decision:
    status: str
    reasons: tuple[str, ...]
    ptbxl: BootstrapEffect
    external: Mapping[str, BootstrapEffect]
    positive_methods: int
    required_positive_methods: int


@dataclass(frozen=True)
class MetaAlphaSelection:
    alpha: float
    table: pd.DataFrame


def _required_columns(robust_score: str) -> set[str]:
    return {
        "patient_id",
        "configuration",
        "outcome_log_rmse",
        *CATEGORICAL_PREDICTORS,
        *SIMPLE_NUMERIC_PREDICTORS,
        robust_score,
    }


def _validate_frame(frame: pd.DataFrame, robust_score: str) -> None:
    missing = _required_columns(robust_score) - set(frame.columns)
    if missing:
        raise ValueError(f"meta-model frame is missing columns: {sorted(missing)}")
    numeric = [*SIMPLE_NUMERIC_PREDICTORS, robust_score, "outcome_log_rmse"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("meta-model predictors/outcome contain non-finite values")
    if frame[["patient_id", "configuration", *CATEGORICAL_PREDICTORS]].isna().any().any():
        raise ValueError("meta-model identifiers contain missing values")


def _pipeline(numeric: Sequence[str], alpha: float):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transform = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_PREDICTORS)),
            ("numeric", StandardScaler(), list(numeric)),
        ]
    )
    return Pipeline([("features", transform), ("ridge", Ridge(alpha=float(alpha)))])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0:
        raise ValueError("R2 is undefined for a constant outcome")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def loco_meta_predictions(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    robust_score: str = ROBUST_SCORE,
) -> PredictionComparison:
    """Fit on fold 9 and predict fold 10 with each test configuration excluded.

    ``alpha`` must already have been selected on fold 8.  No fold-10 outcome is
    touched during fitting.
    """

    _validate_frame(calibration, robust_score)
    _validate_frame(test, robust_score)
    if alpha < 0:
        raise ValueError("ridge alpha must be non-negative")

    chunks: list[pd.DataFrame] = []
    simple_numeric = list(SIMPLE_NUMERIC_PREDICTORS)
    augmented_numeric = [*SIMPLE_NUMERIC_PREDICTORS, robust_score]
    for configuration in sorted(test["configuration"].astype(str).unique()):
        train_rows = calibration[calibration["configuration"].astype(str) != configuration]
        test_rows = test[test["configuration"].astype(str) == configuration].copy()
        if train_rows.empty:
            raise ValueError(f"no calibration configurations remain after excluding {configuration}")
        simple = _pipeline(simple_numeric, alpha)
        augmented = _pipeline(augmented_numeric, alpha)
        simple.fit(train_rows, train_rows["outcome_log_rmse"])
        augmented.fit(train_rows, train_rows["outcome_log_rmse"])
        test_rows["prediction_simple"] = simple.predict(test_rows)
        test_rows["prediction_augmented"] = augmented.predict(test_rows)
        chunks.append(test_rows)

    predictions = pd.concat(chunks, ignore_index=True)
    y = predictions["outcome_log_rmse"].to_numpy(dtype=float)
    r2_simple = _r2(y, predictions["prediction_simple"].to_numpy(dtype=float))
    r2_augmented = _r2(y, predictions["prediction_augmented"].to_numpy(dtype=float))
    return PredictionComparison(
        predictions=predictions,
        r2_simple=r2_simple,
        r2_augmented=r2_augmented,
        delta_r2=r2_augmented - r2_simple,
    )


def tune_meta_ridge_alpha(
    tune: pd.DataFrame,
    *,
    grid: Sequence[float] = META_RIDGE_ALPHA_GRID,
    robust_score: str = ROBUST_SCORE,
) -> MetaAlphaSelection:
    """Select the shared meta-model ridge penalty using fold 8 only.

    Each candidate is evaluated by leave-one-configuration-out prediction within
    fold 8. The neutral criterion is the mean MSE of the simple and augmented
    models, so tuning does not privilege the proposed score.
    """

    values = tuple(float(alpha) for alpha in grid)
    if not values or len(set(values)) != len(values) or any(
        not np.isfinite(alpha) or alpha < 0 for alpha in values
    ):
        raise ValueError("meta alpha grid must contain unique finite non-negative values")
    rows = []
    for alpha in values:
        comparison = loco_meta_predictions(tune, tune, alpha=alpha, robust_score=robust_score)
        truth = comparison.predictions["outcome_log_rmse"].to_numpy(dtype=float)
        simple_error = float(np.mean(
            (truth - comparison.predictions["prediction_simple"].to_numpy(dtype=float)) ** 2
        ))
        augmented_error = float(np.mean(
            (truth - comparison.predictions["prediction_augmented"].to_numpy(dtype=float)) ** 2
        ))
        rows.append({
            "alpha": alpha,
            "mse_simple": simple_error,
            "mse_augmented": augmented_error,
            "mean_mse": 0.5 * (simple_error + augmented_error),
        })
    table = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    best = float(table["mean_mse"].min())
    alpha = float(
        table.loc[np.isclose(table["mean_mse"], best, rtol=1e-12), "alpha"].min()
    )
    return MetaAlphaSelection(alpha=alpha, table=table)


def aggregate_model_seed_outcomes(
    frame: pd.DataFrame,
    *,
    robust_score: str = ROBUST_SCORE,
) -> pd.DataFrame:
    """Average neural-seed outcomes without giving neural methods extra weight.

    The scientific estimand is one patient/method/segment/configuration/target
    cell.  Deterministic methods already have one row per cell; neural methods
    contribute the mean patient-level log-RMSE over their preregistered seeds.
    Predictors must be identical across seeds, otherwise aggregation would hide
    a broken evaluation contract.
    """

    _validate_frame(frame, robust_score)
    identifiers = [
        *(column for column in ("cohort", "partition") if column in frame.columns),
        *SEED_CELL_COLUMNS,
    ]
    static_columns = [
        *SIMPLE_NUMERIC_PREDICTORS,
        robust_score,
    ]
    grouped = frame.groupby(identifiers, sort=False, dropna=False)
    for column in static_columns:
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"meta-model predictor {column!r} changes across model seeds")
    aggregated = grouped[static_columns].first().reset_index()
    aggregated["outcome_log_rmse"] = grouped["outcome_log_rmse"].mean().to_numpy()
    aggregated["model_seed"] = "seed-mean"
    return aggregated


def attach_seed_bootstrap_predictions(
    raw_seed_rows: pd.DataFrame,
    point_predictions: pd.DataFrame,
    *,
    robust_score: str = ROBUST_SCORE,
) -> pd.DataFrame:
    """Attach seed-mean meta predictions to seed-specific held-out outcomes."""

    _validate_frame(raw_seed_rows, robust_score)
    _validate_frame(point_predictions, robust_score)
    identifiers = [
        *(
            column
            for column in ("cohort", "partition")
            if column in raw_seed_rows.columns and column in point_predictions.columns
        ),
        *SEED_CELL_COLUMNS,
    ]
    prediction_columns = [*identifiers, "prediction_simple", "prediction_augmented"]
    missing = set(prediction_columns) - set(point_predictions.columns)
    if missing:
        raise ValueError(f"point predictions are missing columns: {sorted(missing)}")
    predictions = point_predictions[prediction_columns]
    if predictions.duplicated(identifiers).any():
        raise ValueError("seed-mean predictions are not unique by scientific cell")
    attached = raw_seed_rows.merge(
        predictions,
        on=identifiers,
        how="left",
        validate="many_to_one",
    )
    if attached[["prediction_simple", "prediction_augmented"]].isna().any().any():
        raise ValueError("seed-specific outcomes contain cells absent from point predictions")
    return attached


def _bootstrap_rows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patients = frame["patient_id"].astype(str).unique()
    sampled = rng.choice(patients, size=len(patients), replace=True)
    chunks = []
    for bootstrap_id, patient in enumerate(sampled):
        rows = frame[frame["patient_id"].astype(str) == patient].copy()
        rows["_bootstrap_patient"] = f"{bootstrap_id}:{patient}"
        chunks.append(rows)
    boot = pd.concat(chunks, ignore_index=True)

    if "model_seed" in boot.columns:
        selected = []
        for method, rows in boot.groupby("method", sort=False):
            seeds = rows["model_seed"].dropna().unique()
            if len(seeds) > 1:
                seed = rng.choice(seeds)
                rows = rows[rows["model_seed"].isna() | (rows["model_seed"] == seed)]
            selected.append(rows)
        boot = pd.concat(selected, ignore_index=True)
    return boot


def cluster_bootstrap_delta_r2(
    predictions: pd.DataFrame,
    *,
    replicates: int = 2_000,
    seed: int = 20260719,
    bootstrap_predictions: pd.DataFrame | None = None,
) -> BootstrapEffect:
    """Patient-cluster bootstrap with nested neural model-seed resampling.

    ``predictions`` contains the one-row-per-cell seed-mean point estimand.
    ``bootstrap_predictions`` may contain seed-specific outcomes with the same
    fitted predictions; each bootstrap replicate selects one seed globally per
    neural method before resampling patients.  This keeps the point estimate and
    confidence interval on the same, equally weighted method/configuration panel.
    """

    required = {"patient_id", "outcome_log_rmse", "prediction_simple", "prediction_augmented"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    point = prediction_delta_r2(predictions)
    bootstrap_source = predictions if bootstrap_predictions is None else bootstrap_predictions
    missing_bootstrap = required - set(bootstrap_source.columns)
    if missing_bootstrap:
        raise ValueError(
            "bootstrap prediction frame is missing columns: "
            f"{sorted(missing_bootstrap)}"
        )
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    max_attempts = max(replicates * 10, replicates + 1_000)
    while len(values) < replicates and attempts < max_attempts:
        attempts += 1
        sample = _bootstrap_rows(bootstrap_source, rng)
        truth = sample["outcome_log_rmse"].to_numpy(dtype=float)
        try:
            delta = _r2(truth, sample["prediction_augmented"].to_numpy(dtype=float)) - _r2(
                truth, sample["prediction_simple"].to_numpy(dtype=float)
            )
        except ValueError:
            continue
        values.append(delta)
    if len(values) != replicates:
        raise RuntimeError(
            "could not obtain the requested number of valid patient bootstrap "
            f"replicates ({len(values)}/{replicates} after {attempts} attempts)"
        )
    lower, upper = np.percentile(values, [2.5, 97.5])
    return BootstrapEffect(float(point), (float(lower), float(upper)), replicates, seed)


def prediction_delta_r2(predictions: pd.DataFrame) -> float:
    """Recompute the paired augmented-minus-simple R2 point estimand."""

    required = {"outcome_log_rmse", "prediction_simple", "prediction_augmented"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    truth = predictions["outcome_log_rmse"].to_numpy(dtype=float)
    return float(
        _r2(truth, predictions["prediction_augmented"].to_numpy(dtype=float))
        - _r2(truth, predictions["prediction_simple"].to_numpy(dtype=float))
    )


def method_specific_delta_r2(predictions: pd.DataFrame) -> dict[str, float]:
    out = {}
    for method, rows in predictions.groupby("method"):
        out[str(method)] = prediction_delta_r2(rows)
    return out


def stage15_decision(
    *,
    ptbxl: BootstrapEffect,
    external: Mapping[str, BootstrapEffect],
    method_deltas: Mapping[str, float],
    required_positive_methods: int = 3,
) -> Stage15Decision:
    """Apply the preregistered PROCEED/PIVOT rule without discretionary overrides."""

    reasons = []
    if ptbxl.ci95[0] <= 0:
        reasons.append("PTB-XL fold-10 delta-R2 lower confidence bound is not positive")
    if not external or not any(effect.ci95[0] > 0 for effect in external.values()):
        reasons.append("no external zero-shot cohort has a positive delta-R2 lower bound")
    positive_methods = sum(float(delta) > 0 for delta in method_deltas.values())
    if positive_methods < required_positive_methods:
        reasons.append(
            f"only {positive_methods} methods have positive point estimates; "
            f"{required_positive_methods} required"
        )
    return Stage15Decision(
        status="PROCEED" if not reasons else "PIVOT",
        reasons=tuple(reasons),
        ptbxl=ptbxl,
        external=external,
        positive_methods=positive_methods,
        required_positive_methods=required_positive_methods,
    )
