"""Locked fold-9 to fold-10 meta-model and Stage-15 evidence gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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
STAGE15_GATE_ELIGIBLE_EXTERNAL_COHORTS = ("chapman",)
STAGE15_COMMON_PANEL_METHODS = ("lowrank", "ridge", "masked-unet", "imputeecg")
STAGE15_REQUIRED_POSITIVE_METHODS = 3


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
    gate_eligible_external_cohorts: tuple[str, ...]
    qualifying_external_cohorts: tuple[str, ...]


@dataclass(frozen=True)
class MetaAlphaSelection:
    alpha: float
    table: pd.DataFrame


@dataclass(frozen=True)
class MetaFeatureEncoding:
    """Frozen one-hot/numeric design shared by every LOCO split and cohort."""

    categorical_levels: tuple[tuple[str, ...], ...]
    numeric_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.categorical_levels) != len(CATEGORICAL_PREDICTORS):
            raise ValueError("meta encoding must define every categorical predictor")
        for name, levels in zip(
            CATEGORICAL_PREDICTORS, self.categorical_levels, strict=True
        ):
            if not levels or tuple(sorted(set(levels))) != levels:
                raise ValueError(f"{name} levels must be sorted, unique, and non-empty")
        if not self.numeric_columns or len(set(self.numeric_columns)) != len(
            self.numeric_columns
        ):
            raise ValueError("meta numeric encoding must be non-empty and unique")

    @property
    def one_hot_columns(self) -> int:
        return sum(len(levels) for levels in self.categorical_levels)

    @property
    def n_features(self) -> int:
        return self.one_hot_columns + len(self.numeric_columns)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        missing = {
            *CATEGORICAL_PREDICTORS,
            *self.numeric_columns,
        } - set(frame.columns)
        if missing:
            raise ValueError(f"meta encoding input lacks columns: {sorted(missing)}")
        matrix = np.zeros((len(frame), self.n_features), dtype=np.float64)
        offset = 0
        rows = np.arange(len(frame), dtype=np.int64)
        for name, levels in zip(
            CATEGORICAL_PREDICTORS, self.categorical_levels, strict=True
        ):
            codes = pd.Categorical(frame[name].astype(str), categories=levels).codes
            if np.any(codes < 0):
                raise ValueError(f"meta encoding encountered an unknown {name}")
            matrix[rows, offset + codes] = 1.0
            offset += len(levels)
        numeric = frame[list(self.numeric_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("meta encoding encountered non-finite numeric predictors")
        matrix[:, offset:] = numeric
        return matrix


@dataclass(frozen=True)
class MetaConfigurationSufficientStatistics:
    """Additive outcome/design moments partitioned by observed configuration."""

    encoding: MetaFeatureEncoding
    configurations: tuple[str, ...]
    row_count: np.ndarray
    feature_sum: np.ndarray
    feature_crossproduct: np.ndarray
    outcome_sum: np.ndarray
    feature_outcome_sum: np.ndarray
    outcome_square_sum: np.ndarray

    def __post_init__(self) -> None:
        configurations = tuple(str(value) for value in self.configurations)
        if not configurations or len(configurations) != len(set(configurations)):
            raise ValueError("meta sufficient configurations must be non-empty and unique")
        c = len(configurations)
        p = self.encoding.n_features
        arrays = {
            "row_count": (self.row_count, (c,)),
            "feature_sum": (self.feature_sum, (c, p)),
            "feature_crossproduct": (self.feature_crossproduct, (c, p, p)),
            "outcome_sum": (self.outcome_sum, (c,)),
            "feature_outcome_sum": (self.feature_outcome_sum, (c, p)),
            "outcome_square_sum": (self.outcome_square_sum, (c,)),
        }
        normalized = {}
        for name, (raw, shape) in arrays.items():
            dtype = np.int64 if name == "row_count" else np.float64
            value = np.asarray(raw, dtype=dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"meta sufficient {name} must be finite with shape {shape}")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            normalized[name] = value
        if np.any(normalized["row_count"] <= 0):
            raise ValueError("every frozen configuration requires at least one meta row")
        object.__setattr__(self, "configurations", configurations)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def total_rows(self) -> int:
        return int(self.row_count.sum())


@dataclass(frozen=True)
class FittedMetaRidge:
    raw_coefficients: np.ndarray
    raw_intercept: float

    def predict_encoded(self, encoded: np.ndarray) -> np.ndarray:
        return np.asarray(encoded, dtype=float) @ self.raw_coefficients + self.raw_intercept


@dataclass(frozen=True)
class LocoMetaModelBank:
    encoding: MetaFeatureEncoding
    alpha: float
    simple: Mapping[str, FittedMetaRidge]
    augmented: Mapping[str, FittedMetaRidge]


def fixed_meta_encoding(
    *,
    categorical_levels: Mapping[str, Sequence[str]],
    robust_score: str = ROBUST_SCORE,
) -> MetaFeatureEncoding:
    levels = []
    for name in CATEGORICAL_PREDICTORS:
        values = tuple(sorted({str(value) for value in categorical_levels.get(name, ())}))
        levels.append(values)
    return MetaFeatureEncoding(
        categorical_levels=tuple(levels),
        numeric_columns=(*SIMPLE_NUMERIC_PREDICTORS, robust_score),
    )


def infer_meta_encoding(
    frame: pd.DataFrame,
    *,
    robust_score: str = ROBUST_SCORE,
) -> MetaFeatureEncoding:
    return fixed_meta_encoding(
        categorical_levels={
            name: tuple(frame[name].astype(str).unique())
            for name in CATEGORICAL_PREDICTORS
        },
        robust_score=robust_score,
    )


def accumulate_meta_sufficient_statistics(
    batches: Iterable[pd.DataFrame],
    *,
    encoding: MetaFeatureEncoding,
    configurations: Sequence[str],
) -> MetaConfigurationSufficientStatistics:
    configuration_order = tuple(str(value) for value in configurations)
    if not configuration_order or len(configuration_order) != len(set(configuration_order)):
        raise ValueError("meta sufficient configuration order must be non-empty and unique")
    configuration_index = {
        configuration: index for index, configuration in enumerate(configuration_order)
    }
    c = len(configuration_order)
    p = encoding.n_features
    count = np.zeros(c, dtype=np.int64)
    feature_sum = np.zeros((c, p), dtype=np.float64)
    feature_crossproduct = np.zeros((c, p, p), dtype=np.float64)
    outcome_sum = np.zeros(c, dtype=np.float64)
    feature_outcome_sum = np.zeros((c, p), dtype=np.float64)
    outcome_square_sum = np.zeros(c, dtype=np.float64)
    saw_rows = False
    for frame in batches:
        if frame.empty:
            continue
        missing = {"configuration", "outcome_log_rmse"} - set(frame.columns)
        if missing:
            raise ValueError(f"meta sufficient batch lacks columns: {sorted(missing)}")
        codes = frame["configuration"].astype(str).map(configuration_index)
        if codes.isna().any():
            raise ValueError("meta sufficient batch contains an unknown configuration")
        encoded = encoding.encode(frame)
        outcome = frame["outcome_log_rmse"].to_numpy(dtype=np.float64)
        if not np.isfinite(outcome).all():
            raise ValueError("meta sufficient batch contains non-finite outcomes")
        code_values = codes.to_numpy(dtype=np.int64)
        for code in np.unique(code_values):
            selected = code_values == code
            x = encoded[selected]
            y = outcome[selected]
            count[code] += len(y)
            feature_sum[code] += x.sum(axis=0)
            feature_crossproduct[code] += x.T @ x
            outcome_sum[code] += y.sum()
            feature_outcome_sum[code] += x.T @ y
            outcome_square_sum[code] += y @ y
        saw_rows = True
    if not saw_rows:
        raise ValueError("meta sufficient input is empty")
    return MetaConfigurationSufficientStatistics(
        encoding=encoding,
        configurations=configuration_order,
        row_count=count,
        feature_sum=feature_sum,
        feature_crossproduct=feature_crossproduct,
        outcome_sum=outcome_sum,
        feature_outcome_sum=feature_outcome_sum,
        outcome_square_sum=outcome_square_sum,
    )


def _fit_meta_ridge_from_moments(
    *,
    row_count: int,
    feature_sum: np.ndarray,
    feature_crossproduct: np.ndarray,
    outcome_sum: float,
    feature_outcome_sum: np.ndarray,
    alpha: float,
    numeric_start: int,
    feature_indices: np.ndarray,
    full_feature_count: int,
) -> FittedMetaRidge:
    if row_count < 1:
        raise ValueError("meta ridge requires at least one training row")
    indices = np.asarray(feature_indices, dtype=np.int64)
    sx = feature_sum[indices]
    gram = feature_crossproduct[np.ix_(indices, indices)]
    sxy = feature_outcome_sum[indices]
    centered_gram = gram - np.outer(sx, sx) / row_count
    centered_rhs = sxy - sx * outcome_sum / row_count
    scales = np.ones(len(indices), dtype=np.float64)
    numeric = indices >= numeric_start
    centered_diagonal = np.diag(centered_gram)[numeric]
    uncentered_diagonal = np.diag(gram)[numeric]
    numeric_sum = sx[numeric]
    roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum(
        uncentered_diagonal, numeric_sum**2 / row_count
    )
    numeric_scales = np.ones(len(centered_diagonal), dtype=np.float64)
    positive_variance = centered_diagonal > roundoff
    numeric_scales[positive_variance] = np.sqrt(
        centered_diagonal[positive_variance] / row_count
    )
    scales[numeric] = numeric_scales
    inverse_scale = 1.0 / scales
    scaled_gram = centered_gram * np.outer(inverse_scale, inverse_scale)
    scaled_rhs = centered_rhs * inverse_scale
    if alpha > 0:
        coefficients = np.linalg.solve(
            scaled_gram + float(alpha) * np.eye(len(indices)), scaled_rhs
        )
    else:
        coefficients = np.linalg.lstsq(scaled_gram, scaled_rhs, rcond=None)[0]
    raw_selected = coefficients * inverse_scale
    raw_coefficients = np.zeros(full_feature_count, dtype=np.float64)
    raw_coefficients[indices] = raw_selected
    intercept = float(outcome_sum / row_count - (sx / row_count) @ raw_selected)
    return FittedMetaRidge(raw_coefficients=raw_coefficients, raw_intercept=intercept)


def fit_loco_meta_model_bank(
    statistics: MetaConfigurationSufficientStatistics,
    *,
    alpha: float,
) -> LocoMetaModelBank:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha must be finite and non-negative")
    total_count = int(statistics.row_count.sum())
    total_feature_sum = statistics.feature_sum.sum(axis=0)
    total_crossproduct = statistics.feature_crossproduct.sum(axis=0)
    total_outcome_sum = float(statistics.outcome_sum.sum())
    total_feature_outcome = statistics.feature_outcome_sum.sum(axis=0)
    one_hot = statistics.encoding.one_hot_columns
    simple_indices = np.arange(
        one_hot + len(SIMPLE_NUMERIC_PREDICTORS), dtype=np.int64
    )
    augmented_indices = np.arange(statistics.encoding.n_features, dtype=np.int64)
    simple = {}
    augmented = {}
    for index, configuration in enumerate(statistics.configurations):
        values = {
            "row_count": total_count - int(statistics.row_count[index]),
            "feature_sum": total_feature_sum - statistics.feature_sum[index],
            "feature_crossproduct": (
                total_crossproduct - statistics.feature_crossproduct[index]
            ),
            "outcome_sum": total_outcome_sum - float(statistics.outcome_sum[index]),
            "feature_outcome_sum": (
                total_feature_outcome - statistics.feature_outcome_sum[index]
            ),
        }
        simple[configuration] = _fit_meta_ridge_from_moments(
            **values,
            alpha=alpha,
            numeric_start=one_hot,
            feature_indices=simple_indices,
            full_feature_count=statistics.encoding.n_features,
        )
        augmented[configuration] = _fit_meta_ridge_from_moments(
            **values,
            alpha=alpha,
            numeric_start=one_hot,
            feature_indices=augmented_indices,
            full_feature_count=statistics.encoding.n_features,
        )
    return LocoMetaModelBank(
        encoding=statistics.encoding,
        alpha=float(alpha),
        simple=simple,
        augmented=augmented,
    )


def predict_with_loco_meta_bank(
    bank: LocoMetaModelBank, frame: pd.DataFrame
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("LOCO prediction batch is empty")
    encoded = bank.encoding.encode(frame)
    output = frame.copy()
    simple = np.empty(len(frame), dtype=np.float64)
    augmented = np.empty(len(frame), dtype=np.float64)
    configurations = frame["configuration"].astype(str).to_numpy()
    for configuration in np.unique(configurations):
        if configuration not in bank.simple or configuration not in bank.augmented:
            raise ValueError(f"LOCO bank lacks held configuration {configuration}")
        selected = configurations == configuration
        simple[selected] = bank.simple[configuration].predict_encoded(encoded[selected])
        augmented[selected] = bank.augmented[configuration].predict_encoded(
            encoded[selected]
        )
    output["prediction_simple"] = simple
    output["prediction_augmented"] = augmented
    return output


def _sse_for_block(
    statistics: MetaConfigurationSufficientStatistics,
    index: int,
    model: FittedMetaRidge,
) -> float:
    n = float(statistics.row_count[index])
    sx = statistics.feature_sum[index]
    gram = statistics.feature_crossproduct[index]
    sy = float(statistics.outcome_sum[index])
    sxy = statistics.feature_outcome_sum[index]
    sy2 = float(statistics.outcome_square_sum[index])
    coef = model.raw_coefficients
    intercept = model.raw_intercept
    return float(
        sy2
        - 2.0 * (intercept * sy + coef @ sxy)
        + n * intercept**2
        + 2.0 * intercept * (coef @ sx)
        + coef @ gram @ coef
    )


def tune_meta_ridge_alpha_from_sufficient(
    statistics: MetaConfigurationSufficientStatistics,
    *,
    grid: Sequence[float] = META_RIDGE_ALPHA_GRID,
) -> MetaAlphaSelection:
    values = tuple(float(alpha) for alpha in grid)
    if not values or len(set(values)) != len(values) or any(
        not np.isfinite(alpha) or alpha < 0 for alpha in values
    ):
        raise ValueError("meta alpha grid must contain unique finite non-negative values")
    rows = []
    for alpha in values:
        bank = fit_loco_meta_model_bank(statistics, alpha=alpha)
        simple_error = 0.0
        augmented_error = 0.0
        for index, configuration in enumerate(statistics.configurations):
            simple_error += _sse_for_block(statistics, index, bank.simple[configuration])
            augmented_error += _sse_for_block(
                statistics, index, bank.augmented[configuration]
            )
        mse_simple = simple_error / statistics.total_rows
        mse_augmented = augmented_error / statistics.total_rows
        rows.append(
            {
                "alpha": alpha,
                "mse_simple": mse_simple,
                "mse_augmented": mse_augmented,
                "mean_mse": 0.5 * (mse_simple + mse_augmented),
            }
        )
    table = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    best = float(table["mean_mse"].min())
    alpha = float(
        table.loc[np.isclose(table["mean_mse"], best, rtol=1e-12), "alpha"].min()
    )
    return MetaAlphaSelection(alpha=alpha, table=table)


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

    encoding = infer_meta_encoding(calibration, robust_score=robust_score)
    configurations = tuple(sorted(calibration["configuration"].astype(str).unique()))
    statistics = accumulate_meta_sufficient_statistics(
        (calibration,), encoding=encoding, configurations=configurations
    )
    bank = fit_loco_meta_model_bank(statistics, alpha=alpha)
    predicted = predict_with_loco_meta_bank(bank, test)
    predictions = predicted.sort_values(
        "configuration", kind="stable"
    ).reset_index(drop=True)
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

    _validate_frame(tune, robust_score)
    values = tuple(float(alpha) for alpha in grid)
    if not values or len(set(values)) != len(values) or any(
        not np.isfinite(alpha) or alpha < 0 for alpha in values
    ):
        raise ValueError("meta alpha grid must contain unique finite non-negative values")
    encoding = infer_meta_encoding(tune, robust_score=robust_score)
    configurations = tuple(sorted(tune["configuration"].astype(str).unique()))
    statistics = accumulate_meta_sufficient_statistics(
        (tune,), encoding=encoding, configurations=configurations
    )
    return tune_meta_ridge_alpha_from_sufficient(statistics, grid=values)


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


def _resample_seed_mean_outcomes(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, tuple[object, ...]]]:
    """Resample the fitted-run distribution while preserving the seed-mean estimand.

    A method represented by ``k`` fitted neural seeds contributes the mean of
    ``k`` seeds drawn with replacement from those same fitted runs.  The draw is
    global to the method (not patient- or cell-specific), matching the fact that
    one trained checkpoint is evaluated for every patient.  Deterministic
    methods have ``k=1`` and are therefore unchanged.
    """

    required = {
        "patient_id",
        "method",
        "outcome_log_rmse",
        "prediction_simple",
        "prediction_augmented",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"seed bootstrap frame is missing columns: {sorted(missing)}")
    source = frame.copy()
    if "model_seed" not in source.columns:
        source["model_seed"] = "seed-mean"
    identifiers = [
        *(
            column
            for column in ("cohort", "partition")
            if column in source.columns
        ),
        *SEED_CELL_COLUMNS,
    ]
    if source[identifiers].isna().any().any():
        raise ValueError("seed bootstrap identifiers contain missing values")

    method_frames: list[pd.DataFrame] = []
    selected_draws: dict[str, tuple[object, ...]] = {}
    reference_patients: set[str] | None = None
    for method, rows in source.groupby("method", sort=False, dropna=False):
        method_name = str(method)
        if rows["model_seed"].isna().any():
            if not rows["model_seed"].isna().all():
                raise ValueError(f"{method_name} mixes null and non-null model seeds")
            rows = rows.copy()
            rows["model_seed"] = "deterministic"
        seeds = tuple(rows["model_seed"].drop_duplicates().tolist())
        if not seeds:
            raise ValueError(f"{method_name} has no model seeds")
        cell_keys = [column for column in identifiers if column != "method"]
        seed_counts = rows.groupby(cell_keys, sort=False, dropna=False)["model_seed"].nunique()
        row_counts = rows.groupby(cell_keys, sort=False, dropna=False).size()
        if (
            not np.equal(seed_counts.to_numpy(dtype=np.int64), len(seeds)).all()
            or not np.equal(row_counts.to_numpy(dtype=np.int64), len(seeds)).all()
        ):
            raise ValueError(
                f"{method_name} does not provide each model seed exactly once per scientific cell"
            )
        patients = set(rows["patient_id"].astype(str))
        if reference_patients is None:
            reference_patients = patients
        elif patients != reference_patients:
            raise ValueError("all methods must cover the same patients for a shared cluster draw")

        if len(seeds) > 1:
            raw_draw = rng.choice(np.asarray(seeds, dtype=object), size=len(seeds), replace=True)
            draw = tuple(value.item() if isinstance(value, np.generic) else value for value in raw_draw)
            selected_draws[method_name] = draw
        else:
            draw = seeds
        multiplicity = pd.Series(draw).value_counts(sort=False).to_dict()
        weighted = rows.copy()
        weighted["_seed_weight"] = (
            weighted["model_seed"].map(multiplicity).fillna(0).to_numpy(dtype=float)
            / len(draw)
        )
        weighted["_weighted_outcome"] = (
            weighted["outcome_log_rmse"].to_numpy(dtype=float)
            * weighted["_seed_weight"].to_numpy(dtype=float)
        )
        grouped = weighted.groupby(identifiers, sort=False, dropna=False)
        static = grouped.first().reset_index()
        static["outcome_log_rmse"] = grouped["_weighted_outcome"].sum().to_numpy()
        static["model_seed"] = "bootstrap-seed-mean"
        method_frames.append(static.drop(columns=["_seed_weight", "_weighted_outcome"], errors="ignore"))
    if not method_frames:
        raise ValueError("seed bootstrap frame is empty")
    return pd.concat(method_frames, ignore_index=True), selected_draws


def _bootstrap_rows_with_audit(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, tuple[object, ...]]]:
    """Draw one shared patient bootstrap and a seed-mean draw per neural method."""

    seed_mean, selected_draws = _resample_seed_mean_outcomes(frame, rng)
    patients = tuple(sorted(seed_mean["patient_id"].astype(str).unique()))
    if not patients:
        raise ValueError("patient bootstrap frame is empty")
    sampled_array = rng.choice(np.asarray(patients, dtype=object), size=len(patients), replace=True)
    sampled = tuple(str(value) for value in sampled_array)
    chunks = []
    patient_values = seed_mean["patient_id"].astype(str)
    for bootstrap_id, patient in enumerate(sampled):
        rows = seed_mean[patient_values == patient].copy()
        rows["_bootstrap_patient"] = f"{bootstrap_id}:{patient}"
        chunks.append(rows)
    return pd.concat(chunks, ignore_index=True), sampled, selected_draws


def _bootstrap_rows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    boot, _sampled, _selected_draws = _bootstrap_rows_with_audit(frame, rng)
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
    fitted predictions.  If a neural method has five fitted seeds, each
    replicate draws five seeds with replacement globally for that method and
    reconstructs the cellwise seed mean before applying one patient-cluster draw
    shared by all methods.  Thus the point and interval estimate the same
    five-run mean rather than a random single-run outcome.
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
    required_positive_methods: int = STAGE15_REQUIRED_POSITIVE_METHODS,
    gate_eligible_external_cohorts: Iterable[str] = (
        STAGE15_GATE_ELIGIBLE_EXTERNAL_COHORTS
    ),
) -> Stage15Decision:
    """Apply the preregistered PROCEED/PIVOT rule without discretionary overrides."""

    def validate_effect(effect: BootstrapEffect, *, label: str) -> None:
        if (
            isinstance(effect.point, (bool, np.bool_))
            or not isinstance(effect.ci95, (tuple, list, np.ndarray))
            or len(effect.ci95) != 2
            or any(isinstance(value, (bool, np.bool_)) for value in effect.ci95)
        ):
            raise ValueError(f"{label} Stage-15 effect has an invalid point/interval")
        try:
            values = np.asarray((effect.point, *effect.ci95), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} Stage-15 effect has an invalid point/interval"
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{label} Stage-15 effect must be finite")
        lower, upper = (float(value) for value in effect.ci95)
        if lower > upper:
            raise ValueError(f"{label} Stage-15 confidence interval is reversed")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, abs(lower), abs(float(effect.point)), abs(upper)
        )
        if float(effect.point) < lower - tolerance or float(effect.point) > upper + tolerance:
            raise ValueError(f"{label} Stage-15 point estimate lies outside its interval")
        if (
            isinstance(effect.replicates, bool)
            or not isinstance(effect.replicates, (int, np.integer))
            or int(effect.replicates) < 1
            or isinstance(effect.seed, bool)
            or not isinstance(effect.seed, (int, np.integer))
        ):
            raise ValueError(f"{label} Stage-15 bootstrap metadata is invalid")

    validate_effect(ptbxl, label="PTB-XL")
    normalized_external: dict[str, BootstrapEffect] = {}
    external_labels: dict[str, str] = {}
    for raw_cohort, effect in external.items():
        cohort = str(raw_cohort).strip()
        normalized = cohort.casefold()
        if not normalized or normalized in normalized_external:
            raise ValueError("Stage-15 external cohort names must be non-empty and unique")
        validate_effect(effect, label=cohort)
        normalized_external[normalized] = effect
        external_labels[normalized] = cohort
    if set(method_deltas) != set(STAGE15_COMMON_PANEL_METHODS):
        raise ValueError(
            "Stage-15 method effects must contain exactly the frozen four-method panel"
        )
    if required_positive_methods != STAGE15_REQUIRED_POSITIVE_METHODS:
        raise ValueError("Stage-15 positive-method threshold is frozen at three")
    normalized_method_deltas: dict[str, float] = {}
    for method in STAGE15_COMMON_PANEL_METHODS:
        raw_delta = method_deltas[method]
        if isinstance(raw_delta, (bool, np.bool_)):
            raise ValueError(f"Stage-15 method effect for {method} must be finite")
        delta = float(raw_delta)
        if not np.isfinite(delta):
            raise ValueError(f"Stage-15 method effect for {method} must be finite")
        normalized_method_deltas[method] = delta

    reasons = []
    if ptbxl.ci95[0] <= 0:
        reasons.append("PTB-XL fold-10 delta-R2 lower confidence bound is not positive")
    eligible = tuple(
        sorted({str(cohort).strip().casefold() for cohort in gate_eligible_external_cohorts})
    )
    if not eligible or any(not cohort for cohort in eligible):
        raise ValueError("Stage-15 external gate eligibility must be non-empty")
    qualifying = tuple(
        sorted(
            external_labels[cohort]
            for cohort, effect in normalized_external.items()
            if cohort in eligible and effect.ci95[0] > 0
        )
    )
    if not qualifying:
        reasons.append(
            "no patient-key-eligible external zero-shot cohort has a positive "
            "delta-R2 lower bound"
        )
    positive_methods = sum(delta > 0 for delta in normalized_method_deltas.values())
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
        gate_eligible_external_cohorts=eligible,
        qualifying_external_cohorts=qualifying,
    )
