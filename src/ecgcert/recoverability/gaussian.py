"""Gaussian-prior, model-conditional ECG recoverability scores.

The score in this module is a predictive uncertainty under a fitted spatial
subspace model.  It is deliberately *not* called a certificate: it depends on
the empirical Gaussian coordinate prior and on one validation-frozen observation
regularizer.  All returned standard deviations are in the signal unit used to fit
the model (mV in the locked protocol).
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecgcert.physics import LEADS, LEAD_INDEX, SpatialSubspaceModel


# Observation-noise variances in mV^2.  The grid is protocol, not data, and must
# be frozen before fold-8 outcomes are inspected.
REGULARIZATION_GRID_MV2: tuple[float, ...] = tuple(10.0 ** exponent for exponent in range(-8, 0))


def _observed_indices(observed_leads: Sequence[str | int]) -> np.ndarray:
    if not observed_leads:
        raise ValueError("at least one observed lead is required")
    indices: list[int] = []
    for lead in observed_leads:
        if isinstance(lead, str):
            if lead not in LEAD_INDEX:
                raise ValueError(f"unknown ECG lead {lead!r}")
            index = LEAD_INDEX[lead]
        elif isinstance(lead, (int, np.integer)):
            index = int(lead)
            if not 0 <= index < len(LEADS):
                raise IndexError(index)
        else:
            raise TypeError(f"lead must be a name or integer index; got {lead!r}")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("observed leads must be unique")
    return np.asarray(indices, dtype=int)


def gaussian_posterior_covariance(
    model: SpatialSubspaceModel,
    observed_leads: Sequence[str | int],
    *,
    observation_variance_mv2: float,
) -> np.ndarray:
    """Posterior coordinate covariance after observing a lead subset.

    For coordinates ``d ~ N(0, Sigma)`` and centred observations
    ``y_S = M_S d + epsilon``, ``epsilon ~ N(0, lambda I)``, this returns

    ``Sigma - Sigma M_S.T (M_S Sigma M_S.T + lambda I)^+ M_S Sigma``.

    ``lambda`` is the single fold-8-frozen regularizer.  A tiny eigenvalue clip
    removes only floating-point negative variance, never an empirical direction.
    """

    variance = float(observation_variance_mv2)
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("observation_variance_mv2 must be finite and non-negative")
    indices = _observed_indices(observed_leads)
    M_observed = model.M[indices]
    covariance = np.asarray(model.covariance, dtype=float)
    innovation = M_observed @ covariance @ M_observed.T
    innovation = 0.5 * (innovation + innovation.T)
    if variance:
        innovation = innovation + variance * np.eye(len(indices))
        gain_right = np.linalg.solve(innovation, M_observed @ covariance)
    else:
        gain_right = np.linalg.pinv(innovation, rcond=1e-12, hermitian=True) @ (
            M_observed @ covariance
        )
    posterior = covariance - covariance @ M_observed.T @ gain_right
    posterior = 0.5 * (posterior + posterior.T)
    eigenvalues, eigenvectors = np.linalg.eigh(posterior)
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(covariance, ord=2)))
    if float(eigenvalues.min()) < -tolerance:
        raise FloatingPointError("Gaussian posterior covariance is not positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def gaussian_prior_ambiguity_per_lead(
    model: SpatialSubspaceModel,
    observed_leads: Sequence[str | int],
    *,
    observation_variance_mv2: float,
) -> np.ndarray:
    """Model-conditional posterior standard deviation for all twelve leads, in mV."""

    posterior = gaussian_posterior_covariance(
        model,
        observed_leads,
        observation_variance_mv2=observation_variance_mv2,
    )
    lead_covariance = model.M @ posterior @ model.M.T
    return np.sqrt(np.clip(np.diag(lead_covariance), 0.0, None))


def gaussian_conditional_mean(
    model: SpatialSubspaceModel,
    signal: np.ndarray,
    observed_leads: Sequence[str | int],
    *,
    observation_variance_mv2: float,
) -> np.ndarray:
    """Conditional-mean twelve-lead reconstruction for ``(samples, 12)`` input.

    The input is used only at observed columns.  Observed columns are copied back
    exactly so that this helper obeys the shared reconstruction contract.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LEADS):
        raise ValueError(f"signal must be (n_samples, {len(LEADS)}); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    indices = _observed_indices(observed_leads)
    variance = float(observation_variance_mv2)
    posterior = gaussian_posterior_covariance(
        model,
        observed_leads,
        observation_variance_mv2=variance,
    )
    # Sigma_post = Sigma - K A Sigma, so K is formed separately for the mean.
    del posterior
    M_observed = model.M[indices]
    covariance = np.asarray(model.covariance, dtype=float)
    innovation = M_observed @ covariance @ M_observed.T
    innovation = 0.5 * (innovation + innovation.T)
    if variance:
        innovation = innovation + variance * np.eye(len(indices))
        gain = np.linalg.solve(innovation, M_observed @ covariance).T
    else:
        gain = (
            np.linalg.pinv(innovation, rcond=1e-12, hermitian=True)
            @ (M_observed @ covariance)
        ).T
    centred = values[:, indices] - model.mu[indices]
    coordinates = centred @ gain.T
    reconstructed = model.mu + coordinates @ model.M.T
    reconstructed[:, indices] = values[:, indices]
    return reconstructed


@dataclass(frozen=True)
class RegularizationSelection:
    """Fold-8 tuning result for the single observation regularizer."""

    selected_variance_mv2: float
    table: pd.DataFrame
    criterion: str = "mean patient-balanced missing-target MSE"


def _patient_balanced_mse(
    squared_error: np.ndarray,
    patient_ids: Sequence[Hashable],
) -> float:
    if squared_error.ndim != 2 or squared_error.shape[1] < 1:
        raise ValueError("squared_error must contain at least one missing target")
    if len(patient_ids) != squared_error.shape[0]:
        raise ValueError("patient_ids must align with squared_error rows")
    per_row = squared_error.mean(axis=1)
    totals: dict[Hashable, list[float]] = {}
    for patient_id, error in zip(patient_ids, per_row):
        try:
            hash(patient_id)
        except TypeError as exc:
            raise ValueError(f"patient id must be hashable; got {patient_id!r}") from exc
        aggregate = totals.setdefault(patient_id, [0.0, 0.0])
        aggregate[0] += float(error)
        aggregate[1] += 1.0
    return float(np.mean([total / count for total, count in totals.values()]))


def tune_gaussian_regularization(
    models_by_segment: Mapping[str, Sequence[SpatialSubspaceModel]],
    validation_by_segment: Mapping[str, tuple[np.ndarray, Sequence[Hashable]]],
    configurations: Sequence[Sequence[str]],
    *,
    grid_mv2: Sequence[float] = REGULARIZATION_GRID_MV2,
) -> RegularizationSelection:
    """Freeze one regularizer on fold 8 using patient-balanced reconstruction MSE.

    Every segment/rank/configuration cell receives equal weight; patients receive
    equal weight within a cell.  Correlated configuration-target rows are therefore
    not treated as independent observations during tuning.
    """

    grid = tuple(float(value) for value in grid_mv2)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("grid_mv2 must be a non-empty sequence of unique values")
    if any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("the preregistered regularization grid must be finite and positive")
    if not configurations:
        raise ValueError("at least one validation configuration is required")

    rows: list[dict[str, float | int]] = []
    for variance in grid:
        cell_losses: list[float] = []
        for segment in sorted(models_by_segment):
            if segment not in validation_by_segment:
                raise ValueError(f"missing validation samples for segment {segment}")
            X, patient_ids = validation_by_segment[segment]
            X = np.asarray(X, dtype=float)
            if X.ndim != 2 or X.shape[1] != len(LEADS) or X.shape[0] < 2:
                raise ValueError(f"invalid validation matrix for segment {segment}: {X.shape}")
            for model in models_by_segment[segment]:
                for configuration in configurations:
                    observed = _observed_indices(tuple(configuration))
                    missing = np.asarray(
                        [index for index in range(len(LEADS)) if index not in set(observed)],
                        dtype=int,
                    )
                    if not missing.size:
                        continue
                    reconstructed = gaussian_conditional_mean(
                        model,
                        X,
                        tuple(configuration),
                        observation_variance_mv2=variance,
                    )
                    error = (reconstructed[:, missing] - X[:, missing]) ** 2
                    cell_losses.append(_patient_balanced_mse(error, patient_ids))
        if not cell_losses:
            raise ValueError("regularization tuning produced no missing-target cells")
        rows.append(
            {
                "observation_variance_mv2": variance,
                "patient_balanced_mse": float(np.mean(cell_losses)),
                "n_segment_rank_configuration_cells": len(cell_losses),
            }
        )

    table = pd.DataFrame(rows).sort_values("observation_variance_mv2").reset_index(drop=True)
    best_loss = float(table["patient_balanced_mse"].min())
    selected = float(
        table.loc[
            np.isclose(table["patient_balanced_mse"], best_loss, rtol=1e-12, atol=0.0),
            "observation_variance_mv2",
        ].min()
    )
    return RegularizationSelection(selected_variance_mv2=selected, table=table)


__all__ = [
    "REGULARIZATION_GRID_MV2",
    "RegularizationSelection",
    "gaussian_conditional_mean",
    "gaussian_posterior_covariance",
    "gaussian_prior_ambiguity_per_lead",
    "tune_gaussian_regularization",
]
