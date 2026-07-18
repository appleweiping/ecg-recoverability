"""Rank-generic empirical spatial-subspace models for ECG recoverability.

The legacy project used a privileged rank-3 ``DipolarModel``.  This module keeps
that API intact while providing an explicit model object for rank paths and basis
uncertainty.  Two basis definitions are supported:

``raw12_pca``
    PCA directly in the twelve recorded channels.  Small acquisition/quantisation
    violations of the ideal limb-lead algebra remain part of the fitted space.

``independent8_lifted``
    PCA in ``[I, II, V1, ..., V6]`` followed by the fixed twelve-lead transform.
    This variant enforces the ideal Einthoven/Goldberger relations by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ecgcert.physics.dipolar_subspace import (
    INDEPENDENT_LEADS,
    LEAD_INDEX,
    LEADS,
    fit_dipolar_subspace,
    lead_transform_T,
)

BasisVariant = Literal["raw12_pca", "independent8_lifted"]
BASIS_VARIANTS: tuple[BasisVariant, ...] = ("raw12_pca", "independent8_lifted")


@dataclass(frozen=True)
class SpatialSubspaceModel:
    """One fitted empirical spatial subspace.

    ``covariance`` is the covariance of coordinates ``(L - mu) @ M`` and therefore
    has shape ``(rank, rank)``.  ``fit_ids`` identifies the source records; repeated
    ids are allowed for record-cluster bootstrap fits.
    """

    rank: int
    basis_variant: BasisVariant
    fit_cohort: str
    fit_ids: tuple[int, ...]
    M: np.ndarray
    mu: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive integer; got {self.rank!r}")
        if self.basis_variant not in BASIS_VARIANTS:
            raise ValueError(
                f"basis_variant must be one of {BASIS_VARIANTS}; got {self.basis_variant!r}"
            )
        if not self.fit_cohort:
            raise ValueError("fit_cohort must be a non-empty label")

        M = np.array(self.M, dtype=float, copy=True)
        mu = np.array(self.mu, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if M.shape != (len(LEADS), self.rank):
            raise ValueError(f"M must be ({len(LEADS)}, {self.rank}); got {M.shape}")
        if mu.shape != (len(LEADS),):
            raise ValueError(f"mu must be ({len(LEADS)},); got {mu.shape}")
        if covariance.shape != (self.rank, self.rank):
            raise ValueError(
                f"covariance must be ({self.rank}, {self.rank}); got {covariance.shape}"
            )
        if not all(np.all(np.isfinite(a)) for a in (M, mu, covariance)):
            raise ValueError("M, mu, and covariance must contain only finite values")
        if not np.allclose(M.T @ M, np.eye(self.rank), atol=1e-8, rtol=1e-8):
            raise ValueError("columns of M must be orthonormal")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
            raise ValueError("covariance must be symmetric")

        fit_ids = tuple(int(i) for i in self.fit_ids)
        M.setflags(write=False)
        mu.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "fit_ids", fit_ids)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "covariance", covariance)


def _validate_samples(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(LEADS):
        raise ValueError(f"X must be (n_samples, {len(LEADS)}); got {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("at least two samples are required to fit a spatial subspace")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def _coordinate_covariance(X: np.ndarray, M: np.ndarray, mu: np.ndarray) -> np.ndarray:
    coordinates = (X - mu) @ M
    covariance = np.atleast_2d(np.cov(coordinates, rowvar=False))
    return 0.5 * (covariance + covariance.T)


def fit_spatial_subspace(
    X: np.ndarray,
    *,
    rank: int,
    basis_variant: BasisVariant = "raw12_pca",
    fit_cohort: str = "unspecified",
    fit_ids: Sequence[int] = (),
) -> SpatialSubspaceModel:
    """Fit one rank-generic spatial model from twelve-lead samples."""

    X = _validate_samples(X)
    if basis_variant not in BASIS_VARIANTS:
        raise ValueError(f"basis_variant must be one of {BASIS_VARIANTS}; got {basis_variant!r}")
    max_rank = len(INDEPENDENT_LEADS) if basis_variant == "independent8_lifted" else len(LEADS)
    if not isinstance(rank, int) or not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}] for {basis_variant}; got {rank!r}")
    # Centering removes one sample degree of freedom.  Refuse to return arbitrary
    # null-space vectors when the requested rank exceeds the estimable dimension.
    if rank > min(X.shape[0] - 1, max_rank):
        raise ValueError(f"rank {rank} exceeds the centred sample dimension")

    if basis_variant == "raw12_pca":
        M, mu, _ = fit_dipolar_subspace(X, rank=rank)
        modeled_X = X
    else:
        independent_idx = [LEAD_INDEX[lead] for lead in INDEPENDENT_LEADS]
        X8 = X[:, independent_idx]
        mu8 = X8.mean(axis=0)
        U8, _, _ = np.linalg.svd((X8 - mu8).T, full_matrices=False)
        transform = lead_transform_T()
        lifted = transform @ U8[:, :rank]
        M, _ = np.linalg.qr(lifted, mode="reduced")
        mu = transform @ mu8
        modeled_X = X8 @ transform.T

    covariance = _coordinate_covariance(modeled_X, M, mu)
    return SpatialSubspaceModel(
        rank=rank,
        basis_variant=basis_variant,
        fit_cohort=fit_cohort,
        fit_ids=tuple(int(i) for i in fit_ids),
        M=M,
        mu=mu,
        covariance=covariance,
    )


__all__ = [
    "BASIS_VARIANTS",
    "BasisVariant",
    "SpatialSubspaceModel",
    "fit_spatial_subspace",
]
