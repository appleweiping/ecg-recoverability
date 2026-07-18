"""Main-paper low-rank and ridge reconstructors using the uniform API."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.api import (
    Reconstructor,
    ReconstructorConfig,
    TrainManifest,
    load_manifest_signals,
)


def _observed_indices(leads) -> np.ndarray:
    return np.asarray([CANONICAL_LEADS.index(lead) for lead in leads], dtype=int)


def _save_npz_exact(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class RidgeLeadReconstructor(Reconstructor):
    method_id = "ridge"

    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig):
        config.validate()
        signals = np.asarray(load_manifest_signals(train_manifest), dtype=float)
        observed = _observed_indices(config.observed_leads)
        samples = np.transpose(signals, (0, 2, 1)).reshape(-1, 12)
        x = samples[:, observed]
        y = samples
        self.x_mean = x.mean(axis=0)
        self.y_mean = y.mean(axis=0)
        xc = x - self.x_mean
        yc = y - self.y_mean
        ridge_lambda = float(config.parameters.get("ridge_lambda", 1e-3))
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        gram = xc.T @ xc + ridge_lambda * np.eye(observed.size)
        self.weights = np.linalg.solve(gram, xc.T @ yc).T
        self.observed = observed
        self._checkpoint_path = Path(config.output_dir) / "ridge.npz"
        _save_npz_exact(
            self._checkpoint_path,
            weights=self.weights,
            x_mean=self.x_mean,
            y_mean=self.y_mean,
            observed=self.observed,
            ridge_lambda=np.asarray([ridge_lambda]),
        )
        self._fitted = True
        return self

    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        if not np.all(observed_mask == observed_mask[:, :1]):
            raise ValueError("ridge lead reconstruction requires whole-lead masks")
        observed = np.flatnonzero(observed_mask[:, 0])
        if not np.array_equal(observed, self.observed):
            raise ValueError("evaluation mask differs from the fitted ridge configuration")
        x = signal[self.observed].T
        return ((x - self.x_mean) @ self.weights.T + self.y_mean).T


class LowRankConditionalMeanReconstructor(Reconstructor):
    method_id = "low_rank_conditional_mean"

    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig):
        config.validate()
        signals = np.asarray(load_manifest_signals(train_manifest), dtype=float)
        samples = np.transpose(signals, (0, 2, 1)).reshape(-1, 12)
        self.mean = samples.mean(axis=0)
        centered = samples - self.mean
        rank = int(config.parameters.get("rank", 3))
        if rank < 1 or rank > 12:
            raise ValueError("rank must lie in [1,12]")
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:rank].T
        denom = max(samples.shape[0] - 1, 1)
        coord_variance = (singular[:rank] ** 2) / denom
        covariance = (basis * coord_variance[None, :]) @ basis.T
        shrinkage = float(config.parameters.get("noise_variance", 1e-6))
        if shrinkage <= 0:
            raise ValueError("noise_variance must be positive")
        covariance = covariance + shrinkage * np.eye(12)
        self.covariance = covariance
        self.observed = _observed_indices(config.observed_leads)
        self._checkpoint_path = Path(config.output_dir) / "low_rank.npz"
        _save_npz_exact(
            self._checkpoint_path,
            mean=self.mean,
            covariance=self.covariance,
            observed=self.observed,
            rank=np.asarray([rank]),
            noise_variance=np.asarray([shrinkage]),
        )
        self._fitted = True
        return self

    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        if not np.all(observed_mask == observed_mask[:, :1]):
            raise ValueError("low-rank lead reconstruction requires whole-lead masks")
        observed = np.flatnonzero(observed_mask[:, 0])
        if not np.array_equal(observed, self.observed):
            raise ValueError("evaluation mask differs from the fitted low-rank configuration")
        c_oo = self.covariance[np.ix_(observed, observed)]
        gain = self.covariance[:, observed] @ np.linalg.pinv(c_oo)
        residual = signal[observed] - self.mean[observed, None]
        return self.mean[:, None] + gain @ residual
