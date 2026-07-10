"""Fit per-segment dipolar models (M_s, mu_s, Sigma_r) from a training population.

These are the population-level objects the certificate needs:

* ``M_s`` -- per-segment dipolar basis (12x3),
* ``mu_s`` -- per-segment population mean (12,),
* ``Sigma_r`` -- per-segment residual covariance (12x12), used by the Bayesian and
  generative reconstructors and by the Tier II prior.

Fitting pools per-segment 12-lead sample vectors across many records and runs the
segment SVD (:func:`ecgcert.physics.fit_dipolar_subspace`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgcert.physics import fit_dipolar_subspace


@dataclass
class SegmentModel:
    M: np.ndarray        # (12, 3) dipolar basis
    mu: np.ndarray       # (12,) mean
    Sigma_r: np.ndarray  # (12, 12) residual covariance
    evr: np.ndarray      # variance-explained ratios


def fit_segment_models(seg_samples: dict[str, np.ndarray], rank: int = 3) -> dict[str, SegmentModel]:
    """Fit ``{segment: SegmentModel}`` from ``{segment: (N, 12) samples}``."""
    out: dict[str, SegmentModel] = {}
    for seg, X in seg_samples.items():
        if X.shape[0] < 50:
            continue
        M_s, mu_s, evr = fit_dipolar_subspace(X, rank=rank)
        # Residual (non-dipolar) covariance of the centred data off the dipole.
        Xc = X - mu_s
        R = Xc - Xc @ M_s @ M_s.T          # (N, 12) non-dipolar residual
        Sigma_r = np.cov(R.T)              # (12, 12)
        out[seg] = SegmentModel(M=M_s, mu=mu_s, Sigma_r=Sigma_r, evr=evr)
    return out
