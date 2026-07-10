"""Reduced-lead reconstructors used as reference estimators and baselines.

* :class:`LinearDipolarReconstructor` -- the honest linear baseline: recover the
  population-dipolar projection ``L_hat = mu + M_s M_{s,S}^+ (y_S - mu_S)``.  This
  *is* Tier I; it never fabricates non-dipolar content (its hallucination energy is
  zero by construction), so it is the natural floor for the certificate.

* :class:`BayesianDipolarReconstructor` -- the MSE-optimal linear estimator under a
  Gaussian model ``x = M_s d + r``, ``d ~ N(0, Sigma_d)``, ``r ~ N(0, Sigma_r)``,
  ``y_S = Sel_S x + n``, ``n ~ N(0, sigma^2 I)``.  Its posterior mean is the best
  possible reconstruction; on the *observation-independent* part of ``r`` it
  provably returns the prior mean, realising the ``Var(u)`` non-identifiability
  lower bound.  This is the estimator that makes the synthetic experiment's
  irreducibility figure exact.

* :func:`prior_mean_reconstructor` -- returns the population mean (the trivial
  baseline; upper bound on error for unrecoverable content).
"""
from __future__ import annotations

import numpy as np

from ecgcert.certify.tier_decomposition import selection_matrix


def prior_mean_reconstructor(mu_s: np.ndarray, T: int) -> np.ndarray:
    """Constant reconstruction at the population mean -> (12, T)."""
    return np.repeat(mu_s[:, None], T, axis=1)


class LinearDipolarReconstructor:
    """Tier I reconstructor: the recovered population-dipolar projection."""

    def __init__(self, M_s: np.ndarray, mu_s: np.ndarray, observed_leads, rcond: float = 1e-10):
        from ecgcert.physics.dipolar_subspace import reconstruct_dipolar

        self.M_s, self.mu_s, self.observed = M_s, mu_s, observed_leads
        self._recon = reconstruct_dipolar
        self.rcond = rcond

    def predict(self, y_S: np.ndarray) -> np.ndarray:
        """``y_S`` is ``(|S|, T)`` -> full ``(12, T)`` reconstruction."""
        return self._recon(self.M_s, self.mu_s, self.observed, y_S, rcond=self.rcond)


class BayesianDipolarReconstructor:
    """MSE-optimal linear (posterior-mean) reconstructor under a Gaussian model.

    Model in 12-lead space:  ``L = mu + M_s d + r``, with ``d ~ N(0, Sigma_d)``
    (dipole, 3-dim) and ``r ~ N(0, Sigma_r)`` (non-dipolar residual, in the
    orthogonal complement of ``M_s``).  Observation ``y_S = Sel_S L + n``,
    ``n ~ N(0, sigma^2 I)``.  Returns the posterior mean E[L | y_S], the best
    possible reconstruction in mean-squared error.
    """

    def __init__(self, M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                 Sigma_d: np.ndarray, Sigma_r: np.ndarray, sigma: float):
        self.M_s = M_s
        self.mu_s = mu_s
        self.Sel = selection_matrix(observed_leads)          # (|S|, 12)
        # Prior covariance of L: Cov = M Sigma_d M^T + Sigma_r  (Sigma_r lives in M^perp).
        self.Cov = M_s @ Sigma_d @ M_s.T + Sigma_r           # (12, 12)
        self.sigma2 = float(sigma) ** 2

    def predict(self, y_S: np.ndarray) -> np.ndarray:
        y = np.asarray(y_S, float)
        S = self.Sel
        Cyy = S @ self.Cov @ S.T + self.sigma2 * np.eye(S.shape[0])   # (|S|,|S|)
        Cxy = self.Cov @ S.T                                          # (12,|S|)
        gain = Cxy @ np.linalg.inv(Cyy)                               # (12,|S|)
        resid = y - S @ self.mu_s[:, None]
        return self.mu_s[:, None] + gain @ resid                     # (12, T)
