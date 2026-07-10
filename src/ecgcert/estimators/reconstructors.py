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
from ecgcert.physics.dipolar_subspace import LEAD_INDEX


def prior_mean_reconstructor(mu_s: np.ndarray, T: int) -> np.ndarray:
    """Constant reconstruction at the population mean -> (12, T)."""
    return np.repeat(mu_s[:, None], T, axis=1)


class LinearDipolarReconstructor:
    """Tier I reconstructor: the recovered population-dipolar projection."""

    def __init__(self, M_s: np.ndarray, mu_s: np.ndarray, observed_leads, rcond=None):
        from ecgcert.physics.dipolar_subspace import RECON_RCOND, reconstruct_dipolar

        rcond = RECON_RCOND if rcond is None else rcond

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


class OLSReconstructor:
    """Learned linear reconstructor: least-squares map ``y_S -> L`` from data.

    This is the population-optimal *linear* reconstruction.  Unlike the pure
    dipolar reconstructor it also picks up the population-correlated non-dipolar
    content (Tier II), but being MSE-optimal it regresses Tier III content to the
    mean -- it does not fabricate (its hallucination energy stays low; it is
    "honestly incomplete", blurring rather than inventing).  A standard baseline.
    """

    def __init__(self, observed_leads):
        self.observed = observed_leads
        self.idx = [LEAD_INDEX[l] if isinstance(l, str) else int(l) for l in observed_leads]
        self.W = None
        self.b = None

    def fit(self, L_train: np.ndarray) -> "OLSReconstructor":
        """``L_train`` is (12, N) training samples (per-time-sample, any segments)."""
        X = L_train[self.idx].T                          # (N, |S|)
        Y = L_train.T                                    # (N, 12)
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])    # bias
        coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)    # (|S|+1, 12)
        self.W = coef[:-1].T                             # (12, |S|)
        self.b = coef[-1]                                # (12,)
        return self

    def predict(self, y_S: np.ndarray) -> np.ndarray:
        y = np.asarray(y_S, float)
        return self.W @ y + self.b[:, None]


class GenerativeSampleReconstructor:
    """Perceptual/generative baseline that *samples* non-dipolar content.

    Mimics what a generative reconstructor (diffusion / GAN) does to look
    realistic: start from the dipolar reconstruction and add a plausible sample of
    non-dipolar texture drawn from the population non-dipolar covariance.  Because
    that sample is independent of the observation on the Tier III subspace, the
    added content is *fabricated* -- realistic-looking but uncorrelated with the
    truth.  This is the hallucination exhibit; the certificate flags exactly its
    Tier III energy.
    """

    def __init__(self, M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                 Sigma_r: np.ndarray, scale: float = 1.0, seed: int = 0):
        from ecgcert.physics.dipolar_subspace import reconstruct_dipolar

        self.M_s, self.mu_s, self.observed = M_s, mu_s, observed_leads
        self._recon = reconstruct_dipolar
        # Non-dipolar sampling covariance (project Sigma_r off the dipole subspace).
        U = np.eye(12) - M_s @ M_s.T
        C = U @ Sigma_r @ U.T
        C = 0.5 * (C + C.T)
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0, None)
        self._chol = vecs @ np.diag(np.sqrt(vals))       # (12, 12) sampler
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def predict(self, y_S: np.ndarray) -> np.ndarray:
        dip = self._recon(self.M_s, self.mu_s, self.observed, y_S)
        T = dip.shape[1] if dip.ndim == 2 else 1
        noise = self._chol @ self.rng.standard_normal((12, T))
        return dip + self.scale * noise
