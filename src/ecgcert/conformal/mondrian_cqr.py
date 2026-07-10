"""Distribution-free calibration: Mondrian CQR and conformal risk control.

Two guarantees are provided, both finite-sample and distribution-free under
exchangeability.

* **Tier II intervals** -- conformalized quantile regression (CQR, Romano et al.
  2019) applied *per group* ``g = (segment, lead)`` (Mondrian / group-conditional
  conformal), so coverage holds conditional on the feature and lead, not merely
  marginally.

* **Tier III flag** -- a threshold ``tau`` on the hallucination energy ``h`` chosen
  so the false-flag rate on faithful reconstructions is ``<= alpha`` (a one-sided
  conformal quantile; the monotone-risk generalisation is :func:`crc_threshold`).

Covariate shift (e.g. PTB-XL calibration -> CinC-2021 test) is handled by
:func:`weighted_conformal_quantile` with likelihood-ratio weights.

All functions are model-agnostic: they consume predicted quantiles / scores as
arrays and never touch a network, so any base reconstructor can be wrapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample ``(1 - alpha)`` conformal quantile of nonconformity scores.

    Returns the ``ceil((n+1)(1-alpha)) / n`` empirical quantile, i.e. the value
    ``Q`` such that a fresh exchangeable score exceeds ``Q`` with probability
    ``<= alpha``.  If the required rank exceeds ``n`` the quantile is ``+inf``
    (coverage cannot be guaranteed with so few calibration points).
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def weighted_conformal_quantile(scores: np.ndarray, weights: np.ndarray,
                                alpha: float) -> float:
    """Weighted conformal quantile for covariate shift (Tibshirani et al. 2019).

    ``weights`` are (unnormalised) likelihood ratios ``w(x) = dP_test/dP_cal`` on
    the calibration points; a point mass ``w_new`` for the test point is appended
    as ``max(weights)`` (worst case) so the guarantee is conservative.  Returns the
    smallest score whose normalised cumulative weight reaches ``1 - alpha``.
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(s)
    s, w = s[order], w[order]
    w_new = w.max() if w.size else 1.0
    total = w.sum() + w_new
    cum = np.cumsum(w) / total
    idx = np.searchsorted(cum, 1.0 - alpha, side="left")
    if idx >= s.size:
        return np.inf
    return float(s[idx])


def cqr_calibrate(y_cal: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                  alpha: float) -> float:
    """CQR nonconformity correction ``Q`` from calibration quantile predictions.

    Score ``E_i = max(q_lo_i - y_i, y_i - q_hi_i)``; returns the conformal quantile
    of ``{E_i}``.  Test interval is ``[q_lo(x) - Q, q_hi(x) + Q]``.
    """
    y, lo, hi = map(lambda a: np.asarray(a, float), (y_cal, q_lo, q_hi))
    scores = np.maximum(lo - y, y - hi)
    return conformal_quantile(scores, alpha)


def cqr_interval(q_lo: np.ndarray, q_hi: np.ndarray, Q: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the CQR correction: ``[q_lo - Q, q_hi + Q]``."""
    return np.asarray(q_lo, float) - Q, np.asarray(q_hi, float) + Q


def empirical_coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """Fraction of targets inside ``[lo, hi]``."""
    lo, hi, y = map(lambda a: np.asarray(a, float), (lo, hi, y))
    return float(np.mean((y >= lo) & (y <= hi)))


@dataclass
class MondrianCQR:
    """Group-conditional (Mondrian) CQR: one conformal correction per group.

    Groups are arbitrary hashable keys (we use ``(segment, lead)``).  Coverage is
    guaranteed conditional on the group, provided each group's calibration points
    are exchangeable with its test points.
    """

    alpha: float
    Q: dict = field(default_factory=dict)
    _fallback: float = np.inf

    def fit(self, groups, y_cal, q_lo, q_hi) -> "MondrianCQR":
        groups = np.asarray(groups)
        y, lo, hi = map(lambda a: np.asarray(a, float), (y_cal, q_lo, q_hi))
        scores = np.maximum(lo - y, y - hi)
        self.Q = {}
        for g in np.unique(groups):
            m = groups == g
            self.Q[g if not isinstance(g, np.generic) else g.item()] = conformal_quantile(scores[m], self.alpha)
        # Marginal fallback for unseen groups.
        self._fallback = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, groups, q_lo, q_hi) -> tuple[np.ndarray, np.ndarray]:
        groups = np.asarray(groups)
        lo, hi = np.asarray(q_lo, float), np.asarray(q_hi, float)
        Qv = np.array([self.Q.get(g.item() if isinstance(g, np.generic) else g, self._fallback)
                       for g in groups])
        return lo - Qv, hi + Qv


def crc_threshold(losses_by_threshold, thresholds: np.ndarray, alpha: float,
                  b: float = 1.0) -> float:
    """Conformal Risk Control (Angelopoulos et al. 2024) for a monotone loss.

    ``losses_by_threshold`` is an ``(n_cal, n_thresh)`` array of per-example losses
    evaluated at each candidate threshold in ``thresholds`` (assumed to give a
    *non-increasing* empirical risk as the threshold increases -- e.g. a 0/1
    false-flag loss ``1{h > tau}`` as ``tau`` grows).  Returns the smallest
    threshold whose CRC-corrected risk ``(n*Rhat + b) / (n + 1) <= alpha``.
    ``b`` is the loss upper bound.
    """
    L = np.asarray(losses_by_threshold, float)
    n = L.shape[0]
    risks = L.mean(axis=0)
    corrected = (n * risks + b) / (n + 1)
    ok = np.where(corrected <= alpha)[0]
    if ok.size == 0:
        return float(thresholds[-1])
    return float(thresholds[ok[0]])


def flag_threshold(h_faithful: np.ndarray, alpha: float) -> float:
    """Distribution-free flag threshold with false-flag rate ``<= alpha``.

    On faithful reconstructions the hallucination energy ``h`` should be small;
    ``tau`` is its one-sided ``(1 - alpha)`` conformal quantile, so a fresh
    faithful example is flagged with probability ``<= alpha``.
    """
    return conformal_quantile(np.asarray(h_faithful, float), alpha)
