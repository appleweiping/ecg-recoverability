"""The ECG dipolar subspace and the closed-form recoverability certificate.

This module encodes two *exact* facts and one *estimated* object.

Exact (deterministic) facts
---------------------------
1. The standard 12-lead ECG has algebraic rank 8.  The six limb leads are fixed
   linear combinations of two independent limb leads (I, II) via Einthoven's law
   and the Goldberger relations::

       III = II - I
       aVR = -(I + II) / 2
       aVL =  I - II / 2
       aVF =  II - I / 2

   so the eight independent leads are ``[I, II, V1, V2, V3, V4, V5, V6]`` and
   ``L = T @ x`` with a fixed ``T in R^{12x8}`` (:func:`lead_transform_T`).

Estimated (physical) object
---------------------------
2. Within the 8 independent leads, the instantaneous potential is *approximately*
   a rank-3 cardiac **dipole** plus a **non-dipolar residual**.  Per waveform
   segment ``s in {P, QRS, ST, T}`` we estimate the dipolar subspace
   ``M_s in R^{12x3}`` as the top-3 left singular vectors of the population,
   segment-``s`` lead covariance.  We never assume exact rank 3; ``M_s`` is a
   *coordinate system* and everything non-dipolar is pushed to the calibration
   residual (Tier II / Tier III).

The certificate
---------------
For an observed lead subset ``S`` with dipolar sub-matrix ``M_{s,S}`` (rows ``S``
of ``M_s``):

* if ``rank(M_{s,S}) = 3`` the population-dipolar projection of **every** lead is
  recovered exactly by ``L_hat = M_s M_{s,S}^+ y_S`` with error ``M_s M_{s,S}^+ n``
  and closed-form noise-amplification constant ``kappa_s(S) = ||M_s M_{s,S}^+||_2``;
* if ``rank(M_{s,S}) < 3`` the unobserved dipolar directions are unrecoverable.

``kappa_s(S)`` is the object that distinguishes a *dipole-spanning* 3-lead set
(low ``kappa``) from a near-coplanar limb triplet (rank-deficient / huge ``kappa``)
-- the "geometry, not lead count" story.  Generic inverse-problem theory cannot
produce it because it has no ``M``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Truncation tolerance for the pseudo-inverse in dipolar recovery.  Dipole
# directions whose observed singular value falls below RECON_RCOND * (largest)
# are treated as effectively unobserved (Tier III) instead of being inverted --
# this keeps recovery numerically stable on ill-conditioned configurations
# (e.g. limb-only leads reconstructing precordials, where kappa ~ 1e4-1e5).
RECON_RCOND: float = 1e-2

# Standard clinical 12-lead order.
LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
LEAD_INDEX: dict[str, int] = {name: i for i, name in enumerate(LEADS)}

# The 8 algebraically independent leads.
INDEPENDENT_LEADS: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

WAVE_SEGMENTS: tuple[str, ...] = ("P", "QRS", "ST", "T")


def lead_transform_T() -> np.ndarray:
    """Return the fixed matrix ``T in R^{12x8}`` with ``L = T @ x``.

    ``x`` is the independent-lead vector ordered as :data:`INDEPENDENT_LEADS`
    ``= [I, II, V1..V6]``.  The limb-lead rows encode Einthoven + Goldberger.
    """
    ind = {name: j for j, name in enumerate(INDEPENDENT_LEADS)}
    T = np.zeros((12, 8), dtype=float)
    # Independent leads map to themselves.
    for name in INDEPENDENT_LEADS:
        T[LEAD_INDEX[name], ind[name]] = 1.0
    # Dependent limb leads (functions of I, II only).
    i, ii = ind["I"], ind["II"]
    T[LEAD_INDEX["III"], [i, ii]] = [-1.0, 1.0]          # III = II - I
    T[LEAD_INDEX["aVR"], [i, ii]] = [-0.5, -0.5]         # aVR = -(I+II)/2
    T[LEAD_INDEX["aVL"], [i, ii]] = [1.0, -0.5]          # aVL = I - II/2
    T[LEAD_INDEX["aVF"], [i, ii]] = [-0.5, 1.0]          # aVF = II - I/2
    return T


def check_lead_algebra(tol: float = 1e-12) -> bool:
    """Verify the encoded lead algebra is internally consistent (rank 8, relations)."""
    T = lead_transform_T()
    if np.linalg.matrix_rank(T, tol=tol) != 8:
        return False
    # Random independent-lead vectors must satisfy the four dependent relations.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 100))
    L = T @ x
    idx = LEAD_INDEX
    ok = True
    ok &= np.allclose(L[idx["III"]], L[idx["II"]] - L[idx["I"]], atol=tol)
    ok &= np.allclose(L[idx["aVR"]], -(L[idx["I"]] + L[idx["II"]]) / 2, atol=tol)
    ok &= np.allclose(L[idx["aVL"]], L[idx["I"]] - L[idx["II"]] / 2, atol=tol)
    ok &= np.allclose(L[idx["aVF"]], L[idx["II"]] - L[idx["I"]] / 2, atol=tol)
    return bool(ok)


def fit_dipolar_subspace(
    X_seg: np.ndarray, rank: int = 3, center: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the segment dipolar subspace from population 12-lead samples.

    Parameters
    ----------
    X_seg : (N, 12) array
        Population lead vectors sampled within a waveform segment (rows = samples,
        columns = the 12 standard leads).
    rank : int
        Dipolar rank (3 by default -- the cardiac dipole).
    center : bool
        Subtract the population segment mean before the SVD (recommended).

    Returns
    -------
    M_s : (12, rank) array
        Orthonormal dipolar basis (top-``rank`` left singular vectors).
    mu_s : (12,) array
        Population segment mean (zeros if ``center=False``).
    evr : (12,) array
        Variance-explained ratio of *all* singular directions (for the risk-2
        dipolarity check).  ``evr[:rank].sum()`` is the dipolar fraction.
    """
    X = np.asarray(X_seg, dtype=float)
    if X.ndim != 2 or X.shape[1] != 12:
        raise ValueError(f"X_seg must be (N, 12); got {X.shape}")
    mu_s = X.mean(axis=0) if center else np.zeros(12)
    Xc = X - mu_s
    # Economy SVD of the centred data; columns of U are lead-space directions.
    U, sv, _ = np.linalg.svd(Xc.T, full_matrices=False)  # U: (12, r), sv: (r,)
    var = sv**2
    evr = var / var.sum() if var.sum() > 0 else var
    M_s = U[:, :rank].copy()
    return M_s, mu_s, evr


def dipolar_projector(M_s: np.ndarray) -> np.ndarray:
    """Orthogonal projector ``Pi_{D_s} = M_s M_s^T`` onto the dipolar subspace."""
    return M_s @ M_s.T


def _observed_idx(observed_leads) -> np.ndarray:
    return np.array([LEAD_INDEX[l] if isinstance(l, str) else int(l) for l in observed_leads])


def kappa(M_s: np.ndarray, observed_leads, rcond: float = 1e-10) -> tuple[float, int]:
    """Closed-form noise-amplification constant ``kappa_s(S)`` and dipole rank.

    Returns ``(kappa, r)`` where ``kappa = ||M_s M_{s,S}^+||_2`` (spectral norm)
    and ``r = rank(M_{s,S})``.  If ``r < 3`` the observed leads do not span the
    dipole and ``kappa`` is reported for the pseudo-inverse on the observed
    directions (finite, but the missing directions are unrecoverable -- Tier III).
    """
    idx = _observed_idx(observed_leads)
    M_S = M_s[idx]                       # (|S|, 3)
    r = int(np.linalg.matrix_rank(M_S, tol=rcond * max(M_S.shape)))
    G = M_s @ np.linalg.pinv(M_S, rcond=rcond)   # (12, |S|)
    k = float(np.linalg.norm(G, 2))
    return k, r


def reconstruct_dipolar(M_s: np.ndarray, mu_s: np.ndarray, observed_leads, y_S: np.ndarray,
                        rcond: float = RECON_RCOND) -> np.ndarray:
    """Recover the population-dipolar projection of the full 12-lead from ``y_S``.

    ``L_hat = mu_s + M_s M_{s,S}^+ (y_S - mu_{s,S})``.  ``y_S`` may be a vector
    ``(|S|,)`` or a batch ``(|S|, T)`` of time samples.
    """
    idx = _observed_idx(observed_leads)
    M_S = M_s[idx]
    y = np.asarray(y_S, dtype=float)
    resid = y - mu_s[idx][..., None] if y.ndim == 2 else y - mu_s[idx]
    d = np.linalg.pinv(M_S, rcond=rcond) @ resid          # dipole coords
    L_dip = M_s @ d
    return (mu_s[..., None] + L_dip) if y.ndim == 2 else (mu_s + L_dip)


def inverse_dower_matrix() -> np.ndarray:
    """Classical inverse-Dower transform: VCG (X, Y, Z) -> 8 independent leads.

    Reference transform used only to *cross-validate* the data-estimated dipolar
    subspace (the certificate does not depend on Dower).  Rows are the eight
    independent leads ``[I, II, V1..V6]``; columns are the Frank X, Y, Z axes.
    Coefficients are the widely used Dower/inverse-Dower values (Edenbrandt &
    Pahlm, 1988).
    """
    # rows: I, II, V1, V2, V3, V4, V5, V6 ; cols: X, Y, Z
    D = np.array([
        [ 0.156, -0.010, -0.172],   # I
        [-0.227,  0.887,  0.057],   # II
        [-0.515,  0.157, -0.917],   # V1
        [ 0.044,  0.164, -1.387],   # V2
        [ 0.882,  0.098, -1.277],   # V3
        [ 1.213,  0.127, -0.601],   # V4
        [ 1.125,  0.127, -0.086],   # V5
        [ 0.831,  0.076,  0.230],   # V6
    ], dtype=float)
    return D


@dataclass
class DipolarModel:
    """Per-segment dipolar models fitted from a population of ECGs.

    Attributes
    ----------
    M : dict[str, (12, 3)]      dipolar basis per segment
    mu : dict[str, (12,)]       population segment mean per segment
    evr : dict[str, (12,)]      variance-explained ratios per segment
    rank : int                  dipolar rank (default 3)
    """

    M: dict[str, np.ndarray]
    mu: dict[str, np.ndarray]
    evr: dict[str, np.ndarray]
    rank: int = 3

    @classmethod
    def fit(cls, seg_samples: dict[str, np.ndarray], rank: int = 3) -> "DipolarModel":
        """Fit from ``{segment: (N_s, 12) population samples}``."""
        M, mu, evr = {}, {}, {}
        for seg, X in seg_samples.items():
            M[seg], mu[seg], evr[seg] = fit_dipolar_subspace(X, rank=rank)
        return cls(M=M, mu=mu, evr=evr, rank=rank)

    def dipolar_fraction(self, seg: str) -> float:
        """Fraction of segment variance captured by the dipole (top-``rank``)."""
        return float(self.evr[seg][: self.rank].sum())

    def kappa(self, seg: str, observed_leads) -> tuple[float, int]:
        return kappa(self.M[seg], observed_leads)

    def reconstruct_dipolar(self, seg: str, observed_leads, y_S: np.ndarray) -> np.ndarray:
        return reconstruct_dipolar(self.M[seg], self.mu[seg], observed_leads, y_S)
