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

The certificate (per target lead)
---------------------------------
For an observed lead subset ``S`` with dipolar sub-matrix ``M_{s,S}`` (rows ``S``
of ``M_s``), the mean-centred dipolar estimate is
``L_hat = mu_s + M_s M_{s,S}^+ (y_S - mu_{s,S})``.  Per target lead ``ell`` we report
two closed-form numbers, both from a single truncated SVD of ``M_{s,S}``:

* ``eta_{s,ell}(S) = ||e_ell^T M_s (I - M_{s,S}^+ M_{s,S})||_2`` -- sensitivity of
  lead ``ell`` to dipole directions the observation cannot see.  ``eta=0`` means the
  dipolar component of lead ``ell`` is *identifiable* from ``S``; ``eta>0`` means an
  observation-indistinguishable dipole direction changes lead ``ell`` (its dipolar
  component is not identifiable at any SNR).
* ``kappa_{s,ell}(S) = ||e_ell^T M_s M_{s,S}^+||_2`` -- amplification of observation
  noise / observed non-dipolar residual into the *identifiable* part of lead ``ell``.

The global spectral ``kappa_s(S) = ||M_s M_{s,S}^+||_2`` is kept only as a
*configuration-level worst-case summary*; it is not a per-lead certificate and the
coarse rule "global rank < 3 => every unobserved lead is unrecoverable" is replaced by
the per-lead ``eta_{s,ell}``.  Generic inverse-problem theory cannot produce these
because it has no ``M``.  NOTE: ``M_s`` is a *population-estimated* (PCA) object, not a
first-principles physical dipole; these numbers depend on the estimated basis.
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


@dataclass
class ObservedDipole:
    """Everything derived from a SINGLE truncated SVD of ``M_{s,S}``.

    All rank/pseudo-inverse/projector/kappa/eta quantities share one relative
    truncation ``threshold = rcond * sigma_max(M_{s,S})`` so the certificate is
    numerically self-consistent (a dipole direction is either observed or not,
    everywhere).

    Attributes
    ----------
    idx : (|S|,)      observed lead indices
    M_S : (|S|, 3)    observed rows of ``M_s``
    pinv : (3, |S|)   truncated pseudo-inverse ``M_{s,S}^+``
    P_obs : (3, 3)    ``M_{s,S}^+ M_{s,S}`` -- projector onto the *observed* dipole
                      coordinate directions (rank ``r``)
    rank : int        number of dipole directions the observation constrains
    sv : (min(|S|,3),) singular values of ``M_{s,S}``
    threshold : float truncation level ``rcond * sv[0]``
    """

    idx: np.ndarray
    M_S: np.ndarray
    pinv: np.ndarray
    P_obs: np.ndarray
    rank: int
    sv: np.ndarray
    threshold: float


def observed_dipole(M_s: np.ndarray, observed_leads, rcond: float = RECON_RCOND) -> ObservedDipole:
    """Single truncated SVD of ``M_{s,S}``; source of every downstream quantity."""
    M_s = np.asarray(M_s, dtype=float)
    idx = _observed_idx(observed_leads)
    M_S = M_s[idx]                                          # (|S|, 3)
    U, sv, Vt = np.linalg.svd(M_S, full_matrices=False)     # U:(|S|,k) sv:(k,) Vt:(k,3)
    smax = float(sv[0]) if sv.size else 0.0
    thr = rcond * smax
    keep = sv > thr
    r = int(keep.sum())
    sv_inv = np.zeros_like(sv)
    sv_inv[keep] = 1.0 / sv[keep]
    pinv = (Vt.T * sv_inv) @ U.T                            # (3, |S|) truncated M_S^+
    P_obs = Vt.T[:, keep] @ Vt[keep, :]                    # (3, 3) M_S^+ M_S
    return ObservedDipole(idx=idx, M_S=M_S, pinv=pinv, P_obs=P_obs, rank=r,
                          sv=sv, threshold=float(thr))


def kappa(M_s: np.ndarray, observed_leads, rcond: float = RECON_RCOND,
          lead=None) -> tuple[float, int]:
    """Noise / observed-residual amplification of the dipolar estimate, and rank.

    Global (``lead=None``): ``kappa_s(S) = ||M_s M_{s,S}^+||_2`` -- a
    *configuration-level worst-case* summary, NOT a per-lead certificate.

    Per target lead ``ell`` (``lead=ell``): ``kappa_{s,ell}(S) =
    ||e_ell^T M_s M_{s,S}^+||_2`` -- how much observation noise / observed
    non-dipolar residual is amplified into the dipolar estimate of lead ``ell``.

    Returns ``(kappa, rank)`` with ``rank = rank(M_{s,S})`` under the shared
    truncation. All quantities come from one :func:`observed_dipole` SVD.
    """
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    G = np.asarray(M_s, float) @ od.pinv                    # (12, |S|)
    if lead is None:
        k = float(np.linalg.norm(G, 2))
    else:
        li = LEAD_INDEX[lead] if isinstance(lead, str) else int(lead)
        k = float(np.linalg.norm(G[li], 2))
    return k, od.rank


def kappa_per_lead(M_s: np.ndarray, observed_leads, rcond: float = RECON_RCOND) -> np.ndarray:
    """Per-lead amplification ``kappa_{s,ell}(S)`` for all 12 leads -> (12,)."""
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    G = np.asarray(M_s, float) @ od.pinv                    # (12, |S|)
    return np.linalg.norm(G, axis=1)


def eta_per_lead(M_s: np.ndarray, observed_leads, rcond: float = RECON_RCOND) -> np.ndarray:
    """Per-lead unidentifiability ``eta_{s,ell}(S) = ||e_ell^T M_s (I - M_{s,S}^+ M_{s,S})||_2``.

    ``eta_{s,ell}=0``: lead ``ell``'s dipolar component is *identifiable* from ``S``
    (it lies entirely in the observed dipole directions).
    ``eta_{s,ell}>0``: there exists a dipole direction unobserved by ``S`` that
    changes lead ``ell`` -- its dipolar component is not identifiable at any SNR.
    Returns a (12,) array.
    """
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    unobs = np.asarray(M_s, float) @ (np.eye(3) - od.P_obs)  # (12, 3)
    return np.linalg.norm(unobs, axis=1)


def lead_dipolar_norm(M_s: np.ndarray) -> np.ndarray:
    """Per-lead total dipolar gain ``||e_ell^T M_s||_2`` (12,).

    The denominator for the *normalized* identifiability ``eta_tilde``: how much of
    lead ``ell``'s signal lives in the dipolar subspace at all. A lead with tiny
    dipolar norm (e.g. a low-amplitude precordial lead in a flat segment) can have a
    small absolute ``eta`` yet a large *fraction* unobserved.
    """
    return np.linalg.norm(np.asarray(M_s, float), axis=1)


def eta_normalized_per_lead(M_s: np.ndarray, observed_leads, rcond: float = RECON_RCOND,
                            eps: float = 1e-9) -> np.ndarray:
    """Normalized identifiability ``eta_tilde_{s,ell}(S) = eta_{s,ell} / ||e_ell^T M_s||_2``.

    In ``[0, 1]``: the *fraction* of lead ``ell``'s dipolar content lying in dipole
    directions unobserved by ``S``. ``eta_tilde ~ 0`` => the identifiable part is nearly
    all of the lead's dipolar signal; ``eta_tilde ~ 1`` => almost none is identifiable.
    This is the honest graded measure: absolute ``eta`` alone conflates ``S``-geometry with
    lead amplitude. Returns a (12,) array (``nan`` where the dipolar norm is ~0).
    """
    eta = eta_per_lead(M_s, observed_leads, rcond=rcond)
    denom = lead_dipolar_norm(M_s)
    out = np.full(12, np.nan)
    ok = denom > eps
    out[ok] = eta[ok] / denom[ok]
    return out


def dipole_coord_cov(M_s: np.ndarray, mu_s: np.ndarray, X_seg: np.ndarray) -> np.ndarray:
    """Covariance ``Sigma_d`` (3x3) of the dipole coordinates ``d = M_s^T (L - mu_s)``."""
    X = np.asarray(X_seg, float)
    d = (X - np.asarray(mu_s, float)) @ np.asarray(M_s, float)     # (N, 3)
    return np.cov(d.T)


def expected_ambiguity_per_lead(M_s: np.ndarray, observed_leads, Sigma_d: np.ndarray,
                                rcond: float = RECON_RCOND) -> np.ndarray:
    """Expected unobserved ambiguity in mV per lead under the dipole-coordinate prior.

    For a dipole ``d ~ (0, Sigma_d)``, the part of ``d`` in directions unobserved by ``S``
    is irrecoverable; its footprint on lead ``ell`` has std
    ``sqrt( e_ell^T M_s (I-P_obs) Sigma_d (I-P_obs) M_s^T e_ell )``. This puts the graded
    identifiability in millivolts (an expected error a perfect reconstructor still incurs on
    the dipolar component). Returns a (12,) array.
    """
    M_s = np.asarray(M_s, float)
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    A = M_s @ (np.eye(3) - od.P_obs)                               # (12, 3)
    cov = A @ np.asarray(Sigma_d, float) @ A.T                     # (12, 12)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def reconstruct_dipolar(M_s: np.ndarray, mu_s: np.ndarray, observed_leads, y_S: np.ndarray,
                        rcond: float = RECON_RCOND) -> np.ndarray:
    """Recover the population-dipolar projection of the full 12-lead from ``y_S``.

    ``L_hat = mu_s + M_s M_{s,S}^+ (y_S - mu_{s,S})`` -- theorem and code use the
    identical, mean-centred estimator. ``y_S`` may be ``(|S|,)`` or ``(|S|, T)``.
    """
    M_s = np.asarray(M_s, float)
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    y = np.asarray(y_S, dtype=float)
    mu_S = mu_s[od.idx]
    resid = y - mu_S[..., None] if y.ndim == 2 else y - mu_S
    L_dip = M_s @ (od.pinv @ resid)                        # dipole coords -> 12-lead
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

    def kappa(self, seg: str, observed_leads, lead=None) -> tuple[float, int]:
        return kappa(self.M[seg], observed_leads, lead=lead)

    def kappa_per_lead(self, seg: str, observed_leads) -> np.ndarray:
        return kappa_per_lead(self.M[seg], observed_leads)

    def eta_per_lead(self, seg: str, observed_leads) -> np.ndarray:
        return eta_per_lead(self.M[seg], observed_leads)

    def reconstruct_dipolar(self, seg: str, observed_leads, y_S: np.ndarray) -> np.ndarray:
        return reconstruct_dipolar(self.M[seg], self.mu[seg], observed_leads, y_S)
