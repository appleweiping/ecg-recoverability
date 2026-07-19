"""Per-target-lead recoverability decomposition (honest, three-layer).

Given a per-segment dipolar basis ``M_s`` (12x3, PCA-estimated) and an observed
lead subset ``S``, we split each *target lead* into three layers:

1. **Physics-identifiable dipolar component** -- the part of a lead's dipolar
   projection the observation constrains.  ``R_s = M_s P_obs M_s^T`` projects onto
   it (``P_obs = M_{s,S}^+ M_{s,S}``); per lead, identifiability is certified by
   ``eta_{s,ell}(S)=0`` and its noise gain by ``kappa_{s,ell}(S)`` (see
   :mod:`ecgcert.physics`).

2. **Empirically predictable residual** -- non-dipolar content (and unobserved
   dipole directions) that a predictor trained on ``S`` *can* recover on held-out
   data.  This lives in the complement ``U_s = I - R_s`` and is calibrated with
   distribution-free intervals (:mod:`ecgcert.conformal`).

3. **Unresolved residual / achievability gap** -- content that no predictor in the
   evaluated family recovers on held-out data.  Only this layer is genuinely
   unrecoverable *for that family*, and it is established by an achievability
   analysis, NOT declared a priori.

IMPORTANT (corrected from an earlier version): the complement ``U_s = I - R_s`` is
**not** "certified unrecoverable" and energy placed in it is **not** by itself
fabrication.  ``U_s`` contains the empirically predictable residual (layer 2).  We
therefore call it the *off-dipole* subspace and its energy the *off-dipole energy*;
whether that energy is fabrication is decided by the held-out achievability
experiment, not by geometry.  A strict Tier-III independence bound (``p(u|y)=p(u)``)
holds only for explicitly-constructed synthetic data, not for real ECG.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from ecgcert.physics.dipolar_subspace import (
    LEAD_INDEX, LEADS, RECON_RCOND, observed_dipole, kappa_per_lead, eta_per_lead)


class Tier(str, Enum):
    OBSERVED = "observed"                # lead is measured directly
    IDENTIFIABLE = "dipole_identifiable"  # eta=0: dipolar component recoverable from S
    UNIDENTIFIABLE = "dipole_unidentifiable"  # eta>0: an unobserved dipole direction changes this lead


def _observed_idx(observed_leads) -> np.ndarray:
    return np.array([LEAD_INDEX[l] if isinstance(l, str) else int(l) for l in observed_leads])


def selection_matrix(observed_leads) -> np.ndarray:
    """Row-selection ``Sel_S in {0,1}^{|S|x12}`` with ``y_S = Sel_S @ L``."""
    idx = _observed_idx(observed_leads)
    Sel = np.zeros((len(idx), 12))
    Sel[np.arange(len(idx)), idx] = 1.0
    return Sel


def recoverable_dipole_projector(M_s: np.ndarray, observed_leads,
                                 rcond: float | None = None) -> tuple[np.ndarray, int]:
    """Projector ``R_s`` onto the dipole subspace the observation constrains.

    ``R_s = M_s P_obs M_s^T`` with ``P_obs = M_{s,S}^+ M_{s,S}`` (all from the shared
    truncated SVD, :func:`observed_dipole`).  Returns ``(R_s, r)`` with
    ``r = rank(M_{s,S})`` recoverable dipole directions.
    """
    rcond = RECON_RCOND if rcond is None else rcond
    od = observed_dipole(M_s, observed_leads, rcond=rcond)
    R_s = np.asarray(M_s, float) @ od.P_obs @ np.asarray(M_s, float).T
    return R_s, od.rank


def off_dipole_projector(M_s: np.ndarray, observed_leads,
                         rcond: float | None = None) -> np.ndarray:
    """``U_s = I - R_s`` -- the complement of the recoverable dipole subspace.

    Contains unobserved dipole directions AND all non-dipolar content, part of
    which is empirically predictable from ``S`` (layer 2).  This is NOT a
    "certified-unrecoverable" subspace; energy here is not by itself fabrication.
    """
    R_s, _ = recoverable_dipole_projector(M_s, observed_leads, rcond=rcond)
    return np.eye(R_s.shape[0]) - R_s


def dipole_supported_reconstruction(M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                                    L_hat: np.ndarray, rcond: float | None = None) -> np.ndarray:
    """The dipole-identifiable part of a reconstruction: ``mu_s + R_s (L_hat - mu_s)``.

    The part the *physics* layer stands behind (layers 2-3 are the calibration's
    responsibility).  ``L_hat`` is ``(12,)`` or ``(12, T)``.
    """
    R_s, _ = recoverable_dipole_projector(M_s, observed_leads, rcond=rcond)
    mu = mu_s[:, None] if L_hat.ndim == 2 else mu_s
    return mu + R_s @ (L_hat - mu)


def off_dipole_energy(M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                      L_hat: np.ndarray, rcond: float | None = None) -> np.ndarray:
    """Per-lead *off-dipole energy* of a reconstruction (renamed from hallucination).

    ``L_hat`` is ``(12, T)`` or ``(12,)``.  Returns a length-12 vector: the RMS over
    time of the component of ``L_hat - mu_s`` lying off the recoverable dipole
    subspace, read at each lead.  Observed leads are zeroed (not reconstructed).

    This measures deviation-from-dipole that the observation does not pin down; it is
    a *deployable scalar* (no ground truth), but it is NOT fabrication by itself --
    it also contains empirically predictable non-dipolar content.  Whether it is
    fabrication is decided by the held-out achievability analysis.
    """
    U_s = off_dipole_projector(M_s, observed_leads, rcond=rcond)
    X = L_hat if L_hat.ndim == 2 else L_hat[:, None]
    resid = X - mu_s[:, None]
    off = U_s @ resid                                     # (12, T)
    h = np.sqrt(np.mean(off**2, axis=1))                  # (12,)
    h[_observed_idx(observed_leads)] = 0.0
    return h


def tier_report(model, observed_leads, rcond: float | None = None,
                eta_tol: float = 1e-6) -> dict:
    """Per (segment, lead) identifiability report.

    Returns ``{segment: {lead: {tier, observed, eta, kappa, dipole_rank,
    seg_dipolar_fraction}}}``.  A target lead is:

    * OBSERVED           -- measured directly;
    * IDENTIFIABLE       -- ``eta_{s,ell}(S) <= eta_tol`` (dipolar component
                            recoverable from ``S``, with noise gain ``kappa_{s,ell}``);
    * UNIDENTIFIABLE     -- ``eta_{s,ell}(S) > eta_tol`` (an unobserved dipole
                            direction changes this lead).

    This is per-lead; it replaces the coarse "global rank < 3 => everything
    unobserved is unrecoverable" rule.
    """
    rcond = RECON_RCOND if rcond is None else rcond
    obs = set(int(i) for i in _observed_idx(observed_leads))
    out: dict = {}
    for seg, M_s in model.M.items():
        eta = eta_per_lead(M_s, observed_leads, rcond=rcond)   # (12,)
        kap = kappa_per_lead(M_s, observed_leads, rcond=rcond) # (12,)
        _, r = recoverable_dipole_projector(M_s, observed_leads, rcond=rcond)
        frac = model.dipolar_fraction(seg)
        out[seg] = {}
        for li, lead in enumerate(LEADS):
            if li in obs:
                tier = Tier.OBSERVED
            elif eta[li] <= eta_tol:
                tier = Tier.IDENTIFIABLE
            else:
                tier = Tier.UNIDENTIFIABLE
            out[seg][lead] = {
                "tier": tier.value,
                "observed": li in obs,
                "eta": float(eta[li]),
                "kappa": float(kap[li]),
                "dipole_rank": int(r),
                "seg_dipolar_fraction": float(frac),
            }
    return out
