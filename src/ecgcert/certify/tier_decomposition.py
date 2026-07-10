"""Tier I / II / III decomposition and the null-space hallucination energy.

Given a per-segment dipolar model ``M_s`` (12x3 orthonormal) and an observed lead
subset ``S``, we split the 12-lead space into

* the **recoverable dipole subspace** ``R_s`` -- the dipole directions the
  observation actually constrains (Tier I, exact up to noise ``kappa_s(S)``), and
* its orthogonal complement ``U_s = I - R_s`` -- the **certified-unrecoverable**
  subspace containing (i) unobserved dipole directions and (ii) all non-dipolar
  content.  Any energy a reconstruction places in ``U_s`` beyond the population
  mean is unsupported by the observation; on the non-dipolar, observation-
  independent part it is, by the non-identifiability lemma, *fabricated*.

The scalar we monitor per (segment, lead) is the **hallucination energy**

    h_{s,l} = RMS_t [ U_s ( L_hat(t) - mu_s ) ]_l ,

the projection of the reconstruction's deviation-from-prior onto the certified-
unrecoverable subspace, read out at lead ``l``.  A distribution-free flag threshold
is calibrated on faithful reconstructions in :mod:`ecgcert.conformal`.

Geometry produces ``h``; calibration turns it into a guaranteed flag.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from ecgcert.physics.dipolar_subspace import LEAD_INDEX


class Tier(str, Enum):
    OBSERVED = "observed"        # lead is measured directly
    RECOVERABLE = "tier1"        # dipolar projection exactly recoverable
    STATISTICAL = "tier2"        # non-dipolar, population-predictable
    UNRECOVERABLE = "tier3"      # non-dipolar / unobserved-dipole, not identifiable


def _observed_idx(observed_leads) -> np.ndarray:
    return np.array([LEAD_INDEX[l] if isinstance(l, str) else int(l) for l in observed_leads])


def selection_matrix(observed_leads) -> np.ndarray:
    """Row-selection ``Sel_S in {0,1}^{|S|x12}`` with ``y_S = Sel_S @ L``."""
    idx = _observed_idx(observed_leads)
    Sel = np.zeros((len(idx), 12))
    Sel[np.arange(len(idx)), idx] = 1.0
    return Sel


def recoverable_dipole_projector(M_s: np.ndarray, observed_leads,
                                 rcond: float = 1e-10) -> tuple[np.ndarray, int]:
    """Projector ``R_s`` onto the dipole subspace the observation constrains.

    ``R_s = M_s P_obs M_s^T`` where ``P_obs = M_{s,S}^+ M_{s,S}`` projects the
    3-D dipole coordinates onto the directions observable from ``S``.  Returns
    ``(R_s, r)`` with ``r = rank(M_{s,S})`` the number of recoverable dipole
    directions (``r = 3`` => the whole dipole is recoverable).
    """
    idx = _observed_idx(observed_leads)
    M_S = M_s[idx]                                   # (|S|, 3)
    P_obs = np.linalg.pinv(M_S, rcond=rcond) @ M_S    # (3, 3) projector on observable dipole
    R_s = M_s @ P_obs @ M_s.T                         # (12, 12)
    r = int(np.linalg.matrix_rank(M_S, tol=rcond * max(M_S.shape)))
    return R_s, r


def certified_unrecoverable_projector(M_s: np.ndarray, observed_leads) -> np.ndarray:
    """``U_s = I - R_s`` -- non-dipolar plus unobserved-dipole directions."""
    R_s, _ = recoverable_dipole_projector(M_s, observed_leads)
    return np.eye(R_s.shape[0]) - R_s


def supported_reconstruction(M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                             L_hat: np.ndarray) -> np.ndarray:
    """Strip unsupported content: ``mu_s + R_s (L_hat - mu_s)``.

    The part of any reconstruction the certificate is willing to stand behind.
    ``L_hat`` is ``(12,)`` or ``(12, T)``.
    """
    R_s, _ = recoverable_dipole_projector(M_s, observed_leads)
    mu = mu_s[:, None] if L_hat.ndim == 2 else mu_s
    return mu + R_s @ (L_hat - mu)


def hallucination_energy(M_s: np.ndarray, mu_s: np.ndarray, observed_leads,
                         L_hat: np.ndarray) -> np.ndarray:
    """Per-lead hallucination energy ``h_l`` of a reconstruction.

    ``L_hat`` is ``(12, T)`` (a segment window) or ``(12,)``.  Returns a length-12
    vector: the RMS over time of the certified-unrecoverable component at each lead.
    Observed leads are excluded (set to 0) since they are not reconstructed.
    """
    U_s = certified_unrecoverable_projector(M_s, observed_leads)
    X = L_hat if L_hat.ndim == 2 else L_hat[:, None]
    resid = X - mu_s[:, None]
    unsupported = U_s @ resid                          # (12, T)
    h = np.sqrt(np.mean(unsupported**2, axis=1))       # (12,)
    h[_observed_idx(observed_leads)] = 0.0
    return h


def tier_report(model, observed_leads, dipolar_threshold: float = 0.8) -> dict:
    """Per (segment, lead) certificate summary for a lead configuration.

    Returns ``{segment: {lead: {tier, observed, dipole_rank, kappa,
    seg_dipolar_fraction}}}``.  A reconstructed lead is labelled RECOVERABLE when
    the observation spans the dipole (rank 3) and the segment is strongly dipolar
    (fraction >= ``dipolar_threshold``); UNRECOVERABLE when the dipole is not
    spanned (rank < 3); STATISTICAL otherwise (dipole spanned but the segment
    carries material non-dipolar content that only a prior can fill).
    """
    from ecgcert.physics.dipolar_subspace import LEADS

    obs = set(int(i) for i in _observed_idx(observed_leads))
    out: dict = {}
    for seg, M_s in model.M.items():
        _, r = recoverable_dipole_projector(M_s, observed_leads)
        kap, _ = _kappa(M_s, observed_leads)
        frac = model.dipolar_fraction(seg)
        out[seg] = {}
        for li, lead in enumerate(LEADS):
            if li in obs:
                tier = Tier.OBSERVED
            elif r < 3:
                tier = Tier.UNRECOVERABLE
            elif frac >= dipolar_threshold:
                tier = Tier.RECOVERABLE
            else:
                tier = Tier.STATISTICAL
            out[seg][lead] = {
                "tier": tier.value,
                "observed": li in obs,
                "dipole_rank": r,
                "kappa": float(kap),
                "seg_dipolar_fraction": float(frac),
            }
    return out


def _kappa(M_s, observed_leads, rcond: float = 1e-10):
    from ecgcert.physics.dipolar_subspace import kappa as _k

    return _k(M_s, observed_leads, rcond=rcond)
