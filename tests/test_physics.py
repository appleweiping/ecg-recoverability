"""Theorem-vs-simulation checks for the dipolar recoverability certificate.

Every property asserted here is a claim that appears (or will appear) in the paper.
House rule: no claim enters the manuscript until its test is green.
"""
import numpy as np
import pytest

from ecgcert.physics import (
    LEADS,
    LEAD_INDEX,
    check_lead_algebra,
    fit_dipolar_subspace,
    inverse_dower_matrix,
    kappa,
    kappa_per_lead,
    eta_per_lead,
    observed_dipole,
    lead_transform_T,
    reconstruct_dipolar,
)


def test_lead_algebra_rank8_and_relations():
    assert check_lead_algebra()
    T = lead_transform_T()
    assert T.shape == (12, 8)
    assert np.linalg.matrix_rank(T) == 8


def _synthetic_dipolar_population(n=4000, seed=0):
    """Population of exactly-dipolar 12-lead vectors: L = T @ (D_ind @ dipole)."""
    rng = np.random.default_rng(seed)
    T = lead_transform_T()                      # (12, 8)
    D = inverse_dower_matrix()                   # (8, 3) independent-lead <- XYZ
    dipoles = rng.standard_normal((3, n)) * np.array([[1.0], [0.7], [0.5]])
    X_ind = D @ dipoles                          # (8, n) independent leads
    L = T @ X_ind                                # (12, n) full 12-lead
    return L.T                                    # (n, 12)


def test_dipolar_subspace_matches_dower():
    """Data-estimated dipolar subspace equals the inverse-Dower column space."""
    L = _synthetic_dipolar_population()
    M_s, mu_s, evr = fit_dipolar_subspace(L, rank=3, center=True)
    # Exactly dipolar data => first 3 directions capture ~all variance.
    assert evr[:3].sum() > 0.999
    # Column space of M_s must equal column space of T @ D (the true dipolar map).
    T, D = lead_transform_T(), inverse_dower_matrix()
    true_map = T @ D                             # (12, 3)
    P_est = M_s @ M_s.T
    P_true = true_map @ np.linalg.pinv(true_map)
    assert np.linalg.norm(P_est - P_true) < 1e-6


def test_tier1_exact_recovery_noiseless():
    """A dipole-spanning lead set recovers ALL 12 leads exactly (noiseless)."""
    L = _synthetic_dipolar_population()
    M_s, mu_s, _ = fit_dipolar_subspace(L, rank=3)
    # A fresh exactly-dipolar test beat.
    Ltest = _synthetic_dipolar_population(n=200, seed=99)  # (200, 12)
    observed = ["I", "II", "V2"]                            # dipole-spanning
    idx = [LEAD_INDEX[l] for l in observed]
    y_S = Ltest[:, idx].T                                   # (3, 200)
    L_hat = reconstruct_dipolar(M_s, mu_s, observed, y_S)   # (12, 200)
    assert np.linalg.norm(L_hat - Ltest.T) / np.linalg.norm(Ltest.T) < 1e-6


def test_kappa_geometry_not_leadcount():
    """Geometry, not lead count, sets identifiability and conditioning."""
    L = _synthetic_dipolar_population()
    M_s, _, _ = fit_dipolar_subspace(L, rank=3)
    # (a) A coplanar limb triplet is EXACTLY rank 2 at any tolerance (III = II - I).
    _, r_limb = kappa(M_s, ["I", "II", "III"])
    assert r_limb == 2
    # (b) A well-spread 5-lead set is robustly rank 3 and well conditioned; a spread
    #     triplet spans too but conditions worse -> geometry, not count.
    k5, r5 = kappa(M_s, ["I", "II", "V1", "V3", "V5"])
    k3, r3 = kappa(M_s, ["I", "II", "V2"])
    assert r5 == 3 and r3 == 3
    assert k5 < k3


def test_kappa_rank_is_rcond_sensitive_for_near_deficient():
    """A near-rank-deficient config's rank/kappa DEPEND on the truncation tolerance:
    {V1,V2,V3}'s third dipole direction is ~0.5% of the first, so it is treated as
    unobserved (rank 2) at the deployment tolerance but observed-yet-wildly-amplified
    (rank 3, kappa ~ 200) at a tight tolerance. This is why a single kappa number for
    such configs is not meaningful (report rcond sensitivity / bootstrap CIs)."""
    L = _synthetic_dipolar_population()
    M_s, _, _ = fit_dipolar_subspace(L, rank=3)
    k_loose, r_loose = kappa(M_s, ["V1", "V2", "V3"], rcond=1e-2)   # deployment
    k_tight, r_tight = kappa(M_s, ["V1", "V2", "V3"], rcond=1e-4)   # tight
    assert r_loose == 2 and r_tight == 3
    assert k_tight > 100 and k_loose < 10


def test_per_lead_eta_and_kappa_certificate():
    """eta_{s,ell} certifies per-lead identifiability; kappa_{s,ell} its noise gain."""
    L = _synthetic_dipolar_population()
    M_s, _, _ = fit_dipolar_subspace(L, rank=3)
    # Spanning config: every lead's dipolar component is identifiable (eta ~ 0).
    eta_span = eta_per_lead(M_s, ["I", "II", "V2"])
    assert np.max(eta_span) < 1e-6
    # Coplanar limb triplet: precordial leads depend on the unobserved transverse
    # dipole direction -> eta > 0 for them, but ~0 for the observed limb leads.
    eta_limb = eta_per_lead(M_s, ["I", "II", "III"])
    assert eta_limb[LEAD_INDEX["V3"]] > 1e-3
    assert eta_limb[LEAD_INDEX["I"]] < 1e-6
    # Per-lead kappa is finite and the global kappa is its max over leads (worst case).
    kpl = kappa_per_lead(M_s, ["I", "II", "V2"])
    kglob, _ = kappa(M_s, ["I", "II", "V2"])
    assert np.all(np.isfinite(kpl))
    # Global kappa (spectral norm) is the per-lead worst case: >= max_ell kappa_{s,ell}.
    assert kglob >= np.max(kpl) - 1e-9


def test_tier1_noise_amplification_matches_kappa():
    """Reconstruction error norm is bounded by kappa * noise level (the certificate)."""
    rng = np.random.default_rng(1)
    L = _synthetic_dipolar_population()
    M_s, mu_s, _ = fit_dipolar_subspace(L, rank=3)
    observed = ["I", "II", "V2"]
    idx = [LEAD_INDEX[l] for l in observed]
    k, r = kappa(M_s, observed)
    assert r == 3
    Ltest = _synthetic_dipolar_population(n=1, seed=7)          # one exact-dipolar beat
    sigma = 0.05
    worst_ratio = 0.0
    for _ in range(2000):
        n = rng.standard_normal(3) * sigma
        y = Ltest[0, idx] + n
        L_hat = reconstruct_dipolar(M_s, mu_s, observed, y)
        err = np.linalg.norm(L_hat - Ltest[0])                 # = ||M_s M_{s,S}^+ n||
        # The certificate is the deterministic spectral-norm bound err <= kappa*||n||.
        worst_ratio = max(worst_ratio, err / (k * np.linalg.norm(n) + 1e-12))
    assert worst_ratio <= 1.0 + 1e-6
    # And kappa is attained: the worst-case noise direction realises the bound.
    assert worst_ratio > 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
