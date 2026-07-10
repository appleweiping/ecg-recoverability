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
    """The Tier I/III boundary is the dipole RANK; kappa ranks conditioning among
    dipole-spanning sets. Both together are the "geometry, not lead count" story."""
    L = _synthetic_dipolar_population()
    M_s, _, _ = fit_dipolar_subspace(L, rank=3)
    # (a) A coplanar limb triplet cannot span the dipole: rank 2 => a direction is
    #     provably unrecoverable (Tier III), regardless of how many limb leads.
    _, r_span = kappa(M_s, ["I", "II", "V2"])        # spans the dipole
    _, r_limb = kappa(M_s, ["I", "II", "III"])       # III = II - I => coplanar
    assert r_span == 3
    assert r_limb == 2
    # (b) Among dipole-spanning triplets, geometry sets the conditioning: three
    #     well-spread leads condition far better than three adjacent precordials.
    k_good, rg = kappa(M_s, ["I", "II", "V2"])
    k_bad, rb = kappa(M_s, ["V1", "V2", "V3"])       # nearly collinear directions
    assert rg == 3 and rb == 3
    assert k_good < k_bad


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
