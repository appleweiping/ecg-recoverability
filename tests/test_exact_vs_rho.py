"""Exact identifiability vs rho-truncated (numerical) recoverability (P0-A).

Three constructed cases pin down the distinction the paper now draws:
  (1) exact rank-deficient observed matrix  -> exact non-identifiability (eta_exact > 0);
  (2) tiny but nonzero singular value       -> exact identifiable, rho-unidentifiable;
  (3) full-rank well-conditioned            -> both identifiable (eta_exact = eta_rho = 0).
"""
import numpy as np

from ecgcert.physics import eta_exact_per_lead, eta_per_lead


def test_exact_rank_deficient_is_exactly_non_identifiable():
    # Observed = 2 leads -> M_{s,S} is 2x3, exact rank 2. A target whose M_s row is
    # outside that 2-D row space is exactly non-identifiable at any SNR.
    M = np.zeros((12, 3))
    M[0] = [1, 0, 0]
    M[1] = [0, 1, 0]
    M[2] = [0, 0, 1]          # target lead 2: needs the unobserved 3rd direction
    obs = [0, 1]
    ee = eta_exact_per_lead(M, obs)
    er = eta_per_lead(M, obs)
    assert ee[2] > 1e-6, "exact rank-deficient must be exactly non-identifiable"
    assert er[2] > 1e-6, "rho version agrees on an exact rank deficiency"
    assert ee[0] < 1e-9 and ee[1] < 1e-9, "observed leads are trivially identifiable"


def test_tiny_singular_value_exact_identifiable_but_rho_unidentifiable():
    # M_{s,S} = diag(1, 1, 1e-6): exact rank 3 (row space = R^3) so eta_exact = 0,
    # but at rho=1e-2 the 1e-6 direction is truncated so eta_rho > 0.
    M = np.zeros((12, 3))
    M[0] = [1, 0, 0]
    M[1] = [0, 1, 0]
    M[2] = [0, 0, 1e-6]       # observed only through a tiny singular value
    M[3] = [0, 0, 1.0]        # target lead 3 lives on that direction
    obs = [0, 1, 2]
    ee = eta_exact_per_lead(M, obs)
    er = eta_per_lead(M, obs, rcond=1e-2)
    assert ee[3] < 1e-6, "tiny-but-nonzero sv is EXACTLY identifiable"
    assert er[3] > 0.5, "the same target is rho-unidentifiable at rho=1e-2"


def test_full_rank_well_conditioned_both_identifiable():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 3)))   # orthonormal columns
    obs = [0, 1, 2, 7, 8]                                 # 5 leads -> full-rank 5x3
    ee = eta_exact_per_lead(Q, obs)
    er = eta_per_lead(Q, obs, rcond=1e-2)
    assert np.max(ee) < 1e-9, "well-conditioned full rank: exactly identifiable"
    assert np.max(er) < 1e-9, "well-conditioned full rank: rho identifiable too"


def test_exact_and_rho_agree_when_well_separated():
    # Away from the truncation boundary the two verdicts coincide.
    rng = np.random.default_rng(1)
    M = rng.standard_normal((12, 3))
    for obs in ([0, 1], [0, 1, 2], [0, 1, 6, 7, 8]):
        ee = eta_exact_per_lead(M, obs)
        er = eta_per_lead(M, obs, rcond=1e-2)
        # same zero/nonzero pattern (both use a rank test, just different tolerances)
        assert np.all((ee > 1e-6) == (er > 1e-6))
