"""Numerical validation of the minimax per-lead recoverability floor (paper/theorem_floor.tex).

The floor a_{s,l}(S)^2 = e_l^T M_s Sigma_{Q|P} M_s^T e_l (= expected_ambiguity_per_lead^2) is the
minimax risk of the dipolar functional theta_l = e_l^T M_s d over the moment class {E[dd^T] <= Sigma_d},
attained by the least-favourable Gaussian prior and its linear posterior mean. We verify, on
controlled Gaussian data (the least-favourable prior), that:
  (1) a_l^2 equals both the Schur-complement form and expected_ambiguity_per_lead^2;
  (2) the Bayes posterior-mean estimator of theta_l achieves MSE = a_l^2 (Monte Carlo);
  (3) NO estimator beats a_l under the Gaussian prior -- the measured dipolar-projection error of
      every linear reconstructor sits on/above a_l (0 floor violations);
  (4) a_l = 0 iff the lead is identifiable (eta = 0).
"""
import numpy as np
from ecgcert.physics import (
    fit_dipolar_subspace, reconstruct_dipolar, dipole_coord_cov,
    eta_per_lead, expected_ambiguity_per_lead, LEAD_INDEX,
)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _setup(seed=0, N=8000):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((12, 3))
    mu = rng.standard_normal(12) * 0.1
    Sig = np.diag([1.0, 0.5, 0.25])
    d = rng.standard_normal((N, 3)) @ np.linalg.cholesky(Sig).T          # Gaussian LF prior
    L = mu + d @ M.T                                                     # exact dipolar (noise-free)
    Mhat, muhat, _ = fit_dipolar_subspace(L, rank=3)
    return rng, Mhat, muhat, dipole_coord_cov(Mhat, muhat, L), L


def test_floor_equals_schur_and_ambiguity():
    _, M, mu, Sig_d, _ = _setup()
    obs = ["I", "II"]
    amb = expected_ambiguity_per_lead(M, obs, Sig_d)
    # explicit Schur complement a_l^2
    oi = [LEAD_INDEX[l] for l in obs]
    MS = M[oi]
    P = np.linalg.pinv(MS) @ MS
    Q = np.eye(3) - P
    Sig_QP = Q @ Sig_d @ Q - Q @ Sig_d @ P @ np.linalg.pinv(P @ Sig_d @ P) @ P @ Sig_d @ Q
    for l in ["III", "V2", "V6"]:
        e = M[LEAD_INDEX[l]]
        a2 = float(e @ Sig_QP @ e)
        assert abs(np.sqrt(max(a2, 0.0)) - amb[LEAD_INDEX[l]]) < 1e-6


def test_bayes_posterior_mean_achieves_floor():
    """The linear posterior-mean estimator of theta_l attains MSE = a_l^2 under the Gaussian prior."""
    _, M, mu, Sig_d, L = _setup()
    obs = ["I", "II"]; oi = [LEAD_INDEX[l] for l in obs]
    amb = expected_ambiguity_per_lead(M, obs, Sig_d)
    d = (L - mu) @ np.linalg.pinv(M).T                                   # (N,3) true dipole coords
    MS = M[oi]
    # posterior mean of d given observed dipole coords P d, then theta_l = e_l^T M E[d|obs]
    P = np.linalg.pinv(MS) @ MS
    Pd = d @ P.T
    # E[d | Pd] for Gaussian N(0,Sig_d): Sig_d P^T (P Sig_d P^T)^+ Pd
    G = Sig_d @ P.T @ np.linalg.pinv(P @ Sig_d @ P.T)
    dhat = Pd @ G.T
    for l in ["III", "V2", "V6"]:
        e = M[LEAD_INDEX[l]]                                  # e_l^T M_s = row l of M_s (3-vector)
        theta = d @ e; theta_hat = dhat @ e                  # theta_l = e_l^T M_s d
        rmse = float(np.sqrt(np.mean((theta_hat - theta) ** 2)))
        assert abs(rmse - amb[LEAD_INDEX[l]]) < 0.03 * (amb[LEAD_INDEX[l]] + 1e-3)


def test_no_reconstructor_beats_floor_under_gaussian_prior():
    """Measured dipolar-projection error of dipolar/ridge/OLS sits on/above a_l (0 violations)."""
    rng, M, mu, Sig_d, L = _setup()
    P12 = M @ np.linalg.pinv(M)
    for obs in (["I", "II"], ["I", "II", "V2"]):
        oi = [LEAD_INDEX[l] for l in obs]
        amb = expected_ambiguity_per_lead(M, obs, Sig_d)
        Yo = L[:, oi]; T1 = np.hstack([Yo, np.ones((Yo.shape[0], 1))])
        recons = {"dipolar": reconstruct_dipolar(M, mu, obs, Yo.T).T}
        for nm, lam in (("ridge", 1.0), ("ols", 0.0)):
            A = T1.T @ T1 + lam * np.eye(T1.shape[1]); A[-1, -1] -= lam
            W = np.linalg.solve(A, T1.T @ L)                              # (|S|+1, 12)
            recons[nm] = Yo @ W[:-1] + W[-1]                             # (N,12)
        for nm, Lhat in recons.items():
            derr = (P12 @ (Lhat - L).T).T
            for l in LEADS:
                if l in obs:
                    continue
                li = LEAD_INDEX[l]
                rmse = float(np.sqrt(np.mean(derr[:, li] ** 2)))
                # allow a tiny finite-sample slack below the point floor
                assert rmse >= amb[li] - 0.02 * (amb[li] + 1e-3), (nm, l, rmse, amb[li])


def test_floor_zero_iff_identifiable():
    _, M, mu, Sig_d, _ = _setup()
    obs = ["I", "II", "V2"]                       # rank-3 spanning -> all identifiable
    eta = eta_per_lead(M, obs); amb = expected_ambiguity_per_lead(M, obs, Sig_d)
    for l in LEADS:
        if l in obs:
            continue
        li = LEAD_INDEX[l]
        assert (eta[li] < 1e-8) == (amb[li] < 1e-6)
