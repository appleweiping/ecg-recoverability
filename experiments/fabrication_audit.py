"""Null-space dipolar ENERGY ratio (exploratory, ground-truth-free descriptor).

A reconstruction is *determined* by the observed leads only in the observed dipole directions
P_obs = M_{s,S}^+ M_{s,S}; the complementary directions Q = I - P_obs are unidentifiable from S.
As a descriptive diagnostic we measure the fraction of a reconstruction's dipolar energy placed in Q,

    R_Q_s(S) = || M_s Q d_hat ||_2^2 / || M_s d_hat ||_2^2  in [0,1],   d_hat = M_s^+ (Lhat - mu_s).

IMPORTANT -- what R_Q is NOT: R_Q > 0 does NOT by itself prove fabrication/hallucination. The true
dipole can carry Q-energy, and under CORRELATED observed/unobserved dipole coordinates the Bayes
posterior mean E[Q d | P d] is generally NONZERO; so a reconstruction with R_Q = 0 (which sets
Q d_hat = 0) is NOT the posterior mean and R_Q = 0 is NOT 'correct abstention'. A calibrated metric
would standardize Q d_hat - E[Q d | P d] by the conditional covariance Sigma_{Q|P}; that is left to
future work. We report R_Q only as an exploratory descriptor of how much a reconstruction asserts in
the unidentifiable subspace. Audited here (CPU): dipolar / ridge / OLS / prior-mean; the diffusion
model's R_Q(w) vs guidance is in the diffusion pipeline.

Output: results/fabrication_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import LEAD_INDEX, fit_dipolar_subspace, reconstruct_dipolar

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "Lead-I": ["I"], "Lead-II": ["II"], "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def null_space_energy_per_sample(M, mu, obs, Lhat):
    """Per-sample R_Q = ||M_s Q d_hat||^2 / ||M_s d_hat||^2 -> (N,). Lhat: (N,12)."""
    oi = [LEAD_INDEX[l] for l in obs]
    Mp = np.linalg.pinv(M)                                   # (3,12)
    P = np.linalg.pinv(M[oi]) @ M[oi]                        # (3,3) observed-dipole projector
    Q = np.eye(3) - P
    dhat = (Lhat - mu) @ Mp.T                                # (N,3)
    num = np.sum((dhat @ Q.T @ M.T) ** 2, axis=1)           # ||M Q d_hat||^2
    den = np.sum((dhat @ M.T) ** 2, axis=1) + 1e-12         # ||M d_hat||^2
    return num / den


def null_space_energy_ratio(M, mu, obs, Lhat):
    """R_Q averaged over samples (scalar).

    Descriptor only: R_Q>0 is energy asserted in the unidentifiable subspace, NOT proof of
    fabrication (see module docstring; the Bayes posterior mean E[Qd|Pd] is generally nonzero).
    """
    return float(np.mean(null_space_energy_per_sample(M, mu, obs, Lhat)))


def run(n_train=1500, n_test=1500, max_per_record=40, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:n_train]
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    te_ids = rng.permutation(np.intersect1d(f10, norm10))[:n_test]
    print(f"[fab] collecting segment samples train={len(tr_ids)} test={len(te_ids)}", flush=True)
    tr = db.collect_all_segments_with_ids(tr_ids, rate=100, max_per_record=max_per_record,
                                          max_records=len(tr_ids), seed=seed)
    te = db.collect_all_segments_with_ids(te_ids, rate=100, max_per_record=max_per_record,
                                          max_records=len(te_ids), seed=seed)
    out = {"metric": "R_Q = ||M_s Q d_hat||^2 / ||M_s d_hat||^2, fraction of a reconstruction's dipolar "
                      "energy in the UNIDENTIFIABLE subspace Q. EXPLORATORY DESCRIPTOR ONLY: R_Q>0 does "
                      "NOT prove fabrication (E[Qd|Pd] is generally nonzero under coordinate correlation).",
           "note": "R_Q=0 means the reconstruction sets Q d_hat=0; since the Bayes posterior mean "
                   "E[Qd|Pd] is generally nonzero, R_Q=0 is a specific choice, NOT 'correct abstention'. "
                   "A calibrated score would standardize Q d_hat - E[Qd|Pd] by Sigma_{Q|P} (future work).",
           "configs": {}, "segments": list(SEGMENTS),
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization="raw mV segment samples",
                                   train_ids=tr_ids, test_ids=te_ids,
                                   extra={"metric": "null_space_dipolar_energy_ratio_exploratory"})}
    def _record_boot_ci(per_sample, rec_ids, B=500):
        """Record-level bootstrap CI of mean R_Q: resample unique records, pool their samples."""
        uniq = np.unique(rec_ids)
        if uniq.size < 5:
            return None
        idx_by_rec = {r: np.where(rec_ids == r)[0] for r in uniq}
        brng = np.random.default_rng(12345)
        means = []
        for _ in range(B):
            pick = brng.choice(uniq, uniq.size, replace=True)
            sel = np.concatenate([idx_by_rec[r] for r in pick])
            means.append(float(np.mean(per_sample[sel])))
        return [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)]

    for cname, obs in CONFIGS.items():
        oi = [LEAD_INDEX[l] for l in obs]
        row, row_ci = {}, {}
        for s in SEGMENTS:
            Xtr = np.asarray(tr[s][0], float)
            Xtr = Xtr[np.all(np.isfinite(Xtr), axis=1) & np.all(np.abs(Xtr) <= 10.0, axis=1)]
            Xte0, rid0 = np.asarray(te[s][0], float), np.asarray(te[s][1])
            m_te = np.all(np.isfinite(Xte0), axis=1) & np.all(np.abs(Xte0) <= 10.0, axis=1)
            Xte, rid = Xte0[m_te], rid0[m_te]
            if Xtr.shape[0] < 200 or Xte.shape[0] < 200:
                continue
            M, mu, _ = fit_dipolar_subspace(Xtr, rank=3)
            Yo_tr = Xtr[:, oi]; T1 = np.hstack([Yo_tr, np.ones((Yo_tr.shape[0], 1))])   # FIT on train
            Yo = Xte[:, oi]                                                             # APPLY to test
            recons = {"dipolar": reconstruct_dipolar(M, mu, obs, Yo.T).T,
                      "prior_mean": np.tile(mu, (Xte.shape[0], 1))}       # sets Q d_hat=0 (not Bayes-optimal)
            for nm, lam in (("ridge", 1.0), ("ols", 0.0)):
                A = T1.T @ T1 + lam * np.eye(T1.shape[1]); A[-1, -1] -= lam
                W = np.linalg.solve(A, T1.T @ Xtr); recons[nm] = Yo @ W[:-1] + W[-1]
            persamp = {nm: null_space_energy_per_sample(M, mu, obs, Lh) for nm, Lh in recons.items()}
            row[s] = {nm: round(float(np.mean(v)), 5) for nm, v in persamp.items()}
            row_ci[s] = {nm: _record_boot_ci(v, rid) for nm, v in persamp.items()}
        out["configs"][cname] = row
        out.setdefault("configs_ci", {})[cname] = row_ci     # record-bootstrap 95% CIs (uncertainty)
        print(f"[fab] {cname}: " + "  ".join(
            f"{s}:{{" + ",".join(f'{nm}={row[s][nm]:.3f}' for nm in row[s]) + "}" for s in row), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fabrication_audit.json").write_text(json.dumps(out, indent=2))
    print("[json] results/fabrication_audit.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
