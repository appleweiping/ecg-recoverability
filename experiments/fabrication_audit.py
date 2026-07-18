"""Null-space dipolar FABRICATION audit (Phase-1 differentiator; ground-truth-free).

A reconstruction can only be *determined* by the observed leads in the observed dipole directions
P_obs = M_{s,S}^+ M_{s,S}. Whatever a reconstruction asserts in the UNOBSERVED dipole directions
Q = I - P_obs is content the observation cannot possibly fix -- if it is nonzero the method is
FABRICATING (hallucinating) in a provably unidentifiable subspace. This needs no ground truth.

For a reconstruction Lhat under config S, the fitted dipole coordinate is d_hat = M_s^+ (Lhat-mu_s),
and the fabrication energy ratio is
    phi_s(S) = || M_s Q d_hat ||_2^2 / || M_s d_hat ||_2^2  in [0,1],
the fraction of the reconstruction's dipolar energy that lies in the unidentifiable subspace. An
HONEST reconstructor (the Bayes posterior mean, which sets the unobserved coordinate to its prior
mean) has phi ~ 0 -- it abstains where the data is silent; a reconstructor that confidently fills
the null space has phi > 0. We audit dipolar / ridge / OLS / prior-mean here (CPU); the diffusion
model's phi(w) vs guidance is measured in the diffusion pipeline. Complements the accuracy view
(certificate_validation.py): phi is what the method *asserts* it knows but cannot.

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


def fabrication_ratio(M, mu, obs, Lhat):
    """phi = ||M_s Q d_hat||^2 / ||M_s d_hat||^2 per sample, mean over samples. Lhat: (N,12)."""
    oi = [LEAD_INDEX[l] for l in obs]
    Mp = np.linalg.pinv(M)                                   # (3,12)
    P = np.linalg.pinv(M[oi]) @ M[oi]                        # (3,3) observed-dipole projector
    Q = np.eye(3) - P
    dhat = (Lhat - mu) @ Mp.T                                # (N,3)
    num = np.sum((dhat @ Q.T @ M.T) ** 2, axis=1)           # ||M Q d_hat||^2
    den = np.sum((dhat @ M.T) ** 2, axis=1) + 1e-12         # ||M d_hat||^2
    return float(np.mean(num / den))


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
    out = {"metric": "phi = ||M_s Q d_hat||^2 / ||M_s d_hat||^2, fraction of reconstruction dipolar "
                      "energy in the UNIDENTIFIABLE subspace (0 = honest/abstains, >0 = fabricates)",
           "note": "prior-mean = Bayes posterior mean (abstains) -> phi~0; dipolar stays in observed "
                   "row space -> phi~0; ridge/OLS put the learned conditional mean in the null space.",
           "configs": {}, "segments": list(SEGMENTS),
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization="raw mV segment samples",
                                   train_ids=tr_ids, test_ids=te_ids,
                                   extra={"metric": "null_space_dipolar_fabrication_ratio"})}
    for cname, obs in CONFIGS.items():
        oi = [LEAD_INDEX[l] for l in obs]
        row = {}
        for s in SEGMENTS:
            Xtr = np.asarray(tr[s][0], float)
            Xtr = Xtr[np.all(np.isfinite(Xtr), axis=1) & np.all(np.abs(Xtr) <= 10.0, axis=1)]
            Xte = np.asarray(te[s][0], float)
            Xte = Xte[np.all(np.isfinite(Xte), axis=1) & np.all(np.abs(Xte) <= 10.0, axis=1)]
            if Xtr.shape[0] < 200 or Xte.shape[0] < 200:
                continue
            M, mu, _ = fit_dipolar_subspace(Xtr, rank=3)
            Yo_tr = Xtr[:, oi]; T1 = np.hstack([Yo_tr, np.ones((Yo_tr.shape[0], 1))])   # FIT on train
            Yo = Xte[:, oi]                                                             # APPLY to test
            recons = {"dipolar": reconstruct_dipolar(M, mu, obs, Yo.T).T,
                      "prior_mean": np.tile(mu, (Xte.shape[0], 1))}       # abstains: outputs prior mean
            for nm, lam in (("ridge", 1.0), ("ols", 0.0)):
                A = T1.T @ T1 + lam * np.eye(T1.shape[1]); A[-1, -1] -= lam
                W = np.linalg.solve(A, T1.T @ Xtr); recons[nm] = Yo @ W[:-1] + W[-1]
            row[s] = {nm: round(fabrication_ratio(M, mu, obs, Lh), 5) for nm, Lh in recons.items()}
        out["configs"][cname] = row
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
