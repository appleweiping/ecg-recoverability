"""Baselines + physics-vs-PCA validation for the target-specific recoverability paper.

CPU (runs well on the many-core box). Two parts, both with record-bootstrap 95% CIs and
per-superclass breakdown (NORM/MI/STTC/CD/HYP):

A. RECONSTRUCTION BASELINES. For each config and target lead, reconstruct the segment
   and report RMSE/MAE (mV) of the unobserved leads for:
     - prior_mean        : population segment mean (trivial lower reference);
     - inverse_dower     : classical VCG inverse-Dower map fit to the observed leads;
     - dipolar           : Tier-I linear dipolar recovery (mu + M_s M_{s,S}^+ (y-mu));
     - ridge_perseg      : per-segment ridge from observed leads (Tier I+II);
     - ols_perseg        : per-segment least squares from observed leads;
     - ols_pooled        : one OLS over all segments (the weak pooled baseline).
   (A representative neural baseline -- arbitrary-mask U-Net -- is a separate GPU item;
   see experiments/neural_baseline.py.)

B. PHYSICS vs PCA. On REAL data, principal angles between the estimated per-segment
   dipolar subspace M_s and the classical inverse-Dower column space, with bootstrap CIs
   and per-superclass stability -- to decide whether to call M_s a "physical cardiac
   dipole" or an "empirical rank-3 spatial subspace".

Output: results/baselines_physics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

from ecgcert.data import PTBXL
from ecgcert.physics import (LEADS, LEAD_INDEX, fit_dipolar_subspace, reconstruct_dipolar,
                             inverse_dower_matrix, lead_transform_T)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}
SUPER = ("NORM", "MI", "STTC", "CD", "HYP")


def _ridge(X, Y, lam=1.0):
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    d = X1.shape[1]
    A = X1.T @ X1 + lam * np.eye(d); A[-1, -1] -= lam
    W = np.linalg.solve(A, X1.T @ Y)
    return W[:-1], W[-1]


def _collect(db, ids, rate=100):
    """Per-record per-segment mean 12-lead vector + superclass."""
    out = {s: {"X": [], "sc": []} for s in SEGMENTS}
    for eid in ids:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        seg = db.segment_indices(sig, fs=rate)
        sc = db.meta.loc[int(eid), "superclass"]; sc0 = sc[0] if isinstance(sc, list) and sc else "NA"
        for s in SEGMENTS:
            idx = seg.get(s)
            if idx is None or idx.size < 8:
                continue
            v = sig[idx].mean(0)
            if np.all(np.isfinite(v)) and np.all(np.abs(v) < 10):
                out[s]["X"].append(v); out[s]["sc"].append(sc0)
    return {s: {"X": np.array(o["X"]), "sc": np.array(o["sc"])} for s, o in out.items()}


def _boot_ci(vals, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(vals)
    if v.size == 0:
        return [None, None]
    bs = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run(n_train=3000, n_test=1500, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm_tr = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))[:n_train]
    test = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())[:n_test]
    tr = _collect(db, norm_tr); te = _collect(db, test)

    Dcols = lead_transform_T() @ inverse_dower_matrix()      # (12,3) inverse-Dower dipole map
    from ecgcert import lineage
    out = {"n_train": int(len(norm_tr)), "part_A_baselines": {}, "part_B_physics_vs_pca": {},
           "lineage": lineage.make(db, seed=seed, targets=["V2", "V4", "V6"], normalization="raw mV",
                                   train_ids=norm_tr, test_ids=test,
                                   extra={"experiment": "classical baselines + physics-vs-PCA subspace angles"})}

    for s in SEGMENTS:
        Xtr, Xte, scte = tr[s]["X"], te[s]["X"], te[s]["sc"]
        if Xtr.shape[0] < 200 or Xte.shape[0] < 100:
            continue
        M, mu, evr = fit_dipolar_subspace(Xtr, rank=3)
        # ---- Part B: physics vs PCA ----
        ang = np.degrees(subspace_angles(M, Dcols))
        bang = []
        brng = np.random.default_rng(seed + 5)
        for _ in range(300):
            Mb, _, _ = fit_dipolar_subspace(Xtr[brng.integers(0, Xtr.shape[0], Xtr.shape[0])], rank=3)
            bang.append(np.degrees(subspace_angles(Mb, Dcols)))
        bang = np.array(bang)
        bmax = bang.max(axis=1)                            # per-bootstrap LARGEST angle
        bmin = bang.min(axis=1)                            # per-bootstrap SMALLEST angle
        out["part_B_physics_vs_pca"][s] = {
            "principal_angles_deg": [round(float(a), 2) for a in np.sort(ang)[::-1]],
            "max_angle_deg": round(float(ang.max()), 2),
            "max_angle_ci": [round(float(np.percentile(bmax, 2.5)), 2),
                             round(float(np.percentile(bmax, 97.5)), 2)],
            "min_angle_deg": round(float(ang.min()), 2),
            "min_angle_ci": [round(float(np.percentile(bmin, 2.5)), 2),
                             round(float(np.percentile(bmin, 97.5)), 2)],
            "dipolar_fraction": round(float(evr[:3].sum()), 3),
        }
        # ---- Part A: baselines ----
        out["part_A_baselines"][s] = {}
        for cname, obs in CONFIGS.items():
            oi = [LEAD_INDEX[l] for l in obs]
            tgt = [l for l in LEADS if l not in set(obs)]
            ti = [LEAD_INDEX[l] for l in tgt]
            Ftr, Fte = Xtr[:, oi], Xte[:, oi]
            preds = {}
            preds["prior_mean"] = np.tile(mu[ti], (Xte.shape[0], 1))
            # inverse-Dower: fit dipole coords from observed rows of Dcols
            Dobs = Dcols[oi]
            dcoef = np.linalg.pinv(Dobs) @ (Fte - mu[oi]).T          # (3, N)
            preds["inverse_dower"] = (mu[:, None] + Dcols @ dcoef).T[:, ti]
            preds["dipolar"] = np.array([reconstruct_dipolar(M, mu, obs, Fte[k])[ti]
                                         for k in range(Fte.shape[0])])
            Wr, br = _ridge(Ftr, Xtr[:, ti]); preds["ridge_perseg"] = Fte @ Wr + br
            Wo, bo = _ridge(Ftr, Xtr[:, ti], lam=1e-6); preds["ols_perseg"] = Fte @ Wo + bo
            out["part_A_baselines"][s][cname] = {}
            for name, P in preds.items():
                err = P - Xte[:, ti]
                rmse = np.sqrt((err ** 2).mean(1))                  # per-record RMSE over target leads
                mae = np.abs(err).mean(1)
                sub = {c: round(float(rmse[scte == c].mean()), 4) for c in SUPER if (scte == c).sum() >= 20}
                out["part_A_baselines"][s][cname][name] = {
                    "rmse_mV": round(float(rmse.mean()), 4), "rmse_ci": _boot_ci(rmse),
                    "mae_mV": round(float(mae.mean()), 4), "subgroup_rmse": sub}
        print(f"[{s}] physics max-angle={out['part_B_physics_vs_pca'][s]['max_angle_deg']} "
              f"dip={out['part_B_physics_vs_pca'][s]['dipolar_fraction']}", flush=True)
        for cname in CONFIGS:
            r = out["part_A_baselines"][s][cname]
            print(f"    {cname}: " + " ".join(f"{k}={v['rmse_mV']}" for k, v in r.items()), flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines_physics.json").write_text(json.dumps(out, indent=2))
    print("[json] results/baselines_physics.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
