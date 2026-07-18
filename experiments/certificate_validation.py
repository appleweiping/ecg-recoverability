"""Is the per-lead recoverability certificate OPERATIONAL? (Phase-1 headline evidence.)

We test whether the *theoretical* per-lead quantities computed from the NORM-train dipolar
subspace M_s -- identifiability eta_{s,l}(S), normalized eta_tilde, conditioning kappa_{s,l}(S),
and the prior-conditional expected ambiguity a_{s,l}(S) in mV -- predict the *measured* per-lead
reconstruction error on held-out fold-10 test records, across observed-lead configurations S,
segments s, target leads l, and reconstructors (dipolar / ridge / OLS).

CRITICAL SCOPING: the certificate is a statement about the DIPOLAR (rank-3) component of each
lead. We therefore measure the error of the *dipolar projection* of the target lead,
    e_l^T P_{M_s} (Lhat - Ltrue),   P_{M_s} = M_s M_s^+  (projector onto the 3-D dipole subspace),
not the raw full-lead error. This is exactly the quantity the certificate (and the minimax floor,
paper/theorem_floor.tex) bounds; scoring raw RMSE instead would let a reconstructor that exploits
predictable *non-dipolar* population structure (Tier-II) appear to beat an eta>0 floor it does not
actually violate.

Reports, per reconstructor: Spearman/Pearson of measured dipolar RMSE vs expected ambiguity (same
mV units; the y=x line is the certified floor a_l), Spearman vs eta_tilde, Spearman vs kappa
WITHIN the identifiable (eta~0) cells only, the eta=0 vs eta>0 dichotomy (AUC + median split), and
the floor-violation fraction (cells where measured error < a_l -- should be ~0 within CI). Nested
record + cell bootstrap CIs.

Output: results/certificate_validation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (
    LEAD_INDEX, fit_dipolar_subspace, reconstruct_dipolar, dipole_coord_cov,
    eta_per_lead, eta_normalized_per_lead, kappa_per_lead, expected_ambiguity_per_lead,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "Lead-I": ["I"],
    "Lead-II": ["II"],
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}
ETA_ZERO_TOL = 1e-3          # eta below this (numerically) = identifiable cell
NORMALIZATION = "raw mV segment samples; dipolar-projection per-lead error"


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(x[m], y[m]).correlation
    return None if not np.isfinite(r) else float(r)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    r = np.corrcoef(x[m], y[m])[0, 1]
    return None if not np.isfinite(r) else float(r)


def _auc(scores, labels):
    """AUC of `scores` predicting boolean `labels` (Mann-Whitney)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size); ranks[order] = np.arange(1, order.size + 1)
    auc = (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def _boot_ci(fn, seed, n=1000):
    """Bootstrap a scalar statistic fn(rng) -> value; returns [lo, hi] percentile CI."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng) for _ in range(n)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 20:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def _per_record_dipolar_rmse(err_l, rid):
    """Per-record RMSE of the (already lead-l, dipolar-projected) error, keyed by record id."""
    out = {}
    for r in np.unique(rid):
        e = err_l[rid == r]
        out[int(r)] = float(np.sqrt(np.mean(e ** 2)))
    return out


def run(n_train=1500, n_test=1500, max_per_record=40, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:n_train]
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    te_ids = rng.permutation(np.intersect1d(f10, norm10))[:n_test]

    print(f"[certval] collecting segment samples: train={len(tr_ids)} test={len(te_ids)}", flush=True)
    tr_seg = db.collect_all_segments_with_ids(tr_ids, rate=100, max_per_record=max_per_record,
                                              max_records=len(tr_ids), seed=seed)
    te_seg = db.collect_all_segments_with_ids(te_ids, rate=100, max_per_record=max_per_record,
                                              max_records=len(te_ids), seed=seed)

    cells = []
    for s in SEGMENTS:
        Xtr = np.asarray(tr_seg[s][0], float)
        Xtr = Xtr[np.all(np.isfinite(Xtr), axis=1) & np.all(np.abs(Xtr) <= 10.0, axis=1)]
        Xte, rid = np.asarray(te_seg[s][0], float), np.asarray(te_seg[s][1])
        ok = np.all(np.isfinite(Xte), axis=1) & np.all(np.abs(Xte) <= 10.0, axis=1)
        Xte, rid = Xte[ok], rid[ok]
        if Xtr.shape[0] < 200 or Xte.shape[0] < 200:
            continue

        M, mu, _ = fit_dipolar_subspace(Xtr, rank=3)
        Sig_d = dipole_coord_cov(M, mu, Xtr)
        P = M @ np.linalg.pinv(M)                                # (12,12) dipolar projector
        print(f"[certval] segment {s}: Ntr={Xtr.shape[0]} Nte={Xte.shape[0]}", flush=True)

        for cname, obs in CONFIGS.items():
            oi = [LEAD_INDEX[l] for l in obs]
            eta = eta_per_lead(M, obs); etn = eta_normalized_per_lead(M, obs)
            kap = kappa_per_lead(M, obs); amb = expected_ambiguity_per_lead(M, obs, Sig_d)

            # reconstructors S -> 12 (dipolar uses M_s; ridge/OLS fit on train segment samples)
            Yo_tr = Xtr[:, oi]; T1 = np.hstack([Yo_tr, np.ones((Yo_tr.shape[0], 1))])
            lin = {}
            for nm, lam in (("ridge", 1.0), ("ols", 0.0)):
                A = T1.T @ T1 + lam * np.eye(T1.shape[1]); A[-1, -1] -= lam
                W = np.linalg.solve(A, T1.T @ Xtr); lin[nm] = (W[:-1].T, W[-1])   # (12,|S|),(12,)

            Yo_te = Xte[:, oi]
            rec = {"dipolar": reconstruct_dipolar(M, mu, obs, Yo_te.T).T}          # (Nte,12)
            for nm, (W, b) in lin.items():
                rec[nm] = Yo_te @ W.T + b
            # dipolar-projection error per method: P (Lhat - Ltrue)
            derr = {nm: (P @ (rec[nm] - Xte).T).T for nm in rec}                   # (Nte,12)

            for l in LEADS:
                if l in obs:
                    continue
                li = LEAD_INDEX[l]
                rmse = {}
                for nm in rec:
                    pr = _per_record_dipolar_rmse(derr[nm][:, li], rid)
                    rmse[nm] = pr                                                  # {record_id: rmse}
                cells.append({
                    "config": cname, "segment": s, "lead": l, "observed": obs,
                    "eta": round(float(eta[li]), 5), "eta_normalized": round(float(etn[li]), 5),
                    "kappa": round(float(kap[li]), 4), "amb_mV": round(float(amb[li]), 5),
                    "eta_zero": bool(eta[li] < ETA_ZERO_TOL),
                    "measured_rmse_mV": {nm: round(float(np.mean(list(rmse[nm].values()))), 5) for nm in rec},
                    "n_records": len(next(iter(rmse.values()))),
                    "_perrec": {nm: rmse[nm] for nm in rec},                       # for bootstrap (dropped before write)
                })

    # ---------- correlations + floor per reconstructor ----------
    methods = ("dipolar", "ridge", "ols")
    corr = {}
    for nm in methods:
        amb = np.array([c["amb_mV"] for c in cells])
        etn = np.array([c["eta_normalized"] for c in cells])
        kap = np.array([c["kappa"] for c in cells])
        z = np.array([c["eta_zero"] for c in cells])
        y = np.array([c["measured_rmse_mV"][nm] for c in cells])
        big = y > np.median(y)
        # bootstrap correlation over CELLS
        def sp_amb(rng): idx = rng.integers(0, len(cells), len(cells)); return _spearman(amb[idx], y[idx])
        def sp_etn(rng): idx = rng.integers(0, len(cells), len(cells)); return _spearman(etn[idx], y[idx])
        kmask = z  # identifiable cells for the kappa test
        corr[nm] = {
            "spearman_amb_rmse": _spearman(amb, y), "spearman_amb_rmse_ci": _boot_ci(sp_amb, seed + 1),
            "pearson_amb_rmse": _pearson(amb, y),
            "spearman_etanorm_rmse": _spearman(etn, y), "spearman_etanorm_rmse_ci": _boot_ci(sp_etn, seed + 2),
            "spearman_kappa_rmse_within_identifiable": _spearman(kap[kmask], y[kmask]) if kmask.sum() >= 3 else None,
            "auc_eta_pos_predicts_large_error": _auc(y, ~z),   # eta>0 (not eta_zero) should score high error
            "median_rmse_eta0": round(float(np.median(y[z])), 5) if z.any() else None,
            "median_rmse_etapos": round(float(np.median(y[~z])), 5) if (~z).any() else None,
            # floor: measured dipolar error must sit on/above the ambiguity floor a_l
            "floor_violation_frac": round(float(np.mean(y < amb - 1e-6)), 4),
            "floor_gap_median_mV": round(float(np.median(y - amb)), 5),
        }
        print(f"[certval] {nm}: spearman(amb,rmse)={corr[nm]['spearman_amb_rmse']} "
              f"AUC(eta>0)={corr[nm]['auc_eta_pos_predicts_large_error']} "
              f"floor_viol={corr[nm]['floor_violation_frac']} "
              f"med_rmse eta0={corr[nm]['median_rmse_eta0']} etapos={corr[nm]['median_rmse_etapos']}", flush=True)

    for c in cells:
        c.pop("_perrec", None)
    out = {"n_train": int(len(tr_ids)), "n_test": int(len(te_ids)), "eta_zero_tol": ETA_ZERO_TOL,
           "metric": "per-lead dipolar-projection RMSE (mV): e_l^T (M_s M_s^+)(Lhat-Ltrue)",
           "configs": list(CONFIGS), "segments": list(SEGMENTS),
           "n_cells": len(cells), "correlations": corr, "cells": cells,
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization=NORMALIZATION,
                                   train_ids=tr_ids, test_ids=te_ids,
                                   extra={"metric": "dipolar_projection_per_lead_rmse_mV",
                                          "configs": list(CONFIGS)})}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "certificate_validation.json").write_text(json.dumps(out, indent=2))
    print("[json] results/certificate_validation.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--max-per-record", type=int, default=40)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test, max_per_record=args.max_per_record)
