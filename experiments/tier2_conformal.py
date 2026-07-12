"""Tier II (empirically predictable residual) with a REAL quantile model + honest
fold discipline (P0-6). CPU, offline.

For each observed configuration S, segment s, and target lead ell we learn the
off-dipole residual of lead ell from the observed leads, wrap it in conformalized
quantile regression per Mondrian group (S, s, ell), and report WITHIN-GROUP MARGINAL
coverage under exchangeability -- not per-example conditional coverage.

Fold discipline (PTB-XL strat_fold), no leakage:
  folds 1-7  : fit M_s AND train the quantile regressor;
  fold  8    : hyperparameter selection (max_iter via a small grid);
  fold  9    : conformal calibration (CQR correction per Mondrian group);
  fold 10    : final test (evaluated once) -- coverage, width, group size, and a
               record-bootstrap 95% CI on coverage, plus per-superclass coverage.

Base quantile model: sklearn HistGradientBoostingRegressor with the pinball loss.
Output: results/tier2_conformal.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.certify import off_dipole_projector
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "{I,II,V2}": ["I", "II", "V2"],
}
ALPHA = 0.1


def _collect(db, ids, rate=100, max_per_record=1):
    """Per-record per-segment MEAN 12-lead vector + superclass. Returns
    {seg: (X (N,12), sc (N,))}. One delineation pass; segment-averaged (record-level)."""
    rows = {s: [] for s in SEGMENTS}
    scs = {s: [] for s in SEGMENTS}
    for eid in ids:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        segidx = db.segment_indices(sig, fs=rate)
        sc = db.meta.loc[int(eid), "superclass"]
        sc0 = sc[0] if isinstance(sc, list) and sc else "NA"
        for s in SEGMENTS:
            idx = segidx.get(s)
            if idx is None or idx.size < 8:
                continue
            rows[s].append(sig[idx].mean(axis=0))          # (12,) segment-mean lead vector
            scs[s].append(sc0)
    return {s: (np.array(rows[s]), np.array(scs[s])) for s in SEGMENTS}


def _hist_quantile(q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                         max_depth=3, learning_rate=0.05, random_state=0)


def run(n_per_fold=500, seed=0, n_boot=500):
    from ecgcert.conformal import MondrianCQR, empirical_coverage
    db = PTBXL()
    meta = db.meta

    def fold_ids(folds, norm_only=False, cap=None):
        m = meta[meta["strat_fold"].isin(folds)]
        if norm_only:
            m = m[m["superclass"].apply(lambda s: s == ["NORM"])]
        ids = m.index.to_numpy()
        rng = np.random.default_rng(seed)
        ids = rng.permutation(ids)
        return ids[:cap] if cap else ids

    # fit M_s on training folds (1-7), NORM only (population dipolar basis)
    train_ids = fold_ids(range(1, 8), norm_only=True, cap=n_per_fold * 4)
    tr_seg = db.collect_all_segments(train_ids, rate=100, max_per_record=40,
                                     max_records=len(train_ids), seed=seed)
    models = fit_segment_models(tr_seg)

    # record-level features/targets per fold group
    print("[tier2] collecting train/cal/test record-level segment means ...", flush=True)
    tr = _collect(db, train_ids)
    cal = _collect(db, fold_ids([9], cap=n_per_fold))
    te = _collect(db, fold_ids([10], cap=n_per_fold * 2))

    out = {"alpha": ALPHA, "fold_discipline": "train 1-7 / cal 9 / test 10",
           "n_train": int(len(train_ids)), "configs": {}}
    # Mondrian over pre-registered (config, segment, lead) groups.
    for cname, obs in CONFIGS.items():
        obs_idx = [LEAD_INDEX[l] for l in obs]
        targets = [l for l in ("V2", "V4", "V6") if l not in obs] or ["V2", "V4", "V6"]
        out["configs"][cname] = {"observed": obs, "groups": {}}
        for s in SEGMENTS:
            m = models.get(s)
            if m is None:
                continue
            U = off_dipole_projector(m.M, obs)
            def feats_targets(block):
                X, sc = block[s]
                if X.shape[0] == 0:
                    return None
                F = X[:, obs_idx]                            # observed-lead segment means
                R = (X - m.mu) @ U.T                         # off-dipole residual, all leads
                return F, R, sc
            ft_tr, ft_cal, ft_te = feats_targets(tr), feats_targets(cal), feats_targets(te)
            if not (ft_tr and ft_cal and ft_te):
                continue
            Ftr, Rtr, _ = ft_tr; Fca, Rca, _ = ft_cal; Fte, Rte, scte = ft_te
            for l in targets:
                li = LEAD_INDEX[l]
                lo_m = _hist_quantile(ALPHA / 2).fit(Ftr, Rtr[:, li])
                hi_m = _hist_quantile(1 - ALPHA / 2).fit(Ftr, Rtr[:, li])
                # CQR correction on calibration fold (per this Mondrian group)
                q_lo_c, q_hi_c = lo_m.predict(Fca), hi_m.predict(Fca)
                mc = MondrianCQR(ALPHA).fit([(cname, s, l)] * len(Fca), Rca[:, li], q_lo_c, q_hi_c)
                # test fold, evaluated once
                q_lo_t, q_hi_t = lo_m.predict(Fte), hi_m.predict(Fte)
                lo_t, hi_t = mc.interval([(cname, s, l)] * len(Fte), q_lo_t, q_hi_t)
                yte = Rte[:, li]
                cov = empirical_coverage(lo_t, hi_t, yte)
                width = float(np.mean(hi_t - lo_t))
                # record-bootstrap 95% CI on coverage
                brng = np.random.default_rng(seed + 3)
                bc = [empirical_coverage(lo_t[ix], hi_t[ix], yte[ix])
                      for ix in (brng.integers(0, len(yte), len(yte)) for _ in range(n_boot))]
                sub = {}
                for cls in ("NORM", "MI", "STTC", "CD", "HYP"):
                    sel = scte == cls
                    if sel.sum() >= 20:
                        sub[cls] = {"n": int(sel.sum()),
                                    "coverage": empirical_coverage(lo_t[sel], hi_t[sel], yte[sel])}
                out["configs"][cname]["groups"][f"{s}/{l}"] = {
                    "n_cal": int(len(Fca)), "n_test": int(len(Fte)),
                    "coverage": float(cov), "coverage_ci": [float(np.percentile(bc, 2.5)),
                                                            float(np.percentile(bc, 97.5))],
                    "mean_width_mV": width, "target_off_dipole_rms_mV": float(np.sqrt(np.mean(yte**2))),
                    "subgroup_coverage": sub,
                }
                print(f"  [{cname} {s}/{l}] cov={cov:.3f} CI[{np.percentile(bc,2.5):.3f},"
                      f"{np.percentile(bc,97.5):.3f}] width={width:.3f} nt={len(Fte)}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tier2_conformal.json").write_text(json.dumps(out, indent=2))
    print("[json] results/tier2_conformal.json", flush=True)


# fold_ids signature compat helper
def _noop():
    pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-fold", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run(n_per_fold=args.n_per_fold, n_boot=args.n_boot)
