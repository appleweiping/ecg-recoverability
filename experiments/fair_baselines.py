"""Fair, like-for-like baseline comparison (per-timepoint WAVEFORM RMSE).

The classical baselines in baselines_physics.py were evaluated on segment-MEAN vectors,
while neural_baseline.py evaluated per-TIMEPOINT waveforms -- not comparable. This script
evaluates ALL linear reconstructors (prior-mean, dipolar Tier-I, per-segment ridge) with the
IDENTICAL per-timepoint protocol as the U-Net (same test records, same segment indices, RMSE
over target leads x timepoints), and merges the neural result, so the paper's baseline table
is apples-to-apples.

Output: results/fair_baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX, reconstruct_dipolar

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _ridge_fit(X, Y, lam=1.0):
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[-1, -1] -= lam
    W = np.linalg.solve(A, X1.T @ Y)
    return W[:-1].T, W[-1]                                # (12,|S|),(12,)


def run(n_train=3000, n_test=800, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm10))[:n_test]

    seg = db.collect_all_segments(tr, rate=100, max_per_record=40, max_records=n_train, seed=seed)
    models = fit_segment_models(seg)

    # load test waveforms + segment indices (same protocol as the U-Net eval)
    sigs, segidxs = [], []
    for eid in test_ids:
        try:
            s = db.signal(int(eid), rate=100)[:1000]
        except Exception:
            continue
        if s.shape[0] == 1000 and np.all(np.isfinite(s)):
            sigs.append(s.astype(float)); segidxs.append(db.segment_indices(s, fs=100))

    out = {"n_test": len(sigs), "protocol": "per-timepoint waveform RMSE (mV)", "configs": {}}
    for cname, obs in CONFIGS.items():
        oi = [LEAD_INDEX[l] for l in obs]
        tgt = [l for l in ("V2", "V4", "V6") if l not in obs] or ["V2", "V4", "V6"]
        ti = [LEAD_INDEX[l] for l in tgt]
        # fit per-segment ridge limb->all on segment means (coefficients apply per-timepoint)
        rw = {}
        for s in SEGMENTS:
            X = seg[s]
            if X.shape[0] >= 100:
                rw[s] = _ridge_fit(X[:, oi], X)
        per = {m: {s: [] for s in SEGMENTS} for m in ("prior_mean", "dipolar", "ridge")}
        for sig, sgi in zip(sigs, segidxs):
            for s in SEGMENTS:
                idx = sgi.get(s); m = models.get(s)
                if idx is None or idx.size < 8 or m is None:
                    continue
                yS = sig[idx][:, oi].T                       # (|S|, Tseg) per-timepoint
                true = sig[idx].T[ti]                        # (|tgt|, Tseg)
                # prior mean (constant)
                pm = np.tile(m.mu[ti][:, None], (1, idx.size))
                per["prior_mean"][s].append(np.sqrt(np.mean((pm - true) ** 2)))
                # dipolar per-timepoint
                dp = reconstruct_dipolar(m.M, m.mu, obs, yS)[ti]
                per["dipolar"][s].append(np.sqrt(np.mean((dp - true) ** 2)))
                # ridge per-timepoint
                if s in rw:
                    W, b = rw[s]; rd = (W @ yS + b[:, None])[ti]
                    per["ridge"][s].append(np.sqrt(np.mean((rd - true) ** 2)))
        cfg = {}
        for mth, segd in per.items():
            cfg[mth] = {}
            for s, vals in segd.items():
                if vals:
                    e = np.array(vals); br = np.random.default_rng(seed + 4)
                    ci = [float(np.percentile([e[br.integers(0, e.size, e.size)].mean() for _ in range(500)], q))
                          for q in (2.5, 97.5)]
                    cfg[mth][s] = {"rmse_mV": round(float(e.mean()), 4), "rmse_ci": [round(ci[0], 4), round(ci[1], 4)]}
        out["configs"][cname] = cfg
        print(f"[{cname}] " + " | ".join(
            f"{mth} QRS={cfg[mth].get('QRS', {}).get('rmse_mV')}" for mth in per), flush=True)

    # merge neural baseline (same protocol)
    nb = RESULTS / "neural_baseline.json"
    if nb.exists():
        nd = json.loads(nb.read_text())
        for cname, segd in nd.get("configs", {}).items():
            out["configs"].setdefault(cname, {})["unet"] = segd
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fair_baselines.json").write_text(json.dumps(out, indent=2))
    print("[json] results/fair_baselines.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=800)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
