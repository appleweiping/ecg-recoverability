"""Fair, like-for-like baseline comparison (per-timepoint WAVEFORM RMSE) with PAIRED CIs.

All methods use the IDENTICAL shared protocol split (train folds 1-7 / lambda-select fold 8 /
test fold 10) and the SAME test records as the neural baseline (protocol.py). For the linear
reconstructors (prior-mean, dipolar Tier-I, per-segment ridge) we evaluate the SAME
per-timepoint target-lead waveform RMSE the U-Net uses, save PER-RECORD errors, and compute
PAIRED record-bootstrap delta CIs (dipolar->ridge, ridge->U-Net, spanning->limb-6). Ridge
lambda is selected per config on fold 8. Before merging the U-Net JSON we ASSERT its lineage
(dataset / split / seed / targets / normalization) matches, so the merged table cannot silently
combine mismatched runs.

Output: results/fair_baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX, reconstruct_dipolar
from protocol import standard_split, fold8_ids, load_windows, NORMALIZATION

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
LAMBDAS = (0.1, 1.0, 10.0)
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _ridge_fit(X, oi, lam):
    """Ridge limb/obs -> all-12 from segment samples X (N,12). Returns (12,|S|),(12,)."""
    Xo = X[:, oi]
    X1 = np.hstack([Xo, np.ones((Xo.shape[0], 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[-1, -1] -= lam
    W = np.linalg.solve(A, X1.T @ X)                          # (|S|+1, 12)
    return W[:-1].T, W[-1]


def _load_eval(db, ids):
    """(sig (12,win), segidx) per kept record + kept ids, sharing protocol filtering."""
    X, kept, _ = load_windows(db, ids)
    seg = [db.segment_indices(X[i].T, fs=100) for i in range(X.shape[0])]
    return X, kept, seg


def _per_record(method, X, seg, models, ridge_w, oi, ti):
    """Per-record per-timepoint target RMSE (mV) -> {seg: array(n_rec) with nan where absent}."""
    out = {s: np.full(X.shape[0], np.nan) for s in SEGMENTS}
    for r in range(X.shape[0]):
        sig = X[r]                                            # (12, win) raw mV
        for s in SEGMENTS:
            idx = seg[r].get(s); m = models.get(s)
            if idx is None or idx.size < 8 or m is None:
                continue
            true = sig[ti][:, idx]                            # (|tgt|, Tseg)
            yS = sig[oi][:, idx]                              # (|S|, Tseg)
            if method == "prior_mean":
                pred = np.tile(m.mu[ti][:, None], (1, idx.size))
            elif method == "dipolar":
                pred = reconstruct_dipolar(m.M, m.mu, oi, yS)[ti]   # oi = lead indices
            else:  # ridge
                W, b = ridge_w[s]; pred = (W @ yS + b[:, None])[ti]
            out[s][r] = np.sqrt(np.mean((pred - true) ** 2))
    return out


def _select_lambda(seg_tr, X8, seg8, oi, ti):
    """Pick lambda minimizing pooled per-timepoint target RMSE on fold-8 records."""
    best, best_err = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        rw = {s: _ridge_fit(seg_tr[s], oi, lam) for s in SEGMENTS if seg_tr[s].shape[0] >= 100}
        errs = []
        for r in range(X8.shape[0]):
            sig = X8[r]
            for s in SEGMENTS:
                idx = seg8[r].get(s)
                if idx is None or idx.size < 8 or s not in rw:
                    continue
                W, b = rw[s]; pred = (W @ sig[oi][:, idx] + b[:, None])[ti]
                errs.append(np.sqrt(np.mean((pred - sig[ti][:, idx]) ** 2)))
        e = float(np.mean(errs)) if errs else np.inf
        if e < best_err:
            best_err, best = e, lam
    return best


def _paired_delta_ci(err_a, err_b, ids_a, ids_b, n=2000, seed=7):
    """Paired record-bootstrap CI of mean(err_b - err_a) over COMMON record ids."""
    da = {i: e for i, e in zip(ids_a, err_a)}
    db_ = {i: e for i, e in zip(ids_b, err_b)}
    common = [i for i in ids_a if i in db_]
    if len(common) < 10:
        return None
    d = np.array([db_[i] - da[i] for i in common])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n_pairs": len(common), "mean_delta_mV": round(float(d.mean()), 4),
            "delta_ci": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def run(n_train=4000, n_test=800, seed=0):    # match neural_baseline for shared train ids
    db = PTBXL()
    tr_ids, te_ids = standard_split(db, n_train, n_test, seed=seed)
    seg_tr = db.collect_all_segments(tr_ids, rate=100, max_per_record=40,
                                     max_records=n_train, seed=seed)
    models = fit_segment_models(seg_tr)
    X8, _, seg8 = _load_eval(db, fold8_ids(db, cap=200, seed=seed))
    Xte, te_kept, segte = _load_eval(db, te_ids)

    out = {"n_test": int(Xte.shape[0]), "protocol": "per-timepoint waveform RMSE (mV), shared split",
           "lambdas_grid": list(LAMBDAS), "configs": {},
           "lineage": lineage.make(db, seed=seed, targets=["V2", "V4", "V6"],
                                   normalization=NORMALIZATION, train_ids=tr_ids, test_ids=te_kept)}
    methods = ("prior_mean", "dipolar", "ridge")
    for cname, obs in CONFIGS.items():
        oi = [LEAD_INDEX[l] for l in obs]
        tgt = [l for l in ("V2", "V4", "V6") if l not in obs] or ["V2", "V4", "V6"]
        ti = [LEAD_INDEX[l] for l in tgt]
        lam = _select_lambda(seg_tr, X8, seg8, oi, ti)
        ridge_w = {s: _ridge_fit(seg_tr[s], oi, lam) for s in SEGMENTS if seg_tr[s].shape[0] >= 100}
        pr = {mth: _per_record(mth, Xte, segte, models, ridge_w, oi, ti) for mth in methods}
        cfg = {"ridge_lambda": lam}
        for mth in methods:
            cfg[mth] = {}
            for s in SEGMENTS:
                arr = pr[mth][s]; ok = np.isfinite(arr)
                if ok.sum() < 5:
                    continue
                e = arr[ok]; br = np.random.default_rng(seed + 4)
                ci = [float(np.percentile([e[br.integers(0, e.size, e.size)].mean() for _ in range(500)], q))
                      for q in (2.5, 97.5)]
                cfg[mth][s] = {"rmse_mV": round(float(e.mean()), 4),
                               "rmse_ci": [round(ci[0], 4), round(ci[1], 4)],
                               "per_record": {"ids": [int(i) for i in te_kept[ok]],
                                              "rmse": [round(float(x), 5) for x in e]}}
        out["configs"][cname] = cfg

    # ---- merge U-Net (assert lineage first) + paired deltas ----
    nb = RESULTS / "neural_baseline.json"
    neural = None
    if nb.exists():
        neural = json.loads(nb.read_text())
        if "lineage" in neural and "lineage" in out:
            lineage.assert_consistent(out["lineage"], neural["lineage"],
                                      label_a="fair_baselines", label_b="neural_baseline")
        for cname, segd in neural.get("configs", {}).items():
            out["configs"].setdefault(cname, {})["unet"] = {
                s: {k: v for k, v in d.items() if k != "per_record"} for s, d in segd.items()}

    # paired delta CIs per config/segment
    def perrec(cname, mth, s):
        d = out["configs"].get(cname, {}).get(mth, {}).get(s)
        if not d or "per_record" not in d:
            return None
        return np.array(d["per_record"]["rmse"]), np.array(d["per_record"]["ids"])

    deltas = {}
    for cname in CONFIGS:
        for s in SEGMENTS:
            dip = perrec(cname, "dipolar", s); rid = perrec(cname, "ridge", s)
            d = {}
            if dip and rid:
                d["dipolar_to_ridge"] = _paired_delta_ci(dip[0], rid[0], dip[1], rid[1])
            if rid and neural is not None:
                nd = neural.get("configs", {}).get(cname, {}).get(s, {}).get("per_record")
                if nd:
                    d["ridge_to_unet"] = _paired_delta_ci(rid[0], np.array(nd["rmse"]),
                                                          rid[1], np.array(nd["ids"]))
            if d:
                deltas[f"{cname}/{s}"] = d
    # spanning -> limb-6 (per method, QRS), paired on common records
    for mth in ("ridge", "unet"):
        for s in SEGMENTS:
            sp = perrec("{I,II,V1,V3,V5}", mth, s) if mth != "unet" else _unet_pr(neural, "{I,II,V1,V3,V5}", s)
            lb = perrec("limb-6", mth, s) if mth != "unet" else _unet_pr(neural, "limb-6", s)
            if sp and lb:
                deltas.setdefault(f"spanning_to_limb6/{mth}/{s}", {})["delta"] = \
                    _paired_delta_ci(sp[0], lb[0], sp[1], lb[1])
    out["paired_deltas"] = deltas

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fair_baselines.json").write_text(json.dumps(out, indent=2))
    for cname in CONFIGS:
        c = out["configs"][cname]
        print(f"[{cname}] lam={c.get('ridge_lambda')} " + " ".join(
            f"{m}:QRS={c.get(m, {}).get('QRS', {}).get('rmse_mV')}" for m in ("prior_mean", "dipolar", "ridge", "unet")), flush=True)
    print("[json] results/fair_baselines.json", flush=True)


def _unet_pr(neural, cname, s):
    if neural is None:
        return None
    d = neural.get("configs", {}).get(cname, {}).get(s, {}).get("per_record")
    if not d:
        return None
    return np.array(d["rmse"]), np.array(d["ids"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)   # match neural_baseline default
    ap.add_argument("--n-test", type=int, default=800)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
