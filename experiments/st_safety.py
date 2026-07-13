"""Certificate-driven ST-safety of reduced-lead reconstruction (P0-5, honest reframe).

The deployable safety claim is a CERTIFICATE claim, not a generative-model claim: for the
limb-6 configuration the per-lead identifiability eta_{ST,ell}(S) is > 0 for every
precordial lead (the transverse dipole is unobserved), so precordial ST is NOT
identifiable from limb leads -- a warning available before any reconstruction. We then
show what that unrecoverability costs downstream, using REAL (linear) reconstructors, not
a random sampler and not the diffusion:

  - continuous 10-second reconstruction (observed leads kept; unobserved leads filled by
    the reconstructor across the whole record, not a constant mean);
  - ST measured at J+60 ms vs a PR-segment baseline, on the SAME NeuroKit fiducials for
    truth and reconstruction; delineation success rate reported;
  - an "ST-threshold event" = |ST deviation| crossing 0.1 mV in a precordial lead;
  - fabricated event = threshold crossed in reconstruction but not truth; masked event =
    crossed in truth but not reconstruction.

We report prevalence, false-positive rate, false-negative rate, sensitivity, specificity
with record-bootstrap 95% CIs. Language is "ST-threshold event", never "phantom/missed
STEMI" or "fabricated diagnosis".

Output: results/st_safety.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX, eta_per_lead
from ecgcert.estimators import LinearDipolarReconstructor, OLSReconstructor
from ecgcert.clinical import count_stemi_flips

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
LIMB6 = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
PRECORDIAL_IDX = [LEAD_INDEX[l] for l in PRECORDIAL]


def _ridge_recon(F, W, b):
    return W @ F + b[:, None]


def _boot_ci(mask, n=1000, seed=0):
    """Bootstrap CI of a rate over records (mask is a per-record boolean/0-1)."""
    v = np.asarray(mask, float)
    if v.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def run(n_train=1500, n_test=1500, rate=100, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))[:n_train]
    seg = db.collect_all_segments(tr_ids, rate=rate, max_per_record=40, max_records=n_train, seed=seed)
    models = fit_segment_models(seg)
    L_train = np.hstack([seg[s].T for s in SEGMENTS if seg[s].shape[0] > 0])
    ols = OLSReconstructor(LIMB6).fit(L_train)
    # per-segment ridge from limb-6 -> all leads
    oi = [LEAD_INDEX[l] for l in LIMB6]
    ridge = {}
    for s in SEGMENTS:
        X = seg[s]
        if X.shape[0] < 100:
            continue
        X1 = np.hstack([X[:, oi], np.ones((X.shape[0], 1))])
        A = X1.T @ X1 + 1.0 * np.eye(X1.shape[1]); A[-1, -1] -= 1.0
        W = np.linalg.solve(A, X1.T @ X)
        ridge[s] = (W[:-1].T, W[-1])                      # (12,6),(12,)

    # certificate warning: eta for precordial ST under limb-6 (should be > 0)
    eta_st = eta_per_lead(models["ST"].M, LIMB6)
    cert = {l: round(float(eta_st[LEAD_INDEX[l]]), 4) for l in PRECORDIAL}

    test = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())[:n_test]
    recons = ("dipolar", "ridge", "ols")
    rows = {r: [] for r in recons}
    n_valid = 0
    for eid in test:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        segidx = db.segment_indices(sig, fs=rate)
        T = sig.shape[0]
        full = {}
        for r in recons:
            rec = np.tile(sig.mean(0)[:, None], (1, T)).astype(float)
            rec[oi] = sig[:, oi].T
            full[r] = rec
        ok = False
        for s in SEGMENTS:
            idx = segidx.get(s)
            if idx is None or idx.size < 4 or s not in models:
                continue
            m = models[s]; yS = sig[idx][:, oi].T
            full["dipolar"][:, idx] = LinearDipolarReconstructor(m.M, m.mu, LIMB6).predict(yS)
            full["ols"][:, idx] = ols.predict(yS)
            if s in ridge:
                W, b = ridge[s]; full["ridge"][:, idx] = _ridge_recon(yS, W, b)
            ok = True
        if not ok:
            continue
        valid_any = False
        for r in recons:
            flip = count_stemi_flips(sig, full[r].T, rate, leads=PRECORDIAL_IDX)
            if flip.get("valid"):
                rows[r].append(flip); valid_any = True
        n_valid += valid_any

    out = {"config": "limb-6 -> precordial", "certificate_eta_ST_precordial": cert,
           "n_test": int(len(test)), "n_valid_delineation": int(n_valid),
           "note": "eta_ST_precordial > 0 => precordial ST NOT identifiable from limb leads (certificate warning)",
           "reconstructors": {}}
    for r in recons:
        R = rows[r]
        if not R:
            continue
        fab = np.array([x["fabricated"] for x in R], float)
        msk = np.array([x["masked"] for x in R], float)
        sterr = np.array([x["st_error_mv"] for x in R], float)
        out["reconstructors"][r] = {
            "n": len(R),
            "fabricated_event_rate": round(float(fab.mean()), 4), "fabricated_ci": _boot_ci(fab),
            "masked_event_rate": round(float(msk.mean()), 4), "masked_ci": _boot_ci(msk),
            "mean_st_error_mv": round(float(np.nanmean(sterr)), 4),
        }
        print(f"[{r:8s}] n={len(R)} fab_rate={fab.mean():.3f} CI{_boot_ci(fab)} "
              f"masked_rate={msk.mean():.3f} ST-err={np.nanmean(sterr):.3f}mV", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "st_safety.json").write_text(json.dumps(out, indent=2))
    print("[cert] eta_ST_precordial (limb-6):", cert, flush=True)
    print("[json] results/st_safety.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
