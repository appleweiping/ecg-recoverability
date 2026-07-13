"""Certificate-driven ST safety of limb-6 -> precordial reconstruction (honest, continuous).

Two separate objects, deliberately kept distinct:

1. CERTIFICATE (reconstructor-agnostic): for the ST segment under limb-6, the per-lead
   identifiability eta_{ST,ell}(S) > 0 for precordial leads (a dipole direction is
   unobserved). We report absolute eta, NORMALIZED eta_tilde = eta/||e_ell^T M_ST|| (the
   fraction of the lead's dipolar content that is unobservable), and the prior-CONDITIONAL
   expected ambiguity in mV (residual a Bayes reconstructor still incurs under the fitted
   Gaussian dipole prior) -- so "graded" identifiability is stated on scale-free / physical
   quantities, not absolute eta alone.

2. DOWNSTREAM COST (reconstructor-specific): using REAL continuous linear reconstructors we
   reconstruct the FULL 10-second waveform (no constant-mean gap fill), keep observed leads
   exact, and measure ST at J+60 ms vs a PR baseline on fiducials located ONCE on the observed
   Lead II and SHARED between truth and reconstruction. An ST-threshold event = |ST| >= 0.1 mV
   (absolute; elevation or depression). false_positive = crossing in reconstruction not truth;
   false_negative = crossing in truth not reconstruction. We report each reconstructor's
   false-positive / false-negative / total wrong-event rate and mean |ST| error, with
   record-bootstrap 95% CIs. We report ST-threshold events only, never a clinical diagnosis.

We do NOT claim the total wrong-event rate is a certified minimax/Bayes floor; we report it as
an empirically similar total error rate across the evaluated reconstructors.

Output: results/st_safety.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.clinical import ST_THRESHOLD_MV, fiducials, st_threshold_events
from ecgcert.data import PTBXL
from ecgcert.physics import (
    LEAD_INDEX, fit_dipolar_subspace, reconstruct_dipolar,
    eta_per_lead, eta_normalized_per_lead, dipole_coord_cov, expected_ambiguity_per_lead,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
LIMB6 = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
PRECORDIAL_IDX = [LEAD_INDEX[l] for l in PRECORDIAL]
OI = [LEAD_INDEX[l] for l in LIMB6]
NORMALIZATION = "raw mV (no per-record scaling)"


def _boot_ci(mask, n=1000, seed=0):
    v = np.asarray(mask, float)
    if v.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def _fit_continuous(db, ids, rate, per_record_pts, seed):
    """Fit continuous limb-6 -> 12-lead reconstructors on pooled per-TIMEPOINT samples.

    Returns global dipolar (M, mu) and ridge/OLS (W, b) maps. Sampling whole-record
    timepoints (not just delineated segments) gives a genuinely continuous reconstructor.
    """
    rng = np.random.default_rng(seed)
    pool = []
    for eid in ids:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        if sig.shape[0] < 50 or not np.all(np.isfinite(sig)):
            continue
        t = rng.choice(sig.shape[0], min(per_record_pts, sig.shape[0]), replace=False)
        pool.append(sig[t])
    X = np.vstack(pool)                                       # (N, 12) per-timepoint
    X = X[np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= 10.0, axis=1)]
    M, mu, _ = fit_dipolar_subspace(X, rank=3)
    # ridge / ols limb-6 -> 12
    Xo = X[:, OI]
    X1 = np.hstack([Xo, np.ones((Xo.shape[0], 1))])
    maps = {}
    for name, lam in (("ridge", 1.0), ("ols", 0.0)):
        A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[-1, -1] -= lam
        W = np.linalg.solve(A, X1.T @ X)                      # (7, 12)
        maps[name] = (W[:-1].T, W[-1])                        # (12,6),(12,)
    return (M, mu), maps, X


def _reconstruct(sig, dip, maps):
    """Continuous (T,12) reconstruction per method; observed limb leads kept exact."""
    y_obs = sig[:, OI]                                        # (T, 6)
    M, mu = dip
    out = {}
    out["dipolar"] = reconstruct_dipolar(M, mu, LIMB6, y_obs.T).T      # (T,12)
    for name, (W, b) in maps.items():
        out[name] = y_obs @ W.T + b                          # (T,12)
    for r in out:
        out[r][:, OI] = sig[:, OI]                            # keep observed exact
    return out


def run(n_train=1500, n_test=1500, rate=100, seed=0, per_record_pts=80):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))[:n_train]

    # per-segment ST model for the certificate (eta / eta_tilde / expected ambiguity)
    st_seg = db.collect_all_segments(tr_ids, rate=rate, max_per_record=40,
                                     max_records=n_train, seed=seed)["ST"]
    st_seg = st_seg[np.all(np.isfinite(st_seg), axis=1) & np.all(np.abs(st_seg) <= 10.0, axis=1)]
    M_st, mu_st, _ = fit_dipolar_subspace(st_seg, rank=3)
    Sig_d = dipole_coord_cov(M_st, mu_st, st_seg)
    eta = eta_per_lead(M_st, LIMB6); etn = eta_normalized_per_lead(M_st, LIMB6)
    amb = expected_ambiguity_per_lead(M_st, LIMB6, Sig_d)
    cert = {l: {"eta": round(float(eta[LEAD_INDEX[l]]), 4),
                "eta_normalized": round(float(etn[LEAD_INDEX[l]]), 4),
                "expected_ambiguity_mV": round(float(amb[LEAD_INDEX[l]]), 4)} for l in PRECORDIAL}

    # continuous reconstructors
    dip, maps, _ = _fit_continuous(db, tr_ids, rate, per_record_pts, seed)
    recons = ("dipolar", "ridge", "ols")

    test = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())[:n_test]
    rows = {r: [] for r in recons}
    n_valid = 0
    for eid in test:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        if sig.shape[0] < 50 or not np.all(np.isfinite(sig)):
            continue
        fid = fiducials(sig[:, LEAD_INDEX["II"]], rate)      # ONCE, on observed Lead II
        if fid is None or np.asarray(fid["R_off"]).size == 0:
            continue
        full = _reconstruct(sig, dip, maps)
        valid_any = False
        for r in recons:
            ev = st_threshold_events(sig, full[r], rate, leads=PRECORDIAL_IDX,
                                     thr=ST_THRESHOLD_MV, fid=fid)
            if ev.get("valid"):
                rows[r].append(ev); valid_any = True
        n_valid += valid_any

    out = {"config": "limb-6 -> precordial", "st_threshold_mV": ST_THRESHOLD_MV,
           "st_rule": "|ST| >= 0.1 mV (absolute; elevation or depression)",
           "certificate_ST_precordial": cert,
           "n_test": int(len(test)), "n_valid_delineation": int(n_valid),
           "note": ("eta_ST_precordial > 0 => precordial ST not identifiable from limb leads; "
                    "graded by eta_normalized (fraction unobservable) and expected ambiguity (mV). "
                    "The total wrong-event rate below is an empirically similar total error rate "
                    "across the evaluated reconstructors, NOT a certified lower bound."),
           "reconstructors": {},
           "lineage": lineage.make(db, seed=seed, targets=PRECORDIAL, normalization=NORMALIZATION,
                                   train_ids=tr_ids, test_ids=test,
                                   extra={"reconstruction": "continuous full-waveform, observed leads exact",
                                          "st_offset_ms": 60.0, "fiducials": "shared, observed Lead II"})}
    for r in recons:
        R = rows[r]
        if not R:
            continue
        fp = np.array([x["false_positive"] for x in R], float)
        fn = np.array([x["false_negative"] for x in R], float)
        sterr = np.array([x["st_error_mv"] for x in R], float)
        out["reconstructors"][r] = {
            "n": len(R),
            "false_positive_rate": round(float(fp.mean()), 4), "false_positive_ci": _boot_ci(fp),
            "false_negative_rate": round(float(fn.mean()), 4), "false_negative_ci": _boot_ci(fn),
            "total_wrong_rate": round(float((fp + fn).mean()), 4),
            "mean_st_error_mv": round(float(np.nanmean(sterr)), 4),
        }
        print(f"[{r:8s}] n={len(R)} FP={fp.mean():.3f} FN={fn.mean():.3f} "
              f"total={(fp+fn).mean():.3f} |ST|err={np.nanmean(sterr):.3f}mV", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "st_safety.json").write_text(json.dumps(out, indent=2))
    print("[cert] ST precordial (limb-6):", cert, flush=True)
    print("[json] results/st_safety.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
