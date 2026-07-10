"""GPU fabrication exhibit (CRAFT): oracle ceiling + recoverability-deficit certificate.

Design synthesized by a multi-agent design workflow + adversarial critique. The core
move that defeats circularity: an ill-posed certified-unrecoverable subspace gives
rho~0 for ANY sampler (even Bayes-optimal), so rho~0 there measures the definition,
not fabrication. Instead we build a HELD-OUT ORACLE that partitions the per-segment
non-dipolar residual into:
  Band A -- recoverable  (rho_oracle >> 0): a supervised predictor demonstrably CAN
            recover this from the observed leads S on held-out data.
  Band B -- aleatoric    (rho_oracle ~ 0): genuinely unrecoverable.
Fabrication = a RECOVERABILITY DEFICIT: large energy h co-located with
Delta_rho = rho_oracle - rho_model > 0 in Band A. Band B is the built-in negative
control (flag must stay silent).

Primary S = {I,II,V1,V3,V5} -> reconstruct {V2,V4,V6} (precordial interpolation, where
recoverability is provable), NOT limb-6 (used only as the Band-B negative control).

STAGE 1 (this file, run first): the ORACLE GATE. If rho_oracle is not > 0 for the
primary (segment, unobserved lead), the exhibit is dead -- stop and switch S.
Later stages add the conditional CFG diffusion model + certificate (see run modes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecgcert.certify import certified_unrecoverable_projector
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T", "P")

CONFIGS = {
    "precordial-interp": {"obs": ["I", "II", "V1", "V3", "V5"], "target": ["V2", "V4", "V6"]},
    "classic-3lead":     {"obs": ["I", "II", "V2"],             "target": ["V1", "V3", "V4", "V5", "V6"]},
    "limb6-negctrl":     {"obs": ["I", "II", "III", "aVR", "aVL", "aVF"], "target": ["V1", "V2", "V3", "V4", "V5", "V6"]},
}


def _ridge_fit(X, y, lam=1.0):
    """Closed-form ridge with bias. X:(N,d), y:(N,). Returns (w, b)."""
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    d = X1.shape[1]
    A = X1.T @ X1 + lam * np.eye(d)
    A[-1, -1] -= lam  # don't regularize bias
    coef = np.linalg.solve(A, X1.T @ y)
    return coef[:-1], coef[-1]


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def oracle_gate(n_train=2000, n_test=800, rate=100, seed=0):
    """Fit per-sample ridge oracle predicting each unobserved lead's non-dipolar
    residual from the observed leads; report held-out rho_oracle per (config, seg, lead)."""
    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())

    # Segment models on TRAIN only.
    tr_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=60,
                                         max_records=n_train, seed=seed)
    models = fit_segment_models(tr_samples)
    te_samples = db.collect_all_segments(test_ids, rate=rate, max_per_record=60,
                                         max_records=n_test, seed=seed + 1)

    out = {"n_train": n_train, "n_test": n_test, "configs": {}}
    for cname, cfg in CONFIGS.items():
        obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]
        out["configs"][cname] = {}
        for seg in SEGMENTS:
            m = models.get(seg)
            if m is None or tr_samples[seg].shape[0] < 200 or te_samples[seg].shape[0] < 200:
                continue
            U = certified_unrecoverable_projector(m.M, cfg["obs"])   # (12,12)
            Xtr = tr_samples[seg]; Xte = te_samples[seg]             # (N,12) mV
            # observed-lead features (raw mV of observed leads)
            Ftr, Fte = Xtr[:, obs_idx], Xte[:, obs_idx]
            # non-dipolar residual of ALL leads
            Rtr = (Xtr - m.mu) @ U.T                                  # (N,12)
            Rte = (Xte - m.mu) @ U.T
            seg_res = {}
            for l in cfg["target"]:
                li = LEAD_INDEX[l]
                w, b = _ridge_fit(Ftr, Rtr[:, li], lam=1.0)
                pred = Fte @ w + b
                rho = _pearson(pred, Rte[:, li])
                true_rms = float(np.sqrt(np.mean(Rte[:, li] ** 2)))
                seg_res[l] = {"rho_oracle": rho, "true_nondip_rms_mV": true_rms}
            out["configs"][cname][seg] = seg_res
        _print_cfg(cname, out["configs"][cname])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gpu_oracle_gate.json").write_text(json.dumps(out, indent=2))
    _verdict(out)


def _print_cfg(cname, segs):
    print(f"\n=== {cname} ===")
    for seg, res in segs.items():
        parts = "  ".join(f"{l}:rho_or={d['rho_oracle']:+.2f}(nd={d['true_nondip_rms_mV']:.3f})"
                          for l, d in res.items())
        print(f"  [{seg}] {parts}")


def _verdict(out):
    prim = out["configs"].get("precordial-interp", {})
    qrs = prim.get("QRS", {})
    band_a = [l for l, d in qrs.items() if d["rho_oracle"] > 0.2]
    print("\n=== ORACLE GATE VERDICT ===")
    print(f"Primary S QRS Band-A leads (rho_oracle>0.2): {band_a}")
    neg = out["configs"].get("limb6-negctrl", {}).get("QRS", {})
    neg_rho = {l: round(d["rho_oracle"], 2) for l, d in neg.items()}
    print(f"limb6 negative-control QRS rho_oracle: {neg_rho}")
    ok = len(band_a) >= 1
    print(f"GATE {'PASS' if ok else 'FAIL'}: primary S has recoverable (Band A) non-dipolar content")
    out["gate_pass"] = bool(ok)
    (RESULTS / "gpu_oracle_gate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gate", choices=["gate"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=800)
    args = ap.parse_args()
    if args.mode == "gate":
        oracle_gate(n_train=args.n_train, n_test=args.n_test)
