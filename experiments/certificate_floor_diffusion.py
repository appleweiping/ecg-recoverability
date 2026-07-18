"""Does the certified minimax floor bind a LEARNED generative reconstructor? (GPU)

certificate_validation.py showed the per-lead dipolar-projection error of the LINEAR reconstructors
(dipolar / ridge / OLS) sits on/above the certified ambiguity floor a_l (paper/theorem_floor.tex).
The floor is proved against ANY estimator, linear or nonlinear, over the moment class -- so a trained
diffusion model must obey it too. This closes that loop empirically: we reconstruct held-out fold-10
NORM records with the arbitrary-mask DDPM (results/gpu_ddpm.pt) at honest guidance w=1, measure the
SAME per-lead dipolar-projection RMSE e_l^T (M_s M_s^+)(Lhat - Ltrue), and check it against the SAME
floor a_l computed from the SAME NORM-train dipolar subspace M_s as certificate_validation.

M_s (hence a_l, eta, kappa) is refit here identically to certificate_validation (folds 1-7,
n_train, max_per_record=40, fit_dipolar_subspace rank 3), so the DDPM is measured against exactly the
floor the linear methods were. We restrict to the UNDERDETERMINED configs (limb-6, {I,II,V1,V3,V5}),
where the floor is non-trivial (the observed leads do not span the dipole); on a rank-3 spanning set
a_l=0 and the test is vacuous.

Output: results/certificate_floor_diffusion.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (
    LEAD_INDEX, fit_dipolar_subspace, dipole_coord_cov,
    eta_per_lead, kappa_per_lead, expected_ambiguity_per_lead,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_diffusion_clean import _preload, _batched                  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SEGMENTS = ("QRS", "ST", "T")
# underdetermined configs only (non-trivial floor). limb-6 misses the antero-posterior dipole
# direction; the precordial-partial set misses one direction too.
CONFIGS = {"limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
           "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"]}
GUIDANCE = 1.0               # honest reconstruction (no classifier-free inflation)
ETA_ZERO_TOL = 1e-3


def run(n_train=1500, n_test=300, max_per_record=40, seed=0, chunk=128):
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor
    db = PTBXL()
    rng = np.random.default_rng(seed)

    # ---- fit M_s per segment IDENTICALLY to certificate_validation (same floor a_l) ----
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:n_train]
    tr_seg = db.collect_all_segments_with_ids(tr_ids, rate=100, max_per_record=max_per_record,
                                              max_records=len(tr_ids), seed=seed)
    Mseg, Sigseg = {}, {}
    for s in SEGMENTS:
        Xtr = np.asarray(tr_seg[s][0], float)
        Xtr = Xtr[np.all(np.isfinite(Xtr), axis=1) & np.all(np.abs(Xtr) <= 10.0, axis=1)]
        if Xtr.shape[0] < 200:
            continue
        M, mu, _ = fit_dipolar_subspace(Xtr, rank=3)
        Mseg[s] = (M, mu)
        Sigseg[s] = dipole_coord_cov(M, mu, Xtr)
    print(f"[certfloor-diff] fitted M_s for segments {list(Mseg)} (n_train={len(tr_ids)})", flush=True)

    # ---- DDPM + held-out fold-10 NORM test ----
    ckpt = str(RESULTS / "gpu_ddpm.pt")
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    scale, T = ck["scale"], int(ck["T"])
    model = DiffusionReconstructor(T=T, base=64, device="cuda", seed=seed)
    model.net.load_state_dict(ck["sd"]); model.net.eval()
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm_f10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm_f10))[:n_test]
    sigs, segidxs, kept = _preload(db, test_ids)
    print(f"[certfloor-diff] {len(sigs)} fold-10 NORM test records; configs {list(CONFIGS)}", flush=True)

    cells = []
    for cname, obs in CONFIGS.items():
        obs_idx = [LEAD_INDEX[l] for l in obs]
        Lhats = _batched(model, sigs, scale, GUIDANCE, T, seed=200, obs_idx=obs_idx, chunk=chunk)  # (12,1000)
        for s in SEGMENTS:
            if s not in Mseg:
                continue
            M, mu = Mseg[s]
            P = M @ np.linalg.pinv(M)                                # (12,12) dipolar projector
            eta = eta_per_lead(M, obs); kap = kappa_per_lead(M, obs)
            amb = expected_ambiguity_per_lead(M, obs, Sigseg[s])
            for l in LEADS:
                if l in obs:
                    continue
                li = LEAD_INDEX[l]
                per_rec = []
                for Lh, tru, seg in zip(Lhats, sigs, segidxs):
                    idx = seg.get(s)
                    if idx is None or len(idx) == 0:
                        continue
                    # coerce both to lead-major (12, T): DDPM _batched returns (12,1000) but the
                    # raw PTB-XL waveform is time-major (1000,12).
                    Lh = np.asarray(Lh); tru = np.asarray(tru)
                    if Lh.shape[0] != 12:
                        Lh = Lh.T
                    if tru.shape[0] != 12:
                        tru = tru.T
                    d = (P @ (Lh[:, idx] - tru[:, idx]))[li]         # dipolar-projected err, lead l, over samples
                    per_rec.append(float(np.sqrt(np.mean(d ** 2))))
                if len(per_rec) < 20:
                    continue
                cells.append({
                    "config": cname, "segment": s, "lead": l,
                    "eta": round(float(eta[li]), 5), "kappa": round(float(kap[li]), 4),
                    "amb_mV": round(float(amb[li]), 5),
                    "eta_zero": bool(eta[li] < ETA_ZERO_TOL),
                    "ddpm_rmse_mV": round(float(np.mean(per_rec)), 5),
                    "n_records": len(per_rec),
                })
        print(f"[certfloor-diff] {cname}: {sum(c['config']==cname for c in cells)} cells", flush=True)

    # ---- floor test for the LEARNED model (same methodology as certificate_validation) ----
    y = np.array([c["ddpm_rmse_mV"] for c in cells])
    amb = np.array([c["amb_mV"] for c in cells])
    z = np.array([c["eta_zero"] for c in cells])
    floor = {
        "floor_violation_frac": round(float(np.mean(y < amb - 1e-6)), 4),
        "floor_gap_median_mV": round(float(np.median(y - amb)), 5),
        "median_rmse_eta0": round(float(np.median(y[z])), 5) if z.any() else None,
        "median_rmse_etapos": round(float(np.median(y[~z])), 5) if (~z).any() else None,
        "n_cells": len(cells),
    }
    print(f"[certfloor-diff] DDPM floor_viol={floor['floor_violation_frac']} "
          f"gap_med={floor['floor_gap_median_mV']} "
          f"med_rmse eta0={floor['median_rmse_eta0']} etapos={floor['median_rmse_etapos']}", flush=True)

    out = {"reconstructor": "DDPM (arbitrary-mask, guidance w=1)", "guidance": GUIDANCE,
           "metric": "per-lead dipolar-projection RMSE (mV): e_l^T (M_s M_s^+)(Lhat-Ltrue)",
           "configs": list(CONFIGS), "segments": [s for s in SEGMENTS if s in Mseg],
           "floor": floor, "cells": cells,
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS),
                                   normalization="raw mV (DDPM reconstruction); dipolar-projection error",
                                   train_ids=tr_ids, test_ids=test_ids, checkpoint=ckpt,
                                   extra={"checkpoint": ckpt, "guidance": GUIDANCE,
                                          "metric": "learned_model_vs_certified_floor",
                                          "configs": list(CONFIGS)})}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "certificate_floor_diffusion.json").write_text(json.dumps(out, indent=2))
    print("[json] results/certificate_floor_diffusion.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test)
