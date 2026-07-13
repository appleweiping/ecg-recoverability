"""Sensitivity of the certificate to duplicated limb leads in the PCA (P1-11).

The 6 limb leads are exact linear combinations of I and II (Einthoven/Goldberger), so a PCA
of all 12 leads UP-WEIGHTS frontal-plane content: I and II effectively appear in 6 of the 12
rows. We therefore refit M_s on only the 8 algebraically-INDEPENDENT leads [I,II,V1..V6],
lift it back to 12-lead space via the fixed transform T (L = T x), re-orthonormalize, and
compare the certificate (eta per lead, global kappa) and the 3-D subspace itself (principal
angles) against the 12-lead fit. Small angles + matching eta/kappa => the certificate is not
an artifact of the duplicated limb weighting.

Output: results/lead_weighting.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (LEADS, LEAD_INDEX, INDEPENDENT_LEADS, lead_transform_T,
                             fit_dipolar_subspace, eta_per_lead, kappa)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
IND_IDX = [LEAD_INDEX[l] for l in INDEPENDENT_LEADS]
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _fit_8lead(X):
    """Fit rank-3 subspace on the 8 independent leads, lift to 12-lead via T, orthonormalize."""
    X8 = X[:, IND_IDX]                                        # (N, 8)
    mu8 = X8.mean(axis=0)
    U, sv, _ = np.linalg.svd((X8 - mu8).T, full_matrices=False)
    M8 = U[:, :3]                                             # (8, 3)
    T = lead_transform_T()                                    # (12, 8)
    Q, _ = np.linalg.qr(T @ M8)                               # (12, 3) orthonormal
    return Q


def run(n_records=1500, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:n_records]
    samples = db.collect_all_segments(norm, rate=100, max_per_record=40, max_records=n_records, seed=seed)
    out = {"n_records": int(len(norm)), "segments": {},
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization="raw mV",
                                   train_ids=norm, extra={"comparison": "12-lead PCA vs 8-independent-lead PCA"})}
    for s in SEGMENTS:
        X = samples[s]
        X = X[np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= 10.0, axis=1)]
        if X.shape[0] < 300:
            continue
        M12, _, _ = fit_dipolar_subspace(X, rank=3)
        M8 = _fit_8lead(X)
        angles = np.degrees(subspace_angles(M12, M8))        # descending
        seg = {"principal_angles_deg": [round(float(a), 3) for a in angles],
               "max_angle_deg": round(float(angles.max()), 3), "configs": {}}
        for cname, obs in CONFIGS.items():
            k12, r12 = kappa(M12, obs); k8, r8 = kappa(M8, obs)
            e12 = eta_per_lead(M12, obs); e8 = eta_per_lead(M8, obs)
            # max abs eta difference over target leads
            tgt = [LEAD_INDEX[l] for l in LEADS if l not in obs]
            seg["configs"][cname] = {
                "kappa_12lead": round(float(k12), 3), "kappa_8lead": round(float(k8), 3),
                "rank_12lead": int(r12), "rank_8lead": int(r8),
                "max_eta_diff": round(float(np.max(np.abs(e12[tgt] - e8[tgt]))), 4),
            }
        out["segments"][s] = seg
        print(f"[{s}] max_angle={seg['max_angle_deg']:.1f}deg " +
              " ".join(f"{c}:dk={abs(v['kappa_12lead']-v['kappa_8lead']):.2f}"
                       for c, v in seg["configs"].items()), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lead_weighting.json").write_text(json.dumps(out, indent=2))
    print("[json] results/lead_weighting.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=1500)
    args = ap.parse_args()
    run(n_records=args.n_records)
