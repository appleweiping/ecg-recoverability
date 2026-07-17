"""Duplicated-limb-lead weighting sensitivity of the certificate (P0-D).

The six limb leads are exact linear combinations of I and II, so a 12-lead PCA up-weights
frontal-plane content. We refit M_s on only the 8 algebraically-INDEPENDENT leads [I,II,V1..V6],
lift back to 12-lead space via the fixed transform T, and compare the per-lead certificate
(eta, normalized eta_tilde, prior-conditional expected ambiguity, kappa) against the 12-lead
fit -- on the IDENTICAL records the primary map uses. We then test whether the V1-V6 ORDERING
(by normalized identifiability) is stable across the two fits: Spearman correlation, whether the
four anterior leads (V1-V4) remain the least recoverable, and record-bootstrap uncertainty.

If the ordering is stable we keep the graded ST claim but flag the magnitudes as
weighting-dependent; if not, the 8-independent-lead fit should be the primary map.

Output: results/lead_weighting.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (LEADS, LEAD_INDEX, INDEPENDENT_LEADS, lead_transform_T,
                             fit_dipolar_subspace, eta_per_lead, eta_normalized_per_lead,
                             dipole_coord_cov, expected_ambiguity_per_lead, kappa)
from recoverability_maps import map_record_ids

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
IND_IDX = [LEAD_INDEX[l] for l in INDEPENDENT_LEADS]
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
ANTERIOR = {"V1", "V2", "V3", "V4"}
LIMB6 = ["I", "II", "III", "aVR", "aVL", "aVF"]


def _fit8(X):
    """Rank-3 subspace on 8 independent leads, lifted to 12 via T, orthonormalized; +mean."""
    X8 = X[:, IND_IDX]
    mu8 = X8.mean(axis=0)
    U, sv, _ = np.linalg.svd((X8 - mu8).T, full_matrices=False)
    Q, _ = np.linalg.qr(lead_transform_T() @ U[:, :3])
    mu12 = lead_transform_T() @ mu8
    return Q, mu12


def _perlead(M, mu, X):
    """(eta, eta_norm, amb_mV, kappa_per_lead) for limb-6 over the 12 leads."""
    Sd = dipole_coord_cov(M, mu, X)
    eta = eta_per_lead(M, LIMB6); etn = eta_normalized_per_lead(M, LIMB6)
    amb = expected_ambiguity_per_lead(M, LIMB6, Sd)
    kap, _ = kappa(M, LIMB6)
    return eta, etn, amb, kap


def _clean(X):
    return X[np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= 10.0, axis=1)]


def run(n_records=1500, n_boot=200, seed=0):
    db = PTBXL()
    ids = map_record_ids(db, n_records, seed)               # SAME records as the primary map
    samples = db.collect_all_segments_with_ids(ids, rate=100, max_per_record=40,
                                               max_records=n_records, seed=seed)
    out = {"n_records": int(len(ids)), "same_ids_as_map": True,
           "record_ids_sha256": lineage.ids_sha256(ids), "segments": {},
           "lineage": lineage.make(db, seed=seed, targets=PRECORDIAL, normalization="raw mV",
                                   train_ids=ids, extra={"comparison": "12-lead PCA vs 8-independent-lead PCA"})}
    stable_flags = []
    for s in SEGMENTS:
        X0, rid0 = samples[s]
        ok = np.all(np.isfinite(X0), axis=1) & np.all(np.abs(X0) <= 10.0, axis=1)
        X, rid = X0[ok], rid0[ok]
        if X.shape[0] < 300:
            continue
        M12, mu12, _ = fit_dipolar_subspace(X, rank=3)
        M8, mu8 = _fit8(X)
        e12, n12, a12, k12 = _perlead(M12, mu12, X)
        e8, n8, a8, k8 = _perlead(M8, mu8, X)
        pl = {}
        for l in PRECORDIAL:
            li = LEAD_INDEX[l]
            pl[l] = {"eta12": round(float(e12[li]), 4), "eta_norm12": round(float(n12[li]), 4),
                     "amb12_mV": round(float(a12[li]), 4),
                     "eta8": round(float(e8[li]), 4), "eta_norm8": round(float(n8[li]), 4),
                     "amb8_mV": round(float(a8[li]), 4)}
        # ordering by normalized identifiability (higher = less recoverable)
        v12 = np.array([n12[LEAD_INDEX[l]] for l in PRECORDIAL])
        v8 = np.array([n8[LEAD_INDEX[l]] for l in PRECORDIAL])
        order12 = [PRECORDIAL[i] for i in np.argsort(-v12)]
        order8 = [PRECORDIAL[i] for i in np.argsort(-v8)]
        rho = float(spearmanr(v12, v8).correlation)
        ant_gt_lat_12 = bool(min(v12[:4]) >= max(v12[4:]))   # anterior all less recoverable
        ant_gt_lat_8 = bool(min(v8[:4]) >= max(v8[4:]))
        # record-bootstrap on Spearman + anterior-vs-lateral separation
        uids = np.unique(rid); id2rows = {u: np.where(rid == u)[0] for u in uids}
        brng = np.random.default_rng(seed + 1); rhos = []; ant_ok = 0
        for _ in range(n_boot):
            draw = uids[brng.integers(0, uids.size, uids.size)]
            Xb = X[np.concatenate([id2rows[u] for u in draw])]
            Mb, mub, _ = fit_dipolar_subspace(Xb, rank=3); M8b, mu8b = _fit8(Xb)
            nb = eta_normalized_per_lead(Mb, LIMB6); n8b = eta_normalized_per_lead(M8b, LIMB6)
            vb = np.array([nb[LEAD_INDEX[l]] for l in PRECORDIAL])
            v8b = np.array([n8b[LEAD_INDEX[l]] for l in PRECORDIAL])
            rr = spearmanr(vb, v8b).correlation
            if np.isfinite(rr):
                rhos.append(rr)
            ant_ok += int(min(vb[:4]) >= max(vb[4:]) and min(v8b[:4]) >= max(v8b[4:]))
        ang = np.degrees(subspace_angles(M12, M8))
        seg = {"principal_angles_deg": [round(float(x), 2) for x in ang],
               "max_angle_deg": round(float(ang.max()), 2), "per_lead": pl,
               "kappa12": round(float(k12), 3), "kappa8": round(float(k8), 3),
               "order_12lead": order12, "order_8lead": order8,
               "spearman_12_vs_8": round(rho, 3),
               "spearman_ci": [round(float(np.percentile(rhos, 2.5)), 3),
                               round(float(np.percentile(rhos, 97.5)), 3)] if rhos else [None, None],
               "anterior_least_recoverable_12": ant_gt_lat_12,
               "anterior_least_recoverable_8": ant_gt_lat_8,
               "anterior_lateral_ordering_stable_frac": round(ant_ok / n_boot, 3)}
        # a segment's flagship ordering is "stable" if Spearman CI stays high and anterior<lateral robust
        seg["ordering_stable"] = bool((seg["spearman_ci"][0] or 0) >= 0.6
                                      and seg["anterior_lateral_ordering_stable_frac"] >= 0.9)
        out["segments"][s] = seg
        if s in ("ST", "T"):
            stable_flags.append(seg["ordering_stable"])
        print(f"[{s}] maxang={seg['max_angle_deg']} rho={rho:.2f} CI{seg['spearman_ci']} "
              f"ant<lat frac={seg['anterior_lateral_ordering_stable_frac']} stable={seg['ordering_stable']}", flush=True)

    out["ST_T_ordering_stable"] = bool(all(stable_flags)) if stable_flags else None
    out["verdict"] = ("binary verdict and coarse V1-V6 ordering are robust across 12- and "
                      "8-independent-lead fits; exact ST/T eta/kappa/ambiguity MAGNITUDES are "
                      "weighting-dependent") if out["ST_T_ordering_stable"] else \
                     ("ordering NOT stable: the 8-independent-lead fit should be the primary map; "
                      "do not keep the exact graded ST/T claim")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lead_weighting.json").write_text(json.dumps(out, indent=2))
    print("[verdict]", out["verdict"], flush=True)
    print("[json] results/lead_weighting.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    run(n_records=args.n_records, n_boot=args.n_boot)
