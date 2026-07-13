"""Cross-dataset transfer of the recoverability certificate (honest scope).

We fit the per-segment dipolar subspace M_s with IDENTICAL processing on two independent
hospital populations -- PTB-XL (German) and Chapman-Shaoxing-Ningbo (Chinese, PhysioNet
ecg-arrhythmia 1.0.0) -- and compare (a) the subspaces by PRINCIPAL ANGLES and (b) the
per-configuration recoverability under the UNIFIED truncated-SVD numerics (rcond=1e-2):
rank + global kappa across an rcond sweep, and per-lead eta / normalized eta_tilde at the
deployment tolerance. We do NOT claim M_s is dataset-independent; we report what transfers
(QRS subspace + conditioning) and what does not (ST/T third direction). For the degenerate
limb-6 configuration we report its effective rank (2) and per-lead eta/eta_tilde rather than
inverting a near-zero singular value into a spurious 1e5-scale kappa (consistent with the
main map's rule for degenerate sets).

Runs on the GPU box (PTB-XL local + PhysioNet reachable). CPU only.

Output: results/cross_dataset.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import resample_poly

import wfdb
from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (LEAD_INDEX, LEADS, kappa, eta_per_lead, eta_normalized_per_lead)

RESULTS = Path(__file__).resolve().parent.parent / "results"
PN = "ecg-arrhythmia/1.0.0"
SEGS = ("P", "QRS", "ST", "T")
RCONDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
DEPLOY_RCOND = 1e-2
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "{V1,V2,V3}": ["V1", "V2", "V3"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def fit_M(samples, rank=3, clip_mV=10.0):
    """Robust per-segment dipolar basis via eigendecomposition of the 12x12 covariance
    (eigh always converges, unlike SVD on extreme-amplitude Chapman rows). Returns
    {seg: (M(12,3), mu(12,), evr, n)}."""
    out = {}
    for s, X in samples.items():
        if X.size == 0:
            continue
        X = X[np.all(np.isfinite(X), axis=1)]
        X = X[np.all(np.abs(X) <= clip_mV, axis=1)]
        if X.shape[0] < 200:
            continue
        mu = X.mean(0)
        C = np.cov((X - mu).T)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        M = V[:, :rank]
        evr = w[:rank] / max(w.sum(), 1e-12)
        out[s] = (M, mu, evr, int(X.shape[0]))
    return out


def chapman_record_specs(folder_stride=8, per_folder=8, cap=350):
    folders = wfdb.get_record_list(PN)[::folder_stride]
    specs = []
    for fol in folders:
        fol = fol.strip("/")
        try:
            names = wfdb.get_record_list(f"{PN}/{fol}")
        except Exception:
            continue
        for nm in names[:per_folder]:
            specs.append((fol, nm.strip("/").split("/")[-1]))
            if len(specs) >= cap:
                return specs
    return specs


def collect_chapman(specs, max_per_record=40, max_ok=350, seed=0):
    rng = np.random.default_rng(seed)
    rows = {s: [] for s in SEGS}
    ok = 0
    for fol, name in specs:
        try:
            r = wfdb.rdrecord(name, pn_dir=f"{PN}/{fol}")
            sig = r.p_signal.astype(float)
        except Exception:
            continue
        if sig.shape[1] != 12 or sig.shape[0] < 2500 or not np.all(np.isfinite(sig)):
            continue
        sig = resample_poly(sig, 1, 5, axis=0)
        if not np.all(np.isfinite(sig)):
            continue
        segs = PTBXL.segment_indices(sig, fs=100)
        any_seg = False
        for s, idx in segs.items():
            if idx.size == 0:
                continue
            if idx.size > max_per_record:
                idx = rng.choice(idx, max_per_record, replace=False)
            rows[s].append(sig[idx]); any_seg = True
        ok += any_seg
        if ok % 100 == 0 and any_seg:
            print(f"  chapman processed {ok} usable records", flush=True)
        if ok >= max_ok:
            break
    return {s: (np.vstack(v) if v else np.zeros((0, 12))) for s, v in rows.items()}, ok


def _recover(M, obs):
    """Unified-numerics recoverability of config `obs` under basis M: rcond sweep (rank +
    global kappa) and per-lead eta / eta_tilde at deploy rcond."""
    sweep = {}
    for rc in RCONDS:
        k, r = kappa(M, obs, rcond=rc)
        sweep[f"{rc:g}"] = {"rank": int(r), "kappa_global": round(float(k), 3)}
    eta = eta_per_lead(M, obs, rcond=DEPLOY_RCOND)
    etn = eta_normalized_per_lead(M, obs, rcond=DEPLOY_RCOND)
    tgt = [l for l in LEADS if l not in set(obs)]
    per_lead = {l: {"eta": round(float(eta[LEAD_INDEX[l]]), 4),
                    "eta_normalized": (None if not np.isfinite(etn[LEAD_INDEX[l]])
                                       else round(float(etn[LEAD_INDEX[l]]), 4))} for l in tgt}
    return {"rcond_sweep": sweep, "rank_deploy": sweep[f"{DEPLOY_RCOND:g}"]["rank"], "leads": per_lead}


def main():
    db = PTBXL()
    print("[cross] fitting PTB-XL M_s (NORM, 100Hz, lead-II dwt) ...", flush=True)
    norm = db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8))
    ptb_samples = db.collect_all_segments(norm, rate=100, max_per_record=40, max_records=1500, seed=0)
    ptb_models = fit_M(ptb_samples)

    print("[cross] streaming Chapman-Shaoxing-Ningbo subset ...", flush=True)
    specs = chapman_record_specs(folder_stride=8, per_folder=8, cap=350)
    print(f"[cross] {len(specs)} record specs across leaf folders", flush=True)
    chap_samples, n_chap = collect_chapman(specs, max_ok=350)
    chap_models = fit_M(chap_samples)
    print(f"[cross] chapman usable records: {n_chap}", flush=True)

    specs_sha = hashlib.sha256(("|".join(f"{a}/{b}" for a, b in specs)).encode()).hexdigest()[:16]
    out = {"n_chapman_records": n_chap, "deploy_rcond": DEPLOY_RCOND, "rconds": list(RCONDS),
           "processing": "100Hz, lead-II dwt, top-3 spatial eig; unified truncated-SVD recoverability",
           "segments": {}, "recoverability_QRS": {},
           "lineage": lineage.make(db, seed=0, targets=list(LEADS), normalization="raw mV",
                                   train_ids=norm[:1500],
                                   extra={"datasets": ["PTB-XL (folds 1-7 NORM)",
                                                       "Chapman-Shaoxing-Ningbo (PhysioNet ecg-arrhythmia 1.0.0)"],
                                          "chapman_n_records": n_chap, "chapman_specs_sha256": specs_sha})}
    for s in SEGS:
        if s not in ptb_models or s not in chap_models:
            continue
        (Mp, _, evp, np_) = ptb_models[s]
        (Mc, _, evc, nc_) = chap_models[s]
        ang = np.degrees(subspace_angles(Mp, Mc))
        out["segments"][s] = {
            "principal_angles_deg": [round(float(a), 2) for a in ang],
            "max_angle_deg": round(float(np.max(ang)), 2),
            "ptbxl_evr3": [round(float(x), 3) for x in np.asarray(evp)[:3]],
            "chapman_evr3": [round(float(x), 3) for x in np.asarray(evc)[:3]],
            "n_ptb": np_, "n_chap": nc_,
        }
    # unified-numerics recoverability on the transfer-stable QRS subspace
    for name, leads in CONFIGS.items():
        out["recoverability_QRS"][name] = {
            "ptbxl": _recover(ptb_models["QRS"][0], leads),
            "chapman": _recover(chap_models["QRS"][0], leads),
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cross_dataset.json").write_text(json.dumps(out, indent=2))
    print("\n=== PRINCIPAL ANGLES (deg) PTB-XL vs Chapman ===")
    for s in ("QRS", "ST", "T", "P"):
        if s in out["segments"]:
            d = out["segments"][s]
            print(f"  {s:3s}: angles={d['principal_angles_deg']} max={d['max_angle_deg']}")
    print("\n=== QRS recoverability (rank@1e-2, kappa@1e-2) PTB-XL vs Chapman ===")
    for name, r in out["recoverability_QRS"].items():
        pp = r["ptbxl"]["rcond_sweep"][f"{DEPLOY_RCOND:g}"]; cc = r["chapman"]["rcond_sweep"][f"{DEPLOY_RCOND:g}"]
        print(f"  {name:18s}: ptb rank{pp['rank']} k={pp['kappa_global']:.2f} | "
              f"chap rank{cc['rank']} k={cc['kappa_global']:.2f}")
    print("\n[json] results/cross_dataset.json")


if __name__ == "__main__":
    main()
