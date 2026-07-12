"""Target-specific recoverability maps on real PTB-XL (CPU, offline).

Realises the corrected certificate (P0-1): for each observed configuration ``S`` and
waveform segment ``s``, and for each TARGET lead ``ell``, we report

  * eta_{s,ell}(S)   -- identifiability (0 => the dipolar component of lead ell is
                        recoverable from S; >0 => an unobserved dipole direction
                        changes it);
  * kappa_{s,ell}(S) -- noise / observed-residual amplification into the identifiable
                        part of lead ell;

with (a) a truncation-tolerance (rcond) sensitivity sweep -- because near-rank-deficient
configurations' rank/kappa depend on rcond -- and (b) record-bootstrap 95% CIs on the
per-lead numbers (M_s is re-estimated on each bootstrap resample of the pooled segment
samples). Global kappa_s(S) is reported only as a configuration-level worst-case
summary. NOTE: M_s is a population-estimated (PCA) object; these numbers depend on it.

Output: results/recoverability_maps.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.data import PTBXL
from ecgcert.physics import LEADS, LEAD_INDEX, fit_dipolar_subspace, kappa, kappa_per_lead, eta_per_lead

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
RCONDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
DEPLOY_RCOND = 1e-2
CONFIGS = {
    "Lead-I": ["I"],
    "Lead-II": ["II"],
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _fit_M(samples, rank=3, clip_mV=10.0):
    """Robust M_s via SVD of finite, non-outlier pooled samples."""
    X = samples[np.all(np.isfinite(samples), axis=1)]
    X = X[np.all(np.abs(X) <= clip_mV, axis=1)]
    M, mu, evr = fit_dipolar_subspace(X, rank=rank)
    return M, mu, evr, X


def maps(n_records=1500, n_boot=200, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    print(f"[maps] collecting segments from {min(n_records, len(norm))} NORM records ...", flush=True)
    samples = db.collect_all_segments(norm, rate=100, max_per_record=40,
                                      max_records=n_records, seed=seed)

    out = {"n_records": int(min(n_records, len(norm))), "n_boot": n_boot,
           "deploy_rcond": DEPLOY_RCOND, "rconds": list(RCONDS), "configs": {}}
    for cname, obs in CONFIGS.items():
        obs_set = set(obs)
        targets = [l for l in LEADS if l not in obs_set]
        out["configs"][cname] = {"observed": obs, "segments": {}}
        for s in SEGMENTS:
            X = samples[s]
            if X.shape[0] < 300:
                continue
            M, mu, evr, Xc = _fit_M(X)
            # rcond sensitivity (rank + global kappa) on the point estimate
            rc_sweep = {}
            for rc in RCONDS:
                k, r = kappa(M, obs, rcond=rc)
                rc_sweep[f"{rc:g}"] = {"rank": int(r), "kappa_global": float(k)}
            # per-lead eta / kappa at deployment rcond
            eta = eta_per_lead(M, obs, rcond=DEPLOY_RCOND)
            kpl = kappa_per_lead(M, obs, rcond=DEPLOY_RCOND)
            # record/sample bootstrap CI on per-lead eta/kappa (re-estimate M each time)
            N = Xc.shape[0]
            boot_eta = np.zeros((n_boot, 12)); boot_kap = np.zeros((n_boot, 12))
            brng = np.random.default_rng(seed + 1)
            for b in range(n_boot):
                idx = brng.integers(0, N, N)
                Mb, _, _ = fit_dipolar_subspace(Xc[idx], rank=3)
                boot_eta[b] = eta_per_lead(Mb, obs, rcond=DEPLOY_RCOND)
                boot_kap[b] = kappa_per_lead(Mb, obs, rcond=DEPLOY_RCOND)
            seg = {"dipolar_fraction": float(evr[:3].sum()), "rcond_sweep": rc_sweep,
                   "leads": {}}
            for l in targets:
                li = LEAD_INDEX[l]
                seg["leads"][l] = {
                    "eta": float(eta[li]),
                    "eta_ci": [float(np.percentile(boot_eta[:, li], 2.5)),
                               float(np.percentile(boot_eta[:, li], 97.5))],
                    "kappa": float(kpl[li]),
                    "kappa_ci": [float(np.percentile(boot_kap[:, li], 2.5)),
                                 float(np.percentile(boot_kap[:, li], 97.5))],
                }
            out["configs"][cname]["segments"][s] = seg
        _print_cfg(cname, out["configs"][cname])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "recoverability_maps.json").write_text(json.dumps(out, indent=2))
    print("[json] results/recoverability_maps.json", flush=True)


def _print_cfg(cname, cfg):
    print(f"\n=== {cname}  (obs={cfg['observed']}) ===", flush=True)
    for s, seg in cfg["segments"].items():
        ranks = {rc: v["rank"] for rc, v in seg["rcond_sweep"].items()}
        # identifiable vs unidentifiable target leads (deployment rcond)
        idf = [l for l, d in seg["leads"].items() if d["eta"] < 1e-3]
        unid = [l for l, d in seg["leads"].items() if d["eta"] >= 1e-3]
        print(f"  [{s}] dip={seg['dipolar_fraction']:.2f} rank(rcond)={ranks} "
              f"identifiable={idf} unidentifiable={unid}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    maps(n_records=args.n_records, n_boot=args.n_boot)
