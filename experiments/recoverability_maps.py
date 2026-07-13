"""Target-specific recoverability maps on real PTB-XL (CPU, offline).

For each observed configuration ``S`` and waveform segment ``s``, and each TARGET lead
``ell``, we report, from the population dipolar subspace ``M_s`` estimated on the training
NORM records:

  * eta_{s,ell}(S)      -- absolute identifiability (0 => dipolar component recoverable);
  * eta_tilde_{s,ell}   -- NORMALIZED identifiability eta / ||e_ell^T M_s|| in [0,1] (the
                           FRACTION of the lead's dipolar content that is unobservable);
  * amb_{s,ell}(S)      -- prior-CONDITIONAL expected ambiguity in mV (the residual a Bayes
                           reconstructor still incurs, unobserved coord conditioned on the
                           observed one via the fitted Gaussian dipole prior Sigma_d);
  * kappa_{s,ell}(S)    -- conditioning of the identifiable part;

with (a) a truncation-tolerance (rcond) sweep for rank/global-kappa, and (b) RECORD-LEVEL
bootstrap 95% CIs: each resample draws whole records with replacement, re-pools their
segment samples, and REFITS M_s, so the CI reflects between-record variability (pooled-sample
bootstrap understates it -- samples from one record are not exchangeable).

The map is a deterministic function of the estimated M_s and the selection S; it is not
"evaluated" on held-out records. M_s is fit on the stated NORM training records and the CI
quantifies its estimation uncertainty. NOTE: M_s is a population-estimated (PCA) object.

Output: results/recoverability_maps.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (
    LEADS, LEAD_INDEX, fit_dipolar_subspace, kappa, kappa_per_lead, eta_per_lead,
    eta_normalized_per_lead, dipole_coord_cov, expected_ambiguity_per_lead,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
RCONDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
DEPLOY_RCOND = 1e-2
NORMALIZATION = "raw mV (no per-record scaling)"
CONFIGS = {
    "Lead-I": ["I"],
    "Lead-II": ["II"],
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _clean(X, rids, clip_mV=10.0):
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= clip_mV, axis=1)
    return X[ok], rids[ok]


def _perlead(M, mu, Xc, obs):
    """(eta, eta_tilde, amb_mV, kappa) per-lead vectors at deploy rcond."""
    Sd = dipole_coord_cov(M, mu, Xc)
    return (eta_per_lead(M, obs, rcond=DEPLOY_RCOND),
            eta_normalized_per_lead(M, obs, rcond=DEPLOY_RCOND),
            expected_ambiguity_per_lead(M, obs, Sd, rcond=DEPLOY_RCOND),
            kappa_per_lead(M, obs, rcond=DEPLOY_RCOND))


def maps(n_records=1500, n_boot=200, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    used_ids = norm[:n_records]
    print(f"[maps] collecting segments from {len(used_ids)} NORM records ...", flush=True)
    samples = db.collect_all_segments_with_ids(used_ids, rate=100, max_per_record=40,
                                               max_records=n_records, seed=seed)

    out = {"n_records": int(len(used_ids)), "n_boot": n_boot, "deploy_rcond": DEPLOY_RCOND,
           "rconds": list(RCONDS), "bootstrap": "record-level (resample records, refit M_s)",
           "configs": {},
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization=NORMALIZATION,
                                   train_ids=used_ids,
                                   extra={"map_is": "deterministic function of estimated M_s and S; not evaluated on held-out records"})}
    for cname, obs in CONFIGS.items():
        obs_set = set(obs)
        targets = [l for l in LEADS if l not in obs_set]
        out["configs"][cname] = {"observed": obs, "segments": {}}
        for s in SEGMENTS:
            X0, rid0 = samples[s]
            if X0.shape[0] < 300:
                continue
            Xc, rid = _clean(X0, rid0)
            M, mu, evr = fit_dipolar_subspace(Xc, rank=3)
            rc_sweep = {}
            for rc in RCONDS:
                k, r = kappa(M, obs, rcond=rc)
                rc_sweep[f"{rc:g}"] = {"rank": int(r), "kappa_global": float(k)}
            eta, etn, amb, kpl = _perlead(M, mu, Xc, obs)

            # ---- record-level bootstrap: resample records, re-pool, refit ----
            uids = np.unique(rid)
            id2rows = {u: np.where(rid == u)[0] for u in uids}
            B_eta = np.zeros((n_boot, 12)); B_etn = np.zeros((n_boot, 12))
            B_amb = np.zeros((n_boot, 12)); B_kap = np.zeros((n_boot, 12))
            # deterministic but INDEPENDENT bootstrap draws per (config, segment) cell,
            # so cross-cell CIs are not perfectly correlated (a shared seed would tie them)
            brng = np.random.default_rng([seed + 1, list(CONFIGS).index(cname), SEGMENTS.index(s)])
            for b in range(n_boot):
                draw = uids[brng.integers(0, uids.size, uids.size)]
                rows = np.concatenate([id2rows[u] for u in draw])
                Xb = Xc[rows]
                Mb, mub, _ = fit_dipolar_subspace(Xb, rank=3)
                Sdb = dipole_coord_cov(Mb, mub, Xb)
                B_eta[b] = eta_per_lead(Mb, obs, rcond=DEPLOY_RCOND)
                B_etn[b] = eta_normalized_per_lead(Mb, obs, rcond=DEPLOY_RCOND)
                B_amb[b] = expected_ambiguity_per_lead(Mb, obs, Sdb, rcond=DEPLOY_RCOND)
                B_kap[b] = kappa_per_lead(Mb, obs, rcond=DEPLOY_RCOND)

            def ci(A, li):
                col = A[:, li]; col = col[np.isfinite(col)]
                if col.size == 0:
                    return [None, None]
                return [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]

            seg = {"dipolar_fraction": float(evr[:3].sum()), "n_records_seg": int(uids.size),
                   "rcond_sweep": rc_sweep, "leads": {}}
            for l in targets:
                li = LEAD_INDEX[l]
                seg["leads"][l] = {
                    "eta": float(eta[li]), "eta_ci": ci(B_eta, li),
                    "eta_normalized": (None if not np.isfinite(etn[li]) else float(etn[li])),
                    "eta_normalized_ci": ci(B_etn, li),
                    "expected_ambiguity_mV": float(amb[li]), "expected_ambiguity_ci": ci(B_amb, li),
                    "kappa": float(kpl[li]), "kappa_ci": ci(B_kap, li),
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
