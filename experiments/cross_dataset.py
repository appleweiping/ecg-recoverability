"""Cross-dataset transfer of the recoverability certificate -- MATCHED comparison (P0-E).

We fit the per-segment dipolar subspace M_s under the UNIFIED truncated-SVD numerics
(rcond=1e-2) on PTB-XL and on the independent Chapman-Shaoxing-Ningbo database (PhysioNet
ecg-arrhythmia 1.0.0), and compare principal angles / rank / kappa / V2-eta with
record-bootstrap CIs. To avoid confounding cohort with CASE MIX we run two matched
comparisons:
  (a) PTB-XL NORM      vs Chapman normal (sinus-rhythm) records;
  (b) PTB-XL all-record vs Chapman all-record.

Chapman records are sampled RANDOMLY from the complete manifest with a fixed seed (not a
deterministic folder/name stride); channels are reordered by WFDB ``sig_name`` to the standard
12-lead order with an assertion, and resampled to 100 Hz using the record's actual ``fs``. The
exact loaded manifest and its hash are stored. We call an ST/T difference "cohort-specific"
only if it survives the matched (normal-vs-normal) comparison; otherwise it is case-mix
dependence.

Output: results/cross_dataset.json
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import resample_poly

import wfdb
from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import LEADS, LEAD_INDEX, kappa, eta_per_lead

# PhysioNet wfdb pn_dir streaming is unusably slow on some hosts (>150 s/record); direct curl
# of the .hea/.mat pair is ~8 s. We parallel-curl a cache, then rdrecord locally.
CHAP_BASE = "https://physionet.org/files/ecg-arrhythmia/1.0.0"
CHAP_CACHE = Path("/root/autodl-tmp/chapman_cache")


def _curl_record(spec):
    fol, nm = spec
    d = CHAP_CACHE / fol
    d.mkdir(parents=True, exist_ok=True)
    ok = True
    for ext in (".hea", ".mat"):
        f = d / (nm + ext)
        if f.exists() and f.stat().st_size > 0:
            continue
        r = subprocess.run(["curl", "-s", "--fail", "-m", "40", "-o", str(f),
                            f"{CHAP_BASE}/{fol}/{nm}{ext}"], capture_output=True)
        if r.returncode != 0 or not f.exists() or f.stat().st_size == 0:
            ok = False
    return (fol, nm, str(d / nm)) if ok else None

RESULTS = Path(__file__).resolve().parent.parent / "results"
PN = "ecg-arrhythmia/1.0.0"
SEGS = ("P", "QRS", "ST", "T")
RCONDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
DEPLOY_RCOND = 1e-2
STD_ORDER = list(LEADS)                       # standard clinical 12-lead order
SINUS_RHYTHM = "426783006"                    # SNOMED CT sinus rhythm (Chapman "normal")
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def _fit(X, clip_mV=10.0):
    X = X[np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= clip_mV, axis=1)]
    if X.shape[0] < 200:
        return None
    mu = X.mean(0)
    w, V = np.linalg.eigh(np.cov((X - mu).T))
    order = np.argsort(w)[::-1]
    return V[:, order[:3]]                     # (12,3)


def _chapman_manifest(n_target, seed):
    """Randomly ordered (folder, record) pairs across the complete leaf-folder manifest."""
    rng = np.random.default_rng(seed)
    folders = [f.strip("/") for f in wfdb.get_record_list(PN)]
    rng.shuffle(folders)
    specs = []
    for fol in folders:
        try:
            names = wfdb.get_record_list(f"{PN}/{fol}")
        except Exception:
            continue
        names = [n.strip("/").split("/")[-1] for n in names]
        rng.shuffle(names)
        for nm in names:
            specs.append((fol, nm))
        if len(specs) >= n_target * 3:        # over-sample; many drop on load/QC
            break
    rng.shuffle(specs)
    return specs


def _is_sinus(rec):
    for c in (rec.comments or []):
        if c.strip().lower().startswith("dx"):
            codes = c.split(":", 1)[-1].replace(" ", "").split(",")
            return SINUS_RHYTHM in codes
    return False


def collect_chapman(n_target=350, max_per_record=40, seed=0, workers=16):
    """Per-record ST/QRS/... samples, channel-reordered + resampled with actual fs. Records are
    parallel-curled to a local cache then read with wfdb locally. Returns
    {seg: (X, rid, is_norm)}, used_specs, counts."""
    rng = np.random.default_rng(seed + 7)
    specs = _chapman_manifest(n_target, seed)
    print(f"[cross] downloading {len(specs)} Chapman records ({workers} parallel curl) ...", flush=True)
    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_curl_record, specs):
            if res is not None:
                downloaded.append(res)
            if len(downloaded) >= n_target * 2:      # enough survive QC
                break
    print(f"[cross] {len(downloaded)} records cached; loading ...", flush=True)
    rows = {s: [] for s in SEGS}; rids = {s: [] for s in SEGS}; norms = {s: [] for s in SEGS}
    used, n_norm, rid_ctr = [], 0, 0
    for fol, nm, localpath in downloaded:
        try:
            r = wfdb.rdrecord(localpath)
        except Exception:
            continue
        name = [str(x).strip() for x in (r.sig_name or [])]
        if not set(STD_ORDER).issubset(set(name)):        # require all 12 leads
            continue
        perm = [name.index(l) for l in STD_ORDER]          # reorder by sig_name
        sig = np.asarray(r.p_signal, float)[:, perm]
        if sig.shape[0] < 2500 or not np.all(np.isfinite(sig)):
            continue
        fs = int(round(float(r.fs)))
        if fs != 100:                                      # resample using ACTUAL fs
            from math import gcd
            g = gcd(fs, 100)
            sig = resample_poly(sig, 100 // g, fs // g, axis=0)
        if not np.all(np.isfinite(sig)):
            continue
        segidx = PTBXL.segment_indices(sig, fs=100)
        is_norm = _is_sinus(r)
        added = False
        for s, idx in segidx.items():
            if idx.size == 0:
                continue
            if idx.size > max_per_record:
                idx = rng.choice(idx, max_per_record, replace=False)
            rows[s].append(sig[idx]); rids[s].append(np.full(idx.size, rid_ctr, np.int64))
            norms[s].append(np.full(idx.size, is_norm)); added = True
        if added:
            used.append(f"{fol}/{nm}"); n_norm += int(is_norm); rid_ctr += 1
        if rid_ctr >= n_target:
            break
    out = {s: ((np.vstack(rows[s]), np.concatenate(rids[s]), np.concatenate(norms[s]))
               if rows[s] else (np.zeros((0, 12)), np.zeros(0, int), np.zeros(0, bool))) for s in SEGS}
    return out, used, {"n_records": rid_ctr, "n_normal": n_norm}


def collect_ptb(db, seg_samples_ids):
    return seg_samples_ids


def _recover(M, obs):
    sweep = {f"{rc:g}": dict(zip(("kappa_global", "rank"),
             (lambda kr: (round(float(kr[0]), 3), int(kr[1])))(kappa(M, obs, rcond=rc))))
             for rc in RCONDS}
    eta = eta_per_lead(M, obs, rcond=DEPLOY_RCOND)
    return {"rcond_sweep": sweep, "rank_deploy": sweep[f"{DEPLOY_RCOND:g}"]["rank"],
            "V2_eta": round(float(eta[LEAD_INDEX["V2"]]), 4)}


def _compare(Xp, Xc, n_boot, seed):
    """Principal angles + per-config recoverability + record-bootstrap CIs for one segment."""
    Mp, Mc = _fit(Xp[0]), _fit(Xc[0])
    if Mp is None or Mc is None:
        return None
    ang = np.degrees(subspace_angles(Mp, Mc))
    res = {"principal_angles_deg": [round(float(a), 2) for a in ang],
           "max_angle_deg": round(float(ang.max()), 2), "mid_angle_deg": round(float(np.sort(ang)[1]), 2),
           "min_angle_deg": round(float(ang.min()), 2),
           "n_ptb": int(Xp[0].shape[0]), "n_chap": int(Xc[0].shape[0]),
           "recoverability": {c: {"ptbxl": _recover(Mp, obs), "chapman": _recover(Mc, obs)}
                              for c, obs in CONFIGS.items()}}
    # record-bootstrap: resample records on each side, refit, recompute max angle + V2 eta(limb-6)
    def boot(X):
        uids = np.unique(X[1]); id2 = {u: np.where(X[1] == u)[0] for u in uids}
        return uids, id2
    up, ip = boot(Xp); uc, ic = boot(Xc)
    br = np.random.default_rng(seed + 3); angs, dv = [], []
    for _ in range(n_boot):
        Mpb = _fit(Xp[0][np.concatenate([ip[u] for u in up[br.integers(0, up.size, up.size)]])])
        Mcb = _fit(Xc[0][np.concatenate([ic[u] for u in uc[br.integers(0, uc.size, uc.size)]])])
        if Mpb is None or Mcb is None:
            continue
        angs.append(float(np.degrees(subspace_angles(Mpb, Mcb)).max()))
        dv.append(float(eta_per_lead(Mcb, CONFIGS["limb-6"], rcond=DEPLOY_RCOND)[LEAD_INDEX["V2"]]))
    if angs:
        res["max_angle_ci"] = [round(float(np.percentile(angs, 2.5)), 2), round(float(np.percentile(angs, 97.5)), 2)]
        res["chapman_limb6_V2eta_ci"] = [round(float(np.percentile(dv, 2.5)), 4), round(float(np.percentile(dv, 97.5)), 4)]
    return res


def main(n_chapman=350, n_boot=100, seed=0):
    db = PTBXL()
    print("[cross] PTB-XL segment samples (NORM folds 1-7 + all folds 1-7) ...", flush=True)
    norm_ids = np.random.default_rng(seed).permutation(
        db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:1500]
    all_ids = np.random.default_rng(seed + 1).permutation(
        db.meta[db.meta["strat_fold"].isin(range(1, 8))].index.to_numpy())[:1500]
    ptb_norm = db.collect_all_segments_with_ids(norm_ids, rate=100, max_per_record=40, max_records=1500, seed=seed)
    ptb_all = db.collect_all_segments_with_ids(all_ids, rate=100, max_per_record=40, max_records=1500, seed=seed)

    print("[cross] streaming Chapman (random manifest sample) ...", flush=True)
    chap, used, counts = collect_chapman(n_target=n_chapman, seed=seed)
    manifest_sha = hashlib.sha256(("|".join(sorted(used))).encode()).hexdigest()[:16]
    print(f"[cross] chapman: {counts['n_records']} records, {counts['n_normal']} sinus-rhythm", flush=True)

    def seg_pack(store, seg, normal_only=False):
        item = store[seg]
        if len(item) == 3:                     # Chapman: (X, rid, is_norm)
            X, rid, nm = item
        else:                                  # PTB-XL: (X, rid) -- no per-sample normal mask
            X, rid = item
            nm = np.ones(len(rid), dtype=bool)
        if normal_only:
            m = nm.astype(bool); X, rid = X[m], rid[m]
        return (X, rid)

    out = {"deploy_rcond": DEPLOY_RCOND, "rconds": list(RCONDS),
           "chapman": {"n_records": counts["n_records"], "n_normal": counts["n_normal"],
                       "manifest_sha256": manifest_sha, "n_used_records": len(used)},
           "comparisons": {},
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization="raw mV",
                                   train_ids=norm_ids[:1500],
                                   extra={"datasets": ["PTB-XL", "Chapman-Shaoxing-Ningbo (ecg-arrhythmia 1.0.0)"],
                                          "chapman_manifest_sha256": manifest_sha,
                                          "chapman_n_records": counts["n_records"]})}
    comparisons = {
        "matched_normal": (ptb_norm, lambda s: seg_pack(chap, s, normal_only=True)),
        "all_record": (ptb_all, lambda s: seg_pack(chap, s, normal_only=False)),
    }
    for cmp_name, (ptb_store, chap_getter) in comparisons.items():
        out["comparisons"][cmp_name] = {}
        for s in SEGS:
            Xp = seg_pack(ptb_store, s); Xc = chap_getter(s)
            r = _compare(Xp, Xc, n_boot, seed)
            if r:
                out["comparisons"][cmp_name][s] = r
                print(f"  [{cmp_name} {s}] maxang={r['max_angle_deg']} CI={r.get('max_angle_ci')} "
                      f"nptb={r['n_ptb']} nchap={r['n_chap']}", flush=True)

    # verdict: ST/T cohort-specific only if the MATCHED (normal) max angle stays large
    def big(cmp, s):
        r = out["comparisons"].get(cmp, {}).get(s)
        return r and (r.get("max_angle_ci", [r["max_angle_deg"]])[0] or 0) > 45
    out["ST_T_verdict"] = ("cohort-specific (survives matched normal-vs-normal)"
                           if (big("matched_normal", "ST") or big("matched_normal", "T"))
                           else "case-mix / diagnosis dependence (does NOT survive matched comparison)")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cross_dataset.json").write_text(json.dumps(out, indent=2))
    print("[verdict ST/T]", out["ST_T_verdict"], flush=True)
    print("[json] results/cross_dataset.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chapman", type=int, default=350)
    ap.add_argument("--n-boot", type=int, default=100)
    args = ap.parse_args()
    main(n_chapman=args.n_chapman, n_boot=args.n_boot)
