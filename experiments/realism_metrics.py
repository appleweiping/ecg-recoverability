"""Does CFG guidance actually improve REALISM (P0-4)? GPU.

The "realism-recoverability trade-off" claim requires showing that, as guidance w rises,
independent REALISM metrics of the generated (unobserved) leads improve toward the real
distribution WHILE the achievability gap worsens. If realism does not improve with w, the
claim is downgraded to "CFG changes the residual-recovery behaviour of this model" -- no
"fabrication is a property of the objective".

Loads the leakage-fixed arbitrary-mask DDPM (results/gpu_ddpm.pt) and, per guidance w,
reconstructs the primary config {I,II,V1,V3,V5}->{V2,V4,V6} on held-out NORM records; for
each target lead it compares GENERATED vs REAL distributions with metrics that need no
extra model and are robust to delineation noise:
  * PSD distance  -- mean L2 distance of the log power spectral density;
  * amp-Wasserstein -- 1-Wasserstein distance of the per-record peak-to-peak amplitude
                       distribution;
  * QRS-width Wass  -- 1-Wasserstein of NeuroKit QRS-duration distribution (best effort;
                       reports delineation success rate);
alongside the achievability gap (oracle - postmean rho). Reports mean +/- record-bootstrap
CI per w and the trade-off (Spearman corr of realism-improvement vs gap across w).

Output: results/realism_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance

from ecgcert.data import PTBXL
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
OBS = ["I", "II", "V1", "V3", "V5"]
TARGETS = ["V2", "V4", "V6"]
GUIDANCES = (1.0, 2.0, 4.0, 6.0)


def _load_model(T_default=200):
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor
    # our own artifact saved by gpu_diffusion_clean.py (tensors + numpy scale); trusted.
    ck = torch.load(RESULTS / "gpu_ddpm.pt", map_location="cuda", weights_only=False)
    T = int(ck.get("T", T_default))
    m = DiffusionReconstructor(T=T, base=64, device="cuda", cond_dropout=0.15, seed=0)
    m.net.load_state_dict(ck["sd"]); m.net.eval()
    return m, np.asarray(ck["scale"], dtype=np.float32), T


def _psd_logdist(a, b, fs=100):
    fa, Pa = welch(a, fs=fs, nperseg=min(256, len(a)))
    fb, Pb = welch(b, fs=fs, nperseg=min(256, len(b)))
    la, lb = np.log(Pa + 1e-12), np.log(Pb + 1e-12)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def _qrs_widths(sig12, fs=100, lead=1):
    import neurokit2 as nk
    try:
        _, rp = nk.ecg_peaks(sig12[:, lead], sampling_rate=fs)
        _, w = nk.ecg_delineate(sig12[:, lead], rp, sampling_rate=fs, method="dwt")
        on = np.asarray(w.get("ECG_R_Onsets", []), float); off = np.asarray(w.get("ECG_R_Offsets", []), float)
        on = on[~np.isnan(on)]; off = off[~np.isnan(off)]
        widths = []
        for a in on:
            later = off[off > a]
            if later.size:
                widths.append((later[0] - a) / fs * 1000.0)  # ms
        return widths
    except Exception:
        return []


def run(n_test=300, seed=0, n_boot=200):
    import torch
    db = PTBXL()
    rng = np.random.default_rng(seed)
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm10))[:n_test]
    obs_idx = tuple(LEAD_INDEX[l] for l in OBS)
    tgt_idx = [LEAD_INDEX[l] for l in TARGETS]

    # load real test signals (track KEPT ids for honest lineage)
    sigs, kept_ids = [], []
    for eid in test_ids:
        try:
            s = db.signal(int(eid), rate=100)[:1000]
        except Exception:
            continue
        if s.shape[0] == 1000 and np.all(np.isfinite(s)):
            sigs.append(s.astype(np.float32)); kept_ids.append(int(eid))
    print(f"[realism] {len(sigs)} test records", flush=True)
    model, scale, T = _load_model()
    yn = np.stack([(s.T / scale[:, None]) for s in sigs])          # (N,12,1000)

    from ecgcert import lineage
    ckpt = str(RESULTS / "gpu_ddpm.pt")
    out = {"n_test": len(sigs), "guidances": list(GUIDANCES), "obs": OBS, "targets": TARGETS,
           "per_w": {},
           "lineage": lineage.make(db, seed=seed, targets=list(TARGETS),
                                   normalization="per-lead 95th-pct |amp| (train)",
                                   test_ids=kept_ids, checkpoint=ckpt,
                                   extra={"experiment": "diffusion realism vs guidance (abandoned pre-submission hypothesis)",
                                          "kept_record_ids_sha256": lineage.ids_sha256(kept_ids)})}
    # real TARGET-lead amplitude distribution (once). We drop the previous QRS-width metric:
    # it was delineated on the OBSERVED Lead II, which is kept exact, so it was vacuous.
    real_pp = {l: np.array([np.ptp(s[:, LEAD_INDEX[l]]) for s in sigs]) for l in TARGETS}
    out["metric_note"] = ("Realism is measured on the TARGET (unobserved) leads only "
                          "(PSD log-distance, amplitude Wasserstein). The prior QRS-width metric "
                          "on the observed Lead II was vacuous and is removed.")

    for w in GUIDANCES:
        # reconstruct in batches; noise seed is per-BATCH (deterministic) and SHARED across
        # guidance, so guidance sweeps use the same noise realization per batch.
        rec = []
        for bi, i in enumerate(range(0, len(sigs), 128)):
            r = model.sample(yn[i:i + 128], obs_idx=obs_idx, guidance=w, replace=True, steps=T, seed=1000 + bi)
            rec.extend([(r[j] * scale[:, None]) for j in range(r.shape[0])])   # (12,1000) mV
        # PSD distance + amplitude Wasserstein per TARGET lead
        psd = {l: [] for l in TARGETS}; gen_pp = {l: [] for l in TARGETS}
        for s, Lhat in zip(sigs, rec):
            for l in TARGETS:
                li = LEAD_INDEX[l]
                psd[l].append(_psd_logdist(Lhat[li], s[:, li]))
                gen_pp[l].append(np.ptp(Lhat[li]))
        row = {"psd_logdist": {l: float(np.mean(psd[l])) for l in TARGETS},
               "amp_wasserstein": {l: float(wasserstein_distance(gen_pp[l], real_pp[l])) for l in TARGETS}}
        out["per_w"][str(w)] = row
        print(f"  [w={w}] PSD={np.mean([row['psd_logdist'][l] for l in TARGETS]):.3f} "
              f"ampW={np.mean([row['amp_wasserstein'][l] for l in TARGETS]):.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "realism_metrics.json").write_text(json.dumps(out, indent=2))
    print("[json] results/realism_metrics.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()
    run(n_test=args.n_test)
