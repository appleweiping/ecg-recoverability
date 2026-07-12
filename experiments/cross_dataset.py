"""Cross-dataset validation of the recoverability certificate's physical core.

Claim: the certificate's only data-estimated object, the per-segment dipolar subspace
M_s (and hence the closed-form conditioning kappa_s(S)), is DATASET-INDEPENDENT. We fit
M_s with IDENTICAL processing on two geographically independent hospital populations:
  - PTB-XL (German, Physikalisch-Technische Bundesanstalt)      [local]
  - Chapman-Shaoxing-Ningbo (Chinese, PhysioNet ecg-arrhythmia 1.0.0)  [streamed]
and compare the subspaces by PRINCIPAL ANGLES; we also recompute kappa on both.

Runs on the GPU box (PTB-XL local + PhysioNet reachable). CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.signal import resample_poly

import wfdb
from ecgcert.data import PTBXL
from ecgcert.physics.dipolar_subspace import kappa


def fit_M(samples, rank=3, clip_mV=10.0):
    """Robust per-segment dipolar basis via eigendecomposition of the 12x12 covariance
    (eigh always converges, unlike LAPACK gesdd SVD on extreme-amplitude rows). Drops
    non-finite and outlier (|x|>clip_mV) rows. Returns {seg: (M(12,3), mu(12,), evr)}."""
    out = {}
    for s, X in samples.items():
        if X.size == 0:
            continue
        X = X[np.all(np.isfinite(X), axis=1)]
        X = X[np.all(np.abs(X) <= clip_mV, axis=1)]
        if X.shape[0] < 200:
            continue
        mu = X.mean(0)
        C = np.cov((X - mu).T)                       # (12,12) symmetric PSD
        w, V = np.linalg.eigh(C)                      # ascending eigenvalues
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        M = V[:, :rank]                              # top-3 spatial directions
        evr = w[:rank] / max(w.sum(), 1e-12)
        out[s] = (M, mu, evr, int(X.shape[0]))
    return out

RESULTS = Path(__file__).resolve().parent.parent / "results"
PN = "ecg-arrhythmia/1.0.0"
SEGS = ("P", "QRS", "ST", "T")
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "{V1,V2,V3}": ["V1", "V2", "V3"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def chapman_record_specs(folder_stride=5, per_folder=12, cap=1000):
    """Return [(pn_subdir, record_name)] spread across the nested leaf folders
    (WFDBRecords/XX/XXX/), which span the 3 source hospitals."""
    folders = wfdb.get_record_list(PN)[::folder_stride]      # ~90 of 452 leaf folders
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


def collect_chapman(specs, max_per_record=40, max_ok=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = {s: [] for s in SEGS}
    ok = 0
    for fol, name in specs:
        try:
            r = wfdb.rdrecord(name, pn_dir=f"{PN}/{fol}")
            sig = r.p_signal.astype(float)                    # (5000,12) @500Hz, standard order
        except Exception:
            continue
        if sig.shape[1] != 12 or sig.shape[0] < 2500:
            continue
        if not np.all(np.isfinite(sig)):                      # some Chapman records carry NaN leads
            continue
        sig = resample_poly(sig, 1, 5, axis=0)                # -> ~100Hz, matches PTB-XL processing
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


def main():
    print("[cross] fitting PTB-XL M_s (NORM, 100Hz, lead-II dwt) ...", flush=True)
    db = PTBXL()
    norm = db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9))
    ptb_samples = db.collect_all_segments(norm, rate=100, max_per_record=40, max_records=1500, seed=0)
    ptb_models = fit_M(ptb_samples)

    print("[cross] streaming Chapman-Shaoxing-Ningbo subset ...", flush=True)
    # ~350 records spread across leaf folders is ample for a stable 12x3 subspace
    # (each contributes ~40 samples/segment); PhysioNet streaming is the bottleneck.
    specs = chapman_record_specs(folder_stride=8, per_folder=8, cap=350)
    print(f"[cross] {len(specs)} record specs across leaf folders", flush=True)
    chap_samples, n_chap = collect_chapman(specs, max_ok=350)
    chap_models = fit_M(chap_samples)
    print(f"[cross] chapman usable records: {n_chap}", flush=True)

    out = {"n_chapman_records": n_chap, "processing": "100Hz, lead-II dwt, top-3 spatial eig",
           "segments": {}, "kappa_QRS": {}}
    for s in SEGS:
        if s not in ptb_models or s not in chap_models:
            continue
        (Mp, _, evp, np_) = ptb_models[s]
        (Mc, _, evc, nc_) = chap_models[s]
        ang = np.degrees(subspace_angles(Mp, Mc))            # 3 principal angles (deg), ascending
        out["segments"][s] = {
            "principal_angles_deg": [round(float(a), 2) for a in ang],
            "max_angle_deg": round(float(np.max(ang)), 2),
            "ptbxl_evr3": [round(float(x), 3) for x in np.asarray(evp)[:3]],
            "chapman_evr3": [round(float(x), 3) for x in np.asarray(evc)[:3]],
            "n_ptb": np_, "n_chap": nc_,
        }
    for name, leads in CONFIGS.items():
        kp, rp = kappa(ptb_models["QRS"][0], leads)
        kc, rc = kappa(chap_models["QRS"][0], leads)
        out["kappa_QRS"][name] = {"ptbxl": round(float(kp), 2), "chapman": round(float(kc), 2),
                                  "rank_ptb": int(rp), "rank_chap": int(rc)}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cross_dataset.json").write_text(json.dumps(out, indent=2))
    print("\n=== PRINCIPAL ANGLES (deg) between PTB-XL and Chapman M_s ===")
    for s in ("QRS", "ST", "T", "P"):
        if s in out["segments"]:
            d = out["segments"][s]
            print(f"  {s:3s}: angles={d['principal_angles_deg']}  max={d['max_angle_deg']}  "
                  f"evr(ptb)={d['ptbxl_evr3']} evr(chap)={d['chapman_evr3']}")
    print("\n=== kappa_QRS (config conditioning) PTB-XL vs Chapman ===")
    for name, k in out["kappa_QRS"].items():
        print(f"  {name:18s}: ptb={k['ptbxl']:>10} (rank {k['rank_ptb']})  "
              f"chap={k['chapman']:>10} (rank {k['rank_chap']})")
    print("\n[json] results/cross_dataset.json")


def emit_macros():
    """Write paper/_cross_macros.tex from results/cross_dataset.json (no box needed)."""
    paper = Path(__file__).resolve().parent.parent / "paper"
    d = json.loads((RESULTS / "cross_dataset.json").read_text())
    import math
    qrs = d["segments"]["QRS"]
    def dip(seg): return sum(d["segments"][seg]["ptbxl_evr3"]), sum(d["segments"][seg]["chapman_evr3"])
    dQ = dip("QRS"); dS = dip("ST"); dT = dip("T")
    spanS = d["kappa_QRS"].get("{I,II,V1,V3,V5}", {})   # the diffusion exhibit's config
    lines = [
        "% auto-generated by experiments/cross_dataset.py --emit-macros",
        f"\\newcommand{{\\crossN}}{{{d['n_chapman_records']}}}",
        f"\\newcommand{{\\crossQRSang}}{{{math.ceil(qrs['max_angle_deg'])}}}",   # QRS dominant dirs align (ceil)
        f"\\newcommand{{\\crossDipPtb}}{{{dQ[0]:.2f}}}",                         # QRS dipolarity PTB-XL
        f"\\newcommand{{\\crossDipChap}}{{{dQ[1]:.2f}}}",                        # QRS dipolarity Chapman
        f"\\newcommand{{\\crossDipSTptb}}{{{dS[0]:.2f}}}",
        f"\\newcommand{{\\crossDipSTchap}}{{{dS[1]:.2f}}}",
        f"\\newcommand{{\\crossDipTptb}}{{{dT[0]:.2f}}}",
        f"\\newcommand{{\\crossDipTchap}}{{{dT[1]:.2f}}}",
        f"\\newcommand{{\\crossKapSpanPtb}}{{{spanS.get('ptbxl', float('nan')):.1f}}}",
        f"\\newcommand{{\\crossKapSpanChap}}{{{spanS.get('chapman', float('nan')):.1f}}}",
    ]
    (paper / "_cross_macros.tex").write_text("\n".join(lines) + "\n")
    print("[tex] paper/_cross_macros.tex\n" + "\n".join(lines))


if __name__ == "__main__":
    import sys
    if "--emit-macros" in sys.argv:
        emit_macros()
    else:
        main()
