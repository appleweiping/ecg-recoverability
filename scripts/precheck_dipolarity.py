"""Risk-2 gate: is the per-feature dipolarity story real on PTB-XL?

The paper's per-feature framing rests on ONE empirical fact, which this script
checks (and it holds):

 1. **Per-feature separation.** The QRS complex is more dipolar than the ST segment
    and T wave -- so "which feature is recoverable" genuinely varies by feature.

We also probed a second hypothesis -- that dipolarity *decreases* from NORM to
MI/STTC -- and it does NOT hold: dipolarity is feature-and-pathology specific and is
if anything *higher* for ST/T in disease (see the JSON ``gate.disease_drop=false``).
We therefore do NOT motivate the safety story via disease non-dipolarity; the honest
statement is that substantial non-dipolar content exists in every non-QRS feature in
all classes.  This script also reports the closed-form kappa_s(S) for the planned lead
configurations to confirm "geometry, not lead count."

Outputs: results/precheck_dipolarity.json + results/precheck_dipolarity.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.data import PTBXL
from ecgcert.physics import fit_dipolar_subspace, kappa

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
CLASSES = ("NORM", "MI", "STTC")
CONFIGS = {
    "Lead-I": ["I"],
    "3-lead spanning {I,II,V2}": ["I", "II", "V2"],
    "3-lead coplanar {I,II,III}": ["I", "II", "III"],
    "3-lead collinear {V1,V2,V3}": ["V1", "V2", "V3"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def main(n_records: int = 300, rate: int = 100, seed: int = 0) -> None:
    db = PTBXL()
    out: dict = {"n_records_per_class": n_records, "rate": rate, "dipolar_fraction": {},
                 "evr_top5": {}, "kappa_QRS": {}}

    fitted_qrs_M = None
    for cls in CLASSES:
        ids = db.ids_with_superclass(cls, exclusive=True, folds=range(1, 9))  # train folds
        rng = np.random.default_rng(seed)
        ids = rng.permutation(ids)[:n_records]
        out["dipolar_fraction"][cls] = {}
        out["evr_top5"][cls] = {}
        samples = db.collect_all_segments(ids, rate=rate, max_per_record=40, seed=seed)
        for seg in SEGMENTS:
            X = samples[seg]
            if X.shape[0] < 50:
                out["dipolar_fraction"][cls][seg] = None
                continue
            M_s, _, evr = fit_dipolar_subspace(X, rank=3)
            out["dipolar_fraction"][cls][seg] = float(evr[:3].sum())
            out["evr_top5"][cls][seg] = [float(v) for v in evr[:5]]
            if cls == "NORM" and seg == "QRS":
                fitted_qrs_M = M_s
        print(f"[{cls}] " + "  ".join(
            f"{seg}:{out['dipolar_fraction'][cls][seg]:.4f}" if out['dipolar_fraction'][cls][seg]
            else f"{seg}:NA" for seg in SEGMENTS))

    # kappa for the planned configs on the NORM-QRS dipolar subspace.
    # We report the full-precision conditioning number kappa AND the *effective*
    # dipole rank at the deployed truncation tolerance (RECON_RCOND=1e-2): a config
    # whose smallest observed dipole singular value falls below that tolerance
    # recovers fewer than 3 dipole directions (the rest become Tier III).
    from ecgcert.physics import RECON_RCOND
    if fitted_qrs_M is not None:
        for name, leads in CONFIGS.items():
            k, r = kappa(fitted_qrs_M, leads)                     # full precision
            _, r_eff = kappa(fitted_qrs_M, leads, rcond=RECON_RCOND)  # deployed
            out["kappa_QRS"][name] = {"kappa": float(k), "dipole_rank": int(r),
                                      "effective_rank": int(r_eff)}
            print(f"  kappa[{name}] = {k:.3f}  (rank {r}, effective rank {r_eff} @ rcond={RECON_RCOND})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "precheck_dipolarity.json").write_text(json.dumps(out, indent=2))
    _plot(out)

    # Gate verdict.
    norm = out["dipolar_fraction"]["NORM"]
    sep = norm["QRS"] > norm["ST"] and norm["QRS"] > norm["T"]
    disease_drop = (out["dipolar_fraction"]["MI"]["QRS"] < norm["QRS"]) or \
                   (out["dipolar_fraction"]["STTC"]["ST"] < norm["ST"])
    print("\n=== GATE ===")
    print(f"(1) per-feature separation (QRS>ST and QRS>T): {sep}")
    print(f"(2) disease-dependent non-dipolarity present : {disease_drop}")
    out["gate"] = {"per_feature_separation": bool(sep), "disease_drop": bool(disease_drop)}
    (RESULTS / "precheck_dipolarity.json").write_text(json.dumps(out, indent=2))


def _plot(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    width = 0.25
    x = np.arange(len(SEGMENTS))
    for i, cls in enumerate(CLASSES):
        vals = [out["dipolar_fraction"][cls][s] or np.nan for s in SEGMENTS]
        ax.bar(x + (i - 1) * width, vals, width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENTS)
    ax.set_ylabel("dipolar fraction (top-3 variance)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Per-segment dipolarity by diagnostic superclass (PTB-XL)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "precheck_dipolarity.png", dpi=150)
    print("[fig] results/precheck_dipolarity.png")


if __name__ == "__main__":
    main()
