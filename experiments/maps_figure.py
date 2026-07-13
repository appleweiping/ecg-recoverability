"""Figure: the target-specific recoverability map (per-lead eta identifiability).

Reads results/recoverability_maps.json and draws a heatmap: rows = (configuration,
segment), columns = the 12 leads. Cell = per-lead identifiability eta_{s,ell}(S) (0 =>
identifiable, green; >0 => unidentifiable, red). Observed leads are hatched; the global
kappa (deployment rcond) is annotated per row. This is the paper's core figure -- a
per-feature, per-lead certificate available before any reconstruction.

Output: results/recoverability_map.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONFIG_ORDER = ("Lead-I", "{I,II,V2}", "{I,II,V1,V3,V5}", "limb-6")
SEGS = ("QRS", "ST", "T")


def main():
    d = json.loads((RESULTS / "recoverability_maps.json").read_text())
    rows, rowlabels, obs_mask = [], [], []
    for cname in CONFIG_ORDER:
        if cname not in d["configs"]:
            continue
        c = d["configs"][cname]; obs = set(c["observed"])
        for s in SEGS:
            seg = c["segments"].get(s)
            if not seg:
                continue
            eta = np.full(12, np.nan)
            for li, l in enumerate(LEADS):
                if l in obs:
                    eta[li] = np.nan                       # observed (not a target)
                elif l in seg["leads"]:
                    eta[li] = seg["leads"][l]["eta"]
            rows.append(eta)
            rowlabels.append(f"{cname}  {s}")
            obs_mask.append([l in obs for l in LEADS])
    E = np.array(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.2))
    # identifiable (eta~0) green -> unidentifiable (eta large) red
    vmax = np.nanpercentile(E, 95) if np.isfinite(E).any() else 1.0
    im = ax.imshow(E, aspect="auto", cmap="RdYlGn_r", norm=Normalize(0, max(vmax, 1e-3)))
    ax.set_xticks(range(12)); ax.set_xticklabels(LEADS, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rowlabels, fontsize=7)
    # hatch observed leads
    for r, om in enumerate(obs_mask):
        for cidx, o in enumerate(om):
            if o:
                ax.add_patch(plt.Rectangle((cidx - 0.5, r - 0.5), 1, 1, fill=True,
                                           facecolor="0.85", edgecolor="0.6", hatch="////", lw=0.3))
    ax.set_title(r"Per-lead identifiability $\eta_{s,\ell}(S)$  (green: identifiable, "
                 r"red: unidentifiable; grey hatched: observed)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(r"$\eta_{s,\ell}$ (mV)", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "recoverability_map.png", dpi=160)
    print("[fig] results/recoverability_map.png")
    # console summary
    for lab, eta, om in zip(rowlabels, rows, obs_mask):
        idf = [LEADS[i] for i in range(12) if not om[i] and np.isfinite(eta[i]) and eta[i] < 1e-3]
        unid = [LEADS[i] for i in range(12) if not om[i] and np.isfinite(eta[i]) and eta[i] >= 1e-3]
        print(f"  {lab:24s} identifiable={idf} unidentifiable={unid}")


if __name__ == "__main__":
    main()
