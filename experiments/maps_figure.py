"""Figure: the target-specific recoverability map.

Reads results/recoverability_maps.json and draws TWO heatmaps (rows = configuration x
segment, columns = the 12 leads):

  (left)  normalized identifiability eta_tilde = eta / ||e_ell^T M_s||_2 -- the PRIMARY,
          scale-free quantity: the fraction of lead ell's dipolar content that lies in
          directions unobserved by S. It is a ratio of L2 norms (a relative unobserved
          GEOMETRIC gain in [0,1]); it is NOT a variance/energy fraction (that would be
          eta_tilde^2). Higher = less recoverable.
  (right) prior-conditional expected ambiguity in mV -- the physical-scale companion.

Observed leads are hatched. Colormap is the colorblind-safe, perceptually-uniform
sequential 'cividis'. Output: results/recoverability_map.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONFIG_ORDER = ("Lead-I", "{I,II,V2}", "{I,II,V1,V3,V5}", "limb-6")
SEGS = ("QRS", "ST", "T")


def _collect(d):
    rows_norm, rows_amb, rowlabels, obs_mask = [], [], [], []
    for cname in CONFIG_ORDER:
        if cname not in d["configs"]:
            continue
        c = d["configs"][cname]; obs = set(c["observed"])
        for s in SEGS:
            seg = c["segments"].get(s)
            if not seg:
                continue
            vn = np.full(12, np.nan); va = np.full(12, np.nan)
            for li, l in enumerate(LEADS):
                if l in obs or l not in seg["leads"]:
                    continue
                ld = seg["leads"][l]
                en = ld.get("eta_normalized")
                vn[li] = np.nan if en is None else en
                va[li] = ld.get("expected_ambiguity_mV", np.nan)
            rows_norm.append(vn); rows_amb.append(va)
            rowlabels.append(f"{cname}  {s}")
            obs_mask.append([l in obs for l in LEADS])
    return np.array(rows_norm), np.array(rows_amb), rowlabels, obs_mask


def main():
    d = json.loads((RESULTS / "recoverability_maps.json").read_text())
    N, A, rowlabels, obs_mask = _collect(d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    nrow = len(rowlabels)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * nrow + 1.4))
    panels = [
        (axes[0], N, r"normalized identifiability $\tilde\eta_{s,\ell}=\eta/\|e_\ell^\top M_s\|_2$"
                     "\n(dimensionless relative unobserved geometric gain; higher = less recoverable)",
         r"$\tilde\eta$ (dimensionless, $[0,1]$)", (0.0, 1.0)),
        (axes[1], A, "prior-conditional expected ambiguity\n(residual a Bayes reconstructor still incurs)",
         "expected ambiguity (mV)", (0.0, float(np.nanpercentile(A, 95)) if np.isfinite(A).any() else 0.3)),
    ]
    for ax, M, title, clabel, (vmin, vmax) in panels:
        im = ax.imshow(M, aspect="auto", cmap="cividis", norm=Normalize(vmin, max(vmax, 1e-3)))
        ax.set_xticks(range(12)); ax.set_xticklabels(LEADS, fontsize=8)
        ax.set_yticks(range(nrow)); ax.set_yticklabels(rowlabels, fontsize=7)
        for r, om in enumerate(obs_mask):
            for cidx, o in enumerate(om):
                if o:
                    ax.add_patch(plt.Rectangle((cidx - 0.5, r - 0.5), 1, 1, fill=True,
                                               facecolor="0.85", edgecolor="0.6", hatch="////", lw=0.3))
        ax.set_title(title, fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(clabel, fontsize=8)
    fig.suptitle("Target-specific recoverability map (grey hatched = observed lead)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "recoverability_map.png", dpi=160)
    print("[fig] results/recoverability_map.png")
    for lab, vn in zip(rowlabels, N):
        hi = [LEADS[i] for i in range(12) if np.isfinite(vn[i]) and vn[i] >= 0.5]
        print(f"  {lab:24s} eta_tilde>=0.5 (least recoverable): {hi}")


if __name__ == "__main__":
    main()
