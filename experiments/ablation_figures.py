"""Ablation / robustness figures for the recoverability certificate.

Promotes the sensitivities already stored in results/recoverability_maps.json (rank sweep,
rho/truncation sweep, NORM-vs-MI subspace angle) and results/lead_weighting.json (12-vs-8-lead
subspace angle) from buried JSON into first-class figures. NO experiment re-run -- pure matplotlib
over existing lineage-stamped results.

Output: results/ablations.png  (three panels)
  (a) rank sweep r=1..5: cumulative explained-variance of the per-segment dipolar subspace, with
      the a-priori rank-3 choice marked -- rank 3 captures the bulk (QRS/ST) a priori.
  (b) rho (truncation) sweep: global kappa vs rho for each observed-lead config on a representative
      segment -- well-conditioned spanning sets are rho-stable; near-rank-deficient sets are not.
  (c) subspace-angle sensitivity: max principal angle per segment for NORM-vs-MI (diagnosis shift)
      and 12-vs-8-lead (limb-weighting), i.e. how much the estimated M_s moves -- the map is a
      NORM-trained reference and the fine ST/T ordering is weighting-sensitive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGS = ("P", "QRS", "ST", "T")
CONFIG_ORDER = ("Lead-I", "{I,II,V2}", "{I,II,V1,V3,V5}", "limb-6")
RANK_CHOICE = 3


def main():
    maps = json.loads((RESULTS / "recoverability_maps.json").read_text())
    lw = json.loads((RESULTS / "lead_weighting.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.2))

    # ---- (a) rank sweep: cumulative EVR per segment ----
    ax = axes[0]
    rs = maps.get("rank_sensitivity", {}).get("segments", {})
    colors = {"P": "0.6", "QRS": "C0", "ST": "C3", "T": "C2"}
    for s in SEGS:
        ce = rs.get(s, {}).get("cumulative_evr")
        if not ce:
            continue
        ranks = sorted(int(k) for k in ce)
        ys = [ce[str(r)] for r in ranks]
        ax.plot(ranks, ys, "-o", ms=4, color=colors.get(s, "0.4"), label=s)
    ax.axvline(RANK_CHOICE, color="0.3", ls="--", lw=1)
    ax.text(RANK_CHOICE + 0.06, 0.06, f"a-priori rank {RANK_CHOICE}\n(cardiac dipole is 3-D)",
            fontsize=7.5, color="0.3")
    ax.set_xlabel("retained rank $r$"); ax.set_ylabel("cumulative explained variance")
    ax.set_title("(a) Rank sweep of the dipolar subspace", fontsize=9.5)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, title="segment")
    ax.grid(alpha=0.3)

    # ---- (b) rho sweep: global kappa vs rho per config (representative segment) ----
    ax = axes[1]
    seg_b = "QRS"
    rconds = sorted(float(r) for r in maps.get("rconds", [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]))
    for ci, cname in enumerate(CONFIG_ORDER):
        node = maps.get("configs", {}).get(cname, {}).get("segments", {}).get(seg_b, {})
        sweep = node.get("rcond_sweep")
        if not sweep:
            continue
        xs, ys, ranks = [], [], []
        for rc in rconds:
            key = f"{rc:g}"
            if key in sweep:
                xs.append(rc); ys.append(sweep[key]["kappa_global"]); ranks.append(sweep[key]["rank"])
        if xs:
            ax.plot(xs, ys, "-o", ms=4, color=f"C{ci}", label=f"{cname} (rank {ranks[-1]})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"truncation tolerance $\varrho$"); ax.set_ylabel(r"global conditioning $\kappa_s(S)$")
    ax.set_title(rf"(b) $\varrho$-stability of conditioning ({seg_b})", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")

    # ---- (c) subspace-angle sensitivity: NORM-vs-MI and 12-vs-8 max principal angle ----
    ax = axes[2]
    ds = maps.get("diagnosis_sensitivity", {})
    segs_c = [s for s in SEGS if s in ds or s in lw.get("segments", {})]
    x = np.arange(len(segs_c)); w = 0.38
    mi = [ds.get(s, {}).get("max_angle_deg", np.nan) for s in segs_c]
    l8 = [lw.get("segments", {}).get(s, {}).get("max_angle_deg", np.nan) for s in segs_c]
    ax.bar(x - w / 2, mi, w, color="C4", label="NORM vs MI subspace")
    ax.bar(x + w / 2, l8, w, color="C1", label="12-lead vs 8-lead fit")
    ax.axhline(45, color="0.5", ls=":", lw=1)
    ax.text(len(segs_c) - 1.2, 46.5, "45$^\\circ$ (orthogonal-ish)", fontsize=7, color="0.5")
    ax.set_xticks(x); ax.set_xticklabels(segs_c)
    ax.set_ylabel("max principal angle (deg)")
    ax.set_title("(c) Estimated-subspace sensitivity", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # ---- (d) limb-6 precordial eta_norm by retained rank (ST): the rank-3 contingency ----
    ax = axes[3]
    prec = ("V1", "V2", "V3", "V4", "V5", "V6")
    byrank = rs.get("ST", {}).get("limb6_precordial_eta_norm_by_rank", {})
    ranks_d = sorted(int(r) for r in byrank)
    x = np.arange(len(prec)); w = 0.8 / max(len(ranks_d), 1)
    for j, r in enumerate(ranks_d):
        vals = [byrank[str(r)].get(l, 0.0) for l in prec]
        ax.bar(x + (j - (len(ranks_d) - 1) / 2) * w, vals, w, label=f"rank {r}")
    ax.set_xticks(x); ax.set_xticklabels(prec)
    ax.set_ylabel(r"limb-6 precordial $\tilde\eta$ (ST)")
    ax.set_title("(d) Rank-3 contingency of the exact verdict", fontsize=9.5)
    ax.text(0.02, 0.92, "rank 2: all $\\tilde\\eta{=}0$\n(rank-2 observation collapse)",
            transform=ax.transAxes, fontsize=7, color="0.3", va="top")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS / "ablations.png", dpi=160)
    print("[fig] results/ablations.png")


if __name__ == "__main__":
    main()
