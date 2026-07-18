"""Figure: the recoverability certificate is operational (Phase-1 headline).

Reads results/certificate_validation.json and draws two panels:
  (left)  calibration scatter -- measured per-lead DIPOLAR-projection RMSE (mV) vs the theoretical
          prior-conditional expected ambiguity a_l (mV), one marker per (config, segment, target
          lead) x reconstructor; the y=x line is the certified minimax floor. Points track the
          floor and sit on/above it; filled = unidentifiable (eta>0), open = identifiable (eta~0).
  (right) the eta=0 vs eta>0 dichotomy -- measured RMSE distribution, showing the certificate's
          binary verdict cleanly separates recoverable from unrecoverable target leads.

Output: results/certificate_validation.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
METHOD_COLOR = {"dipolar": "C0", "ridge": "C1", "ols": "C2"}


def main():
    d = json.loads((RESULTS / "certificate_validation.json").read_text())
    cells = d["cells"]
    corr = d["correlations"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ---- (left) calibration scatter vs the floor ----
    ax = axes[0]
    amb_all = []
    for nm, col in METHOD_COLOR.items():
        amb = np.array([c["amb_mV"] for c in cells])
        y = np.array([c["measured_rmse_mV"][nm] for c in cells])
        z = np.array([c["eta_zero"] for c in cells])
        amb_all.append(amb)
        ax.scatter(amb[~z], y[~z], s=18, c=col, alpha=0.7, label=f"{nm} (eta>0)", edgecolors="none")
        ax.scatter(amb[z], y[z], s=18, facecolors="none", edgecolors=col, alpha=0.7, linewidths=0.9,
                   label=f"{nm} (eta=0)")
    hi = float(np.nanpercentile(np.concatenate(amb_all), 99)) * 1.1 + 0.02
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="certified floor (y=x)")
    ax.set_xlim(-0.01, hi); ax.set_ylim(-0.01, None)
    ax.set_xlabel("expected ambiguity $a_{s,\\ell}(S)$ (mV, theory)")
    ax.set_ylabel("measured dipolar-projection RMSE (mV)")
    sp = corr["dipolar"]["spearman_amb_rmse"]; pe = corr["dipolar"]["pearson_amb_rmse"]
    au = corr["dipolar"]["auc_eta_pos_predicts_large_error"]
    ax.set_title(f"(a) Certificate predicts measured error\nSpearman {sp:.2f}, Pearson {pe:.2f}, "
                 f"AUC($\\eta{{>}}0$) {au:.2f}", fontsize=9.5)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left"); ax.grid(alpha=0.3)

    # ---- (right) eta=0 vs eta>0 dichotomy ----
    ax = axes[1]
    y = np.array([c["measured_rmse_mV"]["dipolar"] for c in cells])
    z = np.array([c["eta_zero"] for c in cells])
    parts = [y[z], y[~z]]
    vp = ax.violinplot(parts, showmedians=True, showextrema=False)
    for b, col in zip(vp["bodies"], ("C7", "C3")):
        b.set_facecolor(col); b.set_alpha(0.5)
    for i, (data, col) in enumerate(zip(parts, ("0.4", "C3")), start=1):
        ax.scatter(np.full(data.size, i) + (np.random.default_rng(0).random(data.size) - 0.5) * 0.12,
                   data, s=8, c=col, alpha=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"identifiable\n$\\eta{{=}}0$ (n={int(z.sum())})",
                                               f"unidentifiable\n$\\eta{{>}}0$ (n={int((~z).sum())})"])
    ax.set_ylabel("measured dipolar-projection RMSE (mV)")
    m0 = corr["dipolar"]["median_rmse_eta0"]; mp = corr["dipolar"]["median_rmse_etapos"]
    ax.set_title(f"(b) Binary verdict separates recoverability\nmedian {m0:.3f} vs {mp:.3f} mV", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS / "certificate_validation.png", dpi=160)
    print("[fig] results/certificate_validation.png")


if __name__ == "__main__":
    main()
