"""Figure: the null-space dipolar fabrication audit.

Reads results/fabrication_audit.json (CPU: linear reconstructors) and, if present,
results/fabrication_diffusion.json (GPU: DDPM vs guidance) and draws:
  (a) fabrication ratio phi per observed-lead config for the abstaining reconstructors
      (dipolar, prior-mean) vs the linear reconstructors (ridge/OLS): principled reconstructors
      assert nothing in the unidentifiable subspace (phi=0); single-lead reconstructors fabricate.
  (b) the diffusion model's phi vs classifier-free guidance on underdetermined configs: whether
      cranking guidance makes the generative model assert MORE unfounded content.

Output: results/fabrication_audit_figure.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEG_FOR_BAR = "QRS"


def main():
    fab = json.loads((RESULTS / "fabrication_audit.json").read_text())
    diff_path = RESULTS / "fabrication_diffusion.json"
    diff = json.loads(diff_path.read_text()) if diff_path.exists() else None
    if diff and "configs" not in diff:      # stale (pre-loop) schema -> ignore until re-run lands
        diff = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 2 if diff else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.6 * ncol, 4.4))
    axes = np.atleast_1d(axes)

    # ---- (a) CPU: phi per config x reconstructor (QRS) ----
    ax = axes[0]
    cfgs = [c for c in ("Lead-I", "Lead-II", "{I,II,V2}", "{I,II,V1,V3,V5}", "limb-6")
            if c in fab["configs"]]
    methods = ("prior_mean", "dipolar", "ridge", "ols")
    colors = {"prior_mean": "C7", "dipolar": "C0", "ridge": "C1", "ols": "C2"}
    x = np.arange(len(cfgs)); w = 0.8 / len(methods)
    for j, nm in enumerate(methods):
        vals = [fab["configs"][c].get(SEG_FOR_BAR, {}).get(nm, 0.0) for c in cfgs]
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, vals, w, color=colors[nm],
               label=nm.replace("_", "-"))
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"fabrication ratio $\phi$ (QRS)")
    ax.set_title("(a) Linear reconstructors: honest abstain vs fabricate", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.text(0.02, 0.95, "dipolar / prior-mean assert nothing\nin the unidentifiable subspace",
            transform=ax.transAxes, fontsize=7, color="0.3", va="top")

    # ---- (b) diffusion phi vs guidance ----
    if diff:
        ax = axes[1]
        ws = [float(w) for w in diff["guidances"]]
        for cname, cd in diff["configs"].items():
            for s, ls in (("QRS", "-o"), ("ST", "--s"), ("T", ":^")):
                ys = [cd["per_w"].get(f"{w:g}", {}).get(s) for w in ws]
                if all(y is not None for y in ys):
                    ax.plot(ws, ys, ls, ms=4, label=f"{cname} {s}")
        ax.set_xlabel("classifier-free guidance $w$")
        ax.set_ylabel(r"diffusion fabrication ratio $\phi$")
        ax.set_title("(b) Guidance inflates unfounded content", fontsize=9.5)
        ax.set_xticks(ws); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "fabrication_audit_figure.png", dpi=160)
    print("[fig] results/fabrication_audit_figure.png")


if __name__ == "__main__":
    main()
