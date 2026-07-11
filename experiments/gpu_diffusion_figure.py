"""Summarise the clean diffusion fabrication exhibit and make the frontier figure.

Reads results/gpu_diffusion.json (from gpu_diffusion_clean.py) with schema:
  primary / negctrl_limb6 -> {oracle_rho, sweep:{w:{sample:panel, postmean:panel}}}
and produces results/gpu_diffusion_frontier.png (two panels):

  (1) Recoverability deficit vs the realism knob. The held-out oracle rho is the
      linear-recoverable reference. The POSTERIOR MEAN of the diffusion (variance-free,
      K-averaged) tracks the oracle at low guidance and pulls away as guidance rises:
      a genuine DISTRIBUTIONAL deficit, not a single-sample artifact. A single deployed
      draw sits a fixed sampling-variance band below the mean.
  (2) The certificate sees what RMSE cannot. The distributional deficit Delta_rho
      (oracle - postmean) rises monotonically with guidance while RMSE stays flat; the
      limb6 negative control (certified rank-2, oracle ~ 0) shows no deficit -- the
      certificate stays silent where nothing is recoverable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
BAND_A = ("V2", "V4", "V6")
SEGS = ("QRS", "ST", "T")


def _agg(panel, key):
    vals = [panel[s][l][key] for s in SEGS if s in panel
            for l in BAND_A if l in panel[s] and key in panel[s][l]]
    return float(np.mean(vals)) if vals else float("nan")


def _rmse(panel):
    vals = [panel[s]["rmse_mV"] for s in SEGS if s in panel and "rmse_mV" in panel[s]]
    return float(np.mean(vals)) if vals else float("nan")


def _oracle_bandA(oracle_rho):
    vals = [oracle_rho.get(s, {}).get(l, 0.0) for s in SEGS for l in BAND_A]
    return float(np.mean(vals))


def _curves(block):
    sw = block["sweep"]
    ws = sorted(sw.keys(), key=float)
    x = [float(w) for w in ws]
    oracle = _oracle_bandA(block["oracle_rho"])
    mean_rho = [_agg(sw[w]["postmean"], "rho_model") for w in ws]
    samp_rho = [_agg(sw[w]["sample"], "rho_model") for w in ws]
    mean_drho = [oracle - r for r in mean_rho]
    rmse = [_rmse(sw[w]["postmean"]) for w in ws]
    h = [_agg(sw[w]["postmean"], "h_mV") for w in ws]
    rhoRec = [_agg(sw[w]["sample"], "rho_recoverable") for w in ws]
    return dict(x=x, oracle=oracle, mean_rho=mean_rho, samp_rho=samp_rho,
                mean_drho=mean_drho, rmse=rmse, h=h, rhoRec=rhoRec)


def main():
    d = json.loads((RESULTS / "gpu_diffusion.json").read_text())
    prim = _curves(d["primary"])
    neg = _curves(d["negctrl_limb6"]) if "negctrl_limb6" in d else None

    print(f"{'w':>4} {'oracle':>7} {'mean_rho':>9} {'dRho_mean':>10} {'samp_rho':>9} "
          f"{'rmse':>6} {'h':>6} {'rhoRec':>7}")
    for i, w in enumerate(prim["x"]):
        print(f"{w:>4} {prim['oracle']:>7.2f} {prim['mean_rho'][i]:>9.2f} "
              f"{prim['mean_drho'][i]:>10.3f} {prim['samp_rho'][i]:>9.2f} "
              f"{prim['rmse'][i]:>6.3f} {prim['h'][i]:>6.3f} {prim['rhoRec'][i]:>7.2f}")
    if neg:
        print("limb6 negctrl mean dRho:", [round(v, 3) for v in neg["mean_drho"]])
    _plot(prim, neg)


def _plot(p, neg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    x = p["x"]

    ax = axes[0]
    ax.axhline(p["oracle"], ls="--", color="k", lw=1.3, label=r"$\rho_{\rm oracle}$ (held-out recoverable)")
    ax.plot(x, p["mean_rho"], "o-", color="tab:red", lw=2, label=r"$\rho$ posterior mean (K-avg)")
    ax.plot(x, p["samp_rho"], "v:", color="tab:orange", lw=1.5, label=r"$\rho$ single draw (deployed)")
    ax.fill_between(x, p["mean_rho"], [p["oracle"]] * len(x), color="tab:red", alpha=0.13)
    ax.fill_between(x, p["samp_rho"], p["mean_rho"], color="tab:orange", alpha=0.10)
    ax.set_xlabel("guidance scale $w$ (realism knob)")
    ax.set_ylabel(r"non-dipolar correlation w/ truth")
    ax.set_title("Recoverability deficit grows with realism")
    ax.legend(fontsize=6.6, frameon=False, loc="center left")
    ax.grid(alpha=0.3); ax.set_ylim(0, max(0.45, p["oracle"] + 0.08))

    ax = axes[1]
    ax.plot(x, p["mean_drho"], "s-", color="tab:red", lw=2,
            label=r"$\Delta\rho$ deficit (posterior mean)")
    if neg:
        ax.plot(neg["x"], neg["mean_drho"], "d--", color="tab:gray", lw=1.5,
                label=r"$\Delta\rho$ limb-6 negctrl ($\rho_{\rm or}\!\approx\!0$)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("guidance scale $w$")
    ax.set_ylabel(r"$\Delta\rho = \rho_{\rm oracle}-\rho_{\rm mean}$", color="tab:red")
    ax.set_title("Certificate sees what RMSE cannot")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, p["rmse"], "o-", color="tab:blue", lw=1.5, label="RMSE (mV)")
    ax2.set_ylabel("RMSE (mV)", color="tab:blue")
    ax2.set_ylim(0, max(p["rmse"]) * 1.6)
    ax2.legend(fontsize=6.6, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULTS / "gpu_diffusion_frontier.png", dpi=150)
    print("[fig] results/gpu_diffusion_frontier.png")


if __name__ == "__main__":
    main()
