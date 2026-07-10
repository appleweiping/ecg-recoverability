"""Synthetic validation of the three theoretical claims (house-rule simulator).

Ground truth is generated so that every tier is known exactly:

    L = mu + M d + r_II + r_III

* ``M d``       -- dipole (Tier I), recoverable from any dipole-spanning S.
* ``r_II``      -- non-dipolar residual *correlated* with the dipole (predictable
                   from y_S; Tier II).  A learned/Bayes prior recovers it.
* ``r_III``     -- non-dipolar residual *independent* of the observation
                   (Tier III).  No estimator can recover it (Var lower bound), and
                   any reconstruction placing energy here is hallucination.

Figures written to results/:
  synthetic_irreducibility.png -- achieved MSE on Tier III vs the Var(u) bound;
                                  Tier I error vs kappa*sigma.
  synthetic_flag_roc.png       -- hallucination detection power vs injected energy.
  synthetic_coverage.png       -- Mondrian-CQR coverage vs nominal.
And results/synthetic.json with the numeric summary used in the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.certify import hallucination_energy
from ecgcert.conformal import conformal_quantile, cqr_calibrate, cqr_interval, empirical_coverage
from ecgcert.estimators import BayesianDipolarReconstructor, LinearDipolarReconstructor
from ecgcert.physics import (
    LEAD_INDEX,
    inverse_dower_matrix,
    kappa,
    lead_transform_T,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
OBS = ["I", "II", "V2"]           # a dipole-spanning 3-lead configuration


def _model(seed=0):
    """Fixed dipolar basis M (12x3) and mean; from the inverse-Dower map."""
    T, D = lead_transform_T(), inverse_dower_matrix()
    M_full = T @ D                                  # (12, 3) dipolar map
    M, _ = np.linalg.qr(M_full)                     # orthonormalise -> M (12,3)
    M = M[:, :3]
    mu = np.zeros(12)
    return M, mu


def _generate(n, M, mu, sigma_d=(1.0, 0.7, 0.5), tier2=0.15, tier3=0.15, seed=1):
    """Return dict with L (12,n) and the tier ground-truth components."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((3, n)) * np.array(sigma_d)[:, None]
    dip = M @ d                                     # (12, n) Tier I
    # Non-dipolar basis (orthogonal complement of M).
    U = np.eye(12) - M @ M.T
    evals, evecs = np.linalg.eigh(U)
    B = evecs[:, evals > 1e-8]                       # (12, 9) non-dipolar basis
    # Tier II: correlated with dipole strength (predictable from y_S).
    c2 = tier2 * (d[0] + 0.5 * d[1])                 # scalar per sample, dipole-driven
    r2 = B[:, 0:1] @ c2[None, :]                     # (12, n) along one non-dipolar dir
    # Tier III: independent of the dipole/observation.
    c3 = tier3 * rng.standard_normal((B.shape[1] - 1, n))
    r3 = B[:, 1:] @ c3                               # (12, n)
    L = mu[:, None] + dip + r2 + r3
    return {"L": L, "dip": dip, "r2": r2, "r3": r3, "d": d, "B": B}


# Two dipole-spanning (full-rank) configurations with very different conditioning.
# The rank-deficient coplanar/limb cases (where part of the dipole is Tier III) are
# demonstrated on real PTB-XL data via the kappa table, where they are far more
# dramatic (kappa ~ 4e5).
CONFIGS_T1 = {
    "spanning {I,II,V2}": ["I", "II", "V2"],
    "collinear {V1,V2,V3}": ["V1", "V2", "V3"],
}


def experiment_tier1_certificate():
    """Tier I noise sensitivity on *purely dipolar* signals matches sigma*||G||_F.

    The certificate bounds the noise-induced error of the dipolar reconstruction
    ``L_hat = M M_S^+ y_S``.  For purely dipolar signals the reconstruction error
    is exactly ``M M_S^+ n``, whose RMS is ``sigma * ||M M_S^+||_F`` and is bounded
    by ``sqrt(|S|) * kappa_s(S) * sigma``.  A coplanar limb triplet (huge kappa)
    blows up while a well-spread triplet stays flat -- geometry, not lead count.
    """
    M, mu = _model()
    n = 8000
    rng = np.random.default_rng(2)
    # Purely dipolar test signals.
    d = rng.standard_normal((3, n)) * np.array([[1.0], [0.7], [0.5]])
    L = mu[:, None] + M @ d
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2]
    out = {"sigmas": sigmas, "configs": {}}
    for name, leads in CONFIGS_T1.items():
        idx = [LEAD_INDEX[l] for l in leads]
        M_S = M[idx]
        G = M @ np.linalg.pinv(M_S)                    # (12, |S|), the certificate operator
        gF = float(np.linalg.norm(G, "fro"))
        k = float(np.linalg.norm(G, 2))
        rec = LinearDipolarReconstructor(M, mu, leads)
        errs, bound = [], []
        for sigma in sigmas:
            noise = sigma * rng.standard_normal((len(idx), n))
            Lhat = rec.predict(L[idx] + noise)
            e = float(np.sqrt(np.mean(np.sum((Lhat - L) ** 2, axis=0))))
            errs.append(e)
            bound.append(float(np.sqrt(len(idx)) * k * sigma))   # sqrt(|S|) kappa sigma
        out["configs"][name] = {"kappa": k, "GF": gF, "err": errs,
                                "predicted": [float(gF * s) for s in sigmas],
                                "bound": bound}
    return out


def experiment_tier3_irreducibility():
    """No estimator beats Var(u) on Tier III; a hallucinator does no better.

    Signals carry a Tier III component (independent of the observation) of energy
    ``delta``.  The MSE-optimal Bayes reconstructor returns the prior mean on it
    (error == Var(u) == delta^2, the lower bound); a hallucinating reconstructor
    that injects plausible non-dipolar content has the same-or-worse error while
    looking confident -- and its hallucination energy is large.
    """
    from ecgcert.estimators import LinearDipolarReconstructor

    M, mu = _model()
    n = 6000
    rng = np.random.default_rng(11)
    idx = [LEAD_INDEX[l] for l in OBS]
    U = np.eye(12) - M @ M.T
    evals, evecs = np.linalg.eigh(U)
    B = evecs[:, evals > 1e-8]                          # non-dipolar basis (12,9)
    u_dir = B[:, 3]                                     # a fixed Tier III direction
    d = rng.standard_normal((3, n)) * np.array([[1.0], [0.7], [0.5]])
    dip = mu[:, None] + M @ d
    lin = LinearDipolarReconstructor(M, mu, OBS)
    deltas = [0.0, 0.1, 0.2, 0.4, 0.8]
    bayes_mse, hall_mse, lower_bound, hall_h = [], [], [], []
    for delta in deltas:
        u = delta * rng.standard_normal(n)             # Tier III amplitude, indep of obs
        L = dip + u_dir[:, None] * u
        yS = L[idx]                                     # noiseless observation
        # Bayes-optimal on Tier III == prior mean (0): recover only the dipole.
        Lhat_bayes = lin.predict(yS)
        # Error attributable to the Tier III component:
        u_hat_bayes = (Lhat_bayes - dip).T @ u_dir     # projection on u_dir (n,)
        bayes_mse.append(float(np.mean((u_hat_bayes - u) ** 2)))
        lower_bound.append(float(np.mean(u ** 2)))     # Var(u) = delta^2
        # A hallucinator injects a plausible but observation-independent guess.
        u_hall = delta * rng.standard_normal(n)        # uncorrelated with true u
        Lhat_hall = Lhat_bayes + u_dir[:, None] * u_hall
        u_hat_hall = (Lhat_hall - dip).T @ u_dir
        hall_mse.append(float(np.mean((u_hat_hall - u) ** 2)))
        hall_h.append(float(np.mean(np.abs(u_hall))))
    return {"deltas": deltas, "bayes_mse": bayes_mse, "hallucinator_mse": hall_mse,
            "lower_bound_var_u": lower_bound, "hallucinator_energy": hall_h}


def experiment_flag_roc():
    """Detection power of the hallucination flag vs injected Tier III energy."""
    M, mu = _model()
    n = 4000
    g = _generate(n, M, mu, tier2=0.15, tier3=0.0, seed=3)   # faithful (no Tier III)
    L = g["L"]
    # Faithful reconstruction = supported (dipole + Tier II mean); use dipole recon.
    from ecgcert.estimators import LinearDipolarReconstructor
    Sel_idx = [LEAD_INDEX[l] for l in OBS]
    rec = LinearDipolarReconstructor(M, mu, OBS)
    # Calibrate flag threshold on faithful reconstructions' hallucination energy.
    h_faithful = np.array([
        hallucination_energy(M, mu, OBS, rec.predict(L[Sel_idx, i:i+1])).max()
        for i in range(0, 1000)
    ])
    alpha = 0.1
    tau = conformal_quantile(h_faithful, alpha)
    # Now inject Tier III of growing energy into a hallucinated reconstruction.
    deltas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    rng = np.random.default_rng(9)
    B = g["B"]
    power = []
    for delta in deltas:
        det = 0
        for i in range(1000, 2000):
            base = rec.predict(L[Sel_idx, i:i+1])            # faithful dipolar recon
            # a hallucinating reconstructor adds independent non-dipolar content
            hall = base + delta * (B[:, 3:4] @ rng.standard_normal((1, 1)))
            h = hallucination_energy(M, mu, OBS, hall).max()
            det += int(h > tau)
        power.append(det / 1000.0)
    return {"alpha": alpha, "tau": float(tau), "deltas": deltas, "power": power,
            "false_flag_rate": float(np.mean(h_faithful > tau))}


def experiment_coverage():
    """Mondrian-CQR coverage on the Tier II residual (per lead)."""
    M, mu = _model()
    n = 12000
    g = _generate(n, M, mu, tier2=0.2, tier3=0.1, seed=4)
    L = g["L"]
    Sel_idx = [LEAD_INDEX[l] for l in OBS]
    # Target: reconstruct V3 (a precordial lead) full value; predictor = dipolar recon.
    from ecgcert.estimators import LinearDipolarReconstructor
    rec = LinearDipolarReconstructor(M, mu, OBS)
    Lhat = rec.predict(L[Sel_idx])                    # (12, n) dipolar recon
    v3 = LEAD_INDEX["V3"]
    y = L[v3]                                          # true V3
    center = Lhat[v3]                                  # dipolar prediction
    # Heteroscedastic quantile guess around the dipolar center.
    spread = np.std(y - center)
    q_lo, q_hi = center - spread, center + spread
    idx = np.random.default_rng(0).permutation(n)
    cal, te = idx[:6000], idx[6000:]
    alpha = 0.1
    Q = cqr_calibrate(y[cal], q_lo[cal], q_hi[cal], alpha)
    lo, hi = cqr_interval(q_lo[te], q_hi[te], Q)
    cov = empirical_coverage(lo, hi, y[te])
    return {"alpha": alpha, "target_coverage": 1 - alpha, "empirical_coverage": float(cov),
            "mean_width": float(np.mean(hi - lo))}


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {
        "tier1_certificate": experiment_tier1_certificate(),
        "tier3_irreducibility": experiment_tier3_irreducibility(),
        "flag_roc": experiment_flag_roc(),
        "coverage": experiment_coverage(),
    }
    (RESULTS / "synthetic.json").write_text(json.dumps(out, indent=2))
    _plot(out)
    # Console summary (sanity-check before it enters the paper).
    t1 = out["tier1_certificate"]
    print("Tier I: RMS error == sigma*||G||_F (predicted), <= sqrt(|S|)*kappa*sigma (bound):")
    for name, c in t1["configs"].items():
        ok = all(e <= b + 1e-6 for e, b in zip(c["err"], c["bound"]))
        match = max(abs(e - p) for e, p in zip(c["err"], c["predicted"]))
        print(f"  {name:22s} kappa={c['kappa']:.2f} ||G||_F={c['GF']:.2f} "
              f"pred-match<{match:.2e} bound-ok={ok}")
    t3 = out["tier3_irreducibility"]
    print("Tier III irreducibility (Bayes MSE vs Var(u) lower bound):")
    for dlt, b, lb, h in zip(t3["deltas"], t3["bayes_mse"], t3["lower_bound_var_u"],
                             t3["hallucinator_mse"]):
        print(f"  delta={dlt:.2f}  bayes={b:.4f}  Var(u)={lb:.4f}  hallucinator={h:.4f}")
    roc = out["flag_roc"]
    print(f"flag: false-flag rate={roc['false_flag_rate']:.3f} (alpha={roc['alpha']}), "
          f"power@delta: {list(zip(roc['deltas'], roc['power']))}")
    cov = out["coverage"]
    print(f"coverage: empirical={cov['empirical_coverage']:.3f} target={cov['target_coverage']}")


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    # (A) Tier I certificate: error vs sigma per config.
    ax = axes[0]
    t1 = out["tier1_certificate"]
    for name, c in t1["configs"].items():
        ax.plot(t1["sigmas"], c["err"], "o-", label=f"{name} ($\\kappa$={c['kappa']:.0f})")
    ax.set_xlabel(r"noise $\sigma$"); ax.set_ylabel("Tier I RMS error")
    ax.set_yscale("log"); ax.set_title("Tier I: geometry, not lead count")
    ax.legend(fontsize=6.5, frameon=False)

    # (B) Tier III irreducibility.
    ax = axes[1]
    t3 = out["tier3_irreducibility"]
    ax.plot(t3["deltas"], t3["bayes_mse"], "o-", label="Bayes-optimal MSE")
    ax.plot(t3["deltas"], t3["lower_bound_var_u"], "k--", label=r"$\mathrm{Var}(u)$ bound")
    ax.plot(t3["deltas"], t3["hallucinator_mse"], "s:", color="tab:red", label="hallucinator MSE")
    ax.set_xlabel(r"Tier III energy $\delta$"); ax.set_ylabel("MSE on Tier III")
    ax.set_title("Tier III irreducibility"); ax.legend(fontsize=6.5, frameon=False)

    # (C) Flag ROC.
    ax = axes[2]
    roc = out["flag_roc"]
    ax.plot(roc["deltas"], roc["power"], "o-")
    ax.axhline(roc["alpha"], color="grey", ls=":", label=r"$\alpha$ (false-flag)")
    ax.set_xlabel(r"injected Tier III energy $\delta$"); ax.set_ylabel("detection power")
    ax.set_title(f"Hallucination flag (FF={roc['false_flag_rate']:.2f})")
    ax.legend(fontsize=7, frameon=False)

    # (D) Coverage.
    ax = axes[3]
    cov = out["coverage"]
    ax.bar(["target", "empirical"], [cov["target_coverage"], cov["empirical_coverage"]],
           color=["grey", "tab:blue"])
    ax.set_ylim(0, 1.0); ax.set_title("Mondrian-CQR coverage")
    for i, v in enumerate([cov["target_coverage"], cov["empirical_coverage"]]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "synthetic_validation.png", dpi=150)
    print("[fig] results/synthetic_validation.png")


if __name__ == "__main__":
    main()
