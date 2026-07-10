"""Trained deep reconstructors: does the fabrication story survive real networks?

We train two 1-D CNNs that share an architecture but differ in objective:
a distortion-optimal MSE network and an MSE+adversarial (perceptual) network.
Applying them to the same reduced-lead reconstruction and scoring them with the
certificate answers the "you built a straw-man generative baseline" objection:

* the MSE-trained deep network does NOT fabricate (low hallucination energy $h$),
  behaving like the linear OLS baseline -- fabrication is a property of the
  *objective*, not of using a neural network;
* the perceptual (adversarial) network DOES fabricate -- large $h$, near-zero
  correlation with the true non-dipolar content -- exactly as the
  perception-distortion tradeoff predicts, now for a genuinely trained model.

Outputs: results/neural_baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.certify import certified_unrecoverable_projector, hallucination_energy
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
OBS = ["I", "II", "V2"]


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def _score(db, model, models, test_ids, obs_idx, recon_leads, rate, n_test):
    """Apply a full-signal reconstructor; return per-segment {rmse,h,corr}."""
    acc = {s: {"rmse": [], "h": [], "corr": []} for s in SEGMENTS}
    for eid in list(test_ids)[:n_test]:
        try:
            sig = db.signal(int(eid), rate=rate)                    # (T, 12)
        except Exception:
            continue
        Lhat_full = model.predict(sig[:, obs_idx].T)                # (12, T)
        segidx = db.segment_indices(sig, fs=rate)
        for s in SEGMENTS:
            m = models.get(s)
            idx = segidx[s]
            if m is None or idx.size < 4:
                continue
            U = certified_unrecoverable_projector(m.M, OBS)
            W = sig[idx].T                                           # (12, Tseg)
            Lh = Lhat_full[:, idx]                                   # (12, Tseg)
            rmse = np.sqrt(np.mean((Lh[recon_leads] - W[recon_leads]) ** 2))
            h = hallucination_energy(m.M, m.mu, OBS, Lh)
            rec_nd = U @ (Lh - m.mu[:, None])
            true_nd = U @ (W - m.mu[:, None])
            corr = _pearson(rec_nd[recon_leads].ravel(), true_nd[recon_leads].ravel())
            acc[s]["rmse"].append(float(rmse))
            acc[s]["h"].append(float(np.mean(h[recon_leads])))
            acc[s]["corr"].append(corr)
    return {s: {k: float(np.mean(v)) if v else None for k, v in acc[s].items()} for s in SEGMENTS}


def _pooled(scored):
    """Average a per-segment {rmse,h,corr} dict into scalars over segments."""
    rmse = np.mean([scored[s]["rmse"] for s in SEGMENTS if scored[s]["rmse"] is not None])
    h = np.mean([scored[s]["h"] for s in SEGMENTS if scored[s]["h"] is not None])
    corr = np.mean([scored[s]["corr"] for s in SEGMENTS if scored[s]["corr"] is not None])
    return float(rmse), float(h), float(corr)


def main(n_train=300, n_test=400, rate=100, seed=0, weights=(0.0, 0.5, 2.0, 8.0)):
    from ecgcert.estimators.neural import train_adversarial

    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False,
                                                       folds=range(1, 9)))
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())
    obs_idx = [LEAD_INDEX[l] for l in OBS]
    recon_leads = [i for i in range(12) if i not in obs_idx]

    seg_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=40,
                                          max_records=n_train, seed=seed)
    models = fit_segment_models(seg_samples)

    # Perception-distortion sweep: adv_weight 0 (pure MSE / distortion-optimal) up to
    # heavily perceptual. As realism is prioritised on the unrecoverable subspace, the
    # certificate's hallucination energy rises and its correlation with truth falls.
    out = {"config": OBS, "n_train": n_train, "n_test": n_test, "weights": list(weights),
           "sweep": [], "per_segment": {}}
    for w in weights:
        print(f"\n[train] adversarial CNN, adv_weight={w} ...", flush=True)
        model = train_adversarial(db, train_ids, obs_idx, rate=rate, max_records=n_train,
                                  epochs=6, adv_weight=w, seed=seed)
        scored = _score(db, model, models, test_ids, obs_idx, recon_leads, rate, n_test)
        rmse, h, corr = _pooled(scored)
        out["sweep"].append({"adv_weight": w, "rmse": rmse, "h": h, "corr": corr})
        out["per_segment"][str(w)] = scored
        print(f"  adv_weight={w}: RMSE={rmse:.3f}  h={h:.3f}  corr={corr:+.2f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "neural_baselines.json").write_text(json.dumps(out, indent=2))
    _plot(out)
    print("\n[json] results/neural_baselines.json")


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = [s["adv_weight"] for s in out["sweep"]]
    h = [s["h"] for s in out["sweep"]]
    corr = [s["corr"] for s in out["sweep"]]
    rmse = [s["rmse"] for s in out["sweep"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    ax.plot(w, h, "o-", label="hallucination energy $h$")
    ax.plot(w, corr, "s--", label="non-dipolar corr. with truth")
    ax.set_xscale("symlog"); ax.set_xlabel("adversarial (perceptual) weight")
    ax.set_title("Perception $\\to$ fabrication"); ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(rmse, h, "o-")
    for wi, ri, hi in zip(w, rmse, h):
        ax.annotate(f"$\\lambda$={wi:g}", (ri, hi), fontsize=6, xytext=(2, 2),
                    textcoords="offset points")
    ax.set_xlabel("distortion (RMSE, mV)"); ax.set_ylabel("fabrication ($h$, mV)")
    ax.set_title("Perception-distortion frontier"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "neural_perception_distortion.png", dpi=150)
    print("[fig] results/neural_perception_distortion.png")


if __name__ == "__main__":
    main()
