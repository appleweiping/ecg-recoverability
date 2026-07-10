"""M4: real PTB-XL reduced-lead reconstruction + hallucination quantification.

Fits per-segment dipolar models on the training folds, then on the held-out test
fold reconstructs the unobserved leads for three lead configurations with several
baselines and measures, per (segment, reconstructed lead):

* per-lead reconstruction RMSE (the number every paper reports);
* the certified-unrecoverable **hallucination energy** ``h``;
* the **correlation** between the reconstruction's certified-unrecoverable
  component and the *true* one -- the "confidently wrong" test.

Headline: the generative reconstructor keeps global RMSE competitive but places
large energy ``h`` in the certified-unrecoverable subspace that is essentially
uncorrelated with truth (fabrication), while the dipolar/OLS baselines do not
fabricate (h ~ 0) but blur the non-dipolar content.  The certificate distinguishes
these cases; global RMSE does not.

Outputs: results/ptbxl_reduced_lead.json + results/ptbxl_hallucination.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.certify import certified_unrecoverable_projector, hallucination_energy
from ecgcert.data import PTBXL
from ecgcert.estimators import (
    GenerativeSampleReconstructor,
    LinearDipolarReconstructor,
    OLSReconstructor,
)
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX, LEADS

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")

CONFIGS = {
    "Lead-I": ["I"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "3-lead {I,II,V2}": ["I", "II", "V2"],
}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _collect_segment_windows(db, ids, rate, max_records):
    """Per record, delineate once and return list of (segment, (12, T_seg)) windows."""
    windows = {s: [] for s in SEGMENTS}
    for eid in ids[:max_records]:
        try:
            sig = db.signal(int(eid), rate=rate)          # (T, 12)
        except Exception:
            continue
        segidx = db.segment_indices(sig, fs=rate)
        for s in SEGMENTS:
            idx = segidx[s]
            if idx.size >= 4:
                windows[s].append(sig[idx].T)              # (12, T_seg)
    return windows


def main(n_train=500, n_test=500, rate=100, seed=0):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False,
                                                       folds=range(1, 9)))
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())

    # --- fit per-segment models on train ---
    seg_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=40,
                                          max_records=n_train, seed=seed)
    models = fit_segment_models(seg_samples)
    # global OLS training samples (all leads, all segments pooled).
    L_train = np.hstack([seg_samples[s].T for s in SEGMENTS if seg_samples[s].shape[0] > 0])

    # --- collect test windows ---
    test_windows = _collect_segment_windows(db, list(test_ids), rate, n_test)

    out = {"config": {}, "n_train": n_train, "n_test": n_test, "rate": rate}
    for cname, obs in CONFIGS.items():
        obs_idx = [LEAD_INDEX[l] for l in obs]
        recon_leads = [i for i in range(12) if i not in obs_idx]
        ols = OLSReconstructor(obs).fit(L_train)
        res = {seg: {} for seg in SEGMENTS}
        for seg in SEGMENTS:
            m = models.get(seg)
            if m is None or not test_windows[seg]:
                continue
            lin = LinearDipolarReconstructor(m.M, m.mu, obs)
            gen = GenerativeSampleReconstructor(m.M, m.mu, obs, m.Sigma_r, scale=1.0, seed=1)
            U = certified_unrecoverable_projector(m.M, obs)
            acc = {r: {"rmse": [], "h": [], "corr": []} for r in ("dipolar", "ols", "generative")}
            true_nondip_energy = []
            for W in test_windows[seg]:                      # W: (12, T)
                yS = W[obs_idx]
                true_nd = U @ (W - m.mu[:, None])
                for rname, rec in (("dipolar", lin), ("ols", ols), ("generative", gen)):
                    Lhat = rec.predict(yS)
                    rmse = np.sqrt(np.mean((Lhat[recon_leads] - W[recon_leads]) ** 2))
                    h = hallucination_energy(m.M, m.mu, obs, Lhat)
                    rec_nd = U @ (Lhat - m.mu[:, None])
                    # correlation of the non-dipolar component, pooled over recon leads.
                    corr = _pearson(rec_nd[recon_leads].ravel(), true_nd[recon_leads].ravel())
                    acc[rname]["rmse"].append(float(rmse))
                    acc[rname]["h"].append(float(np.mean(h[recon_leads])))
                    acc[rname]["corr"].append(corr)
                true_nondip_energy.append(float(np.sqrt(np.mean((true_nd[recon_leads]) ** 2))))
            res[seg] = {
                "true_nondipolar_rms": float(np.mean(true_nondip_energy)),
                "dipolar_fraction": float(m.evr[:3].sum()),
                **{r: {"rmse": float(np.mean(acc[r]["rmse"])),
                       "h": float(np.mean(acc[r]["h"])),
                       "corr": float(np.mean(acc[r]["corr"]))}
                   for r in acc},
            }
        out["config"][cname] = {"observed": obs, "segments": res}
        _print_config(cname, res)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ptbxl_reduced_lead.json").write_text(json.dumps(out, indent=2))
    _plot(out)


def _print_config(cname, res):
    print(f"\n=== {cname} ===")
    for seg in SEGMENTS:
        r = res.get(seg)
        if not r:
            continue
        print(f"  [{seg}] true-nondip RMS={r['true_nondipolar_rms']:.3f} "
              f"dipfrac={r['dipolar_fraction']:.2f}")
        for rname in ("dipolar", "ols", "generative"):
            d = r[rname]
            print(f"     {rname:11s} RMSE={d['rmse']:.3f}  h={d['h']:.3f}  "
                  f"nondip-corr={d['corr']:+.2f}")


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Focus figure: limb-6 config, ST + T segments (where non-dipolar content lives).
    cfg = out["config"].get("limb-6", {}).get("segments", {})
    segs = [s for s in SEGMENTS if s in cfg and cfg[s]]
    recons = ("dipolar", "ols", "generative")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    # Panel 1: RMSE looks similar; Panel 2: h vs corr reveals hallucination.
    ax = axes[0]
    x = np.arange(len(segs)); w = 0.25
    for i, r in enumerate(recons):
        ax.bar(x + (i - 1) * w, [cfg[s][r]["rmse"] for s in segs], w, label=r)
    ax.set_xticks(x); ax.set_xticklabels(segs); ax.set_ylabel("recon RMSE (mV)")
    ax.set_title("Global RMSE looks fine for all"); ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for r, mk in zip(recons, ("o", "s", "^")):
        hs = [cfg[s][r]["h"] for s in segs]
        cs = [cfg[s][r]["corr"] for s in segs]
        ax.scatter(cs, hs, marker=mk, s=60, label=r)
        for s, c, h in zip(segs, cs, hs):
            ax.annotate(s, (c, h), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("non-dipolar correlation with truth")
    ax.set_ylabel("hallucination energy h (mV)")
    ax.set_title("Certificate reveals fabrication")
    ax.axvline(0, color="grey", lw=0.5); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "ptbxl_hallucination.png", dpi=150)
    print("\n[fig] results/ptbxl_hallucination.png")


if __name__ == "__main__":
    main()
