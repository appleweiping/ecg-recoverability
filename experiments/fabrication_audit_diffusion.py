"""Diffusion tier of the null-space fabrication audit: phi(w) vs classifier-free guidance (GPU).

Loads the trained arbitrary-mask DDPM (results/gpu_ddpm.pt), reconstructs the primary config
{I,II,V1,V3,V5} -> {V2,V4,V6} on held-out NORM fold-10 records at each guidance weight w, and
measures the null-space dipolar FABRICATION ratio phi = ||M_s Q d_hat||^2 / ||M_s d_hat||^2 (the
fraction of the reconstruction's dipolar energy in the provably UNIDENTIFIABLE subspace;
fabrication_audit.fabrication_ratio). If phi rises with w, cranking guidance makes the generative
model assert more content the observation cannot fix -- the rigorous, ground-truth-free form of the
abandoned "fabrication is a property of the objective" claim, and the counterpart to the
achievability result (accuracy does NOT improve with w, gpu_deficit_ci.json).

Output: results/fabrication_diffusion.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fabrication_audit import fabrication_ratio, SEGMENTS          # noqa: E402
from gpu_diffusion_clean import _preload, _batched                  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
# UNDERDETERMINED configs only: the fabrication null space Q is nonzero exactly when the observed
# leads do NOT span the dipole (limb-6 misses the antero-posterior direction; a single lead misses
# two). On a rank-3 spanning set there is nothing to fabricate (phi=0 trivially).
CONFIGS_DIFF = {"limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"], "Lead-I": ["I"]}
GUIDANCES = [1.0, 2.0, 4.0, 6.0]


def run(n_test=300, seed=0, chunk=128):
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor
    db = PTBXL()
    ckpt = str(RESULTS / "gpu_ddpm.pt")
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    scale, T = ck["scale"], int(ck["T"])
    model = DiffusionReconstructor(T=T, base=64, device="cuda", seed=seed)
    model.net.load_state_dict(ck["sd"]); model.net.eval()

    rng = np.random.default_rng(seed)
    norm_train = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    tr_samples = db.collect_all_segments(norm_train, rate=100, max_per_record=60, max_records=3000, seed=seed)
    models = fit_segment_models(tr_samples)                          # M_s per segment
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm_f10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm_f10))[:n_test]
    sigs, segidxs, kept = _preload(db, test_ids)
    print(f"[fab-diff] {len(sigs)} NORM fold-10 records; configs {list(CONFIGS_DIFF)}", flush=True)

    configs_out = {}
    ws = [f"{w:g}" for w in GUIDANCES]
    for cname, obs in CONFIGS_DIFF.items():
        obs_idx = [LEAD_INDEX[l] for l in obs]
        per_w = {}
        for w in GUIDANCES:
            Lhats = _batched(model, sigs, scale, w, T, seed=200, obs_idx=obs_idx, chunk=chunk)  # list (12,1000)
            phis = {s: [] for s in SEGMENTS}
            for Lh, seg in zip(Lhats, segidxs):
                for s in SEGMENTS:
                    m = models.get(s); idx = seg.get(s)
                    if m is None or idx is None or len(idx) == 0:
                        continue
                    phis[s].append(fabrication_ratio(m.M, m.mu, obs, np.asarray(Lh)[:, idx].T))
            per_w[f"{w:g}"] = {s: round(float(np.mean(phis[s])), 5) for s in SEGMENTS if phis[s]}
            print(f"[fab-diff] {cname} w={w}: {per_w[f'{w:g}']}", flush=True)
        trend = {s: (per_w[ws[-1]].get(s, np.nan) - per_w[ws[0]].get(s, np.nan))
                 for s in SEGMENTS if s in per_w[ws[0]] and s in per_w[ws[-1]]}
        configs_out[cname] = {"obs": obs, "per_w": per_w,
                              "fabrication_trend_w1_to_w6": {s: round(float(trend[s]), 5) for s in trend}}
    out = {"configs": configs_out, "guidances": GUIDANCES,
           "metric": ("phi = ||M_s Q d_hat||^2 / ||M_s d_hat||^2, fraction of the diffusion "
                      "reconstruction's dipolar energy in the UNIDENTIFIABLE subspace (0 = honest, "
                      ">0 = fabricates content the observation cannot fix)"),
           "n_test": len(sigs),
           "lineage": lineage.make(db, seed=seed, targets=["V2", "V4", "V6"],
                                   normalization="raw mV (DDPM reconstruction)",
                                   train_ids=norm_train[:3000], test_ids=test_ids,
                                   extra={"checkpoint": ckpt, "guidances": GUIDANCES,
                                          "metric": "null_space_dipolar_fabrication_vs_guidance"})}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fabrication_diffusion.json").write_text(json.dumps(out, indent=2))
    print("[json] results/fabrication_diffusion.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()
    run(n_test=args.n_test)
