"""Strong neural reconstruction baseline: arbitrary-mask conditional 1-D U-Net (GPU).

A direct supervised reconstructor (NOT diffusion): given the observed leads (unobserved
zeroed) and the 12-channel binary mask, a 1-D U-Net regresses the full 12-lead window;
trained with MSE under random lead masks so one model serves any configuration (no
target-lead leakage). This is the "strong neural baseline" the reduced-lead literature
uses (U-Net / masked reconstruction). At eval the observed leads are kept exact.

Reports RMSE (mV) of the target leads per (config, segment), to sit alongside the
classical baselines in baselines_physics.json.

Output: results/neural_baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.data import PTBXL
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
CONFIGS = {
    "{I,II,V2}": ["I", "II", "V2"],
    "{I,II,V1,V3,V5}": ["I", "II", "V1", "V3", "V5"],
    "limb-6": ["I", "II", "III", "aVR", "aVL", "aVF"],
}
_DEPLOY = ((0, 1, 7), (0, 1, 6, 8, 10), (0, 1, 2, 3, 4, 5), (0,), (1,))


def _unet(in_ch, out_ch, base=64):
    import torch; nn = torch.nn

    class B(nn.Module):
        def __init__(s, ci, co):
            super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv1d(ci, co, 5, padding=2)
            s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv1d(co, co, 5, padding=2)
            s.sk = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
        def forward(s, x):
            h = s.c1(torch.nn.functional.silu(s.n1(x))); h = s.c2(torch.nn.functional.silu(s.n2(h)))
            return h + s.sk(x)

    class U(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Conv1d(in_ch, base, 5, padding=2)
            s.d1 = B(base, base); s.d2 = B(base, base * 2); s.d3 = B(base * 2, base * 4)
            s.dn = nn.AvgPool1d(2); s.mid = B(base * 4, base * 4)
            s.u3 = B(base * 4 + base * 4, base * 2); s.u2 = B(base * 2 + base * 2, base)
            s.u1 = B(base + base, base); s.up = nn.Upsample(scale_factor=2, mode="nearest")
            s.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, out_ch, 5, padding=2))
        def forward(s, x):
            h = s.inp(x); h1 = s.d1(h); h2 = s.d2(s.dn(h1)); h3 = s.d3(s.dn(h2))
            m = s.mid(h3); u = s.u3(torch.cat([m, h3], 1))
            u = s.u2(torch.cat([s.up(u), h2], 1)); u = s.u1(torch.cat([s.up(u), h1], 1))
            return s.out(u)
    return U()


def _sample_masks(rng, B, device):
    import torch
    mk = torch.zeros(B, 12, device=device)
    for b in range(B):
        obs = _DEPLOY[rng.integers(len(_DEPLOY))] if rng.random() < 0.5 else \
            rng.choice(12, size=int(rng.integers(3, 9)), replace=False)
        mk[b, list(obs)] = 1.0
    return mk


def run(n_train=4000, n_test=800, epochs=60, seed=0):
    import torch
    from gpu_fabrication import _load_full
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm10))[:n_test]

    X, _, scale = _load_full(db, tr, 100, max_records=n_train)
    Xn = torch.tensor((X / scale[None, :, None]).astype(np.float32))
    net = _unet(24, 12, base=64).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    mrng = np.random.default_rng(seed)
    n = Xn.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 64):
            x0 = Xn[perm[i:i + 64]].cuda(); Bn = x0.shape[0]
            mk = _sample_masks(mrng, Bn, "cuda")[:, :, None]
            inp = torch.cat([x0 * mk, mk.expand(Bn, 12, x0.shape[2])], 1)
            pred = net(inp)
            loss = (((pred - x0) * (1 - mk)) ** 2).sum() / ((1 - mk).sum() * x0.shape[2] + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * Bn
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"  [unet] epoch {ep+1}/{epochs} loss={tot/n:.5f}", flush=True)

    # eval: per config, RMSE of target leads per segment
    net.eval()
    sigs, segidxs = [], []
    for eid in test_ids:
        try:
            s = db.signal(int(eid), rate=100)[:1000]
        except Exception:
            continue
        if s.shape[0] == 1000 and np.all(np.isfinite(s)):
            sigs.append(s.astype(np.float32)); segidxs.append(db.segment_indices(s, fs=100))
    yn = np.stack([(s.T / scale[:, None]) for s in sigs]).astype(np.float32)
    out = {"n_train": n_train, "n_test": len(sigs), "epochs": epochs, "configs": {}}
    with torch.no_grad():
        for cname, obs in CONFIGS.items():
            oi = [LEAD_INDEX[l] for l in obs]; tgt = [l for l in ("V2", "V4", "V6") if l not in obs] or ["V2", "V4", "V6"]
            ti = [LEAD_INDEX[l] for l in tgt]
            mk = np.zeros((1, 12, 1), np.float32); mk[0, oi, 0] = 1.0
            rec = []
            for i in range(0, len(sigs), 128):
                yb = torch.tensor(yn[i:i + 128]).cuda(); B = yb.shape[0]
                m = torch.tensor(mk).cuda().expand(B, 12, 1000)
                inp = torch.cat([yb * m, m], 1)
                p = net(inp)
                p = p * (1 - m) + yb * m                           # keep observed leads exact
                rec.extend([(p[j].cpu().numpy() * scale[:, None]) for j in range(B)])
            per_seg = {}
            for s in SEGMENTS:
                errs = []
                for sig, seg, Lhat in zip(sigs, segidxs, rec):
                    idx = seg.get(s)
                    if idx is None or idx.size < 8:
                        continue
                    errs.append(np.sqrt(np.mean((Lhat[ti][:, idx] - sig[:, ti].T[:, idx]) ** 2)))
                if errs:
                    e = np.array(errs); brng = np.random.default_rng(seed + 2)
                    ci = [float(np.percentile([e[brng.integers(0, e.size, e.size)].mean() for _ in range(500)], q)) for q in (2.5, 97.5)]
                    per_seg[s] = {"rmse_mV": round(float(e.mean()), 4), "rmse_ci": [round(ci[0], 4), round(ci[1], 4)]}
            out["configs"][cname] = per_seg
            print(f"  [{cname}] " + " ".join(f"{s}={v['rmse_mV']}" for s, v in per_seg.items()), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "neural_baseline.json").write_text(json.dumps(out, indent=2))
    print("[json] results/neural_baseline.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test, epochs=args.epochs)
