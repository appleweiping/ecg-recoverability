"""Representative neural reconstruction baseline: arbitrary-mask conditional 1-D U-Net (GPU).

A direct supervised reconstructor (NOT diffusion): given the observed leads (unobserved
zeroed) and the 12-channel binary mask, a 1-D U-Net regresses the full 12-lead window;
trained with MSE under random lead masks so one model serves any configuration (no
target-lead leakage). We call this a REPRESENTATIVE masked-reconstruction baseline, not a
claim of state of the art (no Transformer/ImputeECG-scale model is trained here).

Determinism + multi-seed (P0-7): torch/cuda seeds are set and cudnn is deterministic; we
train N_SEEDS models and report mean +/- across-seed std of the aggregate RMSE, plus
seed-averaged PER-RECORD RMSE (keyed by record id) so fair_baselines can pair it against the
linear baselines. Uses the shared protocol split so train/test ids match every other method.

Output: results/neural_baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import LEAD_INDEX
from protocol import standard_split, load_windows, NORMALIZATION

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


def _set_determinism(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_one(Xn, epochs, seed):
    import torch
    _set_determinism(seed)
    net = _unet(24, 12, base=64).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    mrng = np.random.default_rng(seed)
    n = Xn.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g); tot = 0.0
        for i in range(0, n, 64):
            x0 = Xn[perm[i:i + 64]].cuda(); Bn = x0.shape[0]
            mk = _sample_masks(mrng, Bn, "cuda")[:, :, None]
            inp = torch.cat([x0 * mk, mk.expand(Bn, 12, x0.shape[2])], 1)
            pred = net(inp)
            loss = (((pred - x0) * (1 - mk)) ** 2).sum() / ((1 - mk).sum() * x0.shape[2] + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * Bn
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    [seed {seed} ep {ep+1}/{epochs}] loss={tot/n:.5f}", flush=True)
    return net


def _per_record_rmse(net, X, scale, segidxs, oi, ti):
    """Seed's per-record target-lead RMSE (mV) for one config/segment set. Returns
    dict seg -> array aligned to X's record order (nan where segment absent)."""
    import torch
    net.eval()
    yn = (X / scale[None, :, None]).astype(np.float32)
    mk = np.zeros((1, 12, 1), np.float32); mk[0, oi, 0] = 1.0
    rec = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 128):
            yb = torch.tensor(yn[i:i + 128]).cuda(); Bn = yb.shape[0]
            m = torch.tensor(mk).cuda().expand(Bn, 12, X.shape[2])
            p = net(torch.cat([yb * m, m], 1))
            p = p * (1 - m) + yb * m
            rec.extend([(p[j].cpu().numpy() * scale[:, None]) for j in range(Bn)])
    out = {s: np.full(X.shape[0], np.nan) for s in SEGMENTS}
    for r, (Lhat, sg) in enumerate(zip(rec, segidxs)):
        sig = X[r]                                            # (12, window) raw mV
        for s in SEGMENTS:
            idx = sg.get(s)
            if idx is None or idx.size < 8:
                continue
            out[s][r] = np.sqrt(np.mean((Lhat[ti][:, idx] - sig[ti][:, idx]) ** 2))
    return out


def run(n_train=4000, n_test=800, epochs=60, seeds=(0, 1, 2)):
    db = PTBXL()
    tr_ids, te_ids = standard_split(db, n_train, n_test, seed=0)
    Xtr, tr_kept, scale = load_windows(db, tr_ids)
    Xte, te_kept, _ = load_windows(db, te_ids, scale=scale)
    import torch
    Xn = torch.tensor((Xtr / scale[None, :, None]).astype(np.float32))
    segidxs = [db.segment_indices(Xte[i].T, fs=100) for i in range(Xte.shape[0])]

    out = {"n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]), "epochs": epochs,
           "seeds": list(seeds), "model": "arbitrary-mask 1-D U-Net (representative baseline)",
           "configs": {},
           "lineage": lineage.make(db, seed=0, targets=["V2", "V4", "V6"],
                                   normalization=NORMALIZATION, train_ids=tr_kept, test_ids=te_kept,
                                   extra={"torch_deterministic": True, "cudnn_deterministic": True})}
    # train all seeds once, reuse across configs
    nets = [_train_one(Xn, epochs, s) for s in seeds]

    for cname, obs in CONFIGS.items():
        oi = [LEAD_INDEX[l] for l in obs]
        tgt = [l for l in ("V2", "V4", "V6") if l not in obs] or ["V2", "V4", "V6"]
        ti = [LEAD_INDEX[l] for l in tgt]
        # per-seed per-record RMSE, then average over seeds
        seed_pr = [_per_record_rmse(net, Xte, scale, segidxs, oi, ti) for net in nets]
        per_seg = {}
        for s in SEGMENTS:
            stack = np.vstack([sp[s] for sp in seed_pr])      # (n_seeds, n_rec)
            pr = np.nanmean(stack, axis=0)                     # seed-averaged per-record
            valid = np.isfinite(pr)
            if valid.sum() < 5:
                continue
            e = pr[valid]; ids = te_kept[valid]
            brng = np.random.default_rng(2)
            ci = [float(np.percentile([e[brng.integers(0, e.size, e.size)].mean() for _ in range(500)], q))
                  for q in (2.5, 97.5)]
            # across-seed std of the aggregate (record-mean per seed)
            seed_means = [np.nanmean(sp[s][valid]) for sp in seed_pr]
            per_seg[s] = {"rmse_mV": round(float(e.mean()), 4), "rmse_ci": [round(ci[0], 4), round(ci[1], 4)],
                          "across_seed_std_mV": round(float(np.std(seed_means)), 4),
                          "per_record": {"ids": [int(x) for x in ids], "rmse": [round(float(x), 5) for x in e]}}
        out["configs"][cname] = per_seg
        print(f"  [{cname}] " + " ".join(
            f"{s}={per_seg[s]['rmse_mV']}(+-{per_seg[s]['across_seed_std_mV']})" for s in per_seg), flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "neural_baseline.json").write_text(json.dumps(out, indent=2))
    print("[json] results/neural_baseline.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    run(n_train=args.n_train, n_test=args.n_test, epochs=args.epochs, seeds=tuple(args.seeds))
