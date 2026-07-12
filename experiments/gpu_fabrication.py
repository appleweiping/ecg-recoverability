"""GPU fabrication exhibit (CRAFT): oracle ceiling + recoverability-deficit certificate.

Design synthesized by a multi-agent design workflow + adversarial critique. The core
move that defeats circularity: an ill-posed certified-unrecoverable subspace gives
rho~0 for ANY sampler (even Bayes-optimal), so rho~0 there measures the definition,
not fabrication. Instead we build a HELD-OUT ORACLE that partitions the per-segment
non-dipolar residual into:
  Band A -- recoverable  (rho_oracle >> 0): a supervised predictor demonstrably CAN
            recover this from the observed leads S on held-out data.
  Band B -- aleatoric    (rho_oracle ~ 0): genuinely unrecoverable.
Fabrication = a RECOVERABILITY DEFICIT: large energy h co-located with
Delta_rho = rho_oracle - rho_model > 0 in Band A. Band B is the built-in negative
control (flag must stay silent).

Primary S = {I,II,V1,V3,V5} -> reconstruct {V2,V4,V6} (precordial interpolation, where
recoverability is provable), NOT limb-6 (used only as the Band-B negative control).

STAGE 1 (this file, run first): the ORACLE GATE. If rho_oracle is not > 0 for the
primary (segment, unobserved lead), the exhibit is dead -- stop and switch S.
Later stages add the conditional CFG diffusion model + certificate (see run modes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecgcert.certify import off_dipole_projector
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T", "P")

CONFIGS = {
    "precordial-interp": {"obs": ["I", "II", "V1", "V3", "V5"], "target": ["V2", "V4", "V6"]},
    "classic-3lead":     {"obs": ["I", "II", "V2"],             "target": ["V1", "V3", "V4", "V5", "V6"]},
    "limb6-negctrl":     {"obs": ["I", "II", "III", "aVR", "aVL", "aVF"], "target": ["V1", "V2", "V3", "V4", "V5", "V6"]},
}


def _ridge_fit(X, y, lam=1.0):
    """Closed-form ridge with bias. X:(N,d), y:(N,). Returns (w, b)."""
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    d = X1.shape[1]
    A = X1.T @ X1 + lam * np.eye(d)
    A[-1, -1] -= lam  # don't regularize bias
    coef = np.linalg.solve(A, X1.T @ y)
    return coef[:-1], coef[-1]


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def oracle_gate(n_train=2000, n_test=800, rate=100, seed=0):
    """Fit per-sample ridge oracle predicting each unobserved lead's non-dipolar
    residual from the observed leads; report held-out rho_oracle per (config, seg, lead)."""
    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())

    # Segment models on TRAIN only.
    tr_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=60,
                                         max_records=n_train, seed=seed)
    models = fit_segment_models(tr_samples)
    te_samples = db.collect_all_segments(test_ids, rate=rate, max_per_record=60,
                                         max_records=n_test, seed=seed + 1)

    out = {"n_train": n_train, "n_test": n_test, "configs": {}}
    for cname, cfg in CONFIGS.items():
        obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]
        out["configs"][cname] = {}
        for seg in SEGMENTS:
            m = models.get(seg)
            if m is None or tr_samples[seg].shape[0] < 200 or te_samples[seg].shape[0] < 200:
                continue
            U = off_dipole_projector(m.M, cfg["obs"])   # (12,12)
            Xtr = tr_samples[seg]; Xte = te_samples[seg]             # (N,12) mV
            # observed-lead features (raw mV of observed leads)
            Ftr, Fte = Xtr[:, obs_idx], Xte[:, obs_idx]
            # non-dipolar residual of ALL leads
            Rtr = (Xtr - m.mu) @ U.T                                  # (N,12)
            Rte = (Xte - m.mu) @ U.T
            seg_res = {}
            for l in cfg["target"]:
                li = LEAD_INDEX[l]
                w, b = _ridge_fit(Ftr, Rtr[:, li], lam=1.0)
                pred = Fte @ w + b
                rho = _pearson(pred, Rte[:, li])
                true_rms = float(np.sqrt(np.mean(Rte[:, li] ** 2)))
                seg_res[l] = {"rho_oracle": rho, "true_nondip_rms_mV": true_rms}
            out["configs"][cname][seg] = seg_res
        _print_cfg(cname, out["configs"][cname])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gpu_oracle_gate.json").write_text(json.dumps(out, indent=2))
    _verdict(out)


def _print_cfg(cname, segs):
    print(f"\n=== {cname} ===")
    for seg, res in segs.items():
        parts = "  ".join(f"{l}:rho_or={d['rho_oracle']:+.2f}(nd={d['true_nondip_rms_mV']:.3f})"
                          for l, d in res.items())
        print(f"  [{seg}] {parts}")


def _verdict(out):
    prim = out["configs"].get("precordial-interp", {})
    qrs = prim.get("QRS", {})
    band_a = [l for l, d in qrs.items() if d["rho_oracle"] > 0.2]
    print("\n=== ORACLE GATE VERDICT ===")
    print(f"Primary S QRS Band-A leads (rho_oracle>0.2): {band_a}")
    neg = out["configs"].get("limb6-negctrl", {}).get("QRS", {})
    neg_rho = {l: round(d["rho_oracle"], 2) for l, d in neg.items()}
    print(f"limb6 negative-control QRS rho_oracle: {neg_rho}")
    ok = len(band_a) >= 1
    print(f"GATE {'PASS' if ok else 'FAIL'}: primary S has recoverable (Band A) non-dipolar content")
    out["gate_pass"] = bool(ok)
    (RESULTS / "gpu_oracle_gate.json").write_text(json.dumps(out, indent=2))


# ============================ STAGE 2: diffusion + certificate ============================

def _load_full(db, ids, rate, max_records, scale=None):
    """Load (N,12,L) records; robust per-lead normalization by train 95th-pct |amp|."""
    sigs, kept = [], []
    for eid in list(ids)[:max_records]:
        try:
            s = db.signal(int(eid), rate=rate)                       # (L,12) mV
        except Exception:
            continue
        L = s.shape[0]
        if L < 1000:
            continue
        sigs.append(s[:1000].T.astype(np.float32))                   # (12,1000)
        kept.append(int(eid))
    X = np.stack(sigs)                                                # (N,12,1000) mV
    if scale is None:
        scale = np.percentile(np.abs(X), 95, axis=(0, 2)).astype(np.float32)  # (12,)
        scale = np.clip(scale, 0.05, None)
    return X, np.asarray(kept), scale


def _score_recon(db, models, oracle_rho, recon_fn, test_ids, cfg, rate, tag):
    """Pool per-segment non-dipolar residuals over test records; return per (seg,lead)
    {h, rho_model, rmse, rho_oracle, delta_rho}. recon_fn(sig_mV)->Lhat_mV (12,1000)."""
    from ecgcert.certify.tier_decomposition import recoverable_dipole_projector

    obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]
    tgt = cfg["target"]
    # accumulate pooled residual samples per (seg, lead): U_s (unrecoverable) and R_s (recoverable)
    acc = {s: {l: {"rec": [], "true": [], "recR": [], "trueR": []} for l in tgt} for s in SEGMENTS}
    rmse_acc = {s: [] for s in SEGMENTS}
    for eid in test_ids:
        try:
            sig = db.signal(int(eid), rate=rate)[:1000]              # (1000,12) mV
        except Exception:
            continue
        if sig.shape[0] < 1000:
            continue
        Lhat = recon_fn(sig)                                          # (12,1000) mV
        segidx = db.segment_indices(sig, fs=rate)
        for s in SEGMENTS:
            m = models.get(s); idx = segidx[s]
            if m is None or idx.size < 8:
                continue
            U = off_dipole_projector(m.M, cfg["obs"])
            R, _ = recoverable_dipole_projector(m.M, cfg["obs"])
            true_nd = U @ (sig[idx].T - m.mu[:, None])               # (12,Tseg)
            rec_nd = U @ (Lhat[:, idx] - m.mu[:, None])
            true_R = R @ (sig[idx].T - m.mu[:, None])
            rec_R = R @ (Lhat[:, idx] - m.mu[:, None])
            rmse_acc[s].append(float(np.sqrt(np.mean(
                (Lhat[[LEAD_INDEX[l] for l in tgt]][:, idx] - sig[idx].T[[LEAD_INDEX[l] for l in tgt]]) ** 2))))
            for l in tgt:
                li = LEAD_INDEX[l]
                acc[s][l]["rec"].append(rec_nd[li]); acc[s][l]["true"].append(true_nd[li])
                acc[s][l]["recR"].append(rec_R[li]); acc[s][l]["trueR"].append(true_R[li])
    out = {}
    for s in SEGMENTS:
        if not rmse_acc[s]:
            continue
        out[s] = {"rmse_mV": float(np.mean(rmse_acc[s]))}
        for l in tgt:
            rec = np.concatenate(acc[s][l]["rec"]) if acc[s][l]["rec"] else np.zeros(1)
            tru = np.concatenate(acc[s][l]["true"]) if acc[s][l]["true"] else np.zeros(1)
            recR = np.concatenate(acc[s][l]["recR"]) if acc[s][l]["recR"] else np.zeros(1)
            truR = np.concatenate(acc[s][l]["trueR"]) if acc[s][l]["trueR"] else np.zeros(1)
            rho_m = _pearson(rec, tru)
            h = float(np.sqrt(np.mean(rec ** 2)))
            ro = oracle_rho.get(s, {}).get(l, 0.0)
            out[s][l] = {"h_mV": h, "rho_model": rho_m, "rho_oracle": ro,
                         "delta_rho": ro - rho_m, "rho_recoverable": _pearson(recR, truR)}
    print(f"  [{tag}] " + "  ".join(
        f"{s}:V4 h={out[s]['V4']['h_mV']:.3f} rho_m={out[s]['V4']['rho_model']:+.2f} "
        f"Drho={out[s]['V4']['delta_rho']:+.2f}" for s in ("QRS", "ST", "T") if s in out and 'V4' in out[s]))
    return out


def diffusion_exhibit(n_train=3000, n_test=400, rate=100, epochs=40, T=200,
                      guidances=(1.0, 2.0, 4.0), seed=0):
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor

    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())[:n_test]
    cfg = CONFIGS["precordial-interp"]
    obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]

    # segment models + oracle rho (train-fit)
    tr_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=60,
                                         max_records=min(n_train, 2500), seed=seed)
    models = fit_segment_models(tr_samples)
    te_samples = db.collect_all_segments(list(test_ids), rate=rate, max_per_record=60, seed=seed + 1)
    oracle_rho = {}
    for s in SEGMENTS:
        m = models.get(s)
        if m is None or tr_samples[s].shape[0] < 200 or te_samples[s].shape[0] < 200:
            continue
        U = off_dipole_projector(m.M, cfg["obs"])
        Ftr = tr_samples[s][:, obs_idx]; Fte = te_samples[s][:, obs_idx]
        Rtr = (tr_samples[s] - m.mu) @ U.T; Rte = (te_samples[s] - m.mu) @ U.T
        oracle_rho[s] = {}
        for l in cfg["target"]:
            li = LEAD_INDEX[l]
            w, b = _ridge_fit(Ftr, Rtr[:, li], 1.0)
            oracle_rho[s][l] = _pearson(Fte @ w + b, Rte[:, li])

    # train diffusion on normalized full records
    X, _, scale = _load_full(db, train_ids, rate, max_records=n_train)
    Xn = X / scale[None, :, None]
    print(f"[diffusion] train {Xn.shape}, scale={np.round(scale,3)}", flush=True)
    model = DiffusionReconstructor(obs_idx, T=T, base=64, device="cuda", cond_dropout=0.15, seed=seed)
    model.train(Xn, epochs=epochs, bs=64, lr=2e-4, log_every=5)

    def make_recon(guidance):
        def recon(sig_mV):                                            # sig (1000,12)
            yn = (sig_mV[:1000].T / scale[:, None]).astype(np.float32)  # (12,1000) norm
            out = model.sample(yn[None], guidance=guidance, replace=True, steps=T, seed=1)[0]
            return out * scale[:, None]                               # back to mV
        return recon

    results = {"config": cfg, "n_train": n_train, "n_test": len(test_ids),
               "epochs": epochs, "T": T, "oracle_rho": oracle_rho, "sweep": {}}
    for g in guidances:
        results["sweep"][str(g)] = _score_recon(db, models, oracle_rho, make_recon(g),
                                                list(test_ids), cfg, rate, f"w={g}")
    # honest MMSE anchor: mean of K conditional samples (guidance 1 = plain conditional)
    def mmse_recon(sig_mV, K=8):
        yn = (sig_mV[:1000].T / scale[:, None]).astype(np.float32)
        outs = [model.sample(yn[None], guidance=1.0, replace=True, steps=T, seed=100 + k)[0]
                for k in range(K)]
        return np.mean(outs, axis=0) * scale[:, None]
    results["mmse_anchor"] = _score_recon(db, models, oracle_rho,
                                          lambda s: mmse_recon(s), list(test_ids), cfg, rate, "MMSE")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gpu_diffusion.json").write_text(json.dumps(results, indent=2))
    print("[json] results/gpu_diffusion.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gate", choices=["gate", "diffusion"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if args.mode == "gate":
        oracle_gate(n_train=args.n_train, n_test=args.n_test)
    elif args.mode == "diffusion":
        diffusion_exhibit(n_train=args.n_train, n_test=args.n_test, epochs=args.epochs)
