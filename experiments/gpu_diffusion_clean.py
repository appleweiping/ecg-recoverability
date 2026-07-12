"""Clean, de-noised, reviewer-hardened re-run of the diffusion fabrication exhibit.

Addresses the adversarial-review findings on the first GPU run:
  (A) The recoverability deficit must not be an artifact of comparing an MSE-optimal
      ridge MEAN (rho_oracle) to a SINGLE diffusion sample. We score, at EVERY
      guidance w, BOTH:
        - postmean: rho of the K-sample posterior mean at that w (variance-free);
                    the DISTRIBUTIONAL deficit is oracle - rho_postmean.
        - sample:   rho of single draws, multi-seed averaged (honest deployment);
                    the gap postmean - sample is the pure sampling-variance band.
  (B) The limb6 negative control is actually RUN through the diffusion (not only the
      oracle gate), so we show the Band-A-gated deficit stays ~0 on QRS-limb6.
  (C) NORM train -> NORM test (fold 10): removes the out-of-distribution confound.
  (D) One shared oracle per config, emitted to paper/_gpu_macros.tex.

Sampling is BATCHED (chunk of records through model.sample) -> ~40x faster than the
per-record path. The scoring math is a faithful transcription of gpu_fabrication.py's
audited _score_recon; `--validate` feeds identical reconstructions to both and asserts
they agree.

Run on GPU:
  python experiments/gpu_diffusion_clean.py --n-train 4000 --n-test 800 \
      --epochs 60 --seeds 3 --K 8 --guidances 1.0,2.0,3.0,4.0
  python experiments/gpu_diffusion_clean.py --validate            # math check only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecgcert.certify.tier_decomposition import (
    off_dipole_projector, recoverable_dipole_projector)
from ecgcert.data import PTBXL
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX
from gpu_fabrication import CONFIGS, SEGMENTS, _ridge_fit, _pearson, _score_recon

RESULTS = Path(__file__).resolve().parent.parent / "results"
PAPER = Path(__file__).resolve().parent.parent / "paper"
BAND_A = ("V2", "V4", "V6")
CORE_SEGS = ("QRS", "ST", "T")


# ------------------------------------------------------------------ oracle + data
def _oracle(models, tr_samples, te_samples, cfg):
    """One held-out ridge oracle per config: rho of E[residual|obs] with truth."""
    obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]
    out = {}
    for s in SEGMENTS:
        m = models.get(s)
        if m is None or tr_samples[s].shape[0] < 200 or te_samples[s].shape[0] < 200:
            continue
        U = off_dipole_projector(m.M, cfg["obs"])
        Ftr, Fte = tr_samples[s][:, obs_idx], te_samples[s][:, obs_idx]
        Rtr, Rte = (tr_samples[s] - m.mu) @ U.T, (te_samples[s] - m.mu) @ U.T
        out[s] = {}
        for l in cfg["target"]:
            li = LEAD_INDEX[l]
            w, b = _ridge_fit(Ftr, Rtr[:, li], 1.0)
            out[s][l] = _pearson(Fte @ w + b, Rte[:, li])
    return out


def _preload(db, test_ids, rate=100):
    """Load test signals + segment indices once (skip logic mirrors _score_recon)."""
    sigs, segs, kept = [], [], []
    for eid in test_ids:
        try:
            sig = db.signal(int(eid), rate=rate)[:1000]          # (1000,12) mV
        except Exception:
            continue
        if sig.shape[0] < 1000:
            continue
        sigs.append(sig)
        segs.append(db.segment_indices(sig, fs=rate))
        kept.append(int(eid))
    return sigs, segs, kept


# ------------------------------------------------------------------ batched recon
def _batched(model, sigs, scale, guidance, T, seed, chunk=128):
    """Reconstruct all `sigs` in GPU batches. Returns list of (12,1000) mV."""
    yn = np.stack([(s.T / scale[:, None]).astype(np.float32) for s in sigs])   # (N,12,1000)
    out = [None] * len(sigs)
    for i in range(0, len(sigs), chunk):
        rec = model.sample(yn[i:i + chunk], guidance=guidance, replace=True, steps=T, seed=seed)
        for j in range(rec.shape[0]):
            out[i + j] = rec[j] * scale[:, None]
    return out


def _postmean(model, sigs, scale, guidance, T, K, chunk=128):
    acc = [np.zeros((12, 1000)) for _ in sigs]
    for k in range(K):
        L = _batched(model, sigs, scale, guidance, T, seed=200 + k, chunk=chunk)
        for i in range(len(sigs)):
            acc[i] += L[i]
    return [a / K for a in acc]


# ------------------------------------------------------------------ scoring (faithful)
def _score(sigs, segidxs, models, oracle_rho, Lhats, cfg):
    """Transcription of gpu_fabrication._score_recon, but over precomputed Lhats."""
    tgt = cfg["target"]
    proj = {}
    for s in SEGMENTS:
        m = models.get(s)
        if m is not None:
            U = off_dipole_projector(m.M, cfg["obs"])
            R, _ = recoverable_dipole_projector(m.M, cfg["obs"])
            proj[s] = (U, R)
    acc = {s: {l: {"rec": [], "true": [], "recR": [], "trueR": []} for l in tgt} for s in SEGMENTS}
    rmse_acc = {s: [] for s in SEGMENTS}
    tgt_idx = [LEAD_INDEX[l] for l in tgt]
    for sig, segidx, Lhat in zip(sigs, segidxs, Lhats):
        for s in SEGMENTS:
            m = models.get(s); idx = segidx.get(s)
            if m is None or idx is None or idx.size < 8:
                continue
            U, R = proj[s]
            true_nd = U @ (sig[idx].T - m.mu[:, None])
            rec_nd = U @ (Lhat[:, idx] - m.mu[:, None])
            true_R = R @ (sig[idx].T - m.mu[:, None])
            rec_R = R @ (Lhat[:, idx] - m.mu[:, None])
            rmse_acc[s].append(float(np.sqrt(np.mean(
                (Lhat[tgt_idx][:, idx] - sig[idx].T[tgt_idx]) ** 2))))
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
            ro = oracle_rho.get(s, {}).get(l, 0.0)
            rho_m = _pearson(rec, tru)
            out[s][l] = {"h_mV": float(np.sqrt(np.mean(rec ** 2))), "rho_model": rho_m,
                         "rho_oracle": ro, "delta_rho": ro - rho_m,
                         "rho_recoverable": _pearson(recR, truR)}
    return out


def _avg_panels(panels):
    out = {}
    for s in set().union(*[p.keys() for p in panels]):
        vals = [p[s] for p in panels if s in p]
        out[s] = {"rmse_mV": float(np.mean([v["rmse_mV"] for v in vals]))}
        for l in set().union(*[set(v) - {"rmse_mV"} for v in vals]):
            ent = [v[l] for v in vals if l in v]
            out[s][l] = {k: float(np.mean([e[k] for e in ent]))
                         for k in ("h_mV", "rho_model", "rho_oracle", "delta_rho", "rho_recoverable")}
    return out


def _agg(panel, key):
    vals = [panel[s][l][key] for s in CORE_SEGS if s in panel
            for l in BAND_A if l in panel[s] and key in panel[s][l]]
    return float(np.mean(vals)) if vals else float("nan")


def _agg_rmse(panel):
    vals = [panel[s]["rmse_mV"] for s in CORE_SEGS if s in panel and "rmse_mV" in panel[s]]
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------ per-config driver
def run_config(sigs, segidxs, models, cfg, tr_samples, te_samples, model, scale, T,
               guidances, seeds, K, cname, chunk):
    oracle_rho = _oracle(models, tr_samples, te_samples, cfg)
    out = {"config": cfg, "oracle_rho": oracle_rho, "sweep": {}}
    for g in guidances:
        samp = _avg_panels([_score(sigs, segidxs, models, oracle_rho,
                                   _batched(model, sigs, scale, g, T, seed=1 + sd, chunk=chunk), cfg)
                            for sd in range(seeds)])
        mean = _score(sigs, segidxs, models, oracle_rho,
                      _postmean(model, sigs, scale, g, T, K, chunk=chunk), cfg)
        out["sweep"][str(g)] = {"sample": samp, "postmean": mean}
        oa = float(np.mean([oracle_rho.get(s, {}).get(l, 0) for s in CORE_SEGS for l in BAND_A]))
        print(f"  [{cname} w={g}] mean dRho={_agg(mean,'delta_rho'):+.3f} h={_agg(mean,'h_mV'):.3f} "
              f"rmse={_agg_rmse(mean):.3f} | samp dRho={_agg(samp,'delta_rho'):+.3f} "
              f"rhoRec={_agg(samp,'rho_recoverable'):+.2f} | oracle={oa:+.2f}", flush=True)
    return out


def _build(n_train, n_test, epochs, T, seed):
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm_train = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm_f10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm_f10))[:n_test]

    tr_samples = db.collect_all_segments(norm_train, rate=100, max_per_record=60,
                                         max_records=min(n_train, 3000), seed=seed)
    models = fit_segment_models(tr_samples)
    te_samples = db.collect_all_segments(list(test_ids), rate=100, max_per_record=60, seed=seed + 1)
    sigs, segidxs, kept = _preload(db, test_ids)
    print(f"[clean] test records loaded {len(sigs)} (NORM fold-10)", flush=True)

    from gpu_fabrication import _load_full
    X, _, scale = _load_full(db, norm_train, 100, max_records=n_train)
    Xn = X / scale[None, :, None]
    print(f"[clean] train {Xn.shape} scale={np.round(scale,3)}", flush=True)
    obs_idx = [LEAD_INDEX[l] for l in CONFIGS["precordial-interp"]["obs"]]
    model = DiffusionReconstructor(obs_idx, T=T, base=64, device="cuda", cond_dropout=0.15, seed=seed)
    model.train(Xn, epochs=epochs, bs=64, lr=2e-4, log_every=10)
    torch.save({"sd": model.net.state_dict(), "scale": scale, "T": T}, RESULTS / "gpu_ddpm.pt")
    return db, models, tr_samples, te_samples, sigs, segidxs, kept, model, scale


def main_ci(n_train, n_test, epochs, T, guidances, K, n_seeds, seed, chunk):
    """Error bars on the recoverability deficit over independent TRAINING seeds.

    Data split + held-out oracle are fixed; only the diffusion model is retrained per
    seed. For each seed we score the primary posterior-mean deficit at each guidance;
    we report per-guidance mean +/- std across seeds (captures training + posterior-mean
    sampling stochasticity). This answers "is the +0.10 deficit above run-to-run noise?".
    """
    import torch
    from ecgcert.estimators.diffusion import DiffusionReconstructor
    db = PTBXL()
    rng = np.random.default_rng(seed)
    norm_train = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 9)))
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm_f10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test_ids = rng.permutation(np.intersect1d(f10, norm_f10))[:n_test]
    cfg = CONFIGS["precordial-interp"]
    obs_idx = [LEAD_INDEX[l] for l in cfg["obs"]]

    tr_samples = db.collect_all_segments(norm_train, rate=100, max_per_record=60,
                                         max_records=min(n_train, 3000), seed=seed)
    models = fit_segment_models(tr_samples)
    te_samples = db.collect_all_segments(list(test_ids), rate=100, max_per_record=60, seed=seed + 1)
    oracle_rho = _oracle(models, tr_samples, te_samples, cfg)
    sigs, segidxs, kept = _preload(db, test_ids)
    from gpu_fabrication import _load_full
    X, _, scale = _load_full(db, norm_train, 100, max_records=n_train)
    Xn = X / scale[None, :, None]
    print(f"[ci] test={len(sigs)} train={Xn.shape} n_seeds={n_seeds}", flush=True)

    per_seed = {str(g): [] for g in guidances}
    for ms in range(n_seeds):
        model = DiffusionReconstructor(obs_idx, T=T, base=64, device="cuda",
                                       cond_dropout=0.15, seed=100 + ms)
        model.train(Xn, epochs=epochs, bs=64, lr=2e-4, log_every=20)
        for g in guidances:
            panel = _score(sigs, segidxs, models, oracle_rho,
                           _postmean(model, sigs, scale, g, T, K, chunk=chunk), cfg)
            d = _agg(panel, "delta_rho")
            per_seed[str(g)].append(d)
            print(f"  [seed {ms} w={g}] dRho={d:+.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    oracle_agg = _oracle_agg({"oracle_rho": oracle_rho})
    out = {"n_seeds": n_seeds, "n_test": len(sigs), "epochs": epochs, "guidances": guidances,
           "oracle_agg": oracle_agg, "per_seed_delta_rho": per_seed,
           "mean": {g: float(np.mean(v)) for g, v in per_seed.items()},
           "std": {g: float(np.std(v)) for g, v in per_seed.items()}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gpu_deficit_ci.json").write_text(json.dumps(out, indent=2))
    print("[json] results/gpu_deficit_ci.json")
    for g in guidances:
        m, s = out["mean"][str(g)], out["std"][str(g)]
        print(f"  w={g}: dRho = {m:+.3f} +/- {s:.3f}  (over {n_seeds} training seeds)")


def main(n_train, n_test, epochs, T, guidances, seeds, K, seed, chunk):
    import torch
    db, models, tr_samples, te_samples, sigs, segidxs, kept, model, scale = _build(
        n_train, n_test, epochs, T, seed)
    res = {"n_train": n_train, "n_test": len(sigs), "epochs": epochs, "T": T, "seeds": seeds,
           "K": K, "guidances": guidances, "test_population": "NORM fold-10"}
    res["primary"] = run_config(sigs, segidxs, models, CONFIGS["precordial-interp"], tr_samples,
                                te_samples, model, scale, T, guidances, seeds, K, "precordial", chunk)
    res["negctrl_limb6"] = run_config(sigs, segidxs, models, CONFIGS["limb6-negctrl"], tr_samples,
                                      te_samples, model, scale, T, guidances, seeds, K, "limb6", chunk)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gpu_diffusion.json").write_text(json.dumps(res, indent=2))
    print("[json] results/gpu_diffusion.json", flush=True)
    _emit_macros(res)


def validate(seed=0):
    """Feed IDENTICAL reconstructions to _score (batched) and _score_recon (per-record);
    assert the stats agree, isolating the scoring-math transcription. `kept` are all
    length>=1000 so _score_recon skips none -> the iterator stays aligned with Lhats."""
    db, models, tr_samples, te_samples, sigs, segidxs, kept, model, scale = _build(
        n_train=800, n_test=24, epochs=2, T=50, seed=seed)
    cfg = CONFIGS["precordial-interp"]
    oracle_rho = _oracle(models, tr_samples, te_samples, cfg)
    Lhats = _batched(model, sigs, scale, guidance=2.0, T=50, seed=7)
    mine = _score(sigs, segidxs, models, oracle_rho, Lhats, cfg)
    it = iter(Lhats)
    ref = _score_recon(db, models, oracle_rho, lambda _s: next(it), kept, cfg, 100, "ref")
    ok = True
    for s in CORE_SEGS:
        for l in cfg["target"]:
            a, b = mine.get(s, {}).get(l, {}), ref.get(s, {}).get(l, {})
            for k in ("h_mV", "rho_model", "delta_rho", "rho_recoverable"):
                if abs(a.get(k, 0) - b.get(k, 0)) > 1e-6:
                    print(f"MISMATCH {s}/{l}/{k}: mine={a.get(k)} ref={b.get(k)}"); ok = False
    print("VALIDATE", "PASS" if ok else "FAIL")


def emit_ci_macros():
    """Write paper/_gpu_ci_macros.tex from results/gpu_deficit_ci.json (error bars)."""
    d = json.loads((RESULTS / "gpu_deficit_ci.json").read_text())
    ws = sorted(d["mean"].keys(), key=float)
    wmax = ws[-1]
    w4 = min(ws, key=lambda w: abs(float(w) - 4.0))
    mmax, smax = d["mean"][wmax], d["std"][wmax]
    m4, s4 = d["mean"][w4], d["std"][w4]
    sig = lambda m, s: abs(m) / s if s > 1e-9 else float("inf")
    # count seeds monotone across the FULL sweep, and monotone for w>=2
    ps = d["per_seed_delta_rho"]
    nseed = len(next(iter(ps.values())))
    def mono(idxs):
        c = 0
        for k in range(nseed):
            seq = [ps[ws[i]][k] for i in idxs]
            if all(seq[j + 1] >= seq[j] for j in range(len(seq) - 1)):
                c += 1
        return c
    mono_full = mono(range(len(ws)))
    mono_from2 = mono(range(1, len(ws)))
    lines = [
        "% auto-generated by experiments/gpu_diffusion_clean.py --emit-ci-macros",
        f"\\newcommand{{\\nSeeds}}{{{d['n_seeds']}}}",
        f"\\newcommand{{\\wCImax}}{{{float(wmax):g}}}",
        f"\\newcommand{{\\dRhoCImax}}{{{mmax:+.3f}}}",
        f"\\newcommand{{\\dRhoCIsd}}{{{smax:.3f}}}",
        f"\\newcommand{{\\dRhoCIsigma}}{{{sig(mmax, smax):.1f}}}",
        f"\\newcommand{{\\dRhoCIfour}}{{{m4:+.2f}}}",
        f"\\newcommand{{\\dRhoCIfoursd}}{{{s4:.2f}}}",
        f"\\newcommand{{\\dRhoCIfoursigma}}{{{sig(m4, s4):.1f}}}",
        f"\\newcommand{{\\monoFull}}{{{mono_full}}}",
        f"\\newcommand{{\\monoFromTwo}}{{{mono_from2}}}",
    ]
    (PAPER / "_gpu_ci_macros.tex").write_text("\n".join(lines) + "\n")
    print("[tex] paper/_gpu_ci_macros.tex\n" + "\n".join(lines))
    print("\nfull CI curve (mean +/- std over seeds):")
    for w in ws:
        print(f"  w={w}: {d['mean'][w]:+.3f} +/- {d['std'][w]:.3f}")


def _oracle_agg(block, segs=CORE_SEGS):
    o = block["oracle_rho"]
    return float(np.mean([o.get(s, {}).get(l, 0.0) for s in segs for l in BAND_A]))


def _emit_macros(res):
    prim = res["primary"]; ora = prim["oracle_rho"]; sw = prim["sweep"]
    ws = sorted(sw.keys(), key=float); wmax = ws[-1]
    oraQRS = float(np.mean([ora.get("QRS", {}).get(l, 0) for l in BAND_A]))
    neg = res["negctrl_limb6"]
    negsw = neg["sweep"][wmax]["postmean"]
    negQRS = float(np.mean([negsw.get("QRS", {}).get(l, {}).get("delta_rho", 0) for l in BAND_A]))
    # rmse range across the sweep (postmean), for an honest "narrow band" statement
    rmses = [_agg_rmse(sw[w]["postmean"]) for w in ws]
    # operational-regime anchor: nearest guidance to w=4 (deficit already grown, RMSE flat)
    w4 = min(ws, key=lambda w: abs(float(w) - 4.0))
    lines = [
        "% auto-generated by experiments/gpu_diffusion_clean.py -- do not edit by hand",
        "% All aggregate macros average over CORE_SEGS (QRS,ST,T) x BAND_A (V2,V4,V6).",
        f"\\newcommand{{\\OraQRS}}{{{oraQRS:+.2f}}}",          # QRS-only Band-A oracle
        f"\\newcommand{{\\OraAgg}}{{{_oracle_agg(prim):+.2f}}}",   # primary aggregate oracle
        f"\\newcommand{{\\negOra}}{{{_oracle_agg(neg):+.2f}}}",    # limb6 aggregate oracle
        f"\\newcommand{{\\RhoRecFour}}{{{_agg(sw[w4]['postmean'],'rho_recoverable'):+.2f}}}",   # w=4 postmean
        f"\\newcommand{{\\RhoRec}}{{{_agg(sw[wmax]['postmean'],'rho_recoverable'):+.2f}}}",      # w=wmax postmean
        f"\\newcommand{{\\dRhoMeanFour}}{{{_agg(sw[w4]['postmean'],'delta_rho'):+.2f}}}",       # w=4
        f"\\newcommand{{\\dRhoMeanOne}}{{{_agg(sw[ws[0]]['postmean'],'delta_rho'):+.2f}}}",
        f"\\newcommand{{\\dRhoMeanMax}}{{{_agg(sw[wmax]['postmean'],'delta_rho'):+.2f}}}",
        f"\\newcommand{{\\dRhoSampMax}}{{{_agg(sw[wmax]['sample'],'delta_rho'):+.2f}}}",
        f"\\newcommand{{\\hMeanOne}}{{{_agg(sw[ws[0]]['postmean'],'h_mV'):.3f}}}",
        f"\\newcommand{{\\hMeanMax}}{{{_agg(sw[wmax]['postmean'],'h_mV'):.3f}}}",
        f"\\newcommand{{\\rmseFour}}{{{_agg_rmse(sw[w4]['postmean']):.3f}}}",   # w=4 RMSE (flat regime)
        f"\\newcommand{{\\rmseLo}}{{{min(rmses):.3f}}}",
        f"\\newcommand{{\\rmseHi}}{{{max(rmses):.3f}}}",
        f"\\newcommand{{\\wfour}}{{{float(w4):g}}}",
        f"\\newcommand{{\\negQRSdRho}}{{{negQRS:+.2f}}}",
        f"\\newcommand{{\\wmaxval}}{{{float(wmax):g}}}",
    ]
    (PAPER / "_gpu_macros.tex").write_text("\n".join(lines) + "\n")
    print("[tex] paper/_gpu_macros.tex\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--guidances", type=str, default="1.0,2.0,3.0,4.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--emit-macros", action="store_true",
                    help="regenerate paper/_gpu_macros.tex from existing results/gpu_diffusion.json (no GPU)")
    ap.add_argument("--ci", action="store_true",
                    help="error bars on the deficit over --n-seeds independent training seeds")
    ap.add_argument("--emit-ci-macros", action="store_true",
                    help="regenerate paper/_gpu_ci_macros.tex from results/gpu_deficit_ci.json (no GPU)")
    ap.add_argument("--n-seeds", type=int, default=4)
    args = ap.parse_args()
    if args.emit_macros:
        _emit_macros(json.loads((RESULTS / "gpu_diffusion.json").read_text()))
    elif getattr(args, "emit_ci_macros", False):
        emit_ci_macros()
    elif args.validate:
        validate()
    elif args.ci:
        main_ci(args.n_train, args.n_test, args.epochs, args.T,
                [float(x) for x in args.guidances.split(",")], args.K, args.n_seeds, args.seed, args.chunk)
    else:
        main(args.n_train, args.n_test, args.epochs, args.T,
             [float(x) for x in args.guidances.split(",")], args.seeds, args.K, args.seed, args.chunk)
