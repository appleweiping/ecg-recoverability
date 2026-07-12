"""M5: STEMI safety case -- fabricated / masked ST elevation, and abstention.

Clinical framing: limb-lead monitors reconstruct the precordial leads (V1-V6),
but the precordial ST content is largely non-dipolar and lies in the observation
null space (kappa(limb-6 -> precordial) ~ 6.7e4).  A generative reconstructor
*fabricates* precordial ST deviation (phantom STEMI); a dipolar / OLS
reconstructor *blurs* it (masking a real STEMI).  We measure both, and show that
abstaining on certificate-flagged (high-``h``) reconstructed ST segments removes
the dangerous cases -- the "error prevented by the certificate" number.

Outputs: results/ptbxl_stemi.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.certify import off_dipole_energy
from ecgcert.clinical import count_stemi_flips, st_deviation, stemi_positive
from ecgcert.conformal import flag_threshold
from ecgcert.data import PTBXL
from ecgcert.estimators import GenerativeSampleReconstructor, LinearDipolarReconstructor, OLSReconstructor
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("P", "QRS", "ST", "T")
LIMB6 = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]
PRECORDIAL_IDX = [LEAD_INDEX[l] for l in PRECORDIAL]


def _reconstruct_full(sig, db, rate, models, obs, recons):
    """Reconstruct the whole (T,12) signal per-segment; unlabelled samples kept as
    the dipolar recon of the QRS model (baseline). Returns {recon_name: (T,12)}."""
    segidx = db.segment_indices(sig, fs=rate)
    obs_idx = [LEAD_INDEX[l] for l in obs]
    T = sig.shape[0]
    out = {}
    # per-record hallucination energy on the ST segment (precordial), for flagging.
    st_h = {}
    for rname in recons:
        rec_full = np.tile(sig.mean(axis=0)[:, None], (1, T)).astype(float)  # (12,T) default=mean
        rec_full[obs_idx] = sig[:, obs_idx].T                                # keep observed
        out[rname] = rec_full
    for seg in SEGMENTS:
        idx = segidx[seg]
        if idx.size < 4 or seg not in models:
            continue
        m = models[seg]
        yS = sig[idx][:, obs_idx].T
        builders = {
            "dipolar": LinearDipolarReconstructor(m.M, m.mu, obs),
            "ols": recons["ols"],
            "generative": GenerativeSampleReconstructor(m.M, m.mu, obs, m.Sigma_r, seed=1),
        }
        for rname, rec in builders.items():
            Lhat = rec.predict(yS)                       # (12, T_seg)
            out[rname][:, idx] = Lhat
            if seg == "ST":
                h = off_dipole_energy(m.M, m.mu, obs, Lhat)
                st_h[rname] = float(np.mean(h[PRECORDIAL_IDX]))
    return out, st_h


def main(n_train=500, n_test=800, rate=100, seed=0, alpha=0.1):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False,
                                                       folds=range(1, 9)))
    seg_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=40,
                                          max_records=n_train, seed=seed)
    models = fit_segment_models(seg_samples)
    L_train = np.hstack([seg_samples[s].T for s in SEGMENTS if seg_samples[s].shape[0] > 0])
    ols = OLSReconstructor(LIMB6).fit(L_train)

    # Test on fold 10, over all superclasses (need genuine ST changes -> include MI/STTC).
    test_ids = rng.permutation(db.meta[db.meta["strat_fold"] == 10].index.to_numpy())[:n_test]

    recons = {"dipolar": None, "ols": ols, "generative": None}
    rows = {r: [] for r in ("dipolar", "ols", "generative")}
    st_h_all = {r: [] for r in recons}
    # First pass: gather ST hallucination energies on NORM (faithful-ish) for tau.
    for eid in test_ids:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        recon, st_h = _reconstruct_full(sig, db, rate, models, LIMB6, recons)
        for rname in ("dipolar", "ols", "generative"):
            # recon[rname] is (12, T); st_deviation expects (T, 12).
            flip = count_stemi_flips(sig, recon[rname].T, rate, leads=PRECORDIAL_IDX)
            if flip.get("valid"):
                flip["st_h"] = st_h.get(rname, np.nan)
                rows[rname].append(flip)

    out = {"config": "limb-6 -> precordial", "n_eval": {r: len(rows[r]) for r in rows},
           "alpha": alpha, "reconstructors": {}}
    for rname in ("dipolar", "ols", "generative"):
        R = rows[rname]
        if not R:
            continue
        fab = np.array([r["fabricated"] for r in R])
        msk = np.array([r["masked"] for r in R])
        sterr = np.array([r["st_error_mv"] for r in R])
        h = np.array([r["st_h"] for r in R])
        # Calibrate the ST-segment flag threshold on a HELD-OUT half of this
        # reconstructor's own faithful (not fabricated/masked) records, and evaluate
        # the flag on the disjoint other half + all fabricated/masked records, so the
        # false-flag rate is a genuine held-out quantity, not guaranteed in-sample.
        # tau is floored at a physical epsilon (1e-6 mV): a reconstructor with h==0
        # (the dipolar one) then never flags -- h is blind inside the recoverable
        # subspace, which is the honest behaviour, not a machine-epsilon "detection".
        eps = 1e-6
        faithful_idx = np.where(~(fab | msk) & ~np.isnan(h))[0]
        cal_rng = np.random.default_rng(seed + 777)
        cal_rng.shuffle(faithful_idx)
        cal_idx = faithful_idx[: faithful_idx.size // 2]
        eval_mask = np.ones(len(R), bool)
        eval_mask[cal_idx] = False                       # held-out evaluation records
        tau = max(flag_threshold(h[cal_idx], alpha), eps) if cal_idx.size else np.inf
        flagged = (h > tau) & eval_mask
        eval_faithful = eval_mask & ~(fab | msk) & ~np.isnan(h)
        fab_prevented = int(np.sum(fab & flagged))
        msk_prevented = int(np.sum(msk & flagged))
        out["reconstructors"][rname] = {
            "n": len(R),
            "fabricated_stemi": int(fab.sum()),
            "masked_stemi": int(msk.sum()),
            "mean_st_error_mv": float(np.nanmean(sterr)),
            "flag_tau": float(tau),
            "fabricated_prevented_by_flag": fab_prevented,
            "masked_prevented_by_flag": msk_prevented,
            "flag_rate_heldout_faithful": float(np.mean((h > tau)[eval_faithful]))
            if eval_faithful.any() else 0.0,
        }
        d = out["reconstructors"][rname]
        print(f"[{rname:11s}] fabricated={d['fabricated_stemi']:3d} masked={d['masked_stemi']:3d} "
              f"ST-err={d['mean_st_error_mv']:.3f}mV  ff={d['flag_rate_heldout_faithful']:.3f}  "
              f"prevented(fab/msk)={fab_prevented}/{msk_prevented}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ptbxl_stemi.json").write_text(json.dumps(out, indent=2))
    print("\n[json] results/ptbxl_stemi.json")


if __name__ == "__main__":
    main()
