"""M5b: cross-device shift within PTB-XL (Schiller CS-family -> AT-family).

PTB-XL records come from several acquisition devices.  We calibrate the Tier II
conformal intervals on the CS family (CS100 / CS-12 / CS-12 E) and test coverage on
the AT family (AT-6 / AT-60) -- a genuine device shift.  Plain split conformal loses
coverage under the shift; weighted conformal (device likelihood-ratio weights) and a
small-slice recalibration restore it.  This isolates exactly what the certificate
promises: Tier I exactness and Tier III soundness are distribution-free, only Tier II
coverage level is shift-sensitive, and it is measurable and repairable.

Outputs: results/cross_device.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgcert.conformal import (
    conformal_quantile,
    cqr_interval,
    empirical_coverage,
    weighted_conformal_quantile,
)
from ecgcert.data import PTBXL
from ecgcert.estimators import LinearDipolarReconstructor
from ecgcert.models import fit_segment_models
from ecgcert.physics import LEAD_INDEX

RESULTS = Path(__file__).resolve().parent.parent / "results"
OBS = ["I", "II", "V2"]
TARGET_LEAD = "V4"           # a reconstructed precordial lead
SEG = "ST"                   # ST segment: clinically salient, non-dipolar content
CS = ("CS100    3", "CS-12", "CS-12   E")
AT = ("AT-6 C 5.5", "AT-6     6", "AT-60    3", "AT-6 C", "AT-6 C 5.8")


def _device_family(name: str) -> str | None:
    if name in CS:
        return "CS"
    if name in AT:
        return "AT"
    return None


def _residual_samples(db, ids, models, rate, max_records):
    """Per record: reconstruct with the dipolar model on segment SEG; return the
    (center, true) at TARGET_LEAD averaged over the segment window."""
    m = models[SEG]
    lin = LinearDipolarReconstructor(m.M, m.mu, OBS)
    tl = LEAD_INDEX[TARGET_LEAD]
    obs_idx = [LEAD_INDEX[l] for l in OBS]
    centers, truths = [], []
    for eid in ids[:max_records]:
        try:
            sig = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        idx = db.segment_indices(sig, fs=rate)[SEG]
        if idx.size < 4:
            continue
        yS = sig[idx][:, obs_idx].T
        Lhat = lin.predict(yS)
        centers.append(float(np.mean(Lhat[tl])))
        truths.append(float(np.mean(sig[idx, tl])))
    return np.array(centers), np.array(truths)


def main(n_train=500, rate=100, seed=0, alpha=0.1):
    db = PTBXL()
    db.meta["dev_family"] = db.meta["device"].map(_device_family)
    rng = np.random.default_rng(seed)

    train_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False,
                                                       folds=range(1, 9)))
    seg_samples = db.collect_all_segments(train_ids, rate=rate, max_per_record=40,
                                          max_records=n_train, seed=seed)
    models = fit_segment_models(seg_samples)

    cs_ids = rng.permutation(db.meta[(db.meta.dev_family == "CS")].index.to_numpy())
    at_ids = rng.permutation(db.meta[(db.meta.dev_family == "AT")].index.to_numpy())

    cs_c, cs_y = _residual_samples(db, list(cs_ids), models, rate, 900)
    at_c, at_y = _residual_samples(db, list(at_ids), models, rate, 900)

    # Split CS into calibration (fit spread + conformal) and test.
    n = len(cs_c); k = n // 2
    cal, cst = slice(0, k), slice(k, n)
    spread = np.std(cs_y[cal] - cs_c[cal])
    q_lo_cal, q_hi_cal = cs_c[cal] - spread, cs_c[cal] + spread
    scores = np.maximum(q_lo_cal - cs_y[cal], cs_y[cal] - q_hi_cal)

    # Plain split conformal.
    Q = conformal_quantile(scores, alpha)
    cov_cs = empirical_coverage(*cqr_interval(cs_c[cst] - spread, cs_c[cst] + spread, Q), cs_y[cst])
    cov_at_plain = empirical_coverage(*cqr_interval(at_c - spread, at_c + spread, Q), at_y)

    # Weighted conformal: LR weights from a simple density ratio on the score feature.
    # Estimate w(x) ~ p_AT(center)/p_CS(center) via Gaussian fits on the center stat.
    mu_cs, sd_cs = cs_c[cal].mean(), cs_c[cal].std() + 1e-6
    mu_at, sd_at = at_c.mean(), at_c.std() + 1e-6
    def logpdf(x, m, s):
        return -0.5 * ((x - m) / s) ** 2 - np.log(s)
    w = np.exp(logpdf(cs_c[cal], mu_at, sd_at) - logpdf(cs_c[cal], mu_cs, sd_cs))
    w = np.clip(w, 1e-3, 1e3)
    Qw = weighted_conformal_quantile(scores, w, alpha)
    cov_at_weighted = empirical_coverage(*cqr_interval(at_c - spread, at_c + spread, Qw), at_y)

    # Small-slice recalibration: a few hundred AT points restore exact coverage.
    n_slice = 300
    at_idx = rng.permutation(len(at_c))
    sl, at_test = at_idx[:n_slice], at_idx[n_slice:]
    q_lo_at, q_hi_at = at_c[sl] - spread, at_c[sl] + spread
    scores_at = np.maximum(q_lo_at - at_y[sl], at_y[sl] - q_hi_at)
    Qs = conformal_quantile(scores_at, alpha)
    cov_at_recal = empirical_coverage(at_c[at_test] - spread - Qs, at_c[at_test] + spread + Qs,
                                      at_y[at_test])

    out = {
        "config": OBS, "segment": SEG, "target_lead": TARGET_LEAD, "alpha": alpha,
        "target_coverage": 1 - alpha,
        "n_cs": int(len(cs_c)), "n_at": int(len(at_c)),
        "coverage_CS_indist": float(cov_cs),
        "coverage_AT_plain": float(cov_at_plain),
        "coverage_AT_weighted": float(cov_at_weighted),
        "coverage_AT_recalibrated": float(cov_at_recal),
        # interval widths (mV): a "restored" coverage is only meaningful if the
        # interval did not merely blow up.
        "width_AT_plain": float(2 * spread + 2 * Q),
        "width_AT_weighted": float(2 * spread + 2 * Qw),
        "width_AT_recalibrated": float(2 * spread + 2 * Qs),
    }
    (RESULTS / "cross_device.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
