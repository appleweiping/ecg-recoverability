"""Delineator x sampling-rate robustness of the ST-safety endpoint.

Every fiducial-dependent number (the ST FP/FN rates, the PR-baseline fallback fraction) inherits the
NeuroKit2 delineator and the sampling rate -- a potential single point of failure. This compares
st_safety across dwt x {100, 500 Hz} (rate) plus a cwt @ 100 Hz cell (delineator), matched n_test,
and reports (NeuroKit2's "peak" method emits only wave peaks -- no onsets/offsets -- so it cannot
populate the boundary-defined ST segment [J, T_onset); "cwt" is the boundary-emitting alternative):

  - VERDICT sign-stability: the limb-6 observed set is STRUCTURALLY rank-2 (the six frontal-plane
    leads span only 2 of the 3 dipole dimensions -- an algebraic identity, cohort/delineator/rate
    independent), so any precordial lead with a component in the unobserved 3rd dimension has
    eta_ST > 0 at EVERY setting. We confirm every flagged precordial eta stays > 0 and that the
    per-lead recoverability ORDERING is preserved (Spearman across settings). eta is fit from the
    delineator-located ST samples, so it is not bitwise-identical across delineators -- only the
    sign (structural) and the ordering are claimed invariant.
  - RATE-stability (dwt @ 100 vs 500 Hz, same delineator): the tight axis -- max |Delta eta| over
    the strong leads should be small (few x 1e-3).
  - MAGNITUDE stability: per-reconstructor FP/FN ST-threshold-event rates; we confirm the endpoint
    stays FN-dominated (FN > FP) in EVERY setting -- it under-warns, never over-warns, the
    clinically conservative failure direction -- and report the FP/FN spread.

The cwt cell delineates far fewer records cleanly (small n, high PR-baseline fallback); it is used
only to corroborate the verdict SIGN and ordering, not the precise magnitudes.

Output: results/delineator_robustness.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ecgcert import lineage  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETTINGS = {  # label -> (json file, delineator, rate)
    "dwt-100": ("st_safety_dwt100.json", "dwt", 100),
    "cwt-100": ("st_safety_cwt100.json", "cwt", 100),
    "dwt-500": ("st_safety_dwt500.json", "dwt", 500),
}
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
STRONG = ("V1", "V2", "V3", "V4")  # eta_normalized clearly > 0 at 100 Hz; V5/V6 are near-recoverable
RECONS = ("dipolar", "ridge", "ols")


def _spearman(a, b):
    """Spearman rho between two equal-length sequences (no scipy dependency)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def main():
    loaded = {}
    for lab, (fn, delin, rate) in SETTINGS.items():
        p = RESULTS / fn
        if p.exists():
            loaded[lab] = (json.loads(p.read_text()), delin, rate)
    if len(loaded) < 2:
        raise FileNotFoundError(f"delineator_robustness needs >=2 of {list(SETTINGS)}; have {list(loaded)}")
    labels = list(loaded)

    # ---- verdict: per-lead eta_ST across settings ----
    eta_by = {lab: {l: loaded[lab][0]["certificate_ST_precordial"][l]["eta"] for l in PRECORDIAL}
              for lab in labels}
    sign_stable = all(eta_by[lab][l] > 0 for lab in labels for l in PRECORDIAL)
    # ordering preserved: Spearman of the 6-lead eta profile, each setting vs dwt-100
    ref = "dwt-100" if "dwt-100" in labels else labels[0]
    ordering = {lab: round(_spearman([eta_by[ref][l] for l in PRECORDIAL],
                                     [eta_by[lab][l] for l in PRECORDIAL]), 4)
                for lab in labels if lab != ref}
    strong_spread = round(max(abs(eta_by[a][l] - eta_by[b][l])
                              for a in labels for b in labels for l in STRONG), 4)

    # ---- rate axis: dwt-100 vs dwt-500 (same delineator) ----
    rate_pair = ("dwt-100", "dwt-500")
    rate_stability = None
    if all(x in labels for x in rate_pair):
        a, b = rate_pair
        rate_stability = {
            "pair": list(rate_pair),
            "max_abs_deta_strong": round(max(abs(eta_by[a][l] - eta_by[b][l]) for l in STRONG), 4),
            "max_abs_deta_all": round(max(abs(eta_by[a][l] - eta_by[b][l]) for l in PRECORDIAL), 4),
        }

    # ---- magnitudes: FP/FN + FN-dominance ----
    mag = {r: {} for r in RECONS}
    fn_dominated_all = True
    for r in RECONS:
        for m in ("false_positive_rate", "false_negative_rate", "total_wrong_rate", "mean_st_error_mv"):
            vals = {lab: loaded[lab][0]["reconstructors"][r].get(m) for lab in labels}
            present = {lab: v for lab, v in vals.items() if v is not None}
            mag[r][m] = {"by_setting": {lab: round(v, 4) for lab, v in present.items()},
                         "max_spread": round(max(present.values()) - min(present.values()), 4)
                         if present else None}
        for lab in labels:
            rc = loaded[lab][0]["reconstructors"][r]
            fp, fn = rc.get("false_positive_rate"), rc.get("false_negative_rate")
            if fp is not None and fn is not None and not (fn > fp):
                fn_dominated_all = False

    n_common = {lab: loaded[lab][0].get("n_valid_common") for lab in labels}
    fb = {lab: loaded[lab][0].get("baseline_fallback_frac") for lab in labels}

    # provenance: this is a meta-analysis over the three st_safety runs; record their lineage.
    src_lineage = {lab: {"file": SETTINGS[lab][0],
                         "commit": loaded[lab][0].get("lineage", {}).get("commit"),
                         "segment_def": loaded[lab][0].get("lineage", {}).get("segment_def")}
                   for lab in labels}
    lin = lineage.make(seed=0, targets=list(PRECORDIAL),
                       normalization="raw mV (no per-record scaling)",
                       extra={"source_runs": src_lineage,
                              "kind": "delineator-x-rate robustness meta-analysis"})

    out = {
        "lineage": lin,
        "settings": {lab: {"delineator": loaded[lab][1], "rate": loaded[lab][2],
                           "n_valid_common": n_common[lab],
                           "baseline_fallback_frac": (round(fb[lab], 4) if fb[lab] is not None else None)}
                     for lab in labels},
        "verdict_eta_ST_precordial": {lab: {l: round(eta_by[lab][l], 4) for l in PRECORDIAL} for lab in labels},
        "verdict_sign_stable": bool(sign_stable),
        "verdict_ordering_spearman_vs_ref": {"ref": ref, "rho": ordering},
        "verdict_strong_lead_max_spread": strong_spread,
        "rate_stability": rate_stability,
        "magnitude_FP_FN_total": mag,
        "endpoint_fn_dominated_all_settings": bool(fn_dominated_all),
        "note": ("The limb-6 observed set is structurally rank-2 (frontal-plane leads span 2 of 3 "
                 "dipole dimensions -- an algebraic identity), so the precordial-not-recoverable "
                 "verdict (eta_ST > 0) holds at every delineator/rate; magnitudes shift but the "
                 "recoverability ordering and the FN-dominated (under-warning) failure direction are "
                 "preserved. The cwt cell delineates few records cleanly (small n, high PR-baseline "
                 "fallback) and corroborates only the verdict sign/ordering, not the magnitudes."),
    }
    (RESULTS / "delineator_robustness.json").write_text(json.dumps(out, indent=2))
    print("[json] results/delineator_robustness.json", flush=True)
    print(f"[rob] sign_stable={out['verdict_sign_stable']} "
          f"ordering_rho={ordering} strong_spread={strong_spread} "
          f"fn_dominated_all={out['endpoint_fn_dominated_all_settings']}", flush=True)
    if rate_stability:
        print(f"[rob] rate(dwt 100 vs 500): max|dEta| strong={rate_stability['max_abs_deta_strong']} "
              f"all={rate_stability['max_abs_deta_all']}", flush=True)
    for lab in labels:
        print(f"  {lab}: n={n_common[lab]} fallback={fb[lab]}", flush=True)


if __name__ == "__main__":
    main()
