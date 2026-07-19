"""Cross-cohort SENSITIVITY of the recoverability verdict (sin-Theta descriptor; NOT a certified bound).

The per-lead identifiability eta_{s,l}(S) = || e_l^T M_s (I - P_obs) ||_2 depends on M_s ONLY through
its column space (a common O(3) rotation of M_s leaves every eta unchanged), so if two cohorts share
a dipolar subspace (all principal angles zero) the verdict is identical, and in general drift is
governed by the principal angle theta* between colspace(M_s^A) and colspace(M_s^B).

A sin-Theta (Davis-Kahan) perturbation gives
    | eta^A - eta^B |  <=  2 sin(theta*/2) * (1 + 2 max(kappa_A, kappa_B)),   kappa = || M_{s,S}^+ ||_2,
using the Stewart-Wedin projector-difference bound and assuming the observed block keeps its rank and
the rho-truncated projector equals the exact one. IMPORTANT: at the conditioning we observe this RHS
is VACUOUS -- it evaluates to ~1.37 (QRS) and ~32.2 (ST), both exceeding the trivial |Delta eta| <= 1
(eta in [0,1]). So the inequality certifies NOTHING here; it does not establish that any verdict
transfers, and is not a pre-deployment guarantee. A non-vacuous statement would need a target-specific
bound from the aligned target-lead rows and the exact projector difference (future work).

This script therefore reports EMPIRICAL cross-cohort sensitivity on the limb-6 -> V2 verdict from the
two cohorts (PTB-XL, Chapman) in results/cross_dataset.json: the measured principal angle, the
measured eta drift (with the target-cohort bootstrap CI), and the (vacuous) RHS for transparency.
Descriptively QRS subspaces are nearly aligned (theta*~13.8 deg) while ST are near-orthogonal
(theta*~85.7 deg); this is a two-cohort observation, not a guarantee.

Output: results/transfer_bound.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ecgcert import lineage  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEPLOY_RCOND = "0.01"          # matches cross_dataset DEPLOY_RCOND
SEGMENTS = ("QRS", "ST")       # flagship dipolar-safety segments
TARGET = "V2"                  # flagship precordial target for the limb-6 verdict


def _bound(theta_deg: float, kappa: float) -> float:
    """RHS = 2 sin(theta*/2) (1 + 2 kappa)  (rigorous general form)."""
    delta = 2.0 * math.sin(math.radians(theta_deg) / 2.0)     # chordal subspace distance
    return delta * (1.0 + 2.0 * kappa)


def main():
    cross = json.loads((RESULTS / "cross_dataset.json").read_text())
    mn = cross["comparisons"]["matched_normal"]

    per_seg = {}
    all_hold = True
    for seg in SEGMENTS:
        blk = mn[seg]
        theta_star = float(blk["max_angle_deg"])
        theta_ci = blk.get("max_angle_ci")
        limb = blk["recoverability"]["limb-6"]
        eta_A = float(limb["ptbxl"]["V2_eta"])                 # PTB-XL (source)
        eta_B = float(limb["chapman"]["V2_eta"])               # Chapman (target)
        kap_A = float(limb["ptbxl"]["rcond_sweep"][DEPLOY_RCOND]["kappa_global"])
        kap_B = float(limb["chapman"]["rcond_sweep"][DEPLOY_RCOND]["kappa_global"])
        kappa = max(kap_A, kap_B)                              # conservative (worst observed set)
        drift = abs(eta_A - eta_B)
        rhs = _bound(theta_star, kappa)
        delta = 2.0 * math.sin(math.radians(theta_star) / 2.0)
        holds = drift <= rhs + 1e-9
        all_hold = all_hold and holds
        # empirical ratio of drift to chordal subspace distance (descriptor only)
        lipschitz = (drift / delta) if delta > 0 else float("nan")
        # drift is a POINT estimate; propagate the target-cohort eta bootstrap CI so the ST cell
        # (whose eta CI is nearly [0,1]) is not read as a precise number.
        eta_ci = blk.get("chapman_limb6_V2eta_ci")
        drift_ci = lip_ci = None
        if eta_ci:
            lo_c, hi_c = float(eta_ci[0]), float(eta_ci[1])
            dlo = 0.0 if lo_c <= eta_A <= hi_c else min(abs(eta_A - lo_c), abs(eta_A - hi_c))
            dhi = max(abs(eta_A - lo_c), abs(eta_A - hi_c))
            drift_ci = [round(dlo, 4), round(dhi, 4)]
            lip_ci = [round(dlo / delta, 4), round(dhi / delta, 4)] if delta > 0 else None
        # The certified RHS is VACUOUS (> the trivial |Delta eta| <= 1) at this conditioning, so we
        # report empirical sensitivity only and emit NO shared/divergent threshold flag.
        per_seg[seg] = {
            "theta_star_deg": round(theta_star, 3),
            "theta_star_ci_deg": theta_ci,
            "sin_theta_star": round(math.sin(math.radians(theta_star)), 4),
            "chordal_delta": round(delta, 4),
            "eta_ptbxl": round(eta_A, 4),
            "eta_chapman": round(eta_B, 4),
            "eta_chapman_ci": blk.get("chapman_limb6_V2eta_ci"),
            "kappa_used": round(kappa, 4),
            "empirical_drift": round(drift, 4),
            "empirical_drift_ci": drift_ci,
            "certified_rhs": round(rhs, 4),
            "rhs_is_vacuous": bool(rhs > 1.0),          # RHS exceeds the trivial |Delta eta| <= 1
            "realized_ratio": round(lipschitz, 4),
            "realized_ratio_ci": lip_ci,
        }

    # The rigorous bound holds in every case; the transfer STORY is carried by (i) the realized
    # Lipschitz constant being small and stable across a wide angle range (drift is genuinely
    # Empirical cross-cohort SENSITIVITY, not a certified bound: the sin-Theta RHS is vacuous
    # (> the trivial |Delta eta| <= 1) at the observed conditioning, so we make NO transfer claim.
    summary = {
        "target": TARGET, "observed_set": "limb-6",
        "framing": "empirical cross-cohort sensitivity (the sin-Theta RHS is vacuous here)",
        "rhs_vacuous_all_segments": bool(all(per_seg[s]["rhs_is_vacuous"] for s in SEGMENTS)),
        "certified_rhs": {s: per_seg[s]["certified_rhs"] for s in SEGMENTS},
        "empirical_drift": {s: per_seg[s]["empirical_drift"] for s in SEGMENTS},
        "principal_angle_deg": {s: per_seg[s]["theta_star_deg"] for s in SEGMENTS},
        "st_drift_ci": per_seg["ST"].get("empirical_drift_ci"),
    }

    lin = lineage.make(seed=0, targets=[TARGET],
                       normalization="raw mV (no per-record scaling)",
                       extra={"kind": "cross-cohort sensitivity; sin-Theta RHS reported but VACUOUS",
                              "source": "cross_dataset.json",
                              "source_commit": cross.get("lineage", {}).get("commit")})
    out = {"lineage": lin,
           "bound": "|eta_A - eta_B| <= 2 sin(theta*/2)(1 + 2 max(kappa_A,kappa_B))  [VACUOUS here: RHS > 1]",
           "segments": per_seg, "summary": summary}
    (RESULTS / "transfer_bound.json").write_text(json.dumps(out, indent=2))
    print("[json] results/transfer_bound.json", flush=True)
    for seg, d in per_seg.items():
        print(f"  {seg}: theta*={d['theta_star_deg']}deg drift={d['empirical_drift']} "
              f"rhs={d['certified_rhs']} vacuous={d['rhs_is_vacuous']} "
              f"realized_ratio={d['realized_ratio']}", flush=True)
    print(f"[transfer] rhs_vacuous_all={summary['rhs_vacuous_all_segments']} "
          f"(empirical sensitivity only; no certified transfer claim)", flush=True)


if __name__ == "__main__":
    main()
