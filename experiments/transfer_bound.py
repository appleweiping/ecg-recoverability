"""Cross-cohort transfer bound for the recoverability certificate (sin-Theta / Davis-Kahan).

The per-lead identifiability eta_{s,l}(S) = || e_l^T M_s (I - P_obs) ||_2 is a function ONLY of the
segment-s dipolar subspace basis M_s (12x3, orthonormal columns; top-3 left singular vectors of the
segment's 12-lead sample matrix) and the observed set S. If two cohorts A, B have dipolar subspaces
that differ by principal angles Theta = (theta_1 >= theta_2 >= theta_3) (largest theta* between
colspace(M_s^A) and colspace(M_s^B) in R^12), then eta transfers with a bounded drift.

Proposition (cross-cohort transfer). With M_s orthonormal-column (so ||m_l|| <= 1 and eta <= 1),
for every target lead l and observed set S,

    | eta^A_{s,l}(S) - eta^B_{s,l}(S) |  <=  2 sin(theta*/2) * (1 + 2 kappa_S),

where kappa_S = || M_{s,S}^+ ||_2 = || M_s M_{s,S}^+ ||_2 (the certificate's observed-set
conditioning; the equality uses that M_s is an isometry on its 3-dim column space). Since eta is
rotation-invariant (a common column-space rotation R in O(3) leaves every eta unchanged), theta*=0
gives eta^A = eta^B EXACTLY; the bound degrades smoothly as the subspaces separate, and becomes
vacuous (>= 1 = max eta) precisely when the cohorts' dipolar geometry is near-orthogonal -- i.e. the
certificate refuses to claim transfer exactly when it should.

Proof sketch. Align bases by the optimal R (rotation-invariance), giving ||M_s^A - M_s^B||_2 =
2 sin(theta*/2) =: delta. Then eta^A_l - eta^B_l = ||(I-P^A_S) m^A_l|| - ||(I-P^B_S) m^B_l||; by the
reverse triangle inequality this is <= ||(I-P^A_S)(m^A_l - m^B_l)|| + ||(P^B_S - P^A_S) m^B_l||
<= delta + ||P^A_S - P^B_S||_2. A projector onto colspace(M_{s,S}) perturbs by
||Delta P||_2 <= 2 ||Delta M_{s,S}||_2 ||M_{s,S}^+||_2 <= 2 delta kappa_S (Stewart/Wedin), and
||m^B_l|| <= 1. []

This script validates the bound on the flagship limb-6 -> precordial(V2) verdict using the two
cohorts (PTB-XL, Chapman) already fit and lineage-stamped in results/cross_dataset.json: it checks
LHS <= RHS per segment and reports the realized Lipschitz ratio drift / (2 sin(theta*/2)), which
should be a small constant well inside the certified envelope. The QRS-vs-ST contrast is the
result: QRS (theta*=13.8 deg) transfers tightly; ST (theta*=85.7 deg) does not, and the bound
correctly separates them.

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
        # realized Lipschitz constant of eta in the chordal subspace distance (empirical slope)
        lipschitz = (drift / delta) if delta > 0 else float("nan")
        # drift is a POINT estimate; propagate the target-cohort eta bootstrap CI so the ST cell
        # (whose eta CI is nearly [0,1]) is not read as a precise number. drift_ci = range of
        # |eta_A - c| over c in the Chapman eta CI; lipschitz_ci = drift_ci / delta.
        eta_ci = blk.get("chapman_limb6_V2eta_ci")
        drift_ci = lip_ci = None
        if eta_ci:
            lo_c, hi_c = float(eta_ci[0]), float(eta_ci[1])
            dlo = 0.0 if lo_c <= eta_A <= hi_c else min(abs(eta_A - lo_c), abs(eta_A - hi_c))
            dhi = max(abs(eta_A - lo_c), abs(eta_A - hi_c))
            drift_ci = [round(dlo, 4), round(dhi, 4)]
            lip_ci = [round(dlo / delta, 4), round(dhi / delta, 4)] if delta > 0 else None
        # angle-based pre-deployment transfer diagnostic (reported heuristic, NOT claimed rigorous):
        # a small principal angle means shared dipolar geometry -> small expected drift.
        flag = ("shared" if theta_star < 30 else "divergent" if theta_star > 60 else "partial")
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
            "bound_holds": bool(holds),
            "realized_lipschitz": round(lipschitz, 4),
            "realized_lipschitz_ci": lip_ci,
            "transfer_flag": flag,
        }

    # The rigorous bound holds in every case; the transfer STORY is carried by (i) the realized
    # Lipschitz constant being small and stable across a wide angle range (drift is genuinely
    # O(chordal distance), not merely bounded by it), and (ii) the measured angle separating the
    # shared-geometry segment (QRS) from the divergent one (ST) BEFORE seeing target-cohort labels.
    lips = [per_seg[s]["realized_lipschitz"] for s in SEGMENTS]
    summary = {
        "target": TARGET, "observed_set": "limb-6",
        "all_bounds_hold": bool(all_hold),
        # point estimates only; the ST realized-Lipschitz is nearly unconstrained (see its CI), so
        # we do NOT claim a stable cross-segment constant -- the rigorous claim is the bound + the
        # angle as a pre-deployment transfer-RISK flag, corroborated by the tight QRS point.
        "realized_lipschitz_point": {s: per_seg[s]["realized_lipschitz"] for s in SEGMENTS},
        "st_lipschitz_ci": per_seg["ST"].get("realized_lipschitz_ci"),
        "qrs_flag": per_seg["QRS"]["transfer_flag"],
        "st_flag": per_seg["ST"]["transfer_flag"],
        "qrs_drift_tight": bool(per_seg["QRS"]["empirical_drift"] < 0.1),
        "chordal_ratio_st_over_qrs": round(per_seg["ST"]["chordal_delta"]
                                           / max(per_seg["QRS"]["chordal_delta"], 1e-9), 2),
    }

    lin = lineage.make(seed=0, targets=[TARGET],
                       normalization="raw mV (no per-record scaling)",
                       extra={"kind": "cross-cohort transfer bound (sin-Theta) validation",
                              "source": "cross_dataset.json",
                              "source_commit": cross.get("lineage", {}).get("commit")})
    out = {"lineage": lin, "bound": "|eta_A - eta_B| <= 2 sin(theta*/2) (1 + 2 kappa_S)",
           "segments": per_seg, "summary": summary}
    (RESULTS / "transfer_bound.json").write_text(json.dumps(out, indent=2))
    print("[json] results/transfer_bound.json", flush=True)
    for seg, d in per_seg.items():
        print(f"  {seg}: theta*={d['theta_star_deg']}deg drift={d['empirical_drift']} "
              f"rhs={d['certified_rhs']} holds={d['bound_holds']} flag={d['transfer_flag']} "
              f"lip={d['realized_lipschitz']}", flush=True)
    print(f"[transfer] all_hold={all_hold} lipschitz_point={summary['realized_lipschitz_point']} "
          f"st_lip_ci={summary['st_lipschitz_ci']} qrs={summary['qrs_flag']} st={summary['st_flag']} "
          f"chordal_ratio={summary['chordal_ratio_st_over_qrs']}x", flush=True)


if __name__ == "__main__":
    main()
