r"""Certificate-guided active lead selection: which electrodes to place under a budget.

The recoverability certificate is diagnostic (given S, what is recoverable). It is also
\emph{prescriptive}: pick the observed set S (an electrode budget) that minimises the total certified
ambiguity of the full 12-lead reconstruction,

    J(S) = sum_{s in segments} sum_{l=1..12} a_{s,l}(S)^2 ,   a_{s,l}(S) = expected_ambiguity (mV),

i.e. choose the k measured leads that leave the least of the ECG dipole-unrecoverable. We compare a
GREEDY forward selection (add the lead giving the largest marginal drop in J) against the EXHAUSTIVE
optimum (feasible for small k over the 8 independent measured leads {I,II,V1..V6}; III/aVR/aVL/aVF are
linear combinations of I,II and add no information) and a RANDOM baseline. We also EMPIRICALLY test
the diminishing-returns (submodularity) inequality of the ambiguity reduction rather than assuming it,
so any (1-1/e) reading of the greedy solution is earned, not asserted.

Output: results/active_selection.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.physics import (
    LEADS, LEAD_INDEX, fit_dipolar_subspace, dipole_coord_cov, expected_ambiguity_per_lead,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("QRS", "ST", "T")
# the 8 physically independent measured leads (electrode budget is spent on these; the 4 augmented
# limb leads III/aVR/aVL/aVF are exact linear combinations of I and II).
CANDIDATES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
NORMALIZATION = "raw mV segment samples; total certified squared ambiguity objective"


def run(n_train=1500, max_per_record=40, seed=0, n_random=200):
    db = PTBXL()
    rng = np.random.default_rng(seed)
    tr_ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))[:n_train]
    tr_seg = db.collect_all_segments_with_ids(tr_ids, rate=100, max_per_record=max_per_record,
                                              max_records=len(tr_ids), seed=seed)
    Mseg, Sigseg = {}, {}
    for s in SEGMENTS:
        Xtr = np.asarray(tr_seg[s][0], float)
        Xtr = Xtr[np.all(np.isfinite(Xtr), axis=1) & np.all(np.abs(Xtr) <= 10.0, axis=1)]
        if Xtr.shape[0] < 200:
            continue
        M, mu, _ = fit_dipolar_subspace(Xtr, rank=3)
        Mseg[s] = M
        Sigseg[s] = dipole_coord_cov(M, mu, Xtr)
    segs = list(Mseg)
    print(f"[active] fitted M_s for {segs} (n_train={len(tr_ids)})", flush=True)

    def J(obs):
        """Total certified squared ambiguity (mV^2) over all 12 target leads and all segments."""
        obs_leads = list(obs)
        tot = 0.0
        for s in segs:
            M, Sig = Mseg[s], Sigseg[s]
            if obs_leads:
                a2 = np.square(expected_ambiguity_per_lead(M, obs_leads, Sig))
            else:                                    # nothing observed -> full prior dipole spread
                a2 = np.clip(np.diag(M @ Sig @ M.T), 0.0, None)
            tot += float(np.sum(a2))
        return tot

    cand = list(CANDIDATES)
    K = len(cand)

    # ---- greedy forward selection ----
    greedy_curve, greedy_sets, chosen = [], [], []
    remaining = set(cand)
    for k in range(1, K + 1):
        best_l, best_J = None, np.inf
        for l in sorted(remaining):
            Jv = J(chosen + [l])
            if Jv < best_J:
                best_J, best_l = Jv, l
        chosen = chosen + [best_l]; remaining.discard(best_l)
        greedy_curve.append(round(best_J, 5)); greedy_sets.append(list(chosen))

    # ---- exhaustive optimum (feasible budgets) + random baseline, per budget ----
    per_budget = {}
    for k in range(1, K + 1):
        combos = list(itertools.combinations(cand, k))
        exhaustive = None
        if len(combos) <= 220:                       # C(8,4)=70, C(8,5)=56, ... all feasible here
            Js = [(J(list(c)), c) for c in combos]
            bestJ, bestC = min(Js, key=lambda t: t[0])
            exhaustive = {"J": round(bestJ, 5), "set": list(bestC), "n_subsets": len(combos)}
        # random baseline: mean J over random k-subsets
        idx = rng.integers(0, len(combos), min(n_random, len(combos)))
        rand_Js = [J(list(combos[i])) for i in idx]
        greedy_k = greedy_curve[k - 1]
        opt_gap = (round(greedy_k - exhaustive["J"], 6) if exhaustive else None)
        per_budget[k] = {
            "greedy_J": greedy_k, "greedy_set": greedy_sets[k - 1],
            "exhaustive": exhaustive,
            "random_J_mean": round(float(np.mean(rand_Js)), 5),
            "random_J_std": round(float(np.std(rand_Js)), 5),
            "greedy_minus_optimal": opt_gap,
            "greedy_is_optimal": (bool(abs(opt_gap) < 1e-6) if opt_gap is not None else None),
        }

    # ---- empirical diminishing-returns (submodularity) test of the ambiguity REDUCTION ----
    # f(S) = J(emptyset) - J(S) is the ambiguity removed. Submodular iff for A subset B, e not in B:
    #   f(A+e)-f(A) >= f(B+e)-f(B)  <=>  J(A)-J(A+e) >= J(B)-J(B+e).
    J0 = J([])
    n_ok = n_tot = 0
    worst = 0.0
    for _ in range(n_random):
        perm = list(rng.permutation(cand))
        a = rng.integers(0, K - 1); b = rng.integers(a, K)     # |A| <= |B|
        A = perm[:a]; B = perm[:b]
        pool = [x for x in cand if x not in B]
        if not pool:
            continue
        e = pool[rng.integers(0, len(pool))]
        margA = J(A) - J(A + [e])
        margB = J(B) - J(B + [e])
        n_tot += 1
        if margA >= margB - 1e-9:
            n_ok += 1
        else:
            worst = max(worst, margB - margA)
    submod = {"fraction_satisfied": round(n_ok / max(n_tot, 1), 4), "n_tested": n_tot,
              "worst_violation_mV2": round(float(worst), 6)}

    # headline: greedy vs optimal (max gap over feasible budgets) + greedy vs random advantage
    feasible = [per_budget[k] for k in per_budget if per_budget[k]["exhaustive"]]
    max_gap = max((abs(b["greedy_minus_optimal"]) for b in feasible), default=None)
    # advantage at the tightest interesting budget (k=3: a 3-electrode wearable)
    k3 = per_budget.get(3, {})
    summary = {
        "objective": "min total certified squared ambiguity over 12 leads x 3 segments (mV^2)",
        "candidates": list(CANDIDATES), "budgets": list(per_budget),
        "greedy_matches_optimal_all_feasible": bool(max_gap is not None and max_gap < 1e-6),
        "greedy_max_gap_to_optimal_mV2": (round(max_gap, 6) if max_gap is not None else None),
        "empirical_submodular_fraction": submod["fraction_satisfied"],
        "k3_greedy_set": k3.get("greedy_set"), "k3_greedy_J": k3.get("greedy_J"),
        "k3_random_J_mean": k3.get("random_J_mean"),
        "k3_greedy_vs_random_ratio": (round(k3.get("greedy_J", 0) / k3.get("random_J_mean", 1), 3)
                                      if k3 else None),
        "J_empty": round(J0, 5),
    }

    out = {"per_budget": per_budget, "submodularity": submod, "summary": summary,
           "segments": segs,
           "lineage": lineage.make(db, seed=seed, targets=list(LEADS), normalization=NORMALIZATION,
                                   train_ids=tr_ids,
                                   extra={"objective": "total_certified_squared_ambiguity_mV2",
                                          "candidates": list(CANDIDATES)})}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "active_selection.json").write_text(json.dumps(out, indent=2))
    print("[json] results/active_selection.json", flush=True)
    print(f"[active] greedy==optimal (all feasible k)={summary['greedy_matches_optimal_all_feasible']} "
          f"max_gap={summary['greedy_max_gap_to_optimal_mV2']} "
          f"submod_frac={submod['fraction_satisfied']}", flush=True)
    print(f"[active] k=3: greedy {summary['k3_greedy_set']} J={summary['k3_greedy_J']} "
          f"vs random {summary['k3_random_J_mean']} (ratio {summary['k3_greedy_vs_random_ratio']})", flush=True)
    for k in per_budget:
        b = per_budget[k]
        print(f"  k={k}: greedy={b['greedy_J']} opt={b['exhaustive']['J'] if b['exhaustive'] else '-'} "
              f"rand={b['random_J_mean']} set={b['greedy_set']}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    args = ap.parse_args()
    run(n_train=args.n_train)
