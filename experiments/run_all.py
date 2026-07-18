"""Reproduce every result JSON and figure of the CURRENT paper in one command.

    python experiments/run_all.py           # CPU pipeline (map, calibration, baselines, safety)
    python experiments/run_all.py --gpu     # also run the GPU/server steps (neural + fair table)

Assumes PTB-XL is downloaded (scripts/download_data.py). Synthetic validation runs in
seconds; the PTB-XL experiments are dominated by NeuroKit2 delineation (a few minutes each
on CPU). The neural baseline needs a GPU and is intended for the server.

This orchestrates the target-specific recoverability-map pipeline. The earlier
"fabrication/hallucination" scripts were removed in the pre-submission rebuild (see README
Sec. "Honest history"); do not resurrect them here.
"""
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path, description) in dependency order. CPU unless marked.
CPU_STEPS = [
    ("scripts/precheck_dipolarity.py", "dipolarity gate (per-segment low-rank check)"),
    ("experiments/synthetic_dipole_injection.py", "synthetic theorem validation (strict limit)"),
    ("experiments/recoverability_maps.py", "per-lead eta/kappa map + rcond sweep + bootstrap CI"),
    ("experiments/tier2_conformal.py", "calibrated intervals (CQR, strict fold discipline)"),
    ("experiments/baselines_physics.py", "classical baselines + physics-vs-PCA subspace angles"),
    ("experiments/st_safety.py", "continuous ST-threshold-event safety across reconstructors"),
    ("experiments/lead_weighting.py", "8-independent-lead vs 12-lead fit sensitivity"),
    ("experiments/certificate_validation.py", "per-lead certificate validation vs measured dipolar error"),
    ("experiments/fabrication_audit.py", "null-space dipolar fabrication audit (linear reconstructors)"),
    ("experiments/transfer_bound.py", "cross-cohort transfer bound (sinTheta) from committed cross_dataset.json"),
    ("experiments/delineator_robustness.py", "delineator x rate robustness from committed st_safety_* JSONs"),
    ("experiments/maps_figure.py", "recoverability-map figure"),
    ("experiments/ablation_figures.py", "rank / rho / subspace-angle ablation figures"),
    ("experiments/certificate_validation_figure.py", "certificate-validation figure"),
    ("experiments/fabrication_figure.py", "fabrication-audit figure"),
]
GPU_STEPS = [
    ("experiments/neural_baseline.py", "representative neural baseline (arbitrary-mask 1-D U-Net, 3 seeds) [GPU]"),
    ("experiments/fair_baselines.py", "fair per-timepoint baselines + paired CIs (merges U-Net) [server]"),
    ("experiments/fabrication_audit_diffusion.py", "DDPM null-space fabrication phi vs guidance [GPU]"),
    ("experiments/certificate_floor_diffusion.py", "DDPM per-lead error vs certified floor a_l (learned-model binding) [GPU]"),
]


def _run(rel, desc):
    print(f"\n{'='*70}\n[run_all] {desc}\n         {rel}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        runpy.run_path(str(ROOT / rel), run_name="__main__")
    except SystemExit as e:                       # fail closed: only a clean exit(0) is OK
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code != 0:
            raise RuntimeError(f"[run_all] step FAILED (exit {code}): {rel}") from e
    # any other exception propagates and aborts the pipeline
    print(f"[run_all] done in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also run GPU/server steps")
    args = ap.parse_args()
    for rel, desc in CPU_STEPS:
        _run(rel, desc)
    if args.gpu:
        for rel, desc in GPU_STEPS:
            _run(rel, desc)
        _run("paper/emit_baseline_table.py", "regenerate paper baseline table + macros from JSON")
    print("\n[run_all] results in results/. Next: pytest -q; "
          "python paper/emit_baseline_table.py; then build paper/main_v2.tex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
