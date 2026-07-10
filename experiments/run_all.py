"""Reproduce every figure and JSON result in one command.

    python experiments/run_all.py

Assumes PTB-XL has been downloaded (scripts/download_data.py). The synthetic
validation runs in seconds; the PTB-XL experiments are dominated by NeuroKit2
delineation (a few minutes each on CPU).
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("scripts/precheck_dipolarity.py", "risk-2 dipolarity gate"),
    ("experiments/synthetic_dipole_injection.py", "synthetic theorem validation"),
    ("experiments/ptbxl_reduced_lead.py", "PTB-XL hallucination quantification"),
    ("experiments/neural_baselines.py", "trained deep reconstructors (MSE vs adversarial)"),
    ("experiments/ptbxl_stemi_safety.py", "STEMI safety case"),
    ("experiments/cross_device.py", "cross-device coverage"),
]


def main() -> int:
    for rel, desc in STEPS:
        print(f"\n{'='*70}\n[run_all] {desc}\n         {rel}\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            runpy.run_path(str(ROOT / rel), run_name="__main__")
        except SystemExit:
            pass
        print(f"[run_all] done in {time.time()-t0:.0f}s", flush=True)
    print("\n[run_all] all results in results/. Now: pytest -q; then build paper/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
