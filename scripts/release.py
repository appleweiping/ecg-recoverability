"""Fail-closed release orchestrator (P0-G).

Runs the full pipeline in order; ANY nonzero step aborts with a nonzero exit. Covers the
experiments, both macro emitters, the test suite, and both PDF builds. Phases can be run
separately (data lives on the server; PDFs build anywhere with LaTeX).

    python scripts/release.py --cpu      # CPU experiments
    python scripts/release.py --gpu      # GPU experiments (needs a trained diffusion ckpt)
    python scripts/release.py --papers   # emitters + tests + both PDF builds
    python scripts/release.py --all       # everything

Every experiment writes a lineage-stamped JSON; the emitters fail closed if any required JSON
is absent; the test suite must pass with ZERO skips; both PDFs must compile.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CPU_STEPS = [
    ["scripts/precheck_dipolarity.py"],
    ["experiments/synthetic_dipole_injection.py"],
    ["experiments/recoverability_maps.py", "--n-records", "1500", "--n-boot", "200"],
    ["experiments/tier2_conformal.py", "--n-per-fold", "500"],
    ["experiments/baselines_physics.py", "--n-train", "3000", "--n-test", "1500"],
    ["experiments/st_safety.py", "--n-train", "1500", "--n-test", "1500"],
    ["experiments/lead_weighting.py", "--n-records", "1500", "--n-boot", "200"],
    ["experiments/cross_dataset.py", "--n-chapman", "350", "--n-boot", "100"],
    ["experiments/gpu_fabrication.py", "--mode", "gate", "--n-train", "2000", "--n-test", "800"],
    ["experiments/maps_figure.py"],
]
GPU_STEPS = [
    ["experiments/neural_baseline.py", "--n-train", "4000", "--n-test", "800", "--epochs", "60", "--seeds", "0", "1", "2"],
    ["experiments/fair_baselines.py", "--n-train", "4000", "--n-test", "800"],
    ["experiments/gpu_fabrication.py", "--mode", "diffusion", "--n-train", "4000", "--epochs", "70"],  # trains gpu_ddpm.pt
    ["experiments/realism_metrics.py", "--n-test", "300"],
    ["experiments/gpu_diffusion_clean.py", "--ci", "--n-seeds", "5", "--epochs", "70", "--guidances", "1.0,2.0,4.0,6.0"],
]


def _run(cmd, label):
    print(f"\n{'='*70}\n[release] {label}: {' '.join(cmd)}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u"] + cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[release] FAILED (exit {r.returncode}): {label} :: {' '.join(cmd)}")


def _build_pdf(doc):
    d = ROOT / "paper"
    for f in (f"{doc}.aux", f"{doc}.bbl", f"{doc}.blg"):
        (d / f).unlink(missing_ok=True)
    def tex(): return subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{doc}.tex"],
                                     cwd=d, capture_output=True, text=True)
    if tex().returncode != 0:
        raise SystemExit(f"[release] pdflatex FAILED: {doc} (pass 1)")
    subprocess.run(["bibtex", doc], cwd=d, capture_output=True, text=True)
    tex(); r = tex()
    if r.returncode != 0 or not (d / f"{doc}.pdf").exists():
        raise SystemExit(f"[release] pdflatex FAILED: {doc} (final)")
    print(f"[release] built {doc}.pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--papers", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="also enforce git_dirty=false / non-null checkpoint SHA / current script hash")
    a = ap.parse_args()
    if not any((a.cpu, a.gpu, a.papers, a.all)):
        ap.error("choose --cpu / --gpu / --papers / --all")

    if a.cpu or a.all:
        for cmd in CPU_STEPS:
            _run(cmd, "cpu")
    if a.gpu or a.all:
        for cmd in GPU_STEPS:
            _run(cmd, "gpu")
    if a.papers or a.all:
        _run(["paper/emit_baseline_table.py"], "emit-baseline")   # fails closed if JSON absent
        _run(["paper/emit_long_results.py"], "emit-long")
        # run the FULL release suite (ECG_RELEASE=1: strict integrity + zero skips), not the
        # reduced default. --strict additionally enforces git_dirty=false / checkpoint / script hash.
        env = {**os.environ, "ECG_RELEASE": "1"}
        if a.strict:
            env["ECG_RELEASE_STRICT"] = "1"
        r = subprocess.run([PY, "-m", "pytest", "-q", "--no-header"], cwd=ROOT, env=env)
        if r.returncode != 0:
            raise SystemExit("[release] test suite FAILED")
        _build_pdf("main_v2")
        _build_pdf("arxiv_long")
    print("\n[release] OK", flush=True)


if __name__ == "__main__":
    main()
