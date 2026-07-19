"""Compatibility wrapper for the single manifest-driven experiment DAG."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("icassp", "extended", "legacy"), default="icassp")
    parser.add_argument("--run-root", default=str(ROOT.parent / "ecg-recoverability-runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--environment-lock", choices=("cpu", "gpu"), required=True)
    arguments = parser.parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "dag_runner.py"),
        "--profile",
        arguments.profile,
        "--run-root",
        arguments.run_root,
        "--environment-lock",
        arguments.environment_lock,
    ]
    if arguments.run_id:
        command.extend(["--run-id", arguments.run_id])
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
