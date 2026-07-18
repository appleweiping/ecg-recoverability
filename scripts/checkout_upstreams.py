"""Materialize clean, exact-commit upstream baseline checkouts outside the repo."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from ecgcert.estimators.official import ECG_RECOVER, IMPUTE_ECG, UpstreamSpec


SPECS = {"imputeecg": IMPUTE_ECG, "ecgrecover": ECG_RECOVER}


def checkout(spec: UpstreamSpec, destination: str | Path) -> Path:
    root = Path(destination).resolve()
    target = root / f"{spec.name}-{spec.commit[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["git", "clone", "--no-checkout", spec.repository, str(target)], check=True)
    if not (target / ".git").exists():
        raise ValueError(f"existing target is not a git checkout: {target}")
    subprocess.run(["git", "-C", str(target), "fetch", "origin", spec.commit], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec.commit], check=True)
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != spec.commit:
        raise RuntimeError(f"failed to pin {spec.name}: {head}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*SPECS, "all"], default="all")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    names = SPECS if args.model == "all" else (args.model,)
    for name in names:
        target = checkout(SPECS[name], args.destination)
        print(f"[{name}] {SPECS[name].commit} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
