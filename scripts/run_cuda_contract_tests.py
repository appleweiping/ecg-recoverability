"""Run the three mandatory real-CUDA reconstruction contract tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from ecgcert import lineage
from ecgcert.estimators.official import (
    ECG_RECOVER,
    IMPUTE_ECG,
    UpstreamSpec,
    validate_pinned_checkout,
)


SCHEMA_VERSION = "cuda-contract-tests-v1"
TESTS = (
    "tests/test_cuda_tiny_e2e.py",
    "tests/test_imputeecg_cuda_tiny_e2e.py",
    "tests/test_ecgrecover_cuda_tiny_e2e.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_checkout(root: Path, spec: UpstreamSpec) -> Path:
    prefix = f"{spec.name}-{spec.commit[:12]}"
    candidates = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and path.name == prefix
    )
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {prefix} checkout under {root}")
    validate_pinned_checkout(candidates[0], spec)
    return candidates[0]


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    repo = arguments.repo.resolve(strict=True)
    upstreams = arguments.upstreams.resolve(strict=True)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    imputeecg = _resolve_checkout(upstreams, IMPUTE_ECG)
    ecgrecover = _resolve_checkout(upstreams, ECG_RECOVER)
    stdout_path = output / "pytest.stdout.log"
    stderr_path = output / "pytest.stderr.log"
    command = [sys.executable, "-m", "pytest", "-q", *TESTS]
    environment = os.environ.copy()
    environment.update(
        {
            "ECGCERT_REQUIRE_CUDA_TEST": "1",
            "ECGCERT_IMPUTEECG_SOURCE_DIR": str(imputeecg),
            "ECGCERT_ECGRECOVER_SOURCE_DIR": str(ecgrecover),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            # These three smokes intentionally use num_workers=0; prevent a
            # standalone invocation from inheriting a larger DAG worker pool.
            "ECGCERT_NUM_WORKERS": "1",
        }
    )
    started_at = _utc_now()
    started = time.monotonic()
    returncode = 1
    failure_kind = "pytest"
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=output) as base_temp:
        command.extend(("--basetemp", base_temp))
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=arguments.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                returncode = 124
                failure_kind = "timeout"
            else:
                returncode = int(completed.returncode)
                failure_kind = "none" if returncode == 0 else "pytest"
    duration = time.monotonic() - started
    # A test process must never mutate either exact upstream checkout.
    try:
        validate_pinned_checkout(imputeecg, IMPUTE_ECG)
        validate_pinned_checkout(ecgrecover, ECG_RECOVER)
    except (OSError, ValueError, subprocess.SubprocessError):
        returncode = returncode or 1
        failure_kind = "upstream_integrity"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if returncode == 0 else "failed",
        "failure_kind": failure_kind,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration, 3),
        "returncode": returncode,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "tests": list(TESTS),
        "command": command,
        "hardware": lineage.hardware_fingerprint(),
        "upstreams": {
            "imputeecg": {
                "commit": IMPUTE_ECG.commit,
                "root_tree": IMPUTE_ECG.root_tree,
            },
            "ecgrecover": {
                "commit": ECG_RECOVER.commit,
                "root_tree": ECG_RECOVER.root_tree,
            },
        },
        "artifacts": {
            "stdout": {
                "path": stdout_path.name,
                "sha256": lineage.artifact_sha256(stdout_path),
            },
            "stderr": {
                "path": stderr_path.name,
                "sha256": lineage.artifact_sha256(stderr_path),
            },
        },
    }
    _atomic_json(output / "report.v1.json", report)
    if returncode:
        raise RuntimeError(
            f"mandatory CUDA contract tests failed with exit code {returncode}"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--upstreams", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    try:
        report = run(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"CUDA contract validation failed closed: {exc}") from exc
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
