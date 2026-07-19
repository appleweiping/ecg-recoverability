"""Emit the strict, versioned server-readiness report before a DAG run."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.preflight import (  # noqa: E402
    PreflightConfig,
    collect_server_preflight,
    failed_preflight_report,
    write_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only, fail-closed inventory of the Linux GPU experiment server."
    )
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--expected-commit",
        help="exact 40-character frozen repository commit; omission fails closed",
    )
    parser.add_argument("--storage-root", type=Path, default=ROOT.parent)
    parser.add_argument(
        "--tools-root",
        type=Path,
        default=ROOT.parent / "ecg-recoverability-tools",
    )
    parser.add_argument("--ptbxl", type=Path, default=ROOT / "data" / "ptbxl")
    parser.add_argument(
        "--chapman", type=Path, default=ROOT / "data" / "external" / "chapman"
    )
    parser.add_argument(
        "--cpsc2018", type=Path, default=ROOT / "data" / "external" / "cpsc2018"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON artifact path; the same JSON is always written to stdout",
    )
    parser.add_argument(
        "--environment-lock",
        choices=("cpu", "gpu"),
        default="gpu",
        help="run-level lock that this exact interpreter must satisfy",
    )
    return parser


def _output_path_error(repo: Path, output: Path | None) -> str | None:
    """Reject a report path that could dirty the frozen source checkout."""

    if output is None:
        return None
    try:
        repository = repo.resolve(strict=True)
        destination = output.resolve()
    except OSError:
        return "preflight repo/output path cannot be resolved safely"
    if destination == repository or repository in destination.parents:
        return "server preflight output must be outside the frozen repository"
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_error = _output_path_error(args.repo, args.output)
    if output_error is not None:
        report = failed_preflight_report("preflight.output_scope", output_error)
        sys.stdout.write(write_report(report, None))
        return 2
    config = PreflightConfig(
        repo=args.repo,
        expected_commit=args.expected_commit,
        storage_root=args.storage_root,
        tools_root=args.tools_root,
        ptbxl_root=args.ptbxl,
        chapman_root=args.chapman,
        cpsc2018_root=args.cpsc2018,
        active_environment_lock=args.environment_lock,
    )
    try:
        report = collect_server_preflight(config)
    except Exception:
        report = failed_preflight_report(
            "preflight.collection",
            "server preflight collection failed before a complete inventory was available",
        )
    try:
        rendered = write_report(report, args.output)
    except (OSError, TypeError, ValueError):
        report = failed_preflight_report(
            "preflight.output",
            "the requested report artifact could not be written",
        )
        rendered = write_report(report, None)
    sys.stdout.write(rendered)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
