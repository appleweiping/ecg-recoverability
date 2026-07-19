"""Safely expose persistent server datasets at the DAG's ignored input paths."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.data.staging import (  # noqa: E402
    DataLinkError,
    ensure_server_data_links,
    render_link_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create exactly three ignored repository symlinks to complete persistent ECG cohorts."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="absolute git worktree root (defaults to this script's repository)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="absolute persistent directory containing ptbxl, chapman, and cpsc2018",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = ensure_server_data_links(repo=arguments.repo, data_root=arguments.data_root)
    except DataLinkError as exc:
        sys.stderr.write(f"data link error: {exc}\n")
        return 2
    sys.stdout.write(render_link_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
