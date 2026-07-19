"""Safely expose persistent official sources at the DAG's ignored input path."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.upstream_staging import (  # noqa: E402
    UpstreamLinkError,
    ensure_server_upstream_link,
    render_upstream_link_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one ignored repository symlink to validated persistent official sources."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="absolute Git worktree root (defaults to this script's repository)",
    )
    parser.add_argument(
        "--tools-root",
        type=Path,
        required=True,
        help="absolute persistent tools root containing upstreams/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = ensure_server_upstream_link(
            repo=arguments.repo,
            tools_root=arguments.tools_root,
        )
    except UpstreamLinkError as exc:
        sys.stderr.write(f"upstream link error: {exc}\n")
        return 2
    sys.stdout.write(render_upstream_link_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
