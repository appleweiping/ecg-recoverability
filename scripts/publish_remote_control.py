"""Atomically publish one declared ARC/reviewer control to an active remote run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.control_publish import (  # noqa: E402
    ControlPublicationError,
    atomic_publish_local_bytes,
    publish_remote_control,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/experiment_manifest.yaml")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--remote-workspace", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report_path = arguments.report.expanduser()
    if report_path.exists() or report_path.is_symlink():
        sys.stderr.write("control publication report already exists; overwrite is forbidden\n")
        return 2
    try:
        report = publish_remote_control(
            local_path=arguments.local,
            manifest_path=arguments.manifest,
            node_id=arguments.node_id,
            run_id=arguments.run_id,
            expected_commit=arguments.expected_commit,
            remote_workspace=arguments.remote_workspace,
            host=arguments.host,
            port=arguments.port,
            username=arguments.username,
            known_hosts=arguments.known_hosts,
            key_path=arguments.key,
        )
        rendered = (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        atomic_publish_local_bytes(
            report_path,
            rendered.encode("utf-8"),
            label="control publication report",
        )
    except (ControlPublicationError, OSError, ValueError) as error:
        sys.stderr.write(f"remote control publication failed closed: {error}\n")
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
