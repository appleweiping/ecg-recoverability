"""Atomically pull one envelope-covered gate/handoff artifact from a remote run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.control_publish import (  # noqa: E402
    ControlPublicationError,
    atomic_publish_local_bytes,
    pull_remote_run_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-artifact", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/experiment_manifest.yaml")
    parser.add_argument("--producer-node-id", required=True)
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
    try:
        report_path = arguments.report.expanduser()
        if report_path.exists() or report_path.is_symlink():
            raise ControlPublicationError(
                "artifact-pull report already exists; overwrite is forbidden"
            )
        destination = Path(os.path.abspath(os.fspath(arguments.destination.expanduser())))
        report_destination = Path(os.path.abspath(os.fspath(report_path)))
        if os.path.normcase(os.fspath(destination)) == os.path.normcase(
            os.fspath(report_destination)
        ):
            raise ControlPublicationError("artifact destination and pull report must differ")
        report = pull_remote_run_artifact(
            remote_artifact=arguments.remote_artifact,
            destination=arguments.destination,
            manifest_path=arguments.manifest,
            producer_node_id=arguments.producer_node_id,
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
            label="artifact-pull report",
        )
    except (ControlPublicationError, OSError, ValueError) as error:
        sys.stderr.write(f"remote run artifact pull failed closed: {error}\n")
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
