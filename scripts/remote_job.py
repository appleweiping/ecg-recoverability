"""Server-side durable launch, status, and log attachment for one DAG run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.remote_job import (  # noqa: E402
    RemoteJobError,
    attach_job,
    audit_recovery_job,
    job_status,
    launch_detached_job,
    supervise_attempt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--repo", type=Path, required=True)
    launch.add_argument("--run-root", type=Path, required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--profile", choices=("icassp", "extended", "legacy"), required=True)
    launch.add_argument("--resource", choices=("cpu", "gpu", "paper"))
    launch.add_argument("--environment-lock", choices=("cpu", "gpu"), required=True)
    launch.add_argument("--manifest", default="scripts/experiment_manifest.yaml")
    launch.add_argument("--resume", action="store_true")

    for name in ("status", "attach", "recover-audit"):
        control = subparsers.add_parser(name)
        control.add_argument("--repo", type=Path, required=True)
        control.add_argument("--run-root", type=Path, required=True)
        control.add_argument("--run-id", required=True)
        if name == "attach":
            control.add_argument("--tail-bytes", type=int, default=65_536)

    supervise = subparsers.add_parser("_supervise")
    supervise.add_argument("--attempt-dir", type=Path, required=True)
    return parser


def _dag_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        str(Path(sys.executable).absolute()),
        "scripts/dag_runner.py",
        "--manifest",
        arguments.manifest,
        "--profile",
        arguments.profile,
        "--run-root",
        str(arguments.run_root.resolve()),
        "--run-id",
        arguments.run_id,
        "--environment-lock",
        arguments.environment_lock,
    ]
    if arguments.resource:
        command.extend(("--resource", arguments.resource))
    if arguments.resume:
        command.append("--resume")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "_supervise":
            return supervise_attempt(arguments.attempt_dir)
        if arguments.action == "launch":
            report = launch_detached_job(
                repo=arguments.repo,
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                command=_dag_command(arguments),
                resume=arguments.resume,
                profile=arguments.profile,
                resource=arguments.resource,
                environment_lock=arguments.environment_lock,
                python_executable=Path(sys.executable),
                supervisor_entry=Path(__file__),
            )
        elif arguments.action == "status":
            report = job_status(
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                expected_repo=arguments.repo,
                expected_python=Path(sys.executable),
            )
        elif arguments.action == "recover-audit":
            report = audit_recovery_job(
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                expected_repo=arguments.repo,
                expected_python=Path(sys.executable),
            )
        else:
            report = attach_job(
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                tail_bytes=arguments.tail_bytes,
                expected_repo=arguments.repo,
                expected_python=Path(sys.executable),
            )
    except RemoteJobError as exc:
        sys.stderr.write(f"remote job control failed closed: {exc}\n")
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
