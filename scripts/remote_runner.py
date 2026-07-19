"""Launch or reconnect to a durable DAG job with pinned key-only SSH."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.remote import run_remote  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--action",
        choices=("launch", "status", "attach", "recover-audit"),
        default="launch",
        help=(
            "launch returns immediately; status/attach reconnect; recover-audit "
            "authenticates a stale reservation without changing it"
        ),
    )
    ap.add_argument("--host", default=os.environ.get("REMOTE_HOST"), required=False)
    ap.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_PORT", "22")))
    ap.add_argument("--user", default=os.environ.get("REMOTE_USER", "root"))
    ap.add_argument("--repo", required=True, help="existing clean remote repository")
    ap.add_argument("--run-root", required=True, help="absolute remote isolated-run root")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--profile", choices=("icassp", "extended", "legacy"))
    ap.add_argument("--resource", choices=("cpu", "gpu", "paper"))
    ap.add_argument(
        "--resume",
        action="store_true",
        help="authenticate and continue the existing remote run-id",
    )
    ap.add_argument("--environment-lock", choices=("cpu", "gpu"))
    ap.add_argument(
        "--remote-python",
        required=True,
        help="absolute POSIX path to the already verified remote venv interpreter",
    )
    ap.add_argument("--known-hosts", default=os.environ.get("REMOTE_KNOWN_HOSTS"), required=False)
    ap.add_argument("--key", default=os.environ.get("REMOTE_KEY_PATH"), required=False)
    ap.add_argument("--tail-bytes", type=int, default=65_536)
    args = ap.parse_args(argv)
    if not args.host:
        ap.error("--host or REMOTE_HOST is required")
    if not args.known_hosts:
        ap.error("--known-hosts or REMOTE_KNOWN_HOSTS is required")
    if not args.key:
        ap.error("--key or REMOTE_KEY_PATH is required")
    if args.action == "launch" and (args.profile is None or args.environment_lock is None):
        ap.error("--profile and --environment-lock are required for --action launch")
    if args.action != "launch" and args.resume:
        ap.error("--resume is accepted only for --action launch")
    result = run_remote(
        host=args.host,
        port=args.port,
        username=args.user,
        repo=args.repo,
        run_root=args.run_root,
        run_id=args.run_id,
        profile=args.profile,
        environment_lock=args.environment_lock,
        remote_python=args.remote_python,
        resource=args.resource,
        known_hosts=args.known_hosts,
        key_path=args.key,
        resume=args.resume,
        action=args.action,
        tail_bytes=args.tail_bytes,
    )
    print(
        json.dumps(
            {
                "action": result.action,
                "exit_code": result.exit_code,
                "run_dir": result.run_dir,
                "state": (
                    result.status.get("status", {}).get("state")
                    if result.action == "attach"
                    else result.status.get("status")
                    if result.action == "recover-audit"
                    else result.status.get("state")
                ),
            },
            sort_keys=True,
        )
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
