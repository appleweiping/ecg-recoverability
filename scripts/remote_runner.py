"""Launch the unified DAG runner with key-only auth and a pinned host key."""
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
    ap.add_argument("--host", default=os.environ.get("REMOTE_HOST"), required=False)
    ap.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_PORT", "22")))
    ap.add_argument("--user", default=os.environ.get("REMOTE_USER", "root"))
    ap.add_argument("--repo", required=True, help="existing clean remote repository")
    ap.add_argument("--run-root", required=True, help="absolute remote isolated-run root")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--profile", choices=("icassp", "extended", "legacy"), required=True)
    ap.add_argument("--resource", choices=("cpu", "gpu", "paper"))
    ap.add_argument("--known-hosts", default=os.environ.get("REMOTE_KNOWN_HOSTS"), required=False)
    ap.add_argument("--key", default=os.environ.get("REMOTE_KEY_PATH"), required=False)
    args = ap.parse_args(argv)
    if not args.host:
        ap.error("--host or REMOTE_HOST is required")
    if not args.known_hosts:
        ap.error("--known-hosts or REMOTE_KNOWN_HOSTS is required")
    if not args.key:
        ap.error("--key or REMOTE_KEY_PATH is required")
    result = run_remote(
        host=args.host,
        port=args.port,
        username=args.user,
        repo=args.repo,
        run_root=args.run_root,
        run_id=args.run_id,
        profile=args.profile,
        resource=args.resource,
        known_hosts=args.known_hosts,
        key_path=args.key,
    )
    print(json.dumps({
        "exit_code": result.exit_code,
        "run_dir": result.run_dir,
        "state": result.status.get("state"),
    }, sort_keys=True))
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
