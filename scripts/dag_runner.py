"""Unified local DAG runner for reproducible ECG experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution import DAGRunner, ExperimentManifest  # noqa: E402


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="scripts/experiment_manifest.yaml")
    ap.add_argument("--profile", choices=("icassp", "extended", "legacy"), required=True)
    ap.add_argument("--resource", choices=("cpu", "gpu", "paper"))
    ap.add_argument(
        "--environment-lock",
        choices=("cpu", "gpu"),
        help=(
            "run-level lock satisfied by this interpreter; required for execution "
            "because resource kinds are scheduling metadata"
        ),
    )
    ap.add_argument("--run-root", default=str(ROOT.parent / "ecg-recoverability-runs"))
    ap.add_argument("--control-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="authenticate and continue an existing logical run from its first incomplete node",
    )
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = ExperimentManifest.from_path(manifest_path)
    selected = manifest.select(args.profile, args.resource)
    if args.validate_only:
        print("\n".join(node.id for node in selected))
        return 0
    if args.environment_lock is None:
        ap.error("--environment-lock is required for execution")
    if args.resume and args.run_id is None:
        ap.error("--resume requires an explicit existing --run-id")
    runner = DAGRunner(
        repo=ROOT,
        manifest=manifest,
        profile=args.profile,
        resource=args.resource,
        environment_lock=args.environment_lock,
        run_root=ROOT / args.run_root if not Path(args.run_root).is_absolute() else args.run_root,
        control_root=args.control_root,
        run_id=args.run_id or _default_run_id(),
        resume=args.resume,
    )
    run_dir = runner.run()
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
