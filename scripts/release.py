"""Fail-closed ICASSP release entrypoint for the single experiment DAG.

The command always starts from a clean committed snapshot, rebuilds into a new
run directory, validates every provenance envelope, and accepts either the
positive paper after a reviewed Stage-15 ``PROCEED`` decision or the transparent
negative-result paper after a reviewed ``PIVOT`` decision. Historical experiments
require the explicit ``legacy`` profile and can never satisfy this release command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcert import lineage  # noqa: E402
from ecgcert.execution import ExperimentManifest, ResultEnvelope  # noqa: E402
from ecgcert.execution.budget import (  # noqa: E402
    BudgetError,
    validate_settlement_snapshot,
)
from ecgcert.execution.runner import (  # noqa: E402
    collect_checkpoint_hashes,
    committed_immutable_sha256,
    declared_path_hashes,
    expected_mutable_roots,
    immutable_workspace_sha256,
)
from ecgcert.stage_gates import validate_reviewed_gate  # noqa: E402


def _run(profile: str, run_root: Path, run_id: str) -> Path:
    command = [
        sys.executable,
        "scripts/dag_runner.py",
        "--profile",
        profile,
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f"release DAG failed with exit code {completed.returncode}")
    return (run_root / run_id).resolve()


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode or len(commit) != 40:
        raise SystemExit("cannot verify the release source commit")
    return commit


def _validate_envelope_lineage(
    *,
    run_dir: Path,
    manifest: ExperimentManifest,
    profile: str,
    status: dict,
) -> None:
    """Recompute DAG lineage from the finished workspace, never from cached hashes."""
    selected = manifest.select(profile)
    expected = {node.id for node in selected}
    envelope_dir = run_dir / "envelopes"
    workspace = run_dir / "workspace"
    if not workspace.is_dir():
        raise SystemExit("release workspace is missing")
    found = {path.stem for path in envelope_dir.glob("*.json")}
    if found != expected:
        raise SystemExit(
            f"envelope set mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    status_nodes = status.get("nodes")
    if not isinstance(status_nodes, dict) or set(status_nodes) != expected:
        raise SystemExit("status node set does not match the selected manifest profile")
    failed_status = [
        node_id for node_id, value in status_nodes.items()
        if not isinstance(value, dict)
        or value.get("state") != "succeeded"
        or value.get("exit_code") != 0
    ]
    if failed_status:
        raise SystemExit(f"node status is not succeeded/0: {sorted(failed_status)}")

    run_id = status.get("run_id")
    commit = status.get("commit")
    if run_id != run_dir.name:
        raise SystemExit("status run_id does not match the run directory")
    if commit != _source_commit():
        raise SystemExit("release commit does not match the current source commit")
    if status.get("profile") != profile:
        raise SystemExit("status profile does not match the requested release profile")
    if status.get("manifest_sha256") != manifest.sha256():
        raise SystemExit("status manifest hash does not match the current manifest")
    if status.get("schema_version") != 2:
        raise SystemExit("run status schema is not the release lineage schema")
    budget = status.get("budget")
    if not isinstance(budget, dict):
        raise SystemExit("run status has no global budget record")
    limits = budget.get("limits")
    if not isinstance(limits, dict):
        raise SystemExit("run status budget limits are missing")
    reserved_gpu_hours = limits.get("reserved_gpu_hours_for_rerun")
    frozen_limits = {
        key: limits.get(key)
        for key in ("cpu_core_hours", "gpu_hours", "artifact_bytes")
    }
    settlement_path = run_dir / "budget-settlement.v1.json"
    try:
        settlement_value = json.loads(settlement_path.read_text(encoding="utf-8"))
        settlement = validate_settlement_snapshot(
            settlement_value,
            expected_run_id=run_id,
            limits=frozen_limits,
            reserved_gpu_hours=reserved_gpu_hours,
        )
    except (OSError, json.JSONDecodeError, BudgetError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid global budget settlement: {exc}") from exc
    if budget.get("global_reservation_sha256") != settlement["reservation_sha256"]:
        raise SystemExit("status budget reservation hash mismatch")
    if budget.get("global_settlement_sha256") != settlement["settlement_sha256"]:
        raise SystemExit("status budget settlement hash mismatch")
    if budget.get("settlement_artifact_sha256") != lineage.artifact_sha256(settlement_path):
        raise SystemExit("status budget settlement artifact hash mismatch")
    if budget.get("used") != settlement["used"]:
        raise SystemExit("status budget usage does not match the global settlement")
    try:
        mutable_roots = expected_mutable_roots(ROOT, commit, selected)
        source_sha256 = immutable_workspace_sha256(workspace, mutable_roots)
        committed_sha256 = committed_immutable_sha256(ROOT, commit, mutable_roots)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot authenticate immutable source snapshot: {exc}") from exc
    if status.get("mutable_workspace_roots") != list(mutable_roots):
        raise SystemExit("status mutable-root inventory does not match the manifest")
    if source_sha256 != committed_sha256:
        raise SystemExit("workspace immutable source snapshot does not match the commit")
    if status.get("source_snapshot_sha256") != source_sha256:
        raise SystemExit("workspace immutable source snapshot does not match status")

    for node in selected:
        envelope_path = envelope_dir / f"{node.id}.json"
        try:
            envelope = ResultEnvelope.read(envelope_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{node.id}: invalid result envelope: {exc}") from exc
        if (
            envelope.run_id != run_id
            or envelope.node_id != node.id
            or envelope.commit != commit
        ):
            raise SystemExit(f"{node.id}: envelope run/node/commit identity mismatch")
        if envelope.config_sha256 != node.config_sha256():
            raise SystemExit(f"{node.id}: envelope config hash does not match the manifest")
        if envelope.source_sha256 != source_sha256:
            raise SystemExit(f"{node.id}: envelope source snapshot hash does not match workspace")
        lock_relative = (
            "environments/gpu.lock.txt"
            if node.resource.kind == "gpu"
            else "environments/cpu.lock.txt"
        )
        lock_path = workspace / lock_relative
        if not lock_path.is_file():
            raise SystemExit(f"{node.id}: required environment lock is missing")
        if envelope.environment_lock_sha256 != lineage.artifact_sha256(lock_path):
            raise SystemExit(f"{node.id}: environment lock hash does not match workspace")
        expected_argv = [
            sys.executable if token == "{python}" else token for token in node.command
        ]
        if envelope.argv != expected_argv or envelope.seed != node.seed:
            raise SystemExit(f"{node.id}: envelope command or seed does not match the manifest")

        try:
            input_hashes = declared_path_hashes(workspace, node.inputs)
            output_hashes = declared_path_hashes(workspace, node.outputs)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{node.id}: cannot hash a declared workspace artifact: {exc}") from exc
        if set(envelope.outputs_sha256) != set(node.outputs):
            raise SystemExit(f"{node.id}: envelope output set does not match the manifest")
        if envelope.outputs_sha256 != output_hashes:
            raise SystemExit(f"{node.id}: workspace outputs do not match the envelope")

        checkpoints = collect_checkpoint_hashes(
            workspace, (*node.inputs, *node.outputs),
        )
        if envelope.checkpoint_sha256 != checkpoints:
            raise SystemExit(f"{node.id}: checkpoint hashes do not match the workspace")
        data_inputs = {
            rel: digest for rel, digest in input_hashes.items() if rel not in checkpoints
        }
        if envelope.data_sha256 != lineage.canonical_sha256(data_inputs):
            raise SystemExit(f"{node.id}: input-data hash does not match the workspace")
        split_sha256 = lineage.canonical_sha256({
            "seed": node.seed,
            "argv": expected_argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        })
        if envelope.split_sha256 != split_sha256:
            raise SystemExit(f"{node.id}: input/split hash does not match the workspace")

        expected_upstream = {
            dep: lineage.artifact_sha256(envelope_dir / f"{dep}.json")
            for dep in node.deps
        }
        if envelope.upstream_sha256 != expected_upstream:
            raise SystemExit(f"{node.id}: upstream envelope hashes do not match the DAG")


def _load_status(run_dir: Path) -> dict:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read run status: {exc}") from exc
    if not isinstance(status, dict):
        raise SystemExit("run status must be a JSON object")
    return status


def _validate_run(run_dir: Path, profile: str, *, allow_pending_gate: bool) -> None:
    status = _load_status(run_dir)
    if status.get("state") != "succeeded" or status.get("exit_code") != 0:
        raise SystemExit("run status is not succeeded/0")
    manifest = ExperimentManifest.from_path(ROOT / "scripts" / "experiment_manifest.yaml")
    _validate_envelope_lineage(
        run_dir=run_dir,
        manifest=manifest,
        profile=profile,
        status=status,
    )

    report_path = run_dir / "workspace" / "artifacts" / "paper" / "submission" / "build_report.v3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("pages") != 5 or report.get("overfull_boxes") != 0:
        raise SystemExit("paper formatting gate failed")
    if not report.get("submission_ready") and not allow_pending_gate:
        raise SystemExit(
            "paper is evidence-complete but Stage 15 is not a reviewed PROCEED/PIVOT; "
            "review the gate rather than overriding evidence"
        )
    stage20_path = (
        run_dir / "workspace" / "artifacts" / "control" / "stage20_review" /
        "decision.v3.json"
    )
    if not stage20_path.is_file():
        if allow_pending_gate:
            return
        raise SystemExit("reviewed Stage 20 artifact is missing")
    stage20 = json.loads(stage20_path.read_text(encoding="utf-8"))
    try:
        validate_reviewed_gate(stage20, require_proceed=True)
    except ValueError as error:
        if allow_pending_gate:
            return
        raise SystemExit(f"Stage 20 quality gate failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("icassp", "extended"), default="icassp")
    parser.add_argument("--run-root", type=Path, default=ROOT.parent / "ecg-recoverability-runs")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help="development only: validate a five-page draft while Stage 15 awaits review",
    )
    arguments = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"release-{arguments.profile}-{stamp}"
    run_root = arguments.run_root.resolve()
    run_dir = _run(arguments.profile, run_root, run_id)
    _validate_run(run_dir, arguments.profile, allow_pending_gate=arguments.allow_pending_gate)
    print(f"validated isolated release: {run_dir}")


if __name__ == "__main__":
    main()
