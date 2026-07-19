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
from ecgcert.arc_control import (  # noqa: E402
    ArcControlValidationError,
    ORDERED_STAGES,
    validate_arc_control_chain,
)
from ecgcert.execution import ExperimentManifest, ResultEnvelope  # noqa: E402
from ecgcert.execution.environment import (  # noqa: E402
    EnvironmentLockError,
    lock_relative_path,
    require_locked_environment,
)
from ecgcert.execution.late_inputs import (  # noqa: E402
    LateControlInputError,
    validate_late_control_snapshot,
)
from ecgcert.execution.runner import (  # noqa: E402
    ExecutionError,
    collect_checkpoint_hashes,
    committed_immutable_sha256,
    declared_path_hashes,
    expected_mutable_roots,
    immutable_workspace_sha256,
    MAX_ARTIFACT_BYTES,
    MAX_CPU_CORE_HOURS,
    MAX_GPU_HOURS,
    RESERVED_GPU_HOURS,
    RUN_STATUS_SCHEMA_VERSION,
    validate_attempt_history,
)
from ecgcert.paper_evidence import FIGURE_ARTIFACTS  # noqa: E402
from ecgcert.stage_gates import validate_reviewed_gate  # noqa: E402


def _run(
    profile: str,
    run_root: Path,
    run_id: str,
    environment_lock: str,
    *,
    resume: bool = False,
) -> Path:
    command = [
        sys.executable,
        "scripts/dag_runner.py",
        "--profile",
        profile,
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--environment-lock",
        environment_lock,
    ]
    if resume:
        command.append("--resume")
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
    if status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION:
        raise SystemExit("run status schema is not the release lineage schema")
    lock_record = status.get("environment_lock")
    if not isinstance(lock_record, dict):
        raise SystemExit("run status has no verified run-level environment lock")
    lock_name = lock_record.get("lock_name")
    try:
        lock_relative = lock_relative_path(str(lock_name)).as_posix()
        active_environment = require_locked_environment(
            repo=ROOT,
            lock_name=str(lock_name),
        )
    except (EnvironmentLockError, OSError, ValueError) as exc:
        raise SystemExit(f"active release environment does not satisfy the run lock: {exc}") from exc
    expected_python = str(Path(sys.executable).resolve())
    if status.get("python_executable") != expected_python:
        raise SystemExit("run status interpreter does not match the release interpreter")
    active_environment_sha256 = lineage.environment_sha256()
    if status.get("environment_sha256") != active_environment_sha256:
        raise SystemExit("run status environment fingerprint does not match the release environment")
    if (
        lock_record.get("lock_path") != lock_relative
        or lock_record.get("lock_sha256") != active_environment.lock_sha256
        or lock_record.get("python_executable") != expected_python
        or lock_record.get("ok") is not True
        or lock_record.get("mismatches") != []
    ):
        raise SystemExit("run status environment-lock record is inconsistent")
    budget = status.get("budget")
    if not isinstance(budget, dict):
        raise SystemExit("run status has no global budget record")
    limits = budget.get("limits")
    expected_limits = {
        "cpu_core_hours": MAX_CPU_CORE_HOURS,
        "gpu_hours": MAX_GPU_HOURS,
        "artifact_bytes": MAX_ARTIFACT_BYTES,
        "reserved_gpu_hours_for_rerun": RESERVED_GPU_HOURS,
    }
    if limits != expected_limits:
        raise SystemExit("run status budget limits do not match the frozen protocol")
    reserved_gpu_hours = RESERVED_GPU_HOURS
    frozen_limits = {
        key: limits.get(key)
        for key in ("cpu_core_hours", "gpu_hours", "artifact_bytes")
    }
    try:
        attempt_history = validate_attempt_history(
            run_dir=run_dir,
            status=status,
            limits=frozen_limits,
            reserved_gpu_hours=reserved_gpu_hours,
            selected_node_ids=[node.id for node in selected],
            expected_control_root=run_dir.parent,
        )
    except (ExecutionError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid global budget attempt history: {exc}") from exc
    if budget.get("used") != attempt_history["used"]:
        raise SystemExit("status budget usage does not match authenticated attempts")
    if attempt_history["completed_node_ids"] != [node.id for node in selected]:
        raise SystemExit("attempt history does not complete the selected DAG")
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
    run_identity = {
        "run_id": run_id,
        "control_root": str(run_dir.parent.resolve()),
        "profile": profile,
        "resource": status.get("resource"),
        "environment_lock": lock_record,
        "environment_sha256": active_environment_sha256,
        "python_executable": expected_python,
        "manifest_sha256": manifest.sha256(),
        "commit": commit,
        "source_snapshot_sha256": source_sha256,
        "mutable_workspace_roots": list(mutable_roots),
    }
    if status.get("run_identity_sha256") != lineage.canonical_sha256(run_identity):
        raise SystemExit("run status identity hash is invalid")

    control_input_root = run_dir / "control-inputs"
    expected_control_snapshots = {
        node.id for node in selected if node.late_control_inputs
    }
    if control_input_root.exists():
        if control_input_root.is_symlink() or not control_input_root.is_dir():
            raise SystemExit("late-control snapshot root is not a regular directory")
        found_control_snapshots = {
            path.name for path in control_input_root.iterdir()
        }
    else:
        found_control_snapshots = set()
    if found_control_snapshots != expected_control_snapshots:
        raise SystemExit(
            "late-control snapshot inventory does not match the selected DAG"
        )

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
        try:
            late_binding = validate_late_control_snapshot(
                snapshot_root=control_input_root / node.id,
                expected_run_id=str(run_id),
                expected_node_id=node.id,
                expected_inputs=node.late_control_inputs,
            )
        except (LateControlInputError, OSError, ValueError) as exc:
            raise SystemExit(
                f"{node.id}: late-control snapshot is invalid: {exc}"
            ) from exc
        expected_config_sha256 = (
            node.config_sha256(
                late_control_inputs_sha256=late_binding.inputs_sha256
            )
            if node.late_control_inputs
            else node.config_sha256()
        )
        if envelope.config_sha256 != expected_config_sha256:
            raise SystemExit(f"{node.id}: envelope config hash does not match the manifest")
        if (
            envelope.late_control_inputs_sha256 != late_binding.inputs_sha256
            or envelope.late_control_snapshot_sha256
            != late_binding.snapshot_sha256
        ):
            raise SystemExit(
                f"{node.id}: envelope late-control binding does not match the snapshot"
            )
        if envelope.source_sha256 != source_sha256:
            raise SystemExit(f"{node.id}: envelope source snapshot hash does not match workspace")
        lock_path = workspace / lock_relative
        if not lock_path.is_file():
            raise SystemExit(f"{node.id}: required environment lock is missing")
        if envelope.environment_lock_sha256 != lineage.artifact_sha256(lock_path):
            raise SystemExit(f"{node.id}: environment lock hash does not match workspace")
        if envelope.env_sha256 != active_environment_sha256:
            raise SystemExit(f"{node.id}: environment fingerprint does not match the run")
        expected_argv = [
            expected_python if token == "{python}" else token for token in node.command
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
        effective_data_inputs = dict(data_inputs)
        effective_data_inputs.update(
            {
                f"late-control:{rel}": digest
                for rel, digest in late_binding.inputs_sha256.items()
            }
        )
        if envelope.data_sha256 != lineage.canonical_sha256(effective_data_inputs):
            raise SystemExit(f"{node.id}: input-data hash does not match the workspace")
        split_material = {
            "seed": node.seed,
            "argv": expected_argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        }
        if node.late_control_inputs:
            split_material["late_control_inputs_sha256"] = late_binding.inputs_sha256
            split_material["late_control_snapshot_sha256"] = (
                late_binding.snapshot_sha256
            )
        split_sha256 = lineage.canonical_sha256(split_material)
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


def _validate_arc_gate_chain(run_dir: Path) -> dict[str, str]:
    """Read and independently replay all four formal ARC control reports."""

    arc_reports: list[dict] = []
    receipt_sha256: dict[str, str] = {}
    for stage in ORDERED_STAGES:
        report_path = (
            run_dir
            / "workspace"
            / "artifacts"
            / "control"
            / "arc"
            / f"stage{stage}"
            / "report.v1.json"
        )
        if not report_path.is_file():
            raise SystemExit(f"formal ARC Stage {stage} report is missing")
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"cannot read formal ARC Stage {stage} report: {error}"
            ) from error
        if not isinstance(value, dict):
            raise SystemExit(f"formal ARC Stage {stage} report must be a JSON object")
        arc_reports.append(value)
        receipt_sha256[str(stage)] = lineage.artifact_sha256(report_path)
    try:
        validate_arc_control_chain(arc_reports)
    except ArcControlValidationError as error:
        raise SystemExit(f"formal ARC four-gate chain failed: {error}") from error
    return receipt_sha256


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def _validate_submission_build(
    *,
    workspace: Path,
    arc_receipts_sha256: dict[str, str],
) -> None:
    submission = workspace / "artifacts" / "paper" / "submission"
    report_path = submission / "build_report.v3.json"
    report = _load_json_object(report_path, label="submission build report")
    required_report_state = {
        "schema_version": "submission-build-v3",
        "status": "complete",
        "submission_ready": True,
        "pages": 5,
        "technical_content_end_page": 4,
        "page_five_content_validated": True,
        "overfull_boxes": 0,
    }
    failed_state = [
        key for key, expected in required_report_state.items()
        if report.get(key) != expected
    ]
    if failed_state:
        raise SystemExit(
            "final submission build state is invalid: " + ", ".join(failed_state)
        )

    sidecar = submission / "build_report.v3.sha256"
    try:
        sidecar_text = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit(f"cannot read submission build checksum: {error}") from error
    expected_sidecar = f"{lineage.artifact_sha256(report_path)}  build_report.v3.json"
    if sidecar_text != expected_sidecar:
        raise SystemExit("submission build-report checksum is invalid")

    pdf = submission / "main_v2.pdf"
    if not pdf.is_file() or report.get("pdf_sha256") != lineage.artifact_sha256(pdf):
        raise SystemExit("submission PDF does not match the build report")
    if report.get("arc_formal_receipts_sha256") != arc_receipts_sha256:
        raise SystemExit("submission build is not bound to the formal ARC four-gate chain")

    stage20_path = (
        workspace / "artifacts" / "control" / "stage20_review" / "decision.v3.json"
    )
    stage20 = _load_json_object(stage20_path, label="reviewed Stage 20 artifact")
    try:
        validate_reviewed_gate(stage20, require_proceed=True)
    except ValueError as error:
        raise SystemExit(f"Stage 20 quality gate failed: {error}") from error
    evidence = stage20.get("evidence")
    if not isinstance(evidence, dict):
        raise SystemExit("reviewed Stage 20 artifact has no evidence object")
    scientific = report.get("reviewed_scientific_input_sha256")
    if (
        not isinstance(scientific, dict)
        or scientific != evidence.get("reviewed_scientific_input_sha256")
    ):
        raise SystemExit(
            "final submission scientific inputs differ from the Stage 20 reviewed draft"
        )
    scientific_bundle_sha256 = lineage.canonical_sha256(scientific)
    if (
        report.get("reviewed_scientific_input_bundle_sha256")
        != scientific_bundle_sha256
        or evidence.get("reviewed_scientific_input_bundle_sha256")
        != scientific_bundle_sha256
    ):
        raise SystemExit("reviewed scientific-input bundle hash is invalid")

    stage20_inputs = evidence.get("input_sha256")
    if not isinstance(stage20_inputs, dict):
        raise SystemExit("reviewed Stage 20 artifact has no input-hash inventory")
    evidence_bindings = {
        "claims_sha256": stage20_inputs.get("claim_sync"),
        "claim_macros_sha256": stage20_inputs.get("claim_macros"),
        "verified_registry_sha256": stage20_inputs.get("verified_registry"),
        "figures_summary_sha256": stage20_inputs.get("figures_summary"),
    }
    for key, expected in evidence_bindings.items():
        if not isinstance(expected, str) or report.get(key) != expected:
            raise SystemExit(f"final submission {key} differs from Stage 20 evidence")
    stage20_figures = {
        name: stage20_inputs.get(f"figure_artifact:{name}")
        for name in FIGURE_ARTIFACTS
    }
    if report.get("figure_artifacts_sha256") != stage20_figures:
        raise SystemExit("final submission figure artifacts differ from Stage 20 evidence")

    compiled = report.get("compiled_input_sha256")
    expected_compiled_keys = {
        *(f"source/{name}" for name in (
            "main_v2.tex", "compliance.tex", "refs.bib", "spconf.sty", "IEEEbib.bst"
        )),
        "auto/robust_map_placeholders.tex",
        "figures_v3/summary.v3.json",
        *(f"figures_v3/{name}" for name in FIGURE_ARTIFACTS),
        "provenance/author_declaration.v1.json",
        "provenance/tool_provenance.v1.json",
        "provenance/venue_policy.v1.json",
        "provenance/author_public_key",
        "provenance/author_kit.source",
        "auto/author_declaration.tex",
        "auto/tool_provenance.tex",
        "auto/venue_policy.tex",
        "author_kit/official-template.tex",
        *(f"provenance/arc-stage{stage}-report.v1.json" for stage in ORDERED_STAGES),
    }
    if not isinstance(compiled, dict) or set(compiled) != expected_compiled_keys:
        raise SystemExit("submission compiled-input inventory is incomplete or unexpected")
    build_root = submission / "build"
    for name, expected in compiled.items():
        relative = name.removeprefix("source/") if name.startswith("source/") else name
        source = build_root.joinpath(*relative.split("/"))
        if not source.is_file() or lineage.artifact_sha256(source) != expected:
            raise SystemExit(f"compiled submission input hash mismatch: {name}")
    compiled_scientific = {
        "paper/main_v2.tex": compiled["source/main_v2.tex"],
        "paper/compliance.tex": compiled["source/compliance.tex"],
        "paper/refs.bib": compiled["source/refs.bib"],
        "claims/claims.v3.json": lineage.artifact_sha256(
            workspace / "artifacts" / "paper" / "claims" / "claims.v3.json"
        ),
        "claims/robust_map_placeholders.tex": compiled[
            "auto/robust_map_placeholders.tex"
        ],
        "claims/verified_registry.v1.json": lineage.artifact_sha256(
            workspace
            / "artifacts"
            / "paper"
            / "claims"
            / "verified_registry.v1.json"
        ),
        "figures/summary.v3.json": compiled["figures_v3/summary.v3.json"],
        **{
            f"figures/{name}": compiled[f"figures_v3/{name}"]
            for name in FIGURE_ARTIFACTS
        },
    }
    if compiled_scientific != scientific:
        raise SystemExit(
            "compiled submission scientific inputs differ from the reviewed binding"
        )

    policy_sources = {
        "author_declaration": workspace
        / "artifacts" / "gates" / "paper" / "author_declaration.v1.json",
        "tool_provenance": workspace
        / "artifacts" / "gates" / "paper" / "tool_provenance.v1.json",
        "venue_policy": workspace
        / "artifacts" / "gates" / "paper" / "venue_policy.v1.json",
        "author_public_key": workspace
        / "artifacts" / "gates" / "paper" / "author_ed25519.pub",
        "author_kit": workspace
        / "artifacts" / "gates" / "paper" / "icassp2027-author-kit.zip",
    }
    actual_policy = {
        name: lineage.artifact_sha256(path)
        for name, path in policy_sources.items()
        if path.is_file()
    }
    if actual_policy != report.get("paper_policy_input_sha256"):
        raise SystemExit("signed paper-policy inputs do not match the submission report")


def _validate_run(run_dir: Path, profile: str) -> None:
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

    arc_receipts_sha256 = _validate_arc_gate_chain(run_dir)
    _validate_submission_build(
        workspace=run_dir / "workspace",
        arc_receipts_sha256=arc_receipts_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("icassp", "extended"), default="icassp")
    parser.add_argument("--run-root", type=Path, default=ROOT.parent / "ecg-recoverability-runs")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="authenticate and continue an existing release run",
    )
    parser.add_argument(
        "--environment-lock",
        choices=("cpu", "gpu"),
        default="gpu",
        help="single lock satisfied by the interpreter executing the complete mixed-resource DAG",
    )
    arguments = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"release-{arguments.profile}-{stamp}"
    if arguments.resume and arguments.run_id is None:
        parser.error("--resume requires an explicit existing --run-id")
    run_root = arguments.run_root.resolve()
    run_dir = _run(
        arguments.profile,
        run_root,
        run_id,
        arguments.environment_lock,
        resume=arguments.resume,
    )
    _validate_run(run_dir, arguments.profile)
    print(f"validated isolated release: {run_dir}")


if __name__ == "__main__":
    main()
