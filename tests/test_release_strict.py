"""Strict v3 ICASSP release audit.

Collected only when ``ECG_RELEASE_STRICT=1``.  The audit targets one isolated
authenticated DAG run named by ``ECGCERT_RELEASE_RUN_DIR``; historical
``results/*.json`` are deliberately outside the submission contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ecgcert.execution import ExperimentManifest, ResultEnvelope
from ecgcert.execution.late_inputs import validate_late_control_snapshot
from ecgcert.execution.runner import collect_checkpoint_hashes, declared_path_hashes
from scripts import release


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ExperimentManifest.from_path(ROOT / "scripts" / "experiment_manifest.yaml")
pytestmark = pytest.mark.skipif(
    os.environ.get("ECG_RELEASE_STRICT") != "1",
    reason="strict release audit requires ECG_RELEASE_STRICT=1 and an isolated run",
)


@pytest.fixture(scope="module")
def release_run() -> tuple[Path, str]:
    raw = os.environ.get("ECGCERT_RELEASE_RUN_DIR")
    assert raw, (
        "ECGCERT_RELEASE_RUN_DIR must identify the completed isolated v3 DAG run "
        "when ECG_RELEASE_STRICT=1"
    )
    run_dir = Path(raw).expanduser().resolve(strict=True)
    profile = os.environ.get("ECGCERT_RELEASE_PROFILE", "icassp")
    assert profile in {"icassp", "extended"}
    assert (run_dir / "workspace" / "artifacts").is_dir()
    return run_dir, profile


def test_strict_v3_release_entrypoint_replays_complete_lineage(release_run) -> None:
    run_dir, profile = release_run
    # This single validator replays the authenticated attempt ledger, immutable
    # source snapshot, every input/output/checkpoint hash, the four formal ARC
    # reports, reviewed Stage 20, and the five-page paper gate.
    release._validate_run(run_dir, profile)


def test_strict_profile_contains_no_legacy_nodes_or_results(release_run) -> None:
    run_dir, profile = release_run
    selected = MANIFEST.select(profile)
    assert selected
    assert all("legacy" not in node.profile for node in selected)
    assert all(not output.startswith("results/") for node in selected for output in node.outputs)
    legacy_ids = {node.id for node in MANIFEST.nodes if "legacy" in node.profile}
    assert not (legacy_ids & {node.id for node in selected})
    envelopes = {path.stem for path in (run_dir / "envelopes").glob("*.json")}
    assert envelopes == {node.id for node in selected}


def test_strict_v3_envelopes_are_complete_clean_and_current(release_run) -> None:
    run_dir, profile = release_run
    workspace = run_dir / "workspace"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    selected = MANIFEST.select(profile)
    for node in selected:
        envelope = ResultEnvelope.read(run_dir / "envelopes" / f"{node.id}.json")
        assert envelope.dirty is False
        assert envelope.commit == status["commit"]
        late_binding = validate_late_control_snapshot(
            snapshot_root=run_dir / "control-inputs" / node.id,
            expected_run_id=run_dir.name,
            expected_node_id=node.id,
            expected_inputs=node.late_control_inputs,
        )
        assert envelope.config_sha256 == (
            node.config_sha256(
                late_control_inputs_sha256=late_binding.inputs_sha256
            )
            if node.late_control_inputs
            else node.config_sha256()
        )
        assert envelope.late_control_inputs_sha256 == late_binding.inputs_sha256
        assert envelope.late_control_snapshot_sha256 == late_binding.snapshot_sha256
        assert envelope.argv[0] == status["python_executable"]
        assert envelope.environment_lock_sha256 == status["environment_lock"][
            "lock_sha256"
        ]
        assert envelope.outputs_sha256 == declared_path_hashes(workspace, node.outputs)
        assert envelope.checkpoint_sha256 == collect_checkpoint_hashes(
            workspace, (*node.inputs, *node.outputs)
        )
        assert set(envelope.upstream_sha256) == set(node.deps)


def test_strict_claim_artifacts_are_v3_and_submission_ready(release_run) -> None:
    run_dir, _profile = release_run
    artifacts = run_dir / "workspace" / "artifacts"
    claims = json.loads(
        (artifacts / "paper" / "claims" / "claims.v3.json").read_text(encoding="utf-8")
    )
    build = json.loads(
        (artifacts / "paper" / "submission" / "build_report.v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert claims["schema_version"] == "paper-claims-v3"
    assert claims["submission_ready"] is True
    assert claims["status"] in {"PROCEED", "PIVOT"}
    assert build["schema_version"] == "submission-build-v3"
    assert build["submission_ready"] is True
    assert build["stage15_status"] == claims["status"]
    assert build["pages"] == 5
    assert build["overfull_boxes"] == 0
