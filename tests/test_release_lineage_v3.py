import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from ecgcert import lineage
from ecgcert.execution import ExperimentManifest, ResultEnvelope
from ecgcert.execution.budget import BudgetLease, SETTLEMENT_SCHEMA
from ecgcert.execution.envelope import SCHEMA_VERSION
from ecgcert.execution.late_inputs import empty_late_control_snapshot_sha256
from ecgcert.execution.runner import collect_checkpoint_hashes, declared_path_hashes
from ecgcert.execution.runner import expected_mutable_roots, immutable_workspace_sha256
from ecgcert.paper_evidence import FIGURE_ARTIFACTS
from scripts import release


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _manifest_payload() -> dict:
    common = {
        "profile": ["icassp", "extended", "legacy"],
        "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
        "timeout": 10,
        "seed": 11,
    }
    return {
        "schema_version": 1,
        "nodes": [
            {
                **common,
                "id": "producer",
                "command": ["{python}", "task.py"],
                "deps": [],
                "inputs": ["source.txt"],
                "outputs": [
                    "artifacts/producer.json",
                    "artifacts/linear_model.npz",
                ],
            },
            {
                **common,
                "id": "consumer",
                "command": ["{python}", "consumer.py"],
                "deps": ["producer"],
                "inputs": ["artifacts/producer.json"],
                "outputs": ["artifacts/consumer.json"],
            },
        ],
    }


def _write_envelope(
    *,
    run_dir: Path,
    manifest: ExperimentManifest,
    node_id: str,
    commit: str,
) -> None:
    workspace = run_dir / "workspace"
    envelope_dir = run_dir / "envelopes"
    node = manifest.by_id()[node_id]
    python = str(Path(sys.executable).resolve())
    argv = [python if token == "{python}" else token for token in node.command]
    input_hashes = declared_path_hashes(workspace, node.inputs)
    output_hashes = declared_path_hashes(workspace, node.outputs)
    checkpoints = collect_checkpoint_hashes(workspace, (*node.inputs, *node.outputs))
    data_inputs = {
        rel: digest for rel, digest in input_hashes.items() if rel not in checkpoints
    }
    upstream = {
        dep: lineage.artifact_sha256(envelope_dir / f"{dep}.json") for dep in node.deps
    }
    envelope = ResultEnvelope(
        schema_version=SCHEMA_VERSION,
        run_id=run_dir.name,
        node_id=node.id,
        status="succeeded",
        exit_code=0,
        started_at="2026-07-19T00:00:00Z",
        finished_at="2026-07-19T00:01:00Z",
        commit=commit,
        dirty=False,
        argv=argv,
        config_sha256=node.config_sha256(),
        data_sha256=lineage.canonical_sha256(data_inputs),
        split_sha256=lineage.canonical_sha256({
            "seed": node.seed,
            "argv": argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        }),
        env_sha256="a" * 64,
        environment_lock_sha256=lineage.artifact_sha256(
            workspace / "environments" / "cpu.lock.txt"
        ),
        source_sha256=immutable_workspace_sha256(
            workspace,
            expected_mutable_roots(release.ROOT, commit, manifest.select("icassp")),
        ),
        hardware={"cpu_count": 1},
        seed=node.seed,
        upstream_sha256=upstream,
        late_control_inputs_sha256={},
        late_control_snapshot_sha256=empty_late_control_snapshot_sha256(),
        checkpoint_sha256=checkpoints,
        outputs_sha256=output_hashes,
    )
    envelope.write(envelope_dir / f"{node.id}.json")


def _write_submission_fixture(
    workspace: Path,
    arc_receipts_sha256: dict[str, str],
) -> tuple[Path, Path]:
    submission = workspace / "artifacts" / "paper" / "submission"
    build = submission / "build"
    build.mkdir(parents=True)
    compiled_paths = {
        **{
            f"source/{name}": build / name
            for name in (
                "main_v2.tex",
                "compliance.tex",
                "refs.bib",
                "spconf.sty",
                "IEEEbib.bst",
            )
        },
        "auto/robust_map_placeholders.tex": build
        / "auto"
        / "robust_map_placeholders.tex",
        "figures_v3/summary.v3.json": build / "figures_v3" / "summary.v3.json",
        **{
            f"figures_v3/{name}": build / "figures_v3" / name
            for name in FIGURE_ARTIFACTS
        },
        "provenance/author_declaration.v1.json": build
        / "provenance"
        / "author_declaration.v1.json",
        "provenance/tool_provenance.v1.json": build
        / "provenance"
        / "tool_provenance.v1.json",
        "provenance/venue_policy.v1.json": build
        / "provenance"
        / "venue_policy.v1.json",
        "provenance/author_public_key": build / "provenance" / "author_public_key",
        "provenance/author_kit.source": build / "provenance" / "author_kit.source",
        "auto/author_declaration.tex": build / "auto" / "author_declaration.tex",
        "auto/tool_provenance.tex": build / "auto" / "tool_provenance.tex",
        "auto/venue_policy.tex": build / "auto" / "venue_policy.tex",
        "author_kit/official-template.tex": build
        / "author_kit"
        / "official-template.tex",
        **{
            f"provenance/arc-stage{stage}-report.v1.json": build
            / "provenance"
            / f"arc-stage{stage}-report.v1.json"
            for stage in release.ORDERED_STAGES
        },
    }
    for index, (name, path) in enumerate(compiled_paths.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"compiled-{index}-{name}".encode())
    compiled = {
        name: lineage.artifact_sha256(path) for name, path in compiled_paths.items()
    }
    claims_dir = workspace / "artifacts" / "paper" / "claims"
    claims_dir.mkdir(parents=True)
    claims_path = claims_dir / "claims.v3.json"
    claims_path.write_text('{"schema_version":"paper-claims-v3"}\n', encoding="utf-8")
    registry_path = claims_dir / "verified_registry.v1.json"
    registry_path.write_text('{"schema_version":"verified-registry-v1"}\n', encoding="utf-8")
    scientific = {
        "paper/main_v2.tex": compiled["source/main_v2.tex"],
        "paper/compliance.tex": compiled["source/compliance.tex"],
        "paper/refs.bib": compiled["source/refs.bib"],
        "claims/claims.v3.json": lineage.artifact_sha256(claims_path),
        "claims/robust_map_placeholders.tex": compiled[
            "auto/robust_map_placeholders.tex"
        ],
        "claims/verified_registry.v1.json": lineage.artifact_sha256(registry_path),
        "figures/summary.v3.json": compiled["figures_v3/summary.v3.json"],
        **{
            f"figures/{name}": compiled[f"figures_v3/{name}"]
            for name in FIGURE_ARTIFACTS
        },
    }
    gate_dir = workspace / "artifacts" / "gates" / "paper"
    gate_dir.mkdir(parents=True)
    policy_paths = {
        "author_declaration": gate_dir / "author_declaration.v1.json",
        "tool_provenance": gate_dir / "tool_provenance.v1.json",
        "venue_policy": gate_dir / "venue_policy.v1.json",
        "author_public_key": gate_dir / "author_ed25519.pub",
        "author_kit": gate_dir / "icassp2027-author-kit.zip",
    }
    for index, path in enumerate(policy_paths.values()):
        path.write_bytes(f"policy-{index}".encode())
    policy_sha256 = {
        name: lineage.artifact_sha256(path) for name, path in policy_paths.items()
    }
    stage20_inputs = {
        "claim_sync": scientific["claims/claims.v3.json"],
        "claim_macros": scientific["claims/robust_map_placeholders.tex"],
        "verified_registry": scientific["claims/verified_registry.v1.json"],
        "figures_summary": scientific["figures/summary.v3.json"],
        **{
            f"figure_artifact:{name}": scientific[f"figures/{name}"]
            for name in FIGURE_ARTIFACTS
        },
    }
    stage20_path = (
        workspace
        / "artifacts"
        / "control"
        / "stage20_review"
        / "decision.v3.json"
    )
    stage20_path.parent.mkdir(parents=True)
    stage20_path.write_text(
        json.dumps(
            {
                "decision": "PROCEED",
                "evidence": {
                    "reviewed_scientific_input_sha256": scientific,
                    "reviewed_scientific_input_bundle_sha256": (
                        lineage.canonical_sha256(scientific)
                    ),
                    "input_sha256": stage20_inputs,
                },
            }
        ),
        encoding="utf-8",
    )
    pdf = submission / "main_v2.pdf"
    pdf.write_bytes(b"submission-pdf-fixture")
    report_path = submission / "build_report.v3.json"
    report = {
        "schema_version": "submission-build-v3",
        "status": "complete",
        "submission_ready": True,
        "pages": 5,
        "technical_content_end_page": 4,
        "page_five_content_validated": True,
        "overfull_boxes": 0,
        "pdf_sha256": lineage.artifact_sha256(pdf),
        "arc_formal_receipts_sha256": arc_receipts_sha256,
        "reviewed_scientific_input_sha256": scientific,
        "reviewed_scientific_input_bundle_sha256": lineage.canonical_sha256(
            scientific
        ),
        "claims_sha256": stage20_inputs["claim_sync"],
        "claim_macros_sha256": stage20_inputs["claim_macros"],
        "verified_registry_sha256": stage20_inputs["verified_registry"],
        "figures_summary_sha256": stage20_inputs["figures_summary"],
        "figure_artifacts_sha256": {
            name: stage20_inputs[f"figure_artifact:{name}"]
            for name in FIGURE_ARTIFACTS
        },
        "compiled_input_sha256": compiled,
        "paper_policy_input_sha256": policy_sha256,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (submission / "build_report.v3.sha256").write_text(
        f"{lineage.artifact_sha256(report_path)}  build_report.v3.json\n",
        encoding="ascii",
    )
    return report_path, stage20_path


def _release_run(tmp_path: Path, monkeypatch) -> tuple[Path, ExperimentManifest]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    payload = _manifest_payload()
    (root / "scripts" / "experiment_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
    )
    (root / "task.py").write_text("print('producer')\n", encoding="utf-8")
    (root / "consumer.py").write_text("print('consumer')\n", encoding="utf-8")
    (root / "environments").mkdir()
    (root / "environments" / "cpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8",
    )
    (root / "environments" / "gpu.lock.txt").write_text(
        "fixture==1.0 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "release fixture")
    commit = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(release, "ROOT", root)
    # ARC's four-report chain has a dedicated adversarial release suite; this
    # fixture isolates workspace/envelope/budget lineage.
    arc_receipts_sha256 = {
        str(stage): f"{stage:064x}" for stage in release.ORDERED_STAGES
    }
    monkeypatch.setattr(
        release,
        "_validate_arc_gate_chain",
        lambda _run_dir: arc_receipts_sha256,
    )
    monkeypatch.setattr(
        release,
        "validate_reviewed_gate",
        lambda _value, require_proceed: None,
    )
    monkeypatch.setattr(release.lineage, "environment_sha256", lambda: "a" * 64)
    lock_digest = lineage.artifact_sha256(root / "environments" / "cpu.lock.txt")

    def fake_locked_environment(**_kwargs):
        return SimpleNamespace(lock_sha256=lock_digest)

    monkeypatch.setattr(release, "require_locked_environment", fake_locked_environment)

    manifest = ExperimentManifest.from_path(root / "scripts" / "experiment_manifest.yaml")
    run_dir = tmp_path / "runs" / "release-test"
    workspace = run_dir / "workspace"
    (workspace / "environments").mkdir(parents=True)
    for name in ("cpu.lock.txt", "gpu.lock.txt"):
        (workspace / "environments" / name).write_bytes(
            (root / "environments" / name).read_bytes()
        )
    (workspace / "task.py").write_bytes((root / "task.py").read_bytes())
    (workspace / "consumer.py").write_bytes((root / "consumer.py").read_bytes())
    (workspace / "scripts").mkdir()
    (workspace / "scripts" / "experiment_manifest.yaml").write_bytes(
        (root / "scripts" / "experiment_manifest.yaml").read_bytes()
    )
    (workspace / "artifacts").mkdir(parents=True)
    (workspace / "source.txt").write_text("frozen input\n", encoding="utf-8")
    (workspace / "artifacts" / "producer.json").write_text(
        '{"producer": true}\n', encoding="utf-8",
    )
    (workspace / "artifacts" / "linear_model.npz").write_bytes(
        b"linear-checkpoint"
    )
    (workspace / "artifacts" / "consumer.json").write_text(
        '{"consumer": true}\n', encoding="utf-8",
    )
    (run_dir / "envelopes").mkdir()
    _write_envelope(
        run_dir=run_dir, manifest=manifest, node_id="producer", commit=commit,
    )
    _write_envelope(
        run_dir=run_dir, manifest=manifest, node_id="consumer", commit=commit,
    )
    _write_submission_fixture(workspace, arc_receipts_sha256)
    limits = {
        "cpu_core_hours": 4000.0,
        "gpu_hours": 500.0,
        "artifact_bytes": 100 * 1024**3,
    }
    budget_lease = BudgetLease(
        control_root=run_dir.parent,
        run_id=run_dir.name,
        limits=limits,
        reserved_gpu_hours=100.0,
    )
    zero_usage = {
        "cpu_core_hours": 0.0,
        "gpu_hours": 0.0,
        "artifact_bytes": 0,
    }
    reservation = budget_lease.acquire(zero_usage)
    settlement = budget_lease.settle(zero_usage, run_state="succeeded")
    budget_lease.release()
    settlement_path = run_dir / "budget-settlement.v1.json"
    settlement_path.write_text(json.dumps({
        "schema_version": SETTLEMENT_SCHEMA,
        "reservation": reservation,
        "settlement": settlement,
    }), encoding="utf-8")
    lock_record = {
        "lock_name": "cpu",
        "lock_path": "environments/cpu.lock.txt",
        "lock_sha256": lock_digest,
        "python_executable": str(Path(sys.executable).resolve()),
        "requirement_count": 1,
        "applicable_requirement_count": 1,
        "checked_requirement_count": 1,
        "mismatches": [],
        "ok": True,
    }
    mutable_roots = expected_mutable_roots(root, commit, manifest.select("icassp"))
    source_sha256 = immutable_workspace_sha256(workspace, mutable_roots)
    run_identity = {
        "run_id": run_dir.name,
        "control_root": str(run_dir.parent.resolve()),
        "profile": "icassp",
        "resource": None,
        "environment_lock": lock_record,
        "environment_sha256": "a" * 64,
        "python_executable": str(Path(sys.executable).resolve()),
        "manifest_sha256": manifest.sha256(),
        "commit": commit,
        "source_snapshot_sha256": source_sha256,
        "mutable_workspace_roots": list(mutable_roots),
    }
    logs = run_dir / "logs"
    logs.mkdir()
    node_results = {}
    for node in manifest.select("icassp"):
        stdout = logs / f"{node.id}.stdout.log"
        stderr = logs / f"{node.id}.stderr.log"
        stdout.write_text("fixture\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        node_results[node.id] = {
            "state": "succeeded",
            "exit_code": 0,
            "stdout": f"logs/{node.id}.stdout.log",
            "stderr": f"logs/{node.id}.stderr.log",
            "stdout_sha256": lineage.artifact_sha256(stdout),
            "stderr_sha256": lineage.artifact_sha256(stderr),
        }
    attempt = {
        "ordinal": 0,
        "attempt_id": "initial",
        "budget_run_id": run_dir.name,
        "resumed": False,
        "resume_from_node": "producer",
        "selected_nodes": [node.id for node in manifest.select("icassp")],
        "started_at": "2026-07-19T00:00:00Z",
        "finished_at": "2026-07-19T00:02:00Z",
        "state": "succeeded",
        "error": "",
        "planned_timeout_upper_bound": zero_usage,
        "used": zero_usage,
        "log_dir": "logs",
        "node_results": node_results,
        "settlement": {
            "path": "budget-settlement.v1.json",
            "reservation_sha256": reservation["event_sha256"],
            "settlement_sha256": settlement["event_sha256"],
            "artifact_sha256": lineage.artifact_sha256(settlement_path),
            "ledger_event_ordinal": 1,
        },
        "run_identity_sha256": lineage.canonical_sha256(run_identity),
        "previous_attempt_sha256": "0" * 64,
    }
    attempt["attempt_sha256"] = lineage.canonical_sha256(attempt)
    status = {
        "schema_version": 3,
        "run_id": run_dir.name,
        "control_root": str(run_dir.parent.resolve()),
        "profile": "icassp",
        "resource": None,
        "environment_lock": lock_record,
        "environment_sha256": "a" * 64,
        "python_executable": str(Path(sys.executable).resolve()),
        "manifest_sha256": manifest.sha256(),
        "commit": commit,
        "state": "succeeded",
        "exit_code": 0,
        "run_identity_sha256": lineage.canonical_sha256(run_identity),
        "attempts": [attempt],
        "nodes": {
            node.id: {"state": "succeeded", "exit_code": 0}
            for node in manifest.select("icassp")
        },
        "budget": {
            "limits": {**limits, "reserved_gpu_hours_for_rerun": 100.0},
            "planned_timeout_upper_bound": zero_usage,
            "used": zero_usage,
            "global_ledger": {
                "ledger_relative_path": "budget-ledger.v1.jsonl",
                "bound_event_count": 2,
                "bound_tail_event_sha256": settlement["event_sha256"],
                "bound_cumulative_after": settlement["cumulative_after"],
            },
        },
    }
    status["mutable_workspace_roots"] = list(mutable_roots)
    status["source_snapshot_sha256"] = source_sha256
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return run_dir, manifest


def test_release_recomputes_workspace_and_accepts_consistent_lineage(tmp_path, monkeypatch):
    run_dir, _ = _release_run(tmp_path, monkeypatch)

    release._validate_run(run_dir, "icassp")
    producer = ResultEnvelope.read(run_dir / "envelopes" / "producer.json")
    assert set(producer.checkpoint_sha256) == {"artifacts/linear_model.npz"}


def test_release_rejects_pending_submission_and_missing_stage20(tmp_path, monkeypatch):
    run_dir, _ = _release_run(tmp_path, monkeypatch)
    report_path = (
        run_dir
        / "workspace"
        / "artifacts"
        / "paper"
        / "submission"
        / "build_report.v3.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submission_ready"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit, match="final submission build state"):
        release._validate_run(run_dir, "icassp")

    report["submission_ready"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (report_path.parent / "build_report.v3.sha256").write_text(
        f"{lineage.artifact_sha256(report_path)}  build_report.v3.json\n",
        encoding="ascii",
    )
    (
        run_dir
        / "workspace"
        / "artifacts"
        / "control"
        / "stage20_review"
        / "decision.v3.json"
    ).unlink()
    with pytest.raises(SystemExit, match="cannot read reviewed Stage 20 artifact"):
        release._validate_run(run_dir, "icassp")


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("scientific_binding", "scientific inputs differ"),
        ("arc_binding", "formal ARC four-gate chain"),
        ("compiled_copy", "compiled submission input hash mismatch"),
        ("signed_policy", "signed paper-policy inputs"),
    ],
)
def test_release_binds_final_submission_to_stage20_and_signed_inputs(
    tmp_path,
    monkeypatch,
    tamper,
    message,
):
    run_dir, _ = _release_run(tmp_path, monkeypatch)
    workspace = run_dir / "workspace"
    submission = workspace / "artifacts" / "paper" / "submission"
    report_path = submission / "build_report.v3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tamper == "scientific_binding":
        report["reviewed_scientific_input_sha256"]["paper/main_v2.tex"] = "f" * 64
    elif tamper == "arc_binding":
        report["arc_formal_receipts_sha256"]["20"] = "f" * 64
    elif tamper == "compiled_copy":
        with (submission / "build" / "main_v2.tex").open("ab") as stream:
            stream.write(b"tampered")
    else:
        with (
            workspace
            / "artifacts"
            / "gates"
            / "paper"
            / "tool_provenance.v1.json"
        ).open("ab") as stream:
            stream.write(b"tampered")
    if tamper in {"scientific_binding", "arc_binding"}:
        report_path.write_text(json.dumps(report), encoding="utf-8")
        (submission / "build_report.v3.sha256").write_text(
            f"{lineage.artifact_sha256(report_path)}  build_report.v3.json\n",
            encoding="ascii",
        )
    with pytest.raises(SystemExit, match=message):
        release._validate_run(run_dir, "icassp")


def test_release_cli_exposes_no_pending_gate_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["release.py", "--allow-pending-gate"])

    with pytest.raises(SystemExit) as error:
        release.main()

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("output", "workspace outputs"),
        ("input", "input-data hash"),
        ("config", "config hash"),
        ("envelope_commit", "identity mismatch"),
        ("status_commit", "current source commit"),
        ("upstream", "upstream envelope hashes"),
        ("source", "source snapshot does not match the commit"),
        ("lock_envelope", "environment lock hash"),
        ("lock_status", "environment-lock record"),
        ("environment_status", "environment fingerprint"),
        ("environment_envelope", "environment fingerprint"),
        ("checkpoint_missing", "invalid result envelope"),
        ("checkpoint_null", "invalid result envelope"),
        ("checkpoint_hash", "checkpoint hashes"),
        ("budget", "invalid global budget attempt history"),
        ("attempt_hash", "invalid global budget attempt history"),
        ("attempt_missing", "invalid global budget attempt history"),
        ("attempt_log", "invalid global budget attempt history"),
        ("settlement_missing", "invalid global budget attempt history"),
        ("global_ledger_missing", "invalid global budget attempt history"),
        ("old_schema", "run status schema"),
        ("active_attempt", "invalid global budget attempt history"),
        ("budget_limit", "budget limits do not match"),
    ],
)
def test_release_fails_closed_on_lineage_tampering(
    tmp_path,
    monkeypatch,
    tamper,
    message,
):
    run_dir, _ = _release_run(tmp_path, monkeypatch)
    workspace = run_dir / "workspace"
    envelope_dir = run_dir / "envelopes"
    if tamper == "output":
        (workspace / "artifacts" / "consumer.json").write_text(
            '{"tampered": true}\n', encoding="utf-8",
        )
    elif tamper == "input":
        (workspace / "source.txt").write_text("tampered input\n", encoding="utf-8")
    elif tamper in {"config", "envelope_commit", "upstream"}:
        node_id = "producer" if tamper != "upstream" else "consumer"
        path = envelope_dir / f"{node_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "config":
            value["config_sha256"] = "f" * 64
        elif tamper == "envelope_commit":
            value["commit"] = "e" * 40
        else:
            value["upstream_sha256"] = {"producer": "f" * 64}
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "status_commit":
        path = run_dir / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["commit"] = "e" * 40
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "source":
        (workspace / "task.py").write_text("print('source tamper')\n", encoding="utf-8")
    elif tamper == "lock_envelope":
        path = envelope_dir / "producer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["environment_lock_sha256"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper in {"lock_status", "environment_status"}:
        path = run_dir / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "lock_status":
            value["environment_lock"]["lock_sha256"] = "f" * 64
        else:
            value["environment_sha256"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "environment_envelope":
        path = envelope_dir / "producer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["env_sha256"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper in {"checkpoint_missing", "checkpoint_null", "checkpoint_hash"}:
        path = envelope_dir / "producer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "checkpoint_missing":
            del value["checkpoint_sha256"]
        elif tamper == "checkpoint_null":
            value["checkpoint_sha256"] = None
        else:
            value["checkpoint_sha256"] = {
                "artifacts/linear_model.npz": "f" * 64
            }
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "budget":
        path = run_dir / "budget-settlement.v1.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["settlement"]["used"]["cpu_core_hours"] = 10.0
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper in {
        "attempt_hash",
        "attempt_missing",
        "old_schema",
        "active_attempt",
        "budget_limit",
    }:
        path = run_dir / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "attempt_hash":
            value["attempts"][0]["attempt_sha256"] = "f" * 64
        elif tamper == "attempt_missing":
            del value["attempts"][0]["settlement"]
        elif tamper == "old_schema":
            value["schema_version"] = 2
        elif tamper == "active_attempt":
            value["active_attempt"] = {"state": "running"}
        else:
            value["budget"]["limits"]["gpu_hours"] = 5_000.0
        path.write_text(json.dumps(value), encoding="utf-8")
    elif tamper == "attempt_log":
        (run_dir / "logs" / "producer.stdout.log").write_text(
            "tampered log\n", encoding="utf-8"
        )
    elif tamper == "settlement_missing":
        (run_dir / "budget-settlement.v1.json").unlink()
    elif tamper == "global_ledger_missing":
        (run_dir.parent / "budget-ledger.v1.jsonl").unlink()
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(tamper)

    with pytest.raises(SystemExit, match=message):
        release._validate_run(run_dir, "icassp")
