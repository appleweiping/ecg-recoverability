from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from ecgcert.arc_control import (
    RECEIPT_SCHEMA,
    validate_arc_control_bundle,
    validate_arc_control_chain,
)
from scripts import run_arc_copilot_bridge as bridge


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _source_config(path: Path, protocol: Path) -> None:
    policies = {
        stage: {
            "pause_before": False,
            "pause_after": stage in bridge.REQUIRED_GATES,
            "require_approval": stage in bridge.REQUIRED_GATES,
        }
        for stage in range(1, 24)
    }
    payload = {
        "knowledge_base": {"backend": "markdown", "root": "sentinel"},
        "llm": {
            "provider": "acp",
            "acp": {"agent": "codex", "session_name": "test-session"},
        },
        "hitl": {"enabled": True, "mode": "co-pilot", "stage_policies": policies},
        "experiment": {"mode": "sandbox", "sandbox": {"python_path": "python"}},
        "prompts": {"custom_file": "", "extra_prompts": {"quality_gate": str(protocol)}},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _stage_records(
    stage_dir: Path,
    artifacts: list[str],
    *,
    status: str = "done",
    decision: str = "proceed",
    run_id: str = "rc-test",
) -> None:
    stage = int(stage_dir.name.split("-")[1])
    stage_slug = {
        5: "literature_screen",
        9: "experiment_design",
        15: "research_decision",
        20: "quality_gate",
    }.get(stage, "stage")
    stage_id = f"{stage:02d}-{stage_slug}"
    decision_at = f"2026-07-19T00:{stage:02d}:00+00:00"
    health_at = f"2026-07-19T00:{stage:02d}:01+00:00"
    _write_json(
        stage_dir / "decision.json",
        {
            "stage_id": stage_id,
            "run_id": run_id,
            "status": status,
            "decision": decision,
            "output_artifacts": artifacts,
            "error": None,
            "evidence_refs": [
                f"stage-{stage:02d}/{artifact}" for artifact in artifacts
            ],
            "ts": decision_at,
            "next_stage": stage if status == "blocked_approval" else stage + 1,
        },
    )
    _write_json(
        stage_dir / "stage_health.json",
        {
            "stage_id": stage_id,
            "run_id": run_id,
            "status": status,
            "artifacts_count": len(artifacts),
            "error": None,
            "duration_sec": 1.0,
            "timestamp": health_at,
        },
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(
        prior + json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_approved_gate(
    output: Path,
    stage: int,
    *,
    run_id: str = "rc-test",
    session_id: str = "session-control-1",
) -> tuple[Path, Path, Path]:
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    source_snapshot = control / "source-config.yaml"
    effective = control / "effective-config.yaml"
    project_state = control / "project-state.v1.json"
    if not source_snapshot.exists():
        source_snapshot.write_text("source: frozen\n", encoding="utf-8")
        effective.write_text("effective: frozen\n", encoding="utf-8")
        _write_json(project_state, {"state_sha256": "frozen"})

    stage_dir = output / f"stage-{stage:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == 5:
        output_name = "shortlist.jsonl"
        (stage_dir / output_name).write_text(
            json.dumps(
                {
                    "title": "Verified primary ECG reconstruction paper",
                    "verified": True,
                    "details": "scientific evidence " * 8,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        output_name = "gate-review.md"
        (stage_dir / output_name).write_text(
            f"# Stage {stage} frozen review artifact\n\n" + "evidence " * 20,
            encoding="utf-8",
        )
    _stage_records(
        stage_dir,
        [output_name],
        status="blocked_approval",
        decision="block",
        run_id=run_id,
    )

    waiting_hash = str((stage + 1) % 10) * 64
    nonce = str((stage + 2) % 10) * 64
    _write_json(output / "checkpoint.json", {"last_completed_stage": stage - 1})
    checkpoint_binding = bridge._checkpoint_binding_for_gate(output, stage)
    bridge._snapshot_preapproval_checkpoint(
        output=output,
        stage=stage,
        checkpoint_binding=checkpoint_binding,
    )
    response = {
        "schema_version": bridge.RESPONSE_SCHEMA,
        "stage": stage,
        "run_id": run_id,
        "session_id": session_id,
        "waiting_sha256": waiting_hash,
        "preapproval_checkpoint_sha256": checkpoint_binding["sha256"],
        "nonce": nonce,
        "action": "approve",
        "issued_at": f"2026-07-19T00:{stage:02d}:02+00:00",
        "message": "reviewed",
    }
    response_source = output / "hitl" / "operator-response.v2.json"
    _write_json(response_source, response)
    response_hash = bridge._sha256(response_source)
    response_snapshot = bridge._snapshot_operator_response(
        output=output,
        response_path=response_source,
        stage=stage,
        response_sha256=response_hash,
    )
    consumption_path = (
        output
        / "control"
        / "operator-response-consumption"
        / f"stage-{stage:02d}-{waiting_hash}.v1.json"
    )
    _write_json(
        consumption_path,
        {
            "schema_version": bridge.CONSUMPTION_SCHEMA,
            "claimed_at": f"2026-07-19T00:{stage:02d}:02+00:00",
            "response_path": response_snapshot.relative_to(output).as_posix(),
            "response_sha256": response_hash,
            "stage": stage,
            "run_id": run_id,
            "session_id": session_id,
            "waiting_sha256": waiting_hash,
            "preapproval_checkpoint_sha256": checkpoint_binding["sha256"],
            "nonce": nonce,
            "response": response,
        },
    )
    _append_jsonl(
        output / "hitl" / "interventions.jsonl",
        {
            "id": f"approval-{stage}",
            "type": "approve",
            "stage": stage,
            "stage_name": bridge.STAGE_NUMBER_NAMES[stage],
            "timestamp": f"2026-07-19T00:{stage:02d}:03+00:00",
            "human_input": {
                "action": "approve",
                "timestamp": f"2026-07-19T00:{stage:02d}:02+00:00",
            },
            "pause_reason": "gate_approval",
            "outcome": "Human chose: approve",
            "accepted": True,
            "duration_sec": 1.0,
        },
    )
    intervention_count = len(
        bridge._read_jsonl_records(output / "hitl" / "interventions.jsonl")
    )
    _write_json(
        output / "hitl" / "session.json",
        {
            "session_id": session_id,
            "run_id": run_id,
            "state": "active",
            "mode": "co-pilot",
            "created_at": "2026-07-19T00:00:00+00:00",
            "last_activity": f"2026-07-19T00:{stage:02d}:04+00:00",
            "waiting": None,
            "interventions_count": intervention_count,
        },
    )
    _append_jsonl(
        output / "bridge.events.v1.jsonl",
        {
            "timestamp": f"2026-07-19T00:{stage:02d}:03+00:00",
            "stage": stage,
            "run_id": run_id,
            "session_id": session_id,
            "action": "approve",
            "issued_at": response["issued_at"],
            "nonce": nonce,
            "response_sha256": response_hash,
            "waiting_sha256": waiting_hash,
            "checkpoint_before_approval": checkpoint_binding,
            "consumption_receipt_path": consumption_path.relative_to(output).as_posix(),
            "consumption_receipt_sha256": bridge._sha256(consumption_path),
        },
    )
    return source_snapshot, effective, project_state


def _add_approved_gate_handoff(
    output: Path,
    stage: int,
    *,
    run_id: str = "rc-test",
    session_id: str = "session-control-1",
) -> Path:
    source_snapshot, effective, project_state = _prepare_approved_gate(
        output,
        stage,
        run_id=run_id,
        session_id=session_id,
    )

    record = bridge._build_gate_handoff(
        output=output,
        stage=stage,
        source_config_snapshot=source_snapshot,
        effective_config=effective,
        project_state_snapshot=project_state,
    )
    handoff = bridge._gate_handoff_path(output, stage)
    _write_json(handoff, record)
    _write_json(
        output / "bridge.status.v2.json",
        {
            "state": "completed",
            "gate_handoff_stage": stage,
            "gate_handoff_path": str(handoff),
            "gate_handoff_sha256": bridge._sha256(handoff),
        },
    )
    return handoff


def _operator_exchange(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    _write_json(
        output / "hitl" / "session.json",
        {
            "run_id": "rc-bound",
            "session_id": "session-bound",
            "state": "active",
        },
    )
    _write_json(output / "checkpoint.json", {"last_completed_stage": 4})
    waiting = {
        "stage": 5,
        "stage_name": "LITERATURE_SCREEN",
        "reason": "gate_approval",
        "since": "2026-07-19T00:00:00+00:00",
        "available_actions": ["approve", "reject", "abort"],
    }
    waiting_path = output / "hitl" / "waiting.json"
    _write_json(waiting_path, waiting)
    challenge = bridge._build_operator_challenge(
        output=output,
        waiting=waiting,
        waiting_sha256=bridge._sha256(waiting_path),
    )
    response = {
        "schema_version": bridge.RESPONSE_SCHEMA,
        "stage": challenge["stage"],
        "run_id": challenge["run_id"],
        "session_id": challenge["session_id"],
        "waiting_sha256": challenge["waiting_sha256"],
        "preapproval_checkpoint_sha256": challenge[
            "preapproval_checkpoint_sha256"
        ],
        "nonce": challenge["nonce"],
        "action": "approve",
        "issued_at": "2026-07-19T12:00:00+00:00",
        "message": "reviewed the frozen gate",
    }
    response_path = output / "hitl" / "operator-response.v2.json"
    _write_json(response_path, response)
    return waiting, challenge, response, response_path


def test_operator_response_rejects_stale_issuance(tmp_path: Path) -> None:
    waiting, challenge, response, _response_path = _operator_exchange(tmp_path / "run")
    response["issued_at"] = "2026-07-20T00:00:01+00:00"

    with pytest.raises(ValueError, match="outside the waiting.since-to-24h window"):
        bridge._validate_response(response, waiting, challenge)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "session_id",
        "waiting_sha256",
        "preapproval_checkpoint_sha256",
        "nonce",
    ],
)
def test_operator_response_must_bind_active_challenge(
    field: str, tmp_path: Path
) -> None:
    waiting, challenge, response, _response_path = _operator_exchange(tmp_path / "run")
    response[field] = "f" * 64

    with pytest.raises(ValueError, match=field):
        bridge._validate_response(response, waiting, challenge)


def test_operator_response_consumption_is_atomic_and_one_time(tmp_path: Path) -> None:
    output = tmp_path / "run"
    waiting, challenge, response, response_path = _operator_exchange(output)
    bridge._validate_response(response, waiting, challenge)
    response_sha256 = bridge._sha256(response_path)
    checkpoint_binding = bridge._checkpoint_binding_for_gate(output, 5)
    bridge._snapshot_preapproval_checkpoint(
        output=output,
        stage=5,
        checkpoint_binding=checkpoint_binding,
    )
    response_snapshot = bridge._snapshot_operator_response(
        output=output,
        response_path=response_path,
        stage=5,
        response_sha256=response_sha256,
    )

    claim = bridge._claim_operator_response(
        output=output,
        response_path=response_snapshot,
        response=response,
        response_sha256=response_sha256,
        challenge=challenge,
    )

    assert claim.is_file()
    with pytest.raises(ValueError, match="replay"):
        bridge._claim_operator_response(
            output=output,
            response_path=response_snapshot,
            response=response,
            response_sha256=response_sha256,
            challenge=challenge,
        )


def test_operator_response_snapshot_is_rechecked_at_consumption(tmp_path: Path) -> None:
    output = tmp_path / "run"
    waiting, challenge, response, response_path = _operator_exchange(output)
    bridge._validate_response(response, waiting, challenge)
    response_sha256 = bridge._sha256(response_path)
    checkpoint_binding = bridge._checkpoint_binding_for_gate(output, 5)
    bridge._snapshot_preapproval_checkpoint(
        output=output,
        stage=5,
        checkpoint_binding=checkpoint_binding,
    )
    response_snapshot = bridge._snapshot_operator_response(
        output=output,
        response_path=response_path,
        stage=5,
        response_sha256=response_sha256,
    )
    response_snapshot.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot is missing, changed, or inconsistent"):
        bridge._claim_operator_response(
            output=output,
            response_path=response_snapshot,
            response=response,
            response_sha256=response_sha256,
            challenge=challenge,
        )


def test_new_output_must_be_empty_and_external(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "runs" / "run-1"
    bridge._assert_external_output(external, project)
    with pytest.raises(ValueError, match="outside"):
        bridge._assert_external_output(project / "arc-run", project)

    external.mkdir(parents=True)
    bridge._assert_empty_new_output(external, resume=False)
    (external / "unexpected.txt").write_text("state", encoding="utf-8")
    with pytest.raises(ValueError, match="completely empty"):
        bridge._assert_empty_new_output(external, resume=False)


def test_effective_config_is_external_run_isolated_and_hashable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    protocol = project / "protocol.md"
    protocol.write_text("frozen protocol", encoding="utf-8")
    source = project / "config.yaml"
    _source_config(source, protocol)
    original = source.read_bytes()
    python = tmp_path / "python.exe"
    python.write_text("runtime", encoding="utf-8")
    acpx = tmp_path / "acpx.cmd"
    acpx.write_text("runtime", encoding="utf-8")

    first_output = tmp_path / "runs" / "first"
    loaded, payload = bridge._prepare_effective_config(
        source_config=source,
        output=first_output,
        project_root=project,
        python=python,
        acpx_command=acpx,
    )
    second, _ = bridge._prepare_effective_config(
        source_config=source,
        output=tmp_path / "runs" / "second",
        project_root=project,
        python=python,
        acpx_command=acpx,
    )

    assert source.read_bytes() == original
    assert loaded["knowledge_base"]["root"] == str((first_output / "kb").resolve())
    assert loaded["llm"]["acp"]["cwd"] == str(project)
    assert loaded["experiment"]["sandbox"]["python_path"] == str(python)
    assert loaded["prompts"]["extra_prompts"]["quality_gate"] == str(protocol)
    assert loaded["llm"]["acp"]["session_name"] != second["llm"]["acp"]["session_name"]
    assert yaml.safe_load(payload) == loaded


def test_repository_arc_gates_pause_once_after_artifacts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repository_root / "arc_audit" / "config.arc.yaml").read_text(
            encoding="utf-8"
        )
    )
    policies = config["hitl"]["stage_policies"]
    assert config["experiment"]["mode"] == "sandbox"
    assert "ssh_remote" not in config["experiment"]

    approval_stages = {
        int(stage)
        for stage, policy in policies.items()
        if policy.get("require_approval") is True
    }
    assert approval_stages == bridge.REQUIRED_GATES
    for stage in bridge.REQUIRED_GATES:
        policy = policies[stage]
        assert policy["pause_before"] is False
        assert policy["pause_after"] is True
        assert policy["require_approval"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("pause_before", True), ("pause_after", False)],
)
def test_effective_config_rejects_non_post_artifact_gate_pause(
    field: str,
    value: bool,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    protocol = project / "protocol.md"
    protocol.write_text("frozen protocol", encoding="utf-8")
    source = project / "config.yaml"
    _source_config(source, protocol)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["hitl"]["stage_policies"][15][field] = value
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_text("runtime", encoding="utf-8")
    acpx = tmp_path / "acpx.cmd"
    acpx.write_text("runtime", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one post-artifact review"):
        bridge._prepare_effective_config(
            source_config=source,
            output=tmp_path / "run",
            project_root=project,
            python=python,
            acpx_command=acpx,
        )


def test_effective_config_rejects_arc_remote_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    protocol = project / "protocol.md"
    protocol.write_text("frozen protocol", encoding="utf-8")
    source = project / "config.yaml"
    _source_config(source, protocol)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["experiment"]["mode"] = "ssh_remote"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_text("runtime", encoding="utf-8")
    acpx = tmp_path / "acpx.cmd"
    acpx.write_text("runtime", encoding="utf-8")

    with pytest.raises(ValueError, match="local sandbox control plane"):
        bridge._prepare_effective_config(
            source_config=source,
            output=tmp_path / "run",
            project_root=project,
            python=python,
            acpx_command=acpx,
        )


def test_codex_acp_provider_config_is_secret_free_and_model_pinned() -> None:
    config, serialized = bridge._build_codex_acp_config(
        base_url="https://gateway.example/v1/",
        model="gpt-5.4",
        reasoning_effort="high",
        provider_id="ecgcert-gateway",
    )

    assert config["model"] == "gpt-5.4"
    assert config["model_provider"] == "ecgcert-gateway"
    provider = config["model_providers"]["ecgcert-gateway"]
    assert provider["base_url"] == "https://gateway.example/v1"
    assert provider["env_key"] == "OPENAI_API_KEY"
    assert '"api_key":' not in serialized.lower()
    assert "sk-" not in serialized.lower()


@pytest.mark.parametrize(
    "base_url",
    ["http://gateway.example/v1", "https://user:secret@gateway.example/v1", ""],
)
def test_codex_acp_provider_rejects_unsafe_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS URL"):
        bridge._build_codex_acp_config(
            base_url=base_url,
            model="gpt-5.4",
            reasoning_effort="high",
            provider_id="ecgcert-gateway",
        )


def test_provider_error_cannot_be_native_success(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage-01"
    stage_dir.mkdir()
    (stage_dir / "goal.md").write_text(
        "exceeded retry limit, last status: 429 Too Many Requests", encoding="utf-8"
    )
    _write_json(
        stage_dir / "hardware_profile.json",
        {"cpu": "test", "memory": "enough", "profile": "unit-test"},
    )
    _stage_records(stage_dir, ["goal.md", "hardware_profile.json"])

    ready, violations = bridge._validate_completed_stage(stage_dir)

    assert ready is True
    assert {item.code for item in violations} >= {
        "artifact-too-short",
        "provider-error-retry-limit",
    }


def test_short_or_empty_core_artifacts_fail_closed(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage-03"
    stage_dir.mkdir()
    (stage_dir / "search_plan.yaml").write_text("query: ecg\n", encoding="utf-8")
    (stage_dir / "sources.json").write_text("[]\n", encoding="utf-8")
    (stage_dir / "queries.json").write_text("{}\n", encoding="utf-8")
    _stage_records(stage_dir, ["search_plan.yaml", "sources.json", "queries.json"])

    ready, violations = bridge._validate_completed_stage(stage_dir)

    assert ready is True
    codes = {item.code for item in violations}
    assert "artifact-too-short" in codes
    assert "artifact-parse-error" in codes


def test_healthy_stage_one_is_accepted(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage-01"
    stage_dir.mkdir()
    (stage_dir / "goal.md").write_text(
        "# Frozen ICASSP 2027 ECG recoverability goal\n\n"
        "PTB-XL folds 1-7 train, fold 8 tunes, fold 9 fits the target-specific "
        "model-conditional meta-model, and fold 10 tests Delta R^2 without "
        "retuning. The rank set {2,3,4,5} forms the robust envelope. Chapman "
        "and CPSC receive zero-transfer evaluation.\n\n"
        "**Success Criteria**:\n\n"
        "- On PTB-XL fold 10, the Delta R^2 confidence-interval lower bound "
        "is above zero.\n"
        "- On Chapman, zero-transfer "
        "Delta R^2 has a confidence-interval lower bound above zero.\n"
        "- At least three of four common-panel primary reconstructors have a "
        "positive Delta R^2 point estimate.\n\n"
        "If any registered gate fails, PIVOT without retuning.\n\n"
        "**Generated**: 2026-07-19 "
        + ("Substantive scientific scope. " * 12),
        encoding="utf-8",
    )
    _write_json(
        stage_dir / "hardware_profile.json",
        {"cpu": "test", "memory": "64GiB", "gpu": "deferred", "workers": 8},
    )
    _stage_records(stage_dir, ["goal.md", "hardware_profile.json"])

    assert bridge._validate_completed_stage(stage_dir) == (True, [])


def test_stage_one_rejects_cpsc_only_external_hard_gate(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage-01"
    stage_dir.mkdir()
    (stage_dir / "goal.md").write_text(
        "# Frozen ICASSP 2027 ECG recoverability goal\n\n"
        "PTB-XL folds 1-7 train, fold 8 tunes, fold 9 fits the meta-model, "
        "and fold 10 tests Delta R^2. Chapman and CPSC are both reported.\n\n"
        "## Success Criteria\n\n"
        "- On PTB-XL fold 10, the Delta R^2 confidence-interval lower bound "
        "is above zero.\n"
        "- On CPSC, zero-transfer Delta R^2 has a confidence-interval lower "
        "bound above zero.\n"
        "- At least three of four common-panel primary reconstructors have a "
        "positive Delta R^2 point estimate.\n\n"
        "If any registered gate fails, PIVOT without retuning.\n\n"
        + ("Substantive scientific scope. " * 12),
        encoding="utf-8",
    )
    _write_json(
        stage_dir / "hardware_profile.json",
        {"cpu": "test", "memory": "64GiB", "gpu": "deferred", "workers": 8},
    )
    _stage_records(stage_dir, ["goal.md", "hardware_profile.json"])

    ready, violations = bridge._validate_completed_stage(stage_dir)

    assert ready is True
    assert "protocol-stage15-gate-missing-external-zero-transfer" in {
        item.code for item in violations
    }


def test_stage_one_rejects_unregistered_fourth_publishability_gate(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "stage-01"
    stage_dir.mkdir()
    (stage_dir / "goal.md").write_text(
        "# Frozen ICASSP 2027 ECG recoverability goal\n\n"
        "PTB-XL folds 1-7 train, fold 8 tunes, fold 9 fits the target-specific "
        "model-conditional meta-model, and fold 10 tests Delta R^2. The rank "
        "set {2,3,4,5} forms the robust envelope. Chapman and CPSC receive "
        "zero-transfer evaluation.\n\n"
        "## Success Criteria\n\n"
        "- On PTB-XL fold 10, the Delta R^2 confidence-interval lower bound "
        "is above zero.\n"
        "- On Chapman, zero-transfer "
        "Delta R^2 has a confidence-interval lower bound above zero.\n"
        "- At least three of four common-panel primary reconstructors have a "
        "positive Delta R^2 point estimate.\n"
        "- Effects are consistent enough across QRS, ST, and T windows to "
        "support the interpretation.\n\n"
        "If any registered gate fails, PIVOT without retuning. "
        + ("Substantive scientific scope. " * 12),
        encoding="utf-8",
    )
    _write_json(
        stage_dir / "hardware_profile.json",
        {"cpu": "test", "memory": "64GiB", "gpu": "deferred", "workers": 8},
    )
    _stage_records(stage_dir, ["goal.md", "hardware_profile.json"])

    ready, violations = bridge._validate_completed_stage(stage_dir)

    assert ready is True
    codes = {item.code for item in violations}
    assert "protocol-stage15-gate-count" in codes
    assert "protocol-unregistered-stage15-gate" in codes


def test_clean_stage_five_gate_remains_pending_not_approved(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage-05"
    stage_dir.mkdir()
    records = [
        json.dumps({"title": f"Paper {index}", "verified": True})
        for index in range(4)
    ]
    (stage_dir / "shortlist.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    _stage_records(
        stage_dir,
        ["shortlist.jsonl"],
        status="blocked_approval",
        decision="block",
    )

    assert bridge._validate_completed_stage(stage_dir) == (False, [])


def test_one_native_process_binds_stage_five_then_stage_nine(tmp_path: Path) -> None:
    output = tmp_path / "run"
    source, effective, project_state = _prepare_approved_gate(output, 5)
    records, violations = bridge._bind_ready_gate_handoffs(
        output=output,
        source_config_snapshot=source,
        effective_config=effective,
        project_state_snapshot=project_state,
    )
    assert violations == []
    assert [record["stage"] for record in records] == [5]

    _prepare_approved_gate(output, 9)
    records, violations = bridge._bind_ready_gate_handoffs(
        output=output,
        source_config_snapshot=source,
        effective_config=effective,
        project_state_snapshot=project_state,
    )

    assert violations == []
    assert [record["stage"] for record in records] == [5, 9]
    assert len({record["native_identity"]["run_id"] for record in records}) == 1
    assert len({record["native_identity"]["session_id"] for record in records}) == 1


def test_one_native_process_binds_stage_nine_then_stage_fifteen(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    source, effective, project_state = _prepare_approved_gate(output, 5)
    for stage in (5, 9, 15):
        if stage != 5:
            _prepare_approved_gate(output, stage)
        records, violations = bridge._bind_ready_gate_handoffs(
            output=output,
            source_config_snapshot=source,
            effective_config=effective,
            project_state_snapshot=project_state,
        )
        assert violations == []

    assert [record["stage"] for record in records] == [5, 9, 15]


def test_persistent_bridge_exports_and_replays_all_four_formal_receipts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    receipt_root = tmp_path / "formal-receipts"
    reports: list[dict[str, Any]] = []
    previous_report: dict[str, Any] | None = None
    latest_handoff: dict[str, Any] | None = None

    for stage in bridge.ORDERED_GATES:
        source, effective, project_state = _prepare_approved_gate(output, stage)
        handoffs, violations = bridge._bind_ready_gate_handoffs(
            output=output,
            source_config_snapshot=source,
            effective_config=effective,
            project_state_snapshot=project_state,
        )
        assert violations == []
        assert [record["stage"] for record in handoffs] == [
            candidate for candidate in bridge.ORDERED_GATES if candidate <= stage
        ]
        latest_handoff = handoffs[-1]

        exported = bridge._export_ready_gate_receipts(
            output=output,
            receipt_root=receipt_root,
            handoffs=[latest_handoff],
        )
        bundle = exported[stage]
        receipt = bridge._read_json(bundle / "receipt.v1.json")
        assert receipt["schema_version"] == RECEIPT_SCHEMA
        assert receipt["run"] == {"run_id": "rc-test", "stage": stage}
        previous_report = validate_arc_control_bundle(
            bundle,
            stage,
            previous_report=previous_report,
        )
        reports.append(previous_report)

        first_receipt_hash = bridge._sha256(bundle / "receipt.v1.json")
        assert bridge._export_gate_receipt(
            output=output,
            receipt_root=receipt_root,
            handoff=latest_handoff,
        ) == bundle
        assert bridge._sha256(bundle / "receipt.v1.json") == first_receipt_hash

    assert [report["stage"] for report in validate_arc_control_chain(reports)] == [
        5,
        9,
        15,
        20,
    ]
    assert not list(receipt_root.glob(".*.tmp"))

    assert latest_handoff is not None
    approval = latest_handoff["approval"]
    consumption_path = output / approval["consumption_receipt_path"]
    consumption = bridge._read_json(consumption_path)
    response_snapshot = output / consumption["response_path"]
    replay_challenge = {
        "stage": consumption["stage"],
        "run_id": consumption["run_id"],
        "session_id": consumption["session_id"],
        "waiting_sha256": consumption["waiting_sha256"],
        "preapproval_checkpoint_sha256": consumption[
            "preapproval_checkpoint_sha256"
        ],
        "nonce": consumption["nonce"],
    }
    with pytest.raises(ValueError, match="replay"):
        bridge._claim_operator_response(
            output=output,
            response_path=response_snapshot,
            response=consumption["response"],
            response_sha256=consumption["response_sha256"],
            challenge=replay_challenge,
        )


def test_approved_gate_artifact_mutation_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _add_approved_gate_handoff(output, 5)
    shortlist = output / "stage-05" / "shortlist.jsonl"
    shortlist.write_text(
        shortlist.read_text(encoding="utf-8")
        + json.dumps({"title": "unreviewed addition", "verified": False})
        + "\n",
        encoding="utf-8",
    )

    _, violations = bridge._validate_gate_handoffs(output)

    assert any("artifact set or hash has changed" in item.detail for item in violations)


def test_duplicate_gate_approval_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _add_approved_gate_handoff(output, 5)
    _append_jsonl(
        output / "hitl" / "interventions.jsonl",
        {
            "id": "duplicate-approval",
            "type": "approve",
            "stage": 5,
            "human_input": {"action": "approve"},
            "pause_reason": "gate_approval",
            "accepted": True,
        },
    )

    _, violations = bridge._validate_gate_handoffs(output)

    assert any("exactly one human action" in item.detail for item in violations)


def test_gate_approval_requires_checkpoint_before_input_is_forwarded(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _add_approved_gate_handoff(output, 5)
    (output / "checkpoint.json").unlink()

    with pytest.raises(FileNotFoundError):
        bridge._checkpoint_binding_for_gate(output, 5)


def test_wrong_gate_next_stage_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    handoff = _add_approved_gate_handoff(output, 5)
    record = json.loads(handoff.read_text(encoding="utf-8"))
    record["next_stage"] = 7
    record["next_stage_name"] = "SYNTHESIS"
    _write_json(handoff, record)

    _, violations = bridge._validate_gate_handoffs(output)

    assert any("incorrect next-stage entrypoint" in item.detail for item in violations)


def test_native_run_or_session_change_between_gates_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _add_approved_gate_handoff(output, 5)
    _add_approved_gate_handoff(
        output,
        9,
        run_id="rc-restarted",
        session_id="session-restarted",
    )

    _, violations = bridge._validate_gate_handoffs(output)

    assert any(
        "run_id/session_id changed between approved gates" in item.detail
        or "current native HITL run/session identity has changed" in item.detail
        for item in violations
    )


def test_non_gate_stage_run_id_change_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _add_approved_gate_handoff(output, 5)
    stage_dir = output / "stage-06"
    stage_dir.mkdir()
    (stage_dir / "extraction.md").write_text("evidence " * 20, encoding="utf-8")
    _stage_records(
        stage_dir,
        ["extraction.md"],
        run_id="rc-restarted",
    )

    _, violations = bridge._validate_gate_handoffs(output)

    assert any(item.code == "native-run-identity-invalid" for item in violations)


def test_bridge_rejects_official_resume_that_would_change_native_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="new run_id and HITL session_id"):
        bridge._build_arc_command(
            python=tmp_path / "python.exe",
            effective_config=tmp_path / "effective.yaml",
            output=tmp_path / "run",
            to_stage="RESEARCH_DECISION",
            resume=True,
        )


def test_continuous_arc_run_must_reach_all_four_gates() -> None:
    with pytest.raises(ValueError, match="Stage 20 or later"):
        bridge._validate_continuous_target(15)

    bridge._validate_continuous_target(20)


def test_exact_process_cleanup_is_leaf_first_and_skips_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = bridge.ProcessIdentity(100, 1, "root-start")
    child = bridge.ProcessIdentity(101, 100, "child-start")
    grandchild = bridge.ProcessIdentity(102, 101, "grandchild-start")
    unrelated = bridge.ProcessIdentity(999, 1, "unrelated-start")
    live = {item.pid: item for item in (root, child, grandchild, unrelated)}
    tracker = bridge.ProcessTreeTracker(
        root_pid=100,
        identities={item.pid: item for item in (root, child, grandchild)},
        captured_parent={100: 1, 101: 100, 102: 101},
    )
    signaled: list[int] = []

    monkeypatch.setattr(bridge, "_current_process_table", lambda: dict(live))

    def fake_signal(identity: bridge.ProcessIdentity, _sig: object) -> bool:
        signaled.append(identity.pid)
        live.pop(identity.pid, None)
        return True

    monkeypatch.setattr(bridge, "_signal_exact_process", fake_signal)

    class FakeProcess:
        def wait(self, timeout: float) -> int:
            return 0

    report = bridge._terminate_process_tree(FakeProcess(), tracker, timeout=0.01)  # type: ignore[arg-type]

    assert signaled == [102, 101, 100]
    assert report.targeted_leaf_first == (102, 101, 100)
    assert report.terminated == (102, 101, 100)
    assert 999 not in signaled

    reused = bridge.ProcessIdentity(101, 1, "different-start")
    table = {100: root, 101: reused}
    assert [item.pid for item in tracker.leaf_first(table)] == [100]


def test_windows_process_snapshot_permission_retry_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {123: bridge.ProcessIdentity(123, 1, "start")}
    attempts = 0
    sleeps: list[float] = []

    def snapshot() -> dict[int, bridge.ProcessIdentity]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient snapshot denial")
        return expected

    monkeypatch.setattr(bridge.os, "name", "nt")
    monkeypatch.setattr(bridge, "_windows_process_table", snapshot)
    monkeypatch.setattr(bridge.time, "sleep", sleeps.append)

    assert bridge._current_process_table() == expected
    assert attempts == 3
    assert sleeps == [0.05, 0.1]


def test_project_state_binds_git_dirty_content_and_prompt_hashes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=project, check=True
    )
    source = project / "config.yaml"
    prompt = project / "protocol.md"
    source.write_text("config: frozen\n", encoding="utf-8")
    prompt.write_text("protocol version one\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=project, check=True)
    effective = {
        "prompts": {"custom_file": str(prompt), "extra_prompts": {}},
    }

    original = bridge._project_state(
        project,
        source_config=source,
        effective_loaded=effective,
    )
    snapshot = tmp_path / "run" / "control" / "project-state.v1.json"
    bridge._write_or_verify_project_state(snapshot, original, resume=False)

    prompt.write_text("protocol version two\n", encoding="utf-8")
    changed = bridge._project_state(
        project,
        source_config=source,
        effective_loaded=effective,
    )

    assert original["project_git_dirty"] is False
    bridge._require_clean_project_state(original)
    assert changed["project_git_dirty"] is True
    assert changed["state_sha256"] != original["state_sha256"]
    with pytest.raises(ValueError, match="clean Git worktree"):
        bridge._require_clean_project_state(changed)
    with pytest.raises(ValueError, match="drifted"):
        bridge._write_or_verify_project_state(snapshot, changed, resume=True)


def test_arc_checkout_cleanliness_rejects_tracked_and_untracked_changes(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "arc-checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=checkout, check=True
    )
    tracked = checkout / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)

    bridge._require_clean_git_worktree(checkout, label="official ARC checkout")

    untracked = checkout / "cache.txt"
    untracked.write_text("cache\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official ARC checkout.*clean Git worktree"):
        bridge._require_clean_git_worktree(checkout, label="official ARC checkout")
    untracked.unlink()

    tracked.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official ARC checkout.*clean Git worktree"):
        bridge._require_clean_git_worktree(checkout, label="official ARC checkout")
    subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
    with pytest.raises(ValueError, match="official ARC checkout.*clean Git worktree"):
        bridge._require_clean_git_worktree(checkout, label="official ARC checkout")


def test_invalidation_binds_original_snapshot_and_effective_hashes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    control = output / "control"
    control.mkdir(parents=True)
    source = tmp_path / "source.yaml"
    source.write_text("source: true\n", encoding="utf-8")
    snapshot = control / "source-config.yaml"
    snapshot.write_bytes(source.read_bytes())
    effective = control / "effective-config.yaml"
    effective.write_text("effective: true\n", encoding="utf-8")
    status = output / "bridge.status.v2.json"
    _write_json(status, {"state": "running"})

    invalidation = bridge._write_invalidation(
        output=output,
        violations=[bridge.Violation("provider-error-http-429", "provider failure")],
        process_pid=123,
        source_config=source,
        source_config_snapshot=snapshot,
        effective_config=effective,
        status_path=status,
        command=["python", "-m", "researchclaw.cli"],
    )
    record = json.loads(invalidation.read_text(encoding="utf-8"))

    assert record["source_config"]["observed_sha256"] == bridge._sha256(snapshot)
    assert record["source_config"]["snapshot_sha256"] == bridge._sha256(snapshot)
    assert record["effective_config"]["sha256"] == bridge._sha256(effective)
    assert record["resume_permitted"] is False
