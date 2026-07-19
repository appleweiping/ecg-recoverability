from pathlib import Path

import pytest

from ecgcert.execution.manifest import ExperimentManifest, ManifestError


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "scripts" / "experiment_manifest.yaml"

PRIMARY_IDS = {
    "repository_secret_scan",
    "ptbxl_manifest",
    "chapman_manifest",
    "cpsc_manifest",
    "public_baseline_checkouts",
    "arc_stage5_waiting",
    "arc_stage5_forward",
    "arc_stage5_control",
    "stage5_gate",
    "stage5_review",
    "arc_stage9_waiting",
    "arc_stage9_forward",
    "arc_stage9_control",
    "stage9_gate",
    "stage9_review",
    "cuda_contract_tests",
    "robust_rank_maps",
    "reconstruction_training_inclusion",
    "reconstruction_candidates",
    "official_baseline_preparation",
    "reconstruction_tuning",
    "benchmark_lowrank",
    "benchmark_ridge",
    "benchmark_masked_unet",
    "benchmark_imputeecg",
    "benchmark_ecgrecover",
    "zero_transfer_chapman",
    "zero_transfer_cpsc",
    "meta_analysis",
    "arc_stage15_waiting",
    "arc_stage15_forward",
    "arc_stage15_control",
    "stage15_gate",
    "stage15_review",
    "primary_figures",
    "claim_sync",
    "stage20_review_draft",
    "compile_submission",
    "arc_stage20_waiting",
    "arc_stage20_forward",
    "arc_stage20_control",
    "stage20_gate",
    "stage20_review",
}

BENCHMARK_METHODS = {
    "benchmark_lowrank": "lowrank",
    "benchmark_ridge": "ridge",
    "benchmark_masked_unet": "masked-unet",
    "benchmark_imputeecg": "imputeecg",
    "benchmark_ecgrecover": "ecgrecover",
}

EXTENDED_ONLY_IDS = {
    "sensitivity_p_wave",
    "sensitivity_100hz",
    "sensitivity_delineator",
    "sensitivity_raw12",
    "sensitivity_sample_cap_80",
    "sensitivity_diagnostic_subgroup_prediction",
    "sensitivity_diagnosis_norm",
    "sensitivity_diagnosis_mi",
    "sensitivity_diagnosis_sttc",
    "sensitivity_diagnosis_cd",
    "sensitivity_diagnosis_hyp",
    "cohort_maps_chapman",
    "cohort_maps_cpsc",
}

FORBIDDEN_PRIMARY_TOKENS = {
    "active_selection",
    "certificate_floor",
    "conformal",
    "cqr",
    "diffusion",
    "fabrication",
    "minimax",
    "st_safety",
    "synthetic",
    "tier2_conformal",
}


def _node(node_id="a", deps=None, outputs=None):
    return {
        "id": node_id,
        "profile": ["icassp", "extended", "legacy"],
        "command": ["{python}", "task.py"],
        "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
        "deps": deps or [],
        "inputs": [],
        "outputs": outputs or [f"artifacts/{node_id}.json"],
        "timeout": 30,
        "seed": 0,
    }


def test_manifest_topological_and_profile_selection():
    manifest = ExperimentManifest.from_dict({
        "schema_version": 1,
        "nodes": [_node("b", ["a"]), _node("a")],
    })
    assert [node.id for node in manifest.select("icassp")] == ["a", "b"]
    assert manifest.nodes[0].config_sha256() == manifest.nodes[0].config_sha256()


def test_resource_selection_includes_dependency_closure():
    cpu = _node("cpu")
    gpu = _node("gpu", deps=["cpu"])
    gpu["resource"] = {"kind": "gpu", "cpus": 2, "memory_gb": 4, "gpus": 1}
    manifest = ExperimentManifest.from_dict({"schema_version": 1, "nodes": [gpu, cpu]})
    assert [node.id for node in manifest.select("icassp", "gpu")] == ["cpu", "gpu"]


def test_manifest_rejects_cycle_duplicate_output_and_unsafe_path():
    with pytest.raises(ManifestError, match="cycle"):
        ExperimentManifest.from_dict({
            "schema_version": 1,
            "nodes": [_node("a", ["b"]), _node("b", ["a"])],
        })
    with pytest.raises(ManifestError, match="duplicate output"):
        ExperimentManifest.from_dict({
            "schema_version": 1,
            "nodes": [_node("a", outputs=["x.json"]), _node("b", outputs=["x.json"])],
        })
    bad = _node()
    bad["inputs"] = ["../secret"]
    with pytest.raises(ManifestError, match="unsafe path"):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [bad]})


@pytest.mark.parametrize(
    "unsafe_path",
    [
        ".",
        "C:/outside/data",
        r"C:\outside\data",
        "//server/share/data",
        "artifacts/result.json:alternate-stream",
        "artifacts/result\x00.json",
    ],
)
def test_manifest_rejects_cross_platform_path_escape_forms(unsafe_path):
    bad = _node()
    bad["inputs"] = [unsafe_path]

    with pytest.raises(ManifestError, match="unsafe path"):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [bad]})


def test_manifest_requires_dependency_for_input_inside_producer_directory():
    producer = _node("producer", outputs=["artifacts/models"])
    consumer = _node("consumer")
    consumer["inputs"] = ["artifacts/models/lowrank/model.npz"]

    with pytest.raises(ManifestError, match="not a dependency"):
        ExperimentManifest.from_dict(
            {"schema_version": 1, "nodes": [producer, consumer]}
        )

    consumer["deps"] = ["producer"]
    manifest = ExperimentManifest.from_dict(
        {"schema_version": 1, "nodes": [producer, consumer]}
    )
    assert manifest.by_id()["consumer"].deps == ("producer",)


def test_manifest_rejects_nested_outputs_and_contained_self_input():
    with pytest.raises(ManifestError, match="nested outputs"):
        ExperimentManifest.from_dict(
            {
                "schema_version": 1,
                "nodes": [
                    _node("parent", outputs=["artifacts/models"]),
                    _node("child", outputs=["artifacts/models/lowrank"]),
                ],
            }
        )

    nested = _node(
        "same_node",
        outputs=["artifacts/models", "artifacts/models/lowrank"],
    )
    with pytest.raises(ManifestError, match="nested outputs"):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [nested]})

    self_overlap = _node("self", outputs=["artifacts/models"])
    self_overlap["inputs"] = ["artifacts/models/seed-0/model.npz"]
    with pytest.raises(ManifestError, match="inputs and outputs overlap"):
        ExperimentManifest.from_dict(
            {"schema_version": 1, "nodes": [self_overlap]}
        )


def test_manifest_path_ownership_is_segment_aware():
    producer = _node("producer", outputs=["artifacts/model"])
    consumer = _node("consumer")
    consumer["inputs"] = ["artifacts/models/independent.json"]

    manifest = ExperimentManifest.from_dict(
        {"schema_version": 1, "nodes": [producer, consumer]}
    )
    assert manifest.by_id()["consumer"].deps == ()


def test_manifest_requires_every_field():
    bad = _node()
    del bad["timeout"]
    with pytest.raises(ManifestError, match="missing fields"):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [bad]})


def test_manifest_late_controls_are_declared_external_inbox_paths():
    node = _node()
    node["command"].extend(("--approval", "artifacts/gates/stage5.approval.json"))
    node["late_control_inputs"] = ["artifacts/gates/stage5.approval.json"]
    manifest = ExperimentManifest.from_dict({"schema_version": 1, "nodes": [node]})
    parsed = manifest.nodes[0]
    first = parsed.config_sha256(
        late_control_inputs_sha256={
            "artifacts/gates/stage5.approval.json": "a" * 64
        }
    )
    second = parsed.config_sha256(
        late_control_inputs_sha256={
            "artifacts/gates/stage5.approval.json": "b" * 64
        }
    )
    assert first != second


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda node: node.update(late_control_inputs=["../approval.json"]), "unsafe"),
        (lambda node: node.update(late_control_inputs=["data/approval.json"]), "confined"),
        (
            lambda node: node.update(
                late_control_inputs=["artifacts/gates/unreferenced.json"]
            ),
            "exact command token",
        ),
        (
            lambda node: (
                node["command"].append("artifacts/gates/control.json"),
                node.update(inputs=["artifacts/gates/control.json"]),
                node.update(late_control_inputs=["artifacts/gates/control.json"]),
            ),
            "inputs and late_control_inputs overlap",
        ),
    ],
)
def test_manifest_rejects_unsafe_or_ambiguous_late_controls(mutate, message):
    node = _node()
    mutate(node)
    with pytest.raises(ManifestError, match=message):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [node]})


def test_manifest_rejects_undeclared_gate_command_and_dag_produced_late_control():
    undeclared = _node()
    undeclared["command"].append("artifacts/gates/control.json")
    with pytest.raises(ManifestError, match="undeclared"):
        ExperimentManifest.from_dict(
            {"schema_version": 1, "nodes": [undeclared]}
        )

    producer = _node("producer", outputs=["artifacts/gates/control.json"])
    consumer = _node("consumer")
    consumer["command"].append("artifacts/gates/control.json")
    consumer["late_control_inputs"] = ["artifacts/gates/control.json"]
    with pytest.raises(ManifestError, match="external inbox"):
        ExperimentManifest.from_dict(
            {"schema_version": 1, "nodes": [producer, consumer]}
        )


def _canonical():
    return ExperimentManifest.from_path(CANONICAL_MANIFEST)


def _node_text(node):
    return " ".join((node.id, *node.command, *node.inputs, *node.outputs)).lower()


def _option(node, name):
    position = node.command.index(name)
    return node.command[position + 1]


def test_canonical_icassp_profile_is_the_frozen_v3_primary_line():
    manifest = _canonical()
    selected = manifest.select("icassp")
    by_id = {node.id: node for node in selected}

    assert set(by_id) == PRIMARY_IDS
    assert {node.resource.kind for node in selected} == {"cpu", "gpu", "paper"}
    assert by_id["compile_submission"].profile == ("icassp",)
    assert all(
        node.profile == ("icassp", "extended")
        for node in selected
        if node.id != "compile_submission"
    )
    assert all(output.startswith("artifacts/") for node in selected for output in node.outputs)
    assert all(node.resource.cpus <= 10 for node in selected)
    assert all(
        not output.startswith(("results/", "paper/"))
        for node in selected
        for output in node.outputs
    )

    primary_text = "\n".join(_node_text(node) for node in selected)
    assert not (FORBIDDEN_PRIMARY_TOKENS & set(primary_text.split()))
    for token in FORBIDDEN_PRIMARY_TOKENS:
        assert token not in primary_text

    for manifest_id in ("ptbxl_manifest", "chapman_manifest", "cpsc_manifest"):
        assert by_id[manifest_id].command[1] == "scripts/prepare_data_manifests.py"
    secret_scan = by_id["repository_secret_scan"]
    assert secret_scan.command[1] == "scripts/scan_repository_secrets.py"
    assert secret_scan.deps == ()
    assert secret_scan.outputs == (
        "artifacts/control/security/repository-secret-scan.v1.json",
    )
    assert "repository_secret_scan" in by_id["stage9_gate"].deps
    assert "--repository-secret-scan" in by_id["stage9_gate"].command
    assert by_id["robust_rank_maps"].command[1] == "experiments/robust_maps_v3.py"
    assert _option(by_id["robust_rank_maps"], "--max-per-record") == "40"
    assert _option(by_id["robust_rank_maps"], "--sampling-seed") == "20260719"
    assert "stage9_review" in by_id["robust_rank_maps"].deps
    assert by_id["reconstruction_candidates"].command[1] == (
        "experiments/reconstruction_candidates_v3.py"
    )
    training_inclusion = by_id["reconstruction_training_inclusion"]
    assert training_inclusion.command[1] == "scripts/prepare_training_inclusion_v3.py"
    assert set(training_inclusion.deps) == {"ptbxl_manifest", "stage9_review"}
    assert training_inclusion.outputs == (
        "artifacts/primary/reconstruction_training_inclusion",
    )
    assert "--release" in by_id["reconstruction_candidates"].command
    assert "reconstruction_training_inclusion" in by_id[
        "reconstruction_candidates"
    ].deps
    assert by_id["official_baseline_preparation"].command[1] == (
        "scripts/prepare_official_baselines_v3.py"
    )
    cuda_contract = by_id["cuda_contract_tests"]
    assert cuda_contract.command[1] == "scripts/run_cuda_contract_tests.py"
    assert cuda_contract.resource.kind == "gpu"
    assert set(cuda_contract.deps) == {"public_baseline_checkouts", "stage9_review"}
    assert "artifacts/upstreams" in cuda_contract.inputs
    assert "cuda_contract_tests" in by_id["reconstruction_candidates"].deps
    assert "cuda_contract_tests" in by_id["official_baseline_preparation"].deps
    assert by_id["reconstruction_tuning"].command[1] == (
        "experiments/tune_reconstructors_v3.py"
    )
    assert {"reconstruction_candidates", "official_baseline_preparation"} <= set(
        by_id["reconstruction_tuning"].deps
    )

    for stage in (5, 9):
        gate = by_id[f"stage{stage}_gate"]
        assert gate.command[1] == "experiments/stage_gates_v3.py"
        assert _option(gate, "--mode") == f"stage{stage}"
        assert by_id[f"stage{stage}_review"].command[1] == "scripts/wait_for_stage_review.py"

    for stage in (5, 9, 15, 20):
        waiting = by_id[f"arc_stage{stage}_waiting"]
        control = by_id[f"arc_stage{stage}_control"]
        forward = by_id[f"arc_stage{stage}_forward"]
        assert waiting.command[1] == "scripts/wait_for_arc_control.py"
        assert _option(waiting, "--phase") == "waiting"
        assert _option(waiting, "--bundle") == f"artifacts/gates/arc-stage{stage}-waiting"
        assert waiting.late_control_inputs == (
            f"artifacts/gates/arc-stage{stage}-waiting",
        )
        assert control.command[1] == "scripts/wait_for_arc_control.py"
        assert _option(control, "--phase") == "final"
        assert _option(control, "--stage") == str(stage)
        assert _option(control, "--bundle") == f"artifacts/gates/arc-stage{stage}"
        assert control.late_control_inputs == (
            f"artifacts/gates/arc-stage{stage}",
        )
        assert forward.command[1] == "scripts/forward_arc_stage_review.py"
        assert f"stage{stage}_review" in forward.deps
        assert f"arc_stage{stage}_forward" in control.deps
        gate = by_id[f"stage{stage}_gate"]
        assert _option(gate, "--arc-control") in gate.inputs
        assert f"arc_stage{stage}_waiting" in gate.deps

    for stage in (5, 9, 15, 20):
        review = by_id[f"stage{stage}_review"]
        assert _option(review, "--public-key") == "security/reviewer_ed25519.pub"
        assert "security/reviewer_ed25519.pub" in review.inputs
        approval = _option(review, "--approval")
        assert approval.startswith("artifacts/gates/")
        assert not any(value.startswith("artifacts/gates/") for value in review.inputs)
        assert review.late_control_inputs == (approval,)

    expected_late_nodes = {
        *(f"arc_stage{stage}_waiting" for stage in (5, 9, 15, 20)),
        *(f"arc_stage{stage}_control" for stage in (5, 9, 15, 20)),
        *(f"stage{stage}_review" for stage in (5, 9, 15, 20)),
    }
    assert {
        node.id for node in selected if node.late_control_inputs
    } == expected_late_nodes

    benchmark_ids = set(BENCHMARK_METHODS)
    for node_id, method in BENCHMARK_METHODS.items():
        node = by_id[node_id]
        assert node.command[1] == "experiments/reconstruction_benchmark_v3.py"
        assert _option(node, "--method") == method
        assert {"ptbxl_manifest", "robust_rank_maps", "reconstruction_tuning"} <= set(
            node.deps
        )
        assert "reconstruction_training_inclusion" in node.deps
        assert "artifacts/primary/reconstruction_training_inclusion" in node.inputs
        assert _option(node, "--training-inclusion") == (
            "artifacts/primary/reconstruction_training_inclusion/"
            "training_inclusion.v1.json"
        )
        assert "--tuning-config" in node.command and "--release" in node.command
    for node_id in ("benchmark_imputeecg", "benchmark_ecgrecover"):
        assert "public_baseline_checkouts" in by_id[node_id].deps

    transfer_specs = {
        "zero_transfer_chapman": ("chapman", "chapman_manifest"),
        "zero_transfer_cpsc": ("cpsc2018", "cpsc_manifest"),
    }
    for node_id, (cohort, cohort_manifest) in transfer_specs.items():
        node = by_id[node_id]
        assert node.command[1] == "experiments/external_validation_v3.py"
        assert _option(node, "--mode") == "zero-transfer"
        assert _option(node, "--cohort") == cohort
        assert _option(node, "--n-bootstrap") == "2000"
        assert "--release" in node.command
        assert benchmark_ids | {"ptbxl_manifest", cohort_manifest, "robust_rank_maps"} <= set(
            node.deps
        )
        assert "public_baseline_checkouts" in node.deps
        assert "artifacts/upstreams" in node.inputs

    meta = by_id["meta_analysis"]
    assert meta.command[1] == "experiments/meta_analysis_v3.py"
    assert _option(meta, "--bootstrap-replicates") == "2000"
    assert _option(meta, "--ptbxl-manifest") == "artifacts/manifests/ptbxl.json"
    external_manifest_positions = [
        index for index, value in enumerate(meta.command) if value == "--external-manifest"
    ]
    assert [meta.command[index + 1] for index in external_manifest_positions] == [
        "artifacts/manifests/chapman.json",
        "artifacts/manifests/cpsc2018.json",
    ]
    assert "--release" in meta.command
    assert benchmark_ids | {
        "ptbxl_manifest",
        "chapman_manifest",
        "cpsc_manifest",
        "robust_rank_maps",
        "zero_transfer_chapman",
        "zero_transfer_cpsc",
    } <= set(meta.deps)
    assert {
        "artifacts/manifests/ptbxl.json",
        "artifacts/manifests/chapman.json",
        "artifacts/manifests/cpsc2018.json",
    } <= set(meta.inputs)
    stage15 = by_id["stage15_gate"]
    assert stage15.command[1] == "experiments/meta_analysis_v3.py"
    assert _option(stage15, "--mode") == "stage15"
    assert set(stage15.deps) == {"meta_analysis", "arc_stage15_waiting"}

    figures = by_id["primary_figures"]
    assert figures.command[1] == "experiments/paper_figures_v3.py"
    assert {"meta_analysis", "stage15_review", "zero_transfer_chapman", "zero_transfer_cpsc"} <= set(
        figures.deps
    )
    assert {"stage15_review", "primary_figures"} <= set(by_id["claim_sync"].deps)
    assert _option(by_id["claim_sync"], "--reviewer-public-key") == (
        "security/reviewer_ed25519.pub"
    )
    review_draft = by_id["stage20_review_draft"]
    assert review_draft.command[1] == "scripts/compile_stage20_review_draft_v3.py"
    assert {"claim_sync", "primary_figures"} <= set(review_draft.deps)
    assert review_draft.outputs == ("artifacts/paper/stage20_review_draft",)
    final_submission = by_id["compile_submission"]
    assert final_submission.command[1] == "scripts/compile_submission_v3.py"
    assert {
        "claim_sync",
        "primary_figures",
        "stage20_review",
        "arc_stage20_control",
    } <= set(final_submission.deps)
    for option in (
        "--author-declaration",
        "--tool-provenance",
        "--venue-policy",
        "--author-public-key",
        "--author-kit",
        "--arc-receipts-root",
    ):
        assert option in final_submission.command
    stage20 = by_id["stage20_gate"]
    assert stage20.command[1] == "experiments/stage_gates_v3.py"
    assert _option(stage20, "--mode") == "stage20"
    assert _option(stage20, "--reviewer-public-key") == "security/reviewer_ed25519.pub"
    assert {
        "stage15_review", "claim_sync", "stage20_review_draft", "arc_stage20_waiting"
    } <= set(stage20.deps)
    assert _option(stage20, "--review-draft") == (
        "artifacts/paper/stage20_review_draft"
    )
    assert by_id["stage20_review"].deps == ("stage20_gate",)


def test_canonical_extended_profile_adds_only_prespecified_sensitivities():
    manifest = _canonical()
    selected = manifest.select("extended")
    by_id = {node.id: node for node in selected}

    assert set(by_id) - PRIMARY_IDS == EXTENDED_ONLY_IDS
    assert "compile_submission" not in by_id
    assert all(by_id[node_id].profile == ("extended",) for node_id in EXTENDED_ONLY_IDS)
    assert all(output.startswith("artifacts/") for node in selected for output in node.outputs)
    extended_text = "\n".join(_node_text(node) for node in selected)
    for token in FORBIDDEN_PRIMARY_TOKENS:
        assert token not in extended_text

    sensitivity_specs = {
        "sensitivity_p_wave": "p-wave",
        "sensitivity_100hz": "100hz",
        "sensitivity_delineator": "delineator",
        "sensitivity_raw12": "raw12",
        "sensitivity_sample_cap_80": "sample-cap",
    }
    for node_id, sensitivity in sensitivity_specs.items():
        node = by_id[node_id]
        assert node.command[1] == "experiments/robust_maps_v3.py"
        assert _option(node, "--sensitivity") == sensitivity
        assert _option(node, "--n-bootstrap") == "2000"
        assert _option(node, "--sampling-seed") == "20260719"
        assert "--release" in node.command
        assert {"ptbxl_manifest", "robust_rank_maps"} <= set(node.deps)
    assert _option(by_id["sensitivity_sample_cap_80"], "--max-per-record") == "80"

    subgroup = by_id["sensitivity_diagnostic_subgroup_prediction"]
    assert subgroup.command[1] == "experiments/diagnostic_subgroup_v3.py"
    assert _option(subgroup, "--bootstrap-replicates") == "2000"
    assert _option(subgroup, "--seed") == "20260719"
    assert "--release" in subgroup.command
    assert set(subgroup.deps) == {"meta_analysis", "ptbxl_manifest"}
    assert "stage15_gate" not in subgroup.deps

    diagnosis_specs = {
        "sensitivity_diagnosis_norm": "NORM",
        "sensitivity_diagnosis_mi": "MI",
        "sensitivity_diagnosis_sttc": "STTC",
        "sensitivity_diagnosis_cd": "CD",
        "sensitivity_diagnosis_hyp": "HYP",
    }
    for node_id, diagnosis_class in diagnosis_specs.items():
        node = by_id[node_id]
        assert _option(node, "--sensitivity") == "diagnosis"
        assert _option(node, "--diagnosis-class") == diagnosis_class
        assert _option(node, "--n-bootstrap") == "2000"
        assert "--release" in node.command
        assert {"ptbxl_manifest", "robust_rank_maps"} <= set(node.deps)

    for node_id, cohort in (
        ("cohort_maps_chapman", "chapman"),
        ("cohort_maps_cpsc", "cpsc2018"),
    ):
        node = by_id[node_id]
        assert node.command[1] == "experiments/external_validation_v3.py"
        assert _option(node, "--mode") == "cohort-maps"
        assert _option(node, "--cohort") == cohort
        assert _option(node, "--n-bootstrap") == "2000"
        assert "--release" in node.command


def test_canonical_legacy_profile_retains_historical_branches_only():
    manifest = _canonical()
    legacy = manifest.select("legacy")
    legacy_ids = {node.id for node in legacy}

    assert not (legacy_ids & (PRIMARY_IDS | EXTENDED_ONLY_IDS))
    assert all(node.profile == ("legacy",) for node in legacy)
    assert {
        "synthetic",
        "tier2_conformal",
        "st_safety",
        "active_selection",
        "diffusion_train",
        "certificate_floor_diffusion",
    } <= legacy_ids
    assert any(output.startswith("results/") for node in legacy for output in node.outputs)
