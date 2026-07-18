from pathlib import Path

import pytest

from ecgcert.execution.manifest import ExperimentManifest, ManifestError


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "scripts" / "experiment_manifest.yaml"

PRIMARY_IDS = {
    "ptbxl_manifest",
    "chapman_manifest",
    "cpsc_manifest",
    "public_baseline_checkouts",
    "arc_stage5_control",
    "stage5_gate",
    "stage5_review",
    "arc_stage9_control",
    "stage9_gate",
    "stage9_review",
    "robust_rank_maps",
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
    "arc_stage15_control",
    "stage15_gate",
    "stage15_review",
    "primary_figures",
    "claim_sync",
    "compile_submission",
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


def test_manifest_requires_every_field():
    bad = _node()
    del bad["timeout"]
    with pytest.raises(ManifestError, match="missing fields"):
        ExperimentManifest.from_dict({"schema_version": 1, "nodes": [bad]})


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
    assert all(node.profile == ("icassp", "extended") for node in selected)
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
    assert by_id["robust_rank_maps"].command[1] == "experiments/robust_maps_v3.py"
    assert "stage9_review" in by_id["robust_rank_maps"].deps
    assert by_id["reconstruction_candidates"].command[1] == (
        "experiments/reconstruction_candidates_v3.py"
    )
    assert "--release" in by_id["reconstruction_candidates"].command
    assert by_id["official_baseline_preparation"].command[1] == (
        "scripts/prepare_official_baselines_v3.py"
    )
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
        control = by_id[f"arc_stage{stage}_control"]
        assert control.command[1] == "scripts/wait_for_arc_control.py"
        assert _option(control, "--stage") == str(stage)
        assert _option(control, "--bundle") == f"artifacts/gates/arc-stage{stage}"
        assert not any(value.startswith("artifacts/gates/") for value in control.inputs)
        gate = by_id[f"stage{stage}_gate"]
        assert _option(gate, "--arc-control") in gate.inputs
        assert f"arc_stage{stage}_control" in gate.deps

    for stage in (5, 9, 15, 20):
        review = by_id[f"stage{stage}_review"]
        assert _option(review, "--public-key") == "security/reviewer_ed25519.pub"
        assert "security/reviewer_ed25519.pub" in review.inputs
        # The approval remains a controlled dynamic inbox artifact rather than
        # a static DAG input; its SHA/signature are embedded by the review node.
        assert _option(review, "--approval").startswith("artifacts/gates/")
        assert not any(value.startswith("artifacts/gates/") for value in review.inputs)

    benchmark_ids = set(BENCHMARK_METHODS)
    for node_id, method in BENCHMARK_METHODS.items():
        node = by_id[node_id]
        assert node.command[1] == "experiments/reconstruction_benchmark_v3.py"
        assert _option(node, "--method") == method
        assert {"ptbxl_manifest", "robust_rank_maps", "reconstruction_tuning"} <= set(
            node.deps
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

    meta = by_id["meta_analysis"]
    assert meta.command[1] == "experiments/meta_analysis_v3.py"
    assert _option(meta, "--bootstrap-replicates") == "2000"
    assert "--release" in meta.command
    assert benchmark_ids | {
        "robust_rank_maps",
        "zero_transfer_chapman",
        "zero_transfer_cpsc",
    } <= set(meta.deps)
    stage15 = by_id["stage15_gate"]
    assert stage15.command[1] == "experiments/meta_analysis_v3.py"
    assert _option(stage15, "--mode") == "stage15"
    assert set(stage15.deps) == {"meta_analysis", "arc_stage15_control"}

    figures = by_id["primary_figures"]
    assert figures.command[1] == "experiments/paper_figures_v3.py"
    assert {"meta_analysis", "stage15_review", "zero_transfer_chapman", "zero_transfer_cpsc"} <= set(
        figures.deps
    )
    assert {"stage15_review", "primary_figures"} <= set(by_id["claim_sync"].deps)
    assert _option(by_id["claim_sync"], "--reviewer-public-key") == (
        "security/reviewer_ed25519.pub"
    )
    assert {"claim_sync", "primary_figures"} <= set(by_id["compile_submission"].deps)
    stage20 = by_id["stage20_gate"]
    assert stage20.command[1] == "experiments/stage_gates_v3.py"
    assert _option(stage20, "--mode") == "stage20"
    assert _option(stage20, "--reviewer-public-key") == "security/reviewer_ed25519.pub"
    assert {
        "stage15_review", "claim_sync", "compile_submission", "arc_stage20_control"
    } <= set(stage20.deps)
    assert by_id["stage20_review"].deps == ("stage20_gate",)


def test_canonical_extended_profile_adds_only_prespecified_sensitivities():
    manifest = _canonical()
    selected = manifest.select("extended")
    by_id = {node.id: node for node in selected}

    assert set(by_id) - PRIMARY_IDS == EXTENDED_ONLY_IDS
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
    }
    for node_id, sensitivity in sensitivity_specs.items():
        node = by_id[node_id]
        assert node.command[1] == "experiments/robust_maps_v3.py"
        assert _option(node, "--sensitivity") == sensitivity
        assert _option(node, "--n-bootstrap") == "2000"
        assert "--release" in node.command
        assert {"ptbxl_manifest", "robust_rank_maps"} <= set(node.deps)

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
