from pathlib import Path

from ecgcert.execution.manifest import ExperimentManifest


ROOT = Path(__file__).resolve().parents[1]


def _option(node, name: str) -> str:
    index = node.command.index(name)
    return node.command[index + 1]


def _nodes():
    manifest = ExperimentManifest.from_path(ROOT / "scripts" / "experiment_manifest.yaml")
    return {node.id: node for node in manifest.select("icassp")}


def test_each_arc_gate_has_wait_review_forward_and_final_handoff() -> None:
    nodes = _nodes()
    for stage in (5, 9, 15, 20):
        waiting = nodes[f"arc_stage{stage}_waiting"]
        forward = nodes[f"arc_stage{stage}_forward"]
        final = nodes[f"arc_stage{stage}_control"]
        assert _option(waiting, "--phase") == "waiting"
        assert _option(final, "--phase") == "final"
        assert f"stage{stage}_review" in forward.deps
        assert f"arc_stage{stage}_waiting" in forward.deps
        assert f"arc_stage{stage}_forward" in final.deps


def test_later_waiting_and_final_receipts_bind_prior_formal_report() -> None:
    nodes = _nodes()
    for stage, previous in ((9, 5), (15, 9), (20, 15)):
        prior_report = f"artifacts/control/arc/stage{previous}/report.v1.json"
        for node_id in (f"arc_stage{stage}_waiting", f"arc_stage{stage}_control"):
            node = nodes[node_id]
            assert _option(node, "--previous-report") == prior_report
            assert prior_report in node.inputs
            assert f"arc_stage{previous}_control" in node.deps


def test_stage9_and_stage15_native_arc_are_held_until_downstream_evidence() -> None:
    nodes = _nodes()
    assert "meta_analysis" in nodes["arc_stage9_forward"].deps
    assert "stage20_review_draft" in nodes["arc_stage15_forward"].deps
    assert "arc_stage9_control" not in nodes["meta_analysis"].deps
    assert "arc_stage15_control" not in nodes["stage20_review_draft"].deps
    assert "arc_stage20_control" in nodes["compile_submission"].deps
    assert "stage20_review" in nodes["compile_submission"].deps


def test_stage15_gate_only_reviews_existing_evidence_at_native_pause() -> None:
    nodes = _nodes()
    waiting = nodes["arc_stage15_waiting"]
    gate = nodes["stage15_gate"]
    assert waiting.resource.gpus == 0
    assert gate.resource.gpus == 0
    assert gate.command[1] == "experiments/meta_analysis_v3.py"
    assert _option(gate, "--mode") == "stage15"
    assert set(gate.deps) == {"meta_analysis", "arc_stage15_waiting"}
    assert set(gate.inputs) == {
        "artifacts/primary/meta_analysis",
        "artifacts/control/arc/stage15/waiting.v1.json",
    }
