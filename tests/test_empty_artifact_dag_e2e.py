from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pandas as pd
from pypdf import PdfReader
import pytest

from ecgcert import lineage
from ecgcert.execution import (
    DAGRunner,
    ExecutionError,
    ExperimentManifest,
    ResultEnvelope,
)
from ecgcert.paper_evidence import FIGURE_ARTIFACTS
from ecgcert.protocol import deep_configuration_panel
from scripts.compile_submission_v3 import (
    paper_artifact_content_sha256,
    paper_artifact_signature_message,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PRODUCER = ROOT / "tests" / "fixtures" / "tiny_icassp_pipeline.py"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sign_paper_input(
    path: Path, value: dict, private_key: Ed25519PrivateKey,
) -> None:
    public_key = private_key.public_key()
    raw_public = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    value.update(
        {
            "status": "VERIFIED",
            "signed_at": "2026-07-19T12:00:00+00:00",
            "signer": "Tiny E2E submitting author",
            "signature_algorithm": "Ed25519",
            "signer_public_key_sha256": hashlib.sha256(raw_public).hexdigest(),
        }
    )
    value["content_sha256"] = paper_artifact_content_sha256(value)
    value["signature_ed25519"] = base64.b64encode(
        private_key.sign(paper_artifact_signature_message(value))
    ).decode("ascii")
    _write_json(path, value)


def _write_signed_policy_inputs(repo: Path) -> None:
    """Create non-scientific, authenticated paper-policy inputs for the tiny DAG."""

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_path = repo / "security" / "author_ed25519.pub"
    public_path.write_bytes(
        public_key.public_bytes(
            serialization.Encoding.OpenSSH,
            serialization.PublicFormat.OpenSSH,
        )
        + b" tiny-e2e-author\n"
    )
    provenance = repo / "paper" / "provenance"
    provenance.mkdir(parents=True)
    author_kit = provenance / "icassp2027-author-kit.fixture.zip"
    kit_members = {
        "kit/spconf.sty": (repo / "paper" / "spconf.sty").read_bytes(),
        "kit/IEEEbib.bst": (repo / "paper" / "IEEEbib.bst").read_bytes(),
        "kit/conference-template.tex": (
            b"% official author-kit template fixture\n\\documentclass{article}\n"
        ),
    }
    with zipfile.ZipFile(author_kit, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in kit_members.items():
            member = zipfile.ZipInfo(name, date_time=(2026, 7, 19, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, content)
    _sign_paper_input(
        provenance / "author_declaration.v1.json",
        {
            "schema_version": "paper-author-declaration-v1",
            "authors": [
                {"name": "Fixture Researcher", "affiliation_ids": ["fixture-lab"]}
            ],
            "affiliations": [
                {"id": "fixture-lab", "text": "Synthetic Signal Laboratory"}
            ],
            "funding_statement": "This synthetic test declares no research funding.",
            "conflict_of_interest_statement": (
                "This synthetic test declares no competing interests."
            ),
            "compliance_statement": (
                "This reproducibility test uses only synthetic records."
            ),
            "facts_verified_by_all_authors": True,
        },
        private_key,
    )
    _sign_paper_input(
        provenance / "tool_provenance.v1.json",
        {
            "schema_version": "paper-tool-provenance-v1",
            "sps_policy_url": (
                "https://signalprocessingsociety.org/publications-resources/"
                "publication-guidelines/policy-on-using-large-language-models-llms"
            ),
            "tools": [
                {
                    "provider": "OpenAI",
                    "product": "Codex",
                    "version": "tiny-e2e-fixture",
                    "uses": ["reproducibility test execution"],
                }
            ],
            "human_verification": {
                "verified_by": "Tiny E2E submitting author",
                "verified_at": "2026-07-19T11:59:00+00:00",
                "citations_checked_against_primary_sources": True,
                "numerical_claims_traced_to_artifacts": True,
                "code_and_statistics_reviewed": True,
                "manuscript_read_and_approved": True,
                "scientific_judgment_retained_by_authors": True,
                "ai_systems_are_not_authors": True,
                "any_submitted_section_entirely_generated_by_ai": False,
                "most_or_significant_manuscript_components_generated_by_ai": False,
                "authors_substantively_rewrote_and_verified_ai_assisted_text": True,
            },
            "autoresearchclaw_formal_receipts": {},
        },
        private_key,
    )
    _sign_paper_input(
        provenance / "venue_policy.v1.json",
        {
            "schema_version": "paper-venue-policy-v1",
            "venue": "ICASSP 2027",
            "policy_url": "https://2027.ieeeicassp.org/about/editorial-policies/",
            "author_kit_url": "https://2027.ieeeicassp.org/author-kit.zip",
            "author_kit_sha256": lineage.artifact_sha256(author_kit),
            "author_kit_status": "OFFICIAL_PUBLISHED",
            "author_kit_members": {
                "style": {
                    "member_path": "kit/spconf.sty",
                    "sha256": hashlib.sha256(kit_members["kit/spconf.sty"]).hexdigest(),
                    "build_target": "spconf.sty",
                },
                "bibliography_style": {
                    "member_path": "kit/IEEEbib.bst",
                    "sha256": hashlib.sha256(
                        kit_members["kit/IEEEbib.bst"]
                    ).hexdigest(),
                    "build_target": "IEEEbib.bst",
                },
                "latex_template": {
                    "member_path": "kit/conference-template.tex",
                    "sha256": hashlib.sha256(
                        kit_members["kit/conference-template.tex"]
                    ).hexdigest(),
                    "build_target": "author_kit/official-template.tex",
                },
            },
            "policy_checked_at": "2026-07-19T11:58:00+00:00",
            "review_model": "single-anonymous",
            "author_identities_visible_to_reviewers": True,
            "technical_page_limit": 4,
            "total_page_limit": 5,
            "page_five_allowed_content": [
                "references",
                "funding acknowledgments",
                "Compliance with Ethical Standards",
            ],
        },
        private_key,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )


def _copy_repository_fixture(
    repo: Path, public_key: Path, *, include_claim_tools: bool = True,
) -> None:
    (repo / "scripts").mkdir(parents=True)
    (repo / "experiments").mkdir()
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "paper" / "auto").mkdir(parents=True)
    (repo / "environments").mkdir()
    (repo / "arc_audit").mkdir()
    (repo / "security").mkdir()
    shutil.copy2(ROOT / ".python-version", repo / ".python-version")
    shutil.copytree(ROOT / "src" / "ecgcert", repo / "src" / "ecgcert")
    shutil.copy2(FIXTURE_PRODUCER, repo / "tests" / "fixtures" / FIXTURE_PRODUCER.name)
    if include_claim_tools:
        shutil.copy2(
            ROOT / "experiments" / "paper_figures_v3.py",
            repo / "experiments" / "paper_figures_v3.py",
        )
        for name in (
            "claim_sync_v3.py",
            "compile_stage20_review_draft_v3.py",
            "compile_submission_v3.py",
        ):
            shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
        for name in (
            "main_v2.tex", "compliance.tex", "refs.bib", "spconf.sty", "IEEEbib.bst",
        ):
            shutil.copy2(ROOT / "paper" / name, repo / "paper" / name)
        shutil.copy2(
            ROOT / "paper" / "auto" / "robust_map_placeholders.tex",
            repo / "paper" / "auto" / "robust_map_placeholders.tex",
        )
        shutil.copy2(
            ROOT / "arc_audit" / "verified_registry.v1.json",
            repo / "arc_audit" / "verified_registry.v1.json",
        )
        _write_signed_policy_inputs(repo)
    shutil.copy2(public_key, repo / "security" / "reviewer_ed25519.pub")
    for name in ("cpu.lock.txt", "gpu.lock.txt"):
        shutil.copy2(ROOT / "environments" / name, repo / "environments" / name)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "tiny claim-bearing ICASSP DAG")


def _node(
    *,
    node_id: str,
    command: list[str],
    deps: list[str],
    inputs: list[str],
    outputs: list[str],
    kind: str = "cpu",
    profiles: list[str] | None = None,
    timeout: int = 300,
) -> dict:
    return {
        "id": node_id,
        "profile": profiles or ["icassp"],
        "command": command,
        "resource": {
            "kind": kind,
            "cpus": 1,
            "memory_gb": 3,
            "gpus": 0,
        },
        "deps": deps,
        "inputs": inputs,
        "outputs": outputs,
        "timeout": timeout,
        "seed": 0,
    }


def _claim_manifest(private_key: Path) -> ExperimentManifest:
    producer = "tests/fixtures/tiny_icassp_pipeline.py"
    nodes = [
        _node(
            node_id="tiny_manifests",
            command=["{python}", producer, "manifests", "--output", "artifacts/manifests"],
            deps=[],
            inputs=[],
            outputs=["artifacts/manifests"],
            profiles=["icassp", "extended", "legacy"],
        ),
        _node(
            node_id="tiny_robust_rank_maps",
            command=[
                "{python}", producer, "map", "--manifest", "artifacts/manifests/ptbxl.json",
                "--output", "artifacts/primary/robust_rank_maps",
            ],
            deps=["tiny_manifests"],
            inputs=["artifacts/manifests"],
            outputs=["artifacts/primary/robust_rank_maps"],
        ),
        _node(
            node_id="tiny_benchmark",
            command=[
                "{python}", producer, "benchmark",
                "--manifest", "artifacts/manifests/ptbxl.json",
                "--rank-maps", "artifacts/primary/robust_rank_maps",
                "--output", "artifacts/primary/reconstruction/tiny",
            ],
            deps=["tiny_manifests", "tiny_robust_rank_maps"],
            inputs=["artifacts/manifests", "artifacts/primary/robust_rank_maps"],
            outputs=["artifacts/primary/reconstruction/tiny"],
        ),
        *[
            _node(
                node_id=f"tiny_external_{cohort}",
                command=[
                    "{python}", producer, "external", "--cohort", cohort,
                    "--manifest", f"artifacts/manifests/{cohort}.json",
                    "--benchmark", "artifacts/primary/reconstruction/tiny",
                    "--output", f"artifacts/primary/zero_transfer/{cohort}",
                ],
                deps=["tiny_manifests", "tiny_benchmark"],
                inputs=["artifacts/manifests", "artifacts/primary/reconstruction/tiny"],
                outputs=[f"artifacts/primary/zero_transfer/{cohort}"],
            )
            for cohort in ("chapman", "cpsc2018")
        ],
        _node(
            node_id="tiny_meta_analysis",
            command=[
                "{python}", producer, "meta",
                "--rank-maps", "artifacts/primary/robust_rank_maps",
                "--benchmark", "artifacts/primary/reconstruction/tiny",
                "--chapman", "artifacts/primary/zero_transfer/chapman",
                "--cpsc", "artifacts/primary/zero_transfer/cpsc2018",
                "--output", "artifacts/primary/meta_analysis",
            ],
            deps=[
                "tiny_robust_rank_maps", "tiny_benchmark",
                "tiny_external_chapman", "tiny_external_cpsc2018",
            ],
            inputs=[
                "artifacts/primary/robust_rank_maps",
                "artifacts/primary/reconstruction/tiny",
                "artifacts/primary/zero_transfer/chapman",
                "artifacts/primary/zero_transfer/cpsc2018",
            ],
            outputs=["artifacts/primary/meta_analysis"],
        ),
        _node(
            node_id="tiny_stage15_review",
            command=[
                "{python}", producer, "stage15",
                "--meta", "artifacts/primary/meta_analysis",
                "--private-key", str(private_key.resolve()),
                "--public-key", "security/reviewer_ed25519.pub",
                "--output", "artifacts/primary/stage15_review",
            ],
            deps=["tiny_meta_analysis"],
            inputs=["artifacts/primary/meta_analysis", "security/reviewer_ed25519.pub"],
            outputs=["artifacts/primary/stage15_review"],
            kind="paper",
        ),
        _node(
            node_id="primary_figures",
            command=[
                "{python}", "experiments/paper_figures_v3.py",
                "--rank-maps", "artifacts/primary/robust_rank_maps",
                "--meta-analysis", "artifacts/primary/meta_analysis",
                "--chapman", "artifacts/primary/zero_transfer/chapman",
                "--cpsc", "artifacts/primary/zero_transfer/cpsc2018",
                "--stage15", "artifacts/primary/stage15_review",
                "--output-dir", "artifacts/paper/figures",
            ],
            deps=[
                "tiny_robust_rank_maps", "tiny_meta_analysis",
                "tiny_external_chapman", "tiny_external_cpsc2018",
                "tiny_stage15_review",
            ],
            inputs=[
                "artifacts/primary/robust_rank_maps",
                "artifacts/primary/meta_analysis",
                "artifacts/primary/zero_transfer/chapman",
                "artifacts/primary/zero_transfer/cpsc2018",
                "artifacts/primary/stage15_review",
            ],
            outputs=["artifacts/paper/figures"],
            kind="paper",
            timeout=600,
        ),
        _node(
            node_id="claim_sync",
            command=[
                "{python}", "scripts/claim_sync_v3.py",
                "--stage15", "artifacts/primary/stage15_review",
                "--figures", "artifacts/paper/figures",
                "--registry", "arc_audit/verified_registry.v1.json",
                "--reviewer-public-key", "security/reviewer_ed25519.pub",
                "--output-dir", "artifacts/paper/claims",
            ],
            deps=["tiny_stage15_review", "primary_figures"],
            inputs=[
                "artifacts/primary/stage15_review", "artifacts/paper/figures",
                "arc_audit/verified_registry.v1.json", "security/reviewer_ed25519.pub",
            ],
            outputs=["artifacts/paper/claims"],
            kind="paper",
        ),
        _node(
            node_id="stage20_review_draft",
            command=[
                "{python}", "scripts/compile_stage20_review_draft_v3.py",
                "--claims", "artifacts/paper/claims",
                "--figures", "artifacts/paper/figures",
                "--output-dir", "artifacts/paper/stage20_review_draft",
            ],
            deps=["primary_figures", "claim_sync"],
            inputs=["artifacts/paper/figures", "artifacts/paper/claims"],
            outputs=["artifacts/paper/stage20_review_draft"],
            kind="paper",
            timeout=600,
        ),
        _node(
            node_id="compile_submission",
            command=[
                "{python}", "scripts/compile_submission_v3.py",
                "--claims", "artifacts/paper/claims",
                "--figures", "artifacts/paper/figures",
                "--author-declaration", "paper/provenance/author_declaration.v1.json",
                "--tool-provenance", "paper/provenance/tool_provenance.v1.json",
                "--venue-policy", "paper/provenance/venue_policy.v1.json",
                "--author-public-key", "security/author_ed25519.pub",
                "--author-kit", "paper/provenance/icassp2027-author-kit.fixture.zip",
                "--output-dir", "artifacts/paper/submission",
            ],
            deps=["primary_figures", "claim_sync", "stage20_review_draft"],
            inputs=[
                "artifacts/paper/figures",
                "artifacts/paper/claims",
                "artifacts/paper/stage20_review_draft",
                "paper/provenance/author_declaration.v1.json",
                "paper/provenance/tool_provenance.v1.json",
                "paper/provenance/venue_policy.v1.json",
                "security/author_ed25519.pub",
                "paper/provenance/icassp2027-author-kit.fixture.zip",
            ],
            outputs=["artifacts/paper/submission"],
            kind="paper",
            timeout=600,
        ),
    ]
    return ExperimentManifest.from_dict({"schema_version": 1, "nodes": nodes})


def _configure_stable_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.environment_sha256", lambda: "a" * 64,
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [], "cuda_runtime": "unavailable"},
    )


def test_empty_artifact_dag_rebuilds_entire_claim_bearing_closure(
    tmp_path: Path,
    monkeypatch,
    reviewer_keys,
) -> None:
    _configure_stable_runner(monkeypatch)
    repo = tmp_path / "repo"
    _copy_repository_fixture(repo, reviewer_keys.public)
    manifest = _claim_manifest(reviewer_keys.private)
    selected = manifest.topological(manifest.select("icassp"))
    expected_order = [
        "tiny_manifests", "tiny_robust_rank_maps", "tiny_benchmark",
        "tiny_external_chapman", "tiny_external_cpsc2018", "tiny_meta_analysis",
        "tiny_stage15_review", "primary_figures", "claim_sync",
        "stage20_review_draft", "compile_submission",
    ]
    assert [node.id for node in selected] == expected_order
    assert selected[-4].command[1] == "experiments/paper_figures_v3.py"
    assert selected[-3].command[1] == "scripts/claim_sync_v3.py"
    assert selected[-2].command[1] == "scripts/compile_stage20_review_draft_v3.py"
    assert selected[-1].command[1] == "scripts/compile_submission_v3.py"

    run_root = tmp_path / "runs"
    assert not (run_root / "tiny" / "workspace" / "artifacts").exists()
    run_dir = DAGRunner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=run_root,
        run_id="tiny",
        environment_lock="cpu",
    ).run()
    workspace = run_dir / "workspace"
    artifacts = workspace / "artifacts"

    map_cells = pd.read_parquet(artifacts / "primary/robust_rank_maps/map_cells.parquet")
    assert len(map_cells) == 3 * len(deep_configuration_panel()) * 12
    figure_map = pd.read_parquet(artifacts / "paper/figures/figure1_source.parquet")
    for frame in (map_cells, figure_map):
        frame["configuration"] = frame["configuration"].astype(str)
        frame["target"] = frame["target"].astype(str)
    map_cells = map_cells.sort_values(
        ["segment", "configuration", "target"]
    ).reset_index(drop=True)
    figure_map = figure_map.sort_values(
        ["segment", "configuration", "target"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        map_cells[["segment", "configuration", "target", "target_observed"]],
        figure_map[["segment", "configuration", "target", "target_observed"]],
        check_dtype=False,
    )
    observed = figure_map["target_observed"].to_numpy(dtype=bool)
    assert figure_map.loc[observed, "ambiguity_robust_mv"].isna().all()
    pd.testing.assert_series_equal(
        map_cells.loc[~observed, "ambiguity_robust_mv"].reset_index(drop=True),
        figure_map.loc[~observed, "ambiguity_robust_mv"].reset_index(drop=True),
        check_dtype=False,
    )
    effects = pd.read_parquet(artifacts / "primary/meta_analysis/effects.parquet")
    figure_effects = pd.read_parquet(artifacts / "paper/figures/figure2_source.parquet")
    for frame in (effects, figure_effects):
        frame["cohort"] = frame["cohort"].astype(str)
    pd.testing.assert_frame_equal(
        effects.sort_values("cohort").reset_index(drop=True),
        figure_effects[["cohort", "point", "ci95"]]
        .sort_values("cohort")
        .reset_index(drop=True),
        check_dtype=False,
    )

    stage15 = artifacts / "primary/stage15_review/decision.v3.json"
    figures_summary = json.loads(
        (artifacts / "paper/figures/summary.v3.json").read_text(encoding="utf-8")
    )
    expected_figure_inputs = {
        "rank_map": artifacts / "primary/robust_rank_maps/map_cells.parquet",
        "effects": artifacts / "primary/meta_analysis/effects.parquet",
        "stage15": stage15,
        "chapman": artifacts / "primary/zero_transfer/chapman/patient_metrics.parquet",
        "cpsc2018": artifacts
        / "primary/zero_transfer/cpsc2018/patient_metrics.parquet",
    }
    assert figures_summary["input_sha256"] == {
        name: lineage.artifact_sha256(path)
        for name, path in expected_figure_inputs.items()
    }
    assert set(figures_summary["artifacts_sha256"]) == set(FIGURE_ARTIFACTS)
    claims = json.loads(
        (artifacts / "paper/claims/claims.v3.json").read_text(encoding="utf-8")
    )
    ptb_point = float(effects.loc[effects.cohort == "PTB-XL", "point"].iloc[0])
    assert f"{ptb_point:.3f}" in claims["values"]["ResultIncrementalValue"]
    assert claims["stage15_sha256"] == lineage.artifact_sha256(stage15)
    assert claims["figures_summary_sha256"] == lineage.artifact_sha256(
        artifacts / "paper/figures/summary.v3.json"
    )
    assert claims["figure_artifacts_sha256"] == figures_summary["artifacts_sha256"]

    review_draft = artifacts / "paper/stage20_review_draft"
    draft_report = json.loads(
        (review_draft / "review_draft_report.v3.json").read_text(encoding="utf-8")
    )
    assert draft_report["schema_version"] == "stage20-review-draft-v3"
    assert draft_report["review_ready"] is True
    assert draft_report["submission_ready"] is False
    assert draft_report["release_eligible"] is False
    assert draft_report["not_for_submission"] is True
    assert draft_report["pages"] == 5
    assert draft_report["technical_content_end_page"] == 4
    assert draft_report["overfull_boxes"] == 0
    assert draft_report["claims_sha256"] == lineage.artifact_sha256(
        artifacts / "paper/claims/claims.v3.json"
    )
    assert len(PdfReader(str(review_draft / "stage20_review_draft.pdf")).pages) == 5

    submission = artifacts / "paper/submission"
    report = json.loads((submission / "build_report.v3.json").read_text(encoding="utf-8"))
    assert report["submission_ready"] is True
    assert report["pages"] == 5
    assert report["overfull_boxes"] == 0
    assert report["review_model"] == "single-anonymous"
    assert report["author_identities_visible_to_reviewers"] is True
    assert report["arc_formal_receipts_sha256"] == {}
    assert report["claims_sha256"] == lineage.artifact_sha256(
        artifacts / "paper/claims/claims.v3.json"
    )
    assert report["figure_artifacts_sha256"] == figures_summary["artifacts_sha256"]
    assert report["reviewed_scientific_input_sha256"] == draft_report[
        "reviewed_scientific_input_sha256"
    ]
    assert report["reviewed_scientific_input_bundle_sha256"] == lineage.canonical_sha256(
        draft_report["reviewed_scientific_input_sha256"]
    )
    policy_paths = {
        "author_declaration": workspace
        / "paper/provenance/author_declaration.v1.json",
        "tool_provenance": workspace / "paper/provenance/tool_provenance.v1.json",
        "venue_policy": workspace / "paper/provenance/venue_policy.v1.json",
        "author_public_key": workspace / "security/author_ed25519.pub",
        "author_kit": workspace
        / "paper/provenance/icassp2027-author-kit.fixture.zip",
    }
    assert report["paper_policy_input_sha256"] == {
        name: lineage.artifact_sha256(path) for name, path in policy_paths.items()
    }
    assert len(PdfReader(str(submission / "main_v2.pdf")).pages) == 5

    envelopes = {
        node.id: ResultEnvelope.read(run_dir / "envelopes" / f"{node.id}.json")
        for node in selected
    }
    assert len({envelope.source_sha256 for envelope in envelopes.values()}) == 1
    for node in selected:
        envelope = envelopes[node.id]
        assert set(envelope.upstream_sha256) == set(node.deps)
        assert set(envelope.outputs_sha256) == set(node.outputs)
        assert envelope.environment_lock_sha256


def test_empty_artifact_resume_rejects_tampered_completed_upstream(
    tmp_path: Path,
    monkeypatch,
    reviewer_keys,
) -> None:
    _configure_stable_runner(monkeypatch)
    repo = tmp_path / "repo"
    _copy_repository_fixture(repo, reviewer_keys.public, include_claim_tools=False)
    producer = "tests/fixtures/tiny_icassp_pipeline.py"
    allow_file = tmp_path / "allow-release"
    guard = _node(
        node_id="release_guard",
        command=[
            "{python}", "tests/fixtures/tiny_icassp_pipeline.py", "guard",
            "--allow-file", str(allow_file),
            "--output", "artifacts/control/release-guard.txt",
        ],
        deps=["tiny_robust_rank_maps"],
        inputs=["artifacts/primary/robust_rank_maps"],
        outputs=["artifacts/control/release-guard.txt"],
    )
    manifest = ExperimentManifest.from_dict(
        {
            "schema_version": 1,
            "nodes": [
                _node(
                    node_id="tiny_manifests",
                    command=[
                        "{python}", producer, "manifests",
                        "--output", "artifacts/manifests",
                    ],
                    deps=[],
                    inputs=[],
                    outputs=["artifacts/manifests"],
                    profiles=["icassp", "extended", "legacy"],
                ),
                _node(
                    node_id="tiny_robust_rank_maps",
                    command=[
                        "{python}", producer, "map",
                        "--manifest", "artifacts/manifests/ptbxl.json",
                        "--output", "artifacts/primary/robust_rank_maps",
                    ],
                    deps=["tiny_manifests"],
                    inputs=["artifacts/manifests"],
                    outputs=["artifacts/primary/robust_rank_maps"],
                ),
                guard,
            ],
        }
    )
    runner = DAGRunner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=tmp_path / "runs",
        run_id="tamper",
        environment_lock="cpu",
    )
    with pytest.raises(ExecutionError, match="command failed"):
        runner.run()
    map_path = runner.run_dir / "workspace/artifacts/primary/robust_rank_maps/map_cells.parquet"
    map_path.write_bytes(map_path.read_bytes() + b"tampered")
    allow_file.write_text("allow\n", encoding="utf-8")
    with pytest.raises(ExecutionError, match="completed outputs changed"):
        DAGRunner(
            repo=repo,
            manifest=manifest,
            profile="icassp",
            run_root=runner.run_dir.parent,
            run_id=runner.run_dir.name,
            environment_lock="cpu",
            resume=True,
        ).run()
