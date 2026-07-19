"""Fail-closed checks for the public project-level scientific narrative."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_arc_disclosure_is_generated_from_authenticated_current_provenance() -> None:
    compliance = (ROOT / "paper" / "compliance.tex").read_text(encoding="utf-8")
    manuscript = (ROOT / "paper" / "main_v2.tex").read_text(encoding="utf-8")
    provenance = json.loads(
        (ROOT / "paper" / "tool_provenance.v1.template.json").read_text(encoding="utf-8")
    )
    assert r"\SubmissionToolDisclosure" in compliance
    assert r"\input{auto/tool_provenance}" in manuscript
    assert "queue owner" not in compliance
    assert "Stage~01 failed" not in manuscript
    assert provenance["status"] == "PENDING_HUMAN_VERIFICATION"
    assert provenance["autoresearchclaw_formal_receipts"] == {}


def _claim_surfaces() -> dict[str, str]:
    paths = [ROOT / "README.md", ROOT / "src" / "ecgcert" / "__init__.py"]
    docs = ROOT / "docs"
    if docs.is_dir():
        for path in docs.rglob("*.md"):
            relative = path.relative_to(docs).as_posix().lower()
            if relative == "research_protocol.md" or "execution" in relative:
                continue
            paths.append(path)
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }


def test_public_claim_surfaces_reject_legacy_headlines():
    banned = {
        r"\bfixed[- ]rank[- ]?3\b": "privileged fixed-rank headline",
        r"\brank[- ]3 spatial (?:basis|subspace)\b": "privileged rank-3 representation",
        r"\bexact(?:ly)? (?:recoverable|identifiable|non-identifiable)\b": (
            "model-conditional geometry promoted to exact recovery"
        ),
        r"\bany\s+SNR\b": "SNR-independent guarantee",
        r"\breconstructor-independent\b": "reconstructor-independent guarantee",
        r"\brecoverability certificate\b|\bcertified (?:recovery|hallucination|safety)\b": (
            "legacy certificate language"
        ),
        r"\bdistribution-free calibrated intervals?\b|\bconformalized quantile regression\b": (
            "legacy calibration headline"
        ),
        r"\bST safety\b|\bclinical safety (?:claim|guarantee|bound)\b": (
            "legacy clinical-safety headline"
        ),
    }
    for name, text in _claim_surfaces().items():
        for pattern, explanation in banned.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            assert match is None, f"{name}: {explanation}: {match.group(0)!r}"


def test_readme_states_robust_model_conditional_fail_closed_status():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "model-conditional",
        "{2, 3, 4, 5}",
        "patient-level",
        "patient-cluster",
        "external zero-transfer",
        "Stage 15",
        "fail-closed",
        "PENDING",
        "Legacy artifacts",
    )
    for statement in required:
        assert statement in readme, f"README.md is missing required status text: {statement!r}"
    normalized = re.sub(r"\s+", " ", readme.replace("**", ""))
    assert "do not block the primary submission" in normalized


def test_package_metadata_matches_primary_research_scope():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    description = metadata["description"].lower()
    assert "robust target-specific" in description
    assert "model-conditional" in description
    assert "certificate" not in description
    assert "calibrated intervals" not in description


def test_readme_contains_no_final_empirical_percentage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"(?<![\w.])\d+(?:\.\d+)?\s*%", readme) is None


def test_arc_gate_register_matches_the_sandboxed_control_plane() -> None:
    register = (ROOT / "arc_audit" / "STAGE_GATES.md").read_text(encoding="utf-8")
    config = (ROOT / "arc_audit" / "config.arc.yaml").read_text(encoding="utf-8")
    assert "experiment profile is real (`ssh_remote`)" not in register
    assert "local sandboxed control plane" in register
    assert "acp_runtime.v1.json" not in config
    assert "acpx_version.txt" in config
    assert "arc_probe_status.v1.json" in config


def test_precompiled_paper_pdfs_are_not_source_artifacts() -> None:
    for relative in ("paper/main_v2.pdf", "paper/arxiv_long.pdf"):
        assert not (ROOT / relative).exists()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "paper/*.pdf" in gitignore
