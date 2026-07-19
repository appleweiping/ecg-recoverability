from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_verified_registry import manuscript_keys, validate
from ecgcert.verified_registry import validate_required_literature


ROOT = Path(__file__).resolve().parents[1]


def test_primary_manuscript_claims_are_registered() -> None:
    report = validate(
        ROOT / "paper" / "main_v2.tex",
        ROOT / "arc_audit" / "verified_registry.v1.json",
        require_verified=False,
    )
    assert report["all_registered"] is True
    assert report["pending_citations"] == []
    assert report["all_verified"] is True


def test_stage20_fails_closed_on_pending_citation(tmp_path: Path) -> None:
    manuscript = tmp_path / "paper.tex"
    manuscript.write_text(r"Claim \cite{a}; value \ResultOne.", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "verified-registry-v1",
                "citations": {"a": {"status": "registered_pending_stage5", "source": "https://x"}},
                "numeric_claims": {"ResultOne": {"artifact": "artifacts/x.json", "status": "blocked"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pending citations"):
        validate(manuscript, registry, require_verified=True)


def test_multiline_citation_parser() -> None:
    citations, numeric = manuscript_keys("\\cite{a,b,%\n c} \\ResultEffect")
    assert citations == {"a", "b", "c"}
    assert numeric == {"ResultEffect"}


def test_required_literature_topics_are_verified_and_cited() -> None:
    registry = json.loads(
        (ROOT / "arc_audit" / "verified_registry.v1.json").read_text(encoding="utf-8")
    )
    cited, _ = manuscript_keys(
        (ROOT / "paper" / "main_v2.tex").read_text(encoding="utf-8")
    )
    report = validate_required_literature(registry, cited_keys=cited)
    assert report["all_required_literature_verified_and_cited"] is True


def test_required_literature_missing_topic_fails_closed() -> None:
    registry = json.loads(
        (ROOT / "arc_audit" / "verified_registry.v1.json").read_text(encoding="utf-8")
    )
    del registry["required_literature_coverage"]["full_configuration_benchmark"]
    with pytest.raises(ValueError, match="must cover"):
        validate_required_literature(registry, cited_keys=set(registry["citations"]))
