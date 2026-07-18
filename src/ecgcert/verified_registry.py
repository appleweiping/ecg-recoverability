"""Validation helpers for the manuscript VerifiedRegistry."""
from __future__ import annotations

import json
from pathlib import Path
import re


CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
RESULT_RE = re.compile(r"\\(Result[A-Za-z]+)\b")
VERIFIED = {"verified_primary", "verified_secondary"}
REQUIRED_LITERATURE_TOPICS = {
    "full_configuration_benchmark",
    "imputeecg",
    "ecgrecover",
}


def manuscript_keys(text: str) -> tuple[set[str], set[str]]:
    citations = {
        key.strip()
        for group in CITE_RE.findall(text)
        for key in group.replace("%", "").replace("\n", "").split(",")
        if key.strip()
    }
    return citations, set(RESULT_RE.findall(text))


def validate(manuscript: Path, registry_path: Path, *, require_verified: bool) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "verified-registry-v1":
        raise ValueError("unsupported VerifiedRegistry schema")
    citations, numeric = manuscript_keys(manuscript.read_text(encoding="utf-8"))
    registered_citations = registry.get("citations", {})
    registered_numeric = registry.get("numeric_claims", {})
    missing_citations = citations - set(registered_citations)
    missing_numeric = numeric - set(registered_numeric)
    if missing_citations or missing_numeric:
        raise ValueError(
            f"unregistered claims: citations={sorted(missing_citations)}, "
            f"numeric={sorted(missing_numeric)}"
        )
    pending = sorted(
        key for key in citations if registered_citations[key].get("status") not in VERIFIED
    )
    for key in citations:
        entry = registered_citations[key]
        if not entry.get("source") or entry.get("status") is None:
            raise ValueError(f"citation {key!r} lacks source/status")
    for key in numeric:
        if not registered_numeric[key].get("artifact"):
            raise ValueError(f"numeric claim {key!r} lacks an artifact path")
    if require_verified and pending:
        raise ValueError(f"Stage 20 cannot pass with pending citations: {pending}")
    return {
        "schema_version": "verified-registry-check-v1",
        "citations": sorted(citations),
        "numeric_claims": sorted(numeric),
        "pending_citations": pending,
        "all_registered": True,
        "all_verified": not pending,
    }


def validate_required_literature(
    registry: dict,
    *,
    cited_keys: set[str],
) -> dict:
    coverage = registry.get("required_literature_coverage")
    if not isinstance(coverage, dict) or set(coverage) != REQUIRED_LITERATURE_TOPICS:
        raise ValueError(
            "VerifiedRegistry must cover the full-configuration benchmark, "
            "ImputeECG, and ECGrecover"
        )
    citations = registry.get("citations")
    if not isinstance(citations, dict):
        raise ValueError("VerifiedRegistry citations must be an object")
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for topic in sorted(REQUIRED_LITERATURE_TOPICS):
        entry = coverage[topic]
        if not isinstance(entry, dict) or set(entry) != {
            "citation_key", "status", "source",
        }:
            raise ValueError(f"required literature topic {topic!r} has an invalid entry")
        key = entry["citation_key"]
        if not isinstance(key, str) or key not in citations:
            raise ValueError(f"required literature topic {topic!r} has no registered citation")
        registered = citations[key]
        if (
            key not in cited_keys
            or entry["status"] not in VERIFIED
            or registered.get("status") != entry["status"]
            or registered.get("source") != entry["source"]
        ):
            unresolved.append(topic)
        resolved[topic] = key
    return {
        "all_required_literature_verified_and_cited": not unresolved,
        "topics": resolved,
        "unresolved_topics": unresolved,
    }


__all__ = [
    "CITE_RE",
    "REQUIRED_LITERATURE_TOPICS",
    "RESULT_RE",
    "VERIFIED",
    "manuscript_keys",
    "validate",
    "validate_required_literature",
]
