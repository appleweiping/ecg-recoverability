"""Generate fail-closed ARC Stage 5, 9, and 20 evidence gates.

Each mode writes ``decision.v3.json`` with frozen input hashes and automatic
eligibility.  The status is always ``PENDING_USER_REVIEW``; use
``scripts/record_stage_review.py`` and ``scripts/wait_for_stage_review.py`` to
create a reviewed decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ecgcert import lineage
from ecgcert.arc_control import validate_arc_control_report
from ecgcert.paper_evidence import (
    FIGURE_ARTIFACTS,
    require_artifact_hashes,
    validate_figure_bundle,
)
from ecgcert.stage_gates import (
    DEFAULT_REVIEWER_PUBLIC_KEY,
    make_pending_gate,
    validate_gate,
    validate_reviewed_gate,
)
from ecgcert.verified_registry import (
    VERIFIED,
    manuscript_keys,
    validate as validate_registry,
    validate_required_literature,
)


SECURITY_SCHEMA = "ecgcert-security-status-v1"
SECURITY_FLAGS = (
    "exposed_password_rotated",
    "key_only_auth_verified",
    "known_hosts_pinned",
    "automatic_host_key_acceptance_disabled",
    "password_fallback_disabled",
    "repository_secret_scan_passed",
)
VERIFIED_NUMERIC = {"verified_artifact", "verified_primary", "verified_secondary"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_PATTERNS: dict[str, str] = {
    "fixed_rank_3": r"\bfixed[- ]rank[- ]?3\b",
    "any_snr": r"\bany\s+SNR\b",
    "reconstructor_independent": r"\breconstructor-independent\b",
    "certificate_claim": (
        r"\brecoverability certificate\b|\bcertified "
        r"(?:recovery|hallucination|safety)\b"
    ),
    "distribution_free_cqr": (
        r"\bdistribution-free calibrated intervals?\b|"
        r"\bconformalized quantile regression\b"
    ),
    "clinical_safety": r"\bST safety\b|\bclinical safety (?:claim|guarantee|bound)\b",
}
_SECRET_PATTERNS: dict[str, str] = {
    "server_hostname": r"connect\.weste\.seetacloud\.com",
    "root_ssh_command": r"ssh\s+(?:-[^\s]+\s+)*root@",
    "credential_assignment": r"\b(?:password|passwd)\s*[:=]",
    "private_key": r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _resolve_artifact(path: Path, filename: str) -> Path:
    resolved = path / filename if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def bind_official_arc_control(
    gate: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    stage = gate.get("stage")
    if stage not in (5, 9, 20):
        raise ValueError("official ARC binding requires Stage 5, 9, or 20")
    report = validate_arc_control_report(_load_json(report_path), int(stage))
    evidence = dict(gate["evidence"])
    evidence["official_arc_control"] = {
        "report_sha256": lineage.artifact_sha256(report_path),
        "receipt_sha256": report["receipt_sha256"],
        "run_id": report["run_id"],
        "session_id": report["session_id"],
        "decision": report["decision"],
        "stage_output_sha256": report["stage_output_sha256"],
    }
    bound = dict(gate)
    bound["evidence"] = evidence
    bound["evidence_sha256"] = lineage.canonical_sha256(evidence)
    return bound


def verified_registry_inventory(
    manuscript: Path,
    registry_path: Path,
    *,
    expected_stage15_sha256: str | None = None,
    expected_stage15_status: str | None = None,
    expected_claim_macros_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate registration and report verification across the whole registry."""
    manuscript_report = validate_registry(
        manuscript, registry_path, require_verified=False
    )
    registry = _load_json(registry_path)
    cited_keys, _ = manuscript_keys(manuscript.read_text(encoding="utf-8"))
    literature_coverage = validate_required_literature(
        registry, cited_keys=cited_keys,
    )
    citations = registry.get("citations")
    numeric_claims = registry.get("numeric_claims")
    if not isinstance(citations, dict) or not isinstance(numeric_claims, dict):
        raise ValueError("VerifiedRegistry citations/numeric_claims must be objects")
    stage15_binding_valid = expected_stage15_sha256 is None or (
        registry.get("stage15_sha256") == expected_stage15_sha256
        and registry.get("stage15_status") == expected_stage15_status
    )
    pending: list[str] = []
    pending_numeric: list[str] = []
    invalid_numeric_bindings: list[str] = []
    for key, entry in citations.items():
        if not key or not isinstance(entry, dict):
            raise ValueError("VerifiedRegistry contains an invalid citation entry")
        if not entry.get("source") or not entry.get("status"):
            raise ValueError(f"citation {key!r} lacks source/status")
        if entry["status"] not in VERIFIED:
            pending.append(key)
    for key, entry in numeric_claims.items():
        if not key or not isinstance(entry, dict):
            raise ValueError("VerifiedRegistry contains an invalid numeric-claim entry")
        if not entry.get("artifact") or not entry.get("status"):
            raise ValueError(f"numeric claim {key!r} lacks artifact/status")
        if entry["status"] not in VERIFIED_NUMERIC:
            pending_numeric.append(key)
        if expected_stage15_sha256 is not None and (
            entry.get("status") != "verified_artifact"
            or entry.get("stage15_sha256") != expected_stage15_sha256
            or entry.get("stage15_status") != expected_stage15_status
            or not _HEX64.fullmatch(str(entry.get("value_sha256")))
            or (
                expected_claim_macros_sha256 is not None
                and entry.get("claim_macros_sha256")
                != expected_claim_macros_sha256
            )
        ):
            invalid_numeric_bindings.append(key)
    return {
        **manuscript_report,
        "registry_citation_count": len(citations),
        "registry_numeric_claim_count": len(numeric_claims),
        "pending_registry_citations": sorted(pending),
        "pending_registry_numeric_claims": sorted(pending_numeric),
        "invalid_numeric_bindings": sorted(invalid_numeric_bindings),
        "stage15_binding_valid": stage15_binding_valid,
        "all_citations_verified": not pending,
        "all_numeric_claims_verified": not pending_numeric and not invalid_numeric_bindings,
        "all_verified": (
            not pending
            and not pending_numeric
            and not invalid_numeric_bindings
            and stage15_binding_valid
            and literature_coverage["all_required_literature_verified_and_cited"]
        ),
        "required_literature_coverage": literature_coverage,
    }


def build_stage5_gate(
    *, manuscript: Path, registry: Path, created_at=None
) -> dict[str, Any]:
    inventory = verified_registry_inventory(manuscript, registry)
    eligible = bool(inventory["all_registered"] and inventory["all_citations_verified"])
    reasons = (
        ["all registered citations are verified"]
        if eligible
        else [
            "citation verification remains pending: "
            + ", ".join(inventory["pending_registry_citations"])
        ]
    )
    evidence = {
        "manuscript_sha256": lineage.artifact_sha256(manuscript),
        "verified_registry_sha256": lineage.artifact_sha256(registry),
        "registry_check": inventory,
        "rule": {
            "all_manuscript_claims_registered": True,
            "proceed_requires_all_registry_citations_verified": True,
        },
    }
    return make_pending_gate(
        stage=5,
        evidence=evidence,
        eligible_for_proceed=eligible,
        automatic_reasons=reasons,
        created_at=created_at,
    )


def validate_security_status(status_path: Path) -> dict[str, Any]:
    """Validate the explicit key-only remote-security attestation used at Stage 9."""
    status = _load_json(status_path)
    if status.get("schema_version") != SECURITY_SCHEMA:
        raise ValueError(f"security status must use {SECURITY_SCHEMA}")
    allowed = {
        "schema_version", *SECURITY_FLAGS, "known_hosts_sha256", "public_key_sha256",
        "verified_at", "verified_by",
    }
    if set(status) != allowed:
        raise ValueError("security status has missing or unexpected fields")
    missing = [field for field in SECURITY_FLAGS if not isinstance(status.get(field), bool)]
    if missing:
        raise ValueError(f"security status lacks boolean fields: {missing}")
    for field in ("known_hosts_sha256", "public_key_sha256"):
        if not _HEX64.fullmatch(str(status.get(field))):
            raise ValueError(f"security status {field} must be a full SHA-256")
    for field in ("verified_at", "verified_by"):
        if not isinstance(status.get(field), str) or not status[field].strip():
            raise ValueError(f"security status {field} must be non-empty")
    checks = {field: status[field] is True for field in SECURITY_FLAGS}
    return {
        "schema_version": SECURITY_SCHEMA,
        "checks": checks,
        "all_secure": all(checks.values()),
        "known_hosts_sha256": status["known_hosts_sha256"],
        "public_key_sha256": status["public_key_sha256"],
        "verified_at": status["verified_at"],
        "verified_by": status["verified_by"],
    }


def build_stage9_gate(
    *,
    protocol: Path,
    experiment_manifest: Path,
    arc_config: Path,
    security_status: Path,
    created_at=None,
) -> dict[str, Any]:
    inputs = {
        "research_protocol": lineage.artifact_sha256(protocol),
        "experiment_manifest": lineage.artifact_sha256(experiment_manifest),
        "arc_config": lineage.artifact_sha256(arc_config),
        "security_status": lineage.artifact_sha256(security_status),
    }
    security = validate_security_status(security_status)
    eligible = bool(security["all_secure"])
    reasons = (
        ["protocol inputs are frozen and remote security is verified"]
        if eligible
        else [
            "remote security checks failed: "
            + ", ".join(key for key, passed in security["checks"].items() if not passed)
        ]
    )
    evidence = {
        "input_sha256": inputs,
        "frozen_inputs_sha256": lineage.canonical_sha256(inputs),
        "security": security,
        "rule": {
            "protocol_manifest_arc_config_and_security_frozen": True,
            "proceed_requires_all_security_checks": True,
        },
    }
    return make_pending_gate(
        stage=9,
        evidence=evidence,
        eligible_for_proceed=eligible,
        automatic_reasons=reasons,
        created_at=created_at,
    )


def static_paper_check(manuscript: Path, pdf: Path) -> dict[str, Any]:
    """Run deterministic source/PDF checks needed by the Stage-20 gate."""
    source = manuscript.read_text(encoding="utf-8")
    legacy_hits = [
        name
        for name, pattern in _LEGACY_PATTERNS.items()
        if re.search(pattern, source, flags=re.IGNORECASE)
    ]
    secret_hits = [
        name
        for name, pattern in _SECRET_PATTERNS.items()
        if re.search(pattern, source, flags=re.IGNORECASE)
    ]
    try:
        from pypdf import PdfReader
    except ImportError as error:  # locked environment must contain pypdf
        raise RuntimeError("pypdf is required for the Stage-20 static paper check") from error
    reader = PdfReader(str(pdf))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    pending_in_pdf = bool(re.search(r"\bPENDING\b", extracted, flags=re.IGNORECASE))
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, flags=re.DOTALL
    )
    conclusion = re.search(
        r"\\section\{Conclusion\}(.*?)(?:\\clearpage|\\bibliographystyle)",
        source,
        flags=re.DOTALL,
    )
    checks = {
        "claim_macros_are_evidence_synchronized": (
            r"\input{auto/robust_map_placeholders}" in source
        ),
        "no_legacy_headline": not legacy_hits,
        "no_private_server_or_credential_surface": not secret_hits,
        "no_pending_text_in_pdf": not pending_in_pdf,
        "abstract_uses_synchronized_headline": (
            abstract is not None and r"\ResultHeadline" in abstract.group(1)
        ),
        "conclusion_uses_synchronized_result": (
            conclusion is not None and r"\ResultConclusion" in conclusion.group(1)
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "legacy_pattern_hits": legacy_hits,
        "secret_pattern_hits": secret_hits,
        "pdf_pages": len(reader.pages),
        "manuscript_sha256": lineage.artifact_sha256(manuscript),
        "pdf_sha256": lineage.artifact_sha256(pdf),
    }


def _stage15_reviewed_paper_decision(
    gate: dict[str, Any],
    reviewer_public_key: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> str | None:
    if validate_gate(gate) != 15:
        raise ValueError("Stage 20 requires an ARC Stage-15 gate")
    status = gate.get("status")
    if status == "PENDING_USER_REVIEW":
        return None
    # A claimed reviewed decision with a broken signature is evidence
    # corruption, not a merely failed quality criterion.
    validate_reviewed_gate(gate, public_key_path=reviewer_public_key)
    return str(status) if status in {"PROCEED", "PIVOT"} else None


def build_stage20_gate(
    *,
    stage15: Path,
    submission_build: Path,
    claims: Path,
    manuscript: Path,
    registry: Path,
    figures: Path | None = None,
    reviewer_public_key: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
    created_at=None,
) -> dict[str, Any]:
    stage15_path = _resolve_artifact(stage15, "decision.v3.json")
    build_path = _resolve_artifact(submission_build, "build_report.v3.json")
    build_dir = build_path.parent
    pdf_path = build_dir / "main_v2.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    claims_path = _resolve_artifact(claims, "claims.v3.json")
    claims_dir = claims_path.parent
    macros_path = claims_dir / "robust_map_placeholders.tex"
    if not macros_path.is_file():
        raise FileNotFoundError(macros_path)
    # The frozen DAG places claims and figures as sibling artifact directories.
    # An explicit path remains available for isolated callers/tests.
    figures_dir = (
        figures.resolve()
        if figures is not None
        else (claims_dir.parent / "figures").resolve()
    )

    stage15_value = _load_json(stage15_path)
    stage15_decision = _stage15_reviewed_paper_decision(
        stage15_value, reviewer_public_key
    )
    claims_value = _load_json(claims_path)
    build_value = _load_json(build_path)
    if claims_value.get("schema_version") != "paper-claims-v3":
        raise ValueError("Stage 20 requires a paper-claims-v3 artifact")
    if build_value.get("schema_version") != "submission-build-v3":
        raise ValueError("Stage 20 requires a submission-build-v3 report")

    stage15_sha256 = lineage.artifact_sha256(stage15_path)
    claims_sha256 = lineage.artifact_sha256(claims_path)
    claim_macros_sha256 = lineage.artifact_sha256(macros_path)
    pdf_sha256 = lineage.artifact_sha256(pdf_path)
    figure_summary_value = _load_json(figures_dir / "summary.v3.json")
    figure_binding = validate_figure_bundle(figures_dir, summary=figure_summary_value)
    if figure_summary_value.get("input_sha256", {}).get("stage15") != stage15_sha256:
        raise ValueError("figure summary is not bound to the supplied reviewed Stage-15 gate")
    claimed_figure_artifacts = require_artifact_hashes(
        claims_value.get("figure_artifacts_sha256"),
        label="Stage-20 claim-sync figure binding",
    )
    if claims_value.get("stage15_sha256") != stage15_sha256:
        raise ValueError("claim sync is not bound to the supplied reviewed Stage-15 gate")
    if build_value.get("claims_sha256") != claims_sha256:
        raise ValueError("submission build is not bound to the supplied claim-sync artifact")
    if claims_value.get("claim_macros_sha256") != claim_macros_sha256:
        raise ValueError("claim macro file does not match its claim-sync SHA-256")
    if build_value.get("claim_macros_sha256") != claim_macros_sha256:
        raise ValueError("submission build is not bound to the supplied claim macro file")
    if (
        claims_value.get("figures_summary_sha256")
        != figure_binding["summary_sha256"]
        or claims_value.get("figures_sha256") != figure_binding["summary_sha256"]
    ):
        raise ValueError("claim sync is not bound to the supplied figure summary")
    if claimed_figure_artifacts != figure_binding["artifacts_sha256"]:
        raise ValueError("claim sync is not bound to every supplied figure artifact")
    if (
        build_value.get("figures_summary_sha256")
        != figure_binding["summary_sha256"]
        or build_value.get("figures_sha256") != figure_binding["summary_sha256"]
    ):
        raise ValueError("submission build is not bound to the supplied figure summary")
    build_figure_artifacts = require_artifact_hashes(
        build_value.get("figure_artifacts_sha256"),
        label="Stage-20 submission-build figure binding",
    )
    if build_figure_artifacts != figure_binding["artifacts_sha256"]:
        raise ValueError("submission build is not bound to every supplied figure artifact")
    if build_value.get("pdf_sha256") != pdf_sha256:
        raise ValueError("submission PDF hash does not match its build report")

    registry_sha256 = lineage.artifact_sha256(registry)
    if claims_value.get("verified_registry_sha256") != registry_sha256:
        raise ValueError("claim sync is not bound to the supplied dynamic VerifiedRegistry")
    if build_value.get("verified_registry_sha256") != registry_sha256:
        raise ValueError("submission build is not bound to the dynamic VerifiedRegistry")
    registry_value = _load_json(registry)
    if registry_value.get("claim_macros_sha256") != claim_macros_sha256:
        raise ValueError("dynamic VerifiedRegistry is not bound to the claim macro file")
    if (
        registry_value.get("figures_summary_sha256")
        != figure_binding["summary_sha256"]
    ):
        raise ValueError("dynamic VerifiedRegistry is not bound to the figure summary")
    registry_figure_artifacts = require_artifact_hashes(
        registry_value.get("figure_artifacts_sha256"),
        label="Stage-20 VerifiedRegistry figure binding",
    )
    if registry_figure_artifacts != figure_binding["artifacts_sha256"]:
        raise ValueError("dynamic VerifiedRegistry is not bound to every figure artifact")

    compiled_root = build_dir / "build"
    compiled_paths = {
        "auto/robust_map_placeholders.tex": (
            compiled_root / "auto" / "robust_map_placeholders.tex"
        ),
        "figures_v3/summary.v3.json": compiled_root / "figures_v3" / "summary.v3.json",
        **{
            f"figures_v3/{name}": compiled_root / "figures_v3" / name
            for name in FIGURE_ARTIFACTS
        },
    }
    compiled_actual = {
        name: lineage.artifact_sha256(path) for name, path in compiled_paths.items()
    }
    compiled_expected = {
        "auto/robust_map_placeholders.tex": claim_macros_sha256,
        "figures_v3/summary.v3.json": figure_binding["summary_sha256"],
        **{
            f"figures_v3/{name}": figure_binding["artifacts_sha256"][name]
            for name in FIGURE_ARTIFACTS
        },
    }
    recorded_compiled = build_value.get("compiled_input_sha256")
    if not isinstance(recorded_compiled, dict) or recorded_compiled != compiled_actual:
        raise ValueError("compiled paper input hashes do not match the build report")
    if compiled_actual != compiled_expected:
        raise ValueError("compiled paper inputs do not match their evidence sources")

    values = claims_value.get("values")
    resolved_values = isinstance(values, dict) and bool(values) and all(
        isinstance(value, str)
        and value.strip()
        and not re.search(r"\b(?:PENDING|PLACEHOLDER)\b", value, flags=re.IGNORECASE)
        for value in values.values()
    )
    values_sha256 = lineage.canonical_sha256(values) if isinstance(values, dict) else None
    numeric_registry = registry_value.get("numeric_claims", {})
    value_bindings = isinstance(values, dict) and isinstance(numeric_registry, dict) and all(
        isinstance(numeric_registry.get(key), dict)
        and numeric_registry[key].get("value_sha256") == lineage.canonical_sha256(value)
        for key, value in values.items()
    )
    claim_checks = {
        "status_is_reviewed_paper_decision": (
            stage15_decision in {"PROCEED", "PIVOT"}
            and claims_value.get("status") == stage15_decision
        ),
        "submission_ready": claims_value.get("submission_ready") is True,
        "numeric_values_resolved": resolved_values,
        "values_hash_bound": (
            values_sha256 is not None
            and claims_value.get("claim_values_sha256") == values_sha256
            and registry_value.get("claim_values_sha256") == values_sha256
        ),
        "per_macro_value_hashes_bound": value_bindings,
        "claim_macro_file_bound": True,
        "figure_summary_bound": True,
        "figure_pdfs_and_source_tables_bound": True,
        "compiled_inputs_match_sources": True,
    }
    static = static_paper_check(manuscript, pdf_path)
    build_checks = {
        "status_complete": build_value.get("status") == "complete",
        "submission_ready": build_value.get("submission_ready") is True,
        "stage15_status_matches": (
            stage15_decision in {"PROCEED", "PIVOT"}
            and build_value.get("stage15_status") == stage15_decision
        ),
        "exactly_five_pages": (
            build_value.get("pages") == 5 and static["pdf_pages"] == 5
        ),
        "zero_overfull_boxes": build_value.get("overfull_boxes") == 0,
    }
    registry_check = verified_registry_inventory(
        manuscript,
        registry,
        expected_stage15_sha256=stage15_sha256,
        expected_stage15_status=stage15_decision,
        expected_claim_macros_sha256=claim_macros_sha256,
    )
    checks = {
        "reviewed_stage15_paper_decision": stage15_decision in {"PROCEED", "PIVOT"},
        "build": all(build_checks.values()),
        "claim_sync": all(claim_checks.values()),
        "verified_registry": registry_check["all_verified"],
        "static_paper": static["passed"],
    }
    reasons = [name for name, passed in checks.items() if not passed]
    if not reasons:
        reasons = ["all Stage-20 automatic quality criteria pass"]
    evidence = {
        "input_sha256": {
            "reviewed_stage15": stage15_sha256,
            "submission_build_report": lineage.artifact_sha256(build_path),
            "claim_sync": claims_sha256,
            "claim_macros": claim_macros_sha256,
            "verified_registry": lineage.artifact_sha256(registry),
            "manuscript": lineage.artifact_sha256(manuscript),
            "submission_pdf": pdf_sha256,
            "figures_summary": figure_binding["summary_sha256"],
            **{
                f"figure_artifact:{name}": digest
                for name, digest in figure_binding["artifacts_sha256"].items()
            },
        },
        "compiled_input_sha256": compiled_actual,
        "checks": checks,
        "build_checks": build_checks,
        "claim_checks": claim_checks,
        "registry_check": registry_check,
        "static_paper_check": static,
        "rule": {
            "reviewed_stage15_proceed_or_pivot_required": True,
            "pages": 5,
            "overfull_boxes": 0,
            "all_registry_citations_verified": True,
        },
    }
    return make_pending_gate(
        stage=20,
        evidence=evidence,
        eligible_for_proceed=all(checks.values()),
        automatic_reasons=reasons,
        created_at=created_at,
    )


def _arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage5", "stage9", "stage20"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arc-control", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, default=root / "paper" / "main_v2.tex")
    parser.add_argument(
        "--registry", type=Path, default=root / "arc_audit" / "verified_registry.v1.json"
    )
    parser.add_argument("--protocol", type=Path, default=root / "docs" / "research_protocol.md")
    parser.add_argument(
        "--experiment-manifest", type=Path,
        default=root / "scripts" / "experiment_manifest.yaml",
    )
    parser.add_argument("--arc-config", type=Path, default=root / "arc_audit" / "config.arc.yaml")
    parser.add_argument("--security-status", type=Path)
    parser.add_argument("--stage15", type=Path)
    parser.add_argument("--submission-build", type=Path)
    parser.add_argument("--claims", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument(
        "--reviewer-public-key",
        type=Path,
        default=DEFAULT_REVIEWER_PUBLIC_KEY,
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.mode == "stage5":
        gate = build_stage5_gate(
            manuscript=arguments.manuscript, registry=arguments.registry
        )
    elif arguments.mode == "stage9":
        if arguments.security_status is None:
            raise SystemExit("stage9 requires --security-status")
        gate = build_stage9_gate(
            protocol=arguments.protocol,
            experiment_manifest=arguments.experiment_manifest,
            arc_config=arguments.arc_config,
            security_status=arguments.security_status,
        )
    else:
        missing = [
            name
            for name in ("stage15", "submission_build", "claims")
            if getattr(arguments, name) is None
        ]
        if missing:
            raise SystemExit(f"stage20 requires: {', '.join('--' + x for x in missing)}")
        gate = build_stage20_gate(
            stage15=arguments.stage15,
            submission_build=arguments.submission_build,
            claims=arguments.claims,
            manuscript=arguments.manuscript,
            registry=arguments.registry,
            figures=arguments.figures,
            reviewer_public_key=arguments.reviewer_public_key,
        )
    gate = bind_official_arc_control(gate, arguments.arc_control)
    output = arguments.output_dir.resolve()
    _atomic_json(output / "decision.v3.json", gate)
    print(
        f"Stage {gate['stage']} gate: status={gate['status']}, "
        f"eligible_for_proceed={gate['eligible_for_proceed']}"
    )


if __name__ == "__main__":
    main()
