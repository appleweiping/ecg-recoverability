from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from scripts.compile_stage20_review_draft_v3 import build_stage20_review_draft
from scripts.compile_submission_v3 import (
    main as compile_submission,
    paper_artifact_content_sha256,
    paper_artifact_signature_message,
    validate_compilation_inputs,
    validate_page_five_source,
    validate_submission_policy_inputs,
)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _bound_inputs(tmp_path: Path, *, compilable: bool = False) -> tuple[Path, Path]:
    stage15_sha256 = "a" * 64
    figures = tmp_path / "figures"
    figures.mkdir()
    if compilable:
        from pypdf import PdfWriter

        for name in FIGURE_ARTIFACTS:
            path = figures / name
            if path.suffix == ".pdf":
                writer = PdfWriter()
                writer.add_blank_page(width=200, height=100)
                with path.open("wb") as stream:
                    writer.write(stream)
            else:
                path.write_bytes(b"authenticated source-table fixture")
    else:
        for index, name in enumerate(FIGURE_ARTIFACTS):
            (figures / name).write_bytes(f"compile-figure-{index}".encode())
    artifacts = direct_artifact_hashes(figures)
    summary = _write_json(
        figures / "summary.v3.json",
        {
            "schema_version": "paper-figures-v3",
            "artifacts_sha256": artifacts,
            "input_sha256": {"stage15": stage15_sha256},
        },
    )

    claims = tmp_path / "claims"
    claims.mkdir()
    macros = claims / "robust_map_placeholders.tex"
    if compilable:
        result_macros = (
            "ResultHeadline",
            "ResultPrimaryAssociation",
            "ResultIncrementalValue",
            "ResultRankWeightStability",
            "ResultExternalAssociation",
            "ResultModelCoverage",
            "ResultBootstrapUncertainty",
            "ResultConclusion",
        )
        macros.write_text(
            "% authenticated reviewed fixture\n"
            + "".join(
                rf"\newcommand{{\{name}}}{{Reviewed evidence fixture.}}" + "\n"
                for name in result_macros
            ),
            encoding="utf-8",
        )
    else:
        macros.write_text("% synchronized\n", encoding="utf-8")
    macro_sha = lineage.artifact_sha256(macros)
    summary_sha = lineage.artifact_sha256(summary)
    registry = _write_json(
        claims / "verified_registry.v1.json",
        {
            "schema_version": "verified-registry-v1",
            "claim_macros_sha256": macro_sha,
            "figures_summary_sha256": summary_sha,
            "figure_artifacts_sha256": artifacts,
        },
    )
    _write_json(
        claims / "claims.v3.json",
        {
            "schema_version": "paper-claims-v3",
            "submission_ready": True,
            "status": "PROCEED",
            "stage15_sha256": stage15_sha256,
            "claim_macros_sha256": macro_sha,
            "figures_sha256": summary_sha,
            "figures_summary_sha256": summary_sha,
            "figure_artifacts_sha256": artifacts,
            "verified_registry_sha256": lineage.artifact_sha256(registry),
        },
    )
    return claims, figures


def _sign_paper_artifact(path: Path, value: dict, reviewer_keys) -> Path:
    private_key = serialization.load_pem_private_key(
        reviewer_keys.private.read_bytes(), password=None
    )
    public_key = serialization.load_ssh_public_key(reviewer_keys.public.read_bytes())
    assert isinstance(public_key, Ed25519PublicKey)
    raw_public = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    value.update(
        {
            "status": "VERIFIED",
            "signed_at": "2026-07-19T12:00:00+00:00",
            "signer": "Submitting author",
            "signature_algorithm": "Ed25519",
            "signer_public_key_sha256": hashlib.sha256(raw_public).hexdigest(),
        }
    )
    value["content_sha256"] = paper_artifact_content_sha256(value)
    value["signature_ed25519"] = base64.b64encode(
        private_key.sign(paper_artifact_signature_message(value))
    ).decode("ascii")
    return _write_json(path, value)


def _signed_policy_inputs(tmp_path: Path, reviewer_keys):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paper = Path(__file__).resolve().parents[1] / "paper"
    author_kit = tmp_path / "author-kit.zip"
    kit_members = {
        "kit/spconf.sty": b"% official fixture replacement\n"
        + (paper / "spconf.sty").read_bytes(),
        "kit/IEEEbib.bst": b"% official fixture replacement\n"
        + (paper / "IEEEbib.bst").read_bytes(),
        "kit/conference-template.tex": (
            b"% official author-kit template fixture\n\\documentclass{article}\n"
        ),
    }
    with zipfile.ZipFile(author_kit, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in kit_members.items():
            archive.writestr(name, content)
    author = _sign_paper_artifact(
        tmp_path / "author.json",
        {
            "schema_version": "paper-author-declaration-v1",
            "authors": [{"name": "A. Researcher", "affiliation_ids": ["lab"]}],
            "affiliations": [{"id": "lab", "text": "Signal Processing Laboratory"}],
            "funding_statement": "This work was supported by the Example Fund.",
            "conflict_of_interest_statement": "The authors declare no competing interests.",
            "compliance_statement": (
                "The study uses public de-identified data and collects no new participant data."
            ),
            "facts_verified_by_all_authors": True,
        },
        reviewer_keys,
    )
    tools = _sign_paper_artifact(
        tmp_path / "tools.json",
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
                    "version": "test-version",
                    "uses": ["code review", "language editing"],
                }
            ],
            "human_verification": {
                "verified_by": "Submitting author",
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
        reviewer_keys,
    )
    venue = _sign_paper_artifact(
        tmp_path / "venue.json",
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
                    "sha256": hashlib.sha256(kit_members["kit/IEEEbib.bst"]).hexdigest(),
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
        reviewer_keys,
    )
    return author, tools, venue, author_kit


def test_compile_rehashes_macro_figures_sources_summary_and_registry(tmp_path: Path) -> None:
    claims, figures = _bound_inputs(tmp_path)
    binding = validate_compilation_inputs(claims, figures)
    assert binding["claim_macros_sha256"] == lineage.artifact_sha256(
        claims / "robust_map_placeholders.tex"
    )
    assert binding["figures_summary_sha256"] == lineage.artifact_sha256(
        figures / "summary.v3.json"
    )
    assert binding["figure_artifacts_sha256"] == direct_artifact_hashes(figures)


@pytest.mark.parametrize(
    "group,name",
    (
        ("claims", "robust_map_placeholders.tex"),
        ("claims", "verified_registry.v1.json"),
        ("figures", "summary.v3.json"),
        *(("figures", name) for name in FIGURE_ARTIFACTS),
    ),
)
def test_compile_revalidation_fails_closed_on_any_bound_input_tamper(
    tmp_path: Path, group: str, name: str
) -> None:
    claims, figures = _bound_inputs(tmp_path)
    path = (claims if group == "claims" else figures) / name
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validate_compilation_inputs(claims, figures)


def test_signed_author_tool_and_venue_inputs_are_hash_bound(
    tmp_path: Path, reviewer_keys
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    binding = validate_submission_policy_inputs(
        author,
        tools,
        venue,
        reviewer_keys.public,
        author_kit,
    )
    assert binding["review_model"] == "single-anonymous"
    assert binding["author_identities_visible_to_reviewers"] is True
    assert binding["arc_formal_receipts_sha256"] == {}
    assert "A. Researcher" in binding["generated_tex"]["auto/author_declaration.tex"]
    assert "OpenAI Codex" in binding["generated_tex"]["auto/tool_provenance.tex"]
    assert binding["binding_sha256"] == lineage.canonical_sha256(
        {
            key: binding[key]
            for key in (
                "input_sha256",
                "generated_tex_sha256",
                "arc_formal_receipts_sha256",
                "author_content_sha256",
                "tool_content_sha256",
                "venue_content_sha256",
                "author_kit_members",
                "author_public_key_fingerprint_sha256",
                "ai_policy_attestation",
                "review_model",
                "author_identities_visible_to_reviewers",
            )
        }
    )
    assert binding["ai_policy_attestation"] == {
        "ai_systems_are_not_authors": True,
        "any_submitted_section_entirely_generated_by_ai": False,
        "most_or_significant_manuscript_components_generated_by_ai": False,
        "authors_substantively_rewrote_and_verified_ai_assisted_text": True,
        "manuscript_read_and_approved": True,
        "scientific_judgment_retained_by_authors": True,
    }
    disclosure = binding["generated_tex"]["auto/tool_provenance.tex"]
    assert "No submitted section was entirely generated by AI" in disclosure


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "any_submitted_section_entirely_generated_by_ai",
            True,
            "forbids entirely generated sections",
        ),
        (
            "most_or_significant_manuscript_components_generated_by_ai",
            True,
            "forbids entirely generated sections",
        ),
        (
            "authors_substantively_rewrote_and_verified_ai_assisted_text",
            False,
            "all human-verification attestations must be true",
        ),
        (
            "ai_systems_are_not_authors",
            False,
            "all human-verification attestations must be true",
        ),
    ),
)
def test_ai_policy_attestation_fails_closed(
    tmp_path: Path,
    reviewer_keys,
    field: str,
    value: bool,
    message: str,
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    tool_value = json.loads(tools.read_text(encoding="utf-8"))
    tool_value["human_verification"][field] = value
    tools = _sign_paper_artifact(
        tmp_path / f"tools-invalid-{field}.json", tool_value, reviewer_keys
    )
    with pytest.raises(ValueError, match=message):
        validate_submission_policy_inputs(
            author, tools, venue, reviewer_keys.public, author_kit
        )


def test_policy_inputs_fail_closed_on_tamper_pending_and_wrong_author_kit(
    tmp_path: Path, reviewer_keys
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    author_value = json.loads(author.read_text(encoding="utf-8"))
    author_value["funding_statement"] = "tampered"
    _write_json(author, author_value)
    with pytest.raises(ValueError, match="content_sha256"):
        validate_submission_policy_inputs(
            author, tools, venue, reviewer_keys.public, author_kit
        )

    author, tools, venue, author_kit = _signed_policy_inputs(
        tmp_path / "fresh", reviewer_keys
    )
    author_kit.write_bytes(b"changed after venue signature")
    with pytest.raises(ValueError, match="author-kit hash mismatch"):
        validate_submission_policy_inputs(
            author, tools, venue, reviewer_keys.public, author_kit
        )

    pending = Path(__file__).resolve().parents[1] / "paper" / "author_declaration.v1.template.json"
    with pytest.raises(ValueError, match="status must be VERIFIED"):
        validate_submission_policy_inputs(
            pending, tools, venue, reviewer_keys.public, author_kit
        )


def test_autoresearchclaw_disclosure_requires_all_formal_receipts(
    tmp_path: Path, reviewer_keys
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    tool_value = json.loads(tools.read_text(encoding="utf-8"))
    for field in ("content_sha256", "signature_ed25519"):
        tool_value.pop(field)
    tool_value["tools"].append(
        {
            "provider": "aiming-lab",
            "product": "AutoResearchClaw",
            "version": "v0.5.0",
            "uses": ["research control"],
        }
    )
    tools = _sign_paper_artifact(tmp_path / "tools-with-arc.json", tool_value, reviewer_keys)
    with pytest.raises(ValueError, match="formal Stage 5/9/15/20 receipts"):
        validate_submission_policy_inputs(
            author, tools, venue, reviewer_keys.public, author_kit
        )


def test_author_kit_rejects_traversal_and_member_hash_mismatch(
    tmp_path: Path, reviewer_keys
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    venue_value = json.loads(venue.read_text(encoding="utf-8"))
    for field in ("content_sha256", "signature_ed25519"):
        venue_value.pop(field)
    venue_value["author_kit_members"]["style"]["sha256"] = "0" * 64
    bad_member_venue = _sign_paper_artifact(
        tmp_path / "bad-member-venue.json", venue_value, reviewer_keys
    )
    with pytest.raises(ValueError, match="member hash mismatch"):
        validate_submission_policy_inputs(
            author, tools, bad_member_venue, reviewer_keys.public, author_kit
        )

    traversal_kit = tmp_path / "traversal-kit.zip"
    with zipfile.ZipFile(author_kit) as source, zipfile.ZipFile(
        traversal_kit, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for info in source.infolist():
            destination.writestr(info.filename, source.read(info.filename))
        destination.writestr("../escape.tex", b"unsafe")
    venue_value = json.loads(venue.read_text(encoding="utf-8"))
    for field in ("content_sha256", "signature_ed25519"):
        venue_value.pop(field)
    venue_value["author_kit_sha256"] = lineage.artifact_sha256(traversal_kit)
    traversal_venue = _sign_paper_artifact(
        tmp_path / "traversal-venue.json", venue_value, reviewer_keys
    )
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        validate_submission_policy_inputs(
            author, tools, traversal_venue, reviewer_keys.public, traversal_kit
        )


def test_author_kit_rejects_special_files_and_reused_role_members(
    tmp_path: Path, reviewer_keys
) -> None:
    author, tools, venue, author_kit = _signed_policy_inputs(tmp_path, reviewer_keys)
    special_kit = tmp_path / "special-kit.zip"
    with zipfile.ZipFile(author_kit) as source, zipfile.ZipFile(
        special_kit, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for info in source.infolist():
            destination.writestr(info.filename, source.read(info.filename))
        special = zipfile.ZipInfo("kit/unsupported-fifo")
        special.create_system = 3
        special.external_attr = (stat.S_IFIFO | 0o644) << 16
        destination.writestr(special, b"not a regular file")
    venue_value = json.loads(venue.read_text(encoding="utf-8"))
    venue_value["author_kit_sha256"] = lineage.artifact_sha256(special_kit)
    special_venue = _sign_paper_artifact(
        tmp_path / "special-venue.json", venue_value, reviewer_keys
    )
    with pytest.raises(ValueError, match="special files"):
        validate_submission_policy_inputs(
            author, tools, special_venue, reviewer_keys.public, special_kit
        )

    venue_value = json.loads(venue.read_text(encoding="utf-8"))
    bibliography_member = venue_value["author_kit_members"]["bibliography_style"]
    venue_value["author_kit_members"]["style"]["member_path"] = (
        bibliography_member["member_path"]
    )
    venue_value["author_kit_members"]["style"]["sha256"] = bibliography_member[
        "sha256"
    ]
    reused_venue = _sign_paper_artifact(
        tmp_path / "reused-member-venue.json", venue_value, reviewer_keys
    )
    with pytest.raises(ValueError, match="distinct ZIP members"):
        validate_submission_policy_inputs(
            author, tools, reused_venue, reviewer_keys.public, author_kit
        )


def test_page_five_source_allows_only_references_funding_and_compliance(
    tmp_path: Path,
) -> None:
    paper = Path(__file__).resolve().parents[1] / "paper"
    validate_page_five_source(paper / "main_v2.tex", paper / "compliance.tex")
    main = tmp_path / "main.tex"
    main.write_text(
        (paper / "main_v2.tex")
        .read_text(encoding="utf-8")
        .replace(r"\bibliography{refs}", r"\bibliography{refs}\nTechnical appendix"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allowlist"):
        validate_page_five_source(main, paper / "compliance.tex")


def test_real_five_page_build_binds_declarations_and_build_report(
    tmp_path: Path, reviewer_keys, monkeypatch
) -> None:
    claims, figures = _bound_inputs(tmp_path, compilable=True)
    author, tools, venue, author_kit = _signed_policy_inputs(
        tmp_path / "policy", reviewer_keys
    )
    output = tmp_path / "submission"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_submission_v3.py",
            "--claims",
            str(claims),
            "--figures",
            str(figures),
            "--author-declaration",
            str(author),
            "--tool-provenance",
            str(tools),
            "--venue-policy",
            str(venue),
            "--author-public-key",
            str(reviewer_keys.public),
            "--author-kit",
            str(author_kit),
            "--output-dir",
            str(output),
        ],
    )
    compile_submission()
    report_path = output / "build_report.v3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pages"] == 5
    assert report["technical_content_end_page"] == 4
    assert report["page_five_content_validated"] is True
    assert report["overfull_boxes"] == 0
    assert report["review_model"] == "single-anonymous"
    assert report["author_identities_visible_to_reviewers"] is True
    assert report["paper_policy_input_sha256"]["author_declaration"] == (
        lineage.artifact_sha256(author)
    )
    with zipfile.ZipFile(author_kit) as archive:
        expected_kit_outputs = {
            "style": archive.read("kit/spconf.sty"),
            "bibliography_style": archive.read("kit/IEEEbib.bst"),
            "latex_template": archive.read("kit/conference-template.tex"),
        }
    paper = Path(__file__).resolve().parents[1] / "paper"
    for role, expected in expected_kit_outputs.items():
        relative = report["author_kit_members"][role]["build_target"]
        compiled = output / "build" / relative
        assert compiled.read_bytes() == expected
        assert report["author_kit_members"][role]["sha256"] == hashlib.sha256(
            expected
        ).hexdigest()
    assert (output / "build" / "spconf.sty").read_bytes() != (
        paper / "spconf.sty"
    ).read_bytes()
    assert (output / "build" / "IEEEbib.bst").read_bytes() != (
        paper / "IEEEbib.bst"
    ).read_bytes()
    assert report["ai_policy_attestation"][
        "any_submitted_section_entirely_generated_by_ai"
    ] is False
    assert report["ai_policy_attestation"][
        "most_or_significant_manuscript_components_generated_by_ai"
    ] is False
    detached = (output / "build_report.v3.sha256").read_text(encoding="ascii")
    assert detached == f"{lineage.artifact_sha256(report_path)}  build_report.v3.json\n"
    from pypdf import PdfReader

    draft_output = tmp_path / "stage20-review-draft"
    draft_report = build_stage20_review_draft(
        claims_dir=claims, figures_dir=figures, output_dir=draft_output
    )
    assert draft_report["schema_version"] == "stage20-review-draft-v3"
    assert draft_report["review_ready"] is True
    assert draft_report["submission_ready"] is False
    assert draft_report["release_eligible"] is False
    assert draft_report["not_for_submission"] is True
    assert draft_report["pages"] == 5
    assert draft_report["technical_content_end_page"] == 4
    assert draft_report["page_five_content_validated"] is True
    assert draft_report["overfull_boxes"] == 0
    assert draft_report["identity_claimed"] is False
    assert draft_report["funding_claimed"] is False
    assert draft_report["final_ai_provenance_claimed"] is False
    assert draft_report["official_author_kit_claimed"] is False
    assert draft_report["reviewed_scientific_input_sha256"] == report[
        "reviewed_scientific_input_sha256"
    ]
    assert draft_report["reviewed_scientific_input_bundle_sha256"] == report[
        "reviewed_scientific_input_bundle_sha256"
    ]
    assert report["reviewed_scientific_input_bundle_sha256"] == (
        lineage.canonical_sha256(report["reviewed_scientific_input_sha256"])
    )
    assert set(report["reviewed_scientific_input_sha256"]) == {
        "paper/main_v2.tex",
        "paper/compliance.tex",
        "paper/refs.bib",
        "claims/claims.v3.json",
        "claims/robust_map_placeholders.tex",
        "claims/verified_registry.v1.json",
        "figures/summary.v3.json",
        *(f"figures/{name}" for name in FIGURE_ARTIFACTS),
    }
    pages = [
        " ".join((page.extract_text() or "").split())
        for page in PdfReader(str(draft_output / "stage20_review_draft.pdf")).pages
    ]
    assert "REVIEW DRAFT" in pages[0] and "NOT FOR SUBMISSION" in pages[0]
    assert "REVIEW DRAFT" in pages[4] and "NOT FOR SUBMISSION" in pages[4]
    assert "A. Researcher" not in " ".join(pages)
    submission_pages = [
        " ".join((page.extract_text() or "").split())
        for page in PdfReader(str(output / "main_v2.pdf")).pages
    ]
    assert "No submitted section was entirely generated by AI" in submission_pages[4]
    draft_report_path = draft_output / "review_draft_report.v3.json"
    draft_detached = (draft_output / "review_draft_report.v3.sha256").read_text(
        encoding="ascii"
    )
    assert draft_detached == (
        f"{lineage.artifact_sha256(draft_report_path)}  review_draft_report.v3.json\n"
    )
