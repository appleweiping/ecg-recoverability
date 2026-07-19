"""Compile the five-page ICASSP paper from signed, evidence-gated inputs."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Any, Mapping
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ecgcert import lineage
from ecgcert.arc_control import ORDERED_STAGES, validate_arc_control_chain
from ecgcert.paper_evidence import (
    FIGURE_ARTIFACTS,
    require_artifact_hashes,
    validate_figure_bundle,
)


AUTHOR_SCHEMA = "paper-author-declaration-v1"
TOOL_SCHEMA = "paper-tool-provenance-v1"
VENUE_SCHEMA = "paper-venue-policy-v1"
SIGNED_STATUS = "VERIFIED"
SIGNATURE_ALGORITHM = "Ed25519"
PAGE_FIVE_ALLOWED_CONTENT = (
    "references",
    "funding acknowledgments",
    "Compliance with Ethical Standards",
)
PAPER_SOURCE_FILES = (
    "main_v2.tex",
    "compliance.tex",
    "refs.bib",
    "spconf.sty",
    "IEEEbib.bst",
)
MANUSCRIPT_SOURCE_FILES = (
    "main_v2.tex",
    "compliance.tex",
    "refs.bib",
)
AUTHOR_KIT_BUILD_TARGETS = {
    "style": "spconf.sty",
    "bibliography_style": "IEEEbib.bst",
    "latex_template": "author_kit/official-template.tex",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TECHNICAL_HEADINGS = (
    "Motivation and contribution",
    "Gaussian posterior ambiguity",
    "Locked evaluation",
    "Evidence gate and responsible scope",
    "Conclusion",
)
REQUIRED_RESULT_MACROS = (
    "ResultHeadline",
    "ResultPrimaryAssociation",
    "ResultIncrementalValue",
    "ResultRankWeightStability",
    "ResultExternalAssociation",
    "ResultModelCoverage",
    "ResultBootstrapUncertainty",
    "ResultConclusion",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _full_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{field} must be a full lowercase SHA-256")
    return value


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise ValueError(f"{field} contains a control character")
    return " ".join(value.split())


def _timestamp(value: Any, *, field: str) -> str:
    rendered = _nonempty_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return rendered


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field} keys differ; missing={missing}, extra={extra}")


def _load_public_key(path: Path) -> Ed25519PublicKey:
    raw = path.resolve(strict=True).read_bytes()
    try:
        key = serialization.load_ssh_public_key(raw)
    except ValueError:
        key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("author public key must be an Ed25519 public key")
    return key


def _public_key_sha256(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def paper_artifact_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash the signed artifact payload, excluding its hash and signature fields."""

    payload = dict(value)
    payload.pop("content_sha256", None)
    payload.pop("signature_ed25519", None)
    return lineage.canonical_sha256(payload)


def paper_artifact_signature_message(value: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes covered by a paper-input Ed25519 signature."""

    payload = dict(value)
    payload.pop("signature_ed25519", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_signed_artifact(
    path: Path,
    *,
    schema: str,
    public_key: Ed25519PublicKey,
) -> dict[str, Any]:
    value = _load_object(path.resolve(strict=True))
    if value.get("schema_version") != schema:
        raise ValueError(f"{path.name} must use {schema}")
    status = value.get("status")
    if status != SIGNED_STATUS:
        raise ValueError(f"{path.name} status must be VERIFIED; got {status!r}")
    if value.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        raise ValueError(f"{path.name} must use an Ed25519 signature")
    _nonempty_text(value.get("signer"), field=f"{path.name}.signer")
    _timestamp(value.get("signed_at"), field=f"{path.name}.signed_at")
    fingerprint = _full_sha256(
        value.get("signer_public_key_sha256"),
        field=f"{path.name}.signer_public_key_sha256",
    )
    if fingerprint != _public_key_sha256(public_key):
        raise ValueError(f"{path.name} is not bound to the pinned author public key")
    content_sha256 = _full_sha256(
        value.get("content_sha256"), field=f"{path.name}.content_sha256"
    )
    if content_sha256 != paper_artifact_content_sha256(value):
        raise ValueError(f"{path.name} content_sha256 does not match its payload")
    encoded = value.get("signature_ed25519")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{path.name} signature_ed25519 is missing")
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path.name} signature is not valid base64") from error
    if len(signature) != 64:
        raise ValueError(f"{path.name} Ed25519 signature has an invalid length")
    try:
        public_key.verify(signature, paper_artifact_signature_message(value))
    except InvalidSignature as error:
        raise ValueError(f"{path.name} Ed25519 signature is invalid") from error
    return value


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _tex_command(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{_tex_escape(value)}}}"


def _validate_author_declaration(
    path: Path, public_key: Ed25519PublicKey
) -> tuple[dict[str, Any], str]:
    value = _validate_signed_artifact(path, schema=AUTHOR_SCHEMA, public_key=public_key)
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "signed_at",
            "signer",
            "signature_algorithm",
            "signer_public_key_sha256",
            "content_sha256",
            "signature_ed25519",
            "authors",
            "affiliations",
            "funding_statement",
            "conflict_of_interest_statement",
            "compliance_statement",
            "facts_verified_by_all_authors",
        },
        field="author declaration",
    )
    if value["facts_verified_by_all_authors"] is not True:
        raise ValueError("author declaration facts must be verified by all authors")
    authors = value["authors"]
    affiliations = value["affiliations"]
    if not isinstance(authors, list) or not authors:
        raise ValueError("author declaration authors must be a non-empty list")
    if not isinstance(affiliations, list) or not affiliations:
        raise ValueError("author declaration affiliations must be a non-empty list")
    affiliation_text: dict[str, str] = {}
    for index, affiliation in enumerate(affiliations):
        if not isinstance(affiliation, Mapping):
            raise ValueError(f"affiliations[{index}] must be an object")
        _exact_keys(affiliation, {"id", "text"}, field=f"affiliations[{index}]")
        affiliation_id = _nonempty_text(
            affiliation["id"], field=f"affiliations[{index}].id"
        )
        if affiliation_id in affiliation_text:
            raise ValueError("affiliation ids must be unique")
        affiliation_text[affiliation_id] = _nonempty_text(
            affiliation["text"], field=f"affiliations[{index}].text"
        )
    author_names: list[str] = []
    referenced_affiliations: set[str] = set()
    for index, author in enumerate(authors):
        if not isinstance(author, Mapping):
            raise ValueError(f"authors[{index}] must be an object")
        _exact_keys(author, {"name", "affiliation_ids"}, field=f"authors[{index}]")
        author_names.append(_nonempty_text(author["name"], field=f"authors[{index}].name"))
        ids = author["affiliation_ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"authors[{index}].affiliation_ids must be non-empty")
        for raw_id in ids:
            affiliation_id = _nonempty_text(raw_id, field=f"authors[{index}].affiliation_ids")
            if affiliation_id not in affiliation_text:
                raise ValueError(f"authors[{index}] references an unknown affiliation")
            referenced_affiliations.add(affiliation_id)
    if referenced_affiliations != set(affiliation_text):
        raise ValueError("every declared affiliation must be referenced by an author")
    funding = _nonempty_text(value["funding_statement"], field="funding_statement")
    conflict = _nonempty_text(
        value["conflict_of_interest_statement"],
        field="conflict_of_interest_statement",
    )
    compliance = _nonempty_text(
        value["compliance_statement"], field="compliance_statement"
    )
    rendered = "\n".join(
        (
            "% Generated from an authenticated author declaration; do not edit.",
            _tex_command("SubmissionAuthorNames", ", ".join(author_names)),
            _tex_command(
                "SubmissionAuthorAffiliations", "; ".join(affiliation_text.values())
            ),
            _tex_command("SubmissionFundingStatement", funding),
            _tex_command("SubmissionConflictStatement", conflict),
            _tex_command("SubmissionComplianceStatement", compliance),
            "",
        )
    )
    return value, rendered


def _safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    rendered = _nonempty_text(value, field=field)
    if "\\" in rendered or "\x00" in rendered or ":" in rendered:
        raise ValueError(f"{field} must use a safe POSIX relative path")
    if any(part in {"", ".", ".."} for part in rendered.split("/")):
        raise ValueError(f"{field} must use a safe POSIX relative path")
    path = PurePosixPath(rendered)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must use a safe POSIX relative path")
    return path


def _validate_author_kit_zip(
    archive_path: Path, raw_members: Any
) -> dict[str, dict[str, str]]:
    if not isinstance(raw_members, Mapping):
        raise ValueError("author_kit_members must be an object")
    if set(raw_members) != set(AUTHOR_KIT_BUILD_TARGETS):
        raise ValueError("author kit must bind style, bibliography_style, and latex_template")
    if archive_path.suffix.casefold() != ".zip" or not zipfile.is_zipfile(archive_path):
        raise ValueError("official author kit must be a valid ZIP archive")
    normalized: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > 2_048:
            raise ValueError("official author kit has an unsafe member count")
        seen: set[str] = set()
        total_size = 0
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            raw_name = info.filename
            candidate = raw_name[:-1] if raw_name.endswith("/") else raw_name
            relative = _safe_relative_path(candidate, field="author-kit ZIP member")
            canonical = relative.as_posix() + ("/" if info.is_dir() else "")
            if canonical != raw_name:
                raise ValueError("author-kit ZIP member path is not canonical")
            if raw_name in seen:
                raise ValueError("author-kit ZIP contains duplicate member names")
            seen.add(raw_name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError("author-kit ZIP cannot contain symbolic links")
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("author-kit ZIP cannot contain special files")
            if info.is_dir() != (mode == stat.S_IFDIR) and mode != 0:
                raise ValueError("author-kit ZIP member type is inconsistent")
            if info.flag_bits & 0x1:
                raise ValueError("author-kit ZIP cannot contain encrypted members")
            if info.file_size < 0 or info.file_size > 64 * 1024 * 1024:
                raise ValueError("author-kit ZIP member exceeds the size limit")
            total_size += info.file_size
            if total_size > 256 * 1024 * 1024:
                raise ValueError("author-kit ZIP exceeds the uncompressed size limit")
            if not info.is_dir():
                by_name[raw_name] = info
        for role, descriptor in raw_members.items():
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"author_kit_members.{role} must be an object")
            _exact_keys(
                descriptor,
                {"member_path", "sha256", "build_target"},
                field=f"author_kit_members.{role}",
            )
            relative = _safe_relative_path(
                descriptor["member_path"],
                field=f"author_kit_members.{role}.member_path",
            )
            member_path = relative.as_posix()
            if member_path not in by_name:
                raise ValueError(f"author-kit member for {role} is missing")
            build_target = _nonempty_text(
                descriptor["build_target"],
                field=f"author_kit_members.{role}.build_target",
            )
            if build_target != AUTHOR_KIT_BUILD_TARGETS[role]:
                raise ValueError(f"author-kit build target for {role} is not permitted")
            expected = _full_sha256(
                descriptor["sha256"], field=f"author_kit_members.{role}.sha256"
            )
            with archive.open(by_name[member_path], "r") as stream:
                actual = hashlib.sha256()
                while chunk := stream.read(1 << 20):
                    actual.update(chunk)
            if actual.hexdigest() != expected:
                raise ValueError(f"author-kit member hash mismatch for {role}")
            normalized[role] = {
                "member_path": member_path,
                "sha256": expected,
                "build_target": build_target,
            }
    selected_paths = [descriptor["member_path"] for descriptor in normalized.values()]
    if len(set(selected_paths)) != len(selected_paths):
        raise ValueError("author-kit roles must bind distinct ZIP members")
    return normalized


def _validate_tool_provenance(
    path: Path,
    public_key: Ed25519PublicKey,
    *,
    arc_receipts_root: Path | None,
) -> tuple[dict[str, Any], str, dict[int, Path]]:
    value = _validate_signed_artifact(path, schema=TOOL_SCHEMA, public_key=public_key)
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "signed_at",
            "signer",
            "signature_algorithm",
            "signer_public_key_sha256",
            "content_sha256",
            "signature_ed25519",
            "sps_policy_url",
            "tools",
            "human_verification",
            "autoresearchclaw_formal_receipts",
        },
        field="tool provenance",
    )
    policy_url = _nonempty_text(value["sps_policy_url"], field="sps_policy_url")
    if not policy_url.startswith("https://signalprocessingsociety.org/"):
        raise ValueError("tool provenance must cite the official IEEE SPS LLM policy")
    tools = value["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool provenance tools must be a non-empty list")
    rendered_tools: list[str] = []
    arc_declared = False
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"tools[{index}] must be an object")
        _exact_keys(
            tool,
            {"provider", "product", "version", "uses"},
            field=f"tools[{index}]",
        )
        provider = _nonempty_text(tool["provider"], field=f"tools[{index}].provider")
        product = _nonempty_text(tool["product"], field=f"tools[{index}].product")
        version = _nonempty_text(tool["version"], field=f"tools[{index}].version")
        uses = tool["uses"]
        if not isinstance(uses, list) or not uses:
            raise ValueError(f"tools[{index}].uses must be a non-empty list")
        clean_uses = [
            _nonempty_text(item, field=f"tools[{index}].uses") for item in uses
        ]
        arc_declared = arc_declared or product.casefold() == "autoresearchclaw"
        rendered_tools.append(
            f"{provider} {product} ({version}) assisted with {', '.join(clean_uses)}."
        )
    verification = value["human_verification"]
    if not isinstance(verification, Mapping):
        raise ValueError("human_verification must be an object")
    required_verification = {
        "verified_by",
        "verified_at",
        "citations_checked_against_primary_sources",
        "numerical_claims_traced_to_artifacts",
        "code_and_statistics_reviewed",
        "manuscript_read_and_approved",
        "scientific_judgment_retained_by_authors",
        "ai_systems_are_not_authors",
        "any_submitted_section_entirely_generated_by_ai",
        "most_or_significant_manuscript_components_generated_by_ai",
        "authors_substantively_rewrote_and_verified_ai_assisted_text",
    }
    _exact_keys(verification, required_verification, field="human_verification")
    _nonempty_text(verification["verified_by"], field="human_verification.verified_by")
    _timestamp(verification["verified_at"], field="human_verification.verified_at")
    true_fields = required_verification - {
        "verified_by",
        "verified_at",
        "any_submitted_section_entirely_generated_by_ai",
        "most_or_significant_manuscript_components_generated_by_ai",
    }
    if any(verification[field] is not True for field in true_fields):
        raise ValueError("all human-verification attestations must be true")
    if any(
        verification[field] is not False
        for field in (
            "any_submitted_section_entirely_generated_by_ai",
            "most_or_significant_manuscript_components_generated_by_ai",
        )
    ):
        raise ValueError(
            "IEEE SPS policy forbids entirely generated sections and mostly AI-generated papers"
        )

    raw_receipts = value["autoresearchclaw_formal_receipts"]
    if not isinstance(raw_receipts, Mapping):
        raise ValueError("autoresearchclaw_formal_receipts must be an object")
    receipt_sources: dict[int, Path] = {}
    if arc_declared:
        if set(raw_receipts) != {str(stage) for stage in ORDERED_STAGES}:
            raise ValueError("AutoResearchClaw use requires formal Stage 5/9/15/20 receipts")
        if arc_receipts_root is None:
            raise ValueError("AutoResearchClaw use requires --arc-receipts-root")
        receipt_root = arc_receipts_root.resolve(strict=True)
        reports: list[dict[str, Any]] = []
        for stage in ORDERED_STAGES:
            descriptor = raw_receipts[str(stage)]
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"ARC Stage {stage} receipt descriptor must be an object")
            _exact_keys(descriptor, {"path", "sha256"}, field=f"ARC Stage {stage} receipt")
            relative = _safe_relative_path(
                descriptor["path"], field=f"ARC Stage {stage} receipt.path"
            )
            expected = _full_sha256(
                descriptor["sha256"], field=f"ARC Stage {stage} receipt.sha256"
            )
            source = receipt_root.joinpath(*relative.parts).resolve(strict=True)
            if not source.is_file() or source.is_symlink() or not source.is_relative_to(receipt_root):
                raise ValueError(f"ARC Stage {stage} receipt path is unsafe")
            if lineage.artifact_sha256(source) != expected:
                raise ValueError(f"ARC Stage {stage} formal receipt hash mismatch")
            reports.append(_load_object(source))
            receipt_sources[stage] = source
        validate_arc_control_chain(reports)
        rendered_tools.append(
            "Formal AutoResearchClaw Stage 5, 9, 15, and 20 co-pilot receipts were validated."
        )
    elif raw_receipts:
        raise ValueError("ARC receipts cannot be claimed without an AutoResearchClaw tool entry")
    elif arc_receipts_root is not None:
        raise ValueError("--arc-receipts-root was provided but ARC use is not declared")

    human_statement = (
        "The authors checked primary-source citations, traced numerical claims to authenticated "
        "artifacts, reviewed code and statistics, and substantively rewrote and verified any "
        "AI-assisted text before reading and approving the manuscript. No submitted section was "
        "entirely generated by AI, and AI did not generate most or significant manuscript "
        "components. Scientific judgment remained with the authors; AI systems are not authors."
    )
    rendered = "\n".join(
        (
            "% Generated from authenticated tool provenance; do not edit.",
            _tex_command("SubmissionToolDisclosure", " ".join(rendered_tools)),
            _tex_command("SubmissionHumanVerificationStatement", human_statement),
            "",
        )
    )
    return value, rendered, receipt_sources


def _validate_venue_policy(
    path: Path,
    public_key: Ed25519PublicKey,
    *,
    author_kit: Path,
) -> tuple[dict[str, Any], str, dict[str, dict[str, str]]]:
    value = _validate_signed_artifact(path, schema=VENUE_SCHEMA, public_key=public_key)
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "signed_at",
            "signer",
            "signature_algorithm",
            "signer_public_key_sha256",
            "content_sha256",
            "signature_ed25519",
            "venue",
            "policy_url",
            "author_kit_url",
            "author_kit_sha256",
            "author_kit_status",
            "author_kit_members",
            "policy_checked_at",
            "review_model",
            "author_identities_visible_to_reviewers",
            "technical_page_limit",
            "total_page_limit",
            "page_five_allowed_content",
        },
        field="venue policy",
    )
    if _nonempty_text(value["venue"], field="venue") != "ICASSP 2027":
        raise ValueError("venue policy must target ICASSP 2027")
    policy_url = _nonempty_text(value["policy_url"], field="policy_url")
    author_kit_url = _nonempty_text(value["author_kit_url"], field="author_kit_url")
    if not policy_url.startswith("https://2027.ieeeicassp.org/"):
        raise ValueError("venue policy must cite an official ICASSP 2027 policy URL")
    if not author_kit_url.startswith("https://2027.ieeeicassp.org/"):
        raise ValueError("venue policy must cite the official ICASSP 2027 author kit")
    if value["author_kit_status"] != "OFFICIAL_PUBLISHED":
        raise ValueError("official ICASSP 2027 author kit remains unresolved")
    _timestamp(value["policy_checked_at"], field="policy_checked_at")
    kit_path = author_kit.resolve(strict=True)
    if not kit_path.is_file() or kit_path.is_symlink():
        raise ValueError("official author kit must be a non-symlink regular file")
    expected_kit = _full_sha256(value["author_kit_sha256"], field="author_kit_sha256")
    if lineage.artifact_sha256(kit_path) != expected_kit:
        raise ValueError("official author-kit hash mismatch")
    kit_members = _validate_author_kit_zip(kit_path, value["author_kit_members"])
    review_model = value["review_model"]
    if review_model not in {"single-anonymous", "double-anonymous"}:
        raise ValueError("review_model must be single-anonymous or double-anonymous")
    visible = value["author_identities_visible_to_reviewers"]
    if not isinstance(visible, bool):
        raise ValueError("author_identities_visible_to_reviewers must be boolean")
    if visible != (review_model == "single-anonymous"):
        raise ValueError("review model and author-identity visibility are inconsistent")
    if value["technical_page_limit"] != 4 or value["total_page_limit"] != 5:
        raise ValueError("ICASSP 2027 page limits must be four technical pages and five total")
    if value["page_five_allowed_content"] != list(PAGE_FIVE_ALLOWED_CONTENT):
        raise ValueError("venue policy has an invalid page-five content allowlist")
    rendered = "\n".join(
        (
            "% Generated from an authenticated venue-policy artifact; do not edit.",
            r"\newif\ifSubmissionAuthorIdentitiesVisible",
            (
                r"\SubmissionAuthorIdentitiesVisibletrue"
                if visible
                else r"\SubmissionAuthorIdentitiesVisiblefalse"
            ),
            _tex_command("SubmissionReviewModel", review_model),
            r"\newcommand{\SubmissionBuildTitleSuffix}{}",
            "",
        )
    )
    return value, rendered, kit_members


def validate_submission_policy_inputs(
    author_declaration: Path,
    tool_provenance: Path,
    venue_policy: Path,
    author_public_key: Path,
    author_kit: Path,
    *,
    arc_receipts_root: Path | None = None,
) -> dict[str, Any]:
    """Validate and hash every signed declaration needed by the paper build."""

    public_key_path = author_public_key.resolve(strict=True)
    public_key = _load_public_key(public_key_path)
    author_path = author_declaration.resolve(strict=True)
    tool_path = tool_provenance.resolve(strict=True)
    venue_path = venue_policy.resolve(strict=True)
    kit_path = author_kit.resolve(strict=True)
    author, author_tex = _validate_author_declaration(author_path, public_key)
    tools, tool_tex, receipts = _validate_tool_provenance(
        tool_path, public_key, arc_receipts_root=arc_receipts_root
    )
    venue, venue_tex, kit_members = _validate_venue_policy(
        venue_path, public_key, author_kit=kit_path
    )
    input_sha256 = {
        "author_declaration": lineage.artifact_sha256(author_path),
        "tool_provenance": lineage.artifact_sha256(tool_path),
        "venue_policy": lineage.artifact_sha256(venue_path),
        "author_public_key": lineage.artifact_sha256(public_key_path),
        "author_kit": lineage.artifact_sha256(kit_path),
    }
    generated_tex = {
        "auto/author_declaration.tex": author_tex,
        "auto/tool_provenance.tex": tool_tex,
        "auto/venue_policy.tex": venue_tex,
    }
    generated_tex_sha256 = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in generated_tex.items()
    }
    receipt_sha256 = {
        str(stage): lineage.artifact_sha256(path)
        for stage, path in sorted(receipts.items())
    }
    binding = {
        "input_sha256": input_sha256,
        "generated_tex_sha256": generated_tex_sha256,
        "arc_formal_receipts_sha256": receipt_sha256,
        "author_content_sha256": author["content_sha256"],
        "tool_content_sha256": tools["content_sha256"],
        "venue_content_sha256": venue["content_sha256"],
        "author_kit_members": kit_members,
        "author_public_key_fingerprint_sha256": _public_key_sha256(public_key),
        "ai_policy_attestation": {
            field: tools["human_verification"][field]
            for field in (
                "ai_systems_are_not_authors",
                "any_submitted_section_entirely_generated_by_ai",
                "most_or_significant_manuscript_components_generated_by_ai",
                "authors_substantively_rewrote_and_verified_ai_assisted_text",
                "manuscript_read_and_approved",
                "scientific_judgment_retained_by_authors",
            )
        },
        "review_model": venue["review_model"],
        "author_identities_visible_to_reviewers": venue[
            "author_identities_visible_to_reviewers"
        ],
    }
    return {
        **binding,
        "binding_sha256": lineage.canonical_sha256(binding),
        "generated_tex": generated_tex,
        "source_paths": {
            "author_declaration": author_path,
            "tool_provenance": tool_path,
            "venue_policy": venue_path,
            "author_public_key": public_key_path,
            "author_kit": kit_path,
        },
        "arc_receipt_paths": receipts,
    }


def validate_compilation_inputs(claims_dir: Path, figures_dir: Path) -> dict[str, Any]:
    """Re-hash every claim and figure input before it may enter the TeX build."""

    claims_dir = claims_dir.resolve()
    figures_dir = figures_dir.resolve()
    claims_path = claims_dir / "claims.v3.json"
    macros_path = claims_dir / "robust_map_placeholders.tex"
    registry_path = claims_dir / "verified_registry.v1.json"
    figures_summary_path = figures_dir / "summary.v3.json"
    for path in (claims_path, macros_path, registry_path, figures_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    claims = _load_object(claims_path)
    registry = _load_object(registry_path)
    figures_summary = _load_object(figures_summary_path)
    if claims.get("schema_version") != "paper-claims-v3":
        raise ValueError("submission build requires paper-claims-v3")
    figure_binding = validate_figure_bundle(figures_dir, summary=figures_summary)
    actual_macros_sha256 = lineage.artifact_sha256(macros_path)
    actual_registry_sha256 = lineage.artifact_sha256(registry_path)
    claimed_artifacts = require_artifact_hashes(
        claims.get("figure_artifacts_sha256"), label="claim-sync figure binding"
    )
    checks = {
        "figure_stage15": (
            figures_summary.get("input_sha256", {}).get("stage15")
            == claims.get("stage15_sha256")
        ),
        "claim_macros": claims.get("claim_macros_sha256") == actual_macros_sha256,
        "figures_summary": (
            claims.get("figures_summary_sha256") == figure_binding["summary_sha256"]
            and claims.get("figures_sha256") == figure_binding["summary_sha256"]
        ),
        "figure_artifacts": claimed_artifacts == figure_binding["artifacts_sha256"],
        "verified_registry": claims.get("verified_registry_sha256")
        == actual_registry_sha256,
        "registry_claim_macros": registry.get("claim_macros_sha256")
        == actual_macros_sha256,
        "registry_figures_summary": registry.get("figures_summary_sha256")
        == figure_binding["summary_sha256"],
        "registry_figure_artifacts": require_artifact_hashes(
            registry.get("figure_artifacts_sha256"),
            label="dynamic registry figure binding",
        )
        == figure_binding["artifacts_sha256"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("paper evidence binding mismatch: " + ", ".join(failed))
    return {
        "claims": claims,
        "claims_sha256": lineage.artifact_sha256(claims_path),
        "claim_macros_sha256": actual_macros_sha256,
        "verified_registry_sha256": actual_registry_sha256,
        "figures_summary_sha256": figure_binding["summary_sha256"],
        "figure_artifacts_sha256": figure_binding["artifacts_sha256"],
    }


def require_reviewed_claim_macros(claims: Mapping[str, Any], macros_path: Path) -> None:
    """Reject pending or incomplete claim text from both review and final PDFs."""

    if claims.get("submission_ready") is not True:
        raise ValueError("paper build requires synchronized reviewed Stage 15 claims")
    if claims.get("status") not in {"PROCEED", "PIVOT"}:
        raise ValueError("paper build requires a reviewed PROCEED or PIVOT decision")
    text = macros_path.resolve(strict=True).read_text(encoding="utf-8")
    if re.search(r"pending|PendingStageFifteen", text, flags=re.IGNORECASE):
        raise ValueError("paper build cannot contain pending Stage 15 claim macros")
    for macro in REQUIRED_RESULT_MACROS:
        if not re.search(rf"\\newcommand\{{\\{macro}\}}\{{[^}}]+\}}", text):
            raise ValueError(f"reviewed claim macro is missing or empty: {macro}")


def _paper_source_hashes(paper_source: Path) -> dict[str, str]:
    return {
        f"source/{name}": lineage.artifact_sha256(paper_source / name)
        for name in PAPER_SOURCE_FILES
    }


def _manuscript_source_hashes(paper_source: Path) -> dict[str, str]:
    return {
        f"source/{name}": lineage.artifact_sha256(paper_source / name)
        for name in MANUSCRIPT_SOURCE_FILES
    }


def _strict_source_hashes(
    paper_source: Path, policy: Mapping[str, Any]
) -> dict[str, str]:
    members = policy["author_kit_members"]
    return {
        **_manuscript_source_hashes(paper_source),
        "source/spconf.sty": members["style"]["sha256"],
        "source/IEEEbib.bst": members["bibliography_style"]["sha256"],
        "author_kit/official-template.tex": members["latex_template"]["sha256"],
    }


def reviewed_scientific_input_sha256(
    paper_source: Path, evidence: Mapping[str, Any]
) -> dict[str, str]:
    """Hash the scientific inputs that must remain identical after Stage 20 review."""

    return {
        **{
            f"paper/{name}": lineage.artifact_sha256(paper_source / name)
            for name in MANUSCRIPT_SOURCE_FILES
        },
        "claims/claims.v3.json": evidence["claims_sha256"],
        "claims/robust_map_placeholders.tex": evidence["claim_macros_sha256"],
        "claims/verified_registry.v1.json": evidence["verified_registry_sha256"],
        "figures/summary.v3.json": evidence["figures_summary_sha256"],
        **{
            f"figures/{name}": evidence["figure_artifacts_sha256"][name]
            for name in FIGURE_ARTIFACTS
        },
    }


def _compiled_input_hashes(build: Path, arc_stages: tuple[int, ...]) -> dict[str, str]:
    paths: dict[str, Path] = {
        **{f"source/{name}": build / name for name in PAPER_SOURCE_FILES},
        "auto/robust_map_placeholders.tex": build
        / "auto"
        / "robust_map_placeholders.tex",
        "figures_v3/summary.v3.json": build / "figures_v3" / "summary.v3.json",
        **{
            f"figures_v3/{name}": build / "figures_v3" / name
            for name in FIGURE_ARTIFACTS
        },
        "provenance/author_declaration.v1.json": build
        / "provenance"
        / "author_declaration.v1.json",
        "provenance/tool_provenance.v1.json": build
        / "provenance"
        / "tool_provenance.v1.json",
        "provenance/venue_policy.v1.json": build
        / "provenance"
        / "venue_policy.v1.json",
        "provenance/author_public_key": build / "provenance" / "author_public_key",
        "provenance/author_kit.source": build / "provenance" / "author_kit.source",
        "auto/author_declaration.tex": build / "auto" / "author_declaration.tex",
        "auto/tool_provenance.tex": build / "auto" / "tool_provenance.tex",
        "auto/venue_policy.tex": build / "auto" / "venue_policy.tex",
        "author_kit/official-template.tex": build
        / "author_kit"
        / "official-template.tex",
        **{
            f"provenance/arc-stage{stage}-report.v1.json": build
            / "provenance"
            / f"arc-stage{stage}-report.v1.json"
            for stage in arc_stages
        },
    }
    return {name: lineage.artifact_sha256(path) for name, path in paths.items()}


def _expected_compiled_input_hashes(
    evidence: Mapping[str, Any],
    policy: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, str]:
    return {
        **source_hashes,
        "auto/robust_map_placeholders.tex": evidence["claim_macros_sha256"],
        "figures_v3/summary.v3.json": evidence["figures_summary_sha256"],
        **{
            f"figures_v3/{name}": evidence["figure_artifacts_sha256"][name]
            for name in FIGURE_ARTIFACTS
        },
        "provenance/author_declaration.v1.json": policy["input_sha256"][
            "author_declaration"
        ],
        "provenance/tool_provenance.v1.json": policy["input_sha256"][
            "tool_provenance"
        ],
        "provenance/venue_policy.v1.json": policy["input_sha256"]["venue_policy"],
        "provenance/author_public_key": policy["input_sha256"]["author_public_key"],
        "provenance/author_kit.source": policy["input_sha256"]["author_kit"],
        **policy["generated_tex_sha256"],
        **{
            f"provenance/arc-stage{stage}-report.v1.json": digest
            for stage, digest in (
                (int(raw_stage), raw_digest)
                for raw_stage, raw_digest in policy[
                    "arc_formal_receipts_sha256"
                ].items()
            )
        },
    }


def _page_count(pdf: Path, log_text: str) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf)).pages)
    except ImportError:
        matches = re.findall(r"Output written on .+?\((\d+) pages?", log_text)
        if not matches:
            raise RuntimeError("cannot determine PDF page count; install locked pypdf")
        return int(matches[-1])


def _resolve_tex_toolchain() -> dict[str, dict[str, str]]:
    toolchain: dict[str, dict[str, str]] = {}
    for name in ("pdflatex", "bibtex"):
        discovered = shutil.which(name)
        if discovered is None:
            raise RuntimeError("locked pdflatex and bibtex executables are required")
        path = Path(discovered).resolve(strict=True)
        version_run = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        version = next(
            (line.strip() for line in version_run.stdout.splitlines() if line.strip()), ""
        )
        if version_run.returncode or not version:
            raise RuntimeError(f"cannot identify locked {name} toolchain executable")
        toolchain[name] = {
            "path": str(path),
            "sha256": lineage.artifact_sha256(path),
            "version": version[:256],
        }
    return toolchain


def _run_tex(build: Path) -> tuple[list[list[str]], str, dict[str, dict[str, str]]]:
    """Run a fail-closed TeX toolchain without depending on latexmk/Perl."""

    toolchain = _resolve_tex_toolchain()
    pdflatex = toolchain["pdflatex"]["path"]
    bibtex = toolchain["bibtex"]["path"]
    latex = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main_v2.tex",
    ]
    commands = [latex, [bibtex, "main_v2"], latex, latex]
    transcript: list[str] = []
    for command in commands:
        run = subprocess.run(
            command,
            cwd=build,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        transcript.extend((f"$ {' '.join(command)}", run.stdout, run.stderr))
        if run.returncode:
            raise RuntimeError(
                f"LaTeX compilation failed with exit code {run.returncode}: "
                f"{' '.join(command)}"
            )
    if _resolve_tex_toolchain() != toolchain:
        raise RuntimeError("TeX toolchain changed during compilation")
    return commands, "\n".join(transcript), toolchain


def validate_page_five_source(main_source: Path, compliance_source: Path) -> None:
    """Ensure no technical source can enter the page-five-only suffix."""

    main = re.sub(
        r"(?m)(?<!\\)%.*$", "", main_source.read_text(encoding="utf-8")
    )
    boundary = main.find(r"\clearpage")
    if boundary < 0:
        raise ValueError("main manuscript has no explicit page-five boundary")
    suffix = main[boundary:]
    expected = re.compile(
        r"^\s*\\clearpage\s*"
        r"\\bibliographystyle\{IEEEbib\}\s*"
        r"\\bibliography\{refs\}\s*"
        r"\\input\{compliance\}\s*"
        r"\\end\{document\}\s*$"
    )
    if not expected.fullmatch(suffix):
        raise ValueError("page-five source contains content outside the approved allowlist")
    compliance = re.sub(
        r"(?m)(?<!\\)%.*$", "", compliance_source.read_text(encoding="utf-8")
    )
    sections = re.findall(r"\\section\*?\{([^}]*)\}", compliance)
    if sections != ["Funding Acknowledgment", "Compliance with Ethical Standards"]:
        raise ValueError("compliance.tex must contain only the approved page-five sections")
    if r"\input{" in compliance or r"\include{" in compliance:
        raise ValueError("compliance.tex cannot import unbound page-five content")


def _technical_end_page(aux_text: str) -> int:
    match = re.search(
        r"\\newlabel\{technical-content-end\}\{\{.*?\}\{(\d+)\}", aux_text
    )
    if match is None:
        raise RuntimeError("technical-content-end page label is missing")
    return int(match.group(1))


def _validate_pdf_page_content(pdf: Path, aux_text: str) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("locked pypdf is required for page-content validation") from error
    reader = PdfReader(str(pdf))
    page_text = [" ".join((page.extract_text() or "").split()) for page in reader.pages]
    if len(page_text) != 5:
        raise RuntimeError("page-content validation requires exactly five pages")
    if _technical_end_page(aux_text) != 4:
        raise RuntimeError("technical content must end on page 4")
    required_page_five = (
        "References",
        "Funding Acknowledgment",
        "Compliance with Ethical Standards",
    )
    for heading in required_page_five:
        if heading.casefold() not in page_text[4].casefold():
            raise RuntimeError(f"page 5 lacks required content: {heading}")
        if any(heading.casefold() in text.casefold() for text in page_text[:4]):
            raise RuntimeError(f"page-five-only content leaked before page 5: {heading}")
    for heading in _TECHNICAL_HEADINGS:
        if heading.casefold() in page_text[4].casefold():
            raise RuntimeError(f"technical content leaked onto page 5: {heading}")


def _copy_policy_inputs(build: Path, policy: Mapping[str, Any]) -> tuple[int, ...]:
    auto = build / "auto"
    auto.mkdir(exist_ok=True)
    for relative, text in policy["generated_tex"].items():
        target = build.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    provenance = build / "provenance"
    provenance.mkdir(exist_ok=True)
    canonical_sources = {
        "author_declaration": provenance / "author_declaration.v1.json",
        "tool_provenance": provenance / "tool_provenance.v1.json",
        "venue_policy": provenance / "venue_policy.v1.json",
        "author_public_key": provenance / "author_public_key",
        "author_kit": provenance / "author_kit.source",
    }
    for name, target in canonical_sources.items():
        shutil.copy2(policy["source_paths"][name], target)
    kit_path = policy["source_paths"]["author_kit"]
    kit_members = _validate_author_kit_zip(kit_path, policy["author_kit_members"])
    if kit_members != policy["author_kit_members"]:
        raise RuntimeError("author-kit member binding changed before compilation")
    with zipfile.ZipFile(kit_path) as archive:
        for descriptor in kit_members.values():
            target = build.joinpath(*PurePosixPath(descriptor["build_target"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(descriptor["member_path"], "r") as source, target.open(
                "wb"
            ) as destination:
                shutil.copyfileobj(source, destination, length=1 << 20)
            if lineage.artifact_sha256(target) != descriptor["sha256"]:
                raise RuntimeError("copied author-kit asset hash mismatch")
    stages = tuple(sorted(policy["arc_receipt_paths"]))
    for stage in stages:
        shutil.copy2(
            policy["arc_receipt_paths"][stage],
            provenance / f"arc-stage{stage}-report.v1.json",
        )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--author-declaration", type=Path, required=True)
    parser.add_argument("--tool-provenance", type=Path, required=True)
    parser.add_argument("--venue-policy", type=Path, required=True)
    parser.add_argument("--author-public-key", type=Path, required=True)
    parser.add_argument("--author-kit", type=Path, required=True)
    parser.add_argument("--arc-receipts-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    evidence = validate_compilation_inputs(arguments.claims, arguments.figures)
    policy = validate_submission_policy_inputs(
        arguments.author_declaration,
        arguments.tool_provenance,
        arguments.venue_policy,
        arguments.author_public_key,
        arguments.author_kit,
        arc_receipts_root=arguments.arc_receipts_root,
    )
    claims = evidence["claims"]
    require_reviewed_claim_macros(
        claims, arguments.claims / "robust_map_placeholders.tex"
    )
    repo = Path(__file__).resolve().parents[1]
    paper_source = repo / "paper"
    validate_page_five_source(
        paper_source / "main_v2.tex", paper_source / "compliance.tex"
    )
    source_hashes = _strict_source_hashes(paper_source, policy)
    manuscript_source_hashes = _manuscript_source_hashes(paper_source)
    scientific_input_hashes = reviewed_scientific_input_sha256(paper_source, evidence)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = (output / "build").resolve()
    if build.parent != output:
        raise RuntimeError("unsafe build directory resolution")
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(
        paper_source,
        build,
        ignore=shutil.ignore_patterns("*.pdf", "*.aux", "*.log", "*.out", "*.bbl", "*.blg"),
    )
    (build / "auto").mkdir(exist_ok=True)
    shutil.copy2(
        arguments.claims / "robust_map_placeholders.tex",
        build / "auto" / "robust_map_placeholders.tex",
    )
    figure_dir = build / "figures_v3"
    figure_dir.mkdir(exist_ok=True)
    shutil.copy2(arguments.figures / "summary.v3.json", figure_dir / "summary.v3.json")
    for name in FIGURE_ARTIFACTS:
        shutil.copy2(arguments.figures / name, figure_dir / name)
    arc_stages = _copy_policy_inputs(build, policy)
    expected_copied = _expected_compiled_input_hashes(evidence, policy, source_hashes)
    copied = _compiled_input_hashes(build, arc_stages)
    if copied != expected_copied:
        raise RuntimeError("copied paper inputs differ from their authenticated source bundle")

    commands, transcript, toolchain = _run_tex(build)
    (output / "tex-build.log").write_text(transcript, encoding="utf-8")
    tex_log = (build / "main_v2.log").read_text(encoding="utf-8", errors="replace")
    if "Overfull \\hbox" in tex_log or "Overfull \\vbox" in tex_log:
        raise RuntimeError("submission contains an overfull box")
    pdf = build / "main_v2.pdf"
    if not pdf.is_file():
        raise RuntimeError("LaTeX reported success without a PDF")
    pages = _page_count(pdf, tex_log)
    if pages != 5:
        raise RuntimeError(
            f"ICASSP paper must be exactly five pages including references; got {pages}"
        )
    aux_text = (build / "main_v2.aux").read_text(encoding="utf-8", errors="replace")
    _validate_pdf_page_content(pdf, aux_text)
    final_pdf = output / "main_v2.pdf"
    shutil.copy2(pdf, final_pdf)

    final_evidence = validate_compilation_inputs(arguments.claims, arguments.figures)
    final_policy = validate_submission_policy_inputs(
        arguments.author_declaration,
        arguments.tool_provenance,
        arguments.venue_policy,
        arguments.author_public_key,
        arguments.author_kit,
        arc_receipts_root=arguments.arc_receipts_root,
    )
    evidence_keys = (
        "claims_sha256",
        "claim_macros_sha256",
        "verified_registry_sha256",
        "figures_summary_sha256",
        "figure_artifacts_sha256",
    )
    if {key: final_evidence[key] for key in evidence_keys} != {
        key: evidence[key] for key in evidence_keys
    }:
        raise RuntimeError("paper evidence changed during compilation")
    if final_policy["binding_sha256"] != policy["binding_sha256"]:
        raise RuntimeError("signed paper policy inputs changed during compilation")
    if _manuscript_source_hashes(paper_source) != manuscript_source_hashes:
        raise RuntimeError("paper source changed during compilation")
    if reviewed_scientific_input_sha256(paper_source, final_evidence) != (
        scientific_input_hashes
    ):
        raise RuntimeError("Stage 20 reviewed scientific inputs changed during compilation")
    copied = _compiled_input_hashes(build, arc_stages)
    if copied != expected_copied:
        raise RuntimeError("compiled paper inputs changed during compilation")
    report = {
        "schema_version": "submission-build-v3",
        "status": "complete",
        "submission_ready": bool(claims.get("submission_ready")),
        "stage15_status": claims.get("status"),
        "pages": pages,
        "technical_content_end_page": 4,
        "page_five_content_validated": True,
        "overfull_boxes": 0,
        "review_model": policy["review_model"],
        "author_identities_visible_to_reviewers": policy[
            "author_identities_visible_to_reviewers"
        ],
        "claims_sha256": evidence["claims_sha256"],
        "claim_macros_sha256": evidence["claim_macros_sha256"],
        "verified_registry_sha256": evidence["verified_registry_sha256"],
        "figures_sha256": evidence["figures_summary_sha256"],
        "figures_summary_sha256": evidence["figures_summary_sha256"],
        "figure_artifacts_sha256": evidence["figure_artifacts_sha256"],
        "paper_policy_binding_sha256": policy["binding_sha256"],
        "paper_policy_input_sha256": policy["input_sha256"],
        "paper_policy_content_sha256": {
            "author_declaration": policy["author_content_sha256"],
            "tool_provenance": policy["tool_content_sha256"],
            "venue_policy": policy["venue_content_sha256"],
        },
        "author_kit_members": policy["author_kit_members"],
        "reviewed_scientific_input_sha256": scientific_input_hashes,
        "reviewed_scientific_input_bundle_sha256": lineage.canonical_sha256(
            scientific_input_hashes
        ),
        "ai_policy_attestation": policy["ai_policy_attestation"],
        "author_public_key_fingerprint_sha256": policy[
            "author_public_key_fingerprint_sha256"
        ],
        "arc_formal_receipts_sha256": policy["arc_formal_receipts_sha256"],
        "compiled_input_sha256": copied,
        "pdf_sha256": lineage.artifact_sha256(final_pdf),
        "commands": commands,
        "toolchain": toolchain,
    }
    report_path = output / "build_report.v3.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "build_report.v3.sha256").write_text(
        lineage.artifact_sha256(report_path) + "  build_report.v3.json\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
