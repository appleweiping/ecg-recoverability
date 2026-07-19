"""Build a conspicuous, non-release five-page PDF for the Stage 20 review gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS
try:
    from scripts import compile_submission_v3 as submission
except (ImportError, ModuleNotFoundError):  # Direct ``python scripts/...`` execution.
    import compile_submission_v3 as submission


PAPER_SOURCE_FILES = submission.PAPER_SOURCE_FILES
_page_count = submission._page_count
_paper_source_hashes = submission._paper_source_hashes
_run_tex = submission._run_tex
_validate_pdf_page_content = submission._validate_pdf_page_content
reviewed_scientific_input_sha256 = submission.reviewed_scientific_input_sha256
require_reviewed_claim_macros = submission.require_reviewed_claim_macros
validate_compilation_inputs = submission.validate_compilation_inputs
validate_page_five_source = submission.validate_page_five_source


DRAFT_SCHEMA = "stage20-review-draft-v3"
DRAFT_MACROS = {
    "auto/venue_policy.tex": "\n".join(
        (
            "% Explicit Stage 20 review draft; never accepted by the release compiler.",
            r"\newif\ifSubmissionAuthorIdentitiesVisible",
            r"\SubmissionAuthorIdentitiesVisiblefalse",
            r"\newcommand{\SubmissionReviewModel}{stage20-review-draft-unresolved}",
            (
                r"\newcommand{\SubmissionBuildTitleSuffix}"
                r"{\\[2pt]\large REVIEW DRAFT---NOT FOR SUBMISSION}"
            ),
            "",
        )
    ),
    "auto/author_declaration.tex": "\n".join(
        (
            "% No author facts are asserted by this Stage 20 review draft.",
            r"\newcommand{\SubmissionAuthorNames}{}",
            r"\newcommand{\SubmissionAuthorAffiliations}{}",
            (
                r"\newcommand{\SubmissionFundingStatement}{\textbf{REVIEW DRAFT---NOT FOR "
                r"SUBMISSION.} The final author-verified funding declaration is intentionally "
                r"absent from this review artifact.}"
            ),
            (
                r"\newcommand{\SubmissionConflictStatement}{This review artifact makes no "
                r"competing-interest claim; the final statement requires a signed author "
                r"declaration.}"
            ),
            (
                r"\newcommand{\SubmissionComplianceStatement}{This review artifact makes no "
                r"author-verified ethical-compliance claim; the final statement requires a "
                r"signed author declaration.}"
            ),
            "",
        )
    ),
    "auto/tool_provenance.tex": "\n".join(
        (
            "% No final tool provenance is asserted by this Stage 20 review draft.",
            (
                r"\newcommand{\SubmissionToolDisclosure}{\textbf{REVIEW DRAFT---NOT FOR "
                r"SUBMISSION.} No final tool-use disclosure is claimed here. The submission "
                r"disclosure requires signed, human-verified provenance and all applicable "
                r"formal control receipts.}"
            ),
            (
                r"\newcommand{\SubmissionHumanVerificationStatement}{This review artifact is "
                r"not a final author attestation and is ineligible for release.}"
            ),
            "",
        )
    ),
}


def _draft_macro_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in DRAFT_MACROS.items()
    }


def _compiled_input_hashes(build: Path) -> dict[str, str]:
    paths: dict[str, Path] = {
        **{f"source/{name}": build / name for name in PAPER_SOURCE_FILES},
        "auto/robust_map_placeholders.tex": build
        / "auto"
        / "robust_map_placeholders.tex",
        **{name: build.joinpath(*name.split("/")) for name in DRAFT_MACROS},
        "figures_v3/summary.v3.json": build / "figures_v3" / "summary.v3.json",
        **{
            f"figures_v3/{name}": build / "figures_v3" / name
            for name in FIGURE_ARTIFACTS
        },
    }
    return {name: lineage.artifact_sha256(path) for name, path in paths.items()}


def _expected_input_hashes(
    evidence: Mapping[str, Any], source_hashes: Mapping[str, str]
) -> dict[str, str]:
    return {
        **source_hashes,
        "auto/robust_map_placeholders.tex": evidence["claim_macros_sha256"],
        **_draft_macro_hashes(),
        "figures_v3/summary.v3.json": evidence["figures_summary_sha256"],
        **{
            f"figures_v3/{name}": evidence["figure_artifacts_sha256"][name]
            for name in FIGURE_ARTIFACTS
        },
    }


def _write_draft_macros(build: Path) -> None:
    for relative, text in DRAFT_MACROS.items():
        target = build.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")


def _validate_draft_markers(pdf: Path) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("locked pypdf is required for draft-marker validation") from error
    pages = [" ".join((page.extract_text() or "").split()) for page in PdfReader(str(pdf)).pages]
    if "REVIEW DRAFT" not in pages[0] or "NOT FOR SUBMISSION" not in pages[0]:
        raise RuntimeError("Stage 20 PDF lacks the page-1 review-draft marker")
    if "REVIEW DRAFT" not in pages[4] or "NOT FOR SUBMISSION" not in pages[4]:
        raise RuntimeError("Stage 20 PDF lacks the page-5 review-draft disclosure")


def build_stage20_review_draft(
    *, claims_dir: Path, figures_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Build a review-only PDF whose schema cannot satisfy final release."""

    evidence = validate_compilation_inputs(claims_dir, figures_dir)
    claims = evidence["claims"]
    require_reviewed_claim_macros(
        claims, claims_dir / "robust_map_placeholders.tex"
    )
    repo = Path(__file__).resolve().parents[1]
    paper_source = repo / "paper"
    validate_page_five_source(
        paper_source / "main_v2.tex", paper_source / "compliance.tex"
    )
    source_hashes = _paper_source_hashes(paper_source)
    scientific_input_hashes = reviewed_scientific_input_sha256(paper_source, evidence)
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = (output / "build").resolve()
    if build.parent != output:
        raise RuntimeError("unsafe Stage 20 build directory resolution")
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(
        paper_source,
        build,
        ignore=shutil.ignore_patterns("*.pdf", "*.aux", "*.log", "*.out", "*.bbl", "*.blg"),
    )
    (build / "auto").mkdir(exist_ok=True)
    shutil.copy2(
        claims_dir / "robust_map_placeholders.tex",
        build / "auto" / "robust_map_placeholders.tex",
    )
    _write_draft_macros(build)
    figure_output = build / "figures_v3"
    figure_output.mkdir(exist_ok=True)
    shutil.copy2(figures_dir / "summary.v3.json", figure_output / "summary.v3.json")
    for name in FIGURE_ARTIFACTS:
        shutil.copy2(figures_dir / name, figure_output / name)
    expected = _expected_input_hashes(evidence, source_hashes)
    copied = _compiled_input_hashes(build)
    if copied != expected:
        raise RuntimeError("Stage 20 draft inputs differ from their authenticated sources")

    commands, transcript, toolchain = _run_tex(build)
    (output / "tex-build.log").write_text(transcript, encoding="utf-8", newline="\n")
    tex_log = (build / "main_v2.log").read_text(encoding="utf-8", errors="replace")
    if "Overfull \\hbox" in tex_log or "Overfull \\vbox" in tex_log:
        raise RuntimeError("Stage 20 review draft contains an overfull box")
    pdf = build / "main_v2.pdf"
    if not pdf.is_file():
        raise RuntimeError("LaTeX reported success without a Stage 20 PDF")
    pages = _page_count(pdf, tex_log)
    if pages != 5:
        raise RuntimeError(f"Stage 20 review draft must be exactly five pages; got {pages}")
    aux_text = (build / "main_v2.aux").read_text(encoding="utf-8", errors="replace")
    _validate_pdf_page_content(pdf, aux_text)
    _validate_draft_markers(pdf)
    final_pdf = output / "stage20_review_draft.pdf"
    shutil.copy2(pdf, final_pdf)

    final_evidence = validate_compilation_inputs(claims_dir, figures_dir)
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
        raise RuntimeError("paper evidence changed during Stage 20 draft compilation")
    if _paper_source_hashes(paper_source) != source_hashes:
        raise RuntimeError("paper source changed during Stage 20 draft compilation")
    if reviewed_scientific_input_sha256(paper_source, final_evidence) != (
        scientific_input_hashes
    ):
        raise RuntimeError("Stage 20 reviewed scientific inputs changed during draft compilation")
    copied = _compiled_input_hashes(build)
    if copied != expected:
        raise RuntimeError("compiled Stage 20 draft inputs changed during compilation")
    report = {
        "schema_version": DRAFT_SCHEMA,
        "status": "complete",
        "review_ready": True,
        "submission_ready": False,
        "release_eligible": False,
        "not_for_submission": True,
        "stage15_status": claims.get("status"),
        "pages": pages,
        "technical_content_end_page": 4,
        "page_five_content_validated": True,
        "overfull_boxes": 0,
        "identity_claimed": False,
        "funding_claimed": False,
        "final_ai_provenance_claimed": False,
        "official_author_kit_claimed": False,
        "claims_sha256": evidence["claims_sha256"],
        "claim_macros_sha256": evidence["claim_macros_sha256"],
        "verified_registry_sha256": evidence["verified_registry_sha256"],
        "figures_summary_sha256": evidence["figures_summary_sha256"],
        "figure_artifacts_sha256": evidence["figure_artifacts_sha256"],
        "paper_source_sha256": source_hashes,
        "reviewed_scientific_input_sha256": scientific_input_hashes,
        "reviewed_scientific_input_bundle_sha256": lineage.canonical_sha256(
            scientific_input_hashes
        ),
        "draft_macro_sha256": _draft_macro_hashes(),
        "compiled_input_sha256": copied,
        "pdf_sha256": lineage.artifact_sha256(final_pdf),
        "commands": commands,
        "toolchain": toolchain,
    }
    report_path = output / "review_draft_report.v3.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "review_draft_report.v3.sha256").write_text(
        lineage.artifact_sha256(report_path) + "  review_draft_report.v3.json\n",
        encoding="ascii",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    build_stage20_review_draft(
        claims_dir=arguments.claims,
        figures_dir=arguments.figures,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()
