"""Static guardrails for the evidence-gated primary ECG manuscript.

Run from any directory with:
    python paper/check_submission_claims.py

The check intentionally inspects manuscript source, not generated PDFs.  It
prevents legacy headline language and result artifacts from leaking back into
the four-page primary paper before the Stage 15 evidence decision.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
PRIMARY = PAPER_DIR / "main_v2.tex"
PLACEHOLDERS = PAPER_DIR / "auto" / "robust_map_placeholders.tex"
LONG = PAPER_DIR / "arxiv_long.tex"
FLOOR = PAPER_DIR / "theorem_floor.tex"
CITATION_STATUS = PAPER_DIR / "citation_status.yaml"
COMPLIANCE = PAPER_DIR / "compliance.tex"
REFERENCES = PAPER_DIR / "refs.bib"
AUTHOR_DECLARATION_TEMPLATE = PAPER_DIR / "author_declaration.v1.template.json"
TOOL_PROVENANCE_TEMPLATE = PAPER_DIR / "tool_provenance.v1.template.json"
VENUE_POLICY_TEMPLATE = PAPER_DIR / "venue_policy.v1.template.json"

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

LEGACY_INPUTS = (
    "theorem_corrected",
    "theorem_floor",
    "theorem_transfer",
    "results_v2_calib",
    "results_v2_maps",
    "results_v2_baseline",
    "results_long_crossdata",
    "results_long_generative",
    "auto/fair_baselines_macros",
    "auto/long_results_macros",
)

PRIMARY_BANNED = {
    r"\bexact(?:ly)?\b": "fixed-model exactness language",
    r"\bany\s+SNR\b": "SNR-independent impossibility language",
    r"\bcertificat(?:e|es|ed|ion|ions)\b": "certificate language",
    r"\bminimax\b": "legacy minimax branch",
    r"\bconformal(?:ized)?\b|\bCQR\b": "legacy calibration branch",
    r"\bST[- ]threshold\b": "legacy clinical-event branch",
    r"\bdiffusion\b": "legacy generative branch",
    r"\bactive\s+(?:lead\s+)?selection\b": "legacy active-selection branch",
    r"(?:\.\./)?results/": "direct dependency on result artifacts",
    r"\bmodel[- ]agnostic\b": "unsupported model-agnostic scope",
    r"\bnormalized[- ]?RMSE\b|\bNRMSE\b": "superseded normalized-RMSE wording",
    r"\bdimensionless\s+(?:recoverability\s+)?score\b": "legacy dimensionless score",
    r"\\mathcal\s*B_q\b": "legacy q-augmented meta-model notation",
    r"\blead[- ]weight(?:ing|ed)?\b": "legacy lead-weight family",
    r"\brank/weight/bootstrap\b": "legacy rank/weight aggregation",
    r"\bmedian\s+(?:score|of\s+.+?score)\b": "legacy median aggregation",
    r"fold~?8\s+selects?\s+(?:the\s+)?(?:primary\s+)?rank": "fold-8 rank selection",
    r"\ba\s+second\s+cohort\b|\bone\s+external\s+cohort\b": "single-external-cohort scope",
}

REQUIRED_PRIMARY_PATTERNS = {
    r"Gaussian\s+posterior\s+ambiguity|posterior\s+ambiguity":
        "Gaussian posterior ambiguity definition",
    r"\\bm\\Sigma\^\{\\mathrm\{post\}\}": "Gaussian posterior covariance",
    r"\\tau\^2": "single observation variance",
    r"10\^\{-8\}.*10\^\{-1\}": "fold-8 log-grid endpoints",
    r"Fold~8\s+never\s+selects\s+a\s+rank": "explicit no-rank-selection statement",
    r"A_\{\\mathrm\{robust\}\}": "A_robust primary score",
    r"Q\^\{\\mathrm\{pat\}\}_\{?\.975\}?": "patient-bootstrap 97.5th percentile",
    r"R_\{\\mathrm\{lower\}\}": "R_lower diagnostic",
    r"\\eta": "eta diagnostic",
    r"\\log_\{10\}\\kappa": "log10-kappa diagnostic",
    r"rank\s+span": "rank-span diagnostic",
    r"\b255\b": "full structural configuration map",
    r"\b64[- ]configuration\b|\b64\s+configurations\b": "fixed deep configuration panel",
    r"SHA-256": "configuration-panel hash",
    r"fold~9": "fold-9 meta-fit",
    r"fold~10": "single fold-10 evaluation",
    r"leave-one-configuration-out|\bLOCO\b": "LOCO meta-prediction",
    r"patient-level\s+log-RMSE|natural-log\s+RMSE": "patient log-RMSE outcome",
    r"nested\s+seed|nested\s+patient/seed": "nested neural-seed resampling",
    r"low-rank\s+conditional\s+mean": "low-rank common-panel method",
    r"ridge\s+regression": "ridge common-panel method",
    r"1-D\s+U-Net": "masked U-Net common-panel method",
    r"ImputeECG": "ImputeECG common-panel method",
    r"ECGrecover": "separate ECGrecover baseline",
    r"not\s+counted\s+among\s+the\s+four": "ECGrecover common-panel exclusion",
    r"Chapman": "Chapman zero-transfer cohort",
    r"CPSC~?2018": "CPSC 2018 zero-transfer cohort",
    r"60/20/20": "external patient split",
    r"at\s+least\s+three\s+of\s+the\s+four": "Stage-15 method threshold",
    r"figures_v3/figure1_robust_map\.pdf": "primary robust-map figure",
    r"figures_v3/figure2_prediction_gain\.pdf": "primary prediction-gain figure",
}


def strip_tex_comments(text: str) -> str:
    """Drop unescaped TeX comments while preserving escaped percent signs."""

    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def check() -> list[str]:
    failures: list[str] = []
    required_files = (
        PRIMARY,
        PLACEHOLDERS,
        LONG,
        FLOOR,
        CITATION_STATUS,
        COMPLIANCE,
        REFERENCES,
        AUTHOR_DECLARATION_TEMPLATE,
        TOOL_PROVENANCE_TEMPLATE,
        VENUE_POLICY_TEMPLATE,
    )
    for path in required_files:
        if not path.is_file():
            failures.append(f"missing required file: {path.relative_to(PAPER_DIR.parent)}")
    if failures:
        return failures

    primary_raw = PRIMARY.read_text(encoding="utf-8")
    primary = strip_tex_comments(primary_raw)
    placeholders = PLACEHOLDERS.read_text(encoding="utf-8")

    if r"\input{auto/robust_map_placeholders}" not in primary:
        failures.append("main_v2.tex must input auto/robust_map_placeholders.tex")
    required_authenticated_inputs = (
        r"\input{auto/venue_policy}",
        r"\input{auto/author_declaration}",
        r"\input{auto/tool_provenance}",
    )
    for token in required_authenticated_inputs:
        if token not in primary:
            failures.append(f"main_v2.tex lacks authenticated build input: {token}")
    if r"\ifSubmissionAuthorIdentitiesVisible" not in primary:
        failures.append("main_v2.tex must derive identity visibility from venue policy")
    if r"\SubmissionBuildTitleSuffix" not in primary:
        failures.append("main_v2.tex lacks the explicit build-mode title suffix")
    if re.search(r"\\anon(?:true|false)|submission\s+is\s+anonymous", primary, re.I):
        failures.append("main_v2.tex hardcodes anonymous submission mode")
    stale_provenance = re.search(
        r"Stage~?0?1\s+failed|queue\s+owner|failed\s+probe|"
        r"no\s+AutoResearchClaw\s+(?:research|review)\s+output",
        primary,
        flags=re.IGNORECASE,
    )
    if stale_provenance:
        failures.append("main_v2.tex contains stale hardcoded tool provenance")

    for pattern, description in PRIMARY_BANNED.items():
        match = re.search(pattern, primary, flags=re.IGNORECASE)
        if match:
            failures.append(
                f"main_v2.tex contains {description}: {match.group(0)!r}"
            )

    for pattern, description in REQUIRED_PRIMARY_PATTERNS.items():
        if not re.search(pattern, primary, flags=re.IGNORECASE | re.DOTALL):
            failures.append(f"main_v2.tex lacks {description}")

    figure_count = len(re.findall(r"\\begin\{figure\}", primary))
    table_count = len(re.findall(r"\\begin\{table\}", primary))
    if figure_count != 2:
        failures.append(f"main_v2.tex must contain two main figures; found {figure_count}")
    if table_count != 1:
        failures.append(f"main_v2.tex must contain one main table; found {table_count}")
    if r"\clearpage" not in primary or primary.find(r"\clearpage") > primary.find(
        r"\bibliographystyle"
    ):
        failures.append("main_v2.tex must force references onto page 5")
    page_five = primary[primary.find(r"\clearpage") :]
    allowed_page_five = re.compile(
        r"^\s*\\clearpage\s*"
        r"\\bibliographystyle\{IEEEbib\}\s*"
        r"\\bibliography\{refs\}\s*"
        r"\\input\{compliance\}\s*"
        r"\\end\{document\}\s*$"
    )
    if not allowed_page_five.fullmatch(page_five):
        failures.append("main_v2.tex page-5 suffix violates the venue content allowlist")
    if r"\label{technical-content-end}" not in primary:
        failures.append("main_v2.tex lacks the technical-page boundary label")

    for legacy_input in LEGACY_INPUTS:
        if re.search(
            rf"\\input\{{{re.escape(legacy_input)}\}}", primary, flags=re.IGNORECASE
        ):
            failures.append(f"main_v2.tex imports legacy branch: {legacy_input}")

    if not re.search(
        r"\\newcommand\{\\PendingStageFifteen\}"
        r"\{[^\n}]*Pending--Stage\s*15[^\n}]*\}",
        placeholders,
        flags=re.IGNORECASE,
    ):
        failures.append("placeholder sentinel must visibly render PENDING--STAGE 15")

    for macro in REQUIRED_RESULT_MACROS:
        declaration = re.search(
            rf"\\newcommand\{{\\{macro}\}}\{{\\PendingStageFifteen\}}",
            placeholders,
        )
        if not declaration:
            failures.append(f"missing gated placeholder declaration for \\{macro}")
        if rf"\{macro}" not in primary:
            failures.append(f"main_v2.tex does not use required placeholder \\{macro}")

    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        primary,
        flags=re.DOTALL,
    )
    if abstract is None or r"\ResultHeadline" not in abstract.group(1):
        failures.append("abstract must use evidence-synchronized \\ResultHeadline")
    conclusion = re.search(
        r"\\section\{Conclusion\}(.*?)(?:\\clearpage|\\bibliographystyle)",
        primary,
        flags=re.DOTALL,
    )
    if conclusion is None or r"\ResultConclusion" not in conclusion.group(1):
        failures.append("conclusion must use evidence-synchronized \\ResultConclusion")

    long_text = LONG.read_text(encoding="utf-8")
    appendix_at = long_text.find(r"\appendix")
    if appendix_at < 0:
        failures.append("arxiv_long.tex must contain an appendix boundary")
    if "Legacy supplement---not evidence for the primary" not in long_text:
        failures.append("arxiv_long.tex lacks the legacy-evidence quarantine banner")
    for legacy_input in LEGACY_INPUTS:
        token = rf"\input{{{legacy_input}}}"
        position = long_text.find(token)
        if position < 0:
            failures.append(f"arxiv_long.tex does not retain legacy branch: {legacy_input}")
        elif legacy_input.startswith("auto/"):
            # Historical macro definitions are necessarily loaded in the preamble.
            continue
        elif appendix_at >= 0 and position < appendix_at:
            failures.append(f"legacy branch appears before appendix boundary: {legacy_input}")

    floor_text = FLOOR.read_text(encoding="utf-8")
    forbidden_equivalence = re.search(
        r"below-floor\s+gap\s+is\s+exactly.*?(?:Tier-II|predictable\s+residual)",
        floor_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if forbidden_equivalence:
        failures.append("theorem_floor.tex restores the invalid floor/CQR equivalence")
    if not re.search(r"different\s+objects", floor_text, flags=re.IGNORECASE):
        failures.append("theorem_floor.tex must state that floor gap and residual differ")

    citation_status = CITATION_STATUS.read_text(encoding="utf-8")
    if "zhang2026systematic" not in citation_status:
        failures.append("citation ledger lacks the verified 2026 full-configuration benchmark")
    if not re.search(
        r"key:\s*[\"']zhang2026systematic[\"'][\s\S]{0,500}"
        r"status:\s*[\"']verified_primary[\"'][\s\S]{0,500}"
        r"10\.3389/fcvm\.2026\.1856211",
        citation_status,
        flags=re.IGNORECASE,
    ):
        failures.append("2026 full-configuration benchmark metadata is not primary-source verified")

    compliance = strip_tex_comments(COMPLIANCE.read_text(encoding="utf-8"))
    sections = re.findall(r"\\section\*?\{([^}]*)\}", compliance)
    if sections != ["Funding Acknowledgment", "Compliance with Ethical Standards"]:
        failures.append(
            "compliance.tex must contain only Funding Acknowledgment and the exactly titled "
            "Compliance with Ethical Standards section"
        )
    required_compliance_macros = (
        r"\SubmissionFundingStatement",
        r"\SubmissionComplianceStatement",
        r"\SubmissionConflictStatement",
        r"\SubmissionToolDisclosure",
        r"\SubmissionHumanVerificationStatement",
    )
    for macro in required_compliance_macros:
        if macro not in compliance:
            failures.append(f"compliance.tex lacks authenticated macro {macro}")
    if re.search(r"queue\s+owner|Stage~?0?1\s+failed|Pending", compliance, re.I):
        failures.append("compliance.tex contains stale or pending hardcoded provenance")

    template_status = {
        AUTHOR_DECLARATION_TEMPLATE: "PENDING_AUTHOR_SIGNATURE",
        TOOL_PROVENANCE_TEMPLATE: "PENDING_HUMAN_VERIFICATION",
        VENUE_POLICY_TEMPLATE: "PENDING_OFFICIAL_AUTHOR_KIT",
    }
    for path, expected_status in template_status.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != expected_status:
            failures.append(f"{path.name} must remain an explicit pending template")
        if value.get("content_sha256") or value.get("signature_ed25519"):
            failures.append(f"{path.name} must not masquerade as a signed declaration")
    tool_template = json.loads(TOOL_PROVENANCE_TEMPLATE.read_text(encoding="utf-8"))
    verification = tool_template.get("human_verification", {})
    required_ai_attestations = {
        "any_submitted_section_entirely_generated_by_ai",
        "most_or_significant_manuscript_components_generated_by_ai",
        "authors_substantively_rewrote_and_verified_ai_assisted_text",
    }
    if not required_ai_attestations.issubset(verification):
        failures.append("tool-provenance template lacks explicit IEEE SPS attestations")
    if "manuscript_section_generation_used" in verification:
        failures.append("tool-provenance template retains an ambiguous AI-generation field")
    venue_template = json.loads(VENUE_POLICY_TEMPLATE.read_text(encoding="utf-8"))
    if set(venue_template.get("author_kit_members", {})) != {
        "style",
        "bibliography_style",
        "latex_template",
    }:
        failures.append("venue-policy template does not bind all required author-kit assets")

    references = REFERENCES.read_text(encoding="utf-8")
    expected_impute_authors = (
        "Fang, Xiaocheng and Wang, Haoyu and Cai, Jieyi and Zhao, Qinghao and Li, Jun "
        "and Zhang, Shanwei and Nie, Guangkun and Xiao, Yujie and Huang, Shun and "
        "Jin, Jiarui and Liu, Hongmin and Wang, Guodong and Chen, Shuohua and "
        "Lin, Liming and Wu, Shouling and Li, Hongyan and Hong, Shenda"
    )
    impute_entry = re.search(
        r"@article\{fang2026imputeecg,(.*?)(?=\n\})",
        references,
        flags=re.DOTALL,
    )
    if impute_entry is None or expected_impute_authors not in impute_entry.group(1):
        failures.append("ImputeECG bibliography author order is incomplete or incorrect")

    return failures


def main() -> int:
    failures = check()
    if failures:
        print("Submission-claim checks FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Submission-claim checks PASSED.")
    print("Primary claims are placeholder-gated; legacy branches remain quarantined.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
