"""Compile the five-page ICASSP paper from synchronized, evidence-gated claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from ecgcert import lineage
from ecgcert.paper_evidence import (
    FIGURE_ARTIFACTS,
    require_artifact_hashes,
    validate_figure_bundle,
)


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def validate_compilation_inputs(claims_dir: Path, figures_dir: Path) -> dict:
    """Re-hash every paper input before it may enter the TeX build."""
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
        "verified_registry": (
            claims.get("verified_registry_sha256") == actual_registry_sha256
        ),
        "registry_claim_macros": (
            registry.get("claim_macros_sha256") == actual_macros_sha256
        ),
        "registry_figures_summary": (
            registry.get("figures_summary_sha256") == figure_binding["summary_sha256"]
        ),
        "registry_figure_artifacts": (
            require_artifact_hashes(
                registry.get("figure_artifacts_sha256"),
                label="dynamic registry figure binding",
            )
            == figure_binding["artifacts_sha256"]
        ),
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


def _compiled_input_hashes(build: Path) -> dict[str, str]:
    paths = {
        "auto/robust_map_placeholders.tex": build / "auto" / "robust_map_placeholders.tex",
        "figures_v3/summary.v3.json": build / "figures_v3" / "summary.v3.json",
        **{
            f"figures_v3/{name}": build / "figures_v3" / name
            for name in FIGURE_ARTIFACTS
        },
    }
    return {name: lineage.artifact_sha256(path) for name, path in paths.items()}


def _expected_compiled_input_hashes(binding: dict) -> dict[str, str]:
    return {
        "auto/robust_map_placeholders.tex": binding["claim_macros_sha256"],
        "figures_v3/summary.v3.json": binding["figures_summary_sha256"],
        **{
            f"figures_v3/{name}": binding["figure_artifacts_sha256"][name]
            for name in FIGURE_ARTIFACTS
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


def _run_tex(build: Path) -> tuple[list[list[str]], str]:
    """Run a fail-closed TeX toolchain without depending on latexmk/Perl."""

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None or bibtex is None:
        raise RuntimeError("locked pdflatex and bibtex executables are required")
    commands = [
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main_v2.tex"],
        [bibtex, "main_v2"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main_v2.tex"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main_v2.tex"],
    ]
    transcript: list[str] = []
    for command in commands:
        run = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=300)
        transcript.extend((f"$ {' '.join(command)}", run.stdout, run.stderr))
        if run.returncode:
            raise RuntimeError(
                f"LaTeX compilation failed with exit code {run.returncode}: {' '.join(command)}"
            )
    return commands, "\n".join(transcript)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    macros_path = arguments.claims / "robust_map_placeholders.tex"
    figures_summary = arguments.figures / "summary.v3.json"
    binding = validate_compilation_inputs(arguments.claims, arguments.figures)
    claims = binding["claims"]

    repo = Path(__file__).resolve().parents[1]
    paper_source = repo / "paper"
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
    shutil.copy2(macros_path, build / "auto" / "robust_map_placeholders.tex")
    figure_dir = build / "figures_v3"
    figure_dir.mkdir(exist_ok=True)
    shutil.copy2(figures_summary, figure_dir / "summary.v3.json")
    for name in FIGURE_ARTIFACTS:
        source = arguments.figures / name
        shutil.copy2(source, figure_dir / name)
    copied = _compiled_input_hashes(build)
    expected_copied = _expected_compiled_input_hashes(binding)
    if copied != expected_copied:
        raise RuntimeError("copied paper evidence differs from the re-hashed source bundle")

    commands, transcript = _run_tex(build)
    (output / "tex-build.log").write_text(transcript, encoding="utf-8")
    tex_log_path = build / "main_v2.log"
    tex_log = tex_log_path.read_text(encoding="utf-8", errors="replace")
    if "Overfull \\hbox" in tex_log or "Overfull \\vbox" in tex_log:
        raise RuntimeError("submission contains an overfull box")
    pdf = build / "main_v2.pdf"
    if not pdf.is_file():
        raise RuntimeError("LaTeX reported success without a PDF")
    pages = _page_count(pdf, tex_log)
    if pages != 5:
        raise RuntimeError(f"ICASSP paper must be exactly five pages including references; got {pages}")
    final_pdf = output / "main_v2.pdf"
    shutil.copy2(pdf, final_pdf)
    # Revalidate both upstream inputs and copied build inputs immediately before
    # recording evidence, closing the window for unnoticed in-place mutation.
    final_binding = validate_compilation_inputs(arguments.claims, arguments.figures)
    if {
        key: final_binding[key]
        for key in (
            "claims_sha256",
            "claim_macros_sha256",
            "verified_registry_sha256",
            "figures_summary_sha256",
            "figure_artifacts_sha256",
        )
    } != {
        key: binding[key]
        for key in (
            "claims_sha256",
            "claim_macros_sha256",
            "verified_registry_sha256",
            "figures_summary_sha256",
            "figure_artifacts_sha256",
        )
    }:
        raise RuntimeError("paper evidence changed during compilation")
    copied = _compiled_input_hashes(build)
    if copied != expected_copied:
        raise RuntimeError("compiled paper evidence changed during compilation")
    report = {
        "schema_version": "submission-build-v3",
        "status": "complete",
        "submission_ready": bool(claims.get("submission_ready")),
        "stage15_status": claims.get("status"),
        "pages": pages,
        "overfull_boxes": 0,
        "anonymous": True,
        "claims_sha256": binding["claims_sha256"],
        "claim_macros_sha256": binding["claim_macros_sha256"],
        "verified_registry_sha256": binding["verified_registry_sha256"],
        "figures_sha256": binding["figures_summary_sha256"],
        "figures_summary_sha256": binding["figures_summary_sha256"],
        "figure_artifacts_sha256": binding["figure_artifacts_sha256"],
        "compiled_input_sha256": copied,
        "pdf_sha256": lineage.artifact_sha256(final_pdf),
        "commands": commands,
    }
    (output / "build_report.v3.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
