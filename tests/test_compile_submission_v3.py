from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from scripts.compile_submission_v3 import validate_compilation_inputs


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _bound_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stage15_sha256 = "a" * 64
    figures = tmp_path / "figures"
    figures.mkdir()
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
