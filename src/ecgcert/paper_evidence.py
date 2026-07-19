"""Fail-closed bindings for the paper figures and their source tables."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from . import lineage


FIGURE_PDFS = (
    "figure1_robust_map.pdf",
    "figure2_prediction_gain.pdf",
)
FIGURE_SOURCE_TABLES = (
    "figure1_source.parquet",
    "figure2_source.parquet",
)
FIGURE_ARTIFACTS = FIGURE_PDFS + FIGURE_SOURCE_TABLES
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def direct_artifact_hashes(directory: Path) -> dict[str, str]:
    """Hash every required figure artifact directly from its bytes."""
    directory = directory.resolve()
    return {
        name: lineage.artifact_sha256(directory / name)
        for name in FIGURE_ARTIFACTS
    }


def require_artifact_hashes(value: Any, *, label: str) -> dict[str, str]:
    """Validate an exact, full-SHA mapping for the frozen figure bundle."""
    if not isinstance(value, Mapping) or set(value) != set(FIGURE_ARTIFACTS):
        raise ValueError(
            f"{label} must bind exactly these figure artifacts: "
            f"{sorted(FIGURE_ARTIFACTS)}"
        )
    result = {str(name): str(digest) for name, digest in value.items()}
    invalid = [name for name, digest in result.items() if not _HEX64.fullmatch(digest)]
    if invalid:
        raise ValueError(f"{label} contains invalid SHA-256 values: {sorted(invalid)}")
    return result


def validate_figure_bundle(
    directory: Path,
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-hash a figure directory and compare every file with its summary."""
    directory = directory.resolve()
    summary_path = directory / "summary.v3.json"
    if summary is None:
        import json

        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("paper figure summary must be a JSON object")
        summary = loaded
    if summary.get("schema_version") != "paper-figures-v3":
        raise ValueError("figure bundle requires a paper-figures-v3 summary")
    recorded = require_artifact_hashes(
        summary.get("artifacts_sha256"), label="paper figure summary"
    )
    actual = direct_artifact_hashes(directory)
    mismatches = [name for name in FIGURE_ARTIFACTS if recorded[name] != actual[name]]
    if mismatches:
        raise ValueError(
            "paper figure artifact hash mismatch: " + ", ".join(sorted(mismatches))
        )
    return {
        "summary_sha256": lineage.artifact_sha256(summary_path),
        "artifacts_sha256": actual,
    }
