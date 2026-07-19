"""Generate paper claims and a dynamic VerifiedRegistry from reviewed Stage 15."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ecgcert import lineage
from ecgcert.paper_evidence import validate_figure_bundle
from ecgcert.stage_gates import DEFAULT_REVIEWER_PUBLIC_KEY, validate_reviewed_gate


MACROS = (
    "ResultHeadline",
    "ResultPrimaryAssociation",
    "ResultIncrementalValue",
    "ResultRankWeightStability",
    "ResultExternalAssociation",
    "ResultModelCoverage",
    "ResultBootstrapUncertainty",
    "ResultConclusion",
)
REVIEWED_PAPER_DECISIONS = {"PROCEED", "PIVOT"}


def _escape(value: str) -> str:
    # Preserve deliberate LaTeX commands emitted by this module while escaping
    # only otherwise-unescaped text metacharacters.
    return re.sub(r"(?<!\\)([%_&])", r"\\\1", value)


def _effect(value: Mapping[str, Any]) -> str:
    point = float(value["point"])
    interval = value["ci95"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("effect ci95 must contain two endpoints")
    return (
        rf"$\Delta R^2={point:.3f}$ "
        rf"(95\% CI {float(interval[0]):.3f} to {float(interval[1]):.3f})"
    )


def _review_is_valid(
    gate: Mapping[str, Any],
    reviewer_public_key: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> bool:
    """Return whether Stage 15 has a signed PROCEED or PIVOT author review."""
    try:
        if gate.get("stage") != 15 or gate.get("status") not in REVIEWED_PAPER_DECISIONS:
            return False
        validate_reviewed_gate(gate, public_key_path=reviewer_public_key)
    except (TimeoutError, ValueError):
        return False
    return True


def _claim_values(gate: Mapping[str, Any]) -> dict[str, str]:
    status = str(gate["status"])
    evidence = gate.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("reviewed Stage-15 gate lacks evidence")
    external_evidence = evidence.get("external")
    method_effects = evidence.get("method_delta_r2")
    if not isinstance(external_evidence, Mapping) or set(external_evidence) != {
        "chapman", "cpsc2018"
    }:
        raise ValueError("Stage-15 evidence must contain Chapman and CPSC2018 effects")
    if not isinstance(method_effects, Mapping) or len(method_effects) != 4:
        raise ValueError("Stage-15 evidence must contain all four common-panel methods")

    ptb = _effect(evidence["ptbxl"])
    external = "; ".join(
        f"{cohort}: {_effect(effect)}"
        for cohort, effect in sorted(external_evidence.items())
    )
    positive = sum(float(value) > 0 for value in method_effects.values())
    bootstrap = int(evidence["bootstrap_replicates"])
    if bootstrap != 2_000:
        raise ValueError("primary claim synchronization requires 2,000 bootstrap replicates")

    if status == "PROCEED":
        headline = (
            "The prespecified robust-ambiguity predictor met the Stage~15 evidence rule; "
            f"on PTB-XL, {ptb}."
        )
        primary = f"Reviewed Stage~15 PROCEED; PTB-XL {ptb}"
        conclusion = (
            "The frozen PTB-XL, external-transfer, and cross-method criteria were met, "
            "supporting incremental predictive value without a clinical guarantee."
        )
    else:
        reasons = gate.get("automatic_reasons")
        failed = len(reasons) if isinstance(reasons, list) else 0
        headline = (
            "The prespecified robust-ambiguity predictor did not meet the combined "
            f"Stage~15 rule; on PTB-XL, {ptb}."
        )
        primary = f"Reviewed Stage~15 PIVOT (transparent negative result); PTB-XL {ptb}"
        conclusion = (
            "The frozen evidence rule was not met"
            + (f" in {failed} criterion" + ("s" if failed != 1 else "") if failed else "")
            + "; no rank, score, or hyperparameter was retuned after test opening."
        )

    return {
        "ResultHeadline": headline,
        "ResultPrimaryAssociation": primary,
        "ResultIncrementalValue": ptb,
        "ResultRankWeightStability": "prespecified ranks 2--5; full envelope reported",
        "ResultExternalAssociation": external,
        "ResultModelCoverage": f"{positive}/4 common-panel reconstructors positive",
        "ResultBootstrapUncertainty": (
            f"{bootstrap} patient-cluster replicates with nested seeds"
        ),
        "ResultConclusion": conclusion,
    }


def _dynamic_registry(
    source: Mapping[str, Any],
    *,
    source_sha256: str,
    stage15_sha256: str,
    stage15_status: str,
    values: Mapping[str, str],
    claim_macros_sha256: str,
    figures_summary_sha256: str,
    figure_artifacts_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if source.get("schema_version") != "verified-registry-v1":
        raise ValueError("claim sync requires a verified-registry-v1 source")
    citations = source.get("citations")
    numeric = source.get("numeric_claims")
    if not isinstance(citations, Mapping) or not isinstance(numeric, Mapping):
        raise ValueError("VerifiedRegistry citations/numeric_claims must be objects")
    missing = set(MACROS) - set(numeric)
    if missing:
        raise ValueError(f"source VerifiedRegistry lacks result macros: {sorted(missing)}")

    registry = deepcopy(dict(source))
    registry["source_registry_sha256"] = source_sha256
    registry["stage15_sha256"] = stage15_sha256
    registry["stage15_status"] = stage15_status
    registry["claim_values_sha256"] = lineage.canonical_sha256(dict(values))
    registry["claim_macros_sha256"] = claim_macros_sha256
    registry["figures_summary_sha256"] = figures_summary_sha256
    registry["figure_artifacts_sha256"] = dict(figure_artifacts_sha256)
    updated = registry["numeric_claims"]
    for macro in MACROS:
        entry = dict(updated[macro])
        if not entry.get("artifact"):
            raise ValueError(f"numeric claim {macro!r} lacks an artifact path")
        entry.update(
            {
                "status": "verified_artifact",
                "stage15_sha256": stage15_sha256,
                "stage15_status": stage15_status,
                "value_sha256": lineage.canonical_sha256(values[macro]),
                "claim_macros_sha256": claim_macros_sha256,
            }
        )
        updated[macro] = entry
    return registry


def synchronize_claims(
    *,
    stage15: Path,
    figures: Path,
    output: Path,
    source_registry: Path,
    reviewer_public_key: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> dict[str, Any]:
    gate_path = stage15 / "decision.v3.json"
    figures_path = figures / "summary.v3.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    figures_value = json.loads(figures_path.read_text(encoding="utf-8"))
    source_value = json.loads(source_registry.read_text(encoding="utf-8"))
    if gate.get("schema_version") != "arc-stage15-v3":
        raise ValueError("claim sync requires an ARC Stage-15 v3 artifact")
    if not _review_is_valid(gate, reviewer_public_key):
        raise ValueError("Stage-15 gate lacks a valid signed PROCEED/PIVOT human review")

    stage15_sha256 = lineage.artifact_sha256(gate_path)
    if figures_value.get("schema_version") != "paper-figures-v3":
        raise ValueError("claim sync requires a paper-figures-v3 artifact")
    figure_binding = validate_figure_bundle(figures, summary=figures_value)
    figure_gate_sha = figures_value.get("input_sha256", {}).get("stage15")
    if figure_gate_sha != stage15_sha256:
        raise ValueError("paper figures are not bound to the reviewed Stage-15 gate")

    values = _claim_values(gate)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated only by scripts/claim_sync_v3.py."]
    for macro in MACROS:
        lines.append(rf"\newcommand{{\{macro}}}{{{_escape(values[macro])}}}")
    macros_path = output / "robust_map_placeholders.tex"
    macros_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    claim_macros_sha256 = lineage.artifact_sha256(macros_path)

    dynamic_registry = _dynamic_registry(
        source_value,
        source_sha256=lineage.artifact_sha256(source_registry),
        stage15_sha256=stage15_sha256,
        stage15_status=str(gate["status"]),
        values=values,
        claim_macros_sha256=claim_macros_sha256,
        figures_summary_sha256=str(figure_binding["summary_sha256"]),
        figure_artifacts_sha256=figure_binding["artifacts_sha256"],
    )
    registry_path = output / "verified_registry.v1.json"
    registry_path.write_text(
        json.dumps(dynamic_registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "paper-claims-v3",
        "status": gate["status"],
        "submission_ready": True,
        "stage15_sha256": stage15_sha256,
        # Keep the legacy summary alias while exposing the direct bindings
        # Stage 20 checks item by item.
        "figures_sha256": figure_binding["summary_sha256"],
        "figures_summary_sha256": figure_binding["summary_sha256"],
        "figure_artifacts_sha256": figure_binding["artifacts_sha256"],
        "claim_macros_sha256": claim_macros_sha256,
        "source_registry_sha256": lineage.artifact_sha256(source_registry),
        "verified_registry_sha256": lineage.artifact_sha256(registry_path),
        "claim_values_sha256": lineage.canonical_sha256(values),
        "values": values,
    }
    (output / "claims.v3.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage15", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "arc_audit" / "verified_registry.v1.json",
    )
    parser.add_argument(
        "--reviewer-public-key",
        type=Path,
        default=DEFAULT_REVIEWER_PUBLIC_KEY,
    )
    arguments = parser.parse_args()
    synchronize_claims(
        stage15=arguments.stage15,
        figures=arguments.figures,
        output=arguments.output_dir,
        source_registry=arguments.registry,
        reviewer_public_key=arguments.reviewer_public_key,
    )


if __name__ == "__main__":
    main()
