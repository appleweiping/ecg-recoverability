"""Generate the two primary ICASSP figures and their exact source tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.physics import LEADS
from ecgcert.protocol import deep_configuration_panel


def _configuration_id(configuration) -> str:
    return "+".join(configuration)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def figure_summary(
    output: Path,
    *,
    stage15_status: str,
    input_paths: dict[str, Path],
) -> dict:
    """Build the summary only after directly hashing all rendered evidence."""
    artifact_sha256 = direct_artifact_hashes(output)
    if set(artifact_sha256) != set(FIGURE_ARTIFACTS):  # defensive schema lock
        raise RuntimeError("incomplete primary-figure evidence bundle")
    return {
        "schema_version": "paper-figures-v3",
        "status": "complete",
        "stage15_status": stage15_status,
        "figures": ["figure1_robust_map.pdf", "figure2_prediction_gain.pdf"],
        "source_tables": ["figure1_source.parquet", "figure2_source.parquet"],
        "artifacts_sha256": artifact_sha256,
        "input_sha256": {
            name: lineage.artifact_sha256(path) for name, path in input_paths.items()
        },
    }


def robust_map_figure(map_cells: pd.DataFrame, output: Path) -> pd.DataFrame:
    panel_ids = [_configuration_id(configuration) for configuration in deep_configuration_panel()]
    source = map_cells[
        map_cells["configuration"].isin(panel_ids) & map_cells["segment"].isin(("QRS", "ST", "T"))
    ].copy()
    source["configuration"] = pd.Categorical(
        source["configuration"], categories=panel_ids, ordered=True
    )
    source["target"] = pd.Categorical(source["target"], categories=LEADS, ordered=True)
    source = source.sort_values(["segment", "configuration", "target"])
    if len(source) != 3 * len(panel_ids) * len(LEADS):
        raise ValueError("primary map figure requires all 64 x 12 x 3 frozen cells")

    values = source["ambiguity_robust_mv"].to_numpy(dtype=float)
    vmax = float(np.nanquantile(values, 0.99))
    vmax = max(vmax, np.finfo(float).eps)
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 6.2), sharex=True, sharey=True)
    image = None
    for axis, segment in zip(axes, ("QRS", "ST", "T")):
        rows = source[source["segment"] == segment]
        matrix = rows.pivot(index="configuration", columns="target", values="ambiguity_robust_mv")
        matrix = matrix.reindex(index=panel_ids, columns=LEADS)
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(segment)
        axis.set_xticks(range(len(LEADS)), LEADS, rotation=90, fontsize=6)
        axis.set_xlabel("target lead")
    axes[0].set_ylabel("frozen observed-lead configuration (64)")
    tick_locations = [0, 7, 35, 42, 49, 56, 63]
    axes[0].set_yticks(tick_locations, [panel_ids[index] for index in tick_locations], fontsize=6)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"robust ambiguity $A_{robust}$ (mV)")
    figure.subplots_adjust(left=0.12, right=0.9, bottom=0.14, top=0.94, wspace=0.08)
    figure.savefig(output / "figure1_robust_map.pdf", bbox_inches="tight")
    figure.savefig(output / "figure1_robust_map.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
    return source


def gain_figure(effects: pd.DataFrame, output: Path) -> pd.DataFrame:
    required = {"cohort", "point", "ci95"}
    if not required <= set(effects):
        raise ValueError(f"effects table lacks {sorted(required - set(effects))}")
    source = effects.copy()
    source["ci_lower"] = source["ci95"].apply(lambda value: float(value[0]))
    source["ci_upper"] = source["ci95"].apply(lambda value: float(value[1]))
    order = [cohort for cohort in ("PTB-XL", "chapman", "cpsc2018") if cohort in set(source.cohort)]
    source["cohort"] = pd.Categorical(source["cohort"], categories=order, ordered=True)
    source = source.sort_values("cohort")
    if len(source) != 3:
        raise ValueError("prediction-gain figure requires PTB-XL, Chapman and CPSC2018")
    x = np.arange(len(source))
    point = source["point"].to_numpy(dtype=float)
    error = np.vstack((point - source["ci_lower"], source["ci_upper"] - point))
    figure, axis = plt.subplots(figsize=(4.8, 2.7))
    axis.axhline(0.0, color="0.35", linewidth=1, linestyle="--")
    axis.errorbar(x, point, yerr=error, fmt="o", color="#1f5a99", capsize=4, linewidth=1.5)
    axis.set_xticks(x, [str(value) for value in source["cohort"]])
    axis.set_ylabel(r"incremental predictive value $Delta R^2$")
    axis.set_xlabel("untouched test cohort")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "figure2_prediction_gain.pdf", bbox_inches="tight")
    figure.savefig(output / "figure2_prediction_gain.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-maps", type=Path, required=True)
    parser.add_argument("--meta-analysis", type=Path, required=True)
    parser.add_argument("--chapman", type=Path, required=True)
    parser.add_argument("--cpsc", type=Path, required=True)
    parser.add_argument("--stage15", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    map_path = arguments.rank_maps / "map_cells.parquet"
    effects_path = arguments.meta_analysis / "effects.parquet"
    gate_path = arguments.stage15 / "decision.v3.json"
    for required in (
        map_path,
        effects_path,
        gate_path,
        arguments.chapman / "patient_metrics.parquet",
        arguments.cpsc / "patient_metrics.parquet",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    map_source = robust_map_figure(pd.read_parquet(map_path), output)
    gain_source = gain_figure(pd.read_parquet(effects_path), output)
    map_source.to_parquet(output / "figure1_source.parquet", index=False, compression="zstd")
    gain_source.to_parquet(output / "figure2_source.parquet", index=False, compression="zstd")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    summary = figure_summary(
        output,
        stage15_status=str(gate["status"]),
        input_paths={
            "rank_map": map_path,
            "effects": effects_path,
            "stage15": gate_path,
            "chapman": arguments.chapman / "patient_metrics.parquet",
            "cpsc2018": arguments.cpsc / "patient_metrics.parquet",
        },
    )
    _write_json(output / "summary.v3.json", summary)


if __name__ == "__main__":
    main()
