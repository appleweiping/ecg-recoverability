from pathlib import Path

import pandas as pd

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.physics import LEADS
from ecgcert.protocol import deep_configuration_panel
from experiments.paper_figures_v3 import figure_summary, gain_figure, robust_map_figure


def test_two_primary_figures_have_complete_source_tables(tmp_path):
    rows = []
    for segment in ("QRS", "ST", "T"):
        for configuration in deep_configuration_panel():
            name = "+".join(configuration)
            for target_index, target in enumerate(LEADS):
                rows.append({
                    "segment": segment,
                    "configuration": name,
                    "target": target,
                    "ambiguity_robust_mv": 0.01 * (target_index + 1),
                })
    source = robust_map_figure(pd.DataFrame(rows), tmp_path)
    assert len(source) == 3 * 64 * 12
    effects = pd.DataFrame([
        {"cohort": "PTB-XL", "point": 0.1, "ci95": [0.02, 0.18]},
        {"cohort": "chapman", "point": 0.05, "ci95": [0.01, 0.09]},
        {"cohort": "cpsc2018", "point": 0.02, "ci95": [-0.01, 0.05]},
    ])
    gain = gain_figure(effects, tmp_path)
    assert len(gain) == 3
    source.to_parquet(tmp_path / "figure1_source.parquet", index=False)
    gain.to_parquet(tmp_path / "figure2_source.parquet", index=False)
    assert (tmp_path / "figure1_robust_map.pdf").is_file()
    assert (tmp_path / "figure2_prediction_gain.pdf").is_file()
    inputs: dict[str, Path] = {}
    for name in ("rank_map", "effects", "stage15", "chapman", "cpsc2018"):
        path = tmp_path / f"input-{name}"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    summary = figure_summary(
        tmp_path, stage15_status="PIVOT", input_paths=inputs
    )
    assert set(summary["artifacts_sha256"]) == set(FIGURE_ARTIFACTS)
    assert summary["artifacts_sha256"] == direct_artifact_hashes(tmp_path)
    assert summary["input_sha256"] == {
        name: lineage.artifact_sha256(path) for name, path in inputs.items()
    }
