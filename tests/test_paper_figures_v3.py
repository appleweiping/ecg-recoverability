from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.physics import LEADS
from ecgcert.protocol import deep_configuration_panel
from experiments.paper_figures_v3 import figure_summary, gain_figure, robust_map_figure


def _rank_map_cells() -> pd.DataFrame:
    rows = []
    for segment in ("QRS", "ST", "T"):
        for configuration in deep_configuration_panel():
            name = "+".join(configuration)
            for target_index, target in enumerate(LEADS):
                observed = target in configuration
                rows.append(
                    {
                        "segment": segment,
                        "configuration": name,
                        "target": target,
                        "target_observed": observed,
                        "ambiguity_robust_mv": (
                            999.0 if observed else 0.01 * (target_index + 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_two_primary_figures_have_complete_source_tables(tmp_path):
    rank_map = _rank_map_cells()
    source = robust_map_figure(rank_map, tmp_path)
    assert len(source) == 3 * 64 * 12
    observed = source["target_observed"].to_numpy(dtype=bool)
    assert observed.any()
    assert source.loc[observed, "ambiguity_robust_mv"].isna().all()
    assert np.isfinite(source.loc[~observed, "ambiguity_robust_mv"]).all()
    assert rank_map.loc[rank_map["target_observed"], "ambiguity_robust_mv"].eq(999.0).all()
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
    assert summary["figure1_population"] == "missing_targets_only"
    assert summary["figure1_observed_target_policy"] == "masked_to_null"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="target_observed"), "lacks columns"),
        (
            lambda frame: frame.assign(target_observed="False"),
            "strict booleans",
        ),
        (
            lambda frame: frame.assign(
                target_observed=lambda value: value["target_observed"].mask(
                    value.index == 0, not bool(value.loc[0, "target_observed"])
                )
            ),
            "disagrees with configuration membership",
        ),
        (lambda frame: frame.iloc[:-1].copy(), "requires all 64 x 12 x 3"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate frozen cells",
        ),
        (
            lambda frame: frame.assign(
                ambiguity_robust_mv=lambda value: value["ambiguity_robust_mv"].mask(
                    value.index == 1, np.nan
                )
            ),
            "finite and non-negative",
        ),
    ],
)
def test_robust_map_figure_fails_closed_on_untrusted_observed_target_mask(
    tmp_path, mutation, message
):
    with pytest.raises(ValueError, match=message):
        robust_map_figure(mutation(_rank_map_cells()), tmp_path)
