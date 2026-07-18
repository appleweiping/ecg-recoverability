import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.benchmarking import EXPECTED_METHODS, RELEASE_NEURAL_SEEDS
from ecgcert.protocol import PRIMARY_SEGMENTS, configuration_panel_sha256, deep_configuration_panel
from ecgcert.reconstruction import SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION
from experiments.meta_analysis_v3 import (
    COMMON_PANEL_METHODS,
    EXTERNAL_VALIDATION_SCHEMA_VERSION,
    _validate_common_panel_metrics,
    _validate_external_release_bundle,
)


def _panel_metrics_with_partition_holes() -> pd.DataFrame:
    configurations = ["+".join(value) for value in deep_configuration_panel()]
    rows = []
    for method in COMMON_PANEL_METHODS:
        seeds = RELEASE_NEURAL_SEEDS if method in {"masked-unet", "imputeecg"} else (0,)
        for seed in seeds:
            for partition_index, partition in enumerate(("tune", "calibration", "test")):
                # The union is the full panel, but every individual partition is incomplete.
                included = [
                    value
                    for index, value in enumerate(configurations)
                    if index != partition_index
                ]
                for segment in PRIMARY_SEGMENTS:
                    for configuration in included:
                        rows.append(
                            {
                                "schema_version": BENCHMARK_SCHEMA_VERSION,
                                "cohort": "PTB-XL",
                                "partition": partition,
                                "patient_id": "patient-1",
                                "method": method,
                                "model_seed": seed,
                                "segment": segment,
                                "configuration": configuration,
                                "target": "III",
                                "outcome_log_rmse": -1.0,
                            }
                        )
    rows.append(
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "cohort": "PTB-XL",
            "partition": "test",
            "patient_id": "patient-1",
            "method": "ecgrecover",
            "model_seed": 0,
            "segment": "QRS",
            "configuration": "I",
            "target": "III",
            "outcome_log_rmse": -1.0,
        }
    )
    return pd.DataFrame(rows)


def test_release_panel_is_complete_within_every_partition() -> None:
    frame = _panel_metrics_with_partition_holes()
    with pytest.raises(ValueError, match="partition .*64-configuration panel"):
        _validate_common_panel_metrics(
            frame,
            cohort="PTB-XL",
            partitions={"tune", "calibration", "test"},
        )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": lineage.artifact_sha256(path)}


def _external_release_fixture(tmp_path: Path):
    source_manifest_sha256 = "a" * 64
    rank_maps_sha256 = "b" * 64
    benchmark_bundles = {}
    inventory = {}
    for method in EXPECTED_METHODS:
        method_root = tmp_path / f"benchmark-{method}"
        method_root.mkdir()
        bundle_path = method_root / "bundle.v3.json"
        bundle_path.write_text(json.dumps({"method": method}), encoding="utf-8")
        seeds = tuple(RELEASE_NEURAL_SEEDS) if method in {
            "masked-unet",
            "imputeecg",
            "ecgrecover",
        } else (0,)
        configurations = (("I",),) if method == "ecgrecover" else deep_configuration_panel()
        predictor_sha256 = (method[0] * 64)[:64]
        benchmark_bundles[method] = SimpleNamespace(
            root=method_root,
            seeds=seeds,
            configurations=configurations,
            training_predictors_sha256=predictor_sha256,
        )
        inventory[method] = {
            "bundle_sha256": lineage.artifact_sha256(bundle_path),
            "seeds": list(seeds),
            "n_configurations": len(configurations),
            "training_predictors_sha256": predictor_sha256,
        }

    root = tmp_path / "external"
    root.mkdir()
    metrics_path = root / "patient_metrics.parquet"
    pd.DataFrame([{"patient_id": "external-1"}]).to_parquet(metrics_path, index=False)
    audit_path = root / "evaluation_audit.json"
    audit = {
        "schema_version": EXTERNAL_VALIDATION_SCHEMA_VERSION,
        "mode": "zero-transfer",
        "cohort": "chapman",
        "partition": "test",
        "no_external_fit": True,
        "source_manifest_sha256": source_manifest_sha256,
        "rank_maps_sha256": rank_maps_sha256,
        "target_manifest_sha256": "c" * 64,
        "target_split_sha256": "d" * 64,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "mode": "zero-transfer",
        "partition": "test",
        "external_training_or_adaptation": "forbidden_and_not_performed",
        "source_manifest_sha256": source_manifest_sha256,
        "rank_maps_sha256": rank_maps_sha256,
        "configuration_panel_sha256": configuration_panel_sha256(),
        "common_panel_methods": list(COMMON_PANEL_METHODS),
        "cohort": "chapman",
        "n_test_records_included": 1,
        "n_patient_metric_rows": 1,
        "target_manifest_sha256": "c" * 64,
        "target_split_sha256": "d" * 64,
        "benchmark_bundles": inventory,
        "artifacts": {
            "patient_metrics": _descriptor(metrics_path),
            "evaluation_audit": _descriptor(audit_path),
        },
    }
    (root / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
    return root, source_manifest_sha256, rank_maps_sha256, benchmark_bundles


def test_external_release_accepts_external_audit_schema(tmp_path: Path) -> None:
    root, source_sha, rank_sha, benchmarks = _external_release_fixture(tmp_path)
    metrics, cohort, report = _validate_external_release_bundle(
        root,
        source_manifest_sha256=source_sha,
        rank_maps_sha256=rank_sha,
        benchmark_bundles=benchmarks,
    )
    assert cohort == "chapman"
    assert len(metrics) == 1
    assert report["no_external_fit"] is True
