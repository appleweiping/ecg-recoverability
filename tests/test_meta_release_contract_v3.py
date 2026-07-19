import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.benchmarking import EXPECTED_METHODS, RELEASE_NEURAL_SEEDS
from ecgcert.data.manifest import (
    RELEASE_COHORT_CONTRACTS,
    DatasetManifest,
    ManifestRecord,
)
from ecgcert.protocol import PRIMARY_SEGMENTS, configuration_panel_sha256, deep_configuration_panel
from ecgcert.reconstruction import SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION
from experiments.meta_analysis_v3 import (
    COMMON_PANEL_METHODS,
    EXTERNAL_VALIDATION_SCHEMA_VERSION,
    _attach_robust_map,
    _external_manifest_index,
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


def test_rank_map_diagnostics_cannot_overwrite_folds1_7_predictors() -> None:
    metrics = pd.DataFrame(
        [
            {
                "segment": "QRS",
                "configuration": "I",
                "target": "II",
                "target_rms": 1.25,
                "max_target_observed_correlation": 0.4,
            }
        ]
    )
    rank_map = pd.DataFrame(
        [
            {
                "segment": "QRS",
                "configuration": "I",
                "target": "II",
                "ambiguity_robust_mv": 0.2,
                "configuration_rank_max": 3,
                "log10_condition_max": 1.5,
                "target_rms": 99.0,
                "max_target_observed_correlation": 0.99,
            }
        ]
    )
    attached = _attach_robust_map(metrics, rank_map)
    assert attached.loc[0, "target_rms"] == 1.25
    assert attached.loc[0, "max_target_observed_correlation"] == 0.4
    assert attached.loc[0, "ambiguity_robust_mv"] == 0.2


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
        predictor_sha256 = "e" * 64
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
    target_manifest = DatasetManifest(
        cohort="chapman",
        version="fixture-v1",
        source_url="https://example.invalid/chapman",
        root=str(tmp_path / "external-data"),
        records=tuple(
            ManifestRecord(
                record_id=f"record-{index:03d}",
                patient_id=f"patient-{index:03d}",
                relative_header=f"record-{index:03d}.hea",
                header_sha256="0" * 64,
                signal_file=f"record-{index:03d}.dat",
                signal_size_bytes=1,
                signal_sha256="1" * 64,
            )
            for index in range(100)
        ),
        split_salt="meta-release-contract-v1",
    )
    target_split = target_manifest.split()
    assert target_split.train and target_split.tune and target_split.test
    target_manifest_path = tmp_path / "chapman-manifest.json"
    target_manifest_path.write_text(
        json.dumps(target_manifest.to_dict()), encoding="utf-8"
    )
    target_records = {
        record.record_id: record.patient_id for record in target_manifest.records
    }
    data_audit_records = [
        {
            "record_id": str(record_id),
            "patient_id": target_records[str(record_id)],
            "status": "included",
            "reason": None,
            "segment_counts": {"QRS": 10, "ST": 8, "T": 6},
        }
        for record_id in target_split.test
    ]
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
        "target_manifest_sha256": target_manifest.sha256(),
        "target_split_sha256": target_split.sha256(),
        "training_predictors_content_sha256": "f" * 64,
        "requested_record_ids_sha256": lineage.canonical_sha256(
            sorted(str(value) for value in target_split.test)
        ),
        "data_audit": {
            "summary": {
                "n_total": len(data_audit_records),
                "n_included": len(data_audit_records),
                "n_excluded": 0,
                "exclusion_reasons": {},
            },
            "records": data_audit_records,
        },
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
        "n_test_records_requested": len(data_audit_records),
        "n_test_records_included": len(data_audit_records),
        "n_test_records_excluded": 0,
        "n_patient_metric_rows": 1,
        "target_manifest_sha256": target_manifest.sha256(),
        "target_split_sha256": target_split.sha256(),
        "training_predictors_content_sha256": "f" * 64,
        "benchmark_bundles": inventory,
        "artifacts": {
            "patient_metrics": _descriptor(metrics_path),
            "evaluation_audit": _descriptor(audit_path),
        },
    }
    (root / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
    return (
        root,
        target_manifest_path,
        source_manifest_sha256,
        rank_maps_sha256,
        benchmark_bundles,
    )


def test_external_release_accepts_external_audit_schema(tmp_path: Path, monkeypatch) -> None:
    root, target_manifest, source_sha, rank_sha, benchmarks = (
        _external_release_fixture(tmp_path)
    )
    fixture_manifest = DatasetManifest.from_path(target_manifest)
    monkeypatch.setitem(
        RELEASE_COHORT_CONTRACTS,
        "chapman",
        {
            "version": fixture_manifest.version,
            "source_url": fixture_manifest.source_url,
            "n_records": len(fixture_manifest.records),
            "n_patient_ids": len(fixture_manifest.records),
            "patient_id_strategy": fixture_manifest.patient_id_strategy,
        },
    )
    metrics_path, cohort, report, coverage = _validate_external_release_bundle(
        root,
        target_manifest=target_manifest,
        source_manifest_sha256=source_sha,
        rank_maps_sha256=rank_sha,
        predictor_content_sha256="f" * 64,
        benchmark_bundles=benchmarks,
    )
    assert cohort == "chapman"
    import pyarrow.parquet as pq

    assert pq.ParquetFile(metrics_path).metadata.num_rows == 1
    assert report["no_external_fit"] is True
    assert len(coverage.included_record_ids) == len(coverage.attempted_record_ids)


def test_external_manifest_cohort_mapping_rejects_duplicate_names(tmp_path: Path) -> None:
    _, target_manifest, *_rest = _external_release_fixture(tmp_path)
    duplicate = tmp_path / "duplicate-chapman.json"
    duplicate.write_bytes(target_manifest.read_bytes())
    with pytest.raises(ValueError, match="map uniquely"):
        _external_manifest_index((target_manifest, duplicate))
