import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.benchmarking import (
    EXPECTED_METHODS,
    evaluate_zero_transfer_bundles,
    load_benchmark_bundles,
)
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.data.manifest import DatasetManifest, ManifestRecord
from ecgcert.estimators.official import ECG_RECOVER, IMPUTE_ECG
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.reconstruction import (
    EvaluationRecord,
    ModelBundleError,
    ObservedSampleViolation,
    SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION,
)
from experiments import external_validation_v3 as external
from experiments.robust_maps_v3 import SCHEMA_VERSION as MAP_SCHEMA_VERSION


SOURCE_MANIFEST_SHA256 = "a" * 64
RANK_MAPS_SHA256 = "b" * 64


@pytest.fixture
def parquet_store(monkeypatch):
    store = {}

    def fake_to_parquet(frame, path, **_kwargs):
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"PAR1-test-fixture")
        store[destination] = frame.copy(deep=True)

    def fake_read_parquet(path, **_kwargs):
        source = Path(path).resolve()
        if source not in store:
            raise FileNotFoundError(source)
        return store[source].copy(deep=True)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    return store


def _external_manifest(tmp_path: Path, n_records: int = 12) -> DatasetManifest:
    records = tuple(
        ManifestRecord(
            record_id=f"record-{index:03d}",
            patient_id=f"patient-{index:03d}",
            relative_header=f"record-{index:03d}.hea",
            header_sha256="0" * 64,
            signal_file=f"record-{index:03d}.dat",
            signal_size_bytes=1,
            signal_sha256="1" * 64,
        )
        for index in range(n_records)
    )
    manifest = DatasetManifest(
        cohort="chapman",
        version="fixture-v1",
        source_url="https://example.invalid/chapman",
        root=str(tmp_path / "unread-external-data"),
        records=records,
        split_salt="external-validation-test-v1",
    )
    split = manifest.split()
    assert split.train and split.tune and split.test
    return manifest


def _evaluation_records(
    manifest: DatasetManifest,
    record_ids,
    *,
    length: int = 6,
) -> tuple[EvaluationRecord, ...]:
    by_id = {record.record_id: record for record in manifest.records}
    segments = {
        "QRS": np.arange(0, length // 3, dtype=np.int64),
        "ST": np.arange(length // 3, 2 * length // 3, dtype=np.int64),
        "T": np.arange(2 * length // 3, length, dtype=np.int64),
    }
    out = []
    for index, raw_record_id in enumerate(record_ids):
        record_id = str(raw_record_id)
        rng = np.random.default_rng(1000 + index)
        signal = rng.normal(0.0, 0.2, size=(12, length))
        signal += np.arange(1, 13, dtype=float)[:, None]
        out.append(
            EvaluationRecord(
                patient_id=by_id[record_id].patient_id,
                record_id=record_id,
                signal=signal,
                segment_indices=segments,
            )
        )
    return tuple(out)


def _artifact_descriptor(path: Path) -> dict:
    return {"path": path.name, "sha256": lineage.artifact_sha256(path)}


def _write_benchmark_bundle(root: Path, method: str) -> Path:
    root.mkdir(parents=True)
    model_path = root / "model.bin"
    model_path.write_bytes(f"{method}-checkpoint".encode("utf-8"))
    checkpoint_sha256 = lineage.artifact_sha256(model_path)
    predictor_path = root / "training_predictors.parquet"
    predictor_path.write_bytes(b"PAR1-identical-folds1-7-predictors")
    predictor_descriptor = {
        "path": predictor_path.name,
        "sha256": lineage.artifact_sha256(predictor_path),
        "source_partition": "PTB-XL/folds1-7/train",
    }
    panel = deep_configuration_panel()
    training = {
        "cohort": "PTB-XL",
        "train_role": "folds1-7",
        "evaluation_roles": {
            "tune": "fold8/tune",
            "calibration": "fold9/calibration",
            "test": "fold10/test",
        },
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "delineator": "dwt",
        "signal_unit": "raw_mV",
        "mask": "whole-lead; identical across methods",
        "simple_predictors": {"heldout_target_statistics_used": False},
        "n_train_records": 7,
        "train_signals_sha256": "d" * 64,
        "model_seeds": [0],
        "release": False,
        "subsampled": False,
        "n_configurations": len(panel),
        "configuration_panel_sha256": configuration_panel_sha256(panel),
    }
    if method in {"lowrank", "ridge"}:
        models = [
            {
                "path": model_path.name,
                "sha256": checkpoint_sha256,
                "seed": 0,
                "configuration": list(configuration),
            }
            for configuration in panel
        ]
    else:
        models = [{"path": model_path.name, "sha256": checkpoint_sha256, "seed": 0}]

    official = None
    if method == "imputeecg":
        official = {
            "repository": IMPUTE_ECG.repository,
            "commit": IMPUTE_ECG.commit,
            "source_dir": "upstreams/ImputeECG",
            "integration_config_sha256": "e" * 64,
        }
    elif method == "ecgrecover":
        bridge = ["python", "bridge.py", "{input}", "{output}", "{checkpoint}"]
        training.update(
            {
                "n_configurations": 1,
                "configuration_panel_sha256": configuration_panel_sha256((("I",),)),
            }
        )
        models[0]["configuration"] = ["I"]
        models[0]["inference_bridge"] = bridge
        official = {
            "repository": ECG_RECOVER.repository,
            "commit": ECG_RECOVER.commit,
            "source_dir": "upstreams/ECGrecover",
            "input_lead": "I",
            "inference_bridge": bridge,
            "integration_config_sha256": "f" * 64,
        }

    metadata = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "method": method,
        "adapter_class": f"fixture.{method}",
        "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
        "models": models,
        "training_config": training,
        "tuning_config": {"fixture": True},
        "tuning_source": "frozen-test-fixture",
        "training_predictors": predictor_descriptor,
    }
    if official is not None:
        metadata["official"] = official
    bundle_path = root / "bundle.v3.json"
    bundle_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    metrics_path = root / "patient_metrics.parquet"
    metrics_path.write_bytes(b"PAR1-benchmark-fixture")
    audit_path = root / "evaluation_audit.json"
    audit_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "method": method,
        "adapter_class": metadata["adapter_class"],
        "load_helper": metadata["load_helper"],
        "training_config": training,
        "tuning_config": metadata["tuning_config"],
        "tuning_source": metadata["tuning_source"],
        "manifest": {
            "sha256": SOURCE_MANIFEST_SHA256,
            "split_sha256": "c" * 64,
        },
        "rank_maps_sha256": RANK_MAPS_SHA256,
        "official": official,
        "artifacts": {
            "bundle": _artifact_descriptor(bundle_path),
            "patient_metrics": _artifact_descriptor(metrics_path),
            "evaluation_audit": _artifact_descriptor(audit_path),
            "training_predictors": _artifact_descriptor(predictor_path),
        },
    }
    (root / "summary.v3.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return root


def _benchmark_bundles(tmp_path: Path) -> list[Path]:
    return [
        _write_benchmark_bundle(tmp_path / f"benchmark-{method}", method)
        for method in EXPECTED_METHODS
    ]


class _MaskedMeanReconstructor:
    def reconstruct(self, signal, observed_mask):
        source = np.asarray(signal, dtype=float)
        mask = np.asarray(observed_mask, dtype=bool)
        denominator = np.maximum(mask.sum(axis=0), 1)
        mean = np.where(mask, source, 0.0).sum(axis=0) / denominator
        return np.where(mask, source, mean[None, :])


class _ObservedCorruptingReconstructor(_MaskedMeanReconstructor):
    def reconstruct(self, signal, observed_mask):
        prediction = super().reconstruct(signal, observed_mask)
        prediction[np.asarray(observed_mask, dtype=bool)] += 1e-9
        return prediction


def _training_predictors(_root):
    return {
        (segment, "+".join(configuration), target): (1.0, 0.5)
        for segment in PRIMARY_SEGMENTS
        for configuration in deep_configuration_panel()
        for target in CANONICAL_LEADS
    }


def _zero_arguments(tmp_path: Path, bundles: list[Path]) -> argparse.Namespace:
    return argparse.Namespace(
        mode="zero-transfer",
        cohort="chapman",
        source_manifest=tmp_path / "unused-source-manifest.json",
        target_manifest=tmp_path / "unused-target-manifest.json",
        rank_maps=tmp_path / "unused-rank-maps",
        primary_rank_maps=None,
        benchmark=bundles,
        output_dir=tmp_path / "zero-output",
        device="cpu",
        delineator="dwt",
        n_bootstrap=BOOTSTRAP_REPLICATES,
        seed=7,
        release=False,
    )


def test_zero_transfer_scores_locked_panel_and_writes_patient_metrics(
    tmp_path, parquet_store
):
    manifest = _external_manifest(tmp_path)
    records = _evaluation_records(manifest, manifest.split().test)
    bundles = _benchmark_bundles(tmp_path)
    load_calls = []

    def loader(root, method, seed, *, device):
        load_calls.append((Path(root), method, seed, device))
        return _MaskedMeanReconstructor()

    arguments = _zero_arguments(tmp_path, bundles)
    summary = external.run_zero_transfer(
        arguments,
        target_manifest=manifest,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        rank_maps_sha256=RANK_MAPS_SHA256,
        records=records,
        model_loader=loader,
        predictor_loader=_training_predictors,
    )

    metrics_path = (arguments.output_dir / "patient_metrics.parquet").resolve()
    metrics = parquet_store[metrics_path]
    assert set(metrics["cohort"]) == {"chapman"}
    assert set(metrics["partition"]) == {"test"}
    assert set(metrics["segment"]) == set(PRIMARY_SEGMENTS)
    assert set(metrics["method"]) == set(EXPECTED_METHODS)
    for method in EXPECTED_METHODS[:-1]:
        assert metrics.loc[metrics["method"] == method, "configuration"].nunique() == 64
    assert set(metrics.loc[metrics["method"] == "ecgrecover", "configuration"]) == {"I"}
    assert np.isfinite(metrics[["log_rmse_mv", "outcome_log_rmse"]]).all().all()
    assert np.array_equal(metrics["log_rmse_mv"], metrics["outcome_log_rmse"])
    assert set(metrics["target_rms"]) == {1.0}
    assert all(
        row.target not in row.observed_leads.split(",")
        for row in metrics.itertuples(index=False)
    )
    assert len(load_calls) == len(EXPECTED_METHODS)
    assert summary["external_training_or_adaptation"] == "forbidden_and_not_performed"
    assert summary["observed_sample_integrity"] == "passed_exact_pointwise"
    assert len(summary["training_predictors_content_sha256"]) == 64
    audit = json.loads(
        (arguments.output_dir / "evaluation_audit.json").read_text(encoding="utf-8")
    )
    assert audit["no_external_fit"] is True
    assert audit["data_audit"]["summary"]["n_total"] == len(manifest.split().test)


def test_zero_transfer_fails_on_observed_sample_change(tmp_path, parquet_store):
    manifest = _external_manifest(tmp_path)
    records = _evaluation_records(manifest, manifest.split().test)
    arguments = _zero_arguments(tmp_path, _benchmark_bundles(tmp_path))

    with pytest.raises(ObservedSampleViolation, match="changed observed"):
        external.run_zero_transfer(
            arguments,
            target_manifest=manifest,
            source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            rank_maps_sha256=RANK_MAPS_SHA256,
            records=records,
            model_loader=lambda *_args, **_kwargs: _ObservedCorruptingReconstructor(),
            predictor_loader=_training_predictors,
        )
    assert (arguments.output_dir / "patient_metrics.parquet").resolve() not in parquet_store


def _reauthenticate_bundle(root: Path) -> None:
    summary_path = root / "summary.v3.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["bundle"]["sha256"] = lineage.artifact_sha256(
        root / "bundle.v3.json"
    )
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")


def test_bundle_validation_rejects_missing_checkpoint_and_seed_bridge(tmp_path):
    missing_checkpoint = _benchmark_bundles(tmp_path / "missing-checkpoint")
    (missing_checkpoint[0] / "model.bin").unlink()
    with pytest.raises(FileNotFoundError):
        load_benchmark_bundles(
            missing_checkpoint,
            source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            rank_maps_sha256=RANK_MAPS_SHA256,
            release=False,
        )

    missing_bridge = _benchmark_bundles(tmp_path / "missing-bridge")
    ecgrecover_root = missing_bridge[-1]
    bundle_path = ecgrecover_root / "bundle.v3.json"
    metadata = json.loads(bundle_path.read_text(encoding="utf-8"))
    metadata["models"][0].pop("inference_bridge")
    bundle_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    _reauthenticate_bundle(ecgrecover_root)
    with pytest.raises(ModelBundleError, match="seed 0 inference bridge"):
        load_benchmark_bundles(
            missing_bridge,
            source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            rank_maps_sha256=RANK_MAPS_SHA256,
            release=False,
        )


def test_zero_transfer_rejects_disagreeing_ptb_predictor_content(tmp_path):
    roots = _benchmark_bundles(tmp_path)
    bundles = load_benchmark_bundles(
        roots,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        rank_maps_sha256=RANK_MAPS_SHA256,
        release=False,
    )
    manifest = _external_manifest(tmp_path)
    record = _evaluation_records(manifest, manifest.split().test[:1])

    def disagreeing_predictors(root):
        predictors = _training_predictors(root)
        if Path(root).name.endswith("ridge"):
            key = ("QRS", "I", "II")
            predictors[key] = (2.0, 0.5)
        return predictors

    with pytest.raises(ModelBundleError, match="disagree on folds1-7 predictor content"):
        evaluate_zero_transfer_bundles(
            bundles,
            record,
            cohort="chapman",
            device="cpu",
            model_loader=lambda *_args, **_kwargs: pytest.fail(
                "models must not load before predictor agreement passes"
            ),
            predictor_loader=disagreeing_predictors,
        )


def _primary_rank_map(root: Path) -> pd.DataFrame:
    configurations = all_independent_configurations()
    rows = []
    for segment_index, segment in enumerate(PRIMARY_SEGMENTS):
        for config_index, configuration in enumerate(configurations):
            for target_index, target in enumerate(CANONICAL_LEADS):
                rows.append(
                    {
                        "schema_version": MAP_SCHEMA_VERSION,
                        "segment": segment,
                        "configuration": "+".join(configuration),
                        "target": target,
                        "target_observed": target in configuration,
                        "recoverability_lower": float(
                            segment_index * 10000 + config_index * 20 + target_index
                        )
                        / 30000.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / "map_cells.parquet", index=False)
    summary = {
        "schema_version": MAP_SCHEMA_VERSION,
        "status": "complete",
        "cohort": "PTB-XL",
        "population": "all",
        "analysis_mode": "primary",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "ranks": list(RANK_GRID),
        "n_configurations": len(configurations),
        "n_map_cells": len(frame),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "patient",
        "deep_panel_sha256": configuration_panel_sha256(),
        "protocol": asdict(StudyProtocol()),
        "protocol_sha256": lineage.canonical_sha256(asdict(StudyProtocol())),
        "observation_variance_mv2": 1e-4,
        "artifacts": {"map_cells": "map_cells.parquet"},
    }
    (root / "summary.v3.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return frame


def test_cohort_maps_fit_only_train60_and_compare_ptb_rankings(
    tmp_path, parquet_store, monkeypatch
):
    manifest = _external_manifest(tmp_path)
    split = manifest.split()
    train_records = _evaluation_records(manifest, split.train)
    segment_data = external._segment_samples(train_records, max_per_record=40, seed=3)
    bad_segment_data = dict(segment_data)
    X, record_ids, patient_ids = bad_segment_data["QRS"]
    leaking_ids = record_ids.copy()
    leaking_ids[0] = str(split.test[0])
    bad_segment_data["QRS"] = (X, leaking_ids, patient_ids)
    with pytest.raises(ValueError, match="outside the 60% train split"):
        external._validate_cohort_segment_data(
            bad_segment_data,
            train_record_ids=split.train,
            test_record_ids=split.test,
            patient_by_record={
                record.record_id: record.patient_id for record in manifest.records
            },
        )
    primary_root = tmp_path / "primary-rank-map"
    primary_root.mkdir()
    _primary_rank_map(primary_root)
    captured_patient_ids = []

    class FakeBank:
        n_boot = 2
        ranks = (2,)
        rejected_draws = 0
        rejection_fraction = 0.0

    def fake_bootstrap(X, patient_ids, **_kwargs):
        assert np.asarray(X).shape[1] == 12
        captured_patient_ids.extend(map(str, patient_ids))
        return FakeBank()

    def fake_summarize(_bank, configurations, *, segment, **_kwargs):
        all_configs = all_independent_configurations()
        global_index = {tuple(config): index for index, config in enumerate(all_configs)}
        rank_rows = []
        map_rows = []
        segment_index = PRIMARY_SEGMENTS.index(segment)
        for configuration in map(tuple, configurations):
            config_id = "+".join(configuration)
            config_index = global_index[configuration]
            for target_index, target in enumerate(CANONICAL_LEADS):
                common = {
                    "schema_version": MAP_SCHEMA_VERSION,
                    "segment": segment,
                    "configuration": config_id,
                    "target": target,
                    "target_observed": target in configuration,
                }
                rank_rows.append({**common, "rank": 2})
                map_rows.append(
                    {
                        **common,
                        "recoverability_lower": float(
                            segment_index * 10000 + config_index * 20 + target_index
                        )
                        / 30000.0,
                    }
                )
        return pd.DataFrame(rank_rows), pd.DataFrame(map_rows)

    monkeypatch.setattr(external, "bootstrap_spatial_model_bank", fake_bootstrap)
    monkeypatch.setattr(external, "summarize_model_bank", fake_summarize)
    arguments = argparse.Namespace(
        cohort="chapman",
        target_manifest=tmp_path / "unused-target-manifest.json",
        primary_rank_maps=primary_root,
        output_dir=tmp_path / "cohort-output",
        delineator="dwt",
        n_bootstrap=2,
        seed=19,
    )
    summary = external.run_cohort_maps(
        arguments,
        target_manifest=manifest,
        records=train_records,
        configurations=(("I",), ("II",)),
        ranks=(2,),
    )

    train_patients = {
        record.patient_id
        for record in manifest.records
        if record.record_id in set(map(str, split.train))
    }
    test_patients = {
        record.patient_id
        for record in manifest.records
        if record.record_id in set(map(str, split.test))
    }
    assert set(captured_patient_ids) == train_patients
    assert not set(captured_patient_ids) & test_patients
    assert summary["test_records_accessed"] == 0
    assert summary["overall_spearman_rho"] == pytest.approx(1.0)
    agreement = parquet_store[
        (arguments.output_dir / "ranking_spearman.parquet").resolve()
    ]
    assert agreement.iloc[0]["ranking_metric"].endswith("missing targets only")
    audit = json.loads(
        (arguments.output_dir / "evaluation_audit.json").read_text(encoding="utf-8")
    )
    assert audit["partition"] == "train"
    assert audit["test_records_accessed"] == 0


def test_release_argument_validation_requires_full_protocol(tmp_path):
    outside_artifacts = tmp_path / "release-output"
    arguments = external.build_parser().parse_args(
        [
            "--mode",
            "cohort-maps",
            "--cohort",
            "chapman",
            "--target-manifest",
            str(tmp_path / "target.json"),
            "--primary-rank-maps",
            str(tmp_path / "primary"),
            "--output-dir",
            str(outside_artifacts),
            "--n-bootstrap",
            "2",
            "--release",
        ]
    )
    with pytest.raises(ValueError, match="exactly 2000 patient bootstraps"):
        external.validate_arguments(arguments)
