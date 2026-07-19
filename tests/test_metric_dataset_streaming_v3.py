import ctypes
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert.benchmarking import (
    EXPECTED_METHODS,
    FittedBenchmarkBundle,
    evaluate_zero_transfer_bundles,
)
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.protocol import PRIMARY_SEGMENTS, deep_configuration_panel
from ecgcert.reconstruction import (
    REQUIRED_METRIC_COLUMNS,
    SCHEMA_VERSION,
    EvaluationRecord,
    MetricDatasetWriter,
    MetricShardKey,
    ObservedSampleViolation,
    ReconstructionContractError,
    metric_coverage_contract,
    metric_frame_coverage_contract,
)
from experiments import reconstruction_benchmark_v3 as benchmark


def _metric_frame(
    key: MetricShardKey,
    *,
    n_patients: int,
    target_rms: float = 1.0,
) -> pd.DataFrame:
    rmse = 0.25
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "cohort": key.cohort,
            "partition": key.partition,
            "patient_id": [f"patient-{index:07d}" for index in range(n_patients)],
            "segment": "QRS",
            "configuration": key.configuration,
            "target": "II",
            "method": key.method,
            "model_seed": key.model_seed,
            "observed_leads": "I",
            "n_observed": 1,
            "n_records": 1,
            "n_samples": 20,
            "rmse_mv": rmse,
            "log_rmse_mv": np.log(rmse),
            "target_rms": target_rms,
            "max_target_observed_correlation": 0.5,
            "target_rms_mv": 1.0,
            "normalized_rmse": rmse,
            "outcome_log_rmse": np.log(rmse),
        },
        columns=REQUIRED_METRIC_COLUMNS,
    )


def _coverage(
    keys: tuple[MetricShardKey, ...], *, n_patients: int
) -> dict[MetricShardKey, object]:
    return {
        key: metric_frame_coverage_contract(_metric_frame(key, n_patients=n_patients))
        for key in keys
    }


def _peak_rss_bytes() -> int:
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if value < 10**10 else 1))

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def test_many_shards_publish_without_concat_and_with_bounded_peak_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    keys = tuple(
        MetricShardKey.from_values(
            cohort="PTB-XL",
            partition="test",
            method="masked-unet",
            model_seed=seed,
            configuration=("I",),
        )
        for seed in range(12)
    )
    writer = MetricDatasetWriter(
        tmp_path,
        expected_shards=keys,
        expected_coverage=_coverage(keys, n_patients=250),
        dataset_identity={"fixture": "many-shards-v1"},
    )

    def forbid_concat(*_args, **_kwargs):
        raise AssertionError("bounded metric publication must not call pandas.concat")

    monkeypatch.setattr(pd, "concat", forbid_concat)
    rss_before = _peak_rss_bytes()
    for key in keys:
        writer.write_shard(
            _metric_frame(key, n_patients=250),
            key,
            observed_sample_integrity=True,
        )
    summary = writer.finalize(summary={"method": "masked-unet"})
    rss_after = _peak_rss_bytes()

    assert rss_after - rss_before < 256 * 1024 * 1024
    assert summary["n_patient_metric_rows"] == len(keys) * 250
    metadata = parquet.ParquetFile(tmp_path / "patient_metrics.parquet").metadata
    assert metadata.num_row_groups == len(keys)
    assert metadata.num_rows == len(keys) * 250
    inventory = json.loads(
        (tmp_path / "patient_metrics.inventory.v1.json").read_text(encoding="utf-8")
    )
    assert inventory["status"] == "complete"
    assert len(inventory["expected_coverage_sha256"]) == 64
    assert inventory["n_completed_shards"] == len(keys)
    assert inventory["total_rows"] == len(keys) * 250
    assert [shard["row_group_start"] for shard in inventory["shards"]] == list(
        range(len(keys))
    )
    for shard in inventory["shards"]:
        assert len(shard["parquet_sha256"]) == 64
        assert shard["patient_id_range"] == {
            "min": "patient-0000000",
            "max": "patient-0000249",
        }
        assert shard["status"] == "published"
        assert shard["path"] is None
        assert shard["observed_sample_integrity"] == "passed_exact_pointwise"


def test_resume_reuses_authenticated_shards_and_rejects_stale_identity(
    tmp_path: Path,
) -> None:
    keys = tuple(
        MetricShardKey.from_values(
            cohort="chapman",
            partition="test",
            method="ridge",
            model_seed=seed,
            configuration=("I",),
        )
        for seed in range(3)
    )
    identity = {"fixture": "resume-v1", "split_sha256": "a" * 64}
    first = MetricDatasetWriter(
        tmp_path,
        expected_shards=keys,
        expected_coverage=_coverage(keys, n_patients=3),
        dataset_identity=identity,
    )
    first.write_shard(
        _metric_frame(keys[0], n_patients=3),
        keys[0],
        observed_sample_integrity=True,
    )

    resumed = MetricDatasetWriter(
        tmp_path,
        expected_shards=keys,
        expected_coverage=_coverage(keys, n_patients=3),
        dataset_identity=identity,
        resume=True,
    )
    assert resumed.is_complete(keys[0])
    for key in keys[1:]:
        resumed.write_shard(
            _metric_frame(key, n_patients=3),
            key,
            observed_sample_integrity=True,
        )
    resumed.finalize(summary={"cohort": "chapman", "mode": "zero-transfer"})

    complete = MetricDatasetWriter(
        tmp_path,
        expected_shards=keys,
        expected_coverage=_coverage(keys, n_patients=3),
        dataset_identity=identity,
        resume=True,
    )
    assert all(complete.is_complete(key) for key in keys)
    with pytest.raises(Exception, match="dataset identity"):
        MetricDatasetWriter(
            tmp_path,
            expected_shards=keys,
            expected_coverage=_coverage(keys, n_patients=3),
            dataset_identity={"fixture": "different"},
            resume=True,
        )


def test_orphan_shard_recovery_is_bound_to_dataset_identity(tmp_path: Path) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    identity = {"fixture": "orphan-recovery-v1", "manifest_sha256": "a" * 64}
    first = MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity=identity,
    )
    first.write_shard(
        _metric_frame(key, n_patients=2),
        key,
        observed_sample_integrity=True,
    )
    (tmp_path / "patient_metrics.inventory.v1.json").unlink()

    with pytest.raises(ReconstructionContractError, match="identity metadata"):
        MetricDatasetWriter(
            tmp_path,
            expected_shards=(key,),
            expected_coverage=_coverage((key,), n_patients=2),
            dataset_identity={"fixture": "stale-run"},
        )
    assert not (tmp_path / "patient_metrics.inventory.v1.json").exists()

    recovered = MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity=identity,
    )
    assert recovered.is_complete(key)
    recovered.finalize(summary={"method": "ridge"})


def test_inventory_counters_and_observed_attestation_fail_closed(tmp_path: Path) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    identity = {"fixture": "inventory-tamper-v1"}
    writer = MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity=identity,
    )
    with pytest.raises(ObservedSampleViolation, match="exact pointwise"):
        writer.write_shard(
            _metric_frame(key, n_patients=2),
            key,
            observed_sample_integrity=False,
        )
    writer.write_shard(
        _metric_frame(key, n_patients=2),
        key,
        observed_sample_integrity=True,
    )
    inventory_path = tmp_path / "patient_metrics.inventory.v1.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["n_completed_shards"] = 0
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ReconstructionContractError, match="counters"):
        MetricDatasetWriter(
            tmp_path,
            expected_shards=(key,),
            expected_coverage=_coverage((key,), n_patients=2),
            dataset_identity=identity,
            resume=True,
        )


def test_staged_and_published_parquet_sha_tampering_fails_closed(tmp_path: Path) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    coverage = _coverage((key,), n_patients=2)

    staged_root = tmp_path / "staged"
    staged = MetricDatasetWriter(
        staged_root,
        expected_shards=(key,),
        expected_coverage=coverage,
        dataset_identity={"fixture": "staged-tamper-v1"},
    )
    staged.write_shard(
        _metric_frame(key, n_patients=2), key, observed_sample_integrity=True
    )
    inventory = json.loads(
        (staged_root / "patient_metrics.inventory.v1.json").read_text(encoding="utf-8")
    )
    staged_path = staged_root / inventory["shards"][0]["path"]
    staged_path.write_bytes(staged_path.read_bytes() + b"tamper")
    with pytest.raises(ReconstructionContractError, match="missing or changed"):
        MetricDatasetWriter(
            staged_root,
            expected_shards=(key,),
            expected_coverage=coverage,
            dataset_identity={"fixture": "staged-tamper-v1"},
            resume=True,
        )

    published_root = tmp_path / "published"
    published = MetricDatasetWriter(
        published_root,
        expected_shards=(key,),
        expected_coverage=coverage,
        dataset_identity={"fixture": "published-tamper-v1"},
    )
    published.write_shard(
        _metric_frame(key, n_patients=2), key, observed_sample_integrity=True
    )
    published.finalize(summary={"method": "ridge"})
    metrics_path = published_root / "patient_metrics.parquet"
    metrics_path.write_bytes(metrics_path.read_bytes() + b"tamper")
    with pytest.raises(ReconstructionContractError, match="SHA-256 mismatch"):
        MetricDatasetWriter(
            published_root,
            expected_shards=(key,),
            expected_coverage=coverage,
            dataset_identity={"fixture": "published-tamper-v1"},
            resume=True,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("normalized_rmse", 9.0, "normalized_rmse"),
        ("log_rmse_mv", 9.0, "log_rmse_mv"),
        ("n_samples", 0, "positive integers"),
    ),
)
def test_metric_semantics_fail_closed(
    tmp_path: Path, column: str, value: float, message: str
) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    writer = MetricDatasetWriter(
        tmp_path / column,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity={"fixture": column},
    )
    frame = _metric_frame(key, n_patients=2)
    frame[column] = value
    if column == "log_rmse_mv":
        frame["outcome_log_rmse"] = value
    with pytest.raises(ValueError, match=message):
        writer.write_shard(frame, key, observed_sample_integrity=True)


def test_independent_evaluation_coverage_rejects_silently_missing_rows(
    tmp_path: Path,
) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    writer = MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=3),
        dataset_identity={"fixture": "independent-coverage-v1"},
    )
    with pytest.raises(ReconstructionContractError, match="coverage disagrees"):
        writer.write_shard(
            _metric_frame(key, n_patients=2),
            key,
            observed_sample_integrity=True,
        )


def test_unknown_staging_entry_is_not_ignored_on_resume(tmp_path: Path) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="ridge",
        model_seed=0,
        configuration=("I",),
    )
    identity = {"fixture": "unknown-staging-entry-v1"}
    MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity=identity,
    )
    staging = tmp_path / ".patient-metric-shards"
    (staging / "unexpected.txt").write_text("untracked", encoding="utf-8")
    with pytest.raises(ReconstructionContractError, match="unknown file"):
        MetricDatasetWriter(
            tmp_path,
            expected_shards=(key,),
            expected_coverage=_coverage((key,), n_patients=2),
            dataset_identity=identity,
            resume=True,
        )


def test_resume_policy_prevents_cross_checkpoint_shard_reuse(tmp_path: Path) -> None:
    key = MetricShardKey.from_values(
        cohort="PTB-XL",
        partition="test",
        method="masked-unet",
        model_seed=0,
        configuration=("I",),
    )
    identity = {"fixture": "new-training-attempt-v1"}
    MetricDatasetWriter(
        tmp_path,
        expected_shards=(key,),
        expected_coverage=_coverage((key,), n_patients=2),
        dataset_identity=identity,
        allow_resume=False,
    )
    with pytest.raises(ValueError, match="resume is disabled"):
        MetricDatasetWriter(
            tmp_path,
            expected_shards=(key,),
            expected_coverage=_coverage((key,), n_patients=2),
            dataset_identity=identity,
            resume=True,
            allow_resume=False,
        )


def test_incremental_predictor_and_observed_target_validation_fails_closed(
    tmp_path: Path,
) -> None:
    keys = (
        MetricShardKey.from_values(
            cohort="PTB-XL",
            partition="tune",
            method="ridge",
            model_seed=0,
            configuration=("I",),
        ),
        MetricShardKey.from_values(
            cohort="PTB-XL",
            partition="test",
            method="ridge",
            model_seed=0,
            configuration=("I",),
        ),
    )
    writer = MetricDatasetWriter(
        tmp_path,
        expected_shards=keys,
        expected_coverage=_coverage(keys, n_patients=2),
        dataset_identity={"fixture": "validation-v1"},
    )
    writer.write_shard(
        _metric_frame(keys[0], n_patients=2),
        keys[0],
        observed_sample_integrity=True,
    )
    with pytest.raises(ValueError, match="simple predictors must be fixed"):
        writer.write_shard(
            _metric_frame(keys[1], n_patients=2, target_rms=2.0),
            keys[1],
            observed_sample_integrity=True,
        )

    observed = _metric_frame(keys[1], n_patients=2)
    observed["target"] = "I"
    with pytest.raises(ValueError, match="observed targets"):
        writer.write_shard(
            observed,
            keys[1],
            observed_sample_integrity=True,
        )


class _ExactCopyReconstructor:
    def reconstruct(self, signal, _observed_mask):
        return np.asarray(signal, dtype=float).copy()


def _external_record() -> EvaluationRecord:
    signal = np.arange(12 * 8, dtype=float).reshape(12, 8) / 100.0 + 0.1
    indices = np.arange(8, dtype=np.int64)
    return EvaluationRecord(
        patient_id="external-patient",
        record_id="external-record",
        signal=signal,
        segment_indices={segment: indices for segment in PRIMARY_SEGMENTS},
    )


def _external_predictors(_root):
    return {
        (segment, "+".join(configuration), target): (1.0, 0.5)
        for segment in PRIMARY_SEGMENTS
        for configuration in deep_configuration_panel()
        for target in CANONICAL_LEADS
    }


def test_zero_transfer_resume_skips_models_for_completed_shards(tmp_path: Path) -> None:
    bundles = {
        method: FittedBenchmarkBundle(
            root=tmp_path / method,
            method=method,
            seeds=(0,),
            configurations=(("I",),),
            training_predictors_sha256="a" * 64,
            metadata={},
            summary={},
        )
        for method in EXPECTED_METHODS
    }
    keys = tuple(
        MetricShardKey.from_values(
            cohort="chapman",
            partition="test",
            method=method,
            model_seed=0,
            configuration=("I",),
        )
        for method in EXPECTED_METHODS
    )
    identity = {"fixture": "external-resume-v1"}
    first = MetricDatasetWriter(
        tmp_path / "output",
        expected_shards=keys,
        expected_coverage={
            key: metric_coverage_contract(
                [_external_record()],
                configuration=("I",),
                segments=PRIMARY_SEGMENTS,
            )
            for key in keys
        },
        dataset_identity=identity,
    )
    first_calls = []

    def failing_loader(_root, method, _seed, *, device):
        assert device == "cpu"
        first_calls.append(method)
        if method == "masked-unet":
            raise RuntimeError("simulated interruption")
        return _ExactCopyReconstructor()

    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluate_zero_transfer_bundles(
            bundles,
            [_external_record()],
            cohort="chapman",
            device="cpu",
            metric_writer=first,
            model_loader=failing_loader,
            predictor_loader=_external_predictors,
        )
    assert first_calls == ["lowrank", "ridge", "masked-unet"]

    resumed = MetricDatasetWriter(
        tmp_path / "output",
        expected_shards=keys,
        expected_coverage={
            key: metric_coverage_contract(
                [_external_record()],
                configuration=("I",),
                segments=PRIMARY_SEGMENTS,
            )
            for key in keys
        },
        dataset_identity=identity,
        resume=True,
    )
    resumed_calls = []

    def resumed_loader(_root, method, _seed, *, device):
        assert device == "cpu"
        resumed_calls.append(method)
        return _ExactCopyReconstructor()

    predictor_sha256 = evaluate_zero_transfer_bundles(
        bundles,
        [_external_record()],
        cohort="chapman",
        device="cpu",
        metric_writer=resumed,
        model_loader=resumed_loader,
        predictor_loader=_external_predictors,
    )
    assert len(predictor_sha256) == 64
    assert resumed_calls == ["masked-unet", "imputeecg", "ecgrecover"]
    resumed.finalize(summary={"mode": "zero-transfer", "cohort": "chapman"})


def test_formal_metric_producers_do_not_contain_dataframe_concat() -> None:
    producers = (
        benchmark.run,
        benchmark._fit_and_score_native_linear,
        benchmark._fit_and_score_masked_unet,
        benchmark._fit_and_score_imputeecg,
        benchmark._fit_and_score_ecgrecover,
        evaluate_zero_transfer_bundles,
    )
    for producer in producers:
        assert "pd.concat" not in inspect.getsource(producer)
