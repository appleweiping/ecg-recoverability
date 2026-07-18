import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert.estimators import TrainManifest
from ecgcert.estimators.api import sha256_file
from ecgcert.reconstruction import (
    EvaluationRecord,
    TrainingPredictorAccumulator,
    training_predictor_lookup,
)
from experiments.reconstruction_benchmark_v3 import _streaming_training_moments
from experiments.reconstruction_candidates_v3 import (
    evaluate_linear_candidate_grid,
    fit_masked_unet_candidate,
    validate_release_arguments,
)
from experiments.tune_reconstructors_v3 import (
    CANDIDATE_SCHEMA_VERSION,
    Candidate,
    TRACE_SCHEMA_VERSION,
    TUNING_SEEDS,
    _validate_early_stop_trace,
    candidate_grid,
    select_tuning_configuration,
)


MANIFEST_SHA = "a" * 64
SPLIT_SHA = "b" * 64
CASE_SHA = "c" * 64
PANEL_SHA = "d" * 64


def _official_config():
    return {
        "schema_version": "official-reconstruction-config-v3",
        "methods": {
            "imputeecg": {
                "selection": "official_fixed",
                "training_parameters": {"epochs": 100},
            },
            "ecgrecover": {
                "selection": "official_fixed",
                "training_parameters": {"published_task": "single-input"},
            },
        },
    }


def test_native_candidate_grid_fits_real_fold8_metrics_and_materializes_checkpoints(tmp_path):
    rng = np.random.default_rng(5)
    signals = rng.normal(size=(3, 12, 32)).astype(np.float32)
    signals_path = tmp_path / "train.npy"
    np.save(signals_path, signals)
    manifest = TrainManifest(
        dataset="PTB-XL",
        split="folds1-7/train",
        signals_path=str(signals_path),
        signals_sha256=sha256_file(signals_path),
        split_sha256=SPLIT_SHA,
        patient_ids_sha256="e" * 64,
        rate_hz=500,
        normalization="raw_mV",
    )
    configuration = ("I",)
    accumulator = TrainingPredictorAccumulator(("QRS",))
    for index, signal in enumerate(signals):
        accumulator.update(
            EvaluationRecord(
                patient_id=f"train-{index}",
                record_id=f"train-record-{index}",
                signal=signal,
                segment_indices={"QRS": np.arange(signal.shape[1])},
            )
        )
    predictors = training_predictor_lookup(accumulator.finalize((configuration,)))
    evaluation_records = [
        EvaluationRecord(
            patient_id=f"tune-{index}",
            record_id=f"tune-record-{index}",
            signal=rng.normal(size=(12, 32)),
            segment_indices={"QRS": np.arange(32)},
        )
        for index in range(2)
    ]
    mean, scatter, count = _streaming_training_moments(manifest)
    output_dir = tmp_path / "candidate-artifact"
    output_dir.mkdir()
    frames = evaluate_linear_candidate_grid(
        "lowrank",
        mean=mean,
        scatter=scatter,
        sample_count=count,
        evaluation_records=evaluation_records,
        configurations=(configuration,),
        segments=("QRS",),
        training_predictors=predictors,
        output_dir=output_dir,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
    )
    metrics = pd.concat(frames, ignore_index=True)
    assert set(metrics["candidate_id"]) == {
        value.candidate_id for value in candidate_grid()["lowrank"]
    }
    assert set(metrics["partition"]) == {"fold8/tune"}
    assert set(metrics["train_partition"]) == {"folds1-7/train"}
    assert set(metrics["epoch"]) == {0}
    assert np.allclose(metrics["log_rmse_mv"], np.log(metrics["rmse_mv"]))
    descriptors = metrics[["checkpoint_path", "checkpoint_sha256"]].drop_duplicates()
    assert len(descriptors) == len(candidate_grid()["lowrank"])
    for row in descriptors.itertuples(index=False):
        checkpoint = output_dir / row.checkpoint_path
        assert checkpoint.is_file()
        assert sha256_file(checkpoint) == row.checkpoint_sha256


def _candidate_row(method, candidate_id, *, seed, epoch, score):
    checkpoint = hashlib.sha256(f"{candidate_id}|{seed}".encode()).hexdigest()
    rmse = float(np.exp(score))
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "cohort": "PTB-XL",
        "train_partition": "folds1-7/train",
        "partition": "fold8/tune",
        "manifest_sha256": MANIFEST_SHA,
        "split_sha256": SPLIT_SHA,
        "method": method,
        "candidate_id": candidate_id,
        "patient_id": "patient-1",
        "segment": "QRS",
        "configuration": "I",
        "target": "V2",
        "model_seed": seed,
        "epoch": epoch,
        "rmse_mv": rmse,
        "log_rmse_mv": float(np.log(rmse)),
        "checkpoint_path": f"checkpoints/{candidate_id}-seed-{seed}.ckpt",
        "checkpoint_sha256": checkpoint,
        "observed_integrity": True,
    }


def _singleton_metrics_and_trace():
    rows = []
    trace_rows = []
    grid = candidate_grid()
    for method in ("lowrank", "ridge"):
        for index, candidate in enumerate(grid[method]):
            rows.append(
                _candidate_row(
                    method,
                    candidate.candidate_id,
                    seed=0,
                    epoch=0,
                    score=-2.0 + index / 100,
                )
            )
    for candidate_index, candidate in enumerate(grid["masked-unet"]):
        for seed in TUNING_SEEDS:
            checkpoint = hashlib.sha256(
                f"{candidate.candidate_id}|{seed}".encode()
            ).hexdigest()
            # Epoch 2 is the strict best, followed by exactly patience=8 stale
            # epochs; the first legal stop is therefore epoch 10.
            scores = [-1.0, -1.5] + [-1.4 + index / 100 for index in range(8)]
            best = float("inf")
            stale = 0
            for epoch, score in enumerate(scores, start=1):
                improved = score < best
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
                trace_rows.append(
                    {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "cohort": "PTB-XL",
                        "train_partition": "folds1-7/train",
                        "partition": "fold8/tune",
                        "manifest_sha256": MANIFEST_SHA,
                        "split_sha256": SPLIT_SHA,
                        "fold8_records_sha256": CASE_SHA,
                        "configuration_panel_sha256": PANEL_SHA,
                        "candidate_id": candidate.candidate_id,
                        "model_seed": seed,
                        "epoch": epoch,
                        "monitor_log_rmse_mv": score,
                        "best_so_far_log_rmse_mv": best,
                        "stale_epochs": stale,
                        "is_strict_improvement": improved,
                        "best_epoch": 2,
                        "stopped_epoch": 10,
                        "checkpoint_path": (
                            f"checkpoints/{candidate.candidate_id}-seed-{seed}.ckpt"
                        ),
                        "checkpoint_sha256": checkpoint,
                    }
                )
            rows.append(
                _candidate_row(
                    "masked-unet",
                    candidate.candidate_id,
                    seed=seed,
                    epoch=2,
                    score=-1.7 + candidate_index / 10,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def test_singleton_unet_metrics_require_and_verify_real_patience_trace():
    metrics, trace = _singleton_metrics_and_trace()
    with pytest.raises(ValueError, match="verifiable early-stop trace"):
        select_tuning_configuration(
            metrics,
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            official_config=_official_config(),
        )
    selected = select_tuning_configuration(
        metrics,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
        official_config=_official_config(),
        early_stop_trace=trace,
        fold8_records_sha256=CASE_SHA,
        configuration_panel_sha256=PANEL_SHA,
    )
    assert selected["methods"]["masked-unet"]["epochs"] == 2
    assert selected["selection_audit"]["masked-unet"]["early_stop_trace"][
        "fold8_records_sha256"
    ] == CASE_SHA

    tampered = trace.copy()
    tampered.loc[tampered.index[-1], "stale_epochs"] = 7
    with pytest.raises(ValueError, match="stale counts"):
        _validate_early_stop_trace(
            tampered,
            metrics,
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            fold8_records_sha256=CASE_SHA,
            configuration_panel_sha256=PANEL_SHA,
        )

    first_candidate = candidate_grid()["masked-unet"][0].candidate_id
    first_seed = (
        (trace["candidate_id"] == first_candidate) & (trace["model_seed"] == 0)
    )
    corruptions = []
    missing_epoch = trace.drop(trace[first_seed & (trace["epoch"] == 3)].index)
    corruptions.append(missing_epoch)
    wrong_best = trace.copy()
    wrong_best.loc[first_seed, "best_epoch"] = 1
    corruptions.append(wrong_best)
    wrong_stop = trace.copy()
    wrong_stop.loc[first_seed, "stopped_epoch"] = 9
    corruptions.append(wrong_stop)
    wrong_checkpoint = trace.copy()
    wrong_checkpoint.loc[first_seed, "checkpoint_sha256"] = "f" * 64
    corruptions.append(wrong_checkpoint)
    wrong_case = trace.copy()
    wrong_case.loc[wrong_case.index[0], "fold8_records_sha256"] = "0" * 64
    corruptions.append(wrong_case)
    for corrupted in corruptions:
        with pytest.raises(ValueError):
            _validate_early_stop_trace(
                corrupted,
                metrics,
                manifest_sha256=MANIFEST_SHA,
                split_sha256=SPLIT_SHA,
                fold8_records_sha256=CASE_SHA,
                configuration_panel_sha256=PANEL_SHA,
            )


def test_masked_unet_candidate_restores_best_epoch_before_full_fold8_scoring(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    signals = rng.normal(size=(1, 12, 12)).astype(np.float32)
    signals_path = tmp_path / "train-unet.npy"
    np.save(signals_path, signals)
    train_manifest = TrainManifest(
        dataset="PTB-XL",
        split="folds1-7/train",
        signals_path=str(signals_path),
        signals_sha256=sha256_file(signals_path),
        split_sha256=SPLIT_SHA,
        patient_ids_sha256="e" * 64,
        rate_hz=500,
        normalization="raw_mV",
    )
    configuration = ("I",)
    train_record = EvaluationRecord(
        patient_id="train",
        record_id="train",
        signal=signals[0],
        segment_indices={"QRS": np.arange(12)},
    )
    accumulator = TrainingPredictorAccumulator(("QRS",))
    accumulator.update(train_record)
    predictors = training_predictor_lookup(accumulator.finalize((configuration,)))
    tune_record = EvaluationRecord(
        patient_id="tune",
        record_id="tune",
        signal=rng.normal(size=(12, 12)),
        segment_indices={"QRS": np.arange(12)},
    )
    candidate = Candidate(
        method="masked-unet",
        candidate_id="masked-unet-test",
        parameters={
            "width": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 1,
            "max_epochs": 10,
            "early_stopping_patience": 2,
            "num_workers": 0,
            "deterministic": True,
        },
    )
    monkeypatch.setattr(
        "experiments.reconstruction_candidates_v3._build_unet",
        lambda width: torch.nn.Conv1d(24, 12, 1),
    )
    scores = iter((-1.0, -2.0, -1.5, -1.4))
    monkeypatch.setattr(
        "experiments.reconstruction_candidates_v3._early_stop_fold8_score",
        lambda *args, **kwargs: next(scores),
    )
    output_dir = tmp_path / "unet-candidate"
    output_dir.mkdir()
    frames, audit = fit_masked_unet_candidate(
        candidate,
        seed=0,
        train_manifest=train_manifest,
        evaluation_records=(tune_record,),
        configurations=(configuration,),
        segments=("QRS",),
        training_predictors=predictors,
        output_dir=output_dir,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
        device_name="cpu",
        normalization_scale=np.ones(12, dtype=np.float32),
    )
    assert audit["best_epoch"] == 2
    assert audit["stopped_epoch"] == 4
    metrics = pd.concat(frames, ignore_index=True)
    assert set(metrics["epoch"]) == {2}
    checkpoint = output_dir / audit["checkpoint_path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["best_epoch"] == 2
    assert payload["stopped_epoch"] == 4
    assert sha256_file(checkpoint) == audit["checkpoint_sha256"]


def test_candidate_release_mode_forbids_subsampling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser_values = {
        "output_dir": Path("artifacts/candidates"),
        "max_records": 1,
        "max_configurations": None,
        "rate": 500,
        "segments": ("QRS", "ST", "T"),
        "delineator": "dwt",
        "release": True,
    }
    arguments = type("Arguments", (), parser_values)()
    with pytest.raises(ValueError, match="max-records is forbidden"):
        validate_release_arguments(arguments)
