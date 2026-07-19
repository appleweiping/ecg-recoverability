import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.data.audit import SignalAudit
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.data.ptbxl import PTBXL
from ecgcert.estimators import (
    MaskedUNetReconstructor,
    ReconstructorConfig,
    RidgeLeadReconstructor,
    TrainManifest,
)
from ecgcert.estimators.api import sha256_file
from ecgcert.protocol import PatientSplit
from ecgcert.reconstruction import (
    EvaluationRecord,
    ModelBundleError,
    ObservedSampleViolation,
    OfficialCommandBridgeReconstructor,
    TrainingPredictorAccumulator,
    checkpoint_descriptor,
    evaluate_reconstructor,
    load_fitted_reconstructor,
    load_training_predictors,
    write_benchmark_artifacts,
    write_bundle_metadata,
)
from experiments.reconstruction_benchmark_v3 import (
    PTBXLManifestV3,
    _load_evaluation_records,
    _resolve_official_source,
    _score_partitions,
    _streaming_training_moments,
    build_parser,
    load_ptbxl_manifest,
    resolve_model_seeds,
    validate_release_arguments,
)


class _ZeroMissing:
    def reconstruct(self, signal, observed_mask):
        return np.where(observed_mask, signal, 0.0)


class _ChangesObserved:
    def reconstruct(self, signal, observed_mask):
        prediction = np.where(observed_mask, signal, 0.0)
        prediction[observed_mask] += 1e-6
        return prediction


class _MutatesObservedInputInPlace:
    def reconstruct(self, signal, observed_mask):
        signal[observed_mask] += 1.0
        return signal


def _evaluation_records():
    first = np.arange(1, 12 * 6 + 1, dtype=float).reshape(12, 6) / 10
    second = first + 0.2
    segments = {
        "QRS": np.asarray([0, 1], dtype=np.int64),
        "ST": np.asarray([2, 3], dtype=np.int64),
        "T": np.asarray([4, 5], dtype=np.int64),
    }
    return [
        EvaluationRecord("patient-a", "record-1", first, segments),
        EvaluationRecord("patient-a", "record-2", second, segments),
    ]


def _training_predictors(configurations=(("I", "II"),)):
    accumulator = TrainingPredictorAccumulator(("QRS", "ST", "T"))
    for record in _evaluation_records():
        accumulator.update(record)
    return accumulator.finalize(configurations)


def test_shared_patient_scorer_emits_only_missing_targets_and_log_rmse():
    frame = evaluate_reconstructor(
        _ZeroMissing(),
        _evaluation_records(),
        configuration=("I", "II"),
        method="ridge",
        model_seed=0,
        segments=("QRS", "ST", "T"),
        training_predictors=_training_predictors(),
    )
    assert len(frame) == 3 * 10
    assert set(frame["cohort"]) == {"PTB-XL"}
    assert set(frame["partition"]) == {"test"}
    assert set(frame["patient_id"]) == {"patient-a"}
    assert set(frame["configuration"]) == {"I+II"}
    assert set(frame["model_seed"]) == {0}
    assert not {"I", "II"} & set(frame["target"])
    assert set(frame["n_records"]) == {2}
    assert np.allclose(frame["log_rmse_mv"], np.log(frame["rmse_mv"]))
    assert np.array_equal(frame["outcome_log_rmse"], frame["log_rmse_mv"])


def test_observed_samples_are_checked_independently_and_fail_closed():
    with pytest.raises(ObservedSampleViolation, match="changed observed"):
        evaluate_reconstructor(
            _ChangesObserved(),
            _evaluation_records(),
            configuration=("I", "V1"),
            method="broken",
            model_seed=0,
            segments=("QRS",),
            training_predictors=_training_predictors((("I", "V1"),)),
        )

    with pytest.raises(ObservedSampleViolation, match="changed observed"):
        evaluate_reconstructor(
            _MutatesObservedInputInPlace(),
            _evaluation_records(),
            configuration=("I", "V1"),
            method="in-place-broken",
            model_seed=0,
            segments=("QRS",),
            training_predictors=_training_predictors((("I", "V1"),)),
        )


def test_official_bridge_receives_no_missing_truth(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.ckpt"
    checkpoint.write_bytes(b"official-checkpoint")
    bridge = source / "bridge.py"
    bridge.write_text(
        "from pathlib import Path\n"
        "import numpy as np, sys\n"
        "x=np.load(sys.argv[1], allow_pickle=False)\n"
        "s=x['observed_signal']; m=x['observed_mask']\n"
        "assert s.ndim == 3 and m.shape == s.shape\n"
        "assert np.all(s[~m] == 0)\n"
        "count=Path(sys.argv[3])\n"
        "count.write_text(str(int(count.read_text()) + 1))\n"
        "np.savez(sys.argv[2], reconstruction=s)\n",
        encoding="utf-8",
    )
    count_path = tmp_path / "bridge-calls.txt"
    count_path.write_text("0", encoding="utf-8")
    model = OfficialCommandBridgeReconstructor(
        command=[
            sys.executable,
            "bridge.py",
            "{input}",
            "{output}",
            str(count_path),
        ],
        checkpoint=checkpoint,
        source_dir=source,
        single_input_only=True,
        records_per_process=2,
    )
    template = _evaluation_records()[0]
    records = [
        EvaluationRecord(
            patient_id=f"batch-patient-{index}",
            record_id=f"batch-record-{index}",
            signal=template.signal + index * 0.01,
            segment_indices=template.segment_indices,
        )
        for index in range(5)
    ]
    frame = evaluate_reconstructor(
        model,
        records,
        configuration=("II",),
        method="ecgrecover",
        model_seed=0,
        segments=("QRS",),
        training_predictors=_training_predictors((("II",),)),
    )
    assert not frame.empty
    assert set(frame["n_observed"]) == {1}
    assert count_path.read_text(encoding="utf-8") == "3"


def test_training_only_simple_predictors_are_fixed_across_fold8_9_10_and_patients():
    predictors = _training_predictors()
    base_records = _evaluation_records()
    shifted_records = [
        EvaluationRecord(
            patient_id="heldout-shifted",
            record_id="shifted",
            signal=base_records[0].signal * 20.0,
            segment_indices=base_records[0].segment_indices,
        )
    ]
    frames = _score_partitions(
        _ZeroMissing(),
        {
            "tune": base_records,
            "calibration": shifted_records,
            "test": [base_records[0]],
        },
        configuration=("I", "II"),
        method="lowrank",
        model_seed=0,
        segments=("QRS", "ST", "T"),
        training_predictors={
            (str(row.segment), str(row.configuration), str(row.target)): (
                float(row.target_rms),
                float(row.max_target_observed_correlation),
            )
            for row in predictors.itertuples(index=False)
        },
    )
    combined = pd.concat(frames, ignore_index=True)
    assert set(combined["partition"]) == {"tune", "calibration", "test"}
    grouped = combined.groupby(["segment", "configuration", "target"])
    assert grouped["target_rms"].nunique().max() == 1
    assert grouped["max_target_observed_correlation"].nunique().max() == 1
    assert grouped["target_rms_mv"].nunique().max() > 1


def _release_arguments(tmp_path, *extra):
    tuning = tmp_path / "tuning.json"
    tuning.write_text("{}", encoding="utf-8")
    output = Path.cwd() / "artifacts" / "test-reconstruction"
    return build_parser().parse_args(
        [
            "--method",
            "masked-unet",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--rank-maps",
            str(tmp_path / "maps"),
            "--output-dir",
            str(output),
            "--tuning-config",
            str(tuning),
            "--training-inclusion",
            str(tmp_path / "training-inclusion.json"),
            "--release",
            *extra,
        ]
    )


def test_release_forbids_subsampling_and_freezes_neural_seeds(tmp_path):
    arguments = _release_arguments(tmp_path)
    validate_release_arguments(arguments)
    assert resolve_model_seeds(arguments) == (0, 1, 2, 3, 4)

    with pytest.raises(ValueError, match="max-records is forbidden"):
        validate_release_arguments(_release_arguments(tmp_path, "--max-records", "2"))
    with pytest.raises(ValueError, match="exactly model seeds"):
        validate_release_arguments(_release_arguments(tmp_path, "--seeds", "0,1"))


def _ptbxl_manifest(tmp_path: Path):
    records = []
    split = {"train": [1], "tune": [8], "calibration": [9], "test": [10]}
    for record_id, fold in ((1, 1), (8, 8), (9, 9), (10, 10)):
        records.append(
            {
                "record_id": str(record_id),
                "patient_id": f"patient-{record_id}",
                "strat_fold": fold,
                "files": {},
            }
        )
    split_sha256 = PatientSplit(
        train=(1,), tune=(8,), calibration=(9,), test=(10,)
    ).sha256()
    payload = {
        "schema_version": "ptbxl-manifest-v3",
        "cohort": "PTB-XL",
        "root": str(tmp_path),
        "records": records,
        "split": split,
        "split_sha256": split_sha256,
    }
    payload["manifest_sha256"] = lineage.canonical_sha256(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_ptbxl_manifest_contract_verifies_hash_folds_and_patient_roles(tmp_path):
    path, payload = _ptbxl_manifest(tmp_path)
    loaded = load_ptbxl_manifest(path)
    assert loaded.record_ids("train") == ("1",)
    assert loaded.split_sha256 == payload["split_sha256"]

    payload["records"][0]["patient_id"] = payload["records"][-1]["patient_id"]
    payload["manifest_sha256"] = lineage.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="patient leakage"):
        load_ptbxl_manifest(path)


def test_ptbxl_release_loader_rejects_self_consistent_partial_manifest(tmp_path):
    path, payload = _ptbxl_manifest(tmp_path)
    payload.update(
        {
            "version": "1.0.3",
            "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
            "population": "all_records_no_diagnosis_filter",
            "split_algorithm": "official-strat-folds-1-7_8_9_10-v1",
            "structure": {
                "n_records": 4,
                "n_patients": 4,
                "folds": {
                    str(fold): {
                        "n_records": int(fold in {1, 8, 9, 10}),
                        "n_patients": int(fold in {1, 8, 9, 10}),
                    }
                    for fold in range(1, 11)
                },
                "split": {
                    role: {"n_records": 1, "n_patients": 1}
                    for role in ("train", "tune", "calibration", "test")
                },
            },
        }
    )
    payload["manifest_sha256"] = lineage.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="official v1.0.3 contract"):
        load_ptbxl_manifest(path, release=True)


def test_primary_evaluation_requests_strict_delineation(monkeypatch, tmp_path):
    signal = np.zeros((100, 12), dtype=np.float32)
    audit = SignalAudit(
        cohort="PTB-XL",
        record_id="1",
        patient_id="patient-1",
        status="included",
        reason=None,
        requested_rate_hz=500,
        source_rate_hz=500,
        n_samples=100,
        input_leads=CANONICAL_LEADS,
        input_units=("mV",) * 12,
        canonical_leads=CANONICAL_LEADS,
        source_channel_indices=tuple(range(12)),
        unit_scales_to_mv=(1.0,) * 12,
        output_unit="mV",
    )

    class FixtureDB:
        def signal_with_audit(self, _record_id, *, rate):
            assert rate == 500
            return signal, audit

    observed = {}

    def delineate(_signal, *, fs, method, strict):
        observed.update(fs=fs, method=method, strict=strict)
        return {
            "QRS": np.asarray([1, 2]),
            "ST": np.asarray([3, 4]),
            "T": np.asarray([5, 6]),
        }

    monkeypatch.setattr(PTBXL, "segment_indices", staticmethod(delineate))
    contract = PTBXLManifestV3(
        path=tmp_path / "manifest.json",
        root=tmp_path,
        records={"1": {"patient_id": "patient-1"}},
        split={"test": ("1",)},
        manifest_sha256="0" * 64,
        split_sha256="1" * 64,
    )
    records, evaluation_audit = _load_evaluation_records(
        FixtureDB(),
        contract,
        ("1",),
        rate=500,
        segments=("QRS", "ST", "T"),
        delineator="dwt",
        partition="fold10/test",
    )

    assert len(records) == 1
    assert observed == {"fs": 500, "method": "dwt", "strict": True}
    assert evaluation_audit["records"][0]["source_channel_indices"] == tuple(range(12))
    assert evaluation_audit["records"][0]["output_unit"] == "mV"


def test_official_method_missing_source_or_config_is_an_explicit_failure(tmp_path):
    arguments = build_parser().parse_args(
        [
            "--method",
            "imputeecg",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--rank-maps",
            str(tmp_path / "maps"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="no surrogate"):
        _resolve_official_source(arguments)


def _tiny_train_manifest(tmp_path):
    rng = np.random.default_rng(9)
    signal = rng.normal(size=(5, 12, 16)).astype(np.float32)
    path = tmp_path / "train.npy"
    np.save(path, signal)
    digest = hashlib.sha256(b"split").hexdigest()
    return signal, TrainManifest(
        dataset="fixture",
        split="train",
        signals_path=str(path),
        signals_sha256=sha256_file(path),
        split_sha256=digest,
        patient_ids_sha256=digest,
        rate_hz=500,
    )


def test_streaming_linear_statistics_use_every_training_sample_once(tmp_path):
    signal, train_manifest = _tiny_train_manifest(tmp_path)
    mean, scatter, count = _streaming_training_moments(train_manifest)
    samples = np.transpose(signal, (0, 2, 1)).reshape(-1, 12).astype(float)
    centered = samples - samples.mean(axis=0)
    assert count == samples.shape[0]
    assert np.allclose(mean, samples.mean(axis=0), atol=1e-12)
    assert np.allclose(scatter, centered.T @ centered, atol=1e-10)


def test_linear_bundle_is_hash_checked_and_reloadable(tmp_path):
    signal, train_manifest = _tiny_train_manifest(tmp_path)
    bundle = tmp_path / "bundle"
    model_dir = bundle / "models" / "seed-0" / "config-000"
    RidgeLeadReconstructor().fit(
        train_manifest,
        ReconstructorConfig(
            observed_leads=("I", "II"),
            seed=0,
            output_dir=str(model_dir),
            parameters={"ridge_lambda": 1e-3},
        ),
    )
    checkpoint = model_dir / "ridge.npz"
    descriptor = checkpoint_descriptor(
        checkpoint, bundle, seed=0, configuration=["I", "II"]
    )
    write_bundle_metadata(
        bundle,
        {
            "method": "ridge",
            "adapter_class": "ecgcert.estimators.RidgeLeadReconstructor",
            "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
            "models": [descriptor],
            "training_config": {},
            "tuning_config": {"ridge_lambda": 1e-3},
        },
    )
    loaded = load_fitted_reconstructor(bundle, "ridge", 0)
    mask = np.zeros(12, dtype=bool)
    mask[:2] = True
    reconstructed = loaded.reconstruct(signal[0], mask)
    assert np.array_equal(reconstructed[:2], signal[0, :2])

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ModelBundleError, match="SHA-256 mismatch"):
        load_fitted_reconstructor(bundle, "ridge", 0)


def test_masked_unet_bundle_reloads_for_external_inference(tmp_path):
    pytest.importorskip("torch")
    signal, train_manifest = _tiny_train_manifest(tmp_path)
    bundle = tmp_path / "unet-bundle"
    model_dir = bundle / "models" / "seed-3"
    MaskedUNetReconstructor().fit(
        train_manifest,
        ReconstructorConfig(
            observed_leads=("I",),
            seed=3,
            output_dir=str(model_dir),
            device="cpu",
            parameters={
                "epochs": 1,
                "batch_size": 2,
                "max_records": 5,
                "normalization_records": 5,
                "width": 8,
                "num_workers": 0,
            },
        ),
    )
    descriptor = checkpoint_descriptor(model_dir / "masked_unet.pt", bundle, seed=3)
    write_bundle_metadata(
        bundle,
        {
            "method": "masked-unet",
            "adapter_class": "ecgcert.estimators.MaskedUNetReconstructor",
            "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
            "models": [descriptor],
            "training_config": {},
            "tuning_config": {},
        },
    )
    loaded = load_fitted_reconstructor(bundle, "masked-unet", 3, device="cpu")
    mask = np.zeros(12, dtype=bool)
    mask[0] = True
    reconstructed = loaded.reconstruct(signal[0], mask)
    assert np.array_equal(reconstructed[0], signal[0, 0])


def test_ecgrecover_bundle_freezes_the_official_single_input_contract(tmp_path, monkeypatch):
    source = tmp_path / "official-source"
    source.mkdir()
    bundle = tmp_path / "ecgrecover-bundle"
    checkpoint = bundle / "models" / "seed-0" / "official.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"pinned-official-model")
    bridge = [sys.executable, "bridge.py", "{input}", "{output}", "{checkpoint}"]
    descriptor = checkpoint_descriptor(
        checkpoint,
        bundle,
        seed=0,
        configuration=["II"],
        inference_bridge=bridge,
    )
    write_bundle_metadata(
        bundle,
        {
            "method": "ecgrecover",
            "adapter_class": "ecgcert.reconstruction.OfficialCommandBridgeReconstructor",
            "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
            "models": [descriptor],
            "training_config": {},
            "tuning_config": {},
            "official": {
                "source_dir": str(source),
                "input_lead": "II",
                "inference_records_per_process": 128,
                "inference_bridge": bridge,
            },
        },
    )
    monkeypatch.setattr(
        "ecgcert.estimators.official.validate_pinned_checkout",
        lambda *_args, **_kwargs: "pinned",
    )
    loaded = load_fitted_reconstructor(bundle, "ecgrecover", 0)
    signal = _evaluation_records()[0].signal
    two_lead_mask = np.zeros(12, dtype=bool)
    two_lead_mask[:2] = True
    with pytest.raises(ValueError, match="single-input"):
        loaded.reconstruct(signal, two_lead_mask)

    metadata_path = bundle / "bundle.v3.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["official"]["input_lead"] = "I"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelBundleError, match="invalid official single-input"):
        load_fitted_reconstructor(bundle, "ecgrecover", 0)


def test_summary_and_parquet_contract_include_reload_metadata(tmp_path):
    frame = evaluate_reconstructor(
        _ZeroMissing(),
        [_evaluation_records()[0]],
        configuration=("I",),
        method="lowrank",
        model_seed=0,
        segments=("QRS",),
        training_predictors=_training_predictors((("I",),)),
    )

    value = write_benchmark_artifacts(
        frame,
        tmp_path,
        summary={
            "method": "lowrank",
            "adapter_class": "fixture.Adapter",
            "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
            "checkpoints": [],
            "training_config": {"rate_hz": 500},
            "tuning_config": {"rank": 3},
        },
    )
    persisted = json.loads((tmp_path / "summary.v3.json").read_text(encoding="utf-8"))
    assert value["status"] == persisted["status"] == "complete"
    assert persisted["method"] == "lowrank"
    assert persisted["artifacts"]["patient_metrics"]["path"] == "patient_metrics.parquet"
    assert len(persisted["artifacts"]["patient_metrics"]["sha256"]) == 64
    inventory = json.loads(
        (tmp_path / "patient_metrics.inventory.v1.json").read_text(encoding="utf-8")
    )
    assert inventory["status"] == "complete"
    assert inventory["total_rows"] == len(frame)
    assert inventory["shards"][0]["row_group_start"] == 0
    assert inventory["shards"][0]["parquet_sha256"]
    assert not inventory["shards"][0]["staging_retained"]


def test_training_predictor_bundle_artifact_is_hash_checked(tmp_path, monkeypatch):
    predictors = _training_predictors()
    predictor_path = tmp_path / "training_predictors.parquet"
    predictor_path.write_bytes(b"PAR1-training-predictors")
    write_bundle_metadata(
        tmp_path,
        {
            "method": "ridge",
            "models": [],
            "training_config": {},
            "tuning_config": {},
            "training_predictors": {
                "path": predictor_path.name,
                "sha256": lineage.artifact_sha256(predictor_path),
            },
        },
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: predictors)
    loaded = load_training_predictors(tmp_path)
    assert len(loaded) == len(predictors)
    predictor_path.write_bytes(b"tampered")
    with pytest.raises(ModelBundleError, match="predictor SHA-256 mismatch"):
        load_training_predictors(tmp_path)
