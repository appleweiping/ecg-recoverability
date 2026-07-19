import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ecgcert import lineage
from ecgcert.data.audit import SignalAudit
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.api import sha256_file
from ecgcert.execution import ExperimentManifest
from ecgcert.execution.runner import declared_path_hashes
from ecgcert.official_baselines import OfficialTrainingRecord, materialize_official_arrays
from ecgcert.training_inclusion import (
    AUDIT_FILENAME,
    MANIFEST_FILENAME,
    build_training_inclusion,
    load_training_inclusion,
)
from experiments.reconstruction_benchmark_v3 import _validate_official_training_cohort
from scripts import prepare_training_inclusion_v3 as preparation


SOURCE_SHA = "a" * 64
SPLIT_SHA = "b" * 64
PANEL_SHA = "c" * 64
SEGMENTS = ("QRS", "ST", "T")
CONFIGURATIONS = (("I",), ("II",))


def _signal(record_id: str) -> np.ndarray:
    time = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)
    return np.stack(
        [
            (lead + 1) * time
            + np.float32(int(record_id) / 1000.0)
            + np.float32(0.01) * np.sin((lead + 1) * time)
            for lead in range(12)
        ],
        axis=1,
    ).astype(np.float32)


class _FakeDB:
    def __init__(self, records, windows=None):
        self.records = records
        self.signals = {record_id: _signal(record_id) for record_id in records}
        self.windows = {
            record_id: {
                "QRS": np.arange(100, 140, dtype=np.int64),
                "ST": np.arange(200, 240, dtype=np.int64),
                "T": np.arange(300, 340, dtype=np.int64),
            }
            for record_id in records
        }
        if windows:
            for record_id, values in windows.items():
                self.windows[record_id].update(values)
        self.signal_delta = {}
        self.audit_rate = {}
        self._active = None

    def signal_with_audit(self, ecg_id, rate):
        record_id = str(ecg_id)
        self._active = record_id
        patient_id = str(self.records[record_id]["patient_id"])
        signal = self.signals[record_id].copy()
        signal += np.float32(self.signal_delta.get(record_id, 0.0))
        audit = SignalAudit(
            cohort="PTB-XL",
            record_id=record_id,
            patient_id=patient_id,
            status="included",
            reason=None,
            requested_rate_hz=rate,
            source_rate_hz=float(self.audit_rate.get(record_id, rate)),
            n_samples=signal.shape[0],
            input_leads=CANONICAL_LEADS,
            input_units=("mV",) * 12,
            canonical_leads=CANONICAL_LEADS,
            source_channel_indices=tuple(range(12)),
            unit_scales_to_mv=(1.0,) * 12,
            output_unit="mV",
        )
        return signal, audit

    def segment_indices(self, signal, *, fs, method, strict):
        assert signal.shape == (5000, 12)
        assert fs == 500 and method == "dwt" and strict is True
        return self.windows[self._active]


def _records(*, folds=(1, 2, 7)):
    patient_ids = ("patient-z", "patient-a", "patient-z")
    return {
        str(11 + index): {
            "record_id": str(11 + index),
            "patient_id": patient_ids[index],
            "strat_fold": fold,
        }
        for index, fold in enumerate(folds)
    }


def _build_and_load(tmp_path: Path, *, records=None, windows=None):
    records = _records() if records is None else records
    record_ids = tuple(records)
    source_manifest = tmp_path / "ptbxl.json"
    source_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    db = _FakeDB(records, windows)
    output = tmp_path / "training-inclusion"
    build_training_inclusion(
        db=db,
        records=records,
        record_ids=record_ids,
        source_manifest_file_sha256=sha256_file(source_manifest),
        source_manifest_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA,
        rate_hz=500,
        segments=SEGMENTS,
        delineator="dwt",
        configurations=CONFIGURATIONS,
        configuration_panel_sha256=PANEL_SHA,
        output_dir=output,
    )
    inclusion = load_training_inclusion(
        output / MANIFEST_FILENAME,
        source_manifest_path=source_manifest,
        source_manifest_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA,
        expected_record_ids=record_ids,
        expected_records=records,
        rate_hz=500,
        segments=SEGMENTS,
        delineator="dwt",
        configuration_panel_sha256=PANEL_SHA,
    )
    return inclusion, db, records, source_manifest


@pytest.mark.parametrize("missing_segment", SEGMENTS)
def test_strict_qrs_st_t_failure_is_one_shared_exclusion_decision(
    tmp_path, missing_segment
):
    records = _records(folds=(1, 2, 3))
    windows = {
        "12": {missing_segment: np.asarray([], dtype=np.int64)},
    }
    inclusion, _db, _records_value, _manifest = _build_and_load(
        tmp_path, records=records, windows=windows
    )
    assert inclusion.included_record_ids == ("11", "13")
    audit = inclusion.audit
    by_id = {value["record_id"]: value for value in audit["records"]}
    assert by_id["11"]["status"] == "included"
    assert by_id["12"]["status"] == "excluded"
    assert missing_segment in by_id["12"]["reason"]
    assert by_id["13"]["status"] == "included"
    assert audit["summary"]["n_included"] == 2
    assert audit["summary"]["n_excluded"] == 1


def test_five_benchmarks_and_official_arrays_share_ordered_record_patient_identity(
    tmp_path,
):
    inclusion, db, records, _manifest = _build_and_load(tmp_path)
    expected_records = ("11", "12", "13")
    expected_patients = ("patient-z", "patient-a", "patient-z")
    assert inclusion.record_ids_sha256 == lineage.canonical_sha256(expected_records)
    assert inclusion.patient_ids_sha256 == lineage.canonical_sha256(expected_patients)
    # A set hash would lose both order and the repeated patient.
    assert inclusion.patient_ids_sha256 != lineage.canonical_sha256(
        sorted(set(expected_patients))
    )

    manifests = {}
    for method in ("lowrank", "ridge", "masked-unet", "imputeecg", "ecgrecover"):
        manifest = inclusion.materialize_signals(
            db, records, tmp_path / f"{method}.train.npy"
        )
        manifests[method] = manifest
        assert manifest.record_ids_sha256 == inclusion.record_ids_sha256
        assert manifest.patient_ids_sha256 == inclusion.patient_ids_sha256
        assert manifest.training_inclusion_sha256 == inclusion.inclusion_sha256
        materialized = np.load(manifest.signals_path, mmap_mode="r")
        assert [float(value) for value in materialized[:, 0, 0]] == pytest.approx(
            [float(db.signals[record_id][0, 0]) for record_id in expected_records]
        )

    official = materialize_official_arrays(
        (
            OfficialTrainingRecord(record_id, patient_id, signal, dict(audit))
            for record_id, patient_id, signal, audit in inclusion.iter_validated_signals(
                db, records
            )
        ),
        output_dir=tmp_path / "official-arrays",
        ecgrecover_input_lead="I",
        n_records=len(expected_records),
    )
    assert official["record_order"]["record_ids_sha256"] == inclusion.record_ids_sha256
    assert official["record_order"]["patient_ids_sha256"] == inclusion.patient_ids_sha256
    config = {
        "split_sha256": inclusion.split_sha256,
        "training_inclusion_sha256": inclusion.inclusion_sha256,
        "training_record_ids_sha256": inclusion.record_ids_sha256,
        "training_patient_ids_sha256": inclusion.patient_ids_sha256,
        "training_records_path": official["record_order"]["path"],
        "array_sha256": {
            "training_records.v3.json": official["record_order"]["sha256"]
        },
    }
    for method in ("ImputeECG", "ECGrecover"):
        assert _validate_official_training_cohort(
            manifests[method.lower()], config, method=method, release=True
        ) == len(expected_records)
    wrong_patient_order = dict(config)
    wrong_patient_order["training_patient_ids_sha256"] = lineage.canonical_sha256(
        sorted(set(expected_patients))
    )
    with pytest.raises(ValueError, match="shared training cohort"):
        _validate_official_training_cohort(
            manifests["imputeecg"],
            wrong_patient_order,
            method="ImputeECG",
            release=True,
        )


def test_included_signal_or_source_audit_drift_fails_closed_without_partial_array(
    tmp_path,
):
    inclusion, db, records, _manifest = _build_and_load(tmp_path)
    db.signal_delta["12"] = 0.25
    signal_destination = tmp_path / "drifted-signal.npy"
    with pytest.raises(ValueError, match="signal content changed"):
        inclusion.materialize_signals(db, records, signal_destination)
    assert not signal_destination.exists()

    db.signal_delta.clear()
    db.audit_rate["12"] = 250
    audit_destination = tmp_path / "drifted-audit.npy"
    with pytest.raises(ValueError, match="source audit changed"):
        inclusion.materialize_signals(db, records, audit_destination)
    assert not audit_destination.exists()


@pytest.mark.parametrize("target", ("manifest", "audit", "predictor"))
def test_inclusion_artifact_manifest_and_predictor_tampering_fail_closed(
    tmp_path, target
):
    inclusion, _db, records, source_manifest = _build_and_load(tmp_path)
    if target == "manifest":
        payload = json.loads(inclusion.path.read_text(encoding="utf-8"))
        payload["status"] = "tampered"
        inclusion.path.write_text(json.dumps(payload), encoding="utf-8")
    elif target == "audit":
        with (inclusion.root / AUDIT_FILENAME).open("ab") as handle:
            handle.write(b"tamper")
    else:
        with inclusion.predictors_path.open("ab") as handle:
            handle.write(b"tamper")
    with pytest.raises((ValueError, OSError)):
        load_training_inclusion(
            inclusion.path,
            source_manifest_path=source_manifest,
            source_manifest_sha256=SOURCE_SHA,
            split_sha256=SPLIT_SHA,
            expected_record_ids=tuple(records),
            expected_records=records,
            rate_hz=500,
            segments=SEGMENTS,
            delineator="dwt",
            configuration_panel_sha256=PANEL_SHA,
        )


@pytest.mark.parametrize("forbidden_fold", (8, 9, 10))
def test_fold8_9_10_cannot_enter_training_inclusion(tmp_path, forbidden_fold):
    records = _records(folds=(1, 2, forbidden_fold))
    source_manifest = tmp_path / "ptbxl.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "forbidden-inclusion"
    with pytest.raises(ValueError, match="outside folds1-7/train"):
        build_training_inclusion(
            db=_FakeDB(records),
            records=records,
            record_ids=tuple(records),
            source_manifest_file_sha256=sha256_file(source_manifest),
            source_manifest_sha256=SOURCE_SHA,
            split_sha256=SPLIT_SHA,
            rate_hz=500,
            segments=SEGMENTS,
            delineator="dwt",
            configurations=CONFIGURATIONS,
            configuration_panel_sha256=PANEL_SHA,
            output_dir=output,
        )
    assert not output.exists()


def test_preparation_script_reads_only_train_role_and_publishes_atomically(
    tmp_path, monkeypatch
):
    records = {
        "11": {"patient_id": "p11", "strat_fold": 1},
        "18": {"patient_id": "p18", "strat_fold": 8},
        "19": {"patient_id": "p19", "strat_fold": 9},
        "20": {"patient_id": "p20", "strat_fold": 10},
    }
    roles_requested = []

    def record_ids(role):
        roles_requested.append(role)
        return {"train": ("11",), "tune": ("18",), "calibration": ("19",), "test": ("20",)}[
            role
        ]

    contract = SimpleNamespace(
        root=tmp_path,
        records=records,
        manifest_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA,
        record_ids=record_ids,
    )
    manifest_path = tmp_path / "ptbxl.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        staging = Path(kwargs["output_dir"])
        staging.mkdir()
        (staging / MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
        return {"status": "complete"}

    verified = []
    monkeypatch.setattr(
        preparation, "load_ptbxl_manifest", lambda *_args, **_kwargs: contract
    )
    monkeypatch.setattr(
        preparation,
        "_verify_manifest_files",
        lambda _contract, ids, **_kwargs: verified.extend(ids),
    )
    monkeypatch.setattr(preparation, "_validate_database_identity", lambda *_args: None)
    monkeypatch.setattr(preparation, "PTBXL", lambda _root: object())
    monkeypatch.setattr(preparation, "build_training_inclusion", fake_build)
    output = tmp_path / "published-inclusion"
    result = preparation.run(
        argparse.Namespace(manifest=manifest_path, output_dir=output, release=False)
    )
    assert result == {"status": "complete"}
    assert roles_requested == ["train"]
    assert verified == ["11"]
    assert captured["record_ids"] == ("11",)
    assert output.is_dir()
    assert not list(tmp_path.glob(".published-inclusion.tmp-*"))


def test_preparation_script_removes_partial_staging_on_failure(tmp_path, monkeypatch):
    contract = SimpleNamespace(
        root=tmp_path,
        records={"11": {"patient_id": "p11", "strat_fold": 1}},
        manifest_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA,
        record_ids=lambda role: ("11",) if role == "train" else (),
    )
    manifest_path = tmp_path / "ptbxl.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    def fail_after_partial_write(**kwargs):
        staging = Path(kwargs["output_dir"])
        staging.mkdir()
        (staging / "partial").write_bytes(b"partial")
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(
        preparation, "load_ptbxl_manifest", lambda *_args, **_kwargs: contract
    )
    monkeypatch.setattr(preparation, "_verify_manifest_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preparation, "_validate_database_identity", lambda *_args: None)
    monkeypatch.setattr(preparation, "PTBXL", lambda _root: object())
    monkeypatch.setattr(
        preparation, "build_training_inclusion", fail_after_partial_write
    )
    output = tmp_path / "failed-inclusion"
    with pytest.raises(RuntimeError, match="fixture failure"):
        preparation.run(
            argparse.Namespace(manifest=manifest_path, output_dir=output, release=False)
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-inclusion.tmp-*"))


def test_dag_and_release_path_hash_bind_every_training_consumer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = ExperimentManifest.from_path(root / "scripts" / "experiment_manifest.yaml")
    by_id = {node.id: node for node in manifest.select("icassp")}
    inclusion_id = "reconstruction_training_inclusion"
    inclusion_dir = "artifacts/primary/reconstruction_training_inclusion"
    inclusion_file = f"{inclusion_dir}/{MANIFEST_FILENAME}"
    consumers = {
        "reconstruction_candidates",
        "official_baseline_preparation",
        "benchmark_lowrank",
        "benchmark_ridge",
        "benchmark_masked_unet",
        "benchmark_imputeecg",
        "benchmark_ecgrecover",
    }
    assert by_id[inclusion_id].outputs == (inclusion_dir,)
    for node_id in consumers:
        node = by_id[node_id]
        assert inclusion_id in node.deps
        assert inclusion_dir in node.inputs
        option = node.command.index("--training-inclusion")
        assert node.command[option + 1] == inclusion_file

    workspace = tmp_path / "workspace"
    artifact = workspace / inclusion_dir
    artifact.mkdir(parents=True)
    (artifact / MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
    (artifact / AUDIT_FILENAME).write_text("{}\n", encoding="utf-8")
    predictor = artifact / "training_predictors.parquet"
    predictor.write_bytes(b"predictor-v1")
    before = declared_path_hashes(workspace, (inclusion_dir,))[inclusion_dir]
    predictor.write_bytes(b"predictor-v2")
    after = declared_path_hashes(workspace, (inclusion_dir,))[inclusion_dir]
    assert before != after
