import json

import pytest

from ecgcert.data.manifest import (
    DatasetManifest,
    ManifestRecord,
    build_wfdb_manifest,
    hash_files,
)


def test_external_manifest_roundtrip_and_file_verification(tmp_path):
    header = tmp_path / "record.hea"
    signal = tmp_path / "record.dat"
    header.write_text("record 12 500 2\n", encoding="utf-8")
    signal.write_bytes(b"fixture-signal")
    manifest = build_wfdb_manifest(
        cohort="fixture",
        version="1",
        source_url="https://example.invalid/fixture",
        root=tmp_path,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    loaded = DatasetManifest.from_path(path)
    assert loaded.sha256() == manifest.sha256()
    assert loaded.structure() == {
        "n_records": 1,
        "n_patients": 1,
        "records_per_patient_min": 1,
        "records_per_patient_max": 1,
        "split": {
            "train": {"n_records": 1, "n_patients": 1},
            "tune": {"n_records": 0, "n_patients": 0},
            "calibration": {"n_records": 0, "n_patients": 0},
            "test": {"n_records": 0, "n_patients": 0},
        },
    }
    loaded.verify_files()
    signal.write_bytes(b"changed")
    try:
        loaded.verify_files()
    except ValueError as exc:
        assert "signal" in str(exc)
    else:
        raise AssertionError("changed signal must fail manifest verification")


def test_parallel_hashes_are_deterministic_and_worker_bounded(tmp_path, monkeypatch):
    paths = []
    for index in range(8):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]) * (index + 1))
        paths.append(path)
    assert hash_files(reversed(paths), workers=3) == hash_files(paths, workers=1)
    monkeypatch.setenv("ECGCERT_NUM_WORKERS", "11")
    with pytest.raises(ValueError, match="workers"):
        hash_files(paths)


def test_external_manifest_fails_closed_on_missing_or_false_structure(tmp_path):
    header = tmp_path / "record.hea"
    signal = tmp_path / "record.dat"
    header.write_text("record 12 500 2\n", encoding="utf-8")
    signal.write_bytes(b"fixture-signal")
    payload = build_wfdb_manifest(
        cohort="fixture",
        version="1",
        source_url="https://example.invalid/fixture",
        root=tmp_path,
    ).to_dict()
    for field in (
        "manifest_sha256",
        "split_sha256",
        "split_algorithm",
        "split_ratios",
        "structure",
    ):
        broken = dict(payload)
        broken.pop(field)
        with pytest.raises(ValueError, match="missing fields"):
            DatasetManifest.from_dict(broken)
    broken = dict(payload)
    broken["structure"] = {**payload["structure"], "n_records": 2}
    with pytest.raises(ValueError, match="structure counts"):
        DatasetManifest.from_dict(broken)


def test_wfdb_manifest_rejects_partial_cohort_and_patient_mapping(tmp_path):
    (tmp_path / "record.hea").write_text("record 12 500 2\n", encoding="utf-8")
    (tmp_path / "record.dat").write_bytes(b"fixture-signal")
    with pytest.raises(ValueError, match="incomplete"):
        build_wfdb_manifest(
            cohort="fixture",
            version="1",
            source_url="https://example.invalid/fixture",
            root=tmp_path,
            expected_record_count=2,
        )
    with pytest.raises(ValueError, match="patient mapping lacks"):
        build_wfdb_manifest(
            cohort="fixture",
            version="1",
            source_url="https://example.invalid/fixture",
            root=tmp_path,
            patient_by_record={},
        )


def test_release_external_contract_rejects_self_consistent_partial_manifest(tmp_path):
    manifest = DatasetManifest(
        cohort="chapman",
        version="1.0.0",
        source_url="https://physionet.org/content/ecg-arrhythmia/1.0.0/",
        root=str(tmp_path),
        records=(
            ManifestRecord(
                record_id="record",
                patient_id="record",
                relative_header="record.hea",
                header_sha256="0" * 64,
                signal_file="record.dat",
                signal_size_bytes=1,
                signal_sha256="1" * 64,
            ),
        ),
        split_salt="ecgcert-chapman-1.0.0-v1",
        patient_id_strategy="official_one_patient_ecg_per_unique_wfdb_record_name",
    )
    with pytest.raises(ValueError, match="n_records"):
        manifest.validate_release_contract("chapman")
