import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.physics import LEADS
from ecgcert.protocol import BOOTSTRAP_REPLICATES, PRIMARY_RATE_HZ, PRIMARY_SEGMENTS
from ecgcert.recoverability import bootstrap_spatial_model_bank
from experiments.reconstruction_benchmark_v3 import PTBXLManifestV3
from experiments.robust_maps_v3 import (
    SCHEMA_VERSION,
    _artifact_hashes,
    _load_primary_summary,
    _resolve_sensitivity,
    _role_ids,
    _verify_manifest_identity,
    summarize_model_bank,
    validate_release_arguments,
)


def test_rank_robust_summary_uses_patient_bootstrap_and_all_targets():
    rng = np.random.default_rng(17)
    latent = rng.normal(size=(80, 3))
    mixing, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    X = latent @ mixing.T + rng.normal(scale=0.01, size=(80, 12))
    patient_ids = np.asarray([f"patient-{index // 10}" for index in range(80)], dtype=object)
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(2, 3),
        basis_variants=("independent8_lifted",),
        n_boot=5,
        seed=4,
    )
    rank_path, cells = summarize_model_bank(
        bank,
        (("I",), ("I", "II")),
        segment="QRS",
        observation_variance_mv2=1e-4,
    )
    assert len(rank_path) == 2 * 2 * len(LEADS)
    assert len(cells) == 2 * len(LEADS)
    assert set(cells["bootstrap_replicates"]) == {5}
    assert np.all(cells["ambiguity_robust_mv"] >= 0)
    assert np.all((cells["recoverability_lower"] >= 0) & (cells["recoverability_lower"] <= 1))
    assert np.all(cells["target_rms"] > 0)
    assert np.all((cells["max_target_observed_correlation"] >= 0) &
                  (cells["max_target_observed_correlation"] <= 1 + 1e-12))


def test_cross_rank_robust_score_contains_each_rank_upper_bound():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(60, 12))
    patients = [f"p{index // 6}" for index in range(60)]
    bank = bootstrap_spatial_model_bank(
        X,
        patients,
        ranks=(2, 3, 4),
        basis_variants=("raw12_pca",),
        n_boot=4,
        seed=1,
    )
    rank_path, cells = summarize_model_bank(
        bank,
        (("I", "V1"),),
        segment="T",
        observation_variance_mv2=1e-3,
    )
    for target, rows in rank_path.groupby("target"):
        robust = float(cells.loc[cells["target"] == target, "ambiguity_robust_mv"].iloc[0])
        assert robust >= float(rows["ambiguity_q975_mv"].max()) - 1e-12


def _release_arguments(**overrides):
    values = {
        "release": True,
        "mode": "primary",
        "sensitivity": None,
        "diagnosis_class": None,
        "primary_rank_maps": None,
        "manifest": Path("manifest.json"),
        "max_records": None,
        "n_bootstrap": BOOTSTRAP_REPLICATES,
        "segments": PRIMARY_SEGMENTS,
        "rate": PRIMARY_RATE_HZ,
        "population": "all",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
        "max_per_record": 40,
        "observation_variance_mv2": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("sensitivity", "diagnosis", "changed_field", "changed_value"),
    [
        ("p-wave", None, "segments", ("P",)),
        ("100hz", None, "rate", 100),
        ("delineator", None, "delineator", "peak"),
        ("raw12", None, "basis_variant", "raw12_pca"),
        ("diagnosis", "MI", None, None),
    ],
)
def test_release_sensitivities_allow_only_the_preregistered_difference(
    sensitivity, diagnosis, changed_field, changed_value
):
    arguments = _release_arguments(
        mode="sensitivity",
        sensitivity=sensitivity,
        diagnosis_class=diagnosis,
        primary_rank_maps=Path("primary"),
    )
    _resolve_sensitivity(arguments, {"observation_variance_mv2": 1e-4})
    validate_release_arguments(arguments)
    if changed_field is not None:
        assert getattr(arguments, changed_field) == changed_value
    arguments.max_records = 1
    with pytest.raises(SystemExit, match="max-records is forbidden"):
        validate_release_arguments(arguments)


def test_primary_release_rejects_sensitivity_only_variants():
    for change in (
        {"segments": ("P",)},
        {"rate": 100},
        {"delineator": "peak"},
        {"basis_variant": "raw12_pca"},
        {"diagnosis_class": "NORM"},
        {"observation_variance_mv2": 1e-4},
    ):
        with pytest.raises(SystemExit, match="release protocol violation"):
            validate_release_arguments(_release_arguments(**change))


def test_diagnosis_role_filter_is_multilabel_and_preserves_manifest_order():
    contract = PTBXLManifestV3(
        path=Path("manifest.json"),
        root=Path("data"),
        records={str(index): {} for index in range(1, 7)},
        split={
            "train": ("3", "1", "2", "4"),
            "tune": ("6", "5"),
            "calibration": (),
            "test": (),
        },
        manifest_sha256="a" * 64,
        split_sha256="b" * 64,
    )
    db = type("DB", (), {})()
    db.meta = pd.DataFrame(
        {
            "superclass": [
                ["MI", "STTC"],
                ["NORM"],
                ["CD", "MI"],
                ["HYP"],
                ["MI"],
                ["STTC"],
            ]
        },
        index=range(1, 7),
    )
    train, tune = _role_ids(contract, db, "MI")
    assert train == ("3", "1")
    assert tune == ("5",)


def test_primary_summary_verifies_direct_artifact_hashes(tmp_path: Path):
    paths = {
        "rank_path": "rank_path.parquet",
        "map_cells": "map_cells.parquet",
        "regularization_tuning": "regularization_tuning.parquet",
        "patient_audit": "patient_audit.json",
    }
    for index, relative in enumerate(paths.values()):
        (tmp_path / relative).write_bytes(f"artifact-{index}".encode())
    hashes = _artifact_hashes(tmp_path, paths)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_mode": "primary",
        "population": "all",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
        "segments": list(PRIMARY_SEGMENTS),
        "rate_hz": PRIMARY_RATE_HZ,
        "ranks": [2, 3, 4, 5],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "patient",
        "observation_variance_mv2": 1e-4,
        "artifacts": paths,
        "artifact_sha256": hashes,
    }
    (tmp_path / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _load_primary_summary(tmp_path)["artifact_sha256"] == hashes
    (tmp_path / "map_cells.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_primary_summary(tmp_path)


def test_release_identity_checks_complete_manifest_and_metadata(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "ptbxl_database.csv"
    statements = tmp_path / "scp_statements.csv"
    metadata.write_bytes(b"metadata")
    statements.write_bytes(b"statements")
    records = {
        "1": {
            "patient_id": "p1",
            "strat_fold": 1,
            "files": {"500": {"record": "records500/1_hr"}},
        },
        "2": {
            "patient_id": "p2",
            "strat_fold": 8,
            "files": {"500": {"record": "records500/2_hr"}},
        },
    }
    payload = {
        "metadata_sha256": lineage.artifact_sha256(metadata),
        "scp_statements_sha256": lineage.artifact_sha256(statements),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    contract = PTBXLManifestV3(
        path=manifest_path,
        root=tmp_path,
        records=records,
        split={"train": ("1",), "tune": ("2",), "calibration": (), "test": ()},
        manifest_sha256="a" * 64,
        split_sha256="b" * 64,
    )
    db = type("DB", (), {})()
    db.meta = pd.DataFrame(
        {
            "strat_fold": [1, 8],
            "filename_hr": ["records500/1_hr", "records500/2_hr"],
        },
        index=[1, 2],
    )
    calls = []
    monkeypatch.setattr(
        "experiments.robust_maps_v3._verify_manifest_files",
        lambda manifest, ids, rate: calls.append((tuple(ids), rate)),
    )
    monkeypatch.setattr(
        "experiments.robust_maps_v3._validate_database_identity",
        lambda db_value, manifest, ids: None,
    )
    _verify_manifest_identity(contract, db, rate=500, release=True)
    assert calls == [(tuple(records), 500)]
    db.meta.loc[2, "filename_hr"] = "different"
    with pytest.raises(ValueError, match="record path mismatch"):
        _verify_manifest_identity(contract, db, rate=500, release=True)
