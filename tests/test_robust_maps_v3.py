import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.physics import LEADS
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    PRIMARY_SEGMENTS,
    SEGMENT_SAMPLING_SEED,
    SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
)
from ecgcert.recoverability import (
    BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
    BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
    BOOTSTRAP_REPLAY_SCHEMA_VERSION,
    bootstrap_spatial_model_bank,
)
from experiments.reconstruction_benchmark_v3 import PTBXLManifestV3
from experiments.robust_maps_v3 import (
    BOOTSTRAP_AUDIT_SCHEMA_VERSION,
    BOOTSTRAP_DRAW_SCHEMA_VERSION,
    ROBUST_MAP_INVENTORY_FILENAME,
    SEGMENT_ARTIFACT_FILENAMES,
    SCHEMA_VERSION,
    RobustMapSegmentStore,
    _artifact_hashes,
    _load_primary_summary,
    _resolve_sensitivity,
    _role_ids,
    _segment_sampling_config,
    _verify_manifest_identity,
    _write_parquet,
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
    assert cells["target_observed"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all()
    assert cells.apply(
        lambda row: bool(row["target_observed"])
        == (str(row["target"]) in str(row["configuration"]).split("+")),
        axis=1,
    ).all()


def test_cross_rank_robust_score_is_exact_max_of_per_rank_upper_bounds():
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
        cell = cells.loc[cells["target"] == target].iloc[0]
        assert float(cell["ambiguity_robust_mv"]) == pytest.approx(
            float(rows["ambiguity_q975_mv"].max()), abs=1e-15
        )
        assert float(cell["recoverability_lower"]) == pytest.approx(
            np.clip(1.0 - float(rows["eta_normalized_q975"].max()), 0.0, 1.0),
            abs=1e-15,
        )
        assert float(cell["log10_kappa_target_upper"]) == pytest.approx(
            np.log10(
                max(float(rows["kappa_target_q975"].max()), np.finfo(float).tiny)
            ),
            abs=1e-15,
        )


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
        "max_per_record": PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        "sampling_seed": SEGMENT_SAMPLING_SEED,
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
        (
            "sample-cap",
            None,
            "max_per_record",
            SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        ),
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
        {"max_per_record": SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD},
        {"sampling_seed": SEGMENT_SAMPLING_SEED + 1},
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
        "bootstrap_draws": "bootstrap_draws.parquet",
        "bootstrap_patients": "bootstrap_patients.parquet",
        "bootstrap_multiplicities": "bootstrap_multiplicities.parquet",
        "bootstrap_audit": "bootstrap_audit.parquet",
        "bootstrap_moments": "bootstrap_moments.parquet",
        "bootstrap_attempts": "bootstrap_attempts.parquet",
    }
    for index, relative in enumerate(paths.values()):
        (tmp_path / relative).write_bytes(f"artifact-{index}".encode())
    hashes = _artifact_hashes(tmp_path, paths)
    sampling = _segment_sampling_config(
        cap_per_record=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        seed=SEGMENT_SAMPLING_SEED,
    )
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
        "bootstrap_evidence_schema_version": BOOTSTRAP_AUDIT_SCHEMA_VERSION,
        "bootstrap_draw_schema_version": BOOTSTRAP_DRAW_SCHEMA_VERSION,
        "bootstrap_replay_schema_version": BOOTSTRAP_REPLAY_SCHEMA_VERSION,
        "bootstrap_moments_schema_version": BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
        "bootstrap_attempt_schema_version": BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
        "segment_sampling": sampling,
        "segment_sampling_sha256": lineage.canonical_sha256(sampling),
        "observation_variance_mv2": 1e-4,
        "artifacts": paths,
        "artifact_sha256": hashes,
    }
    (tmp_path / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _load_primary_summary(tmp_path)["artifact_sha256"] == hashes
    changed = json.loads(json.dumps(summary))
    changed["segment_sampling"]["active_cap_per_record_per_segment"] = 41
    (tmp_path / "summary.v3.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen primary map"):
        _load_primary_summary(tmp_path)
    (tmp_path / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
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


def _write_segment_fixture(store: RobustMapSegmentStore, segment: str) -> None:
    for index, name in enumerate(SEGMENT_ARTIFACT_FILENAMES):
        _write_parquet(
            pd.DataFrame(
                {
                    "segment": [segment, segment],
                    "artifact_index": [index, index],
                    "value": [1.0, 2.0],
                }
            ),
            store.segment_artifact(segment, name),
        )
    store.commit_segment(
        segment,
        metadata={
            "n_rank_rows": 2,
            "n_map_cells": 2,
            "n_bootstrap_draw_rows": 2,
            "n_bootstrap_attempt_rows": 2,
            "bootstrap_rejection": {
                "rejected_draws": 0,
                "rejection_fraction": 0.0,
            },
        },
    )


def test_segment_store_recovers_completed_wave_and_streams_atomic_merge(tmp_path: Path):
    pytest.importorskip("pyarrow")
    output = tmp_path / "maps"
    identity = {"manifest_sha256": "a" * 64, "seed": 7}
    first = RobustMapSegmentStore(
        output, identity=identity, segments=("QRS", "ST")
    )
    _write_segment_fixture(first, "QRS")

    resumed = RobustMapSegmentStore(
        output, identity=identity, segments=("QRS", "ST")
    )
    assert resumed.is_complete("QRS")
    assert not resumed.is_complete("ST")
    _write_segment_fixture(resumed, "ST")
    final_paths = {}
    for name, filename in SEGMENT_ARTIFACT_FILENAMES.items():
        resumed.merge_parquet(name, output / filename)
        final_paths[name] = filename
    resumed.mark_complete(final_paths)
    resumed.cleanup_staging()

    complete = RobustMapSegmentStore(
        output, identity=identity, segments=("QRS", "ST")
    )
    assert complete.status == "complete"
    assert not complete.staging_dir.exists()
    assert (output / ROBUST_MAP_INVENTORY_FILENAME).is_file()
    import pyarrow.parquet as pq

    merged = pq.ParquetFile(output / SEGMENT_ARTIFACT_FILENAMES["bootstrap_draws"])
    assert merged.metadata.num_row_groups == 2
    assert merged.metadata.num_rows == 4


def test_segment_store_fails_closed_if_completed_segment_changes(tmp_path: Path):
    pytest.importorskip("pyarrow")
    output = tmp_path / "maps"
    identity = {"manifest_sha256": "a" * 64, "seed": 7}
    store = RobustMapSegmentStore(output, identity=identity, segments=("QRS",))
    _write_segment_fixture(store, "QRS")
    path = store.segment_artifact("QRS", "map_cells")
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        RobustMapSegmentStore(output, identity=identity, segments=("QRS",))


def test_segment_store_resume_identity_binds_timepoint_sampling(tmp_path: Path):
    output = tmp_path / "maps"
    primary_sampling = _segment_sampling_config(
        cap_per_record=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        seed=SEGMENT_SAMPLING_SEED,
    )
    RobustMapSegmentStore(
        output,
        identity={"segment_sampling": primary_sampling},
        segments=("QRS",),
    )
    changed_sampling = _segment_sampling_config(
        cap_per_record=SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        seed=SEGMENT_SAMPLING_SEED,
    )
    with pytest.raises(ValueError, match="resume identity changed"):
        RobustMapSegmentStore(
            output,
            identity={"segment_sampling": changed_sampling},
            segments=("QRS",),
        )
