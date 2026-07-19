from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.panel_completeness import (
    PartitionCoverage,
    SourcePartition,
    load_ptbxl_source_partitions,
    require_identical_coverages,
    validate_metric_panel,
    validate_partition_audit,
    validate_predictor_tables,
)
from ecgcert.protocol import PatientSplit, PRIMARY_SEGMENTS, deep_configuration_panel
from ecgcert.reconstruction import SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION


def _source(partition: str, *, excluded: bool = False) -> SourcePartition:
    records = [(f"{partition}-included", f"{partition}-patient")]
    if excluded:
        records.append((f"{partition}-excluded", f"{partition}-excluded-patient"))
    return SourcePartition(
        cohort="PTB-XL",
        partition=partition,
        manifest_sha256="a" * 64,
        split_sha256="b" * 64,
        records=tuple(records),
    )


def _coverage(partition: str, *, excluded: bool = False) -> PartitionCoverage:
    source = _source(partition, excluded=excluded)
    records = [
        {
            "record_id": f"{partition}-included",
            "patient_id": f"{partition}-patient",
            "status": "included",
            "reason": None,
            "segment_counts": {"QRS": 10, "ST": 6, "T": 4},
        }
    ]
    if excluded:
        records.append(
            {
                "record_id": f"{partition}-excluded",
                "patient_id": f"{partition}-excluded-patient",
                "status": "excluded",
                "reason": "ValueError: bad signal",
                "segment_counts": {},
            }
        )
    audit = {
        "n_requested": len(records),
        "n_included": 1,
        "n_excluded": int(excluded),
        "audit_sha256": lineage.canonical_sha256(records),
        "records": records,
    }
    return validate_partition_audit(
        audit,
        source=source,
        audit_artifact_sha256=(partition[0] * 64),
    )


def _metric_rows(
    *,
    partitions=("test",),
    configurations=("I",),
    all_targets: bool = False,
) -> pd.DataFrame:
    rows = []
    for partition in partitions:
        for configuration in configurations:
            observed = set(configuration.split("+"))
            targets = CANONICAL_LEADS if all_targets else tuple(
                target for target in CANONICAL_LEADS if target not in observed
            )
            for segment, n_samples in {"QRS": 10, "ST": 6, "T": 4}.items():
                for target in targets:
                    rows.append(
                        {
                            "schema_version": BENCHMARK_SCHEMA_VERSION,
                            "cohort": "PTB-XL",
                            "partition": partition,
                            "patient_id": f"{partition}-patient",
                            "method": "lowrank",
                            "model_seed": 0,
                            "segment": segment,
                            "configuration": configuration,
                            "target": target,
                            "n_records": 1,
                            "n_samples": n_samples,
                            "target_rms": 1.0,
                            "max_target_observed_correlation": 0.5,
                            "outcome_log_rmse": -1.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_explicit_exclusions_are_audited_and_never_become_metric_patients() -> None:
    coverage = _coverage("test", excluded=True)
    assert coverage.attempted_record_ids == ("test-excluded", "test-included")
    assert coverage.included_record_ids == ("test-included",)
    assert coverage.excluded_record_ids == ("test-excluded",)
    assert coverage.excluded_reasons == (
        ("test-excluded", "ValueError: bad signal"),
    )
    report = validate_metric_panel(
        _metric_rows(),
        cohort="PTB-XL",
        coverages={"test": coverage},
        method_seeds={"lowrank": (0,)},
        configurations=("I",),
        chunk_rows=7,
    )
    assert report["n_evaluable_patients"] == 1
    assert report["coverage"]["test"]["n_excluded_records"] == 1


def test_audit_rejects_silent_manifest_truncation() -> None:
    source = _source("test", excluded=True)
    records = [
        {
            "record_id": "test-included",
            "patient_id": "test-patient",
            "status": "included",
            "reason": None,
            "segment_counts": {"QRS": 1, "ST": 1, "T": 1},
        }
    ]
    audit = {
        "n_requested": 1,
        "n_included": 1,
        "n_excluded": 0,
        "records": records,
    }
    with pytest.raises(ValueError, match="complete manifest partition"):
        validate_partition_audit(
            audit,
            source=source,
            audit_artifact_sha256="c" * 64,
        )


def test_science_audit_6912_row_full_target_fixture_is_rejected() -> None:
    """The audited bad fixture includes observed targets: 3*64*3*12 = 6912 rows."""

    configurations = tuple("+".join(value) for value in deep_configuration_panel())
    partitions = ("tune", "calibration", "test")
    frame = _metric_rows(
        partitions=partitions,
        configurations=configurations,
        all_targets=True,
    )
    assert len(frame) == 6_912
    with pytest.raises(ValueError, match="canonical12 minus the observed"):
        validate_metric_panel(
            frame,
            cohort="PTB-XL",
            coverages={partition: _coverage(partition) for partition in partitions},
            method_seeds={"lowrank": (0,)},
            configurations=configurations,
            chunk_rows=113,
        )


def test_metric_panel_rejects_one_missing_patient_target_cell() -> None:
    frame = _metric_rows()
    frame = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    with pytest.raises(ValueError, match="omit or duplicate|exactly one row"):
        validate_metric_panel(
            frame,
            cohort="PTB-XL",
            coverages={"test": _coverage("test")},
            method_seeds={"lowrank": (0,)},
            configurations=("I",),
        )


def test_metric_panel_rejects_audit_patient_or_sample_disagreement() -> None:
    frame = _metric_rows()
    frame.loc[0, "n_samples"] = 9
    with pytest.raises(ValueError, match="record or sample counts"):
        validate_metric_panel(
            frame,
            cohort="PTB-XL",
            coverages={"test": _coverage("test")},
            method_seeds={"lowrank": (0,)},
            configurations=("I",),
        )


def _predictors(configurations=("I",)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_partition": "PTB-XL/folds1-7/train",
                "segment": segment,
                "configuration": configuration,
                "target": target,
                "target_rms": 1.0,
                "max_target_observed_correlation": 0.5,
            }
            for segment in PRIMARY_SEGMENTS
            for configuration in configurations
            for target in CANONICAL_LEADS
        ]
    )


def test_predictor_tables_and_metric_values_have_one_folds1_7_source() -> None:
    predictors = _predictors()
    contract = validate_predictor_tables(
        {"lowrank": predictors, "ridge": predictors.copy()},
        artifact_sha256={"lowrank": "d" * 64, "ridge": "d" * 64},
        methods=("lowrank", "ridge"),
        configurations=("I",),
    )
    frame = _metric_rows()
    report = validate_metric_panel(
        frame,
        cohort="PTB-XL",
        coverages={"test": _coverage("test")},
        method_seeds={"lowrank": (0,)},
        configurations=("I",),
        predictor=contract,
    )
    assert report["predictor"]["source_partition"] == "PTB-XL/folds1-7/train"
    frame.loc[0, "target_rms"] = 2.0
    with pytest.raises(ValueError, match="training_predictors"):
        validate_metric_panel(
            frame,
            cohort="PTB-XL",
            coverages={"test": _coverage("test")},
            method_seeds={"lowrank": (0,)},
            configurations=("I",),
            predictor=contract,
        )


def test_predictor_tables_reject_hash_or_value_disagreement() -> None:
    predictors = _predictors()
    with pytest.raises(ValueError, match="artifact SHA-256"):
        validate_predictor_tables(
            {"lowrank": predictors, "ridge": predictors.copy()},
            artifact_sha256={"lowrank": "d" * 64, "ridge": "e" * 64},
            methods=("lowrank", "ridge"),
            configurations=("I",),
        )
    changed = predictors.copy()
    changed.loc[0, "target_rms"] = 2.0
    with pytest.raises(ValueError, match="predictor values"):
        validate_predictor_tables(
            {"lowrank": predictors, "ridge": changed},
            artifact_sha256={"lowrank": "d" * 64, "ridge": "d" * 64},
            methods=("lowrank", "ridge"),
            configurations=("I",),
        )


def test_method_audits_must_have_identical_scientific_coverage() -> None:
    reference = _coverage("test")
    assert require_identical_coverages((reference, reference)) == reference
    changed = PartitionCoverage(
        **{
            **reference.__dict__,
            "patient_segment_stats": (("test-patient", "QRS", 1, 9),),
        }
    )
    with pytest.raises(ValueError, match="disagree"):
        require_identical_coverages((reference, changed))


def test_ptbxl_manifest_membership_is_hash_and_fold_authenticated(tmp_path: Path) -> None:
    records = []
    split = {"train": [], "tune": [], "calibration": [], "test": []}
    for role, fold in (("train", 1), ("tune", 8), ("calibration", 9), ("test", 10)):
        record_id = str(fold)
        split[role].append(record_id)
        records.append(
            {
                "record_id": record_id,
                "patient_id": f"patient-{fold}",
                "strat_fold": fold,
                "files": {},
            }
        )
    split_sha256 = PatientSplit(
        train=("1",), tune=("8",), calibration=("9",), test=("10",)
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
    path = tmp_path / "ptbxl.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sources = load_ptbxl_source_partitions(
        path,
        expected_manifest_sha256=payload["manifest_sha256"],
        expected_split_sha256=split_sha256,
    )
    assert sources["test"].records == (("10", "patient-10"),)
    payload["records"][-1]["patient_id"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_ptbxl_source_partitions(
            path,
            expected_manifest_sha256=payload["manifest_sha256"],
        )
