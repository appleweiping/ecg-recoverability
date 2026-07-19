"""Fail-closed release contracts for reconstruction-panel evidence.

The patient metric tables are intentionally large.  The validators in this
module therefore authenticate the much smaller manifest/audit objects first and
then scan metric rows in bounded chunks.  Exact cell coverage is accumulated as
compact bit masks; no patient x configuration x target Cartesian DataFrame is
materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.data.manifest import DatasetManifest
from ecgcert.protocol import PatientSplit, PRIMARY_SEGMENTS
from ecgcert.reconstruction import (
    SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION,
    load_training_predictors,
    training_predictor_lookup,
)


@dataclass(frozen=True)
class SourcePartition:
    """Authenticated record/patient membership for one evaluation partition."""

    cohort: str
    partition: str
    manifest_sha256: str
    split_sha256: str
    records: tuple[tuple[str, str], ...]

    @property
    def record_to_patient(self) -> dict[str, str]:
        return dict(self.records)


@dataclass(frozen=True)
class PartitionCoverage:
    """Manifest-authenticated attempted, excluded, and evaluable patient sets."""

    cohort: str
    partition: str
    manifest_sha256: str
    split_sha256: str
    audit_artifact_sha256: str
    audit_records_sha256: str
    attempted_record_ids: tuple[str, ...]
    included_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    excluded_reasons: tuple[tuple[str, str], ...]
    patient_segment_stats: tuple[tuple[str, str, int, int], ...]

    def scientific_payload(self) -> dict[str, Any]:
        """Return the content that must agree across method-specific audits."""

        return {
            "cohort": self.cohort,
            "partition": self.partition,
            "manifest_sha256": self.manifest_sha256,
            "split_sha256": self.split_sha256,
            "audit_records_sha256": self.audit_records_sha256,
            "attempted_record_ids": list(self.attempted_record_ids),
            "included_record_ids": list(self.included_record_ids),
            "excluded_record_ids": list(self.excluded_record_ids),
            "excluded_reasons": [list(value) for value in self.excluded_reasons],
            "patient_segment_stats": [list(value) for value in self.patient_segment_stats],
        }

    @property
    def scientific_sha256(self) -> str:
        return lineage.canonical_sha256(self.scientific_payload())


@dataclass(frozen=True)
class PredictorContract:
    """One immutable folds-1--7 simple-predictor table shared by all methods."""

    source_partition: str
    artifact_sha256: str
    content_sha256: str
    methods: tuple[str, ...]
    lookup: Mapping[tuple[str, str, str], tuple[float, float]]

    def evidence(self) -> dict[str, Any]:
        return {
            "source_partition": self.source_partition,
            "artifact_sha256": self.artifact_sha256,
            "content_sha256": self.content_sha256,
            "methods": list(self.methods),
            "n_cells": len(self.lookup),
        }


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def load_ptbxl_source_partitions(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_split_sha256: str | None = None,
) -> dict[str, SourcePartition]:
    """Authenticate the fold-aware PTB-XL manifest without opening signal files."""

    payload = _read_json_object(path)
    if payload.get("schema_version") != "ptbxl-manifest-v3" or payload.get(
        "cohort"
    ) != "PTB-XL":
        raise ValueError("release metrics require a ptbxl-manifest-v3 source")
    embedded_manifest_sha256 = str(payload.get("manifest_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if (
        embedded_manifest_sha256 != expected_manifest_sha256
        or lineage.canonical_sha256(unhashed) != embedded_manifest_sha256
    ):
        raise ValueError("PTB-XL source manifest SHA-256 mismatch")

    records_payload = payload.get("records")
    split_payload = payload.get("split")
    if not isinstance(records_payload, list) or not isinstance(split_payload, Mapping):
        raise ValueError("PTB-XL source manifest lacks records/split membership")
    records: dict[str, tuple[str, int]] = {}
    for raw in records_payload:
        if not isinstance(raw, Mapping):
            raise ValueError("PTB-XL manifest contains a non-object record")
        record_id = str(raw.get("record_id", ""))
        patient_id = str(raw.get("patient_id", ""))
        try:
            fold = int(raw.get("strat_fold"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PTB-XL record {record_id!r} lacks strat_fold") from exc
        if not record_id or not patient_id or record_id in records:
            raise ValueError("PTB-XL manifest has empty or duplicate record identity")
        records[record_id] = (patient_id, fold)

    roles = ("train", "tune", "calibration", "test")
    if not set(roles) <= set(split_payload):
        raise ValueError("PTB-XL source split lacks a frozen role")
    split = {
        role: tuple(str(record_id) for record_id in split_payload[role])
        for role in roles
    }
    flat = [record_id for role in roles for record_id in split[role]]
    if len(flat) != len(set(flat)) or set(flat) != set(records):
        raise ValueError("PTB-XL source split is not an exact record partition")
    computed_split_sha256 = PatientSplit(
        train=split["train"],
        tune=split["tune"],
        calibration=split["calibration"],
        test=split["test"],
    ).sha256()
    embedded_split_sha256 = str(payload.get("split_sha256", ""))
    if computed_split_sha256 != embedded_split_sha256 or (
        expected_split_sha256 is not None
        and embedded_split_sha256 != expected_split_sha256
    ):
        raise ValueError("PTB-XL source split SHA-256 mismatch")

    expected_folds = {
        "train": set(range(1, 8)),
        "tune": {8},
        "calibration": {9},
        "test": {10},
    }
    patient_roles: dict[str, set[str]] = {}
    for role in roles:
        for record_id in split[role]:
            patient_id, fold = records[record_id]
            if fold not in expected_folds[role]:
                raise ValueError(f"PTB-XL record {record_id} has the wrong fold for {role}")
            patient_roles.setdefault(patient_id, set()).add(role)
    if any(len(values) != 1 for values in patient_roles.values()):
        raise ValueError("PTB-XL source manifest leaks patients across roles")

    return {
        role: SourcePartition(
            cohort="PTB-XL",
            partition=role,
            manifest_sha256=embedded_manifest_sha256,
            split_sha256=embedded_split_sha256,
            records=tuple(sorted((record_id, records[record_id][0]) for record_id in split[role])),
        )
        for role in ("tune", "calibration", "test")
    }


def load_external_test_source(
    path: str | Path,
    *,
    cohort: str,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
) -> SourcePartition:
    """Authenticate an external 60/20/20 manifest and return its test membership."""

    manifest = DatasetManifest.from_path(path)
    manifest.validate_release_contract(cohort)
    split = manifest.split()
    if (
        manifest.cohort != cohort
        or manifest.sha256() != expected_manifest_sha256
        or split.sha256() != expected_split_sha256
    ):
        raise ValueError(f"{cohort} source manifest/split SHA-256 mismatch")
    if not split.train or not split.tune or not split.test or split.calibration:
        raise ValueError(f"{cohort} source manifest does not realize frozen 60/20/20 roles")
    by_record = {str(record.record_id): str(record.patient_id) for record in manifest.records}
    return SourcePartition(
        cohort=cohort,
        partition="test",
        manifest_sha256=manifest.sha256(),
        split_sha256=split.sha256(),
        records=tuple(sorted((str(record_id), by_record[str(record_id)]) for record_id in split.test)),
    )


def validate_partition_audit(
    audit: Mapping[str, Any],
    *,
    source: SourcePartition,
    audit_artifact_sha256: str,
    segments: Sequence[str] = PRIMARY_SEGMENTS,
) -> PartitionCoverage:
    """Prove that every authorized record was attempted exactly once.

    Explicitly excluded records are accepted only when they retain a reason.
    Evaluable patient/segment record and sample counts are derived solely from
    included audit rows and later matched against every metric cell.
    """

    if len(audit_artifact_sha256) != 64:
        raise ValueError("partition audit lacks its artifact SHA-256")
    raw_records = audit.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("partition audit lacks record-level attempts")
    expected = source.record_to_patient
    seen: dict[str, Mapping[str, Any]] = {}
    segment_names = tuple(str(segment) for segment in segments)
    if not segment_names or len(segment_names) != len(set(segment_names)):
        raise ValueError("expected segments must be non-empty and unique")
    patient_stats: dict[tuple[str, str], list[int]] = {}
    included: list[str] = []
    excluded: list[str] = []
    excluded_reasons: list[tuple[str, str]] = []
    raw_exclusion_counts: dict[str, int] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("partition audit contains a non-object record")
        record_id = str(raw.get("record_id", ""))
        patient_id = str(raw.get("patient_id", ""))
        status = str(raw.get("status", ""))
        if not record_id or record_id in seen:
            raise ValueError("partition audit contains an empty or duplicate record ID")
        if record_id not in expected or patient_id != expected[record_id]:
            raise ValueError("partition audit record/patient membership disagrees with manifest")
        seen[record_id] = raw
        if status == "excluded":
            raw_reason = str(raw.get("reason") or "")
            normalized_reason = " ".join(raw_reason.split())
            if not normalized_reason:
                raise ValueError("explicitly excluded records require a retained reason")
            excluded.append(record_id)
            excluded_reasons.append((record_id, normalized_reason))
            raw_exclusion_counts[raw_reason] = raw_exclusion_counts.get(raw_reason, 0) + 1
            continue
        if status != "included" or raw.get("reason") not in (None, ""):
            raise ValueError("partition audit has an invalid inclusion status/reason")
        counts = raw.get("segment_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(segment_names):
            raise ValueError("included audit rows must report every primary segment")
        parsed_counts: dict[str, int] = {}
        for segment in segment_names:
            value = counts[segment]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("audit segment counts must be non-negative integers")
            parsed_counts[segment] = int(value)
        if not any(parsed_counts.values()):
            raise ValueError("included audit record has no evaluable primary segment")
        included.append(record_id)
        for segment, n_samples in parsed_counts.items():
            if n_samples:
                stats = patient_stats.setdefault((patient_id, segment), [0, 0])
                stats[0] += 1
                stats[1] += n_samples

    if set(seen) != set(expected) or len(seen) != len(expected):
        raise ValueError("partition audit does not attempt the complete manifest partition")
    n_included = len(included)
    n_excluded = len(excluded)
    summary = audit.get("summary")
    if isinstance(summary, Mapping):
        declared_total = summary.get("n_total")
        declared_included = summary.get("n_included")
        declared_excluded = summary.get("n_excluded")
        declared_exclusion_reasons = summary.get("exclusion_reasons")
    else:
        declared_total = audit.get("n_requested")
        declared_included = audit.get("n_included")
        declared_excluded = audit.get("n_excluded")
        declared_exclusion_reasons = audit.get("exclusion_reasons")
    if (
        declared_total != len(expected)
        or declared_included != n_included
        or declared_excluded != n_excluded
        or n_included + n_excluded != len(expected)
    ):
        raise ValueError("partition audit attempted/included/excluded counts are inconsistent")
    if declared_exclusion_reasons is not None and declared_exclusion_reasons != dict(
        sorted(raw_exclusion_counts.items())
    ):
        raise ValueError("partition audit exclusion-reason counts are inconsistent")

    records_sha256 = lineage.canonical_sha256(raw_records)
    embedded_sha256 = audit.get("audit_sha256")
    if embedded_sha256 is not None and embedded_sha256 != records_sha256:
        raise ValueError("partition audit record SHA-256 mismatch")
    stats_tuple = tuple(
        sorted(
            (patient_id, segment, values[0], values[1])
            for (patient_id, segment), values in patient_stats.items()
        )
    )
    if not stats_tuple:
        raise ValueError("partition audit contains no evaluable patient/segment")
    return PartitionCoverage(
        cohort=source.cohort,
        partition=source.partition,
        manifest_sha256=source.manifest_sha256,
        split_sha256=source.split_sha256,
        audit_artifact_sha256=audit_artifact_sha256,
        audit_records_sha256=records_sha256,
        attempted_record_ids=tuple(sorted(seen)),
        included_record_ids=tuple(sorted(included)),
        excluded_record_ids=tuple(sorted(excluded)),
        excluded_reasons=tuple(sorted(excluded_reasons)),
        patient_segment_stats=stats_tuple,
    )


def require_identical_coverages(
    coverages: Sequence[PartitionCoverage],
) -> PartitionCoverage:
    """Require method-specific audits to describe the same scientific cases."""

    if not coverages:
        raise ValueError("no partition coverage audits were provided")
    reference = coverages[0]
    for candidate in coverages[1:]:
        if candidate.scientific_payload() != reference.scientific_payload():
            raise ValueError("method bundles disagree on attempted/excluded patient coverage")
    return reference


def validate_predictor_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    artifact_sha256: Mapping[str, str],
    methods: Sequence[str],
    configurations: Sequence[str],
    segments: Sequence[str] = PRIMARY_SEGMENTS,
) -> PredictorContract:
    """Require byte-identical and value-identical folds-1--7 predictors."""

    method_order = tuple(methods)
    if set(tables) != set(method_order) or set(artifact_sha256) != set(method_order):
        raise ValueError("simple predictor inventory does not match the common methods")
    hashes = {str(artifact_sha256[method]) for method in method_order}
    if len(hashes) != 1 or any(len(value) != 64 for value in hashes):
        raise ValueError("common methods do not share one predictor artifact SHA-256")
    expected_keys = {
        (str(segment), str(configuration), str(target))
        for segment in segments
        for configuration in configurations
        for target in CANONICAL_LEADS
    }
    reference_lookup: dict[tuple[str, str, str], tuple[float, float]] | None = None
    content_sha256: str | None = None
    for method in method_order:
        lookup = training_predictor_lookup(tables[method])
        if set(lookup) != expected_keys:
            raise ValueError(f"{method} predictors do not cover the frozen simple-predictor grid")
        payload = [
            {
                "segment": segment,
                "configuration": configuration,
                "target": target,
                "target_rms": values[0],
                "max_target_observed_correlation": values[1],
            }
            for (segment, configuration, target), values in sorted(lookup.items())
        ]
        candidate_sha256 = lineage.canonical_sha256(payload)
        if reference_lookup is None:
            reference_lookup = lookup
            content_sha256 = candidate_sha256
        elif lookup != reference_lookup or candidate_sha256 != content_sha256:
            raise ValueError("common methods disagree on folds-1--7 predictor values")
    assert reference_lookup is not None and content_sha256 is not None
    return PredictorContract(
        source_partition="PTB-XL/folds1-7/train",
        artifact_sha256=next(iter(hashes)),
        content_sha256=content_sha256,
        methods=method_order,
        lookup=reference_lookup,
    )


def load_common_predictor_contract(
    bundles: Mapping[str, Any],
    *,
    methods: Sequence[str],
    configurations: Sequence[str],
    segments: Sequence[str] = PRIMARY_SEGMENTS,
) -> PredictorContract:
    """Load authenticated bundle predictor artifacts and compare them exactly."""

    method_order = tuple(methods)
    tables = {
        method: load_training_predictors(bundles[method].root) for method in method_order
    }
    hashes = {
        method: str(bundles[method].training_predictors_sha256)
        for method in method_order
    }
    return validate_predictor_tables(
        tables,
        artifact_sha256=hashes,
        methods=method_order,
        configurations=configurations,
        segments=segments,
    )


def _predictor_arrays(
    predictor: PredictorContract | None,
    *,
    segments: Sequence[str],
    configurations: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if predictor is None:
        return None
    rms = np.empty((len(segments), len(configurations), len(CANONICAL_LEADS)))
    correlation = np.empty_like(rms)
    for segment_index, segment in enumerate(segments):
        for configuration_index, configuration in enumerate(configurations):
            for target_index, target in enumerate(CANONICAL_LEADS):
                key = (str(segment), str(configuration), str(target))
                if key not in predictor.lookup:
                    raise ValueError(f"simple predictor contract lacks cell {key}")
                rms[segment_index, configuration_index, target_index], correlation[
                    segment_index, configuration_index, target_index
                ] = predictor.lookup[key]
    return rms, correlation


def validate_metric_panel(
    frame: pd.DataFrame,
    *,
    cohort: str,
    coverages: Mapping[str, PartitionCoverage],
    method_seeds: Mapping[str, Sequence[int]],
    configurations: Sequence[str],
    segments: Sequence[str] = PRIMARY_SEGMENTS,
    predictor: PredictorContract | None = None,
    chunk_rows: int = 500_000,
) -> dict[str, Any]:
    """Validate every patient/configuration/segment/target release cell exactly.

    For each cohort/partition/method/seed/configuration/segment group, the target
    mask must equal ``canonical12 - observed(configuration)`` for every patient
    with an evaluable audited segment.  Explicitly excluded records never enter
    that expected set; no un-audited or silently truncated patient can enter it.
    """

    if frame.empty:
        raise ValueError("release patient metrics are empty")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("metric validation chunk_rows must be a positive integer")
    segment_order = tuple(str(value) for value in segments)
    configuration_order = tuple(str(value) for value in configurations)
    method_order = tuple(str(value) for value in method_seeds)
    partition_order = tuple(sorted(str(value) for value in coverages))
    if (
        not segment_order
        or len(segment_order) != len(set(segment_order))
        or not configuration_order
        or len(configuration_order) != len(set(configuration_order))
        or not method_order
        or not partition_order
    ):
        raise ValueError("metric-panel release dimensions must be non-empty and unique")

    required = {
        "schema_version",
        "cohort",
        "partition",
        "patient_id",
        "method",
        "model_seed",
        "segment",
        "configuration",
        "target",
        "n_records",
        "n_samples",
        "target_rms",
        "max_target_observed_correlation",
        "outcome_log_rmse",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark metrics lack release columns: {sorted(missing)}")

    canonical_index = {lead: index for index, lead in enumerate(CANONICAL_LEADS)}
    all_targets_mask = (1 << len(CANONICAL_LEADS)) - 1
    missing_target_masks = np.empty(len(configuration_order), dtype=np.uint16)
    missing_target_counts = np.empty(len(configuration_order), dtype=np.uint8)
    for index, configuration in enumerate(configuration_order):
        observed = tuple(configuration.split("+"))
        if (
            not observed
            or len(observed) != len(set(observed))
            or any(lead not in canonical_index for lead in observed)
        ):
            raise ValueError(f"invalid observed configuration {configuration!r}")
        observed_mask = sum(1 << canonical_index[lead] for lead in observed)
        missing_mask = all_targets_mask ^ observed_mask
        missing_target_masks[index] = missing_mask
        missing_target_counts[index] = int(missing_mask).bit_count()

    pairs = tuple(
        (method, int(seed))
        for method in method_order
        for seed in method_seeds[method]
    )
    if not pairs or len(pairs) != len(set(pairs)) or any(seed < 0 for _, seed in pairs):
        raise ValueError("method/seed release panel is empty or duplicated")
    pair_by_method: dict[str, dict[int, int]] = {method: {} for method in method_order}
    for pair_index, (method, seed) in enumerate(pairs):
        pair_by_method[method][seed] = pair_index

    cases: list[tuple[str, str]] = []
    stats_by_case: list[dict[str, tuple[int, int]]] = []
    coverage_evidence: dict[str, Any] = {}
    patient_partition: dict[str, str] = {}
    for partition in partition_order:
        coverage = coverages[partition]
        if coverage.cohort != cohort or coverage.partition != partition:
            raise ValueError("metric coverage cohort/partition does not match its key")
        coverage_evidence[partition] = {
            "manifest_sha256": coverage.manifest_sha256,
            "split_sha256": coverage.split_sha256,
            "audit_artifact_sha256": coverage.audit_artifact_sha256,
            "audit_records_sha256": coverage.audit_records_sha256,
            "coverage_sha256": coverage.scientific_sha256,
            "n_attempted_records": len(coverage.attempted_record_ids),
            "n_included_records": len(coverage.included_record_ids),
            "n_excluded_records": len(coverage.excluded_record_ids),
        }
        per_patient: dict[str, dict[str, tuple[int, int]]] = {}
        for patient_id, segment, n_records, n_samples in coverage.patient_segment_stats:
            per_patient.setdefault(patient_id, {})[segment] = (n_records, n_samples)
        for patient_id in sorted(per_patient):
            previous = patient_partition.setdefault(patient_id, partition)
            if previous != partition:
                raise ValueError("audited patient appears in multiple evaluation partitions")
            cases.append((partition, patient_id))
            stats_by_case.append(per_patient[patient_id])
    if not cases:
        raise ValueError("release audit contains no evaluable patients")

    case_by_patient = {patient_id: index for index, (_partition, patient_id) in enumerate(cases)}
    partition_index = {value: index for index, value in enumerate(partition_order)}
    expected_case_partition = np.asarray(
        [partition_index[partition] for partition, _patient in cases], dtype=np.int16
    )
    expected_n_records = np.zeros((len(cases), len(segment_order)), dtype=np.int32)
    expected_n_samples = np.zeros((len(cases), len(segment_order)), dtype=np.int64)
    for case_index, stats in enumerate(stats_by_case):
        for segment_index, segment in enumerate(segment_order):
            n_records, n_samples = stats.get(segment, (0, 0))
            expected_n_records[case_index, segment_index] = n_records
            expected_n_samples[case_index, segment_index] = n_samples
    segment_available = expected_n_records > 0

    n_groups = len(cases) * len(pairs) * len(configuration_order) * len(segment_order)
    actual_target_masks = np.zeros(n_groups, dtype=np.uint16)
    actual_target_counts = np.zeros(n_groups, dtype=np.uint32)
    predictor_arrays = _predictor_arrays(
        predictor, segments=segment_order, configurations=configuration_order
    )

    def categorical_codes(values: pd.Series, categories: Sequence[str], label: str) -> np.ndarray:
        codes = pd.Categorical(values, categories=categories).codes
        if np.any(codes < 0):
            raise ValueError(f"benchmark metrics contain an unexpected {label}")
        return np.asarray(codes, dtype=np.int32)

    for start in range(0, len(frame), chunk_rows):
        chunk = frame.iloc[start : start + chunk_rows]
        if (
            not chunk["schema_version"].eq(BENCHMARK_SCHEMA_VERSION).all()
            or not chunk["cohort"].eq(cohort).all()
        ):
            raise ValueError("benchmark metrics use the wrong schema or cohort")
        partition_codes = categorical_codes(
            chunk["partition"], partition_order, "partition"
        )
        patient_codes_raw = chunk["patient_id"].map(case_by_patient)
        if patient_codes_raw.isna().any():
            raise ValueError("benchmark metrics contain a patient absent from the audit")
        case_codes = patient_codes_raw.to_numpy(dtype=np.int32)
        if not np.array_equal(
            partition_codes, expected_case_partition[case_codes]
        ):
            raise ValueError("benchmark patient appears in the wrong audited partition")
        segment_codes = categorical_codes(chunk["segment"], segment_order, "segment")
        configuration_codes = categorical_codes(
            chunk["configuration"], configuration_order, "configuration"
        )
        target_codes = categorical_codes(chunk["target"], CANONICAL_LEADS, "target")
        method_codes = categorical_codes(chunk["method"], method_order, "method")

        seed_numeric = pd.to_numeric(chunk["model_seed"], errors="coerce").to_numpy()
        if not np.isfinite(seed_numeric).all() or not np.equal(seed_numeric, np.floor(seed_numeric)).all():
            raise ValueError("benchmark metrics contain a non-integer model seed")
        seed_values = seed_numeric.astype(np.int64)
        pair_codes = np.full(len(chunk), -1, dtype=np.int32)
        for method_index, method in enumerate(method_order):
            method_rows = method_codes == method_index
            for seed, pair_index_value in pair_by_method[method].items():
                pair_codes[method_rows & (seed_values == seed)] = pair_index_value
        if np.any(pair_codes < 0):
            raise ValueError("benchmark metrics contain an unexpected method/seed pair")

        target_bits = np.left_shift(
            np.uint16(1), target_codes.astype(np.uint16)
        )
        legal_masks = missing_target_masks[configuration_codes]
        if np.any(np.bitwise_and(legal_masks, target_bits) == 0):
            raise ValueError(
                "metric target set is not canonical12 minus the observed configuration"
            )
        n_records = pd.to_numeric(chunk["n_records"], errors="coerce").to_numpy()
        n_samples = pd.to_numeric(chunk["n_samples"], errors="coerce").to_numpy()
        expected_records = expected_n_records[case_codes, segment_codes]
        expected_samples = expected_n_samples[case_codes, segment_codes]
        if (
            not np.isfinite(n_records).all()
            or not np.isfinite(n_samples).all()
            or not np.array_equal(n_records, expected_records)
            or not np.array_equal(n_samples, expected_samples)
        ):
            raise ValueError(
                "metric patient/segment record or sample counts disagree with the audit"
            )
        outcome = pd.to_numeric(chunk["outcome_log_rmse"], errors="coerce").to_numpy()
        if not np.isfinite(outcome).all():
            raise ValueError("benchmark metrics contain non-finite primary outcomes")

        if predictor_arrays is not None:
            expected_rms = predictor_arrays[0][
                segment_codes, configuration_codes, target_codes
            ]
            expected_correlation = predictor_arrays[1][
                segment_codes, configuration_codes, target_codes
            ]
            actual_rms = pd.to_numeric(chunk["target_rms"], errors="coerce").to_numpy()
            actual_correlation = pd.to_numeric(
                chunk["max_target_observed_correlation"], errors="coerce"
            ).to_numpy()
            if not np.array_equal(actual_rms, expected_rms) or not np.array_equal(
                actual_correlation, expected_correlation
            ):
                raise ValueError(
                    "metric simple predictors disagree with folds1-7 training_predictors"
                )

        group_codes = (
            (
                (case_codes.astype(np.int64) * len(pairs) + pair_codes)
                * len(configuration_order)
                + configuration_codes
            )
            * len(segment_order)
            + segment_codes
        )
        np.bitwise_or.at(actual_target_masks, group_codes, target_bits)
        np.add.at(actual_target_counts, group_codes, 1)

    actual_masks = actual_target_masks.reshape(
        len(cases), len(pairs), len(configuration_order), len(segment_order)
    )
    actual_counts = actual_target_counts.reshape(actual_masks.shape)
    expected_masks = (
        segment_available[:, None, None, :]
        * missing_target_masks[None, None, :, None]
    )
    expected_counts = (
        segment_available[:, None, None, :]
        * missing_target_counts[None, None, :, None]
    )
    if not np.array_equal(actual_masks, np.broadcast_to(expected_masks, actual_masks.shape)):
        raise ValueError("release metrics silently omit or duplicate patient target cells")
    if not np.array_equal(actual_counts, np.broadcast_to(expected_counts, actual_counts.shape)):
        raise ValueError("release metrics do not contain exactly one row per expected cell")

    return {
        "cohort": cohort,
        "n_metric_rows": int(len(frame)),
        "n_evaluable_patients": len(cases),
        "methods": list(method_order),
        "method_seeds": {
            method: [int(seed) for seed in method_seeds[method]] for method in method_order
        },
        "n_configurations": len(configuration_order),
        "segments": list(segment_order),
        "target_policy": "canonical12_minus_observed_configuration",
        "coverage": coverage_evidence,
        "predictor": None if predictor is None else predictor.evidence(),
    }


__all__ = [
    "PartitionCoverage",
    "PredictorContract",
    "SourcePartition",
    "load_common_predictor_contract",
    "load_external_test_source",
    "load_ptbxl_source_partitions",
    "require_identical_coverages",
    "validate_metric_panel",
    "validate_partition_audit",
    "validate_predictor_tables",
]
