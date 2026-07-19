"""Locked ICASSP 2027 study protocol.

This module contains the choices that must be frozen before outcome inspection:
patient splits, the independent-lead configuration universe, the neural benchmark
panel, primary sample rate/segments, segment-timepoint sampling, and the
bootstrap/rank grids. Experiment scripts import these values rather than defining
local variants.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from ecgcert.physics.dipolar_subspace import INDEPENDENT_LEADS


PRIMARY_RATE_HZ = 500
PRIMARY_SEGMENTS = ("QRS", "ST", "T")
SUPPLEMENTARY_SEGMENTS = ("P",)
RANK_GRID = (2, 3, 4, 5)
BOOTSTRAP_REPLICATES = 2_000
CONFIG_PANEL_SALT = "ecgcert-icassp27-v1"
CONFIG_PANEL_SHA256 = (
    "b5d629b911e38858e5512d685046a0cd05bc81f5afa26b8e0a30b81c5926c775"
)
EXTERNAL_SPLIT_RATIOS = (0.6, 0.2, 0.2)
EXTERNAL_SPLIT_ALGORITHM = "sha256-order-largest-remainder-60-20-20-v1"
PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD = 40
SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD = 80
SEGMENT_SAMPLING_SEED = 20260719
SEGMENT_SAMPLING_ALGORITHM = (
    "sha256-namespace-record-segment-pcg64-permutation-prefix-v1"
)
EXTERNAL_MAP_PRIMARY_METRIC = "ambiguity_robust_mv"
EXTERNAL_MAP_PRIMARY_ORDER = "lower_is_more_recoverable"
EXTERNAL_MAP_SECONDARY_DIAGNOSTIC = "recoverability_lower"


@dataclass(frozen=True)
class StudyProtocol:
    """Serializable, decision-complete protocol used by every main experiment."""

    rate_hz: int = PRIMARY_RATE_HZ
    primary_segments: tuple[str, ...] = PRIMARY_SEGMENTS
    supplementary_segments: tuple[str, ...] = SUPPLEMENTARY_SEGMENTS
    rank_grid: tuple[int, ...] = RANK_GRID
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    basis_variant: str = "independent8_lifted"
    sensitivity_basis_variant: str = "raw12_pca"
    configuration_salt: str = CONFIG_PANEL_SALT
    configuration_panel_sha256: str = CONFIG_PANEL_SHA256
    primary_segment_sample_cap_per_record: int = (
        PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD
    )
    sensitivity_segment_sample_cap_per_record: int = (
        SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
    )
    segment_sampling_seed: int = SEGMENT_SAMPLING_SEED
    segment_sampling_algorithm: str = SEGMENT_SAMPLING_ALGORITHM
    external_map_primary_metric: str = EXTERNAL_MAP_PRIMARY_METRIC
    external_map_primary_order: str = EXTERNAL_MAP_PRIMARY_ORDER
    external_map_secondary_diagnostic: str = EXTERNAL_MAP_SECONDARY_DIAGNOSTIC

    def validate(self) -> None:
        if self.rate_hz != 500:
            raise ValueError("The locked primary protocol uses 500 Hz")
        if self.primary_segments != PRIMARY_SEGMENTS:
            raise ValueError(f"Primary segments must be {PRIMARY_SEGMENTS!r}")
        if self.supplementary_segments != SUPPLEMENTARY_SEGMENTS:
            raise ValueError(
                f"Supplementary segments must be {SUPPLEMENTARY_SEGMENTS!r}"
            )
        if self.rank_grid != RANK_GRID:
            raise ValueError(f"Rank grid must be {RANK_GRID!r}")
        if self.bootstrap_replicates != BOOTSTRAP_REPLICATES:
            raise ValueError(
                f"Main-paper uncertainty requires exactly {BOOTSTRAP_REPLICATES} bootstraps"
            )
        if self.basis_variant != "independent8_lifted":
            raise ValueError("The primary basis must be independent8_lifted")
        if self.sensitivity_basis_variant != "raw12_pca":
            raise ValueError("The sensitivity basis must be raw12_pca")
        if self.configuration_salt != CONFIG_PANEL_SALT:
            raise ValueError(f"Configuration salt must be {CONFIG_PANEL_SALT!r}")
        if self.configuration_panel_sha256 != CONFIG_PANEL_SHA256:
            raise ValueError(
                f"Configuration panel SHA-256 must be {CONFIG_PANEL_SHA256!r}"
            )
        if (
            self.primary_segment_sample_cap_per_record
            != PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD
        ):
            raise ValueError(
                "Primary per-record/segment sample cap must be "
                f"{PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD}"
            )
        if (
            self.sensitivity_segment_sample_cap_per_record
            != SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
        ):
            raise ValueError(
                "Cap sensitivity must use "
                f"{SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD} samples per record/segment"
            )
        if self.segment_sampling_seed != SEGMENT_SAMPLING_SEED:
            raise ValueError(f"Segment sampling seed must be {SEGMENT_SAMPLING_SEED}")
        if self.segment_sampling_algorithm != SEGMENT_SAMPLING_ALGORITHM:
            raise ValueError(
                f"Segment sampling algorithm must be {SEGMENT_SAMPLING_ALGORITHM!r}"
            )
        if self.external_map_primary_metric != EXTERNAL_MAP_PRIMARY_METRIC:
            raise ValueError(
                f"External map primary metric must be {EXTERNAL_MAP_PRIMARY_METRIC!r}"
            )
        if self.external_map_primary_order != EXTERNAL_MAP_PRIMARY_ORDER:
            raise ValueError(
                f"External map primary order must be {EXTERNAL_MAP_PRIMARY_ORDER!r}"
            )
        if (
            self.external_map_secondary_diagnostic
            != EXTERNAL_MAP_SECONDARY_DIAGNOSTIC
        ):
            raise ValueError(
                "External map secondary diagnostic must be "
                f"{EXTERNAL_MAP_SECONDARY_DIAGNOSTIC!r}"
            )


@dataclass(frozen=True)
class PatientSplit:
    """Patient-disjoint train/tune/calibration/test record identifiers."""

    train: tuple[int | str, ...]
    tune: tuple[int | str, ...]
    calibration: tuple[int | str, ...]
    test: tuple[int | str, ...]

    def validate(self) -> None:
        groups: dict[str, set[str]] = {}
        for name in ("train", "tune", "calibration", "test"):
            normalized = tuple(str(value) for value in getattr(self, name))
            if any(not value for value in normalized):
                raise ValueError(f"{name} contains an empty record identifier")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} contains duplicate record identifiers")
            groups[name] = set(normalized)
        names = tuple(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    sample = sorted(map(str, overlap))[:5]
                    raise ValueError(f"{left}/{right} leakage: {sample}")

    def sha256(self) -> str:
        self.validate()
        payload = json.dumps(
            {
                name: sorted(str(value) for value in getattr(self, name))
                for name in ("train", "tune", "calibration", "test")
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ptbxl_split(db, *, population: str = "all") -> PatientSplit:
    """Return the locked PTB-XL fold split.

    The main analysis is all-diagnosis. ``population='norm'`` is available only
    for the predeclared sensitivity analysis.  PTB-XL stratified folds are
    patient-disjoint; we nevertheless validate patient IDs when present.
    """

    if population not in {"all", "norm"}:
        raise ValueError("population must be 'all' or 'norm'")
    if not db.meta.index.is_unique:
        raise ValueError("PTB-XL metadata contains duplicate record identifiers")
    try:
        folds_present = {int(value) for value in db.meta["strat_fold"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PTB-XL metadata has invalid strat_fold values") from exc
    if not folds_present <= set(range(1, 11)) or folds_present != set(range(1, 11)):
        raise ValueError("PTB-XL metadata must realize every official fold 1--10")

    def ids(folds: Sequence[int]) -> tuple[int, ...]:
        if population == "norm":
            values = db.ids_with_superclass("NORM", exclusive=False, folds=folds)
        else:
            values = db.meta[db.meta["strat_fold"].isin(folds)].index.to_numpy()
        return tuple(int(x) for x in sorted(values))

    split = PatientSplit(
        train=ids(range(1, 8)),
        tune=ids((8,)),
        calibration=ids((9,)),
        test=ids((10,)),
    )
    split.validate()
    assigned = set(split.train) | set(split.tune) | set(split.calibration) | set(split.test)
    expected = (
        {int(value) for value in db.meta.index}
        if population == "all"
        else {
            int(value)
            for value in db.ids_with_superclass(
                "NORM", exclusive=False, folds=range(1, 11)
            )
        }
    )
    if assigned != expected:
        raise ValueError("PTB-XL fold roles do not cover the requested population exactly")

    if "patient_id" in db.meta.columns:
        patient_values = db.meta["patient_id"].astype(str)
        if patient_values.eq("").any() or patient_values.str.lower().eq("nan").any():
            raise ValueError("PTB-XL metadata contains an empty patient identifier")
        patient_sets = []
        for record_ids in (split.train, split.tune, split.calibration, split.test):
            patient_sets.append(set(db.meta.loc[list(record_ids), "patient_id"].astype(str)))
        for i, left in enumerate(patient_sets):
            for right in patient_sets[i + 1 :]:
                overlap = left & right
                if overlap:
                    raise ValueError(f"PTB-XL patient leakage across folds: {sorted(overlap)[:5]}")
    return split


def patient_hash_split(
    record_to_patient: Mapping[int | str, int | str],
    *,
    salt: str,
    ratios: tuple[float, float, float] = EXTERNAL_SPLIT_RATIOS,
) -> PatientSplit:
    """Deterministically split an external cohort 60/20/20 by patient hash.

    The first partition is cohort-map fitting, the second is development/QC,
    and the third is the untouched zero-shot test.  ``calibration`` is empty
    because external outcomes are never used to fit the PTB-XL meta-model.

    Patients are ordered by salted SHA-256 and then sliced using the largest-
    remainder integer allocation.  This realizes the declared proportions to
    within one patient, unlike independent hash thresholds whose realized split
    sizes can drift randomly.  Record identifiers retain their input type, but
    their string representations must be unique because manifests serialize
    identifiers as strings.
    """

    if not isinstance(salt, str) or not salt:
        raise ValueError("external split salt must be a non-empty string")
    if len(ratios) != 3 or not np.isfinite(ratios).all() or not np.isclose(
        sum(ratios), 1.0
    ):
        raise ValueError("ratios must contain three values summing to one")
    if any(x <= 0 for x in ratios):
        raise ValueError("all external split ratios must be positive")

    normalized_records = [str(record) for record in record_to_patient]
    if len(normalized_records) != len(set(normalized_records)):
        raise ValueError("record identifiers collide after string normalization")
    normalized_patients = [str(patient) for patient in record_to_patient.values()]
    if any(not patient for patient in normalized_patients):
        raise ValueError("external split contains an empty patient identifier")
    unique_patients = sorted(
        set(normalized_patients),
        key=lambda patient: (
            hashlib.sha256(f"{salt}|{patient}".encode("utf-8")).hexdigest(),
            patient,
        ),
    )
    raw_counts = np.asarray(ratios, dtype=float) * len(unique_patients)
    counts = np.floor(raw_counts).astype(int)
    remainder = len(unique_patients) - int(counts.sum())
    fractional_order = sorted(
        range(3), key=lambda index: (-(raw_counts[index] - counts[index]), index)
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    boundaries = np.cumsum(counts)
    roles = ("train", "tune", "test")
    patient_partition = {
        patient: roles[int(np.searchsorted(boundaries, index, side="right"))]
        for index, patient in enumerate(unique_patients)
    }

    buckets: dict[str, list[int | str]] = {"train": [], "tune": [], "test": []}
    for record, patient in sorted(record_to_patient.items(), key=lambda item: str(item[0])):
        buckets[patient_partition[str(patient)]].append(record)
    split = PatientSplit(
        train=tuple(buckets["train"]),
        tune=tuple(buckets["tune"]),
        calibration=(),
        test=tuple(buckets["test"]),
    )
    split.validate()
    assigned = set(split.train) | set(split.tune) | set(split.test)
    if assigned != set(record_to_patient) or sum(map(len, buckets.values())) != len(
        record_to_patient
    ):
        raise AssertionError("patient hash split did not assign every record exactly once")
    return split


def canonical_configuration(leads: Iterable[str]) -> tuple[str, ...]:
    """Return a configuration in canonical independent-lead order."""

    requested = tuple(leads)
    unknown = set(requested) - set(INDEPENDENT_LEADS)
    if unknown:
        raise ValueError(f"Configuration contains non-independent leads: {sorted(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("Configuration contains duplicate leads")
    return tuple(lead for lead in INDEPENDENT_LEADS if lead in requested)


def all_independent_configurations() -> tuple[tuple[str, ...], ...]:
    """All 255 non-empty subsets of ``[I, II, V1..V6]``."""

    out = []
    for size in range(1, len(INDEPENDENT_LEADS) + 1):
        out.extend(combinations(INDEPENDENT_LEADS, size))
    return tuple(out)


def _hash_order(config: tuple[str, ...], salt: str) -> str:
    canonical = ",".join(canonical_configuration(config))
    return hashlib.sha256(f"{salt}|{canonical}".encode("utf-8")).hexdigest()


def deep_configuration_panel(salt: str = CONFIG_PANEL_SALT) -> tuple[tuple[str, ...], ...]:
    """Return the locked, outcome-independent 64-configuration neural panel."""

    by_size: dict[int, list[tuple[str, ...]]] = {size: [] for size in range(1, 9)}
    for config in all_independent_configurations():
        by_size[len(config)].append(config)

    chosen: list[tuple[str, ...]] = []
    chosen.extend(by_size[1])
    chosen.extend(by_size[2])
    quotas = {3: 7, 4: 7, 5: 7, 6: 4, 7: 2, 8: 1}
    for size, quota in quotas.items():
        ranked = sorted(by_size[size], key=lambda config: (_hash_order(config, salt), config))
        chosen.extend(ranked[:quota])

    panel = tuple(chosen)
    if len(panel) != 64 or len(set(panel)) != 64:
        raise AssertionError("The locked neural configuration panel must contain 64 unique sets")
    if salt == CONFIG_PANEL_SALT:
        payload = "\n".join(",".join(configuration) for configuration in panel)
        observed_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if observed_sha256 != CONFIG_PANEL_SHA256:
            raise AssertionError(
                "The locked neural configuration panel changed from its preregistered SHA-256"
            )
    return panel


def configuration_panel_sha256(panel: Sequence[Sequence[str]] | None = None) -> str:
    panel = deep_configuration_panel() if panel is None else panel
    payload = "\n".join(",".join(canonical_configuration(config)) for config in panel)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
