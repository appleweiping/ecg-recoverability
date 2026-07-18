"""Locked ICASSP 2027 study protocol.

This module contains the choices that must be frozen before outcome inspection:
patient splits, the independent-lead configuration universe, the neural benchmark
panel, primary sample rate/segments, and the bootstrap/rank grids.  Experiment
scripts import these values rather than defining local variants.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from ecgcert.physics.dipolar_subspace import INDEPENDENT_LEADS


PRIMARY_RATE_HZ = 500
PRIMARY_SEGMENTS = ("QRS", "ST", "T")
SUPPLEMENTARY_SEGMENTS = ("P",)
RANK_GRID = (2, 3, 4, 5)
BOOTSTRAP_REPLICATES = 2_000
CONFIG_PANEL_SALT = "ecgcert-icassp27-v1"


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

    def validate(self) -> None:
        if self.rate_hz != 500:
            raise ValueError("The locked primary protocol uses 500 Hz")
        if self.primary_segments != PRIMARY_SEGMENTS:
            raise ValueError(f"Primary segments must be {PRIMARY_SEGMENTS!r}")
        if self.rank_grid != RANK_GRID:
            raise ValueError(f"Rank grid must be {RANK_GRID!r}")
        if self.bootstrap_replicates < 2_000:
            raise ValueError("Main-paper uncertainty requires at least 2,000 bootstraps")


@dataclass(frozen=True)
class PatientSplit:
    """Patient-disjoint train/tune/calibration/test record identifiers."""

    train: tuple[int | str, ...]
    tune: tuple[int | str, ...]
    calibration: tuple[int | str, ...]
    test: tuple[int | str, ...]

    def validate(self) -> None:
        groups = {
            "train": set(self.train),
            "tune": set(self.tune),
            "calibration": set(self.calibration),
            "test": set(self.test),
        }
        names = tuple(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    sample = sorted(map(str, overlap))[:5]
                    raise ValueError(f"{left}/{right} leakage: {sample}")

    def sha256(self) -> str:
        payload = "|".join(
            f"{name}:{','.join(sorted(map(str, getattr(self, name))))}"
            for name in ("train", "tune", "calibration", "test")
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

    if "patient_id" in db.meta.columns:
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
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> PatientSplit:
    """Deterministically split an external cohort 60/20/20 by patient.

    The first partition is cohort-map fitting, the second is development/QC,
    and the third is the untouched zero-shot test.  ``calibration`` is empty
    because external outcomes are never used to fit the PTB-XL meta-model.
    """

    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three values summing to one")
    if any(x <= 0 for x in ratios):
        raise ValueError("all external split ratios must be positive")

    unique_patients = sorted({str(v) for v in record_to_patient.values()})
    patient_partition: dict[str, str] = {}
    cut1, cut2 = ratios[0], ratios[0] + ratios[1]
    for patient in unique_patients:
        digest = hashlib.sha256(f"{salt}|{patient}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        patient_partition[patient] = "train" if u < cut1 else "tune" if u < cut2 else "test"

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
    return panel


def configuration_panel_sha256(panel: Sequence[Sequence[str]] | None = None) -> str:
    panel = deep_configuration_panel() if panel is None else panel
    payload = "\n".join(",".join(canonical_configuration(config)) for config in panel)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
