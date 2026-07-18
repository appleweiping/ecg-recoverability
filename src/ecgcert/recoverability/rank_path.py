"""Rank paths, patient-cluster bootstrap, and robust recoverability envelopes."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from operator import index as integer_index
from typing import Iterable, Sequence

import numpy as np

from ecgcert.physics import (
    BASIS_VARIANTS,
    LEADS,
    LEAD_INDEX,
    RECON_RCOND,
    BasisVariant,
    SpatialSubspaceModel,
    eta_normalized_per_lead,
    eta_per_lead,
    fit_spatial_subspace,
    kappa,
    kappa_per_lead,
)
from ecgcert.recoverability.gaussian import gaussian_prior_ambiguity_per_lead

DEFAULT_RANK_GRID: tuple[int, ...] = (2, 3, 4, 5)
ModelKey = tuple[int, str, str]


def _readonly(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _lead_names(observed_leads) -> tuple[str, ...]:
    names: list[str] = []
    for lead in observed_leads:
        if isinstance(lead, str):
            if lead not in LEAD_INDEX:
                raise ValueError(f"unknown ECG lead {lead!r}")
            names.append(lead)
            continue
        try:
            lead_index = integer_index(lead)
        except TypeError as exc:
            raise ValueError(f"lead indices must be integers; got {lead!r}") from exc
        if not 0 <= lead_index < len(LEADS):
            raise IndexError(f"lead index must be in [0, {len(LEADS) - 1}]; got {lead_index}")
        names.append(LEADS[lead_index])
    canonical = tuple(names)
    if not canonical:
        raise ValueError("at least one observed lead is required")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"observed leads must be unique; got {canonical!r}")
    return canonical


@dataclass(frozen=True)
class RankPathEntry:
    """Recoverability quantities for one fitted model and observed-lead set."""

    rank: int
    basis_variant: str
    fit_cohort: str
    fit_ids: tuple[int, ...]
    observed_leads: tuple[str, ...]
    rcond: float | None
    effective_rank: int
    eta: np.ndarray
    eta_normalized: np.ndarray
    kappa_per_lead: np.ndarray
    kappa_global: float
    ambiguity: np.ndarray
    bootstrap_index: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("eta", "eta_normalized", "kappa_per_lead", "ambiguity"):
            value = np.asarray(getattr(self, field_name), dtype=float)
            if value.shape != (len(LEADS),):
                raise ValueError(f"{field_name} must be ({len(LEADS)},); got {value.shape}")
            object.__setattr__(self, field_name, _readonly(value))

    @property
    def model_key(self) -> ModelKey:
        return (self.rank, self.basis_variant, self.fit_cohort)


@dataclass(frozen=True)
class BootstrapRankPath:
    """Full-sample rank path plus patient-cluster bootstrap refits."""

    point: tuple[RankPathEntry, ...]
    replicates: tuple[RankPathEntry, ...]
    seed: int
    n_boot: int
    patient_ids: tuple[Hashable, ...]

    @property
    def record_ids(self) -> tuple[Hashable, ...]:
        """Deprecated compatibility alias; bootstrap clusters are patients."""

        return self.patient_ids


@dataclass(frozen=True)
class RecoverabilityEnvelope:
    """Conservative envelope across ranks, basis variants, cohorts, and bootstrap CIs."""

    observed_leads: tuple[str, ...]
    confidence: float
    eta_normalized_lower: np.ndarray
    eta_normalized_upper: np.ndarray
    kappa_lower: np.ndarray
    kappa_upper: np.ndarray
    ambiguity_lower: np.ndarray
    ambiguity_upper: np.ndarray
    recoverability_lower: np.ndarray
    model_sensitivity_span: np.ndarray
    kappa_global_upper: float
    worst_eta_member: tuple[ModelKey, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "eta_normalized_lower",
            "eta_normalized_upper",
            "kappa_lower",
            "kappa_upper",
            "ambiguity_lower",
            "ambiguity_upper",
            "recoverability_lower",
            "model_sensitivity_span",
        ):
            value = np.asarray(getattr(self, field_name), dtype=float)
            if value.shape != (len(LEADS),):
                raise ValueError(f"{field_name} must be ({len(LEADS)},); got {value.shape}")
            object.__setattr__(self, field_name, _readonly(value))


def evaluate_spatial_model(
    model: SpatialSubspaceModel,
    observed_leads,
    *,
    observation_variance_mv2: float,
    rcond: float | None,
    bootstrap_index: int | None = None,
) -> RankPathEntry:
    observed = _lead_names(observed_leads)
    eta = eta_per_lead(model.M, observed, rcond=rcond)
    eta_normalized = eta_normalized_per_lead(model.M, observed, rcond=rcond)
    per_lead_kappa = kappa_per_lead(model.M, observed, rcond=rcond)
    global_kappa, effective_rank = kappa(model.M, observed, rcond=rcond)
    ambiguity = gaussian_prior_ambiguity_per_lead(
        model,
        observed,
        observation_variance_mv2=observation_variance_mv2,
    )
    return RankPathEntry(
        rank=model.rank,
        basis_variant=model.basis_variant,
        fit_cohort=model.fit_cohort,
        fit_ids=model.fit_ids,
        observed_leads=observed,
        rcond=rcond,
        effective_rank=effective_rank,
        eta=eta,
        eta_normalized=eta_normalized,
        kappa_per_lead=per_lead_kappa,
        kappa_global=global_kappa,
        ambiguity=ambiguity,
        bootstrap_index=bootstrap_index,
    )


def compute_rank_path(
    models: Iterable[SpatialSubspaceModel],
    observed_leads,
    *,
    observation_variance_mv2: float,
    rcond: float | None = RECON_RCOND,
) -> tuple[RankPathEntry, ...]:
    """Compute one path entry for every supplied fitted model.

    The function does not select a preferred rank.  Callers should pre-register the
    candidate grid and aggregate it with :func:`aggregate_recoverability_envelope`.
    """

    models = tuple(models)
    if not models:
        raise ValueError("at least one SpatialSubspaceModel is required")
    keys = [(m.rank, m.basis_variant, m.fit_cohort) for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"model keys must be unique within a rank path; got {keys!r}")
    return tuple(
        evaluate_spatial_model(
            model,
            observed_leads,
            observation_variance_mv2=observation_variance_mv2,
            rcond=rcond,
        )
        for model in models
    )


def bootstrap_rank_path(
    X: np.ndarray,
    patient_ids: Sequence[Hashable],
    observed_leads,
    *,
    ranks: Sequence[int] = DEFAULT_RANK_GRID,
    basis_variants: Sequence[BasisVariant] = BASIS_VARIANTS,
    fit_cohort: str = "unspecified",
    n_boot: int = 2_000,
    seed: int = 0,
    observation_variance_mv2: float,
    rcond: float | None = RECON_RCOND,
) -> BootstrapRankPath:
    """Fit a rank path and refit it under a reproducible patient-cluster bootstrap.

    Every bootstrap draw samples whole patient ids with replacement and includes all
    rows belonging to each drawn id.  Repeated patients therefore repeat their entire
    row block rather than being collapsed to unique ids.
    """

    X = np.asarray(X, dtype=float)
    pid = np.asarray(patient_ids, dtype=object)
    if X.ndim != 2 or X.shape[1] != len(LEADS):
        raise ValueError(f"X must be (n_samples, {len(LEADS)}); got {X.shape}")
    if pid.ndim != 1 or pid.shape[0] != X.shape[0]:
        raise ValueError("patient_ids must be one-dimensional and aligned with X rows")
    for patient_id in pid:
        try:
            hash(patient_id)
        except TypeError as exc:
            raise ValueError(f"patient ids must be hashable; got {patient_id!r}") from exc
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}")
    ranks = tuple(int(rank) for rank in ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be a non-empty sequence of unique integers")
    variants = tuple(basis_variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("basis_variants must be a non-empty sequence of unique values")

    unique_ids = tuple(dict.fromkeys(pid.tolist()))
    if len(unique_ids) < 2:
        raise ValueError("patient-cluster bootstrap requires at least two unique patients")
    patient_code = {patient_id: code for code, patient_id in enumerate(unique_ids)}
    coded = np.asarray([patient_code[value] for value in pid], dtype=np.int64)
    unique_codes = np.arange(len(unique_ids), dtype=np.int64)
    row_lookup = {int(code): np.flatnonzero(coded == code) for code in unique_codes}

    def fit_models(samples: np.ndarray, fit_ids: Sequence[int]) -> tuple[SpatialSubspaceModel, ...]:
        return tuple(
            fit_spatial_subspace(
                samples,
                rank=rank,
                basis_variant=variant,
                fit_cohort=fit_cohort,
                fit_ids=fit_ids,
            )
            for variant in variants
            for rank in ranks
        )

    point_models = fit_models(X, unique_codes)
    point = compute_rank_path(
        point_models,
        observed_leads,
        observation_variance_mv2=observation_variance_mv2,
        rcond=rcond,
    )

    rng = np.random.default_rng(seed)
    replicates: list[RankPathEntry] = []
    for bootstrap_index in range(n_boot):
        drawn_ids = unique_codes[rng.integers(0, len(unique_codes), len(unique_codes))]
        rows = np.concatenate([row_lookup[int(code)] for code in drawn_ids])
        models = fit_models(X[rows], drawn_ids)
        replicates.extend(
            evaluate_spatial_model(
                model,
                observed_leads,
                observation_variance_mv2=observation_variance_mv2,
                rcond=rcond,
                bootstrap_index=bootstrap_index,
            )
            for model in models
        )

    return BootstrapRankPath(
        point=point,
        replicates=tuple(replicates),
        seed=int(seed),
        n_boot=n_boot,
        patient_ids=unique_ids,
    )


def _column_extreme(values: np.ndarray, fn) -> np.ndarray:
    out = np.full(values.shape[1], np.nan)
    for lead_index in range(values.shape[1]):
        finite = values[:, lead_index][np.isfinite(values[:, lead_index])]
        if finite.size:
            out[lead_index] = fn(finite)
    return out


def _column_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """Columnwise quantile that keeps structurally undefined leads as NaN."""

    out = np.full(values.shape[1], np.nan)
    for lead_index in range(values.shape[1]):
        finite = values[:, lead_index][np.isfinite(values[:, lead_index])]
        if finite.size:
            out[lead_index] = np.quantile(finite, quantile)
    return out


def aggregate_recoverability_envelope(
    path: BootstrapRankPath | Sequence[RankPathEntry],
    *,
    confidence: float = 0.95,
) -> RecoverabilityEnvelope:
    """Aggregate a conservative uncertainty envelope across all path members.

    With bootstrap input, each model's percentile interval is computed first and
    the outer envelope is then taken across model keys.  Full-sample point estimates
    are explicitly included, so the returned bounds always contain every rank-path
    point even when a percentile interval happens not to contain its point estimate.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    if isinstance(path, BootstrapRankPath):
        point = path.point
        replicates = path.replicates
    else:
        point = tuple(path)
        replicates = ()
    if not point:
        raise ValueError("cannot aggregate an empty rank path")
    observed = point[0].observed_leads
    if any(entry.observed_leads != observed for entry in point + tuple(replicates)):
        raise ValueError("all rank-path entries must use the same observed leads")

    grouped: dict[ModelKey, list[RankPathEntry]] = {entry.model_key: [] for entry in point}
    for entry in replicates:
        if entry.model_key not in grouped:
            raise ValueError(f"bootstrap entry has no matching point model: {entry.model_key!r}")
        grouped[entry.model_key].append(entry)

    alpha = (1.0 - confidence) / 2.0
    eta_lo, eta_hi, kap_lo, kap_hi, amb_lo, amb_hi = [], [], [], [], [], []
    global_kappa_hi: list[float] = []
    keys: list[ModelKey] = []
    for entry in point:
        reps = grouped[entry.model_key]
        keys.append(entry.model_key)
        if reps:
            eta_values = np.stack([r.eta_normalized for r in reps])
            kap_values = np.stack([r.kappa_per_lead for r in reps])
            amb_values = np.stack([r.ambiguity for r in reps])
            eta_lo.append(np.fmin(_column_quantile(eta_values, alpha), entry.eta_normalized))
            eta_hi.append(
                np.fmax(_column_quantile(eta_values, 1.0 - alpha), entry.eta_normalized)
            )
            kap_lo.append(np.fmin(_column_quantile(kap_values, alpha), entry.kappa_per_lead))
            kap_hi.append(
                np.fmax(_column_quantile(kap_values, 1.0 - alpha), entry.kappa_per_lead)
            )
            amb_lo.append(np.fmin(_column_quantile(amb_values, alpha), entry.ambiguity))
            amb_hi.append(
                np.fmax(_column_quantile(amb_values, 1.0 - alpha), entry.ambiguity)
            )
            global_kappa_hi.append(
                max(
                    float(np.nanquantile([r.kappa_global for r in reps], 1.0 - alpha)),
                    entry.kappa_global,
                )
            )
        else:
            eta_lo.append(entry.eta_normalized)
            eta_hi.append(entry.eta_normalized)
            kap_lo.append(entry.kappa_per_lead)
            kap_hi.append(entry.kappa_per_lead)
            amb_lo.append(entry.ambiguity)
            amb_hi.append(entry.ambiguity)
            global_kappa_hi.append(entry.kappa_global)

    point_eta = np.stack([entry.eta_normalized for entry in point])
    point_kappa = np.stack([entry.kappa_per_lead for entry in point])
    point_ambiguity = np.stack([entry.ambiguity for entry in point])
    eta_lower_candidates = np.vstack((np.stack(eta_lo), point_eta))
    eta_upper_candidates = np.vstack((np.stack(eta_hi), point_eta))
    kappa_lower_candidates = np.vstack((np.stack(kap_lo), point_kappa))
    kappa_upper_candidates = np.vstack((np.stack(kap_hi), point_kappa))
    ambiguity_lower_candidates = np.vstack((np.stack(amb_lo), point_ambiguity))
    ambiguity_upper_candidates = np.vstack((np.stack(amb_hi), point_ambiguity))

    eta_lower = _column_extreme(eta_lower_candidates, np.min)
    eta_upper = _column_extreme(eta_upper_candidates, np.max)
    kappa_lower = _column_extreme(kappa_lower_candidates, np.min)
    kappa_upper = _column_extreme(kappa_upper_candidates, np.max)
    ambiguity_lower = _column_extreme(ambiguity_lower_candidates, np.min)
    ambiguity_upper = _column_extreme(ambiguity_upper_candidates, np.max)
    sensitivity = _column_extreme(point_eta, np.max) - _column_extreme(point_eta, np.min)

    eta_hi_by_model = np.stack(eta_hi)
    worst_members: list[ModelKey] = []
    for lead_index in range(len(LEADS)):
        column = eta_hi_by_model[:, lead_index]
        if np.any(np.isfinite(column)):
            worst_members.append(keys[int(np.nanargmax(column))])
        else:
            worst_members.append(keys[0])

    return RecoverabilityEnvelope(
        observed_leads=observed,
        confidence=confidence,
        eta_normalized_lower=eta_lower,
        eta_normalized_upper=eta_upper,
        kappa_lower=kappa_lower,
        kappa_upper=kappa_upper,
        ambiguity_lower=ambiguity_lower,
        ambiguity_upper=ambiguity_upper,
        recoverability_lower=np.clip(1.0 - eta_upper, 0.0, 1.0),
        model_sensitivity_span=sensitivity,
        kappa_global_upper=float(max(global_kappa_hi)),
        worst_eta_member=tuple(worst_members),
    )


__all__ = [
    "DEFAULT_RANK_GRID",
    "BootstrapRankPath",
    "RankPathEntry",
    "RecoverabilityEnvelope",
    "aggregate_recoverability_envelope",
    "bootstrap_rank_path",
    "compute_rank_path",
    "evaluate_spatial_model",
]
