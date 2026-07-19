"""Reusable patient-cluster bootstrap banks for spatial-subspace models.

The expensive unit of resampling is the fitted spatial model, not an observed-lead
configuration.  This module therefore fits every rank/basis model once per patient
bootstrap draw from sufficient statistics.  Any number of observed configurations
can subsequently consume the same bank without refitting PCA models.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from operator import index as integer_index

import numpy as np

from ecgcert.physics import (
    BASIS_VARIANTS,
    INDEPENDENT_LEADS,
    LEADS,
    LEAD_INDEX,
    RECON_RCOND,
    BasisVariant,
    SpatialSubspaceModel,
    lead_transform_T,
)
from ecgcert.recoverability.rank_path import (
    DEFAULT_RANK_GRID,
    BootstrapRankPath,
    compute_rank_path,
    evaluate_spatial_model,
)


BOOTSTRAP_MOMENT_BATCH_SIZE = 64


def _patient_key(patient_id: Hashable) -> tuple[type, Hashable]:
    """Validate an id and retain type information in its internal codebook key."""

    if patient_id is None:
        raise ValueError("patient ids cannot be missing (None)")
    if isinstance(patient_id, (bool, np.bool_)):
        raise ValueError("boolean patient ids are ambiguous with integer ids")
    try:
        hash(patient_id)
    except TypeError as exc:
        raise ValueError(f"patient id must be hashable; got {patient_id!r}") from exc
    try:
        reflexive = patient_id == patient_id
    except Exception as exc:
        raise ValueError(f"patient id has invalid equality semantics: {patient_id!r}") from exc
    if not isinstance(reflexive, (bool, np.bool_)) or not bool(reflexive):
        raise ValueError(f"patient id must be non-missing and reflexive; got {patient_id!r}")
    return type(patient_id), patient_id


@dataclass(frozen=True)
class PatientClusterSufficientStatistics:
    """Per-patient raw moments in stable first-appearance order.

    The moments use ``Y = X - origin``: ``sums[p]`` is ``sum(Y_p)`` and
    ``crossproducts[p]`` is ``Y_p.T @ Y_p``.  A shared origin leaves PCA unchanged
    while avoiding catastrophic cancellation for signals with a large common offset.
    Arrays are immutable owned copies so a cached bank cannot drift with caller data.
    """

    patient_ids: tuple[Hashable, ...]
    origin: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    crossproducts: np.ndarray

    def __post_init__(self) -> None:
        patient_ids = tuple(self.patient_ids)
        if len(patient_ids) < 2:
            raise ValueError("patient-cluster bootstrap requires at least two patients")
        seen: dict[tuple[type, Hashable], None] = {}
        for patient_id in patient_ids:
            key = _patient_key(patient_id)
            if key in seen:
                raise ValueError(f"patient_ids must be unique; duplicate {patient_id!r}")
            seen[key] = None

        origin = np.array(self.origin, dtype=float, copy=True)
        supplied_counts = np.asarray(self.counts)
        if not np.issubdtype(supplied_counts.dtype, np.integer):
            raise ValueError("counts must contain integers")
        counts = np.array(supplied_counts, dtype=np.int64, copy=True)
        sums = np.array(self.sums, dtype=float, copy=True)
        crossproducts = np.array(self.crossproducts, dtype=float, copy=True)
        n_patients = len(patient_ids)
        if origin.shape != (len(LEADS),):
            raise ValueError(f"origin must be ({len(LEADS)},); got {origin.shape}")
        if counts.shape != (n_patients,):
            raise ValueError(f"counts must be ({n_patients},); got {counts.shape}")
        if sums.shape != (n_patients, len(LEADS)):
            raise ValueError(
                f"sums must be ({n_patients}, {len(LEADS)}); got {sums.shape}"
            )
        expected_cross_shape = (n_patients, len(LEADS), len(LEADS))
        if crossproducts.shape != expected_cross_shape:
            raise ValueError(
                f"crossproducts must be {expected_cross_shape}; got {crossproducts.shape}"
            )
        if np.any(counts < 1):
            raise ValueError("every patient must contribute at least one sample")
        if (
            not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(sums))
            or not np.all(np.isfinite(crossproducts))
        ):
            raise ValueError("patient sufficient statistics must be finite")
        if not np.allclose(
            crossproducts,
            np.swapaxes(crossproducts, 1, 2),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("patient crossproducts must be symmetric")

        origin.setflags(write=False)
        counts.setflags(write=False)
        sums.setflags(write=False)
        crossproducts.setflags(write=False)
        object.__setattr__(self, "patient_ids", patient_ids)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "sums", sums)
        object.__setattr__(self, "crossproducts", crossproducts)

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    @property
    def n_samples(self) -> int:
        return int(self.counts.sum())


@dataclass(frozen=True)
class PatientBootstrapAttemptLedger:
    """Immutable audit trail for every proposed patient-bootstrap draw.

    Rows retain proposal order, including draws rejected because their centred
    scatter cannot support the preregistered rank grid.  Accepted indices are
    dense and zero based; rejected attempts use ``-1``.  Multiplicities use the
    narrowest lossless unsigned representation so the complete ledger remains
    practical for large patient cohorts.
    """

    multiplicities: np.ndarray
    accepted: np.ndarray
    accepted_bootstrap_index: np.ndarray

    def __post_init__(self) -> None:
        supplied_multiplicities = np.asarray(self.multiplicities)
        if supplied_multiplicities.ndim != 2:
            raise ValueError("attempt multiplicities must be a two-dimensional array")
        n_attempts, n_patients = supplied_multiplicities.shape
        if n_attempts < 1 or n_patients < 2:
            raise ValueError("attempt ledger requires at least one draw and two patients")
        if not np.issubdtype(supplied_multiplicities.dtype, np.integer):
            raise ValueError("attempt multiplicities must contain integers")
        if np.any(supplied_multiplicities < 0):
            raise ValueError("attempt multiplicities cannot be negative")
        if n_patients > np.iinfo(np.uint32).max:
            raise ValueError("patient count exceeds the uint32 multiplicity contract")
        multiplicity_dtype = (
            np.uint16 if n_patients <= np.iinfo(np.uint16).max else np.uint32
        )
        multiplicities = np.array(
            supplied_multiplicities,
            dtype=multiplicity_dtype,
            copy=True,
        )
        if not np.all(multiplicities.sum(axis=1, dtype=np.uint64) == n_patients):
            raise ValueError("each attempted draw must contain n_patients clusters")

        supplied_accepted = np.asarray(self.accepted)
        if supplied_accepted.dtype != np.bool_ or supplied_accepted.shape != (n_attempts,):
            raise ValueError("accepted must be a one-dimensional boolean array")
        accepted = np.array(supplied_accepted, dtype=np.bool_, copy=True)
        if not bool(accepted[-1]):
            raise ValueError("the final attempt must be accepted when bootstrap sampling stops")

        supplied_indices = np.asarray(self.accepted_bootstrap_index)
        if not np.issubdtype(supplied_indices.dtype, np.integer):
            raise ValueError("accepted_bootstrap_index must contain integers")
        if supplied_indices.shape != (n_attempts,):
            raise ValueError(
                "accepted_bootstrap_index must be aligned with attempted draws"
            )
        if int(accepted.sum()) > np.iinfo(np.int32).max:
            raise ValueError("accepted draw count exceeds the int32 index contract")
        accepted_bootstrap_index = np.array(
            supplied_indices,
            dtype=np.int32,
            copy=True,
        )
        if np.any(accepted_bootstrap_index[~accepted] != -1):
            raise ValueError("rejected attempts must use accepted_bootstrap_index=-1")
        expected_accepted_indices = np.arange(int(accepted.sum()), dtype=np.int32)
        if not np.array_equal(
            accepted_bootstrap_index[accepted],
            expected_accepted_indices,
        ):
            raise ValueError(
                "accepted bootstrap indices must be dense and follow attempt order"
            )

        multiplicities.setflags(write=False)
        accepted.setflags(write=False)
        accepted_bootstrap_index.setflags(write=False)
        object.__setattr__(self, "multiplicities", multiplicities)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(
            self,
            "accepted_bootstrap_index",
            accepted_bootstrap_index,
        )

    @property
    def n_attempts(self) -> int:
        return int(self.multiplicities.shape[0])

    @property
    def n_patients(self) -> int:
        return int(self.multiplicities.shape[1])

    @property
    def n_boot(self) -> int:
        return int(self.accepted.sum())

    @property
    def rejected_draws(self) -> int:
        return self.n_attempts - self.n_boot

    @property
    def accepted_multiplicities(self) -> np.ndarray:
        """Return accepted rows in bootstrap-index order as an immutable copy."""

        result = np.array(self.multiplicities[self.accepted], copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class PatientBootstrapModelBank:
    """Point models and bootstrap models grouped by patient-cluster draw.

    The row ``bootstrap_multiplicities[b, p]`` records how often patient ``p``
    appears in bootstrap draw ``b``.  Patient ids live once in ``statistics``;
    bootstrap model ``fit_ids`` are intentionally empty to avoid copying a long id
    tuple into every rank/variant model.
    """

    statistics: PatientClusterSufficientStatistics
    point_models: tuple[SpatialSubspaceModel, ...]
    bootstrap_models: tuple[tuple[SpatialSubspaceModel, ...], ...]
    bootstrap_multiplicities: np.ndarray
    ranks: tuple[int, ...]
    basis_variants: tuple[BasisVariant, ...]
    fit_cohort: str
    seed: int
    attempt_ledger: PatientBootstrapAttemptLedger
    rejected_draws: int = 0

    def __post_init__(self) -> None:
        point_models = tuple(self.point_models)
        bootstrap_models = tuple(tuple(group) for group in self.bootstrap_models)
        ranks = tuple(self.ranks)
        variants = tuple(self.basis_variants)
        if (
            isinstance(self.rejected_draws, bool)
            or not isinstance(self.rejected_draws, int)
            or self.rejected_draws < 0
        ):
            raise ValueError("rejected_draws must be a non-negative integer")
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer, not a boolean")
        try:
            normalized_seed = integer_index(self.seed)
        except TypeError as exc:
            raise ValueError("seed must be an integer") from exc
        expected_models = len(ranks) * len(variants)
        if len(point_models) != expected_models:
            raise ValueError(
                f"point_models must contain {expected_models} rank/variant models"
            )
        if any(len(group) != expected_models for group in bootstrap_models):
            raise ValueError(
                f"every bootstrap group must contain {expected_models} rank/variant models"
            )

        multiplicities = np.array(self.bootstrap_multiplicities, copy=True)
        expected_shape = (len(bootstrap_models), self.statistics.n_patients)
        if multiplicities.shape != expected_shape:
            raise ValueError(
                f"bootstrap_multiplicities must be {expected_shape}; got {multiplicities.shape}"
            )
        if not np.issubdtype(multiplicities.dtype, np.integer):
            raise ValueError("bootstrap_multiplicities must be an integer array")
        if np.any(multiplicities < 0):
            raise ValueError("bootstrap multiplicities cannot be negative")
        if not np.all(multiplicities.sum(axis=1) == self.statistics.n_patients):
            raise ValueError("each bootstrap draw must contain n_patients clusters")
        if not isinstance(self.attempt_ledger, PatientBootstrapAttemptLedger):
            raise ValueError("attempt_ledger must be a PatientBootstrapAttemptLedger")
        if self.attempt_ledger.n_patients != self.statistics.n_patients:
            raise ValueError("attempt ledger patient dimension does not match statistics")
        if self.attempt_ledger.n_boot != len(bootstrap_models):
            raise ValueError("attempt ledger accepted count does not match model groups")
        if self.attempt_ledger.rejected_draws != self.rejected_draws:
            raise ValueError("attempt ledger rejected count does not match rejected_draws")
        if not np.array_equal(
            multiplicities,
            self.attempt_ledger.accepted_multiplicities,
        ):
            raise ValueError(
                "accepted bootstrap multiplicities do not match the attempt ledger"
            )

        expected_keys = tuple(
            (variant, rank) for variant in variants for rank in ranks
        )
        point_keys = tuple((model.basis_variant, model.rank) for model in point_models)
        if point_keys != expected_keys:
            raise ValueError("point model ordering does not match basis_variants x ranks")
        for group in bootstrap_models:
            group_keys = tuple((model.basis_variant, model.rank) for model in group)
            if group_keys != expected_keys:
                raise ValueError("bootstrap model ordering does not match point models")

        multiplicities.setflags(write=False)
        object.__setattr__(self, "point_models", point_models)
        object.__setattr__(self, "bootstrap_models", bootstrap_models)
        object.__setattr__(self, "bootstrap_multiplicities", multiplicities)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "basis_variants", variants)
        object.__setattr__(self, "seed", normalized_seed)

    @property
    def patient_ids(self) -> tuple[Hashable, ...]:
        return self.statistics.patient_ids

    @property
    def n_boot(self) -> int:
        return len(self.bootstrap_models)

    @property
    def rejection_fraction(self) -> float:
        attempts = self.n_boot + self.rejected_draws
        return self.rejected_draws / attempts if attempts else 0.0

    def fit_patient_ids(self, bootstrap_index: int | None = None) -> tuple[Hashable, ...]:
        """Return the original ids in a point fit or bootstrap multiset."""

        if bootstrap_index is None:
            return self.patient_ids
        if isinstance(bootstrap_index, bool):
            raise IndexError("bootstrap_index must be an integer, not a boolean")
        try:
            normalized_index = integer_index(bootstrap_index)
        except TypeError as exc:
            raise IndexError("bootstrap_index must be an integer") from exc
        if not 0 <= normalized_index < self.n_boot:
            raise IndexError(
                f"bootstrap_index must be in [0, {self.n_boot - 1}]; got {bootstrap_index}"
            )
        row = self.bootstrap_multiplicities[normalized_index]
        return tuple(
            patient_id
            for patient_id, multiplicity in zip(self.patient_ids, row)
            for _ in range(int(multiplicity))
        )


def cache_patient_cluster_statistics(
    X: np.ndarray,
    patient_ids: Iterable[Hashable],
) -> PatientClusterSufficientStatistics:
    """Cache ``count``, ``sum``, and ``crossproduct`` once for each patient."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(LEADS):
        raise ValueError(f"X must be (n_samples, {len(LEADS)}); got {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    row_patient_ids = tuple(patient_ids)
    if len(row_patient_ids) != X.shape[0]:
        raise ValueError("patient_ids must be one-dimensional and aligned with X rows")

    ordered_ids: list[Hashable] = []
    code_by_id: dict[tuple[type, Hashable], int] = {}
    row_groups: list[list[int]] = []
    for row_index, patient_id in enumerate(row_patient_ids):
        key = _patient_key(patient_id)
        if key not in code_by_id:
            code_by_id[key] = len(ordered_ids)
            ordered_ids.append(patient_id)
            row_groups.append([])
        row_groups[code_by_id[key]].append(row_index)

    n_patients = len(ordered_ids)
    origin = X.mean(axis=0)
    shifted_X = X - origin
    counts = np.empty(n_patients, dtype=np.int64)
    sums = np.empty((n_patients, len(LEADS)), dtype=float)
    crossproducts = np.empty((n_patients, len(LEADS), len(LEADS)), dtype=float)
    for patient_code, rows in enumerate(row_groups):
        patient_X = shifted_X[np.asarray(rows, dtype=int)]
        counts[patient_code] = patient_X.shape[0]
        sums[patient_code] = patient_X.sum(axis=0)
        crossproducts[patient_code] = patient_X.T @ patient_X

    return PatientClusterSufficientStatistics(
        patient_ids=tuple(ordered_ids),
        origin=origin,
        counts=counts,
        sums=sums,
        crossproducts=crossproducts,
    )


def _canonicalize_column_signs(basis: np.ndarray) -> np.ndarray:
    basis = np.array(basis, dtype=float, copy=True)
    for column_index in range(basis.shape[1]):
        column = basis[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            basis[:, column_index] *= -1.0
    return basis


class _InsufficientSpatialRank(ValueError):
    """Internal signal that a bootstrap draw cannot support the rank grid."""


def _leading_eigenvectors(scatter: np.ndarray, rank: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scatter + scatter.T))
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    ordered_eigenvalues = eigenvalues[order]
    leading_value = float(ordered_eigenvalues[0])
    scale = max(leading_value, abs(float(np.trace(scatter))) / scatter.shape[0], 0.0)
    tolerance = np.finfo(float).eps * max(scatter.shape) * scale
    if ordered_eigenvalues[-1] < -100.0 * tolerance:
        raise ValueError("centred scatter is materially non-positive-semidefinite")
    if leading_value <= 0.0 or ordered_eigenvalues[rank - 1] <= tolerance:
        effective_rank = int(np.count_nonzero(ordered_eigenvalues > tolerance))
        raise _InsufficientSpatialRank(
            f"centred scatter rank {effective_rank} cannot support requested rank {rank}"
        )
    return _canonicalize_column_signs(eigenvectors[:, order[:rank]])


def _coordinate_covariance(
    basis: np.ndarray,
    scatter: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    covariance = basis.T @ scatter @ basis / float(sample_count - 1)
    return 0.5 * (covariance + covariance.T)


def _models_from_moments(
    sample_count: int,
    sample_sum: np.ndarray,
    sample_crossproduct: np.ndarray,
    *,
    origin: np.ndarray,
    ranks: tuple[int, ...],
    basis_variants: tuple[BasisVariant, ...],
    fit_cohort: str,
    fit_ids: Sequence[int] = (),
) -> tuple[SpatialSubspaceModel, ...]:
    mean12 = origin + sample_sum / float(sample_count)
    scatter12 = sample_crossproduct - np.outer(sample_sum, sample_sum) / float(sample_count)
    scatter12 = 0.5 * (scatter12 + scatter12.T)
    max_rank = max(ranks)
    models: list[SpatialSubspaceModel] = []

    raw_basis = None
    independent_basis = None
    independent_scatter = None
    independent_mean = None
    lifted_basis = None
    transform = None
    for variant in basis_variants:
        if variant == "raw12_pca":
            if raw_basis is None:
                raw_basis = _leading_eigenvectors(scatter12, max_rank)
            for rank in ranks:
                basis = raw_basis[:, :rank]
                models.append(
                    SpatialSubspaceModel(
                        rank=rank,
                        basis_variant=variant,
                        fit_cohort=fit_cohort,
                        fit_ids=tuple(fit_ids),
                        M=basis,
                        mu=mean12,
                        covariance=_coordinate_covariance(basis, scatter12, sample_count),
                    )
                )
            continue

        if independent_basis is None:
            independent_indices = [LEAD_INDEX[lead] for lead in INDEPENDENT_LEADS]
            independent_sum = sample_sum[independent_indices]
            independent_crossproduct = sample_crossproduct[np.ix_(
                independent_indices,
                independent_indices,
            )]
            independent_indices_array = np.asarray(independent_indices, dtype=int)
            independent_mean = (
                origin[independent_indices_array]
                + independent_sum / float(sample_count)
            )
            independent_scatter = independent_crossproduct - (
                np.outer(independent_sum, independent_sum) / float(sample_count)
            )
            independent_scatter = 0.5 * (
                independent_scatter + independent_scatter.T
            )
            independent_basis = _leading_eigenvectors(independent_scatter, max_rank)
            transform = lead_transform_T()
            lifted_basis, _ = np.linalg.qr(
                transform @ independent_basis,
                mode="reduced",
            )
            lifted_basis = _canonicalize_column_signs(lifted_basis)

        assert independent_scatter is not None
        assert independent_mean is not None
        assert transform is not None
        assert lifted_basis is not None
        modeled_scatter = transform @ independent_scatter @ transform.T
        modeled_mean = transform @ independent_mean
        for rank in ranks:
            basis = lifted_basis[:, :rank]
            models.append(
                SpatialSubspaceModel(
                    rank=rank,
                    basis_variant=variant,
                    fit_cohort=fit_cohort,
                    fit_ids=tuple(fit_ids),
                    M=basis,
                    mu=modeled_mean,
                    covariance=_coordinate_covariance(
                        basis,
                        modeled_scatter,
                        sample_count,
                    ),
                )
            )
    return tuple(models)


def _validated_grid(
    statistics: PatientClusterSufficientStatistics,
    ranks: Sequence[int],
    basis_variants: Sequence[BasisVariant],
) -> tuple[tuple[int, ...], tuple[BasisVariant, ...]]:
    supplied_ranks = tuple(ranks)
    if any(isinstance(rank, (bool, np.bool_)) for rank in supplied_ranks):
        raise ValueError("ranks must contain integers, not booleans")
    try:
        normalized_ranks = tuple(integer_index(rank) for rank in supplied_ranks)
    except TypeError as exc:
        raise ValueError("ranks must contain integers") from exc
    variants = tuple(basis_variants)
    if not normalized_ranks or len(set(normalized_ranks)) != len(normalized_ranks):
        raise ValueError("ranks must be a non-empty sequence of unique integers")
    if any(rank < 1 for rank in normalized_ranks):
        raise ValueError("ranks must be positive")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("basis_variants must be a non-empty sequence of unique values")
    if any(variant not in BASIS_VARIANTS for variant in variants):
        raise ValueError(f"basis_variants must be selected from {BASIS_VARIANTS}")

    max_rank = max(normalized_ranks)
    dimension_limit = min(
        len(INDEPENDENT_LEADS) if "independent8_lifted" in variants else len(LEADS),
        len(LEADS),
    )
    if max_rank > dimension_limit:
        raise ValueError(f"rank {max_rank} exceeds basis dimension {dimension_limit}")
    minimum_bootstrap_samples = statistics.n_patients * int(statistics.counts.min())
    if max_rank > minimum_bootstrap_samples - 1:
        raise ValueError(
            "requested rank can exceed the centred sample dimension in a bootstrap draw"
        )
    return normalized_ranks, variants


def rebuild_spatial_model_bank(
    statistics: PatientClusterSufficientStatistics,
    attempt_ledger: PatientBootstrapAttemptLedger,
    *,
    ranks: Sequence[int] = DEFAULT_RANK_GRID,
    basis_variants: Sequence[BasisVariant] = BASIS_VARIANTS,
    fit_cohort: str = "unspecified",
    seed: int = 0,
    verify_rng_sequence: bool = True,
) -> PatientBootstrapModelBank:
    """Rebuild and authenticate a model bank from moments and its attempt ledger.

    The complete proposal sequence is authoritative evidence, but it is also
    checked against ``numpy.default_rng(seed)`` by default.  Every accepted row
    must produce the requested model grid and every rejected row must reproduce
    the same rank-deficiency decision.  This is the shared replay primitive used
    by artifact readers and evidence validators.
    """

    if not isinstance(statistics, PatientClusterSufficientStatistics):
        raise ValueError("statistics must be PatientClusterSufficientStatistics")
    if not isinstance(attempt_ledger, PatientBootstrapAttemptLedger):
        raise ValueError("attempt_ledger must be PatientBootstrapAttemptLedger")
    if attempt_ledger.n_patients != statistics.n_patients:
        raise ValueError("attempt ledger patient dimension does not match statistics")
    normalized_ranks, variants = _validated_grid(statistics, ranks, basis_variants)
    if isinstance(seed, bool):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    try:
        normalized_seed = integer_index(seed)
    except TypeError as exc:
        raise ValueError(f"seed must be an integer; got {seed!r}") from exc
    if not isinstance(verify_rng_sequence, bool):
        raise ValueError("verify_rng_sequence must be boolean")

    if verify_rng_sequence:
        rng = np.random.default_rng(normalized_seed)
        for attempt_index, recorded in enumerate(attempt_ledger.multiplicities):
            drawn_codes = rng.integers(
                0,
                statistics.n_patients,
                size=statistics.n_patients,
            )
            expected = np.bincount(
                drawn_codes,
                minlength=statistics.n_patients,
            )
            if not np.array_equal(recorded, expected):
                raise ValueError(
                    "attempt multiplicities do not match the declared RNG seed "
                    f"at attempt {attempt_index}"
                )

    try:
        point_models = _models_from_moments(
            statistics.n_samples,
            statistics.sums.sum(axis=0),
            statistics.crossproducts.sum(axis=0),
            origin=statistics.origin,
            ranks=normalized_ranks,
            basis_variants=variants,
            fit_cohort=fit_cohort,
            fit_ids=tuple(range(statistics.n_patients)),
        )
    except _InsufficientSpatialRank as exc:
        raise ValueError(f"point sample cannot support the requested rank grid: {exc}") from exc

    accepted_models: list[tuple[SpatialSubspaceModel, ...] | None] = [
        None
    ] * attempt_ledger.n_boot
    attempt_start = 0
    accepted_before_batch = 0
    while attempt_start < attempt_ledger.n_attempts:
        candidate_count = min(
            BOOTSTRAP_MOMENT_BATCH_SIZE,
            attempt_ledger.n_boot - accepted_before_batch,
        )
        attempt_stop = attempt_start + candidate_count
        if candidate_count < 1 or attempt_stop > attempt_ledger.n_attempts:
            raise ValueError("attempt ledger cannot arise from the declared batch sampler")
        batch_multiplicities = attempt_ledger.multiplicities[
            attempt_start:attempt_stop
        ]
        weights = batch_multiplicities.astype(float)
        sample_counts = weights @ statistics.counts
        sample_sums = weights @ statistics.sums
        sample_crossproducts = np.tensordot(
            weights,
            statistics.crossproducts,
            axes=(1, 0),
        )
        for local_index in range(candidate_count):
            attempt_index = attempt_start + local_index
            try:
                models = _models_from_moments(
                    int(round(float(sample_counts[local_index]))),
                    sample_sums[local_index],
                    sample_crossproducts[local_index],
                    origin=statistics.origin,
                    ranks=normalized_ranks,
                    basis_variants=variants,
                    fit_cohort=fit_cohort,
                )
            except _InsufficientSpatialRank as exc:
                if bool(attempt_ledger.accepted[attempt_index]):
                    raise ValueError(
                        f"attempt {attempt_index} is marked accepted but replays as rank deficient"
                    ) from exc
                continue
            if not bool(attempt_ledger.accepted[attempt_index]):
                raise ValueError(
                    f"attempt {attempt_index} is marked rejected but replays as accepted"
                )
            bootstrap_index = int(
                attempt_ledger.accepted_bootstrap_index[attempt_index]
            )
            accepted_models[bootstrap_index] = models
        accepted_before_batch += int(
            attempt_ledger.accepted[attempt_start:attempt_stop].sum()
        )
        attempt_start = attempt_stop

    if any(models is None for models in accepted_models):  # pragma: no cover - ledger guards.
        raise ValueError("attempt ledger does not reconstruct every accepted model")
    bootstrap_models = tuple(
        models for models in accepted_models if models is not None
    )
    return PatientBootstrapModelBank(
        statistics=statistics,
        point_models=point_models,
        bootstrap_models=bootstrap_models,
        bootstrap_multiplicities=attempt_ledger.accepted_multiplicities,
        ranks=normalized_ranks,
        basis_variants=variants,
        fit_cohort=fit_cohort,
        seed=normalized_seed,
        attempt_ledger=attempt_ledger,
        rejected_draws=attempt_ledger.rejected_draws,
    )


def bootstrap_spatial_model_bank(
    data: np.ndarray | PatientClusterSufficientStatistics,
    patient_ids: Iterable[Hashable] | None = None,
    *,
    ranks: Sequence[int] = DEFAULT_RANK_GRID,
    basis_variants: Sequence[BasisVariant] = BASIS_VARIANTS,
    fit_cohort: str = "unspecified",
    n_boot: int = 2000,
    seed: int = 0,
) -> PatientBootstrapModelBank:
    """Build one sufficient-statistics model bank reusable by all configurations.

    Each bootstrap draw samples ``n_patients`` patient clusters with replacement.
    ``data`` may be raw ``X`` (with aligned ``patient_ids``) or a previously cached
    :class:`PatientClusterSufficientStatistics` object.  The latter path never reads
    or aggregates the raw sample matrix again.
    Only fixed-origin sufficient statistics are combined; every accepted draw performs at
    most one ``12 x 12`` and one ``8 x 8`` eigendecomposition, independent of rank-grid size.
    Numerically rank-deficient draws are deterministically rejected and redrawn;
    ``rejected_draws`` records this conditioning of the ordinary cluster bootstrap.
    """

    if isinstance(data, PatientClusterSufficientStatistics):
        if patient_ids is not None:
            raise ValueError("patient_ids must be omitted when data is cached statistics")
        statistics = data
    else:
        if patient_ids is None:
            raise ValueError("patient_ids are required when data is a raw sample matrix")
        statistics = cache_patient_cluster_statistics(data, patient_ids)
    normalized_ranks, variants = _validated_grid(statistics, ranks, basis_variants)
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}")
    if isinstance(seed, bool):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    try:
        normalized_seed = integer_index(seed)
    except TypeError as exc:
        raise ValueError(f"seed must be an integer; got {seed!r}") from exc

    n_patients = statistics.n_patients
    multiplicity_dtype = np.uint16 if n_patients <= np.iinfo(np.uint16).max else np.uint32
    multiplicities = np.empty((n_boot, n_patients), dtype=multiplicity_dtype)
    rng = np.random.default_rng(normalized_seed)
    try:
        point_models = _models_from_moments(
            statistics.n_samples,
            statistics.sums.sum(axis=0),
            statistics.crossproducts.sum(axis=0),
            origin=statistics.origin,
            ranks=normalized_ranks,
            basis_variants=variants,
            fit_cohort=fit_cohort,
            fit_ids=tuple(range(n_patients)),
        )
    except _InsufficientSpatialRank as exc:
        raise ValueError(f"point sample cannot support the requested rank grid: {exc}") from exc

    bootstrap_groups: list[tuple[SpatialSubspaceModel, ...]] = []
    attempt_multiplicity_chunks: list[np.ndarray] = []
    attempt_acceptance_chunks: list[np.ndarray] = []
    attempt_index_chunks: list[np.ndarray] = []
    # Batch moment aggregation uses dense BLAS while bounding temporary memory for
    # large cohorts.  The eigendecompositions remain one pair per bootstrap draw.
    accepted_draws = 0
    rejected_draws = 0
    maximum_attempts = max(1000, 100 * n_boot)
    while accepted_draws < n_boot:
        candidate_count = min(BOOTSTRAP_MOMENT_BATCH_SIZE, n_boot - accepted_draws)
        candidate_multiplicities = np.empty(
            (candidate_count, n_patients),
            dtype=multiplicity_dtype,
        )
        for candidate_index in range(candidate_count):
            drawn_codes = rng.integers(0, n_patients, size=n_patients)
            candidate_multiplicities[candidate_index] = np.bincount(
                drawn_codes,
                minlength=n_patients,
            )

        weights = candidate_multiplicities.astype(float)
        sample_counts = weights @ statistics.counts
        sample_sums = weights @ statistics.sums
        sample_crossproducts = np.tensordot(
            weights,
            statistics.crossproducts,
            axes=(1, 0),
        )
        candidate_accepted = np.zeros(candidate_count, dtype=np.bool_)
        candidate_accepted_indices = np.full(candidate_count, -1, dtype=np.int32)
        for candidate_index in range(candidate_count):
            try:
                models = _models_from_moments(
                    int(round(float(sample_counts[candidate_index]))),
                    sample_sums[candidate_index],
                    sample_crossproducts[candidate_index],
                    origin=statistics.origin,
                    ranks=normalized_ranks,
                    basis_variants=variants,
                    fit_cohort=fit_cohort,
                )
            except _InsufficientSpatialRank:
                rejected_draws += 1
                if accepted_draws + rejected_draws >= maximum_attempts:
                    raise RuntimeError(
                        "too many rank-deficient patient-bootstrap draws; "
                        "reduce the rank grid or increase within-patient support"
                    ) from None
                continue
            multiplicities[accepted_draws] = candidate_multiplicities[candidate_index]
            bootstrap_groups.append(models)
            candidate_accepted[candidate_index] = True
            candidate_accepted_indices[candidate_index] = accepted_draws
            accepted_draws += 1
        attempt_multiplicity_chunks.append(candidate_multiplicities)
        attempt_acceptance_chunks.append(candidate_accepted)
        attempt_index_chunks.append(candidate_accepted_indices)

    attempt_ledger = PatientBootstrapAttemptLedger(
        multiplicities=np.concatenate(attempt_multiplicity_chunks, axis=0),
        accepted=np.concatenate(attempt_acceptance_chunks),
        accepted_bootstrap_index=np.concatenate(attempt_index_chunks),
    )

    return PatientBootstrapModelBank(
        statistics=statistics,
        point_models=point_models,
        bootstrap_models=tuple(bootstrap_groups),
        bootstrap_multiplicities=multiplicities,
        ranks=normalized_ranks,
        basis_variants=variants,
        fit_cohort=fit_cohort,
        seed=normalized_seed,
        attempt_ledger=attempt_ledger,
        rejected_draws=rejected_draws,
    )


def rank_path_from_model_bank(
    bank: PatientBootstrapModelBank,
    observed_leads,
    *,
    observation_variance_mv2: float,
    rcond: float | None = RECON_RCOND,
) -> BootstrapRankPath:
    """Evaluate one observed configuration without fitting any spatial models."""

    point = compute_rank_path(
        bank.point_models,
        observed_leads,
        observation_variance_mv2=observation_variance_mv2,
        rcond=rcond,
    )
    replicates = tuple(
        evaluate_spatial_model(
            model,
            observed_leads,
            observation_variance_mv2=observation_variance_mv2,
            rcond=rcond,
            bootstrap_index=bootstrap_index,
        )
        for bootstrap_index, model_group in enumerate(bank.bootstrap_models)
        for model in model_group
    )
    return BootstrapRankPath(
        point=point,
        replicates=replicates,
        seed=bank.seed,
        n_boot=bank.n_boot,
        patient_ids=bank.patient_ids,
    )


__all__ = [
    "BOOTSTRAP_MOMENT_BATCH_SIZE",
    "PatientBootstrapAttemptLedger",
    "PatientBootstrapModelBank",
    "PatientClusterSufficientStatistics",
    "bootstrap_spatial_model_bank",
    "cache_patient_cluster_statistics",
    "rank_path_from_model_bank",
    "rebuild_spatial_model_bank",
]
