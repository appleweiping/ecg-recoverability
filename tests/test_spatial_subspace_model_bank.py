"""Sufficient-statistics patient bootstrap model-bank tests."""

from collections import Counter
from uuid import UUID

import numpy as np
import pytest

import ecgcert.recoverability.model_bank as model_bank_module
from ecgcert.physics import fit_spatial_subspace
from ecgcert.recoverability import (
    aggregate_recoverability_envelope,
    bootstrap_spatial_model_bank,
    cache_patient_cluster_statistics,
    compute_rank_path,
    rank_path_from_model_bank,
)


RANKS = (2, 3)
VARIANTS = ("raw12_pca", "independent8_lifted")


def _string_patient_population():
    rng = np.random.default_rng(330)
    patient_names = ("patient-c", "patient-a", "patient-e", "patient-b", "patient-d")
    row_counts = (4, 5, 3, 6, 4)
    rows = []
    patient_ids = []
    for patient_index, (patient_id, row_count) in enumerate(
        zip(patient_names, row_counts)
    ):
        patient_effect = rng.normal(scale=0.7, size=12)
        trend = np.linspace(-0.2, 0.3, 12) * patient_index
        rows.append(
            patient_effect
            + trend
            + rng.normal(scale=0.35, size=(row_count, 12))
        )
        patient_ids.extend([patient_id] * row_count)
    return np.vstack(rows), tuple(patient_ids), patient_names


def _direct_models(X):
    return tuple(
        fit_spatial_subspace(
            X,
            rank=rank,
            basis_variant=variant,
            fit_cohort="development",
        )
        for variant in VARIANTS
        for rank in RANKS
    )


def _assert_rotation_equivalent(left, right):
    assert left.rank == right.rank
    assert left.basis_variant == right.basis_variant
    np.testing.assert_allclose(left.mu, right.mu, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        left.M @ left.M.T,
        right.M @ right.M.T,
        rtol=1e-8,
        atol=1e-8,
    )
    # Coordinate covariance rotates with the basis; its lead-space image is invariant.
    np.testing.assert_allclose(
        left.M @ left.covariance @ left.M.T,
        right.M @ right.covariance @ right.M.T,
        rtol=1e-8,
        atol=1e-8,
    )


def _rows_for_multiplicity(X, patient_ids, patient_order, multiplicities):
    patient_array = np.asarray(patient_ids, dtype=object)
    blocks = []
    for patient_id, multiplicity in zip(patient_order, multiplicities):
        rows = np.flatnonzero(patient_array == patient_id)
        blocks.extend([rows] * int(multiplicity))
    return X[np.concatenate(blocks)]


def test_patient_statistics_support_strings_and_match_raw_moments():
    X, patient_ids, patient_order = _string_patient_population()
    statistics = cache_patient_cluster_statistics(X, patient_ids)

    assert statistics.patient_ids == patient_order
    assert statistics.n_patients == len(patient_order)
    assert statistics.n_samples == X.shape[0]
    patient_array = np.asarray(patient_ids, dtype=object)
    for patient_index, patient_id in enumerate(patient_order):
        patient_X = X[patient_array == patient_id]
        shifted_patient_X = patient_X - statistics.origin
        assert statistics.counts[patient_index] == patient_X.shape[0]
        np.testing.assert_allclose(
            statistics.sums[patient_index],
            shifted_patient_X.sum(axis=0),
        )
        np.testing.assert_allclose(
            statistics.crossproducts[patient_index],
            shifted_patient_X.T @ shifted_patient_X,
        )
    np.testing.assert_allclose(statistics.origin, X.mean(axis=0))
    assert not statistics.crossproducts.flags.writeable

    invalid_ids = list(patient_ids)
    invalid_ids[0] = None
    with pytest.raises(ValueError, match="cannot be missing"):
        cache_patient_cluster_statistics(X, invalid_ids)

    for invalid_patient_id, message in (
        (float("nan"), "reflexive"),
        (True, "boolean"),
    ):
        invalid_ids = list(patient_ids)
        invalid_ids[0] = invalid_patient_id
        with pytest.raises(ValueError, match=message):
            cache_patient_cluster_statistics(X, invalid_ids)


def test_cached_statistics_and_uuid_ids_are_reusable_without_reading_x(monkeypatch):
    X, patient_ids, _ = _string_patient_population()
    uuid_by_string = {
        patient_id: UUID(int=index + 1)
        for index, patient_id in enumerate(dict.fromkeys(patient_ids))
    }
    uuid_ids = tuple(uuid_by_string[patient_id] for patient_id in patient_ids)
    statistics = cache_patient_cluster_statistics(X, uuid_ids)

    def fail_if_raw_data_is_read(*args, **kwargs):
        raise AssertionError("cached-statistics path must not rebuild patient moments")

    monkeypatch.setattr(
        model_bank_module,
        "cache_patient_cluster_statistics",
        fail_if_raw_data_is_read,
    )
    bank = bootstrap_spatial_model_bank(
        statistics,
        ranks=RANKS,
        basis_variants=VARIANTS,
        n_boot=2,
        seed=12,
    )

    assert bank.statistics is statistics
    assert bank.patient_ids == tuple(uuid_by_string.values())
    assert all(isinstance(patient_id, UUID) for patient_id in bank.patient_ids)


def test_point_and_small_bootstrap_models_equal_direct_fits_up_to_rotation():
    X, patient_ids, _ = _string_patient_population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="development",
        n_boot=4,
        seed=91,
    )

    for sufficient_model, direct_model in zip(bank.point_models, _direct_models(X)):
        _assert_rotation_equivalent(sufficient_model, direct_model)

    for bootstrap_index, sufficient_group in enumerate(bank.bootstrap_models):
        bootstrap_X = _rows_for_multiplicity(
            X,
            patient_ids,
            bank.patient_ids,
            bank.bootstrap_multiplicities[bootstrap_index],
        )
        for sufficient_model, direct_model in zip(
            sufficient_group,
            _direct_models(bootstrap_X),
        ):
            _assert_rotation_equivalent(sufficient_model, direct_model)

    # Invariant certificate quantities also agree for the point estimate.
    sufficient_path = compute_rank_path(
        bank.point_models, ["I", "II", "V2"], observation_variance_mv2=1e-4
    )
    direct_path = compute_rank_path(
        _direct_models(X), ["I", "II", "V2"], observation_variance_mv2=1e-4
    )
    for sufficient_entry, direct_entry in zip(sufficient_path, direct_path):
        assert sufficient_entry.effective_rank == direct_entry.effective_rank
        np.testing.assert_allclose(
            sufficient_entry.eta_normalized,
            direct_entry.eta_normalized,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            sufficient_entry.kappa_per_lead,
            direct_entry.kappa_per_lead,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            sufficient_entry.ambiguity,
            direct_entry.ambiguity,
            rtol=1e-7,
            atol=5e-8,
        )


def test_model_bank_is_deterministic_and_retains_shared_string_id_provenance():
    X, patient_ids, patient_order = _string_patient_population()
    kwargs = dict(
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="development",
        n_boot=5,
        seed=17,
    )
    first = bootstrap_spatial_model_bank(X, patient_ids, **kwargs)
    second = bootstrap_spatial_model_bank(X, patient_ids, **kwargs)

    assert first.patient_ids == second.patient_ids == patient_order
    np.testing.assert_array_equal(
        first.bootstrap_multiplicities,
        second.bootstrap_multiplicities,
    )
    for first_group, second_group in zip(
        (first.point_models,) + first.bootstrap_models,
        (second.point_models,) + second.bootstrap_models,
    ):
        for first_model, second_model in zip(first_group, second_group):
            np.testing.assert_array_equal(first_model.M, second_model.M)
            np.testing.assert_array_equal(first_model.mu, second_model.mu)
            np.testing.assert_array_equal(first_model.covariance, second_model.covariance)

    for bootstrap_index in range(first.n_boot):
        drawn_ids = first.fit_patient_ids(bootstrap_index)
        assert len(drawn_ids) == len(patient_order)
        assert Counter(drawn_ids) == Counter(
            {
                patient_id: int(multiplicity)
                for patient_id, multiplicity in zip(
                    patient_order,
                    first.bootstrap_multiplicities[bootstrap_index],
                )
                if multiplicity
            }
        )

    with pytest.raises(IndexError):
        first.fit_patient_ids(-1)


def test_fixed_origin_moments_are_stable_for_large_common_offsets():
    X, patient_ids, _ = _string_patient_population()
    offset_X = X * 1e-3 + 1e12
    # Subtracting the exactly represented common offset gives a stable reference
    # for the values actually representable at this floating-point scale.
    centered_reference = offset_X - 1e12
    bank = bootstrap_spatial_model_bank(
        offset_X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="development",
        n_boot=1,
        seed=3,
    )
    for sufficient_model, direct_model in zip(
        bank.point_models,
        _direct_models(centered_reference),
    ):
        np.testing.assert_allclose(
            sufficient_model.M @ sufficient_model.M.T,
            direct_model.M @ direct_model.M.T,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            sufficient_model.M @ sufficient_model.covariance @ sufficient_model.M.T,
            direct_model.M @ direct_model.covariance @ direct_model.M.T,
            rtol=1e-8,
            atol=1e-8,
        )
        assert np.linalg.norm(sufficient_model.covariance, 2) < 1e-4


def test_rank_deficient_draws_are_deterministically_rejected_and_reported():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(7, 12))
    patient_ids = tuple(f"singleton-{index}" for index in range(7))
    first = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(5,),
        basis_variants=VARIANTS,
        n_boot=3,
        seed=2,
    )
    second = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(5,),
        basis_variants=VARIANTS,
        n_boot=3,
        seed=2,
    )

    assert first.rejected_draws > 0
    assert first.rejection_fraction > 0.0
    assert first.rejected_draws == second.rejected_draws
    np.testing.assert_array_equal(
        first.bootstrap_multiplicities,
        second.bootstrap_multiplicities,
    )


def test_one_eigendecomposition_per_variant_and_draw_not_per_rank(monkeypatch):
    X, patient_ids, _ = _string_patient_population()
    original_eigh = model_bank_module.np.linalg.eigh
    observed_shapes = []

    def tracked_eigh(matrix):
        observed_shapes.append(matrix.shape)
        return original_eigh(matrix)

    monkeypatch.setattr(model_bank_module.np.linalg, "eigh", tracked_eigh)
    bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(2, 3, 4),
        basis_variants=VARIANTS,
        n_boot=3,
        seed=5,
    )

    # One point fit plus three bootstrap fits for each basis variant.
    assert observed_shapes.count((12, 12)) == 4
    assert observed_shapes.count((8, 8)) == 4
    assert len(observed_shapes) == 8


def test_all_observed_configurations_reuse_bank_without_refitting(monkeypatch):
    X, patient_ids, _ = _string_patient_population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        n_boot=3,
        seed=7,
    )

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("rank-path evaluation must not refit model-bank members")

    monkeypatch.setattr(model_bank_module, "_models_from_moments", fail_if_refit)
    limb_path = rank_path_from_model_bank(
        bank, ["I", "II", "III"], observation_variance_mv2=1e-4
    )
    spread_path = rank_path_from_model_bank(
        bank, ["I", "II", "V2", "V5"], observation_variance_mv2=1e-4
    )

    expected_entries = bank.n_boot * len(bank.point_models)
    assert len(limb_path.replicates) == expected_entries
    assert len(spread_path.replicates) == expected_entries
    assert limb_path.record_ids == spread_path.record_ids == bank.patient_ids
    assert {entry.bootstrap_index for entry in limb_path.replicates} == set(
        range(bank.n_boot)
    )
    # The existing envelope API consumes paths generated from the reusable bank.
    assert aggregate_recoverability_envelope(limb_path).eta_normalized_upper.shape == (12,)
