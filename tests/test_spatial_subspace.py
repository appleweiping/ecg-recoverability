"""Rank-generic spatial-subspace geometry and legacy compatibility."""

import numpy as np
import pytest

from ecgcert.physics import (
    DipolarModel,
    LEAD_INDEX,
    SpatialSubspaceModel,
    fit_spatial_subspace,
    kappa,
    lead_transform_T,
    observed_dipole,
)
from ecgcert.recoverability import compute_rank_path


def _population(n: int = 320, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.linspace(3.0, 0.5, 12)
    return rng.standard_normal((n, 12)) * scales


@pytest.mark.parametrize("basis_variant", ["raw12_pca", "independent8_lifted"])
@pytest.mark.parametrize("rank", range(1, 7))
def test_spatial_model_supports_rank_one_through_six(basis_variant, rank):
    X = _population()
    model = fit_spatial_subspace(
        X,
        rank=rank,
        basis_variant=basis_variant,
        fit_cohort="development",
        fit_ids=(4, 7, 9),
    )

    assert model.rank == rank
    assert model.M.shape == (12, rank)
    assert model.mu.shape == (12,)
    assert model.covariance.shape == (rank, rank)
    np.testing.assert_allclose(model.M.T @ model.M, np.eye(rank), atol=1e-10)
    np.testing.assert_allclose(model.covariance, model.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(model.covariance).min() > -1e-10

    entry = compute_rank_path(
        [model], ["I", "II", "V2"], observation_variance_mv2=1e-4
    )[0]
    assert entry.rank == rank
    assert 0 <= entry.effective_rank <= rank
    assert entry.eta.shape == (12,)
    assert entry.kappa_per_lead.shape == (12,)
    assert entry.ambiguity.shape == (12,)


def test_legacy_dipolar_model_matches_rank_three_raw12_fit():
    X = _population()
    legacy = DipolarModel.fit({"QRS": X}, rank=3)
    generic = fit_spatial_subspace(
        X,
        rank=3,
        basis_variant="raw12_pca",
        fit_cohort="development",
    )

    np.testing.assert_allclose(legacy.mu["QRS"], generic.mu)
    np.testing.assert_allclose(
        legacy.M["QRS"] @ legacy.M["QRS"].T,
        generic.M @ generic.M.T,
        atol=1e-12,
    )
    assert legacy.rank == generic.rank == 3


def test_quantized_raw12_is_near_full_limb_rank_but_lifted_is_exact_rank_two():
    rng = np.random.default_rng(123)
    scales = np.array([3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8])
    independent = rng.standard_normal((500, 8)) * scales
    # Independent rounding of the stored channels mimics finite acquisition
    # precision and breaks exact limb algebra at a very small scale.
    quantized_raw12 = np.round(independent @ lead_transform_T().T, decimals=3)
    raw = fit_spatial_subspace(
        quantized_raw12,
        rank=3,
        basis_variant="raw12_pca",
        fit_cohort="quantized",
    )
    lifted = fit_spatial_subspace(
        quantized_raw12,
        rank=3,
        basis_variant="independent8_lifted",
        fit_cohort="quantized",
    )
    limb = ["I", "II", "III", "aVR", "aVL", "aVF"]

    raw_exact = observed_dipole(raw.M, limb, rcond=None)
    raw_deployed = observed_dipole(raw.M, limb, rcond=1e-2)
    lifted_exact = observed_dipole(lifted.M, limb, rcond=None)

    assert raw_exact.rank == 3
    assert raw_deployed.rank == 2
    assert raw_exact.sv[-1] / raw_exact.sv[0] < 1e-3
    assert lifted_exact.rank == 2
    assert lifted_exact.sv[-1] / lifted_exact.sv[0] < 1e-14

    # The no-rcond kappa call is the full-precision diagnostic used by the
    # precheck, while an explicit rcond is the deployed stability verdict.
    full_precision_kappa, full_precision_rank = kappa(raw.M, limb)
    deployed_kappa, deployed_rank = kappa(raw.M, limb, rcond=1e-2)
    legacy = DipolarModel.fit({"QRS": quantized_raw12}, rank=3)
    _, legacy_default_rank = legacy.kappa("QRS", limb)
    assert full_precision_rank == 3
    assert deployed_rank == 2
    assert legacy_default_rank == 2
    assert full_precision_kappa > 100 * deployed_kappa


def test_rank_path_metrics_are_invariant_to_coordinate_rotation():
    rng = np.random.default_rng(41)
    model = fit_spatial_subspace(
        _population(seed=17),
        rank=4,
        basis_variant="raw12_pca",
        fit_cohort="development",
        fit_ids=(1, 2, 3),
    )
    rotation, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    rotated = SpatialSubspaceModel(
        rank=model.rank,
        basis_variant=model.basis_variant,
        fit_cohort=model.fit_cohort,
        fit_ids=model.fit_ids,
        M=model.M @ rotation,
        mu=model.mu,
        covariance=rotation.T @ model.covariance @ rotation,
    )
    observed = ["I", "II", "V2"]
    original_entry = compute_rank_path(
        [model], observed, observation_variance_mv2=1e-4, rcond=1e-8
    )[0]
    rotated_entry = compute_rank_path(
        [rotated], observed, observation_variance_mv2=1e-4, rcond=1e-8
    )[0]

    assert original_entry.effective_rank == rotated_entry.effective_rank
    assert original_entry.kappa_global == pytest.approx(
        rotated_entry.kappa_global, rel=1e-10, abs=1e-12
    )
    for field_name in ("eta", "eta_normalized", "kappa_per_lead", "ambiguity"):
        np.testing.assert_allclose(
            getattr(original_entry, field_name),
            getattr(rotated_entry, field_name),
            rtol=1e-9,
            # Ambiguity is a square root of a covariance diagonal; round-off at
            # mathematically zero entries is therefore O(sqrt(machine epsilon)).
            atol=5e-8,
        )

    # A frozen model owns immutable copies, preventing accidental post-fit drift.
    assert not model.M.flags.writeable
    assert not model.covariance.flags.writeable
    assert LEAD_INDEX["V2"] == 7
