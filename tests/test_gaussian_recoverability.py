import numpy as np

from ecgcert.physics import LEADS, fit_spatial_subspace
from ecgcert.recoverability import (
    gaussian_conditional_mean,
    gaussian_posterior_covariance,
    gaussian_prior_ambiguity_per_lead,
    tune_gaussian_regularization,
)


def _model(seed=0, rank=4):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(300, rank))
    mixing, _ = np.linalg.qr(rng.normal(size=(len(LEADS), rank)))
    X = latent @ mixing.T + 0.01 * rng.normal(size=(300, len(LEADS)))
    return X, fit_spatial_subspace(X, rank=rank, fit_cohort="fixture")


def test_posterior_shrinks_and_ambiguity_is_mv():
    _, model = _model()
    prior = model.covariance
    posterior = gaussian_posterior_covariance(
        model, ("I", "II"), observation_variance_mv2=1e-4
    )
    assert posterior.shape == prior.shape
    assert np.linalg.eigvalsh(prior - posterior).min() > -1e-9
    ambiguity = gaussian_prior_ambiguity_per_lead(
        model, ("I", "II"), observation_variance_mv2=1e-4
    )
    assert ambiguity.shape == (12,)
    assert np.all(ambiguity >= 0)


def test_conditional_mean_copies_observed_samples_exactly():
    X, model = _model()
    reconstructed = gaussian_conditional_mean(
        model, X[:20], ("I", "V2", "V6"), observation_variance_mv2=1e-5
    )
    indices = [LEADS.index(lead) for lead in ("I", "V2", "V6")]
    assert np.array_equal(reconstructed[:, indices], X[:20, indices])


def test_fold8_tuning_is_deterministic_and_patient_balanced():
    X, model = _model()
    patient_ids = np.asarray([f"p{index // 10}" for index in range(100)], dtype=object)
    arguments = dict(
        models_by_segment={"QRS": (model,)},
        validation_by_segment={"QRS": (X[:100], patient_ids)},
        configurations=(("I",), ("I", "II")),
        grid_mv2=(1e-6, 1e-4),
    )
    first = tune_gaussian_regularization(**arguments)
    second = tune_gaussian_regularization(**arguments)
    assert first.selected_variance_mv2 in {1e-6, 1e-4}
    assert first.selected_variance_mv2 == second.selected_variance_mv2
    assert first.table.equals(second.table)
