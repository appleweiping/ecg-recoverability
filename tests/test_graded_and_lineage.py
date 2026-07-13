"""Unit tests for the correctness-pass additions: normalized/expected ambiguity, the
absolute |ST| threshold, and lineage consistency assertions.
"""
import numpy as np
import pytest

from ecgcert.physics import (
    fit_dipolar_subspace, eta_per_lead, eta_normalized_per_lead, lead_dipolar_norm,
    dipole_coord_cov, expected_ambiguity_per_lead,
)
from ecgcert import lineage
from ecgcert import clinical


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((800, 3)) * np.array([2.0, 1.0, 0.4])
    M_true, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    X = d @ M_true.T + 0.05 * rng.standard_normal((800, 12))
    return X


def test_eta_normalized_is_fraction():
    X = _synthetic()
    M, mu, _ = fit_dipolar_subspace(X, rank=3)
    obs = ["I", "II", "V2"]
    eta = eta_per_lead(M, obs)
    etn = eta_normalized_per_lead(M, obs)
    denom = lead_dipolar_norm(M)
    # eta_tilde = eta / ||row|| where defined, and lies in [0, 1]
    ok = np.isfinite(etn)
    assert np.allclose(etn[ok], eta[ok] / denom[ok], atol=1e-9)
    assert np.all(etn[ok] >= -1e-9) and np.all(etn[ok] <= 1 + 1e-6)


def test_expected_ambiguity_nonneg_and_zero_when_identifiable():
    X = _synthetic()
    M, mu, _ = fit_dipolar_subspace(X, rank=3)
    Sd = dipole_coord_cov(M, mu, X)
    # spanning set -> P_obs = I -> zero unobserved ambiguity
    amb_span = expected_ambiguity_per_lead(M, ["I", "II", "V2"], Sd)
    assert np.all(amb_span >= 0)
    assert np.max(amb_span) < 1e-6
    # a single observed lead is rank-deficient for ANY M -> positive ambiguity elsewhere
    amb_one = expected_ambiguity_per_lead(M, ["V2"], Sd)
    assert np.max(amb_one) > 1e-3


def test_st_threshold_is_absolute():
    dev = np.zeros(12)
    dev[7] = -0.15                       # V2 depression
    assert clinical.st_threshold_positive(dev, leads=[7])       # |−0.15| >= 0.1 fires
    dev[7] = 0.15
    assert clinical.st_threshold_positive(dev, leads=[7])       # elevation fires too
    dev[7] = 0.05
    assert not clinical.st_threshold_positive(dev, leads=[7])   # below threshold


def test_no_stemi_language_in_clinical():
    import inspect
    src = inspect.getsource(clinical)
    for banned in ("STEMI", "phantom", "fabricated"):
        assert banned.lower() not in src.lower(), f"banned term {banned!r} in clinical.py"


def test_lineage_make_and_assert():
    lin = lineage.make(None, seed=0, targets=["V2"], normalization="raw",
                       train_ids=[3, 1, 2], test_ids=[5, 4])
    for k in ("commit", "protocol", "seed", "targets", "train_ids_sha256", "test_ids_sha256"):
        assert k in lin
    # order-independent id hash
    assert lin["train_ids_sha256"] == lineage.ids_sha256([1, 2, 3])
    lineage.assert_consistent(lin, dict(lin))                  # identical -> ok
    with pytest.raises(ValueError):
        lineage.assert_consistent(lin, {**lin, "seed": 99})
