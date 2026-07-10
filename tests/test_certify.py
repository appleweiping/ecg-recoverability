"""Tier decomposition + hallucination-energy checks (theory vs simulation)."""
import numpy as np
import pytest

from ecgcert.certify import (
    Tier,
    certified_unrecoverable_projector,
    hallucination_energy,
    recoverable_dipole_projector,
    supported_reconstruction,
    tier_report,
)
from ecgcert.physics import (
    DipolarModel,
    LEAD_INDEX,
    fit_dipolar_subspace,
    inverse_dower_matrix,
    lead_transform_T,
)


def _dipolar_population(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    T, D = lead_transform_T(), inverse_dower_matrix()
    dip = rng.standard_normal((3, n)) * np.array([[1.0], [0.7], [0.5]])
    return (T @ (D @ dip)).T  # (n, 12)


def test_full_dipole_recovery_projector_is_MMt():
    L = _dipolar_population()
    M_s, _, _ = fit_dipolar_subspace(L, rank=3)
    R_s, r = recoverable_dipole_projector(M_s, ["I", "II", "V2"])
    assert r == 3
    # A dipole-spanning set recovers the whole dipole subspace: R_s == M_s M_s^T.
    assert np.linalg.norm(R_s - M_s @ M_s.T) < 1e-8
    # R_s is an orthogonal projector.
    assert np.linalg.norm(R_s @ R_s - R_s) < 1e-8


def test_hallucination_energy_zero_for_faithful_dipolar():
    L = _dipolar_population()
    M_s, mu_s, _ = fit_dipolar_subspace(L, rank=3)
    Ltest = _dipolar_population(n=64, seed=7).T          # (12, 64) exactly dipolar
    h = hallucination_energy(M_s, mu_s, ["I", "II", "V2"], Ltest)
    # Purely dipolar signal has no certified-unrecoverable energy.
    assert np.max(h) < 1e-8


def test_hallucination_energy_scales_with_injection():
    L = _dipolar_population()
    M_s, mu_s, _ = fit_dipolar_subspace(L, rank=3)
    rng = np.random.default_rng(1)
    base = _dipolar_population(n=64, seed=3).T          # (12, 64)
    # Inject non-dipolar content on V3 (a precordial lead) with growing energy.
    v3 = LEAD_INDEX["V3"]
    energies, hs = [], []
    for delta in [0.0, 0.05, 0.1, 0.2, 0.4]:
        L_hat = base.copy()
        L_hat[v3] += delta * rng.standard_normal(base.shape[1])
        h = hallucination_energy(M_s, mu_s, ["I", "II", "V2"], L_hat)
        energies.append(delta)
        hs.append(h[v3])
    # Monotone increasing hallucination energy with injected non-dipolar energy.
    assert all(hs[i] < hs[i + 1] for i in range(len(hs) - 1))
    assert hs[0] < 1e-8


def test_supported_reconstruction_strips_injection():
    L = _dipolar_population()
    M_s, mu_s, _ = fit_dipolar_subspace(L, rank=3)
    base = _dipolar_population(n=32, seed=5).T
    L_hat = base.copy()
    L_hat[LEAD_INDEX["V3"]] += 0.3
    supported = supported_reconstruction(M_s, mu_s, ["I", "II", "V2"], L_hat)
    # The supported reconstruction lives in the dipole subspace: its residual has
    # no certified-unrecoverable energy.
    h = hallucination_energy(M_s, mu_s, ["I", "II", "V2"], supported)
    assert np.max(h) < 1e-8
    # And the supported reconstruction equals the recovered dipolar projection.
    R_s, _ = recoverable_dipole_projector(M_s, ["I", "II", "V2"])
    expect = mu_s[:, None] + R_s @ (L_hat - mu_s[:, None])
    assert np.linalg.norm(supported - expect) < 1e-10


def test_tier_report_labels():
    L = _dipolar_population()
    # Build a 4-segment model (reuse the same dipolar pop for all segments here).
    segs = {s: _dipolar_population(n=2000, seed=i) for i, s in enumerate(["P", "QRS", "ST", "T"])}
    model = DipolarModel.fit(segs, rank=3)
    # Dipole-spanning config: observed leads OBSERVED, others RECOVERABLE (dipolar pop).
    rep = tier_report(model, ["I", "II", "V2"], dipolar_threshold=0.8)
    assert rep["QRS"]["I"]["tier"] == Tier.OBSERVED.value
    assert rep["QRS"]["V6"]["tier"] == Tier.RECOVERABLE.value
    # Coplanar limb triplet (rank 2) => reconstructed leads UNRECOVERABLE.
    rep2 = tier_report(model, ["I", "II", "III"], dipolar_threshold=0.8)
    assert rep2["QRS"]["V6"]["tier"] == Tier.UNRECOVERABLE.value


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
