"""Leakage guards for the arbitrary-mask conditional DDPM.

These tests exist because an earlier version fixed the observed set at construction and
then reused the same model to score a different configuration, leaking the originally
observed target leads. They assert that the model conditions on EXACTLY the requested
observed leads and nothing else.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgcert.estimators.diffusion import DiffusionReconstructor, _DEPLOY_CONFIGS

PRECORDIAL = (6, 7, 8, 9, 10, 11)  # V1..V6
LIMB6 = (0, 1, 2, 3, 4, 5)
PRECORDIAL_INTERP = (0, 1, 6, 8, 10)  # I,II,V1,V3,V5


def _tiny():
    return DiffusionReconstructor(T=8, base=8, device="cpu", cond_dropout=0.0, seed=0)


def test_mask_from_obs_is_exact():
    m = _tiny()
    mask = m._mask_from_obs(LIMB6, B=2, W=16)          # (2,12,16)
    row = mask[0, :, 0].numpy()
    assert list(np.where(row == 1)[0]) == list(LIMB6)
    assert row.sum() == len(LIMB6)


def test_conditioning_zeroes_unobserved_values_and_mask():
    m = _tiny()
    B, W = 3, 16
    x0 = torch.randn(B, 12, W)
    mask = m._mask_from_obs(PRECORDIAL_INTERP, B, W)
    cond = m._cond(x0, mask)                            # (B,24,W): [obs(12), mask(12)]
    obs, mask_ch = cond[:, :12], cond[:, 12:]
    unobs = [i for i in range(12) if i not in PRECORDIAL_INTERP]
    # observed-value channels are zero on unobserved leads; mask channels are 0/1.
    assert torch.allclose(obs[:, unobs], torch.zeros_like(obs[:, unobs]))
    assert torch.allclose(mask_ch[:, list(PRECORDIAL_INTERP)], torch.ones(B, len(PRECORDIAL_INTERP), W))
    assert torch.allclose(mask_ch[:, unobs], torch.zeros(B, len(unobs), W))


def test_perturbing_unobserved_leads_does_not_change_conditioning():
    """The conditioning must be invariant to the (arbitrary) values on unobserved leads."""
    m = _tiny()
    B, W = 2, 16
    x0 = torch.randn(B, 12, W)
    mask = m._mask_from_obs(LIMB6, B, W)
    cond_a = m._cond(x0, mask)
    x0b = x0.clone()
    x0b[:, list(PRECORDIAL)] += 5.0                    # change ONLY unobserved (precordial) leads
    cond_b = m._cond(x0b, mask)
    assert torch.allclose(cond_a, cond_b)


def test_limb6_conditioning_excludes_precordial():
    """A limb-6 reconstruction must not see any precordial (V1-V6) content."""
    m = _tiny()
    B, W = 2, 16
    x0 = torch.randn(B, 12, W)
    mask = m._mask_from_obs(LIMB6, B, W)
    cond = m._cond(x0, mask)
    obs, mask_ch = cond[:, :12], cond[:, 12:]
    assert torch.count_nonzero(obs[:, list(PRECORDIAL)]) == 0
    assert torch.count_nonzero(mask_ch[:, list(PRECORDIAL)]) == 0


def test_repaint_preserves_observed_and_ignores_unobserved_truth():
    """End-to-end: sampling with obs_idx=S returns the true observed leads (RePaint),
    and the reconstruction of a config is invariant to the truth on unobserved leads
    (no target leakage). Uses a tiny untrained model -- we test data flow, not quality."""
    m = _tiny()
    B, W = 2, 16
    y = np.random.default_rng(0).standard_normal((B, 12, W)).astype(np.float32)
    rec = m.sample(y, obs_idx=LIMB6, guidance=1.0, replace=True, steps=8, seed=1)
    # observed leads are exactly preserved by RePaint.
    assert np.allclose(rec[:, list(LIMB6)], y[:, list(LIMB6)], atol=1e-5)
    # changing the truth on the UNOBSERVED precordial leads must not change the limb-6
    # reconstruction (same seed) -- otherwise target leads are leaking in.
    y2 = y.copy(); y2[:, list(PRECORDIAL)] += 3.0
    rec2 = m.sample(y2, obs_idx=LIMB6, guidance=1.0, replace=True, steps=8, seed=1)
    assert np.allclose(rec, rec2, atol=1e-5)


def test_deploy_configs_have_disjoint_target():
    """Sanity: for each deployment config the reconstructed (target) leads are disjoint
    from the observed leads."""
    for obs in _DEPLOY_CONFIGS:
        target = [i for i in range(12) if i not in obs]
        assert set(obs).isdisjoint(target)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
