"""Lock the paper's synthetic theorem-validations into the test suite.

These call the experiment functions directly and assert the invariants that the
figures claim, so no figure can silently drift away from its theorem.
"""
import pytest

from experiments import synthetic_dipole_injection as S


def test_tier1_certificate_matches_GF_and_bound():
    out = S.experiment_tier1_certificate()
    for name, c in out["configs"].items():
        # Full-rank configs: RMS error equals sigma*||G||_F and respects the bound.
        for e, p, b in zip(c["err"], c["predicted"], c["bound"]):
            assert e <= b + 1e-6, (name, e, b)
            assert abs(e - p) <= 0.02 * (p + 1.0), (name, e, p)


def test_tier3_bayes_hits_var_lower_bound():
    out = S.experiment_tier3_irreducibility()
    for b, lb, h in zip(out["bayes_mse"], out["lower_bound_var_u"], out["hallucinator_mse"]):
        # Bayes-optimal reconstruction achieves the Var(u) lower bound (can't beat it).
        assert abs(b - lb) <= 0.02 * (lb + 1e-3)
        # A hallucinator does strictly worse (roughly 2x) while looking confident.
        if lb > 1e-4:
            assert h >= 1.5 * lb


def test_flag_false_flag_and_power():
    out = S.experiment_flag_roc()
    assert out["false_flag_rate_heldout"] <= out["alpha"] + 0.02
    # Power reaches 1.0 for any non-trivial injected energy.
    assert min(out["power"][1:]) >= 0.95


def test_coverage_at_nominal():
    out = S.experiment_coverage()
    assert out["empirical_coverage"] >= out["target_coverage"] - 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
