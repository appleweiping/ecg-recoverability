"""Distribution-free guarantee checks (Monte-Carlo vs nominal level)."""
import numpy as np
import pytest

from ecgcert.conformal import (
    MondrianCQR,
    conformal_quantile,
    cqr_calibrate,
    cqr_interval,
    empirical_coverage,
    flag_threshold,
    weighted_conformal_quantile,
)


def test_conformal_quantile_marginal_coverage():
    """A one-sided conformal quantile controls exceedance at <= alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    exceed = []
    for _ in range(400):
        cal = rng.standard_normal(200)
        Q = conformal_quantile(cal, alpha)
        test = rng.standard_normal(500)
        exceed.append(np.mean(test > Q))
    # Mean exceedance <= alpha (finite-sample conformal is conservative).
    assert np.mean(exceed) <= alpha + 0.01


def test_cqr_coverage_clean():
    rng = np.random.default_rng(1)
    alpha = 0.1
    covs = []
    for _ in range(300):
        x = rng.uniform(-2, 2, 3000)
        sigma = 0.3 + 0.5 * np.abs(x)
        y = np.sin(x) + sigma * rng.standard_normal(x.size)
        q_lo = np.sin(x) - 1.0 * sigma
        q_hi = np.sin(x) + 1.0 * sigma
        idx = rng.permutation(x.size)
        cal, te = idx[:1500], idx[1500:]
        Q = cqr_calibrate(y[cal], q_lo[cal], q_hi[cal], alpha)
        lo, hi = cqr_interval(q_lo[te], q_hi[te], Q)
        covs.append(empirical_coverage(lo, hi, y[te]))
    assert np.mean(covs) >= 1 - alpha - 0.01


def test_mondrian_group_conditional_coverage():
    """Per-group CQR covers each group at >= 1 - alpha even when groups differ."""
    rng = np.random.default_rng(2)
    alpha = 0.1
    # Two groups with very different noise scales.
    def make(n):
        g = rng.integers(0, 2, n)
        scale = np.where(g == 0, 0.2, 1.5)
        y = scale * rng.standard_normal(n)
        q_lo = -0.5 * scale * np.ones(n)   # deliberately too-narrow base quantiles
        q_hi = 0.5 * scale * np.ones(n)
        return g, y, q_lo, q_hi

    g_c, y_c, lo_c, hi_c = make(4000)
    m = MondrianCQR(alpha).fit(g_c, y_c, lo_c, hi_c)
    g_t, y_t, lo_t, hi_t = make(8000)
    lo, hi = m.interval(g_t, lo_t, hi_t)
    for grp in (0, 1):
        sel = g_t == grp
        assert empirical_coverage(lo[sel], hi[sel], y_t[sel]) >= 1 - alpha - 0.02


def test_mondrian_tuple_groups_not_flattened():
    """Regression: tuple groups (segment, lead) must be kept intact, not collapsed
    into their elements by np.asarray. Two tuple groups -> exactly two corrections,
    each covering its group."""
    rng = np.random.default_rng(7)
    alpha = 0.1
    labels = [("QRS", "V2"), ("ST", "V4")]

    def make(n):
        gi = rng.integers(0, 2, n)
        groups = [labels[i] for i in gi]
        scale = np.where(gi == 0, 0.2, 1.5)
        y = scale * rng.standard_normal(n)
        return groups, y, -0.5 * scale, 0.5 * scale

    gc, yc, lo_c, hi_c = make(4000)
    m = MondrianCQR(alpha).fit(gc, yc, lo_c, hi_c)
    assert set(m.Q.keys()) == {"QRS|V2", "ST|V4"}          # two tuple groups, not four strings
    assert all(n > 100 for n in m.n_group.values())
    gt, yt, lo_t, hi_t = make(8000)
    lo, hi = m.interval(gt, lo_t, hi_t)
    for lab in labels:
        sel = np.array([g == lab for g in gt])
        assert empirical_coverage(lo[sel], hi[sel], yt[sel]) >= 1 - alpha - 0.03


def test_flag_false_flag_rate_controlled():
    """Flag threshold keeps false-flag rate <= alpha on faithful examples."""
    rng = np.random.default_rng(3)
    alpha = 0.05
    rates = []
    for _ in range(400):
        h_faithful_cal = np.abs(rng.standard_normal(300)) * 0.1
        tau = flag_threshold(h_faithful_cal, alpha)
        h_faithful_test = np.abs(rng.standard_normal(1000)) * 0.1
        rates.append(np.mean(h_faithful_test > tau))
    assert np.mean(rates) <= alpha + 0.01


def test_weighted_conformal_recovers_coverage_under_shift():
    """Under covariate shift, weighted conformal restores coverage that plain
    conformal loses."""
    rng = np.random.default_rng(4)
    alpha = 0.1
    plain_cov, weighted_cov = [], []
    for _ in range(200):
        # Source x ~ N(0,1); target x ~ N(1,1) (covariate shift). y|x heteroscedastic.
        xs = rng.standard_normal(3000)
        xt = rng.standard_normal(3000) + 1.0
        f = lambda x: 0.5 * x
        s = lambda x: 0.2 + 0.4 * np.abs(x)
        ys = f(xs) + s(xs) * rng.standard_normal(xs.size)
        yt = f(xt) + s(xt) * rng.standard_normal(xt.size)
        q_lo_s, q_hi_s = f(xs) - s(xs), f(xs) + s(xs)
        q_lo_t, q_hi_t = f(xt) - s(xt), f(xt) + s(xt)
        score = np.maximum(q_lo_s - ys, ys - q_hi_s)
        # Plain conformal (ignores shift).
        Qp = conformal_quantile(score, alpha)
        cov_p = empirical_coverage(q_lo_t - Qp, q_hi_t + Qp, yt)
        plain_cov.append(cov_p)
        # Weighted conformal with true LR w(x)=N(1,1)/N(0,1).
        lr = np.exp(xs - 0.5)                # dN(1,1)/dN(0,1) at xs
        Qw = weighted_conformal_quantile(score, lr, alpha)
        cov_w = empirical_coverage(q_lo_t - Qw, q_hi_t + Qw, yt)
        weighted_cov.append(cov_w)
    # Weighted conformal is at least as good and closer to nominal from below.
    assert np.mean(weighted_cov) >= 1 - alpha - 0.03
    assert np.mean(weighted_cov) >= np.mean(plain_cov) - 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
