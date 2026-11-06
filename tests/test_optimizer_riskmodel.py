import numpy as np
import pytest

from quantbot.optimizer import objective, solve
from quantbot.riskmodel import ledoit_wolf


def _problem(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) * 0.01
    cov = A @ A.T + np.eye(n) * 1e-4
    return rng.normal(0, 0.002, n), cov, rng.uniform(0, 0.2, n), np.full(n, 3e-4)


@pytest.mark.parametrize("seed", range(5))
def test_fista_matches_scipy(seed):
    scipy_opt = pytest.importorskip("scipy.optimize")
    n = 8
    mu, cov, w0, cost = _problem(n, seed)
    kw = dict(horizon=5, risk_aversion=10.0, tcost_aversion=1.0)
    lo, hi = np.zeros(n), np.full(n, 0.3)
    res = solve(mu, cov, w0, cost, lo=lo, hi=hi, max_gross=10, target_vol=10, calibrate=False, tol=1e-12,
                max_iter=5000, **kw)
    # reference: split w = w0 + u - v to make the L1 term smooth
    f = lambda z: -objective(w0 + z[:n] - z[n:], mu, cov, w0, cost, **kw)  # noqa: E731
    bounds = [(0, None)] * (2 * n)
    cons = [{"type": "ineq", "fun": lambda z, i=i: (w0 + z[:n] - z[n:])[i] - lo[i]} for i in range(n)] + \
           [{"type": "ineq", "fun": lambda z, i=i: hi[i] - (w0 + z[:n] - z[n:])[i]} for i in range(n)]
    ref = scipy_opt.minimize(f, np.zeros(2 * n), bounds=bounds, constraints=cons, method="SLSQP",
                             options={"ftol": 1e-14, "maxiter": 1000})
    assert objective(res.w, mu, cov, w0, cost, **kw) >= -ref.fun - 1e-9


def test_no_trade_band_when_edge_below_cost():
    n = 5
    mu, cov, w0, _ = _problem(n, 1)
    huge_cost = np.full(n, 1.0)
    res = solve(mu * 1e-3, cov, w0, huge_cost, horizon=5, risk_aversion=10, tcost_aversion=1,
                lo=np.zeros(n), hi=np.ones(n), max_gross=10, target_vol=10, calibrate=False)
    np.testing.assert_allclose(res.w, w0, atol=1e-9)


def test_caps_and_vol_target_respected():
    n = 10
    mu, cov, w0, cost = _problem(n, 2)
    res = solve(np.abs(mu) * 50, cov, np.zeros(n), cost, horizon=5, risk_aversion=0.1, tcost_aversion=1,
                lo=np.zeros(n), hi=np.full(n, 0.2), max_gross=1.0, target_vol=0.05)
    assert res.w.min() >= -1e-12 and res.w.max() <= 0.2 + 1e-12
    assert np.abs(res.w).sum() <= 1.0 + 1e-9
    assert res.ex_ante_vol <= 0.05 + 1e-9


def test_calibration_uses_the_risk_budget():
    n = 10
    mu, cov, _, cost = _problem(n, 3)
    res = solve(np.abs(mu), cov, np.zeros(n), cost * 0, horizon=5, risk_aversion=1000, tcost_aversion=1,
                lo=np.zeros(n), hi=np.ones(n), max_gross=5.0, target_vol=0.10)
    assert res.ex_ante_vol == pytest.approx(0.10, rel=0.06)


def test_ledoit_wolf_is_valid_and_shrinks_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))                     # true cov = I, few observations
    S, shrink = ledoit_wolf(X)
    assert 0 <= shrink <= 1 and shrink > 0.3
    assert np.linalg.eigvalsh(S).min() > 0
    sample = np.cov(X, rowvar=False, bias=True)
    assert np.linalg.norm(S - np.eye(20)) < np.linalg.norm(sample - np.eye(20))
