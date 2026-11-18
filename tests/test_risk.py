import numpy as np
import pandas as pd
import pytest

from jse_backtester import risk


def _returns(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=pd.bdate_range("2018-01-01", periods=n))


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.20, 0.05])
    # wealth: 1.10 -> 0.88 -> 0.924 ; drawdown trough = 0.88/1.10 - 1
    assert risk.max_drawdown(r) == pytest.approx(-0.20)


def test_var_ordering_and_normal_agreement():
    r = _returns()
    assert risk.historical_var(r, 0.99) > risk.historical_var(r, 0.95)
    assert risk.historical_cvar(r, 0.95) >= risk.historical_var(r, 0.95)
    # for near-Gaussian data CF-VaR ~ parametric VaR
    assert risk.cornish_fisher_var(r, 0.95) == pytest.approx(risk.parametric_var(r, 0.95), rel=0.1)


def test_risk_contributions_sum_to_one():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), columns=list("ABCDE"))
    cov = risk.shrunk_cov(rets, 0.2)
    w = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=cov.index)
    rc = risk.risk_contributions(w, cov)
    assert rc["pct"].sum() == pytest.approx(1.0)
    vol = np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert rc["contribution"].sum() == pytest.approx(vol)


def test_beta_recovers_known_value():
    b = _returns(seed=3)
    rng = np.random.default_rng(4)
    p = 1.3 * b + rng.normal(0, 0.002, len(b))
    beta, _, r2 = risk.beta_alpha(p, b, rf=0.0)
    assert beta == pytest.approx(1.3, abs=0.03)
    assert r2 > 0.9
