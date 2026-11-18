import numpy as np
import pandas as pd
import pytest

from jse_backtester.backtest import BacktestConfig, build_factor_returns, run_backtest
from jse_backtester.data import synthetic_prices
from jse_backtester.factors import (SIGNALS, factor_regression, ic_summary,
                                    information_coefficient, momentum)
from jse_backtester.universe import DEFAULT_BENCHMARK, JSE_TOP_TICKERS


@pytest.fixture(scope="module")
def prices():
    px = synthetic_prices(JSE_TOP_TICKERS[:25], "2016-01-01", "2022-12-31", seed=7)
    return px.drop(columns=[DEFAULT_BENCHMARK])


def test_momentum_value():
    px = pd.DataFrame({"A": np.arange(1.0, 301.0)})
    m = momentum(px, lookback=252, skip=21)
    assert m["A"].iloc[-1] == pytest.approx(px["A"].iloc[-22] / px["A"].iloc[-253] - 1)


def test_first_trade_costs_and_turnover(prices):
    cfg = BacktestConfig(signal="momentum")
    res = run_backtest(prices, cfg)
    first = res.turnover.iloc[0]
    # turnover = sum|dw|/2, so building a 100% book from cash is 0.5; costs hit all 100% bought
    assert first == pytest.approx(0.5)
    assert res.costs.iloc[0] == pytest.approx(cfg.stt + cfg.brokerage)
    assert (res.returns <= res.gross_returns + 1e-12).all()


def test_no_trading_before_signal_exists_and_lag(prices):
    res = run_backtest(prices, BacktestConfig(signal="momentum", lag=1))
    # momentum needs 252 days of history, so nothing can be traded before that
    assert res.returns.index[0] > prices.index[251]
    # every execution date is strictly after the signal date it came from (a month-end)
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    assert not set(res.weights.index) & set(month_ends)


def test_zero_costs_means_net_equals_gross(prices):
    res = run_backtest(prices, BacktestConfig(stt=0.0, brokerage=0.0))
    pd.testing.assert_series_equal(res.returns, res.gross_returns, check_names=False)


def test_long_short_is_roughly_dollar_neutral(prices):
    res = run_backtest(prices, BacktestConfig(long_only=False, stt=0, brokerage=0))
    assert res.weights.sum(axis=1).abs().max() < 1e-9
    assert res.weights.abs().sum(axis=1).round(6).eq(2.0).all()


def test_inverse_vol_weights_sum_to_one(prices):
    res = run_backtest(prices, BacktestConfig(weighting="inv_vol"))
    assert res.weights.sum(axis=1).round(8).eq(1.0).all()


def test_regression_recovers_betas():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    X = pd.DataFrame(rng.normal(0, 0.01, (1500, 2)), index=idx, columns=["F1", "F2"])
    y = 0.5 * X["F1"] - 0.3 * X["F2"] + rng.normal(0, 0.002, 1500)
    out = factor_regression(y, X, rf=0.0)
    assert out.table.loc["F1", "coef"] == pytest.approx(0.5, abs=0.03)
    assert out.table.loc["F2", "coef"] == pytest.approx(-0.3, abs=0.03)
    assert out.r2 > 0.85  # theoretical R^2 is ~0.895 by construction


def test_momentum_ic_positive_on_synthetic(prices):
    # the synthetic generator embeds persistent drift, so momentum should carry signal
    ic = information_coefficient(SIGNALS["momentum"](prices), prices)
    assert ic_summary(ic)["Mean IC"] > 0


def test_factor_returns_shape(prices):
    f = build_factor_returns(prices)
    assert set(f.columns) == {"MOMENTUM", "LOW_VOL", "REVERSAL", "MKT"}
