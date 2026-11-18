import numpy as np
import pandas as pd
import pytest

from jse_backtester.backtest import BacktestConfig, run_backtest
from jse_backtester.costgrid import DEFAULT_COSTS_BPS, cost_grid, grid_pivot
from jse_backtester.data import synthetic_prices
from jse_backtester.universe import DEFAULT_BENCHMARK, JSE_TOP_TICKERS


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(JSE_TOP_TICKERS[:25], "2016-01-01", "2021-12-31", seed=11).drop(columns=[DEFAULT_BENCHMARK])


@pytest.fixture(scope="module")
def grid(prices):
    return cost_grid(prices)


def test_grid_shape_and_columns(grid):
    assert len(grid) == 3 * 3 * len(DEFAULT_COSTS_BPS)
    assert {"factor", "rebalance", "cost_bps", "net_cagr", "net_sharpe", "annual_turnover"} <= set(grid.columns)
    assert set(grid["cost_bps"]) == {0, 15, 25, 40, 65, 100}


def test_costs_only_ever_hurt(grid):
    for _, g in grid.groupby(["factor", "rebalance"]):
        g = g.sort_values("cost_bps")
        assert g["net_cagr"].is_monotonic_decreasing
        assert g["net_sharpe"].is_monotonic_decreasing
        assert g["annual_turnover"].nunique() == 1  # turnover does not depend on cost
        assert g.iloc[0]["net_cagr"] == pytest.approx(g.iloc[0]["gross_cagr"])


def test_faster_rebalancing_trades_more(grid):
    t = grid_pivot(grid, "annual_turnover")[0]
    for f in ("momentum", "low_vol", "reversal"):
        assert t[(f, "W")] > t[(f, "M")] > t[(f, "Q")]


@pytest.mark.parametrize("factor,freq,bps", [("momentum", "M", 40), ("reversal", "W", 100), ("low_vol", "Q", 25)])
def test_shortcut_matches_full_engine(prices, grid, factor, freq, bps):
    """Net = gross - turnover x cost must equal the full simulator with symmetric costs."""
    full = run_backtest(prices, BacktestConfig(signal=factor, rebalance=freq, stt=0.0, brokerage=bps / 2e4))
    from jse_backtester.risk import annualised_return, sharpe
    row = grid[(grid.factor == factor) & (grid.rebalance == freq) & (grid.cost_bps == bps)].iloc[0]
    assert row["net_cagr"] == pytest.approx(annualised_return(full.returns), rel=1e-9)
    assert row["net_sharpe"] == pytest.approx(sharpe(full.returns), rel=1e-9)
    assert row["annual_turnover"] == pytest.approx(full.annual_turnover, rel=1e-9)
