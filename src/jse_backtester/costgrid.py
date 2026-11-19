"""Transaction-cost sensitivity grid: factor x rebalance frequency x round-trip cost.

Round-trip cost = what you pay to buy and later sell 100% of a position, split evenly across
the buy and sell legs (a simplification: real STT is buy-only). For reference, the default
JSE model in BacktestConfig (0.25% STT + 0.15% brokerage each way) is ~55 bps round trip.

Efficiency/exactness: weights drift on *gross* returns, so costs never change the weight path.
Net return = gross return - one-way turnover x round-trip cost on each execution date. One
cost-free backtest per (factor, frequency) therefore gives every cost level exactly; a test
checks this against the full engine.
"""
from __future__ import annotations

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .optimize import DEFAULT_MAX_WEIGHT
from .risk import DEFAULT_RF, annualised_return, sharpe
from .utils import TRADING_DAYS

DEFAULT_COSTS_BPS = (0, 15, 25, 40, 65, 100)
DEFAULT_FACTORS = ("momentum", "low_vol", "reversal")
DEFAULT_REBALANCES = ("W", "M", "Q")


def cost_grid(
    prices: pd.DataFrame,
    factors: tuple[str, ...] = DEFAULT_FACTORS,
    rebalances: tuple[str, ...] = DEFAULT_REBALANCES,
    costs_bps: tuple[float, ...] = DEFAULT_COSTS_BPS,
    top_frac: float = 0.3,
    weighting: str = "equal",
    rf: float = DEFAULT_RF,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    """Tidy DataFrame: factor, rebalance, cost_bps, net_cagr, net_sharpe, annual_turnover, gross_cagr."""
    rows = []
    for factor in factors:
        for freq in rebalances:
            cfg = BacktestConfig(signal=factor, top_frac=top_frac, rebalance=freq,
                                 weighting=weighting, stt=0.0, brokerage=0.0, max_weight=max_weight)
            res = run_backtest(prices, cfg)
            gross = res.gross_returns
            turn = res.turnover.reindex(gross.index).fillna(0.0)
            ann_turn = float(turn.sum() / (len(gross) / TRADING_DAYS))
            gross_cagr = annualised_return(gross)
            for bps in costs_bps:
                net = gross - turn * (bps / 1e4)
                rows.append({
                    "factor": factor, "rebalance": freq, "cost_bps": bps,
                    "net_cagr": annualised_return(net), "net_sharpe": sharpe(net, rf),
                    "annual_turnover": ann_turn, "gross_cagr": gross_cagr,
                })
    return pd.DataFrame(rows)


def grid_pivot(grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows = (factor, rebalance), columns = round-trip cost in bps."""
    order = {f: i for i, f in enumerate(DEFAULT_FACTORS)}
    freq_order = {f: i for i, f in enumerate(DEFAULT_REBALANCES)}
    p = grid.pivot_table(index=["factor", "rebalance"], columns="cost_bps", values=metric)
    idx = sorted(p.index, key=lambda t: (order.get(t[0], 99), freq_order.get(t[1], 99)))
    return p.loc[idx]
