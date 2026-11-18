"""Rank-based factor backtester with JSE transaction costs and an execution lag.

Timeline for one rebalance:
    close of day d  : signal computed from data <= d, target weights fixed
    close of d+lag  : trades executed (default lag = 1 trading day), costs deducted
    afterwards      : weights drift with returns until the next rebalance
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import SIGNALS
from .risk import CURRENCY_ROLL_WINDOW, DEFAULT_RF, CurrencyExposure, currency_exposure, risk_summary, usdzar_betas
from .universe import FX_TICKERS
from .utils import TRADING_DAYS, rebalance_dates, returns_from_prices


@dataclass
class BacktestConfig:
    signal: str = "momentum"
    signal_params: dict = field(default_factory=dict)
    top_frac: float = 0.3            # fraction of the universe in each leg
    long_only: bool = True           # False -> long top / short bottom (factor portfolio)
    weighting: str = "equal"         # "equal" or "inv_vol"
    rebalance: str = "M"             # "W", "M" or "Q"
    lag: int = 1                     # trading days between signal and execution
    min_names: int = 6               # hold cash if fewer names have a valid signal
    vol_window: int = 60             # for inverse-vol weighting
    # JSE costs (fractions of traded value). Defaults are assumptions: set your own.
    stt: float = 0.0025              # Securities Transfer Tax, 0.25% on purchases
    brokerage: float = 0.0015        # brokerage + STRATE + levies, each way


@dataclass
class BacktestResult:
    returns: pd.Series               # net of costs
    gross_returns: pd.Series
    weights: pd.DataFrame            # post-trade weights at each execution date
    turnover: pd.Series              # one-way turnover per rebalance
    costs: pd.Series                 # cost drag per rebalance (fraction of NAV)
    config: BacktestConfig

    @property
    def annual_turnover(self) -> float:
        years = len(self.returns) / TRADING_DAYS
        return float(self.turnover.sum() / years)

    def summary(self, benchmark: pd.Series | None = None, rf: float = DEFAULT_RF) -> pd.Series:
        s = risk_summary(self.returns, benchmark, rf)
        s["Annual turnover (one-way)"] = self.annual_turnover
        s["Total cost drag (ann.)"] = float(self.costs.sum() / (len(self.returns) / TRADING_DAYS))
        return s


def build_weights(
    signal: pd.DataFrame,
    dates: pd.DatetimeIndex,
    vol: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Target weights at each signal date. Long leg sums to +1, short leg (if any) to -1."""
    W = pd.DataFrame(0.0, index=dates, columns=signal.columns)
    for d in dates:
        s = signal.loc[d].dropna()
        if len(s) < cfg.min_names:
            continue
        k = max(1, int(round(len(s) * cfg.top_frac)))
        legs = [(s.nlargest(k).index, 1.0)]
        if not cfg.long_only:
            legs.append((s.nsmallest(k).index, -1.0))
        for names, sign in legs:
            w = pd.Series(1.0 / len(names), index=names)
            if cfg.weighting == "inv_vol":
                inv = 1.0 / vol.loc[d, names].replace(0, np.nan)
                if inv.notna().any():  # names lacking a vol estimate get the average weight
                    inv = inv.fillna(inv.mean())
                    w = inv / inv.sum()
            W.loc[d, names] = sign * w.to_numpy()
    return W


def simulate(rets: pd.DataFrame, targets: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """Walk forward day by day: drift weights, execute trades, charge JSE costs."""
    idx, R = rets.index, rets.fillna(0.0).to_numpy()
    pos = {d: i for i, d in enumerate(idx)}
    exec_map = {}
    for d in targets.index:
        j = pos[d] + cfg.lag
        if j < len(idx):
            exec_map[idx[j]] = targets.loc[d].to_numpy()
    # Ignore leading rebalances with no valid signal (all-zero targets): the book starts at
    # the first date we actually hold something.
    live = [d for d, t in exec_map.items() if np.abs(t).sum() > 0]
    if not live:
        raise ValueError("No rebalance could be executed: not enough data for the chosen signal")
    exec_map = {d: t for d, t in exec_map.items() if d >= min(live)}

    n_days = len(idx)
    w = np.zeros(R.shape[1])
    gross, net = np.zeros(n_days), np.zeros(n_days)
    turn, cost = {}, {}
    held = {}
    for i, d in enumerate(idx):
        g = float(w @ R[i])
        w = w * (1.0 + R[i]) / (1.0 + g)  # weights drift; equity = 1 + g
        c = 0.0
        if d in exec_map:
            new = exec_map[d]
            trade = new - w
            buys, sells = trade[trade > 0].sum(), -trade[trade < 0].sum()
            c = buys * (cfg.stt + cfg.brokerage) + sells * cfg.brokerage
            turn[d], cost[d] = (buys + sells) / 2.0, c
            w = new.copy()
            held[d] = new
        gross[i], net[i] = g, g - c

    start = idx.get_loc(min(exec_map))
    sl = slice(start, None)
    return BacktestResult(
        returns=pd.Series(net[sl], index=idx[sl], name="net"),
        gross_returns=pd.Series(gross[sl], index=idx[sl], name="gross"),
        weights=pd.DataFrame(held, index=rets.columns).T,
        turnover=pd.Series(turn, name="turnover"),
        costs=pd.Series(cost, name="cost"),
        config=cfg,
    )


def run_backtest(
    prices: pd.DataFrame, cfg: BacktestConfig, benchmark: str | None = None
) -> BacktestResult:
    px = prices.drop(columns=[benchmark], errors="ignore") if benchmark else prices
    px = px.drop(columns=list(FX_TICKERS), errors="ignore")  # a currency is not a tradable name
    signal = SIGNALS[cfg.signal](px, **cfg.signal_params)
    rets = returns_from_prices(px)
    vol = rets.rolling(cfg.vol_window).std()
    targets = build_weights(signal, rebalance_dates(px.index, cfg.rebalance), vol, cfg)
    return simulate(rets, targets, cfg)


def build_factor_returns(
    prices: pd.DataFrame,
    factors: tuple[str, ...] = ("momentum", "low_vol", "reversal"),
    frac: float = 0.3,
    rebalance: str = "M",
    rf: float = DEFAULT_RF,
) -> pd.DataFrame:
    """Cost-free long/short factor returns plus an equal-weight market excess return (MKT)."""
    prices = prices.drop(columns=list(FX_TICKERS), errors="ignore")
    out = {}
    for f in factors:
        cfg = BacktestConfig(
            signal=f, top_frac=frac, long_only=False, rebalance=rebalance, stt=0.0, brokerage=0.0
        )
        out[f.upper()] = run_backtest(prices, cfg).returns
    out["MKT"] = returns_from_prices(prices).mean(axis=1) - rf / TRADING_DAYS
    return pd.DataFrame(out)


# ---------------------------------------------------------------- USD/ZAR / rand-hedge exposure
def daily_weights(res: BacktestResult, rets: pd.DataFrame) -> pd.DataFrame:
    """Start-of-day weights the strategy actually held (the weights each day's return was earned on).

    Rebuilt from the post-trade weights in ``res.weights`` with the same drift and execution
    timing as ``simulate``, so ``(daily_weights * rets).sum(axis=1)`` equals ``res.gross_returns``.
    """
    idx = res.returns.index
    cols = list(res.weights.columns)
    R = rets.reindex(index=idx, columns=cols).fillna(0.0).to_numpy()
    targets = {d: row.to_numpy() for d, row in res.weights.iterrows()}
    w = np.zeros(len(cols))
    out = np.zeros_like(R)
    for i, d in enumerate(idx):
        out[i] = w
        g = float(w @ R[i])
        w = w * (1.0 + R[i]) / (1.0 + g)
        if d in targets:
            w = targets[d].copy()
    return pd.DataFrame(out, index=idx, columns=cols)


def rand_hedge_exposure(
    res: BacktestResult,
    rets: pd.DataFrame,
    fx_ret: pd.Series,
    betas: pd.DataFrame | None = None,
    window: int = CURRENCY_ROLL_WINDOW,
) -> CurrencyExposure:
    """Is this strategy secretly a macro rand bet?

    Combines the realised USD/ZAR beta and rolling correlation of the strategy's net returns with
    a look-through of what it held each day: weight-averaged stock USD/ZAR beta and the share of
    capital in Rand Hedge / Neutral / SA Inc. names. Stock betas come from each stock's full
    history (``usdzar_betas``), so the look-through is a diagnostic of the holdings' structure, not
    a tradable signal (it uses the whole sample, i.e. is not point-in-time).
    """
    rets = rets.drop(columns=list(FX_TICKERS), errors="ignore")
    if betas is None:
        betas = usdzar_betas(rets, fx_ret)
    return currency_exposure(res.returns, fx_ret, weights=daily_weights(res, rets), betas=betas, window=window)
