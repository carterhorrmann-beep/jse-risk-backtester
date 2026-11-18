"""Price-based factor signals, rank-IC analysis and Newey-West factor regressions.

All signals are "higher = more attractive" and use only data up to and including
the date they are stamped with (no look-ahead).

Fundamental factors (value, quality) are deliberately absent: free data feeds only
provide *current* fundamentals, which would leak the future into a backtest. Add them
via a point-in-time CSV (see README roadmap).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import DEFAULT_RF
from .utils import TRADING_DAYS, newey_west_ols, rebalance_dates, returns_from_prices


# ---------------------------------------------------------------- signals
def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: total return from t-lookback to t-skip (skips the reversal month)."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def low_volatility(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Negative trailing realised volatility."""
    vol = returns_from_prices(prices).rolling(window, min_periods=int(window * 0.8)).std()
    return -vol


def reversal(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Short-term reversal: negative trailing 1-month return."""
    return -(prices / prices.shift(window) - 1.0)


SIGNALS = {"momentum": momentum, "low_vol": low_volatility, "reversal": reversal}


# ---------------------------------------------------------------- information coefficient
def information_coefficient(
    signal: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21, freq: str = "M"
) -> pd.Series:
    """Cross-sectional Spearman rank correlation between signal and forward return."""
    fwd = prices.shift(-horizon) / prices - 1.0
    out = {}
    for d in rebalance_dates(prices.index, freq):
        s, f = signal.loc[d], fwd.loc[d]
        m = s.notna() & f.notna()
        if m.sum() >= 8:
            out[d] = s[m].rank().corr(f[m].rank())
    return pd.Series(out, name="IC")


def ic_summary(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    ir = ic.mean() / ic.std(ddof=1) if len(ic) > 1 else np.nan
    return pd.Series(
        {
            "Mean IC": ic.mean(),
            "IC std": ic.std(ddof=1),
            "IC IR": ir,
            "t-stat": ir * np.sqrt(len(ic)),
            "Hit rate": (ic > 0).mean(),
            "Periods": len(ic),
        }
    )


# ---------------------------------------------------------------- factor regression
@dataclass
class RegressionResult:
    table: pd.DataFrame  # coef + Newey-West t-stat; alpha annualised
    r2: float
    n_obs: int


def factor_regression(
    y: pd.Series, X: pd.DataFrame, rf: float = DEFAULT_RF, lags: int = 5
) -> RegressionResult:
    """OLS of daily excess returns on factor returns with Newey-West (HAC) t-stats."""
    df = pd.concat([y.rename("__y__"), X], axis=1, join="inner").dropna()
    yv = df["__y__"].to_numpy() - rf / TRADING_DAYS
    Xv = np.column_stack([np.ones(len(df)), df[list(X.columns)].to_numpy()])
    beta, se, resid = newey_west_ols(yv, Xv, lags)

    table = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=["alpha"] + list(X.columns))
    table.loc["alpha", "coef"] *= TRADING_DAYS  # annualise intercept
    r2 = 1.0 - resid.var() / yv.var()
    return RegressionResult(table=table, r2=float(r2), n_obs=len(df))
