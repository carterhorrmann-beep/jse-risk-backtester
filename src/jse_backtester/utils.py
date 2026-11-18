from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with Newey-West (Bartlett kernel) HAC standard errors.

    ``X`` must already contain any intercept column. Returns (coefficients, standard errors,
    residuals).
    """
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    Z = X * resid[:, None]
    S = Z.T @ Z
    for lag in range(1, lags + 1):
        G = Z[lag:].T @ Z[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ S @ bread))
    return beta, se, resid


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. NaN where either price is missing (no implicit fill)."""
    return prices / prices.shift(1) - 1.0


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each period. freq: 'W', 'M' or 'Q'."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(freq)).max().to_numpy())
