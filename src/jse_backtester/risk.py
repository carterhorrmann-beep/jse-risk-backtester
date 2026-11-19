"""Portfolio risk analytics: performance ratios, VaR/CVaR, drawdowns, risk decomposition."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .universe import FX_TICKERS, sector_weights
from .utils import TRADING_DAYS, newey_west_ols

# Approximate SA cash rate (3m T-bill / JIBAR area). Update to the prevailing rate.
DEFAULT_RF = 0.07

# Illustrative JSE stress windows (inclusive). Edit freely.
SCENARIOS: dict[str, tuple[str, str]] = {
    "Nenegate (Dec 2015)": ("2015-12-08", "2015-12-18"),
    "Steinhoff collapse (Dec 2017)": ("2017-12-05", "2017-12-29"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
}


# ---------------------------------------------------------------- return / risk ratios
def annualised_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def annualised_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = DEFAULT_RF) -> float:
    ex = r.dropna() - rf / TRADING_DAYS
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: float = DEFAULT_RF) -> float:
    ex = r.dropna() - rf / TRADING_DAYS
    downside = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    return float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


# ---------------------------------------------------------------- VaR / CVaR (as positive loss)
def historical_var(r: pd.Series, level: float = 0.95) -> float:
    return float(-np.quantile(r.dropna(), 1 - level))


def historical_cvar(r: pd.Series, level: float = 0.95) -> float:
    r = r.dropna()
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def parametric_var(r: pd.Series, level: float = 0.95) -> float:
    z = NormalDist().inv_cdf(1 - level)
    r = r.dropna()
    return float(-(r.mean() + z * r.std(ddof=1)))


def cornish_fisher_var(r: pd.Series, level: float = 0.95) -> float:
    """VaR adjusted for skew and excess kurtosis (better for fat-tailed JSE equities)."""
    r = r.dropna()
    z = NormalDist().inv_cdf(1 - level)
    s, k = r.skew(), r.kurt()  # pandas kurt() is excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(r.mean() + z_cf * r.std(ddof=1)))


# ---------------------------------------------------------------- relative to benchmark
def beta_alpha(r: pd.Series, bench: pd.Series, rf: float = DEFAULT_RF) -> tuple[float, float, float]:
    """CAPM beta, annualised Jensen alpha and R^2 from daily excess returns."""
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    df.columns = ["p", "b"]
    y, x = df["p"] - rf / TRADING_DAYS, df["b"] - rf / TRADING_DAYS
    beta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    alpha = (y.mean() - beta * x.mean()) * TRADING_DAYS
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return float(beta), float(alpha), float(r2)


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    active = (r - bench).dropna()
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    active = (r - bench).dropna()
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(active.mean() * TRADING_DAYS / te)


# ---------------------------------------------------------------- summaries
def risk_summary(
    r: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = DEFAULT_RF,
    level: float = 0.95,
) -> pd.Series:
    r = r.dropna()
    pct = f"{level:.0%}"
    cagr = annualised_return(r)
    mdd = max_drawdown(r)
    out = {
        "CAGR": cagr,
        "Ann. volatility": annualised_vol(r),
        "Sharpe": sharpe(r, rf),
        "Sortino": sortino(r, rf),
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        f"1d VaR {pct} (historical)": historical_var(r, level),
        f"1d VaR {pct} (Cornish-Fisher)": cornish_fisher_var(r, level),
        f"1d CVaR {pct} (historical)": historical_cvar(r, level),
        "Skew": float(r.skew()),
        "Excess kurtosis": float(r.kurt()),
    }
    if benchmark is not None:
        b = benchmark.dropna()
        beta, alpha, r2 = beta_alpha(r, b, rf)
        out.update(
            {
                "Beta vs benchmark": beta,
                "Alpha (ann.)": alpha,
                "R^2 vs benchmark": r2,
                "Tracking error": tracking_error(r, b),
                "Information ratio": information_ratio(r, b),
            }
        )
    return pd.Series(out)


def scenario_table(r: pd.Series, scenarios: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Compounded return and worst drawdown of a return series inside each stress window."""
    rows = {}
    for name, (a, b) in (scenarios or SCENARIOS).items():
        w = r.loc[a:b].dropna()
        if len(w) >= 3:
            rows[name] = {"Return": float((1 + w).prod() - 1), "Max drawdown": max_drawdown(w), "Days": len(w)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- USD/ZAR & rand-hedge decomposition
# USD/ZAR is quoted in rand per dollar, so a *rise* is a weaker rand. A stock with a positive
# beta earns more when the rand falls (miners, Naspers/Prosus, BAT, Richemont, Bidcorp...): a
# Rand Hedge. A negative beta is SA Inc.: domestic banks, retailers, property, hurt by a weak rand.
USDZAR_HEDGE_THRESHOLD = 0.20   # |beta| above this leaves the Neutral bucket
CURRENCY_ROLL_WINDOW = 126      # trading days (~6 months) for the rolling correlation
BUCKET_HEDGE, BUCKET_NEUTRAL, BUCKET_DOMESTIC = "Rand Hedge", "Neutral", "SA Inc."
BUCKET_UNKNOWN = "Unclassified"  # too little history to estimate a beta
BUCKETS = (BUCKET_HEDGE, BUCKET_NEUTRAL, BUCKET_DOMESTIC)


def classify_usdzar_beta(beta: float, threshold: float = USDZAR_HEDGE_THRESHOLD) -> str:
    """Rand Hedge if beta > +threshold; SA Inc. if beta < -threshold; Neutral in [-t, +t]."""
    if pd.isna(beta):
        return BUCKET_UNKNOWN
    if beta > threshold:
        return BUCKET_HEDGE
    if beta < -threshold:
        return BUCKET_DOMESTIC
    return BUCKET_NEUTRAL


def usdzar_betas(
    returns: pd.DataFrame,
    fx_ret: pd.Series,
    lags: int = 5,
    min_obs: int = 250,
    threshold: float = USDZAR_HEDGE_THRESHOLD,
) -> pd.DataFrame:
    """USD/ZAR beta of each stock: OLS of its daily ZAR return on the daily % change in USDZAR=X.

    Returns one row per ticker (sorted, most rand-hedged first) with ``beta``, Newey-West
    ``t_stat`` (``lags`` lags), ``r2``, ``n_obs`` and the ``bucket`` (see classify_usdzar_beta).
    Each stock uses its own history (pairwise with the FX series); fewer than ``min_obs`` common
    days gives NaN / "Unclassified". The beta is univariate on purpose: it is the stock's total
    sensitivity to the rand, including whatever the rand shares with the market.
    """
    fx = fx_ret.dropna().rename("__fx__")
    rows: dict[str, tuple] = {}
    for t in returns.columns:
        if t in FX_TICKERS:
            continue
        df = pd.concat([returns[t].rename("__y__"), fx], axis=1, join="inner").dropna()
        if len(df) < min_obs:
            rows[t] = (np.nan, np.nan, np.nan, len(df))
            continue
        y, x = df["__y__"].to_numpy(), df["__fx__"].to_numpy()
        beta, se, resid = newey_west_ols(y, np.column_stack([np.ones(len(x)), x]), lags)
        rows[t] = (beta[1], beta[1] / se[1], 1.0 - resid.var() / y.var(), len(df))
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["beta", "t_stat", "r2", "n_obs"])
    out["bucket"] = out["beta"].map(lambda b: classify_usdzar_beta(b, threshold))
    return out.sort_values("beta", ascending=False)


def rolling_correlation(r: pd.Series, fx_ret: pd.Series, window: int = CURRENCY_ROLL_WINDOW) -> pd.Series:
    """Rolling correlation of a return series with USD/ZAR % changes (first value after ``window`` days)."""
    df = pd.concat([r.rename("r"), fx_ret.rename("fx")], axis=1, join="inner").dropna()
    return df["r"].rolling(window).corr(df["fx"]).dropna().rename(f"{window}d corr")


def equal_weight_path(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily weights of the equal-weight universe (1/n over the names with a return that day)."""
    avail = returns.notna()
    return avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


@dataclass
class CurrencyExposure:
    """USD/ZAR exposure of one return stream, optionally decomposed by holding.

    ``summary`` holds the realised regression of the stream's returns on USD/ZAR changes and its
    rolling correlation; ``lookthrough_beta`` is the weight-averaged stock beta (what the
    *holdings* say the exposure should be); ``bucket_weights`` is the average net weight sitting
    in each bucket and ``bucket_gross`` the average gross (they differ only for long/short books).
    """
    summary: pd.Series
    rolling_corr: pd.Series
    betas: pd.DataFrame | None = None
    lookthrough_beta: pd.Series | None = None      # per day (constant for a static portfolio)
    bucket_weights: pd.Series | None = None
    bucket_gross: pd.Series | None = None
    window: int = CURRENCY_ROLL_WINDOW


def currency_exposure(
    returns: pd.Series,
    fx_ret: pd.Series,
    weights: pd.Series | pd.DataFrame | None = None,
    betas: pd.DataFrame | None = None,
    window: int = CURRENCY_ROLL_WINDOW,
    lags: int = 5,
) -> CurrencyExposure:
    """Realised USD/ZAR beta, rolling correlation and (given weights + stock betas) a bucket split.

    ``weights`` is a static Series, or a dates x tickers frame of start-of-day weights (use
    ``backtest.daily_weights``); it needs ``betas`` (from ``usdzar_betas``) to be decomposed.
    Days with no position are ignored in the bucket averages.
    """
    if (weights is None) != (betas is None):
        raise ValueError("weights and betas must be given together (look-through needs both)")
    df = pd.concat([returns.rename("r"), fx_ret.rename("fx")], axis=1, join="inner").dropna()
    nan = float("nan")
    out = {"USD/ZAR Beta (realised)": nan, "Beta t-stat (Newey-West)": nan, "R^2 vs USD/ZAR": nan,
           "Corr with USD/ZAR (full sample)": nan, "Observations": float(len(df))}
    if len(df) > 30:
        y, x = df["r"].to_numpy(), df["fx"].to_numpy()
        b, se, resid = newey_west_ols(y, np.column_stack([np.ones(len(x)), x]), lags)
        out.update({"USD/ZAR Beta (realised)": b[1], "Beta t-stat (Newey-West)": b[1] / se[1],
                    "R^2 vs USD/ZAR": 1.0 - resid.var() / y.var(),
                    "Corr with USD/ZAR (full sample)": float(np.corrcoef(x, y)[0, 1])})
    roll = rolling_correlation(returns, fx_ret, window)
    tag = f"Rolling {window}d corr"
    out.update({f"{tag} (latest)": float(roll.iloc[-1]) if len(roll) else nan,
                f"{tag} (average)": float(roll.mean()) if len(roll) else nan,
                f"{tag} (min)": float(roll.min()) if len(roll) else nan,
                f"{tag} (max)": float(roll.max()) if len(roll) else nan})

    look = bw = bg = None
    out["Look-through USD/ZAR Beta (weighted)"] = nan
    if weights is not None and betas is not None:
        cols = [t for t in betas.index if t not in FX_TICKERS]
        if isinstance(weights, pd.Series):
            W = pd.DataFrame(np.tile(weights.reindex(cols).fillna(0.0).to_numpy(), (len(df), 1)),
                             index=df.index, columns=cols)
        else:
            W = weights.reindex(index=df.index, columns=cols).fillna(0.0)
        W = W.loc[W.abs().sum(axis=1) > 0]
        if not W.empty:
            b_vec = betas.loc[cols, "beta"].fillna(0.0)  # unknown betas count as 0 (see 'Unclassified')
            look = (W @ b_vec).rename("look-through beta")
            bucket = betas.loc[cols, "bucket"]
            names = [b for b in (*BUCKETS, BUCKET_UNKNOWN) if (bucket == b).any()]
            bw = pd.Series({b: W.loc[:, bucket == b].sum(axis=1).mean() for b in names})
            bg = pd.Series({b: W.loc[:, bucket == b].abs().sum(axis=1).mean() for b in names})
            out["Look-through USD/ZAR Beta (weighted)"] = float(look.mean())
    return CurrencyExposure(pd.Series(out), roll, betas, look, bw, bg, window)


# ---------------------------------------------------------------- covariance & risk decomposition
def shrunk_cov(returns: pd.DataFrame, shrinkage: float = 0.1) -> pd.DataFrame:
    """Annualised sample covariance shrunk toward a constant-correlation target."""
    S = returns.cov() * TRADING_DAYS
    vol = np.sqrt(np.diag(S))
    corr = S.to_numpy() / np.outer(vol, vol)
    n = len(S)
    rho = (corr.sum() - n) / (n * (n - 1))
    F = rho * np.outer(vol, vol)
    np.fill_diagonal(F, np.diag(S))
    return pd.DataFrame((1 - shrinkage) * S.to_numpy() + shrinkage * F, index=S.index, columns=S.columns)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility. 'pct' sums to 1."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    marginal = cov.to_numpy() @ w / vol
    contrib = w * marginal
    return pd.DataFrame(
        {"weight": w, "marginal": marginal, "contribution": contrib, "pct": contrib / vol},
        index=cov.index,
    )


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted-average stock volatility over portfolio volatility (1.0 = no diversification)."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    port_vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    return float((w * np.sqrt(np.diag(cov))).sum() / port_vol)


# Trailing-window estimation (used to re-estimate the covariance before each backtest rebalance)
TRAILING_WINDOW = 252   # trading days
MIN_TRAILING_OBS = 126  # fewer usable days than this and the caller should not trust a covariance


def trailing_returns(
    returns: pd.DataFrame,
    end: pd.Timestamp,
    window: int = TRAILING_WINDOW,
    min_obs: int = MIN_TRAILING_OBS,
    names=None,
) -> pd.DataFrame:
    """The last ``window`` daily returns up to and including ``end`` (nothing later is ever read).

    Only names with a *complete* window are kept (a stock that listed or resumed trading part-way
    through the window would otherwise force a pairwise, possibly non-positive-definite covariance).
    Returns an empty frame when fewer than ``min_obs`` days of history exist.
    """
    hist = returns.loc[:end]
    if names is not None:
        hist = hist[list(names)]
    hist = hist.dropna(how="all").tail(window)  # drops the all-NaN first return row too
    if len(hist) < min_obs:
        return hist.iloc[0:0, 0:0]
    return hist.loc[:, hist.notna().all()]


def trailing_shrunk_cov(
    returns: pd.DataFrame,
    end: pd.Timestamp,
    window: int = TRAILING_WINDOW,
    shrinkage: float = 0.1,
    min_obs: int = MIN_TRAILING_OBS,
    names=None,
) -> pd.DataFrame | None:
    """``shrunk_cov`` of the trailing window ending at ``end`` (no look-ahead); None if < 2 usable names."""
    win = trailing_returns(returns, end, window, min_obs, names)
    return shrunk_cov(win, shrinkage) if win.shape[1] >= 2 else None


@dataclass
class PortfolioRiskReport:
    summary: pd.Series
    contributions: pd.DataFrame
    sectors: pd.Series
    portfolio_returns: pd.Series
    diversification_ratio: float
    sample: tuple[pd.Timestamp, pd.Timestamp]
    currency: CurrencyExposure | None = None  # set when fx_returns is supplied


def portfolio_risk_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    benchmark: pd.Series | None = None,
    rf: float = DEFAULT_RF,
    level: float = 0.95,
    shrinkage: float = 0.1,
    fx_returns: pd.Series | None = None,
    window: int = CURRENCY_ROLL_WINDOW,
) -> PortfolioRiskReport:
    """Risk report for a static-weight portfolio (weights normalised to sum to 1).

    Pass ``fx_returns`` (daily % changes in USDZAR=X) to add the USD/ZAR exposure & rand-hedge
    breakdown. Holding betas are estimated on the same common sample as the portfolio, so the
    realised portfolio beta equals the weighted sum of holding betas (OLS is linear).
    """
    w = weights[weights.index.isin(returns.columns)]
    if w.empty:
        raise ValueError("None of the portfolio tickers are in the price data")
    w = w / w.sum()
    rets = returns[w.index].dropna()  # common sample across holdings
    port = rets @ w
    cov = shrunk_cov(rets, shrinkage)
    contrib = risk_contributions(w, cov).sort_values("pct", ascending=False)
    div_ratio = diversification_ratio(w, cov)
    currency = None
    if fx_returns is not None:
        betas = usdzar_betas(rets, fx_returns)
        currency = currency_exposure(port, fx_returns, weights=w, betas=betas, window=window)
    return PortfolioRiskReport(
        summary=risk_summary(port, benchmark, rf, level),
        contributions=contrib,
        sectors=sector_weights(w),
        portfolio_returns=port,
        diversification_ratio=div_ratio,
        sample=(rets.index[0], rets.index[-1]),
        currency=currency,
    )
