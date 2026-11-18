"""Price loading: Yahoo Finance (cached), CSV, or a synthetic generator for tests/demos."""
from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .sanitize import EVENT_COLS, sanitize_units, summarize_events
from .universe import DEFAULT_BENCHMARK, FX_TICKERS, JSE_TOP_TICKERS, SECTORS, USDZAR_TICKER


def clean_prices(px: pd.DataFrame, sanitize: bool = True) -> pd.DataFrame:
    """Drop empty columns/rows and non-positive prices, repair Yahoo ZAc/ZAR glitches,
    then bridge short gaps (<=5 days). Read sanitiser events with ``glitch_events(px)``."""
    px = px.sort_index().where(lambda d: d > 0)
    px = px.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # FX trades on days the JSE is shut (public holidays). Drop those rows so stock returns and
    # USD/ZAR changes cover exactly the same days; the FX move over a holiday then lands in the
    # next JSE session's change, matching the stock's own multi-day return.
    equity_cols = [c for c in px.columns if c not in FX_TICKERS]
    if equity_cols and len(equity_cols) < px.shape[1]:
        px = px.loc[px[equity_cols].notna().any(axis=1)]
    events = pd.DataFrame(columns=EVENT_COLS)
    if sanitize:
        px, events = sanitize_units(px)
        msg = summarize_events(events)
        if msg:
            warnings.warn(f"Data sanitiser: {msg}", UserWarning, stacklevel=2)
    px = px.ffill(limit=5)
    # Plain records, NOT a DataFrame: pandas compares .attrs with == inside concat/nlargest,
    # which raises "truth value of a DataFrame is ambiguous" for DataFrame values.
    px.attrs["glitches"] = events.to_dict("records")
    return px


def glitch_events(px: pd.DataFrame) -> pd.DataFrame:
    """Sanitiser events recorded on a price frame returned by load_prices/clean_prices."""
    return pd.DataFrame(px.attrs.get("glitches", []), columns=EVENT_COLS)


def split_fx(px: pd.DataFrame, ticker: str = USDZAR_TICKER) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate currency columns from the share prices.

    Returns (prices without any FX columns, ``ticker`` level series or None when absent). Call
    this before ranking, equal-weighting or covariance work so USD/ZAR is never treated as a stock.
    """
    fx = px[ticker].dropna().rename(ticker) if ticker in px.columns else None
    rest = px.drop(columns=[c for c in FX_TICKERS if c in px.columns])
    return rest, fx


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    source: str = "yfinance",
    benchmark: str = DEFAULT_BENCHMARK,
    cache_dir: str | Path = ".cache",
    csv_path: str | Path | None = None,
    seed: int = 42,
    with_fx: bool = True,
) -> pd.DataFrame:
    """Return adjusted close prices (columns = tickers + benchmark + ``USDZAR=X``).

    ``USDZAR=X`` (rand per dollar) rides along for the currency / rand-hedge analysis; use
    ``split_fx`` to take it out before treating the frame as a stock universe. With
    ``with_fx=False`` it is neither downloaded nor simulated.
    """
    tickers = list(tickers or JSE_TOP_TICKERS)
    if source == "synthetic":
        return synthetic_prices(tickers, start, end or "2025-12-31", benchmark, seed, include_fx=with_fx)
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when source='csv'")
        px = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return clean_prices(px)
    if source == "yfinance":
        wanted = tickers + [benchmark] + ([USDZAR_TICKER] if with_fx else [])
        px = _from_yfinance(wanted, start, end, Path(cache_dir))
        missing = [t for t in wanted if t not in px.columns]
        if missing:  # delisted / renamed / unavailable: carry on without them
            fx_note = (f" Without {USDZAR_TICKER} the USD/ZAR exposure sections are skipped."
                       if USDZAR_TICKER in missing else "")
            warnings.warn(f"No Yahoo data for {missing}; continuing without them "
                          f"(check for ticker changes or delistings).{fx_note}", UserWarning, stacklevel=2)
        if px.shape[1] < 6:
            raise RuntimeError(f"Only {px.shape[1]} tickers returned data; refusing to continue")
        return px
    raise ValueError(f"unknown source {source!r}")


def _from_yfinance(tickers: list[str], start: str, end: str | None, cache_dir: Path) -> pd.DataFrame:
    """Download adjusted closes. The *raw* download is cached (24h); sanitising runs on every
    load, so sanitiser upgrades apply without re-downloading and glitches can't hide in the cache."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install yfinance (or use --source synthetic)") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    path = cache_dir / f"raw_{key}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 24 * 3600:
        return clean_prices(pd.read_csv(path, index_col=0, parse_dates=True))

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (offline? try --source synthetic)")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px.to_csv(path)  # raw, unsanitised
    return clean_prices(px)


# True USD/ZAR sensitivities used only by the simulator (per cent move in the stock per 1% rise
# in rand per dollar). Sector defaults with a few well-known rand hedges / domestic names.
# The univariate OLS betas the analytics report differ a little: the simulated rand also moves
# against the market (risk-off weakens the rand), exactly as in real data.
_SIM_FX_SECTOR = {"Resources": 0.9, "Financials": -0.45, "Industrials": -0.40,
                  "Tech & Telecom": 0.20, "Property": -0.35}
_SIM_FX_NAME = {"NPN.JO": 0.85, "PRX.JO": 0.85, "BTI.JO": 0.90, "CFR.JO": 0.65, "MNP.JO": 0.85,
                "BID.JO": 0.65, "GLN.JO": 1.0, "S32.JO": 1.0, "BHG.JO": 1.0, "AGL.JO": 0.9,
                "REM.JO": 0.20, "BVT.JO": 0.20, "MTN.JO": 0.20, "VOD.JO": 0.0, "SOL.JO": 0.6}


def synthetic_usdzar(
    idx: pd.DatetimeIndex, market_shock: np.ndarray, rng: np.random.Generator, rho: float = 0.20
) -> pd.Series:
    """Simulated USD/ZAR (rand per dollar), roughly 12-20 with a gentle rand-weakening trend.

    Fat-tailed daily moves (Student-t, 5 d.o.f.) with slowly varying volatility around ~13%
    annualised, a *negative* link to the equity market (a risk-off day weakens the rand, so
    USD/ZAR rises when stocks fall; correlation ``rho``), and mean reversion around a log-linear
    trend (12.5 -> 18.2) so the level stays in a realistic band. Not market data.
    """
    T = len(idx)
    z = (market_shock - market_shock.mean()) / market_shock.std()
    eps = rng.standard_t(5, T) / np.sqrt(5 / 3)  # unit variance
    shock = -rho * z + np.sqrt(1 - rho**2) * eps
    logv = np.zeros(T)
    for t in range(1, T):  # slowly varying volatility regime
        logv[t] = 0.98 * logv[t - 1] + rng.normal(0, 0.08)
    sigma = 0.0085 * np.exp(logv - 0.5 * logv.var())
    target = np.log(12.5) + (np.log(18.2) - np.log(12.5)) * np.arange(T) / max(T - 1, 1)
    kappa, x = 0.003, np.empty(T)
    x[0] = target[0]
    for t in range(1, T):
        x[t] = x[t - 1] + (target[t] - target[t - 1]) + kappa * (target[t - 1] - x[t - 1]) + sigma[t] * shock[t]
    return pd.Series(np.exp(x), index=idx, name=USDZAR_TICKER)


def synthetic_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    benchmark: str = DEFAULT_BENCHMARK,
    seed: int = 42,
    include_fx: bool = False,
) -> pd.DataFrame:
    """Simulated prices: market + sector factors + slowly-mean-reverting stock drift + noise.

    The persistent drift component gives momentum something to find. This is NOT market
    data: use it only for tests, CI and demos.

    With ``include_fx=True`` a simulated ``USDZAR=X`` column is appended and every stock also
    loads on it (resources / offshore earners positively = Rand Hedges, domestic banks and
    retailers negatively = SA Inc.), so the rand-hedge analytics have something real to find.
    The FX draws come from a separate generator: with ``include_fx=False`` the output is
    bit-identical to earlier versions.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T, n = len(idx), len(tickers)
    sector_names = sorted({SECTORS.get(t, "Other") for t in tickers})
    sec_idx = np.array([sector_names.index(SECTORS.get(t, "Other")) for t in tickers])

    mkt = rng.normal(0.0003, 0.009, T)
    sec = rng.normal(0.0, 0.007, (T, len(sector_names)))
    beta = rng.uniform(0.6, 1.4, n)
    idio_vol = rng.uniform(0.012, 0.025, n)

    mu = np.zeros(n)
    rets = np.empty((T, n))
    for t in range(T):
        mu = 0.995 * mu + rng.normal(0, 0.00004, n)  # OU drift, ~6 month half-life
        rets[t] = beta * mkt[t] + sec[t, sec_idx] + mu + rng.normal(0, idio_vol)

    fx = None
    if include_fx:
        fx_rng = np.random.default_rng([seed, 7919])  # independent stream
        fx = synthetic_usdzar(idx, mkt, fx_rng)
        load = np.array([_SIM_FX_NAME.get(t, _SIM_FX_SECTOR.get(SECTORS.get(t, "Other"), 0.0)) for t in tickers])
        load = load + fx_rng.normal(0, 0.10, n)
        rets = rets + np.outer(fx.pct_change().fillna(0.0).to_numpy(), load)

    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
    px[benchmark] = 100 * np.cumprod(1 + rets.mean(axis=1))
    if fx is not None:
        px[USDZAR_TICKER] = fx
    return px
