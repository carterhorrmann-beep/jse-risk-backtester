import warnings

import numpy as np
import pandas as pd
import pytest

import jse_backtester.data as data
from jse_backtester.universe import DEFAULT_BENCHMARK, JSE_TOP_TICKERS


def _fake_yahoo(drop):
    def fake(tickers, start, end, cache_dir):
        px = data.synthetic_prices([t for t in tickers if t != DEFAULT_BENCHMARK], "2020-01-01", "2021-12-31")
        return px.drop(columns=drop, errors="ignore")
    return fake


def test_universe_uses_the_jse_ticker_for_bhp():
    assert "BHG.JO" in JSE_TOP_TICKERS and "BHP.JO" not in JSE_TOP_TICKERS


def test_failed_tickers_are_skipped_with_a_warning(monkeypatch):
    monkeypatch.setattr(data, "_from_yfinance", _fake_yahoo(["BHG.JO"]))
    with pytest.warns(UserWarning, match=r"No Yahoo data for \['BHG.JO'\]"):
        px = data.load_prices(source="yfinance")
    assert "BHG.JO" not in px.columns and px.shape[1] >= 30


def test_refuses_to_continue_when_almost_everything_failed(monkeypatch):
    keep = JSE_TOP_TICKERS[:3]
    monkeypatch.setattr(data, "_from_yfinance", _fake_yahoo([t for t in JSE_TOP_TICKERS if t not in keep]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="refusing to continue"):
            data.load_prices(source="yfinance")


# ---------------------------------------------------------------- USD/ZAR ingestion
from jse_backtester.universe import USDZAR_TICKER  # noqa: E402


def _recording_yahoo(calls, drop=()):
    def fake(tickers, start, end, cache_dir):
        calls.append(list(tickers))
        stocks = [t for t in tickers if t not in (DEFAULT_BENCHMARK, USDZAR_TICKER)]
        px = data.synthetic_prices(stocks, "2020-01-01", "2021-12-31", include_fx=True)
        return px.drop(columns=list(drop), errors="ignore")
    return fake


def test_yfinance_source_requests_usdzar_alongside_the_universe(monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_from_yfinance", _recording_yahoo(calls))
    px = data.load_prices(source="yfinance")
    assert USDZAR_TICKER in calls[0] and DEFAULT_BENCHMARK in calls[0]
    assert set(JSE_TOP_TICKERS) <= set(calls[0])
    assert USDZAR_TICKER in px.columns


def test_usdzar_can_be_switched_off(monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_from_yfinance", _recording_yahoo(calls, drop=[USDZAR_TICKER]))
    px = data.load_prices(source="yfinance", with_fx=False)
    assert USDZAR_TICKER not in calls[0] and USDZAR_TICKER not in px.columns
    assert USDZAR_TICKER not in data.load_prices(source="synthetic", with_fx=False).columns


def test_missing_usdzar_warns_that_the_currency_sections_are_skipped(monkeypatch):
    monkeypatch.setattr(data, "_from_yfinance", _recording_yahoo([], drop=[USDZAR_TICKER]))
    with pytest.warns(UserWarning, match=r"No Yahoo data for \['USDZAR=X'\].*USD/ZAR exposure sections are skipped"):
        px = data.load_prices(source="yfinance")
    assert USDZAR_TICKER not in px.columns and px.shape[1] >= 30


def test_synthetic_source_ships_usdzar_for_offline_runs():
    px = data.load_prices(source="synthetic", start="2019-01-01", end="2020-12-31")
    assert USDZAR_TICKER in px.columns and DEFAULT_BENCHMARK in px.columns
    assert 8 < px[USDZAR_TICKER].min() and px[USDZAR_TICKER].max() < 24  # rand, not cents


def test_split_fx_removes_every_currency_column():
    px = data.load_prices(source="synthetic", start="2019-01-01", end="2020-12-31")
    stocks, fx = data.split_fx(px)
    assert USDZAR_TICKER not in stocks.columns and stocks.shape[1] == px.shape[1] - 1
    assert fx.name == USDZAR_TICKER and fx.index.equals(px[USDZAR_TICKER].dropna().index)
    stocks2, fx2 = data.split_fx(stocks)
    assert fx2 is None and stocks2.equals(stocks)


def test_fx_only_days_are_dropped_so_stock_and_fx_calendars_match():
    """USD/ZAR trades on JSE holidays; those rows must not survive, or stock returns would show a
    spurious zero day (forward-filled) against a real FX move."""
    idx = pd.bdate_range("2021-03-01", periods=12)
    holiday = idx[5]
    stock = pd.Series(np.linspace(100, 111, 12), index=idx, name="SBK.JO")
    fx = pd.Series(np.linspace(14.0, 15.1, 12), index=idx, name=USDZAR_TICKER)
    px = pd.DataFrame({"SBK.JO": stock, USDZAR_TICKER: fx})
    px.loc[holiday, "SBK.JO"] = np.nan         # JSE closed, FX still trading
    clean = data.clean_prices(px)
    assert holiday not in clean.index and len(clean) == 11
    assert clean["SBK.JO"].notna().all() and clean[USDZAR_TICKER].notna().all()
    # the FX move over the holiday lands in the next session, like the stock's own return
    nxt = idx[6]
    assert clean[USDZAR_TICKER].loc[nxt] == fx.loc[nxt]


def test_frames_without_fx_are_cleaned_exactly_as_before():
    idx = pd.bdate_range("2021-03-01", periods=8)
    px = pd.DataFrame({"A.JO": np.linspace(100, 108, 8), "B.JO": np.linspace(50, 58, 8)}, index=idx)
    px.iloc[3, 0] = np.nan                     # an isolated gap is forward-filled, never dropped
    clean = data.clean_prices(px)
    assert len(clean) == 8 and clean["A.JO"].iloc[3] == px["A.JO"].iloc[2]
