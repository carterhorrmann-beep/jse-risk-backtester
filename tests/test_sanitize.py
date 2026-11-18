import numpy as np
import pandas as pd
import pytest

from jse_backtester.backtest import BacktestConfig, run_backtest
from jse_backtester.data import clean_prices, glitch_events, load_prices
from jse_backtester.sanitize import sanitize_series, sanitize_units
from jse_backtester.utils import returns_from_prices


def _base(n=600, seed=0, start=15000.0):
    """A well-behaved ZAc price series (daily vol ~1.5%)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx, name="SBK.JO")


def _kinds(events):
    return [e[2] for e in events]


def test_clean_series_is_untouched():
    s = _base()
    fixed, ev = sanitize_series(s, "SBK.JO")
    pd.testing.assert_series_equal(fixed, s)
    assert ev == []


@pytest.mark.parametrize("factor", [0.01, 100.0])
def test_isolated_one_day_flip_is_repaired(factor):
    s = _base()
    g = s.copy()
    g.iloc[100] *= factor  # one day served in the wrong unit
    fixed, ev = sanitize_series(g, "SBK.JO")
    np.testing.assert_allclose(fixed.to_numpy(), s.to_numpy(), rtol=1e-9)
    assert _kinds(ev).count("unit_flip") == 2  # into and out of the glitch


def test_multi_day_flip_run_is_repaired():
    s = _base()
    g = s.copy()
    g.iloc[200:206] *= 100.0
    fixed, _ = sanitize_series(g)
    np.testing.assert_allclose(fixed.to_numpy(), s.to_numpy(), rtol=1e-9)


def test_persistent_tail_in_wrong_unit_is_repaired_to_dominant_unit():
    s = _base()
    g = s.copy()
    g.iloc[-40:] *= 0.01  # last 40 days in rands, 560 days in cents
    fixed, ev = sanitize_series(g)
    np.testing.assert_allclose(fixed.to_numpy(), s.to_numpy(), rtol=1e-9)
    assert _kinds(ev) == ["unit_flip"]


def test_glitch_at_start_of_history_is_repaired():
    s = _base()
    g = s.copy()
    g.iloc[:3] *= 100.0
    fixed, _ = sanitize_series(g)
    np.testing.assert_allclose(fixed.to_numpy(), s.to_numpy(), rtol=1e-9)


def test_flip_on_a_day_with_a_real_move_preserves_the_real_return():
    s = _base()
    s.iloc[50:] *= 0.97   # a genuine -3% day at t=50 ...
    g = s.copy()
    g.iloc[50] *= 100.0   # ... that Yahoo also served in the wrong unit
    fixed, _ = sanitize_series(g)
    np.testing.assert_allclose(fixed.to_numpy(), s.to_numpy(), rtol=1e-9)
    # only the 100x factor is removed; the real move through the glitch day survives
    assert fixed.iloc[50] / fixed.iloc[49] == pytest.approx(s.iloc[50] / s.iloc[49])


def test_genuine_crash_and_big_rally_are_not_touched_as_unit_flips():
    s = _base()
    g = s.copy()
    g.iloc[300:] *= 0.55   # persistent -45% (corporate action / Steinhoff-style crash)
    g.iloc[450:] *= 1.30   # persistent +30%
    fixed, ev = sanitize_series(g)
    pd.testing.assert_series_equal(fixed, g)
    assert "unit_flip" not in _kinds(ev) and "spike_removed" not in _kinds(ev)
    assert _kinds(ev).count("flag_only") >= 1  # big persistent moves are flagged for review only


def test_reverting_bad_print_is_removed_and_filled():
    s = _base()
    g = s.copy()
    g.iloc[400] *= 1.8  # +80% print that reverts next day
    fixed, ev = sanitize_series(g)
    assert np.isnan(fixed.iloc[400])
    assert _kinds(ev) == ["spike_removed"]
    mask = np.arange(len(s)) != 400
    np.testing.assert_allclose(fixed.to_numpy()[mask], s.to_numpy()[mask], rtol=1e-12)
    filled = clean_prices(pd.DataFrame({"X": g}))["X"]
    assert filled.iloc[400] == pytest.approx(s.iloc[399])  # forward-filled


def test_frame_level_returns_are_sane_after_sanitising():
    bench = _base(seed=1, start=6000.0).rename("STX40.JO")
    sbk = _base(seed=2).rename("SBK.JO")
    bad = pd.DataFrame({"SBK.JO": sbk, "STX40.JO": bench})
    bad.iloc[100, 0] *= 0.01
    bad.iloc[300:303, 1] *= 100.0
    raw_ret = returns_from_prices(bad)
    assert raw_ret.abs().max().max() > 50  # the +9,900% symptom

    with pytest.warns(UserWarning, match="Data sanitiser"):
        px = clean_prices(bad)
    ret = returns_from_prices(px)
    assert ret.abs().max().max() < 0.12
    ann_vol = ret.std() * np.sqrt(252)
    assert (ann_vol < 0.35).all()  # was ~375% / ~3000% before
    assert set(glitch_events(px)["ticker"]) == {"SBK.JO", "STX40.JO"}


def test_csv_loader_runs_sanitiser(tmp_path):
    df = pd.DataFrame({"AAA.JO": _base(seed=3), "BBB.JO": _base(seed=4)})
    df.iloc[10, 1] *= 100.0
    path = tmp_path / "prices.csv"
    df.to_csv(path)
    with pytest.warns(UserWarning, match="Data sanitiser"):
        px = load_prices(source="csv", csv_path=path)
    assert returns_from_prices(px).abs().max().max() < 0.12
    assert len(glitch_events(px)) == 2


def test_sanitize_units_returns_empty_event_frame_for_clean_data():
    px = pd.DataFrame({"A": _base(seed=5), "B": _base(seed=6)})
    out, ev = sanitize_units(px)
    pd.testing.assert_frame_equal(out, px)
    assert ev.empty and list(ev.columns) == ["ticker", "date", "kind", "detail"]


def test_sanitised_prices_flow_through_the_backtester():
    """Regression: events used to live in px.attrs as a DataFrame, which crashed pandas
    (nlargest/concat compare attrs with ==) as soon as the sanitiser found anything."""
    from jse_backtester.data import synthetic_prices
    from jse_backtester.universe import JSE_TOP_TICKERS
    px = synthetic_prices(JSE_TOP_TICKERS[:20], "2016-01-01", "2020-12-31", seed=5) * 150
    px.iloc[600, 3] *= 0.01
    with pytest.warns(UserWarning, match="Data sanitiser"):
        clean = clean_prices(px)
    assert len(glitch_events(clean)) == 2
    res = run_backtest(clean.drop(columns=["STX40.JO"]), BacktestConfig(signal="momentum"))
    assert res.returns.abs().max() < 0.2


# ---------------------------------------------------------------- USD/ZAR (rand per dollar, ~12-20)
def _usdzar(n=1500, seed=0, start=15.0, vol=0.0085):
    """Fat-tailed USD/ZAR level series (rand per dollar), nothing like a ZAc price."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = rng.standard_t(5, n) / np.sqrt(5 / 3) * vol
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx, name="USDZAR=X")


def _frame(s):
    return pd.DataFrame({"USDZAR=X": s})


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_usdzar_triggers_no_unit_flip_false_positives(seed):
    fx = _usdzar(seed=seed)
    out, ev = sanitize_units(_frame(fx))
    pd.testing.assert_frame_equal(out, _frame(fx))
    assert ev.empty


def test_real_world_usdzar_shocks_are_left_alone():
    """Nenegate-style: -9% (rand) day plus a multi-day slide, then a COVID-style grind: persistent
    moves of a size USD/ZAR genuinely makes must never be rescaled, removed or flagged as units."""
    fx = _usdzar()
    fx.iloc[300:] *= 1.09                      # one-day +9% that sticks (Nenegate, Dec 2015)
    fx.iloc[400:] *= 1.05
    fx.iloc[401:] *= 1.04
    fx.iloc[402:] *= 1.03                      # three-day slide
    fx.iloc[800:] *= 1 / 1.06                  # relief rally
    out, ev = sanitize_units(_frame(fx))
    pd.testing.assert_frame_equal(out, _frame(fx))
    assert ev.empty


@pytest.mark.parametrize("factor", [100.0, 0.01])
def test_usdzar_100x_bad_tick_is_still_repaired(factor):
    fx = _usdzar()
    bad = fx.copy()
    bad.iloc[200] *= factor
    out, ev = sanitize_units(_frame(bad))
    np.testing.assert_allclose(out["USDZAR=X"].to_numpy(), fx.to_numpy(), rtol=1e-9)
    assert list(ev["kind"]) == ["unit_flip", "unit_flip"] and set(ev["ticker"]) == {"USDZAR=X"}


def test_usdzar_decimal_slip_is_a_reverting_spike_not_a_unit_flip():
    fx = _usdzar()
    bad = fx.copy()
    bad.iloc[500] *= 10.0                      # 15.x printed as 1xx.x for a day
    out, ev = sanitize_units(_frame(bad))
    assert list(ev["kind"]) == ["spike_removed"]
    assert np.isnan(out["USDZAR=X"].iloc[500])
    mask = np.arange(len(fx)) != 500
    np.testing.assert_allclose(out["USDZAR=X"].to_numpy()[mask], fx.to_numpy()[mask], rtol=1e-12)


def test_fx_uses_tighter_bad_print_thresholds_than_equities():
    fx = _usdzar()
    bad = fx.copy()
    bad.iloc[600] *= 1.18                      # +18% print that reverts: absurd for USD/ZAR
    _, ev_fx = sanitize_units(_frame(bad))
    assert list(ev_fx["kind"]) == ["spike_removed"]
    # the same move on an equity (default 35% threshold) is left untouched: equities can do that
    _, ev_eq = sanitize_units(pd.DataFrame({"AAA.JO": bad.rename("AAA.JO")}))
    assert ev_eq.empty


def test_fx_and_cents_columns_keep_their_own_units():
    stock = _base(seed=11, start=15000.0).rename("SBK.JO")   # ZAc, ~R150
    fx = _usdzar(n=len(stock), seed=12).set_axis(stock.index).rename("USDZAR=X")
    px = pd.DataFrame({"SBK.JO": stock, "USDZAR=X": fx})
    out, ev = sanitize_units(px)
    pd.testing.assert_frame_equal(out, px)     # neither is "corrected" toward the other's unit
    assert ev.empty

    bad = px.copy()
    bad.iloc[100, 0] *= 0.01                   # the stock flips to rands; the currency must not care
    out, ev = sanitize_units(bad)
    np.testing.assert_allclose(out["SBK.JO"].to_numpy(), stock.to_numpy(), rtol=1e-9)
    pd.testing.assert_series_equal(out["USDZAR=X"], fx)
    assert set(ev["ticker"]) == {"SBK.JO"}


@pytest.mark.parametrize("level", [0.055, 1500.0])
def test_wrong_unit_or_inverted_fx_level_is_flagged_not_changed(level):
    """A whole series quoted as ZARUSD (~0.055) or in a 100x unit has no jump for the flip
    detector to find, so a level check flags it (and leaves the data as supplied)."""
    fx = _usdzar() * level / 15.0
    out, ev = sanitize_units(_frame(fx))
    pd.testing.assert_frame_equal(out, _frame(fx))
    assert list(ev["kind"]) == ["fx_level"]
    from jse_backtester.sanitize import summarize_events
    assert "implausible FX levels flagged" in summarize_events(ev)


def test_usdzar_flows_through_clean_prices_silently():
    stock = _base(seed=21, start=15000.0)
    px = pd.DataFrame({"SBK.JO": stock, "USDZAR=X": _usdzar(n=len(stock), seed=22).set_axis(stock.index)})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")         # no "Data sanitiser: ..." warning for clean FX
        clean = clean_prices(px)
    assert glitch_events(clean).empty
    pd.testing.assert_series_equal(clean["USDZAR=X"], px["USDZAR=X"])
