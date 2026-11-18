"""USD/ZAR beta, Rand-hedge classification, exposure decomposition and the reports built on them."""
import numpy as np
import pandas as pd
import pytest

from jse_backtester import risk
from jse_backtester.backtest import (BacktestConfig, daily_weights, rand_hedge_exposure, run_backtest,
                                     simulate)
from jse_backtester.cli import main
from jse_backtester.costgrid import cost_grid
from jse_backtester.data import load_prices, split_fx, synthetic_prices
from jse_backtester.universe import DEFAULT_BENCHMARK, JSE_TOP_TICKERS, USDZAR_TICKER
from jse_backtester.utils import newey_west_ols, rebalance_dates, returns_from_prices

TRUE_BETAS = {"HEDGE_HI": 0.8, "HEDGE_LO": 0.30, "FLAT": 0.0, "DOM_LO": -0.30, "DOM_HI": -0.8}


def _fx_returns(n=2500, seed=0):
    """Fat-tailed daily USD/ZAR % changes (~0.85% a day)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(5, n) / np.sqrt(5 / 3) * 0.0085
    return pd.Series(r, index=pd.bdate_range("2016-01-01", periods=n), name="USDZAR")


def _stocks(fx, betas=TRUE_BETAS, noise=0.015, seed=1):
    """ZAR stock returns = beta x USD/ZAR change + idiosyncratic noise (no market factor)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({t: b * fx + rng.normal(0, noise, len(fx)) for t, b in betas.items()})


# ---------------------------------------------------------------- beta maths
def test_usdzar_beta_recovers_known_betas_and_buckets():
    fx = _fx_returns()
    table = risk.usdzar_betas(_stocks(fx), fx)
    for t, b in TRUE_BETAS.items():
        assert table.loc[t, "beta"] == pytest.approx(b, abs=0.06)
    assert table.loc["HEDGE_HI", "bucket"] == risk.BUCKET_HEDGE
    assert table.loc["HEDGE_LO", "bucket"] == risk.BUCKET_HEDGE      # 0.30 > 0.20
    assert table.loc["FLAT", "bucket"] == risk.BUCKET_NEUTRAL
    assert table.loc["DOM_LO", "bucket"] == risk.BUCKET_DOMESTIC     # -0.30 < -0.20
    assert table.loc["DOM_HI", "bucket"] == risk.BUCKET_DOMESTIC
    assert table.loc["HEDGE_HI", "t_stat"] > 10 and table.loc["DOM_HI", "t_stat"] < -10
    assert abs(table.loc["FLAT", "t_stat"]) < 3                      # nothing there to find
    assert list(table.index) == list(table["beta"].sort_values(ascending=False).index)
    assert (table["n_obs"] == len(fx)).all()


@pytest.mark.parametrize("beta, bucket", [
    (0.2001, risk.BUCKET_HEDGE), (0.20, risk.BUCKET_NEUTRAL), (0.0, risk.BUCKET_NEUTRAL),
    (-0.20, risk.BUCKET_NEUTRAL), (-0.2001, risk.BUCKET_DOMESTIC), (float("nan"), risk.BUCKET_UNKNOWN),
])
def test_classification_thresholds_are_exactly_plus_minus_020(beta, bucket):
    assert risk.classify_usdzar_beta(beta) == bucket


def test_sign_convention_usdzar_up_means_rand_weaker():
    """A stock that rises when USDZAR rises is a Rand Hedge; the same stock vs the inverted
    quote (ZARUSD) flips sign, which is why the sanitiser flags an inverted series."""
    fx = _fx_returns()
    stock = pd.DataFrame({"X": 0.7 * fx})
    assert risk.usdzar_betas(stock, fx).loc["X", "beta"] == pytest.approx(0.7, abs=1e-9)
    inverse = 1 / (1 + fx) - 1  # ZAR per... dollar per rand
    assert risk.usdzar_betas(stock, inverse).loc["X", "beta"] < 0


def test_newey_west_tstat_matches_independent_hac_formula():
    """The t-stat must equal a hand-rolled Bartlett-kernel HAC calculation for the slope. With
    a persistent regressor AND persistent errors (so x*e is autocorrelated) HAC must also come
    out below the naive OLS t-stat. (With a serially uncorrelated regressor, as real daily FX
    changes nearly are, HAC ~ naive: the correction only bites when x*e is autocorrelated.)"""
    rng = np.random.default_rng(5)
    n, lags = 3000, 5
    x = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 0.0085)
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.012)
    fx = pd.Series(x, index=pd.bdate_range("2016-01-01", periods=n))
    stock = pd.DataFrame({"S": 0.5 * fx + e})
    row = risk.usdzar_betas(stock, fx, lags=lags).loc["S"]

    y = stock["S"].to_numpy()
    xt = x - x.mean()
    slope = (xt @ y) / (xt @ xt)
    resid = y - y.mean() - slope * xt
    u = xt * resid
    S = u @ u + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) for k in range(1, lags + 1))
    manual_t = slope / (np.sqrt(S) / (xt @ xt))
    assert row["beta"] == pytest.approx(slope)
    assert row["t_stat"] == pytest.approx(manual_t, rel=1e-9)

    naive_t = slope / np.sqrt(resid.var(ddof=2) / (xt @ xt))
    assert abs(row["t_stat"]) < 0.9 * abs(naive_t)


def test_newey_west_helper_reduces_to_white_with_zero_lags():
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = 1 + 2 * X[:, 1] + rng.normal(size=500)
    beta, se, resid = newey_west_ols(y, X, lags=0)
    bread = np.linalg.inv(X.T @ X)
    white = np.sqrt(np.diag(bread @ (X * resid[:, None]).T @ (X * resid[:, None]) @ bread))
    np.testing.assert_allclose(se, white)


def test_too_little_history_is_unclassified_not_guessed():
    fx = _fx_returns(400)
    rets = _stocks(fx)
    rets.iloc[:300, 0] = np.nan  # HEDGE_HI has 100 observations only
    table = risk.usdzar_betas(rets, fx, min_obs=250)
    assert np.isnan(table.loc["HEDGE_HI", "beta"]) and table.loc["HEDGE_HI", "n_obs"] == 100
    assert table.loc["HEDGE_HI", "bucket"] == risk.BUCKET_UNKNOWN
    assert table.loc["DOM_HI", "bucket"] == risk.BUCKET_DOMESTIC


def test_the_fx_column_is_never_treated_as_a_stock():
    fx = _fx_returns()
    rets = _stocks(fx).assign(**{USDZAR_TICKER: fx})
    assert USDZAR_TICKER not in risk.usdzar_betas(rets, fx).index


def test_rolling_correlation_window_and_value():
    fx = _fx_returns(400)
    roll = risk.rolling_correlation(2.0 * fx, fx, window=126)
    assert len(roll) == 400 - 126 + 1
    np.testing.assert_allclose(roll.to_numpy(), 1.0)
    assert (risk.rolling_correlation(-fx, fx, 126) < -0.999).all()


# ---------------------------------------------------------------- portfolio exposure (risk)
def test_portfolio_realised_beta_equals_weighted_holding_betas():
    """OLS is linear: on a common sample the portfolio's USD/ZAR beta is exactly sum(w * beta)."""
    fx = _fx_returns()
    rets = _stocks(fx)
    w = pd.Series({"HEDGE_HI": 0.4, "HEDGE_LO": 0.1, "FLAT": 0.1, "DOM_LO": 0.1, "DOM_HI": 0.3})
    rep = risk.portfolio_risk_report(w, rets, fx_returns=fx)
    ce = rep.currency
    expected = sum(w[t] * ce.betas.loc[t, "beta"] for t in w.index)
    assert ce.summary["USD/ZAR Beta (realised)"] == pytest.approx(expected, abs=1e-10)
    assert ce.summary["Look-through USD/ZAR Beta (weighted)"] == pytest.approx(expected, abs=1e-10)
    # weights split by bucket
    assert ce.bucket_weights[risk.BUCKET_HEDGE] == pytest.approx(0.5)
    assert ce.bucket_weights[risk.BUCKET_NEUTRAL] == pytest.approx(0.1)
    assert ce.bucket_weights[risk.BUCKET_DOMESTIC] == pytest.approx(0.4)
    assert ce.bucket_weights.sum() == pytest.approx(1.0)
    assert ce.summary["Beta t-stat (Newey-West)"] != 0 and 0 <= ce.summary["R^2 vs USD/ZAR"] <= 1


def test_look_through_needs_both_weights_and_betas():
    fx = _fx_returns()
    with pytest.raises(ValueError, match="together"):
        risk.currency_exposure(fx, fx, weights=pd.Series({"A": 1.0}))
    with pytest.raises(ValueError, match="together"):
        risk.currency_exposure(fx, fx, betas=risk.usdzar_betas(_stocks(fx), fx))


def test_risk_report_has_no_currency_block_without_fx():
    fx = _fx_returns()
    rep = risk.portfolio_risk_report(pd.Series({"HEDGE_HI": 1.0, "DOM_HI": 1.0}), _stocks(fx))
    assert rep.currency is None


def test_rolling_correlation_in_report_uses_126_day_window():
    fx = _fx_returns()
    rep = risk.portfolio_risk_report(pd.Series({"HEDGE_HI": 1.0, "HEDGE_LO": 1.0}), _stocks(fx), fx_returns=fx)
    assert rep.currency.window == 126
    assert len(rep.currency.rolling_corr) == len(fx) - 126 + 1
    s = rep.currency.summary
    assert s["Rolling 126d corr (min)"] <= s["Rolling 126d corr (average)"] <= s["Rolling 126d corr (max)"]
    assert s["Rolling 126d corr (latest)"] == pytest.approx(rep.currency.rolling_corr.iloc[-1])


# ---------------------------------------------------------------- synthetic data
@pytest.fixture(scope="module")
def synth():
    px = load_prices(source="synthetic", start="2015-01-01", end="2022-12-31")
    stocks, fx = split_fx(px.drop(columns=[DEFAULT_BENCHMARK]))
    return px, stocks, fx


def test_synthetic_source_includes_a_realistic_usdzar_series(synth):
    px, stocks, fx = synth
    assert USDZAR_TICKER in px.columns and USDZAR_TICKER not in stocks.columns
    assert 12 < fx.median() < 20 and 8 < fx.min() and fx.max() < 24     # rand per dollar, not cents
    r = fx.pct_change().dropna()
    assert 0.08 < r.std() * np.sqrt(252) < 0.22                          # ~13% annualised, not stock-like
    assert r.kurt() > 2                                                   # fat tails
    assert r.abs().max() < 0.12                                           # no 100x style jumps


def test_synthetic_rand_hedges_and_sa_inc_are_found_by_the_classifier(synth):
    _, stocks, fx = synth
    b = risk.usdzar_betas(returns_from_prices(stocks), returns_from_prices(fx.to_frame()).iloc[:, 0])
    for miner in ("AGL.JO", "GFI.JO", "S32.JO", "BHG.JO"):
        assert b.loc[miner, "bucket"] == risk.BUCKET_HEDGE, miner
    for domestic in ("SBK.JO", "ABG.JO", "NED.JO", "SHP.JO", "WHL.JO"):
        assert b.loc[domestic, "bucket"] == risk.BUCKET_DOMESTIC, domestic
    assert set(b["bucket"]) >= {risk.BUCKET_HEDGE, risk.BUCKET_DOMESTIC}
    assert b.loc[["NPN.JO", "PRX.JO", "BTI.JO"], "beta"].min() > 0.2


def test_synthetic_fx_is_opt_in_and_leaves_the_main_random_stream_untouched():
    tick = JSE_TOP_TICKERS[:12]
    off = synthetic_prices(tick, "2018-01-01", "2020-12-31", seed=3)
    on = synthetic_prices(tick, "2018-01-01", "2020-12-31", seed=3, include_fx=True)
    assert USDZAR_TICKER not in off.columns and USDZAR_TICKER in on.columns
    fx_ret = on[USDZAR_TICKER].pct_change()
    for t in tick:
        # enabling FX adds exactly load x fx_return to each day's stock return, nothing else
        diff = on[t].pct_change() - off[t].pct_change()
        load = (diff / fx_ret).dropna()
        assert load.std() < 1e-9, t
    again = synthetic_prices(tick, "2018-01-01", "2020-12-31", seed=3, include_fx=True)
    pd.testing.assert_frame_equal(on, again)  # deterministic


# ---------------------------------------------------------------- backtest look-through
def _book(fx_beta_by_stock, seed=3, n=1500):
    fx = _fx_returns(n, seed=seed)
    rets = _stocks(fx, fx_beta_by_stock)
    rets.index = fx.index
    return rets, fx


def _targets(rets, held):
    dates = rebalance_dates(rets.index, "M")[2:]
    t = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    t.loc[:, held] = 1.0 / len(held)
    return t


def test_daily_weights_reproduce_the_gross_return_stream(synth):
    _, stocks, _ = synth
    res = run_backtest(stocks, BacktestConfig(signal="momentum"))
    rets = returns_from_prices(stocks)
    W = daily_weights(res, rets)
    rebuilt = (W * rets.reindex(res.returns.index).fillna(0.0)).sum(axis=1)
    np.testing.assert_allclose(rebuilt.to_numpy(), res.gross_returns.to_numpy(), atol=1e-12)
    live = W.abs().sum(axis=1) > 0
    np.testing.assert_allclose(W.loc[live].sum(axis=1).to_numpy(), 1.0, atol=1e-9)  # fully invested long-only


def test_rand_hedge_book_reports_hedge_weight_and_positive_beta():
    betas = {f"H{i}": 0.8 for i in range(6)} | {f"D{i}": -0.8 for i in range(6)}
    rets, fx = _book(betas)
    cfg = BacktestConfig(stt=0.0, brokerage=0.0)
    hedge_book = simulate(rets, _targets(rets, [f"H{i}" for i in range(6)]), cfg)
    dom_book = simulate(rets, _targets(rets, [f"D{i}" for i in range(6)]), cfg)

    stock_betas = risk.usdzar_betas(rets, fx)
    h = rand_hedge_exposure(hedge_book, rets, fx, stock_betas)
    d = rand_hedge_exposure(dom_book, rets, fx, stock_betas)

    assert h.bucket_weights[risk.BUCKET_HEDGE] == pytest.approx(1.0)
    assert h.bucket_weights.get(risk.BUCKET_DOMESTIC, 0.0) == pytest.approx(0.0)
    assert h.summary["USD/ZAR Beta (realised)"] == pytest.approx(0.8, abs=0.08)
    assert h.summary["Look-through USD/ZAR Beta (weighted)"] == pytest.approx(0.8, abs=0.08)
    assert h.summary["Rolling 126d corr (latest)"] > 0.2

    assert d.bucket_weights[risk.BUCKET_DOMESTIC] == pytest.approx(1.0)
    assert d.summary["USD/ZAR Beta (realised)"] == pytest.approx(-0.8, abs=0.08)
    assert d.summary["Rolling 126d corr (latest)"] < -0.2


def test_balanced_book_is_currency_neutral_against_the_universe_tilt():
    betas = {f"H{i}": 0.8 for i in range(6)} | {f"D{i}": -0.8 for i in range(6)}
    rets, fx = _book(betas)
    book = simulate(rets, _targets(rets, list(rets.columns)), BacktestConfig(stt=0.0, brokerage=0.0))
    ex = rand_hedge_exposure(book, rets, fx)
    uni = risk.currency_exposure(rets.mean(axis=1), fx, risk.equal_weight_path(rets), risk.usdzar_betas(rets, fx))
    assert abs(ex.summary["Look-through USD/ZAR Beta (weighted)"]) < 0.05
    assert ex.bucket_weights[risk.BUCKET_HEDGE] == pytest.approx(0.5, abs=0.05)   # drift moves it a little
    gap = (ex.summary["Look-through USD/ZAR Beta (weighted)"]
           - uni.summary["Look-through USD/ZAR Beta (weighted)"])
    assert abs(gap) < 0.05


def test_usdzar_column_is_ignored_by_the_backtester_and_cost_grid(synth):
    px, stocks, _ = synth
    with_fx = px.drop(columns=[DEFAULT_BENCHMARK])
    assert USDZAR_TICKER in with_fx.columns
    a = run_backtest(with_fx, BacktestConfig(signal="low_vol"))
    b = run_backtest(stocks, BacktestConfig(signal="low_vol"))
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert USDZAR_TICKER not in a.weights.columns
    g1 = cost_grid(with_fx, factors=("momentum",), rebalances=("M",), costs_bps=(25,))
    g2 = cost_grid(stocks, factors=("momentum",), rebalances=("M",), costs_bps=(25,))
    pd.testing.assert_frame_equal(g1, g2)


# ---------------------------------------------------------------- CLI reports
SHORT = ["--source", "synthetic", "--start", "2018-01-01", "--end", "2021-12-31"]
SAMPLE = "examples/sample_portfolio.csv"


def test_backtest_report_has_currency_section_chart_and_csv(tmp_path, capsys):
    main(["backtest", *SHORT, "--out", str(tmp_path), "--save-charts", str(tmp_path / "docs")])
    report = (tmp_path / "report.md").read_text()
    assert "## USD/ZAR exposure & rand-hedge breakdown" in report
    for needle in ("USD/ZAR Beta (realised)", "Look-through USD/ZAR Beta", "Rolling 126d corr (latest)",
                   "Rand Hedge", "SA Inc.", "tilt vs universe", "Equal-weight universe"):
        assert needle in report, needle
    assert "![rand hedge](rand_hedge.png)" in report
    assert (tmp_path / "rand_hedge.png").stat().st_size > 10_000
    assert (tmp_path / "docs" / "rand_hedge.png").exists()
    betas = pd.read_csv(tmp_path / "usdzar_betas.csv", index_col=0)
    assert USDZAR_TICKER not in betas.index and {"beta", "t_stat", "bucket"} <= set(betas.columns)
    assert "USD/ZAR exposure & rand-hedge breakdown" in capsys.readouterr().out  # also printed to the console


def test_risk_report_has_currency_section(tmp_path):
    main(["risk", *SHORT, "--weights", SAMPLE, "--out", str(tmp_path)])
    report = (tmp_path / "report.md").read_text()
    assert "## USD/ZAR exposure & rand-hedge breakdown" in report
    assert "Stock USD/ZAR betas (holdings, common sample)" in report
    assert "NPN.JO" in report.split("Stock USD/ZAR betas")[1]
    assert (tmp_path / "risk_rand_hedge.png").exists()


def test_charts_command_writes_docs_rand_hedge_png(tmp_path):
    main(["charts", *SHORT, "--out", str(tmp_path / "reports"), "--dir", str(tmp_path / "docs"),
          "--weights", SAMPLE])
    assert (tmp_path / "docs" / "rand_hedge.png").exists()


def test_report_skips_currency_cleanly_when_prices_have_no_usdzar(tmp_path, capsys):
    px = synthetic_prices(JSE_TOP_TICKERS[:20], "2018-01-01", "2021-12-31", seed=9)  # include_fx=False
    csv = tmp_path / "prices.csv"
    px.to_csv(csv)
    args = ["--source", "csv", "--csv", str(csv), "--out", str(tmp_path / "out")]
    main(["backtest", *args])
    main(["risk", *args, "--weights", SAMPLE])
    out = capsys.readouterr().out
    assert f"{USDZAR_TICKER} not in data" in out
    assert "rand-hedge breakdown" not in (tmp_path / "out" / "report.md").read_text()
    assert not (tmp_path / "out" / "rand_hedge.png").exists()
