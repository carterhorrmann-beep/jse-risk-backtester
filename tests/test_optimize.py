"""Portfolio optimisation: min-variance, risk parity, the comparison table, trailing-window
backtest weighting (no look-ahead), portfolio CSV parsing, and the CLI / chart plumbing."""
import io

import numpy as np
import pandas as pd
import pytest

from jse_backtester import optimize, plots, risk
from jse_backtester.backtest import BacktestConfig, build_weights, run_backtest
from jse_backtester.cli import main
from jse_backtester.data import split_fx, synthetic_prices
from jse_backtester.optimize import (compare_portfolios, equal_weight, min_variance, optimise_weights,
                                     read_portfolio_csv, risk_parity)
from jse_backtester.universe import DEFAULT_BENCHMARK, JSE_TOP_TICKERS
from jse_backtester.utils import returns_from_prices

SAMPLE = "examples/sample_portfolio.csv"
SHORT = ["--source", "synthetic", "--start", "2018-01-01", "--end", "2021-12-31"]
METHODS = ["equal", "min_var", "risk_parity"]


@pytest.fixture(scope="module")
def market():
    """(prices without benchmark/FX, daily returns, USD/ZAR returns) on 25 simulated JSE names."""
    px = synthetic_prices(JSE_TOP_TICKERS[:25], "2016-01-01", "2022-12-31", seed=7, include_fx=True)
    px, fx = split_fx(px.drop(columns=[DEFAULT_BENCHMARK]))
    fx_ret = returns_from_prices(fx.reindex(px.index).to_frame()).iloc[:, 0]
    return px, returns_from_prices(px).iloc[1:], fx_ret


@pytest.fixture(scope="module")
def cov(market):
    return risk.shrunk_cov(market[1], 0.1)


def _random_cov(n, seed):
    """Heterogeneous vols (5-70%) and factor-driven correlations."""
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 1, (n, 2))
    C = L @ L.T + np.diag(rng.uniform(0.2, 2.0, n))
    d = np.sqrt(np.diag(C))
    S = (C / np.outer(d, d)) * np.outer(*(2 * [rng.uniform(0.05, 0.70, n)]))
    names = [f"S{i}" for i in range(n)]
    return pd.DataFrame(S, index=names, columns=names)


def _var(w, S):
    return float(w.to_numpy() @ S.to_numpy() @ w.to_numpy())


# ---------------------------------------------------------------- weights sum to one, long-only
@pytest.mark.parametrize("method", METHODS)
def test_weights_sum_to_one_and_are_long_only(cov, method):
    w = optimise_weights(cov, method, 0.20)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w >= 0).all()
    assert list(w.index) == list(cov.index)


@pytest.mark.parametrize("n", [3, 8, 20, 40])
def test_weights_sum_to_one_on_random_covariances(n):
    S = _random_cov(n, seed=n)
    for method in METHODS:
        w = optimise_weights(S, method, max(0.20, 1.0 / n))
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert (w >= 0).all()


# ---------------------------------------------------------------- max_weight cap
@pytest.mark.parametrize("cap", [0.06, 0.10, 0.15, 0.20, 0.35])
def test_min_variance_respects_max_weight(cov, cap):
    w = min_variance(cov, cap)
    assert w.max() <= cap + 1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_default_max_weight_is_20_percent(cov):
    assert optimize.DEFAULT_MAX_WEIGHT == 0.20
    assert BacktestConfig().max_weight == 0.20
    pd.testing.assert_series_equal(min_variance(cov), min_variance(cov, 0.20))


def test_cap_binds_and_costs_variance_versus_uncapped(cov):
    free, capped = min_variance(cov, 1.0), min_variance(cov, 0.06)
    assert free.max() > 0.06 + 1e-3                  # the unconstrained optimum concentrates
    assert capped.max() == pytest.approx(0.06, abs=1e-8)  # and the cap is actually what limits it
    assert _var(capped, cov) >= _var(free, cov) - 1e-12


def test_infeasible_cap_raises_a_readable_error():
    S = _random_cov(4, seed=1)
    with pytest.raises(ValueError, match="at least 5 holdings"):
        min_variance(S, 0.20)
    with pytest.raises(ValueError, match="max_weight"):
        min_variance(S, 0.0)


def test_cap_of_exactly_one_over_n_gives_equal_weights():
    S = _random_cov(5, seed=2)
    pd.testing.assert_series_equal(min_variance(S, 0.20), equal_weight(S))


# ---------------------------------------------------------------- min-variance is actually minimal
def test_min_var_has_lower_in_sample_variance_than_equal_weight(market, cov):
    rets = market[1]
    ew, mv = equal_weight(cov), min_variance(cov, 0.20)
    assert _var(mv, cov) < _var(ew, cov)                          # the shrunk-covariance objective
    assert (rets @ mv).var() < (rets @ ew).var()                   # and on the raw daily returns
    assert _var(mv, cov) <= _var(risk_parity(cov), cov) + 1e-12    # nothing feasible beats it


def test_min_var_beats_random_feasible_portfolios(cov):
    rng = np.random.default_rng(0)
    cap, mv = 0.20, min_variance(cov, 0.20)
    n, best = len(cov), np.inf
    for _ in range(500):
        x = rng.dirichlet(np.ones(n))
        for _ in range(60):                               # push mass off names above the cap
            x = np.minimum(x, cap)
            room = cap - x
            x = x + (1 - x.sum()) * room / room.sum()
        if abs(x.sum() - 1) < 1e-9 and x.max() <= cap + 1e-9:
            best = min(best, x @ cov.to_numpy() @ x)
    assert _var(mv, cov) <= best + 1e-12


def test_min_variance_closed_form_when_assets_are_uncorrelated():
    S = pd.DataFrame(np.diag([0.04, 0.09, 0.16, 0.25]), index=list("ABCD"), columns=list("ABCD"))
    w = min_variance(S, 1.0)
    inv = 1.0 / np.diag(S)                      # with zero correlations the optimum is w_i ~ 1 / variance_i
    np.testing.assert_allclose(w.to_numpy(), inv / inv.sum(), atol=1e-7)


# ---------------------------------------------------------------- risk parity equalises risk shares
def test_risk_parity_equalises_euler_risk_shares(cov):
    w = risk_parity(cov)
    rc = risk.risk_contributions(w, cov)
    assert rc["pct"].to_numpy() == pytest.approx(np.full(len(cov), 1.0 / len(cov)), abs=1e-6)
    assert (w > 0).all()
    # it is the Euler decomposition of the spec: RC_i = w_i (S w)_i / sqrt(w' S w), in volatility units
    vol = np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert rc["contribution"].to_numpy() == pytest.approx(np.full(len(cov), vol / len(cov)), abs=1e-6)


@pytest.mark.parametrize("n", [3, 5, 12, 30, 40])
def test_risk_parity_equalises_risk_on_random_covariances(n):
    S = _random_cov(n, seed=100 + n)
    w = risk_parity(S)
    rc = risk.risk_contributions(w, S)["pct"]
    assert (rc - 1.0 / n).abs().max() < 1e-6
    assert (w > 0).all() and w.sum() == pytest.approx(1.0, abs=1e-12)


def test_risk_parity_closed_form_when_uncorrelated_is_inverse_vol():
    S = pd.DataFrame(np.diag([0.04, 0.16, 0.09]), index=list("ABC"), columns=list("ABC"))
    w = risk_parity(S)
    inv = 1.0 / np.sqrt(np.diag(S))
    np.testing.assert_allclose(w.to_numpy(), inv / inv.sum(), atol=1e-7)


def test_risk_parity_ignores_the_scale_of_the_covariance(cov):
    np.testing.assert_allclose(risk_parity(cov * 7.0).to_numpy(), risk_parity(cov).to_numpy(), atol=1e-7)


def test_unknown_method_is_rejected(cov):
    with pytest.raises(ValueError, match="unknown method"):
        optimise_weights(cov, "max_sharpe")


# ---------------------------------------------------------------- consistency with risk.py
def test_compare_portfolios_uses_the_shared_shrunk_covariance_and_risk_math(market):
    px, rets, fx_ret = market
    tickers = list(px.columns[:11])
    res = compare_portfolios(rets, tickers, max_weight=0.20, fx_returns=fx_ret)
    common = rets[tickers].dropna()
    pd.testing.assert_frame_equal(res.cov, risk.shrunk_cov(common, 0.1))
    assert list(res.weights.columns) == METHODS and list(res.risk_shares.columns) == METHODS
    assert res.weights.sum().to_numpy() == pytest.approx(np.ones(3))
    assert res.risk_shares.sum().to_numpy() == pytest.approx(np.ones(3))
    assert res.risk_parity_gap < 1e-6
    # the same portfolio through the existing risk report gives identical numbers
    rep = risk.portfolio_risk_report(res.weights["min_var"], common, fx_returns=fx_ret)
    assert res.metrics.loc["Diversification ratio", "min_var"] == pytest.approx(rep.diversification_ratio)
    pd.testing.assert_series_equal(res.risk_shares["min_var"], rep.contributions["pct"].reindex(tickers), check_names=False)
    assert res.metrics.loc["Look-through USD/ZAR Beta", "min_var"] == pytest.approx(
        rep.currency.summary["Look-through USD/ZAR Beta (weighted)"])
    assert res.metrics.loc["Ann. volatility", "min_var"] < res.metrics.loc["Ann. volatility", "equal"]


def test_compare_portfolios_with_current_holdings_adds_a_column(market):
    px, rets, _ = market
    cur = pd.Series([3.0, 2.0, 1.0, 1.0, 1.0], index=list(px.columns[:5]))
    res = compare_portfolios(rets, current=cur / cur.sum(), max_weight=0.20)
    assert list(res.weights.columns) == ["current", *METHODS]
    assert res.weights["current"].sum() == pytest.approx(1.0)
    assert res.metrics.loc["Look-through USD/ZAR Beta"].isna().all()   # no FX supplied


def test_compare_portfolios_input_errors(market):
    px, rets, _ = market
    with pytest.raises(ValueError, match="at least two"):
        compare_portfolios(rets, [px.columns[0]])
    with pytest.raises(ValueError, match="holdings to reach 100%"):
        compare_portfolios(rets, list(px.columns[:4]), max_weight=0.20)
    with pytest.raises(ValueError, match="common daily observations"):
        compare_portfolios(rets.iloc[:30], list(px.columns[:8]))


# ---------------------------------------------------------------- trailing window: no look-ahead
def _toy_returns(n_days=700, n=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    scale = rng.uniform(0.008, 0.03, n)
    f = rng.normal(0, 0.008, (n_days, 1))
    return pd.DataFrame(f + rng.normal(0, 1, (n_days, n)) * scale, index=idx, columns=[f"S{i}" for i in range(n)])


def test_trailing_returns_is_exactly_the_last_252_days_up_to_the_date():
    rets = _toy_returns()
    d = rets.index[500]
    win = risk.trailing_returns(rets, d)
    assert len(win) == 252 and win.index[-1] == d
    assert win.index[0] == rets.index[500 - 251]
    assert risk.trailing_returns(rets, rets.index[40]).empty        # too little history -> nothing usable


def test_trailing_returns_keeps_only_names_with_a_complete_window():
    rets = _toy_returns()
    rets.loc[rets.index[:300], "S3"] = np.nan                       # S3 lists part-way through
    assert "S3" not in risk.trailing_returns(rets, rets.index[400]).columns
    assert "S3" in risk.trailing_returns(rets, rets.index[600]).columns


def test_backtest_weights_use_only_the_trailing_252_days_before_the_rebalance():
    rets = _toy_returns()
    d = rets.index[520]
    signal = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)   # hold everything: isolates weighting
    vol = rets.rolling(60).std()
    for weighting in ("min_var", "risk_parity"):
        cfg = BacktestConfig(weighting=weighting, top_frac=1.0, max_weight=0.25)
        base = build_weights(signal, pd.DatetimeIndex([d]), vol, cfg, rets).loc[d]
        changed = rets.copy()
        changed.iloc[521:] = np.random.default_rng(1).normal(0, 0.5, changed.iloc[521:].shape)   # the future
        changed.iloc[:520 - 251] = np.random.default_rng(2).normal(0, 0.5, changed.iloc[:520 - 251].shape)  # too old
        again = build_weights(signal, pd.DatetimeIndex([d]), vol, cfg, changed).loc[d]
        np.testing.assert_allclose(again.to_numpy(), base.to_numpy(), atol=1e-12)
        # ...but the window itself does matter (guards against the test passing trivially)
        inside = rets.copy()
        inside.iloc[400:500] = np.random.default_rng(3).normal(0, 0.5, inside.iloc[400:500].shape)
        moved = build_weights(signal, pd.DatetimeIndex([d]), vol, cfg, inside).loc[d]
        assert np.abs(moved.to_numpy() - base.to_numpy()).max() > 1e-3


def test_truncating_the_future_does_not_change_earlier_backtest_weights(market):
    px = market[0]
    cut = px.index[-300]
    for weighting in ("min_var", "risk_parity"):
        cfg = BacktestConfig(weighting=weighting)
        full, trunc = run_backtest(px, cfg).weights, run_backtest(px.loc[:cut], cfg).weights
        safe = trunc.index[trunc.index < cut - pd.Timedelta(days=45)]
        assert len(safe) > 20
        np.testing.assert_allclose(trunc.loc[safe].to_numpy(), full.loc[safe].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------- backtest --weighting
@pytest.mark.parametrize("weighting", ["equal", "inv_vol", "min_var", "risk_parity"])
def test_every_weighting_produces_valid_long_only_books(market, weighting):
    res = run_backtest(market[0], BacktestConfig(weighting=weighting, max_weight=0.20))
    assert res.weights.sum(axis=1).round(8).eq(1.0).all()
    assert (res.weights >= -1e-12).all().all()
    if weighting == "min_var":
        assert res.weights.max().max() <= 0.20 + 1e-9
    if weighting == "risk_parity":
        assert (res.weights.where(res.weights > 0).min().min()) > 0.0


def test_optimised_weighting_differs_from_equal_and_costs_are_still_charged(market):
    eq, mv = (run_backtest(market[0], BacktestConfig(weighting=w)) for w in ("equal", "min_var"))
    assert not np.allclose(eq.weights.to_numpy(), mv.weights.to_numpy())
    assert (mv.returns <= mv.gross_returns + 1e-12).all() and mv.annual_turnover > 0


def test_min_var_cap_relaxes_to_equal_weight_when_too_few_names_are_held(market):
    res = run_backtest(market[0], BacktestConfig(weighting="min_var", top_frac=0.12, max_weight=0.20))  # 3 names
    assert res.weights.sum(axis=1).round(8).eq(1.0).all()
    assert res.weights.max().max() <= 1 / 3 + 1e-9


def test_long_short_legs_are_optimised_separately(market):
    res = run_backtest(market[0], BacktestConfig(weighting="risk_parity", long_only=False))
    assert res.weights.sum(axis=1).abs().max() < 1e-9
    assert res.weights.abs().sum(axis=1).round(6).eq(2.0).all()


def test_unknown_weighting_and_missing_returns_are_rejected(market):
    with pytest.raises(ValueError, match="unknown weighting"):
        run_backtest(market[0], BacktestConfig(weighting="max_sharpe"))
    sig = pd.DataFrame(1.0, index=market[0].index, columns=market[0].columns)
    with pytest.raises(ValueError, match="needs the daily returns"):
        build_weights(sig, sig.index[300:301], sig, BacktestConfig(weighting="min_var"))


# ---------------------------------------------------------------- portfolio CSV
def test_read_portfolio_csv_parses_normalises_and_aggregates():
    csv = io.StringIO("Ticker,Weight\nnpn,30\nSBK.JO,20\nAGL,20\nnpn,10\nUSDZAR=X,0\n")
    w = read_portfolio_csv(csv)
    assert list(w.index) == ["NPN.JO", "SBK.JO", "AGL.JO", "USDZAR=X"]
    assert w.sum() == pytest.approx(1.0)
    assert w["NPN.JO"] == pytest.approx(40 / 80)


def test_read_portfolio_csv_reads_the_same_file_twice_and_the_repo_sample():
    buf = io.BytesIO(b"ticker,weight\nNPN.JO,1\nSBK.JO,1\n")
    pd.testing.assert_series_equal(read_portfolio_csv(buf), read_portfolio_csv(buf))   # rewinds like a re-run upload
    sample = read_portfolio_csv(SAMPLE)
    assert len(sample) == 11 and sample.sum() == pytest.approx(1.0)


def test_read_portfolio_csv_falls_back_to_the_first_two_columns():
    w = read_portfolio_csv(io.StringIO("symbol,alloc\nNPN.JO,2\nSBK.JO,2\n"))
    assert w.to_dict() == {"NPN.JO": 0.5, "SBK.JO": 0.5}


@pytest.mark.parametrize("text, match", [
    ("ticker\nNPN.JO\n", "two columns"),
    ("ticker,weight\nNPN.JO,abc\n", "must be a number"),
    ("ticker,weight\nNPN.JO,-1\nSBK.JO,2\n", "non-negative"),
    ("ticker,weight\nNPN.JO,0\nSBK.JO,0\n", "sum to zero"),
    ("", "could not read"),
])
def test_read_portfolio_csv_rejects_bad_files(text, match):
    with pytest.raises(ValueError, match=match):
        read_portfolio_csv(io.StringIO(text))


# ---------------------------------------------------------------- CLI + charts
def test_optimize_command_writes_report_tables_csvs_and_chart(tmp_path, capsys):
    main(["optimize", *SHORT, "--weights", SAMPLE, "--out", str(tmp_path), "--save-charts", str(tmp_path / "docs")])
    report = (tmp_path / "report.md").read_text()
    for needle in ("Equal Weight", "Min Variance", "Risk Parity", "Current", "risk share", "weight",
                   "Ann. volatility", "Sharpe", "Max drawdown", "Diversification ratio",
                   "Look-through USD/ZAR Beta", "![optimisation](optimize.png)", "max weight 20%"):
        assert needle in report, needle
    assert "Total" in report and "100.0%" in report
    assert "Risk Parity: largest gap between a risk share and 1/N = 0.0000" in report
    assert "Capital weight vs Euler risk share" in capsys.readouterr().out          # printed too
    assert (tmp_path / "optimize.png").stat().st_size > 10_000
    assert (tmp_path / "docs" / "optimize.png").exists()
    w = pd.read_csv(tmp_path / "weights.csv", index_col="ticker")
    assert w[["Equal Weight", "Min Variance", "Risk Parity"]].sum().to_numpy() == pytest.approx(np.ones(3))
    assert w["Min Variance"].max() <= 0.20 + 1e-9
    shares = pd.read_csv(tmp_path / "risk_shares.csv", index_col="ticker")
    assert shares["Risk Parity"].to_numpy() == pytest.approx(np.full(len(shares), 1 / len(shares)), abs=1e-6)
    assert {"Ann. volatility", "Look-through USD/ZAR Beta"} <= set(pd.read_csv(tmp_path / "metrics.csv", index_col=0).index)


def test_optimize_command_accepts_tickers_and_max_weight(tmp_path):
    main(["optimize", *SHORT, "--tickers", "npn", "SBK.JO", "AGL", "GFI.JO", "SHP", "MTN", "--max-weight", "0.25",
          "--out", str(tmp_path)])
    w = pd.read_csv(tmp_path / "weights.csv", index_col="ticker")
    assert list(w.columns) == ["Equal Weight", "Min Variance", "Risk Parity"]          # no 'Current' without a CSV
    assert list(w.index) == ["NPN.JO", "SBK.JO", "AGL.JO", "GFI.JO", "SHP.JO", "MTN.JO"]
    assert w["Min Variance"].max() <= 0.25 + 1e-9
    assert "max weight 25%" in (tmp_path / "report.md").read_text()


def test_optimize_command_fails_cleanly_on_bad_input(tmp_path):
    with pytest.raises(SystemExit, match="cap needs at least"):
        main(["optimize", *SHORT, "--weights", SAMPLE, "--max-weight", "0.05", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="not found"):
        main(["optimize", *SHORT, "--weights", str(tmp_path / "missing.csv"), "--out", str(tmp_path)])


@pytest.mark.parametrize("weighting", ["min_var", "risk_parity"])
def test_backtest_command_accepts_the_new_weightings(tmp_path, capsys, weighting):
    main(["backtest", *SHORT, "--weighting", weighting, "--max-weight", "0.15", "--out", str(tmp_path)])
    header = capsys.readouterr().out
    assert weighting in header and "trailing 252d shrunk cov" in header
    assert (tmp_path / "report.md").exists()
    w = pd.read_csv(tmp_path / "weights.csv", index_col=0)
    assert w.sum(axis=1).round(8).eq(1.0).all()
    if weighting == "min_var":
        assert "cap 15%" in header and w.max().max() <= 0.15 + 1e-9


def test_charts_command_writes_docs_optimize_png(tmp_path):
    main(["charts", *SHORT, "--out", str(tmp_path / "reports"), "--dir", str(tmp_path / "docs"), "--weights", SAMPLE])
    assert (tmp_path / "docs" / "optimize.png").stat().st_size > 10_000


def test_plot_optimisation_draws_one_grouped_panel_per_method(market):
    px, rets, fx_ret = market
    res = compare_portfolios(rets, list(px.columns[:11]), fx_returns=fx_ret)
    fig = plots.plot_optimisation(res.weights, res.risk_shares, res.metrics)
    assert len(fig.axes) == 3
    assert all(len(ax.patches) == 2 * len(res.weights) for ax in fig.axes)       # weight + risk bar per stock
    assert [ax.get_title(loc="left").split("|")[0].strip() for ax in fig.axes] == \
        ["Equal Weight", "Min Variance", "Risk Parity"]
    with pytest.raises(ValueError, match="none of the columns"):
        plots.plot_optimisation(res.weights.iloc[:, :0], res.risk_shares.iloc[:, :0])
