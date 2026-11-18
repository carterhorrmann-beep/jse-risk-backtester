"""Interactive dashboard:  streamlit run app/streamlit_app.py"""
import pandas as pd
import streamlit as st

from jse_backtester import plots
from jse_backtester.backtest import BacktestConfig, build_factor_returns, rand_hedge_exposure, run_backtest
from jse_backtester.costgrid import DEFAULT_COSTS_BPS, cost_grid, grid_pivot
from jse_backtester.data import glitch_events, load_prices, split_fx
from jse_backtester.factors import SIGNALS, factor_regression, ic_summary, information_coefficient
from jse_backtester.risk import (BUCKET_DOMESTIC, BUCKET_HEDGE, DEFAULT_RF, USDZAR_HEDGE_THRESHOLD,
                                 currency_exposure, equal_weight_path, portfolio_risk_report,
                                 risk_summary, scenario_table, usdzar_betas)
from jse_backtester.universe import DEFAULT_BENCHMARK, USDZAR_TICKER
from jse_backtester.utils import returns_from_prices

st.set_page_config(page_title="JSE Risk & Factor Backtester", layout="wide")
st.title("JSE Portfolio Risk & Factor Backtester")


@st.cache_data(show_spinner="Loading prices...")
def get_prices(source: str, start: str):
    return load_prices(source=source, start=start)


with st.sidebar:
    source = st.radio("Data source", ["synthetic", "yfinance"], help="Synthetic data is for demos only")
    start = st.text_input("Start date", "2015-01-01")
    rf = st.number_input("Risk-free rate (annual)", 0.0, 0.25, DEFAULT_RF, 0.005)
    st.header("Strategy")
    factor = st.selectbox("Factor", list(SIGNALS))
    top = st.slider("Top fraction", 0.1, 0.5, 0.3, 0.05)
    rebalance = st.selectbox("Rebalance", ["M", "Q", "W"])
    weighting = st.selectbox("Weighting", ["equal", "inv_vol"])
    stt = st.number_input("STT on purchases", 0.0, 0.01, 0.0025, 0.0005, format="%.4f")
    brk = st.number_input("Brokerage (each way)", 0.0, 0.01, 0.0015, 0.0005, format="%.4f")

try:
    px_all = get_prices(source, start)
except Exception as exc:  # network / data errors
    st.error(f"Could not load data: {exc}")
    st.stop()

events = glitch_events(px_all)
if not events.empty:
    with st.expander(f"Data quality: sanitiser repaired/flagged {len(events)} Yahoo price events"):
        st.caption("ZAc/ZAR unit flips are rescaled to the dominant unit; reverting bad prints are removed; "
                   "large persistent moves are only flagged.")
        st.dataframe(events, use_container_width=True)

bench = returns_from_prices(px_all[[DEFAULT_BENCHMARK]])[DEFAULT_BENCHMARK] if DEFAULT_BENCHMARK in px_all else None
px, fx_px = split_fx(px_all.drop(columns=[DEFAULT_BENCHMARK], errors="ignore"))  # USD/ZAR is not a stock
fx_ret = returns_from_prices(fx_px.reindex(px.index).to_frame()).iloc[:, 0] if fx_px is not None else None


@st.cache_data(show_spinner="Estimating USD/ZAR betas...")
def get_betas(prices, fx):
    return usdzar_betas(returns_from_prices(prices), fx)


def currency_panel(exposures: dict, betas: pd.DataFrame, chart_title: str, subject: str) -> None:
    """Currency / Rand-Hedge Exposure: headline numbers, bucket split, betas chart. The first
    exposure in ``exposures`` is the subject (strategy / portfolio)."""
    st.subheader("Currency / Rand-Hedge Exposure")
    if fx_ret is None:
        st.info(f"No {USDZAR_TICKER} data was loaded, so the USD/ZAR exposure panel is unavailable. "
                "Use the synthetic or yfinance source, or add a USDZAR=X column to a CSV.")
        return
    st.caption("USD/ZAR is rand per dollar: positive beta = gains when the rand weakens (Rand Hedge), "
               f"negative = SA Inc. Buckets at +/-{USDZAR_HEDGE_THRESHOLD:.2f}. Stock betas are OLS of daily ZAR "
               "returns on daily USDZAR=X % changes with Newey-West t-stats.")
    ex = exposures[subject]
    sm = ex.summary
    w = ex.bucket_weights if ex.bucket_weights is not None else pd.Series(dtype=float)
    c = st.columns(4)
    c[0].metric("USD/ZAR beta (realised)", f"{sm['USD/ZAR Beta (realised)']:+.2f}",
                help=f"Newey-West t-stat {sm['Beta t-stat (Newey-West)']:.1f}")
    c[1].metric("Look-through beta (weighted)", f"{sm['Look-through USD/ZAR Beta (weighted)']:+.2f}",
                help="Weight-average of the holdings' own USD/ZAR betas")
    c[2].metric(f"Rolling {ex.window}d correlation", f"{sm[f'Rolling {ex.window}d corr (latest)']:+.2f}",
                help="Latest value of the rolling correlation with USD/ZAR % changes")
    c[3].metric("Rand Hedge vs SA Inc. weight",
                f"{w.get(BUCKET_HEDGE, 0.0):.0%} / {w.get(BUCKET_DOMESTIC, 0.0):.0%}")
    rolling = {k: e.rolling_corr.rename(k) for k, e in exposures.items()}
    st.pyplot(plots.plot_rand_hedge(betas, rolling, ex.window, title=chart_title))
    left, right = st.columns(2)
    wt = pd.DataFrame({k: e.bucket_weights for k, e in exposures.items() if e.bucket_weights is not None})
    left.markdown("**Average weight by bucket**")
    left.dataframe(wt.style.format("{:.1%}"))
    summ = pd.DataFrame({k: e.summary for k, e in exposures.items()}).drop(index="Observations")
    right.markdown("**Exposure summary**")
    right.dataframe(summ.style.format("{:.3f}", na_rep="n/a"))
    with st.expander("Stock-level USD/ZAR betas"):
        st.dataframe(betas.style.format({"beta": "{:+.2f}", "t_stat": "{:.1f}", "r2": "{:.1%}"}, na_rep="n/a"),
                     use_container_width=True)

tab_bt, tab_risk, tab_cost = st.tabs(["Factor backtest", "Portfolio risk", "Cost sensitivity"])

with tab_bt:
    cfg = BacktestConfig(signal=factor, top_frac=top, rebalance=rebalance,
                         weighting=weighting, stt=stt, brokerage=brk)
    res = run_backtest(px, cfg)
    b = bench.loc[res.returns.index[0]:] if bench is not None else None
    summary = res.summary(b, rf)
    c = st.columns(4)
    c[0].metric("CAGR", f"{summary['CAGR']:.1%}")
    c[1].metric("Sharpe", f"{summary['Sharpe']:.2f}")
    c[2].metric("Max drawdown", f"{summary['Max drawdown']:.1%}")
    c[3].metric("Annual turnover", f"{summary['Annual turnover (one-way)']:.0%}")
    series = {f"{factor} (net)": res.returns, f"{factor} (gross)": res.gross_returns}
    if b is not None:
        series["Benchmark"] = b
    st.pyplot(plots.plot_equity(series))
    st.pyplot(plots.plot_drawdowns({k: v for k, v in series.items() if "gross" not in k}))
    left, right = st.columns(2)
    reg = factor_regression(res.returns, build_factor_returns(px, frac=top, rebalance=rebalance, rf=rf), rf)
    left.subheader("Factor exposures"), left.pyplot(plots.plot_factor_loadings(reg.table))
    right.subheader("Signal quality (rank IC)")
    right.dataframe(ic_summary(information_coefficient(SIGNALS[factor](px), px)).to_frame("IC"))
    if fx_ret is not None:
        rets_all = returns_from_prices(px)
        betas_all = get_betas(px, fx_ret)
        ew = rets_all.mean(axis=1).loc[res.returns.index[0]:]
        exposures = {f"{factor} (net)": rand_hedge_exposure(res, rets_all, fx_ret, betas_all),
                     "Equal-weight universe": currency_exposure(
                         ew, fx_ret, equal_weight_path(rets_all).loc[res.returns.index[0]:], betas_all)}
        currency_panel(exposures, betas_all, f"Rand-hedge exposure: {factor} vs universe", f"{factor} (net)")
    else:
        currency_panel({}, pd.DataFrame(), "", "")
    st.subheader("All metrics")
    st.dataframe(summary.to_frame("Value"))

with tab_risk:
    st.write("Enter holdings (ticker, weight). Weights are normalised to 100%.")
    default = pd.read_csv("examples/sample_portfolio.csv")
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True)
    w = edited.dropna().set_index("ticker")["weight"].astype(float)
    if w.sum() > 0:
        rep = portfolio_risk_report(w, returns_from_prices(px).iloc[1:], bench, rf, fx_returns=fx_ret)
        st.caption(f"Sample: {rep.sample[0].date()} to {rep.sample[1].date()} | "
                   f"diversification ratio {rep.diversification_ratio:.2f}")
        st.dataframe(rep.summary.to_frame("Portfolio"))
        st.pyplot(plots.plot_risk_contributions(rep.contributions))
        st.subheader("Sector weights"), st.bar_chart(rep.sectors)
        st.subheader("Stress windows"), st.dataframe(scenario_table(rep.portfolio_returns))
        if rep.currency is not None:
            exposures = {"Portfolio": rep.currency}
            if bench is not None:
                exposures["Benchmark"] = currency_exposure(bench.loc[rep.sample[0]:rep.sample[1]], fx_ret)
            currency_panel(exposures, rep.currency.betas, "Rand-hedge exposure: portfolio holdings", "Portfolio")
        else:
            currency_panel({}, pd.DataFrame(), "", "")

with tab_cost:
    st.write("Net performance of every factor at each rebalance frequency across **round-trip** trading "
             "costs (buy + later sell of a position, split evenly across both legs). The default JSE model "
             "in the sidebar (0.25% STT + 0.15% brokerage each way) is about 55 bps.")
    costs = st.multiselect("Round-trip costs (bps)", [0, 15, 25, 40, 65, 100, 150, 200],
                           default=list(DEFAULT_COSTS_BPS))

    @st.cache_data(show_spinner="Running 9 backtests...")
    def get_grid(prices, costs, top, weighting, rf):
        return cost_grid(prices, costs_bps=tuple(costs), top_frac=top, weighting=weighting, rf=rf)

    if costs and st.button("Run cost grid", type="primary"):
        grid = get_grid(px, tuple(sorted(costs)), top, weighting, rf)
        st.pyplot(plots.plot_cost_grid(grid, "net_sharpe"))
        st.pyplot(plots.plot_cost_grid(grid, "net_cagr"))
        cols = st.columns(3)
        for col, (metric, label, f) in zip(cols, [("net_cagr", "Net CAGR", "{:.2%}"),
                                                  ("net_sharpe", "Net Sharpe", "{:.2f}"),
                                                  ("annual_turnover", "Annual turnover", "{:.0%}")]):
            t = grid_pivot(grid, metric)
            if metric == "annual_turnover":
                t = t.iloc[:, [0]].rename(columns=lambda c: "one-way")
            else:
                t = t.rename(columns=lambda c: f"{c:g} bps")
            t.index = [f"{a} / {b}" for a, b in t.index]
            col.subheader(label)
            col.dataframe(t.style.format(f))
        st.download_button("Download grid (CSV)", grid.to_csv(index=False), "cost_grid.csv")
