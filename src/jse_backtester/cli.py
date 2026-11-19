"""Command-line interface: ``jse-bt backtest | risk | optimize | cost-grid | charts``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import WEIGHTINGS, BacktestConfig, build_factor_returns, rand_hedge_exposure, run_backtest
from .costgrid import DEFAULT_COSTS_BPS, DEFAULT_FACTORS, DEFAULT_REBALANCES, cost_grid, grid_pivot
from .data import glitch_events, load_prices, split_fx
from .factors import SIGNALS, factor_regression, ic_summary, information_coefficient
from .optimize import DEFAULT_MAX_WEIGHT, compare_portfolios, normalise_ticker, read_portfolio_csv
from .risk import (BUCKET_DOMESTIC, BUCKET_HEDGE, BUCKET_NEUTRAL, BUCKET_UNKNOWN, CURRENCY_ROLL_WINDOW,
                   DEFAULT_RF, SCENARIOS, USDZAR_HEDGE_THRESHOLD, CurrencyExposure, currency_exposure,
                   equal_weight_path, portfolio_risk_report, risk_summary, scenario_table, usdzar_betas)
from .universe import DEFAULT_BENCHMARK, USDZAR_TICKER
from .utils import returns_from_prices

_PLAIN = ("Sharpe", "Sortino", "Calmar", "Beta", "Skew", "kurtosis", "Information", "R^2", "t-stat", "IC")
DEFAULT_CHARTS_DIR = "docs"
DEFAULT_PORTFOLIO = "examples/sample_portfolio.csv"


# ---------------------------------------------------------------- formatting
def _fmt(name: str, v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if name == "Periods":
        return f"{v:.0f}"
    return f"{v:,.2f}" if any(k in name for k in _PLAIN) else f"{v:.2%}"


def fmt_summary(df: pd.DataFrame) -> str:
    out = df.copy().astype(object)
    for i in out.index:
        for c in out.columns:
            out.loc[i, c] = _fmt(i, df.loc[i, c])
    return out.to_string()


def _fmt_grid(p: pd.DataFrame, f) -> str:
    t = p.apply(lambda c: c.map(f))
    t.columns = [f"{c:g} bps" for c in t.columns]
    t.index = [f"{a} / {b}" for a, b in t.index]
    return t.to_string()


def _dq_text(events: pd.DataFrame) -> str:
    if events is None or events.empty:
        return "## Data quality\nNo price glitches detected.\n"
    counts = events.groupby(["ticker", "kind"]).size().unstack(fill_value=0)
    shown = events.head(25).assign(date=lambda d: d["date"].dt.date.astype(str))
    more = f"\n... and {len(events) - 25} more (see data_quality.csv)" if len(events) > 25 else ""
    return ("## Data quality (Yahoo ZAc/ZAR sanitiser)\n```\n" + counts.to_string() + "\n\n"
            + shown.to_string(index=False) + more + "\n```\n")


# ---------------------------------------------------------------- USD/ZAR section
_LEAN = 0.10  # look-through beta gap vs the universe that is called out as a lean


def _fmt_ce(key: str, v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if key == "Observations":
        return f"{v:.0f}"
    if key.startswith("R^2"):
        return f"{v:.1%}"
    if "t-stat" in key:
        return f"{v:.1f}"
    return f"{v:+.2f}"


def _currency_section(
    exposures: dict[str, CurrencyExposure],
    universe: str | None = None,
    stock_table: pd.DataFrame | None = None,
    stock_note: str = "",
) -> str:
    """Markdown/text block shared by ``risk`` and ``backtest``. The first exposure is the subject;
    ``universe`` names the exposure to measure the tilt against (an equal-weight universe)."""
    names = list(exposures)
    subject = exposures[names[0]]
    win = subject.window
    thr = USDZAR_HEDGE_THRESHOLD
    summ = pd.DataFrame({n: e.summary for n, e in exposures.items()})
    tbl = pd.DataFrame({n: [_fmt_ce(k, v) for k, v in summ[n].items()] for n in names}, index=summ.index)
    if all(e.lookthrough_beta is None for e in exposures.values()):
        tbl = tbl.drop(index="Look-through USD/ZAR Beta (weighted)")
    parts = [
        "## USD/ZAR exposure & rand-hedge breakdown\n"
        "USD/ZAR is rand per dollar: a positive beta means a holding gains when the rand weakens "
        f"(Rand Hedge), a negative beta means it suffers (SA Inc.). Stock betas are OLS of daily ZAR returns "
        f"on daily USDZAR=X % changes with Newey-West t-stats (5 lags). Buckets: Rand Hedge > +{thr:.2f}, "
        f"SA Inc. < -{thr:.2f}, Neutral in between. 'Realised' regresses the portfolio's own returns on "
        "USD/ZAR; 'look-through' averages its holdings' stock betas by weight.\n",
        "```\n" + tbl.to_string() + "\n```\n",
    ]

    wcols = {n: e.bucket_weights for n, e in exposures.items() if e.bucket_weights is not None}
    if wcols:
        rows = [b for b in (BUCKET_HEDGE, BUCKET_NEUTRAL, BUCKET_DOMESTIC, BUCKET_UNKNOWN)
                if any(b in w.index for w in wcols.values())]
        wt = pd.DataFrame({n: w.reindex(rows).fillna(0.0) for n, w in wcols.items()})
        out = wt.apply(lambda c: c.map("{:.1%}".format))
        out.index.name = "avg net weight"
        if universe in wt.columns and names[0] in wt.columns and universe != names[0]:
            out["tilt vs universe"] = (wt[names[0]] - wt[universe]).map(lambda v: f"{v * 100:+.1f} pts")
        parts.append("### Weight by bucket\n```\n" + out.to_string() + "\n```\n")
        gross = exposures[names[0]].bucket_gross
        if gross is not None and not np.allclose(gross.reindex(wt.index).fillna(0), wt[names[0]].abs(), atol=1e-6):
            g = gross.reindex(rows).fillna(0.0).map("{:.1%}".format).to_frame("avg gross weight")
            parts.append("Long/short book: gross exposure by bucket\n```\n" + g.to_string() + "\n```\n")

    if universe in exposures and exposures[names[0]].lookthrough_beta is not None and universe != names[0]:
        d = exposures[names[0]].summary["Look-through USD/ZAR Beta (weighted)"] \
            - exposures[universe].summary["Look-through USD/ZAR Beta (weighted)"]
        hedge = wt.loc[BUCKET_HEDGE, names[0]] - wt.loc[BUCKET_HEDGE, universe] if BUCKET_HEDGE in wt.index else 0.0
        dom = wt.loc[BUCKET_DOMESTIC, names[0]] - wt.loc[BUCKET_DOMESTIC, universe] if BUCKET_DOMESTIC in wt.index else 0.0
        lean = ("leans toward Rand Hedge names" if d > _LEAN else
                "leans toward SA Inc. names" if d < -_LEAN else "carries no material currency tilt")
        parts.append(f"Versus the equal-weight universe, {names[0].replace(' (net)', '')} {lean}: look-through beta {d:+.2f}, "
                     f"Rand Hedge weight {hedge * 100:+.1f} pts, SA Inc. weight {dom * 100:+.1f} pts "
                     f"(a gap beyond +/-{_LEAN:.2f} in look-through beta is called out as a lean).\n")

    if stock_table is not None:
        t = stock_table.copy()
        for c, f in (("weight", "{:.1%}"), ("beta", "{:+.2f}"), ("t_stat", "{:.1f}")):
            if c in t:
                t[c] = t[c].map(lambda v, f=f: "n/a" if pd.isna(v) else f.format(v))
        cols = [c for c in ("weight", "beta", "t_stat", "bucket") if c in t]
        parts.append(f"### Stock USD/ZAR betas{stock_note}\n```\n" + t[cols].to_string() + "\n```\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- shared plumbing
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="yfinance")
    p.add_argument("--csv", help="price CSV (date index, one column per ticker) when --source csv")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--rf", type=float, default=DEFAULT_RF, help="annual risk-free rate, e.g. 0.07")
    p.add_argument("--out", default="reports/latest", help="output directory")
    p.add_argument("--save-charts", nargs="?", const=DEFAULT_CHARTS_DIR, default=None, metavar="DIR",
                   help=f"also write README-ready PNGs to DIR (default: {DEFAULT_CHARTS_DIR}/)")


def _load(args, tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame, pd.Series | None]:
    """Returns (share prices, benchmark returns, sanitiser events, USD/ZAR daily % changes).

    USDZAR=X is split off here so it is never ranked, weighted or risk-decomposed as a stock.
    ``tickers`` (default: the JSE universe) is what yfinance downloads / the simulator generates.
    """
    px = load_prices(tickers=tickers, start=args.start, end=args.end, source=args.source,
                     benchmark=args.benchmark, csv_path=args.csv)
    events = glitch_events(px)
    bench = None
    if args.benchmark in px.columns:
        bench = returns_from_prices(px[[args.benchmark]])[args.benchmark].rename("Benchmark")
        px = px.drop(columns=[args.benchmark])
    else:
        print(f"[warn] benchmark {args.benchmark} not in data; skipping benchmark metrics")
    px, fx = split_fx(px)
    fx_ret = None
    if fx is not None:
        fx_ret = returns_from_prices(fx.reindex(px.index).to_frame()).iloc[:, 0].rename("USDZAR")
    else:
        print(f"[warn] {USDZAR_TICKER} not in data; skipping the USD/ZAR exposure section "
              "(csv source: add a USDZAR=X column)")
    if not events.empty:
        print(f"[data] sanitiser repaired/flagged {len(events)} price events "
              f"({(events['kind'] == 'unit_flip').sum()} unit flips); details in data_quality.csv")
    return px, bench, events, fx_ret


def _note(args, px: pd.DataFrame) -> str:
    if args.source == "synthetic":
        return "SIMULATED DATA - illustrative only, not market performance"
    src = "Yahoo Finance" if args.source == "yfinance" else "user-supplied prices"
    return f"Source: {src}, adjusted closes {px.index[0]:%b %Y} - {px.index[-1]:%b %Y} | jse-bt"


def _emit(fig, name: str, out: Path, args, note: str) -> None:
    """Save into the report folder, and (with --save-charts) a README-ready copy."""
    targets = [(out / name, 130)]
    if args.save_charts:
        charts = Path(args.save_charts)
        charts.mkdir(parents=True, exist_ok=True)
        targets.append((charts / name, 160))
    plots.save_figure(fig, targets, note)


def _prep_out(args, events: pd.DataFrame) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "data_quality.csv", index=False)
    return out


# ---------------------------------------------------------------- commands
def _weighting_label(args) -> str:
    if args.weighting == "min_var":
        return f"min_var (cap {args.max_weight:.0%}, trailing 252d shrunk cov)"
    if args.weighting == "risk_parity":
        return "risk_parity (trailing 252d shrunk cov)"
    return args.weighting


def cmd_backtest(args) -> None:
    px, bench, events, fx_ret = _load(args)
    out = _prep_out(args, events)
    note = _note(args, px)
    cfg = BacktestConfig(
        signal=args.factor, top_frac=args.top_frac, long_only=not args.long_short,
        weighting=args.weighting, rebalance=args.rebalance, stt=args.stt, brokerage=args.brokerage,
        max_weight=args.max_weight,
    )
    res = run_backtest(px, cfg)
    start = res.returns.index[0]
    ew = returns_from_prices(px).mean(axis=1).loc[start:]
    b = bench.loc[start:] if bench is not None else None

    table = pd.DataFrame({f"{args.factor} (net)": res.summary(b, args.rf)})
    table["Equal-weight universe"] = risk_summary(ew, b, args.rf)
    if b is not None:
        table["Benchmark"] = risk_summary(b, None, args.rf)
    text = [f"# JSE factor backtest: {args.factor}\n",
            f"Universe: {px.shape[1]} stocks | {start.date()} to {res.returns.index[-1].date()} | "
            f"rebalance {args.rebalance} | top {args.top_frac:.0%} | {_weighting_label(args)} | "
            f"STT {args.stt:.2%} + brokerage {args.brokerage:.2%}\n",
            "## Performance & risk\n```\n" + fmt_summary(table) + "\n```\n"]

    ic = ic_summary(information_coefficient(SIGNALS[args.factor](px), px))
    text.append("## Signal quality (21d forward rank IC)\n```\n" + fmt_summary(ic.to_frame("IC")) + "\n```\n")

    factors = build_factor_returns(px, frac=args.top_frac, rebalance=args.rebalance, rf=args.rf)
    reg = factor_regression(res.returns, factors, args.rf)
    text.append(f"## Factor exposures (R² {reg.r2:.2f}, n={reg.n_obs})\n```\n"
                + reg.table.round(3).to_string() + "\n```\n")

    currency_brief = ""
    rolling = {}
    if fx_ret is not None:
        rets = returns_from_prices(px)
        betas = usdzar_betas(rets, fx_ret)
        label = f"{args.factor} (net)"
        exposures = {label: rand_hedge_exposure(res, rets, fx_ret, betas)}
        exposures["Equal-weight universe"] = currency_exposure(ew, fx_ret, equal_weight_path(rets).loc[start:], betas)
        if b is not None:
            exposures["Benchmark"] = currency_exposure(b, fx_ret)
        currency_brief = _currency_section(exposures, universe="Equal-weight universe")
        text.append(currency_brief)
        text.append("### Stock USD/ZAR betas (whole universe, each stock's full history)\n```\n"
                    + betas.assign(beta=betas.beta.map("{:+.2f}".format), t_stat=betas.t_stat.map("{:.1f}".format))
                    [["beta", "t_stat", "bucket"]].to_string() + "\n```\n")
        betas.to_csv(out / "usdzar_betas.csv")
        # chart: strategy vs the equal-weight universe (the benchmark stays in the table above)
        rolling = {k: e.rolling_corr.rename(k) for k, e in exposures.items() if k != "Benchmark"}
    scen = scenario_table(res.returns)
    if not scen.empty:
        text.append("## Stress windows\n```\n" + scen.round(4).to_string() + "\n```\n")
    text.append(_dq_text(events))

    series = {f"{args.factor} (net)": res.returns, "Equal-weight": ew}
    if b is not None:
        series["Benchmark"] = b
    _emit(plots.plot_equity(series), "equity.png", out, args, note)
    _emit(plots.plot_drawdowns(series), "drawdowns.png", out, args, note)
    _emit(plots.plot_factor_loadings(reg.table), "factor_loadings.png", out, args, note)
    images = "![equity](equity.png)\n\n![drawdowns](drawdowns.png)\n\n![factors](factor_loadings.png)\n"
    if rolling:
        _emit(plots.plot_rand_hedge(betas, rolling, CURRENCY_ROLL_WINDOW,
                                    title=f"Rand-hedge exposure: {args.factor} vs universe"),
              "rand_hedge.png", out, args, note)
        images += "\n![rand hedge](rand_hedge.png)\n"
    text.append(images)

    table.to_csv(out / "summary.csv")
    res.weights.to_csv(out / "weights.csv")
    (out / "report.md").write_text("\n".join(text))
    print("\n".join(text[:4] + ([currency_brief] if currency_brief else [])))
    print(f"Report written to {out}/" + (f" (charts also in {args.save_charts}/)" if args.save_charts else ""))


def cmd_risk(args) -> None:
    px, bench, events, fx_ret = _load(args)
    out = _prep_out(args, events)
    note = _note(args, px)
    wdf = pd.read_csv(args.weights)
    weights = wdf.set_index(wdf.columns[0])[wdf.columns[1]].astype(float)
    missing = [t for t in weights.index if t not in px.columns]
    if missing:
        print(f"[warn] holdings with no price data (dropped, weights renormalised): {missing}")
    rets = returns_from_prices(px).iloc[1:]
    rep = portfolio_risk_report(weights, rets, bench, args.rf, fx_returns=fx_ret)

    text = ["# JSE portfolio risk report\n",
            f"Sample: {rep.sample[0].date()} to {rep.sample[1].date()} (common history of all holdings)\n",
            "## Risk summary\n```\n" + fmt_summary(rep.summary.to_frame("Portfolio")) + "\n```\n",
            f"Diversification ratio: {rep.diversification_ratio:.2f}\n",
            "## Sector weights\n```\n" + rep.sectors.map("{:.1%}".format).to_string() + "\n```\n",
            "## Risk contributions\n```\n"
            + rep.contributions.assign(weight=lambda d: d.weight.map("{:.1%}".format),
                                       pct=lambda d: d.pct.map("{:.1%}".format))[["weight", "pct"]]
            .rename(columns={"pct": "risk share"}).to_string() + "\n```\n"]

    factors = build_factor_returns(px, rf=args.rf)
    reg = factor_regression(rep.portfolio_returns, factors, args.rf)
    text.append(f"## Factor exposures (R² {reg.r2:.2f})\n```\n" + reg.table.round(3).to_string() + "\n```\n")
    rolling = {}
    if rep.currency is not None:
        ce = rep.currency
        w_norm = weights[weights.index.isin(rets.columns)]
        w_norm = w_norm / w_norm.sum()
        exposures = {"Portfolio": ce}
        if bench is not None:
            exposures["Benchmark"] = currency_exposure(bench.loc[rep.sample[0]:rep.sample[1]], fx_ret)
        held = ce.betas.assign(weight=w_norm.reindex(ce.betas.index)).sort_values("weight", ascending=False)
        text.append(_currency_section(exposures, stock_table=held, stock_note=" (holdings, common sample)"))
        ce.betas.to_csv(out / "usdzar_betas.csv")
        rolling = {k: e.rolling_corr.rename(k) for k, e in exposures.items()}
    scen = scenario_table(rep.portfolio_returns, SCENARIOS)
    if not scen.empty:
        text.append("## Stress windows (static weights)\n```\n" + scen.round(4).to_string() + "\n```\n")
    text.append(_dq_text(events))

    if rolling:
        _emit(plots.plot_rand_hedge(rep.currency.betas, rolling, CURRENCY_ROLL_WINDOW,
                                    title="Rand-hedge exposure: portfolio holdings"),
              "risk_rand_hedge.png", out, args, note)
    _emit(plots.plot_risk_contributions(rep.contributions), "risk_contributions.png", out, args, note)
    _emit(plots.plot_factor_loadings(reg.table), "risk_factor_loadings.png", out, args, note)
    series = {"Portfolio": rep.portfolio_returns}
    if bench is not None:
        series["Benchmark"] = bench.loc[rep.sample[0]:rep.sample[1]]
    _emit(plots.plot_drawdowns(series), "risk_drawdowns.png", out, args, note)
    (out / "report.md").write_text("\n".join(text))
    print("\n".join(text))
    print(f"Report written to {out}/" + (f" (charts also in {args.save_charts}/)" if args.save_charts else ""))


def _pct(v: float) -> str:
    return f"{v:.1%}"


def optimisation_tables(res) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-stock weight / risk-share table, portfolio metrics table) as display strings."""
    cols = {}
    for m in res.weights.columns:
        cols[(res.labels[m], "weight")] = res.weights[m]
        cols[(res.labels[m], "risk share")] = res.risk_shares[m]
    raw = pd.DataFrame(cols)
    raw.loc["Total"] = raw.sum()
    per_stock = raw.apply(lambda c: c.map(_pct))
    fmt = {"Ann. volatility": _pct, "Max drawdown": _pct, "Sharpe": "{:.2f}".format,
           "Diversification ratio": "{:.2f}".format, "Look-through USD/ZAR Beta": "{:+.2f}".format}
    metrics = res.metrics.copy().astype(object)
    for row, f in fmt.items():
        metrics.loc[row] = [("n/a" if pd.isna(v) else f(v)) for v in res.metrics.loc[row]]
    metrics.columns = [res.labels[m] for m in res.metrics.columns]
    return per_stock, metrics


def cmd_optimize(args) -> None:
    if args.tickers:
        holdings, tickers = None, list(dict.fromkeys(normalise_ticker(t) for t in args.tickers))
    else:
        path = args.weights or DEFAULT_PORTFOLIO
        if not Path(path).exists():
            sys.exit(f"jse-bt optimize: portfolio file {path!r} not found (pass --weights FILE or --tickers A B C ...)")
        try:
            holdings = read_portfolio_csv(path)
        except ValueError as exc:
            sys.exit(f"jse-bt optimize: {exc}")
        tickers = list(holdings.index)

    px, bench, events, fx_ret = _load(args, tickers)
    out = _prep_out(args, events)
    note = _note(args, px)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        print(f"[warn] holdings with no price data (dropped): {missing}")
    rets = returns_from_prices(px).iloc[1:]
    try:
        res = compare_portfolios(rets, tickers, max_weight=args.max_weight, rf=args.rf,
                                 fx_returns=fx_ret, current=holdings)
    except ValueError as exc:
        sys.exit(f"jse-bt optimize: {exc}")

    per_stock, metrics = optimisation_tables(res)
    n = len(res.weights)
    text = [
        "# JSE portfolio optimisation: Equal Weight vs Min Variance vs Risk Parity\n",
        f"Universe: {n} stocks | sample {res.sample[0].date()} to {res.sample[1].date()} "
        f"({res.n_obs} common daily returns) | max weight {args.max_weight:.0%} (Min Variance) | "
        f"risk-free {args.rf:.1%}\n",
        "Covariance: sample covariance shrunk 10% toward constant correlation (`risk.shrunk_cov`), the same "
        "estimator as the risk report. Risk share = Euler contribution to portfolio volatility "
        "`w_i (S w)_i / (w' S w)`; Risk Parity sets every share to 1/N. Everything here is in-sample; for "
        "an out-of-sample test use `jse-bt backtest --weighting min_var|risk_parity` (trailing 252-day "
        "re-estimation, no look-ahead).\n",
        "## Capital weight vs Euler risk share\n```\n" + per_stock.to_string() + "\n```\n",
        f"Risk Parity: largest gap between a risk share and 1/N = {res.risk_parity_gap * 100:.6f} percentage "
        f"points (target share {1 / n:.2%}).\n",
        "## Portfolio metrics (in-sample, static weights)\n```\n" + metrics.to_string() + "\n```\n",
        _dq_text(events), "![optimisation](optimize.png)\n",
    ]
    _emit(plots.plot_optimisation(res.weights, res.risk_shares, res.metrics, res.labels), "optimize.png", out, args, note)

    res.weights.rename(columns=res.labels).rename_axis("ticker").to_csv(out / "weights.csv")
    res.risk_shares.rename(columns=res.labels).rename_axis("ticker").to_csv(out / "risk_shares.csv")
    res.metrics.rename(columns=res.labels).to_csv(out / "metrics.csv")
    (out / "report.md").write_text("\n".join(text))
    print("\n".join(text[:-1]))
    print(f"Report written to {out}/" + (f" (chart also in {args.save_charts}/)" if args.save_charts else ""))


def cmd_cost_grid(args) -> None:
    px, _, events, _ = _load(args)
    out = _prep_out(args, events)
    note = _note(args, px)
    grid = cost_grid(px, tuple(args.factors), tuple(args.rebalances), tuple(args.costs_bps),
                     args.top_frac, args.weighting, args.rf)
    grid.to_csv(out / "cost_grid.csv", index=False)

    turn = grid_pivot(grid, "annual_turnover").iloc[:, [0]]
    turn.columns = [0]
    text = [
        "# Transaction-cost sensitivity grid\n",
        f"Universe: {px.shape[1]} stocks | top {args.top_frac:.0%} | {args.weighting} weights | "
        f"risk-free {args.rf:.1%}\n",
        "Round-trip cost = buy + later sell of 100% of a position, split evenly across both legs. "
        "Reference: the default JSE model (0.25% STT + 0.15% brokerage each way) is about 55 bps.\n",
        "## Net CAGR\n```\n" + _fmt_grid(grid_pivot(grid, "net_cagr"), lambda v: f"{v:.2%}") + "\n```\n",
        "## Net Sharpe\n```\n" + _fmt_grid(grid_pivot(grid, "net_sharpe"), lambda v: f"{v:.2f}") + "\n```\n",
        "## Annual turnover (one-way, independent of cost)\n```\n"
        + _fmt_grid(turn, lambda v: f"{v:.0%}").replace("0 bps", "turnover") + "\n```\n",
        _dq_text(events),
    ]
    _emit(plots.plot_cost_grid(grid, "net_sharpe"), "cost_grid_sharpe.png", out, args, note)
    _emit(plots.plot_cost_grid(grid, "net_cagr"), "cost_grid_cagr.png", out, args, note)
    text.append("![sharpe](cost_grid_sharpe.png)\n\n![cagr](cost_grid_cagr.png)\n")
    (out / "cost_grid.md").write_text("\n".join(text))
    print("\n".join(text[:6]))
    print(f"Grid written to {out}/" + (f" (charts also in {args.save_charts}/)" if args.save_charts else ""))


def cmd_charts(args) -> None:
    """Regenerate every README chart into DIR from the chosen data source."""
    charts = args.dir
    common = ["--source", args.source, "--start", args.start, "--benchmark", args.benchmark,
              "--rf", str(args.rf), "--save-charts", charts]
    if args.end:
        common += ["--end", args.end]
    if args.csv:
        common += ["--csv", args.csv]
    base = Path(args.out)
    main(["backtest", *common, "--out", str(base / "backtest")])
    main(["risk", *common, "--out", str(base / "risk"), "--weights", args.weights])
    main(["optimize", *common, "--out", str(base / "optimize"), "--weights", args.weights])
    main(["cost-grid", *common, "--out", str(base / "cost_grid")])
    names = sorted(p.name for p in Path(charts).glob("*.png"))
    print(f"\nREADME charts in {charts}/: {', '.join(names)}")


# ---------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jse-bt", description="JSE portfolio risk & factor backtester")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="backtest a rank-based factor strategy")
    _add_common(bt)
    bt.add_argument("--factor", choices=list(SIGNALS), default="momentum")
    bt.add_argument("--top-frac", type=float, default=0.3)
    bt.add_argument("--long-short", action="store_true", help="long top / short bottom (research only)")
    bt.add_argument("--weighting", choices=list(WEIGHTINGS), default="equal",
                    help="min_var / risk_parity re-estimate the shrunk covariance on the trailing 252 days "
                         "before each rebalance (no look-ahead)")
    bt.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT,
                    help="per-stock cap for --weighting min_var (default 0.20)")
    bt.add_argument("--rebalance", choices=["W", "M", "Q"], default="M")
    bt.add_argument("--stt", type=float, default=0.0025, help="Securities Transfer Tax on purchases")
    bt.add_argument("--brokerage", type=float, default=0.0015, help="brokerage + levies, each way")
    bt.set_defaults(func=cmd_backtest)

    rk = sub.add_parser("risk", help="risk report for a static-weight portfolio")
    _add_common(rk)
    rk.add_argument("--weights", default="examples/sample_portfolio.csv", help="CSV: ticker,weight")
    rk.set_defaults(func=cmd_risk)

    op = sub.add_parser("optimize", help="Equal Weight vs Min Variance vs Risk Parity for a set of JSE holdings")
    _add_common(op)
    grp = op.add_mutually_exclusive_group()
    grp.add_argument("--weights", help=f"CSV: ticker,weight (default {DEFAULT_PORTFOLIO}); also shown as 'Current'")
    grp.add_argument("--tickers", nargs="+", metavar="TICKER", help="e.g. NPN.JO SBK.JO AGL.JO (the .JO suffix is optional)")
    op.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT,
                    help="per-stock cap for Min Variance (default 0.20)")
    op.set_defaults(func=cmd_optimize)

    cg = sub.add_parser("cost-grid", help="net CAGR / Sharpe / turnover across costs and rebalance frequencies")
    _add_common(cg)
    cg.add_argument("--factors", nargs="+", choices=list(SIGNALS), default=list(DEFAULT_FACTORS))
    cg.add_argument("--rebalances", nargs="+", choices=["W", "M", "Q"], default=list(DEFAULT_REBALANCES))
    cg.add_argument("--costs-bps", nargs="+", type=float, default=list(DEFAULT_COSTS_BPS),
                    help="round-trip costs in basis points")
    cg.add_argument("--top-frac", type=float, default=0.3)
    cg.add_argument("--weighting", choices=["equal", "inv_vol"], default="equal")
    cg.set_defaults(func=cmd_cost_grid)

    ch = sub.add_parser("charts", help="regenerate all README charts (backtest + risk + cost grid)")
    _add_common(ch)
    ch.add_argument("--dir", default=DEFAULT_CHARTS_DIR, help="where README PNGs go")
    ch.add_argument("--weights", default="examples/sample_portfolio.csv")
    ch.set_defaults(func=cmd_charts, out="reports/charts")

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
