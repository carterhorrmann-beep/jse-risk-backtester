"""Matplotlib charts. Each function returns a Figure so callers can save or embed it."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from .optimize import METHOD_LABELS  # noqa: E402
from .risk import BUCKET_DOMESTIC, BUCKET_HEDGE, BUCKET_NEUTRAL, USDZAR_HEDGE_THRESHOLD, drawdown_series  # noqa: E402

_STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25}


def plot_equity(series: dict[str, pd.Series]) -> plt.Figure:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, r in series.items():
            ax.plot((1 + r.fillna(0)).cumprod(), label=name, lw=1.6)
        ax.set_title("Growth of R1"), ax.set_yscale("log"), ax.legend(frameon=False)
        fig.tight_layout()
    return fig


def plot_drawdowns(series: dict[str, pd.Series]) -> plt.Figure:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for name, r in series.items():
            dd = drawdown_series(r.fillna(0))
            ax.fill_between(dd.index, dd, 0, alpha=0.35, label=name)
        ax.set_title("Drawdowns"), ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


def plot_risk_contributions(contrib: pd.DataFrame, top: int = 15) -> plt.Figure:
    df = contrib.sort_values("pct", ascending=False).head(top)[::-1]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
        y = range(len(df))
        ax.barh([i + 0.2 for i in y], df["weight"], height=0.4, label="Weight")
        ax.barh([i - 0.2 for i in y], df["pct"], height=0.4, label="Share of risk")
        ax.set_yticks(list(y)), ax.set_yticklabels(df.index)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title("Weight vs contribution to volatility"), ax.legend(frameon=False)
        fig.tight_layout()
    return fig


WEIGHT_COLOR, RISK_COLOR = "#4c72b0", "#dd8452"


def plot_optimisation(
    weights: pd.DataFrame,
    risk_shares: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Grouped bars of capital weight vs Euler risk contribution per stock, one panel per method.

    Panels follow ``METHOD_LABELS`` (Equal Weight, Min Variance, Risk Parity) for whichever of those
    columns ``weights`` has; the dashed line is 1/N, where every Risk Parity risk bar sits. With
    ``metrics`` (from ``compare_portfolios``) each panel title also shows volatility and the
    diversification ratio.
    """
    labels = {**METHOD_LABELS, **(labels or {})}
    cols = [m for m in METHOD_LABELS if m in weights.columns]
    if not cols:
        raise ValueError("weights has none of the columns equal / min_var / risk_parity")
    n = len(weights)
    x = np.arange(n)
    names = [t.replace(".JO", "") for t in weights.index]
    top = max(weights[cols].to_numpy().max(), risk_shares[cols].to_numpy().max()) * 1.18
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(len(cols), 1, figsize=(max(8.0, 0.62 * n + 2.5), 2.75 * len(cols) + 0.9),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, m in zip(axes[:, 0], cols):
            ax.bar(x - 0.2, weights[m], 0.4, color=WEIGHT_COLOR, label="Capital weight")
            ax.bar(x + 0.2, risk_shares[m], 0.4, color=RISK_COLOR, label="Risk contribution")
            ax.axhline(1.0 / n, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.text(1.005, 1.0 / n, "1/N", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=8, alpha=0.7)
            title = labels[m]
            if metrics is not None and m in metrics.columns:
                title += (f"   |   volatility {metrics.loc['Ann. volatility', m]:.1%}"
                          f"   diversification ratio {metrics.loc['Diversification ratio', m]:.2f}")
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylim(0, top)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))  # 0, 5%, 10%, ...
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axes[0, 0].legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
        axes[-1, 0].set_xticks(x)
        axes[-1, 0].set_xticklabels(names, rotation=90, fontsize=8)
        fig.suptitle("Capital weights vs risk contributions: Equal Weight, Min Variance, Risk Parity", fontsize=11)
        fig.tight_layout()
    return fig


def plot_factor_loadings(table: pd.DataFrame) -> plt.Figure:
    df = table.drop(index="alpha")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(df.index, df["coef"], color=["#2a7" if t > 2 else "#999" for t in df["t_stat"].abs()])
        for b, t in zip(bars, df["t_stat"]):
            ax.annotate(f"t={t:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("Factor loadings (green = |t| > 2, Newey-West)")
        fig.tight_layout()
    return fig


def plot_cost_grid(grid: pd.DataFrame, metric: str = "net_sharpe") -> plt.Figure:
    """One heatmap per factor: rows = rebalance frequency, columns = round-trip cost (bps)."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    pct = metric.endswith("cagr")
    title = {"net_sharpe": "Net Sharpe ratio", "net_cagr": "Net CAGR"}.get(metric, metric)
    factors = list(dict.fromkeys(grid["factor"]))
    vals = grid[metric]
    lo, hi = float(vals.min()), float(vals.max())
    norm = TwoSlopeNorm(vmin=min(lo, -1e-9), vcenter=0.0, vmax=max(hi, 1e-9)) if lo < 0 < hi else None
    with plt.rc_context({"axes.grid": False}):
        fig, axes = plt.subplots(1, len(factors), figsize=(4.2 * len(factors), 3.3), squeeze=False, sharey=True)
        for ax, f in zip(axes[0], factors):
            sub = grid[grid["factor"] == f].pivot(index="rebalance", columns="cost_bps", values=metric)
            sub = sub.reindex([r for r in ("W", "M", "Q") if r in sub.index])
            im = ax.imshow(sub.to_numpy(), cmap="RdYlGn", norm=norm, aspect="auto",
                           vmin=None if norm else lo, vmax=None if norm else hi)
            ax.set_xticks(range(sub.shape[1]), [f"{c:g}" for c in sub.columns])
            ax.set_yticks(range(sub.shape[0]), sub.index)
            ax.set_title(f.replace("_", " "), fontsize=10)
            ax.set_xlabel("round-trip cost (bps)")
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub.iat[i, j]
                    ax.text(j, i, f"{v:.1%}" if pct else f"{v:.2f}", ha="center", va="center", fontsize=8)
        axes[0][0].set_ylabel("rebalance")
        fig.suptitle(title + " by trading cost and rebalance frequency", fontsize=11)
        fig.tight_layout()
    return fig


BUCKET_COLORS = {BUCKET_HEDGE: "#2a7", BUCKET_NEUTRAL: "#999999", BUCKET_DOMESTIC: "#d9534f"}


def plot_rand_hedge(
    betas: pd.DataFrame,
    rolling: dict[str, pd.Series],
    window: int = 126,
    threshold: float = USDZAR_HEDGE_THRESHOLD,
    title: str = "Rand-hedge exposure",
) -> plt.Figure:
    """Left: each stock's USD/ZAR beta by bucket. Right: rolling correlation with USD/ZAR.

    Bars with |Newey-West t| < 2 are faded (the beta is not distinguishable from zero). ``rolling``
    maps a label (strategy, universe, benchmark...) to its rolling-correlation series.
    """
    df = betas.dropna(subset=["beta"]).sort_values("beta", ascending=False)
    with plt.rc_context(_STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
        x = range(len(df))
        faded = df["t_stat"].abs() < 2
        ax.bar(x, df["beta"], color=[BUCKET_COLORS.get(b, "#999999") for b in df["bucket"]],
               alpha=1.0, width=0.8)
        for i, f in zip(x, faded):
            if f:
                ax.patches[i].set_alpha(0.35)
        for y, ls in ((threshold, "--"), (-threshold, "--"), (0, "-")):
            ax.axhline(y, color="k", lw=0.8 if y == 0 else 0.7, ls=ls, alpha=0.6)
        ax.set_xticks(list(x)), ax.set_xticklabels([t.replace(".JO", "") for t in df.index], rotation=90, fontsize=8)
        ax.set_xlim(-0.8, len(df) - 0.2)
        counts = df["bucket"].value_counts()
        handles = [plt.Rectangle((0, 0), 1, 1, color=BUCKET_COLORS[b]) for b in BUCKET_COLORS]
        labels = [f"{b} ({counts.get(b, 0)})" for b in BUCKET_COLORS]
        ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
        ax.set_ylabel("USD/ZAR beta")
        ax.set_title(f"Stock USD/ZAR betas (dashed = +/-{threshold:.2f}; faded = |t| < 2)", fontsize=10)

        # first series is the subject (strategy / portfolio): bold; the rest are muted references
        muted = [("#888888", "-"), ("#c77d2b", "--"), ("#6a5acd", ":")]
        for k, (name, r) in enumerate(rolling.items()):
            if not len(r):
                continue
            if k == 0:
                bx.plot(r.index, r, lw=1.9, color="#1f77b4", label=name, zorder=3)
            else:
                c, ls = muted[(k - 1) % len(muted)]
                bx.plot(r.index, r, lw=1.4, color=c, ls=ls, label=name, zorder=2)
        bx.axhline(0, color="k", lw=0.8)
        bx.set_ylabel("correlation with USD/ZAR")
        bx.set_title(f"Rolling {window}-day correlation with USD/ZAR", fontsize=10)
        bx.legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(title + "  (USD/ZAR up = rand weaker; positive = Rand Hedge)", fontsize=11)
        fig.tight_layout()
    return fig


def add_footer(fig: plt.Figure, note: str) -> None:
    fig.text(0.995, 0.005, note, ha="right", va="bottom", fontsize=7, color="#777777")


def save_figure(fig: plt.Figure, targets: list[tuple], note: str = "") -> None:
    """Save to every (path, dpi) target with a source footer, then close the figure."""
    if note:
        add_footer(fig, note)
    for path, dpi in targets:
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
