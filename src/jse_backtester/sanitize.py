"""Sanitiser for Yahoo Finance ``.JO`` price glitches.

Yahoo quotes JSE shares in South African cents (ZAc) but occasionally serves a day (or a
run of days) in rands (ZAR), or the reverse. In a price series that appears as a ~100x or
~0.01x jump, which becomes a +9,900% / -99% "return" and wrecks volatility, covariance and
risk-contribution numbers. Isolated one-day bad prints that revert next day show up too.

Stages, applied per ticker on its valid prices:

1. **Unit flips.** A day-on-day ratio within ``tol`` (log units) of 100^m, m = +/-1, +/-2...
   marks a unit change. The series is cut into segments at those points, each segment is
   assigned a unit exponent, and every segment is rescaled to the *dominant* unit (the one
   covering most days; ties go to the most recent). Real returns are preserved because only
   the ~100x factor is removed, not the genuine move that happened that day. Nothing real
   moves 100x in a day, so genuine corporate-action drops (-40%, say) are never touched.
2. **Reverting spikes.** A move beyond ``spike`` (log) that is undone by the next day
   (|ln(p[t+1]/p[t-1])| < ``revert``) is a bad print: set to NaN so it can be forward-filled.
   Persistent moves (corporate actions, genuine crashes) don't revert and are left alone.
3. **Flags.** Remaining moves beyond ``spike`` are reported but NOT modified.

**FX series (``USDZAR=X``).** Yahoo quotes USD/ZAR in rands per dollar (~12-20), not cents, so
there is no ZAc/ZAR unit to flip and the 100x detector must stay silent on it. That already
holds: stage 1 only fires on day-on-day ratios within ``tol`` of 100^m, and no genuine currency
move (even Nenegate's ~-10% day) is anywhere near 10x. Bad prints that *do* happen (a x100
tick, a decimal slip) are still repaired by stages 1-2. FX columns get tighter spike/revert
thresholds (``FX_PARAMS``) because a 35% one-day move that reverts is far beyond any real
USD/ZAR move, but a 12-15% bad print is not. A level check (``FX_PLAUSIBLE``) flags a whole
series quoted in the wrong unit or inverted (ZARUSD ~0.055), which a jump detector cannot see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import FX_TICKERS

LN100 = float(np.log(100.0))
EVENT_COLS = ["ticker", "date", "kind", "detail"]

# Tighter bad-print thresholds for currency pairs (log moves). Real USD/ZAR days are rarely
# beyond +/-6% (Nenegate, COVID peak); 15% that reverts within 5% the next day is a bad tick.
FX_PARAMS: dict = {"spike": 0.15, "revert": 0.05}
# Median rand-per-dollar level that is plausible (history: ~5.6 to ~19.9). Outside -> flag.
FX_PLAUSIBLE: tuple[float, float] = (3.0, 60.0)


def sanitize_series(
    s: pd.Series,
    ticker: str = "",
    tol: float = 0.4,
    spike: float = 0.35,
    revert: float = 0.15,
) -> tuple[pd.Series, list[tuple]]:
    """Return (cleaned series on the original index, list of event tuples)."""
    orig_index = s.index
    s = s.dropna()
    events: list[tuple] = []
    if len(s) < 5:
        return s.reindex(orig_index), events

    # ---- 1) unit flips -------------------------------------------------------------
    lr = np.log(s).diff()
    m = np.round(lr / LN100)  # which power of 100 (NaN on first row)
    flip = ((m != 0) & ((lr - m * LN100).abs() < tol)).to_numpy()
    if flip.any():
        exponent = pd.Series(np.where(flip, m.to_numpy(), 0.0), index=s.index).cumsum()
        counts = exponent.value_counts()
        top = counts[counts == counts.max()].index
        ref = exponent.iloc[-1] if exponent.iloc[-1] in top else top[0]
        s = s * 100.0 ** (ref - exponent)
        for d, ratio in zip(s.index[flip], np.exp(lr.to_numpy()[flip])):
            events.append((ticker, d, "unit_flip", f"day-on-day price ratio {ratio:.4g}x; rescaled to dominant unit"))

    # ---- 2) reverting spikes -------------------------------------------------------
    ls = np.log(s)
    one = ls.diff()
    two = ls.shift(-1) - ls.shift(1)
    bad = ((one.abs() > spike) & (two.abs() < revert)).to_numpy()
    if bad.any():
        for d, mv in zip(s.index[bad], np.expm1(one.to_numpy()[bad])):
            events.append((ticker, d, "spike_removed", f"{mv:+.1%} one-day move reverted next day; set to NaN"))
        s = s.mask(bad)

    # ---- 3) flag-only --------------------------------------------------------------
    v = s.dropna()
    big = np.log(v).diff().abs() > spike
    for d in v.index[big.to_numpy()]:
        r = v.iloc[v.index.get_loc(d)] / v.iloc[v.index.get_loc(d) - 1] - 1
        events.append((ticker, d, "flag_only", f"{r:+.1%} one-day move kept (not reverted); verify corporate action"))

    return s.reindex(orig_index), events


def sanitize_units(
    px: pd.DataFrame, overrides: dict[str, dict] | None = None, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sanitise every column. Returns (clean prices, events DataFrame).

    ``kwargs`` apply to every column; ``overrides`` maps a column name to its own
    ``sanitize_series`` parameters. FX tickers default to ``FX_PARAMS``.
    """
    per_col = {t: FX_PARAMS for t in FX_TICKERS} | (overrides or {})
    out, all_events = px.copy(), []
    for col in px.columns:
        out[col], ev = sanitize_series(px[col], ticker=str(col), **(kwargs | per_col.get(str(col), {})))
        all_events += ev
        if str(col) in FX_TICKERS:
            level = out[col].dropna()
            lo, hi = FX_PLAUSIBLE
            if len(level) and not lo <= float(level.median()) <= hi:
                all_events.append((str(col), level.index[-1], "fx_level",
                                   f"median level {level.median():.4g} outside {lo:g}-{hi:g} rand per "
                                   "dollar; wrong unit or inverted quote? Left unchanged"))
    events = pd.DataFrame(all_events, columns=EVENT_COLS)
    if not events.empty:
        events = events.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out, events


def summarize_events(events: pd.DataFrame) -> str:
    """One-paragraph human summary, empty string if nothing happened."""
    if events.empty:
        return ""
    parts = []
    for kind, label in [("unit_flip", "ZAc/ZAR unit flips fixed"),
                        ("spike_removed", "reverting bad prints removed"),
                        ("flag_only", "large moves flagged (unchanged)"),
                        ("fx_level", "implausible FX levels flagged (unchanged)")]:
        sub = events[events["kind"] == kind]
        if not sub.empty:
            who = ", ".join(f"{t} ({n})" for t, n in sub["ticker"].value_counts().items())
            parts.append(f"{len(sub)} {label}: {who}")
    return "; ".join(parts)
