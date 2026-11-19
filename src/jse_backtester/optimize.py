"""Portfolio optimisation on the shrunk covariance matrix: Minimum Variance and Risk Parity.

Every optimiser takes a covariance matrix, and the matrix always comes from
``risk.shrunk_cov`` (sample covariance shrunk 10% toward a constant-correlation target), the same
estimator behind ``risk_contributions`` and the risk report, so weights, Euler risk shares and
diversification ratios agree across the whole codebase.

* ``min_variance``: ``min w' S w`` s.t. ``sum(w) = 1``, ``0 <= w_i <= max_weight`` (default 20%).
* ``risk_parity``: long-only weights whose Euler risk shares ``w_i (S w)_i / (w' S w)`` all equal
  ``1/N`` (Equal Risk Contribution), ``sum(w) = 1``, ``w_i > 0``.

Both are solved with ``scipy.optimize.minimize(method="SLSQP")``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk import (DEFAULT_RF, annualised_vol, diversification_ratio, max_drawdown,
                   risk_contributions, sharpe, shrunk_cov, usdzar_betas)
from .universe import FX_TICKERS

DEFAULT_MAX_WEIGHT = 0.20
DEFAULT_SHRINKAGE = 0.10
MIN_OBS = 60  # fewest common daily returns we will estimate a covariance from

METHOD_LABELS = {"equal": "Equal Weight", "min_var": "Min Variance", "risk_parity": "Risk Parity"}
_ALIASES = {"equal_weight": "equal", "minimum_variance": "min_var", "erc": "risk_parity"}
_EPS = 1e-9


# ---------------------------------------------------------------- helpers
def _cov_array(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("covariance matrix must be square")
    if not np.isfinite(S).all():
        raise ValueError("covariance matrix contains NaN/inf (need a complete return history)")
    return (S + S.T) / 2.0


def _index(cov) -> pd.Index:
    return cov.index if isinstance(cov, pd.DataFrame) else pd.RangeIndex(np.asarray(cov).shape[0])


def check_max_weight(n: int, max_weight: float) -> None:
    """Raise a readable error when ``n`` names cannot sum to 100% under the per-name cap."""
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    if n * max_weight < 1.0 - 1e-12:
        need = math.ceil(1.0 / max_weight - 1e-12)
        raise ValueError(f"a {max_weight:.0%} cap needs at least {need} holdings to reach 100% "
                         f"(got {n}): add holdings or raise --max-weight")


def _finish(w: np.ndarray, index, cap: float = 1.0) -> pd.Series:
    """Clip solver noise into [0, cap] and renormalise to exactly 100%."""
    w = np.clip(w, 0.0, cap)
    total = w.sum()
    if not total > 0:
        raise RuntimeError("optimiser returned an all-zero weight vector")
    return pd.Series(w / total, index=index, name="weight")


# ---------------------------------------------------------------- optimisers
def equal_weight(cov: pd.DataFrame | np.ndarray) -> pd.Series:
    """1/N in every name."""
    n = _cov_array(cov).shape[0]
    return pd.Series(np.full(n, 1.0 / n), index=_index(cov), name="weight")


def min_variance(cov: pd.DataFrame | np.ndarray, max_weight: float = DEFAULT_MAX_WEIGHT) -> pd.Series:
    """Long-only minimum-variance weights with a per-name cap.

    ``min w' S w`` s.t. ``sum(w) = 1`` and ``0 <= w_i <= max_weight``. Convex, so SLSQP finds the
    global optimum. Raises ``ValueError`` if ``n * max_weight < 1`` (the cap makes 100% unreachable).
    """
    S = _cov_array(cov)
    n = S.shape[0]
    check_max_weight(n, max_weight)
    cap = min(max_weight, 1.0)
    if n == 1 or abs(n * cap - 1.0) < 1e-12:  # only one feasible point: equal weights
        return equal_weight(cov)
    Sn = S / float(np.mean(np.diag(S)))  # unit-scale objective keeps SLSQP's tolerances meaningful
    res = minimize(
        lambda w: w @ Sn @ w,
        x0=np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * Sn @ w,
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    w = res.x
    feasible = abs(w.sum() - 1.0) < 1e-6 and (w > -1e-8).all() and (w < cap + 1e-8).all()
    if not feasible or not np.isfinite(res.fun):
        raise RuntimeError(f"min-variance optimisation failed: {res.message}")
    return _finish(w, _index(cov), cap)


def risk_parity(cov: pd.DataFrame | np.ndarray) -> pd.Series:
    """Equal Risk Contribution weights: every name's Euler risk share equals ``1/N``.

    The ERC condition ``w_i (S w)_i = (w' S w) / N`` is scale-invariant in ``w``, so it is solved in
    its strictly convex form (Spinu 2013)::

        min_y  0.5 y' S y  -  (1/N) * sum(log y_i)        y > 0

    whose first-order condition ``y_i (S y)_i = 1/N`` is exactly the ERC condition. SLSQP finds the
    unique minimiser ``y`` and the weights are ``w = y / sum(y)``, which are long-only (``w_i > 0``),
    sum to one, and have Euler risk shares ``RC_i = w_i (S w)_i / (w' S w) = 1/N``.
    """
    S = _cov_array(cov)
    n = S.shape[0]
    if n == 1:
        return equal_weight(cov)
    Sn = S / float(np.mean(np.diag(S)))
    inv_vol = 1.0 / np.sqrt(np.diag(Sn))
    w0 = inv_vol / inv_vol.sum()
    y0 = w0 / math.sqrt(w0 @ Sn @ w0)  # start with y' S y = 1, where the optimum has it
    floor = 1e-12
    res = minimize(
        lambda y: 0.5 * y @ Sn @ y - np.log(y).sum() / n,
        x0=y0,
        jac=lambda y: Sn @ y - 1.0 / (n * y),
        method="SLSQP",
        bounds=[(floor, None)] * n,
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    y = res.x
    w = y / y.sum()
    rc = w * (Sn @ w) / (w @ Sn @ w)
    if not np.isfinite(rc).all() or np.abs(rc - 1.0 / n).max() > 1e-5:
        raise RuntimeError(f"risk-parity optimisation failed to equalise risk shares: {res.message}")
    return _finish(w, _index(cov))


def optimise_weights(
    cov: pd.DataFrame | np.ndarray, method: str, max_weight: float = DEFAULT_MAX_WEIGHT
) -> pd.Series:
    """Dispatch on ``method``: ``equal`` | ``min_var`` | ``risk_parity`` (``max_weight`` applies to min_var)."""
    method = _ALIASES.get(method, method)
    if method == "equal":
        return equal_weight(cov)
    if method == "min_var":
        return min_variance(cov, max_weight)
    if method == "risk_parity":
        return risk_parity(cov)
    raise ValueError(f"unknown method {method!r}; choose from {sorted(METHOD_LABELS)}")


# ---------------------------------------------------------------- portfolio CSV
def normalise_ticker(t: str) -> str:
    """Upper-case and add the JSE ``.JO`` suffix when the symbol has no exchange suffix."""
    t = str(t).strip().upper()
    return t if ("." in t or "=" in t) else f"{t}.JO"


def read_portfolio_csv(source) -> pd.Series:
    """Read a ``ticker,weight`` CSV (path or file-like) into a weight Series (sums to 100%).

    Columns are matched by name (``ticker``/``weight``, any case) and otherwise taken as the first two
    columns, like ``jse-bt risk``. Duplicate tickers are summed; weights must be non-negative.
    """
    if hasattr(source, "seek"):  # an uploaded file read earlier (or by another widget) sits at EOF
        source.seek(0)
    try:
        df = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"could not read the portfolio CSV: {exc}") from exc
    if df.shape[1] < 2:
        raise ValueError("portfolio CSV needs two columns: ticker,weight")
    cols = {str(c).strip().lower(): c for c in df.columns}
    tc, wc = cols.get("ticker", df.columns[0]), cols.get("weight", df.columns[1])
    df = df[[tc, wc]].dropna()
    df.columns = ["ticker", "weight"]
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        raise ValueError("every weight in the portfolio CSV must be a number")
    if (df["weight"] < 0).any():
        raise ValueError("weights must be non-negative (the optimiser is long-only)")
    df["ticker"] = df["ticker"].map(normalise_ticker)
    w = df.groupby("ticker", sort=False)["weight"].sum()
    if not w.sum() > 0:
        raise ValueError("portfolio weights sum to zero")
    return (w / w.sum()).rename("weight")


# ---------------------------------------------------------------- side-by-side comparison
@dataclass
class OptimisationResult:
    weights: pd.DataFrame      # tickers x methods: capital weights (each column sums to 1)
    risk_shares: pd.DataFrame  # tickers x methods: Euler share of portfolio volatility (sums to 1)
    metrics: pd.DataFrame      # metric x methods (in-sample)
    cov: pd.DataFrame          # the shrunk covariance everything was computed from
    sample: tuple[pd.Timestamp, pd.Timestamp]
    n_obs: int
    max_weight: float
    labels: dict[str, str]

    @property
    def risk_parity_gap(self) -> float:
        """Largest |risk share - 1/N| of the Risk Parity column (should be ~0)."""
        return float((self.risk_shares["risk_parity"] - 1.0 / len(self.risk_shares)).abs().max())


def compare_portfolios(
    returns: pd.DataFrame,
    tickers: list[str] | None = None,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    shrinkage: float = DEFAULT_SHRINKAGE,
    rf: float = DEFAULT_RF,
    fx_returns: pd.Series | None = None,
    current: pd.Series | None = None,
) -> OptimisationResult:
    """Equal Weight vs Min Variance vs Risk Parity (plus ``current`` holdings, if given).

    Everything is estimated and measured on the common sample of the chosen names (in-sample: it
    shows what the optimiser targets, not what it would have earned; see ``backtest --weighting``
    for the out-of-sample, no-look-ahead version). ``metrics`` holds realised annualised volatility,
    Sharpe and max drawdown of the static-weight portfolio, the diversification ratio from the shrunk
    covariance, and (with ``fx_returns``) the look-through USD/ZAR beta ``sum_i w_i * beta_i``.
    """
    wanted = list(dict.fromkeys(tickers if tickers is not None else
                                (current.index if current is not None else returns.columns)))
    cols = [t for t in wanted if t in returns.columns and t not in FX_TICKERS]
    if len(cols) < 2:
        raise ValueError("need at least two holdings with price data to optimise")
    check_max_weight(len(cols), max_weight)
    rets = returns[cols].dropna()
    if len(rets) < MIN_OBS:
        raise ValueError(f"only {len(rets)} common daily observations across the holdings; need at least {MIN_OBS}")

    cov = shrunk_cov(rets, shrinkage)
    weights = {}
    if current is not None:
        cur = current.reindex(cols).fillna(0.0)
        if not cur.sum() > 0:
            raise ValueError("none of the current holdings have price data")
        weights["current"] = cur / cur.sum()
        labels = {"current": "Current", **METHOD_LABELS}
    else:
        labels = dict(METHOD_LABELS)
    for m in METHOD_LABELS:
        weights[m] = optimise_weights(cov, m, max_weight)
    W = pd.DataFrame(weights).loc[cols]

    betas = None
    if fx_returns is not None:
        b = usdzar_betas(rets, fx_returns)["beta"].reindex(cols)
        betas = b if b.notna().any() else None  # no estimable beta (too little history): report n/a

    shares, metrics = {}, {}
    for m in W:
        w = W[m]
        port = rets @ w
        shares[m] = risk_contributions(w, cov)["pct"]
        metrics[m] = {
            "Ann. volatility": annualised_vol(port),
            "Sharpe": sharpe(port, rf),
            "Max drawdown": max_drawdown(port),
            "Diversification ratio": diversification_ratio(w, cov),
            "Look-through USD/ZAR Beta": float((w * betas.fillna(0.0)).sum()) if betas is not None else float("nan"),
        }
    return OptimisationResult(
        weights=W, risk_shares=pd.DataFrame(shares).loc[cols], metrics=pd.DataFrame(metrics), cov=cov,
        sample=(rets.index[0], rets.index[-1]), n_obs=len(rets), max_weight=max_weight, labels=labels,
    )
