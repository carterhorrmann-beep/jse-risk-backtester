"""JSE universe, sector map and defaults.

Tickers use Yahoo Finance's ``.JO`` suffix. The list is a *current-constituent*
snapshot of large/liquid JSE names, so historical backtests on it carry
survivorship bias (see README). Verify tickers before relying on results:
corporate actions and renames (e.g. Anglo American Platinum -> Valterra
Platinum) can break history on free data feeds.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_BENCHMARK = "STX40.JO"  # Satrix 40 ETF, a tradable proxy for the Top 40

# USD/ZAR spot: rand per US dollar (~12-20, quoted in rands, NOT cents like the .JO shares).
# A *rise* means the rand weakens. It is a risk factor, never a tradable name in the universe.
USDZAR_TICKER = "USDZAR=X"
FX_TICKERS: tuple[str, ...] = (USDZAR_TICKER,)

SECTORS: dict[str, str] = {
    # Resources
    "AGL.JO": "Resources", "BHG.JO": "Resources", "GFI.JO": "Resources",
    "ANG.JO": "Resources", "SSW.JO": "Resources", "IMP.JO": "Resources",
    "HAR.JO": "Resources", "EXX.JO": "Resources", "GLN.JO": "Resources",
    "S32.JO": "Resources", "NPH.JO": "Resources", "VAL.JO": "Resources",
    "SOL.JO": "Resources",
    # Financials
    "SBK.JO": "Financials", "FSR.JO": "Financials", "ABG.JO": "Financials",
    "NED.JO": "Financials", "CPI.JO": "Financials", "SLM.JO": "Financials",
    "OMU.JO": "Financials", "DSY.JO": "Financials",
    # Industrials & consumer
    "SHP.JO": "Industrials", "WHL.JO": "Industrials", "CLS.JO": "Industrials",
    "BID.JO": "Industrials", "BVT.JO": "Industrials", "APN.JO": "Industrials",
    "TBS.JO": "Industrials", "MNP.JO": "Industrials", "PPH.JO": "Industrials",
    "CFR.JO": "Industrials", "BTI.JO": "Industrials", "REM.JO": "Industrials",
    # Technology & telecoms
    "NPN.JO": "Tech & Telecom", "PRX.JO": "Tech & Telecom",
    "MTN.JO": "Tech & Telecom", "VOD.JO": "Tech & Telecom",
    # Property
    "GRT.JO": "Property",
}

JSE_TOP_TICKERS: list[str] = list(SECTORS)


def sector_weights(weights: pd.Series) -> pd.Series:
    """Aggregate portfolio weights by sector (unknown tickers -> 'Other')."""
    sectors = weights.index.map(lambda t: SECTORS.get(t, "Other"))
    return weights.groupby(sectors).sum().sort_values(ascending=False)
