"""Sector relative strength.

About 60-70% of single-stock variance is sector-driven. The IDIOSYNCRATIC
return (symbol vs sector ETF) is where the alpha lives. We compute:

  rs_1m = symbol_return_1m − sector_etf_return_1m
  rs_3m = symbol_return_3m − sector_etf_return_3m

and a composite `relative_strength_score` ∈ [-1, +1] for use in the opinion.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.services import market_data as md


# Sector → sector SPDR ETF used as the benchmark
_SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def sector_etf_for(symbol: str) -> str | None:
    """Look up the sector ETF for a symbol via yfinance.profile."""
    try:
        sector = md.get_profile(symbol).get("sector")
    except Exception:
        return None
    if not sector:
        return None
    return _SECTOR_TO_ETF.get(sector)


def _period_return(df: pd.DataFrame, period_bars: int) -> float | None:
    if df.empty or len(df) < period_bars + 1:
        return None
    try:
        return float(df["close"].iloc[-1] / df["close"].iloc[-(period_bars + 1)] - 1)
    except Exception:
        return None


def sector_relative_strength(symbol: str) -> dict[str, Any]:
    """Compute symbol returns minus sector-ETF returns over 1M and 3M windows."""
    etf = sector_etf_for(symbol)
    if etf is None:
        return {"score": 0.0, "summary": "No sector mapping available.",
                "sector_etf": None}

    try:
        df_sym = md.get_history(symbol, interval="1d", range_="6mo")
        df_etf = md.get_history(etf, interval="1d", range_="6mo")
    except Exception:
        return {"score": 0.0, "summary": "History fetch failed.", "sector_etf": etf}

    ret_sym_1m = _period_return(df_sym, 21)
    ret_etf_1m = _period_return(df_etf, 21)
    ret_sym_3m = _period_return(df_sym, 63)
    ret_etf_3m = _period_return(df_etf, 63)

    rs_1m = (ret_sym_1m - ret_etf_1m) if (ret_sym_1m is not None and ret_etf_1m is not None) else None
    rs_3m = (ret_sym_3m - ret_etf_3m) if (ret_sym_3m is not None and ret_etf_3m is not None) else None

    # Composite — weight 3M more heavily (the durable signal)
    components = []
    if rs_1m is not None: components.append(("1m", rs_1m, 0.4))
    if rs_3m is not None: components.append(("3m", rs_3m, 0.6))
    if not components:
        return {"score": 0.0, "summary": f"Not enough history vs {etf}.",
                "sector_etf": etf}
    weighted = sum(v * w for _, v, w in components) / sum(w for _, _, w in components)
    # ±15% relative move saturates the score
    score = math.tanh(weighted * 6.7)

    direction = "outperforming" if weighted > 0.01 else ("underperforming" if weighted < -0.01 else "matching")
    summary_parts = [f"vs {etf}:"]
    if rs_1m is not None:
        summary_parts.append(f"1M {rs_1m*100:+.1f}%")
    if rs_3m is not None:
        summary_parts.append(f"3M {rs_3m*100:+.1f}%")
    summary_parts.append(f"({direction})")

    return {
        "score": round(float(score), 3),
        "rs_1m_pct": round((rs_1m or 0) * 100, 2),
        "rs_3m_pct": round((rs_3m or 0) * 100, 2),
        "sector_etf": etf,
        "summary": " ".join(summary_parts),
    }
