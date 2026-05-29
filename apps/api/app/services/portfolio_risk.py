"""Portfolio-level risk aggregation — the view a PM actually manages from.

Per-position analysis (which portfolio_review already does) is necessary but
not sufficient. A book of 8 "great" mega-cap tech names is ONE bet, not eight.
This module computes the aggregate exposures a professional watches:

  - **Concentration**: largest single position, top-3 weight, sector weights
  - **Beta**: portfolio-weighted beta vs SPY (systematic exposure)
  - **Correlation clusters**: groups of holdings whose daily returns are
    >0.7 correlated over 90d — these are effectively one position
  - **Total capital at risk**: if every position hit a 2× ATR stop at once,
    what % of the book is lost (the "everything goes wrong at once" number)
  - **Cash / diversification**: number of effective independent bets

Returns structured warnings a pro would flag in a risk meeting.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import log
from app.services import market_data as md


def analyze_risk(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """positions: [{symbol, quantity, avg_buy_price, market_value, ...}]

    Returns aggregate risk metrics + a list of typed warnings.
    """
    holdings = [p for p in positions if (p.get("market_value") or 0) > 0]
    if not holdings:
        return {"ok": False, "reason": "no positions"}

    total_mv = sum(float(p["market_value"]) for p in holdings)
    warnings: list[dict[str, str]] = []

    # ---- Concentration ----
    weights = {p["symbol"]: float(p["market_value"]) / total_mv for p in holdings}
    sorted_w = sorted(weights.items(), key=lambda kv: -kv[1])
    largest_sym, largest_w = sorted_w[0]
    top3_w = sum(w for _, w in sorted_w[:3])

    if largest_w > 0.35:
        warnings.append({"severity": "critical", "kind": "concentration",
                         "message": f"{largest_sym} is {largest_w*100:.0f}% of the book — single-name risk is extreme"})
    elif largest_w > 0.20:
        warnings.append({"severity": "warning", "kind": "concentration",
                         "message": f"{largest_sym} is {largest_w*100:.0f}% of the book — consider trimming"})
    if top3_w > 0.60 and len(holdings) > 3:
        warnings.append({"severity": "warning", "kind": "concentration",
                         "message": f"Top 3 holdings are {top3_w*100:.0f}% of the book — thin diversification"})

    # ---- Sector exposure ----
    sector_w: dict[str, float] = {}
    betas: list[tuple[float, float]] = []   # (weight, beta)
    closes: dict[str, pd.Series] = {}
    atr_stops: list[tuple[float, float]] = []  # (weight, stop_loss_pct)

    for p in holdings:
        sym = p["symbol"]
        w = weights[sym]
        try:
            profile = md.get_profile(sym)
            sector = profile.get("sector") or "Unknown"
            sector_w[sector] = sector_w.get(sector, 0.0) + w
        except Exception:
            sector_w["Unknown"] = sector_w.get("Unknown", 0.0) + w
        try:
            stats = md.get_key_stats(sym)
            b = stats.get("beta")
            if b is not None:
                betas.append((w, float(b)))
        except Exception:
            pass
        # daily closes for correlation + ATR stop distance
        try:
            df = md.get_history(sym, interval="1d", range_="6mo")
            if not df.empty and len(df) > 30:
                closes[sym] = df["close"]
                atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
                last = float(df["close"].iloc[-1])
                stop_pct = (2 * atr / last) if last else 0.05    # 2× ATR stop
                atr_stops.append((w, min(stop_pct, 0.5)))
        except Exception:
            pass

    # Only warn on sectors we actually identified — don't cry "concentrated"
    # just because the sector feed (yfinance) was unavailable and everything
    # bucketed into "Unknown". That would be a false alarm.
    known_sectors = {k: v for k, v in sector_w.items() if k != "Unknown"}
    top_sector = max(known_sectors.items(), key=lambda kv: kv[1]) if known_sectors else (None, 0)
    if top_sector[1] > 0.50:
        warnings.append({"severity": "warning", "kind": "sector",
                         "message": f"{top_sector[0]} is {top_sector[1]*100:.0f}% of the book — sector-concentrated"})

    # ---- Portfolio beta ----
    port_beta = None
    if betas:
        wsum = sum(w for w, _ in betas)
        port_beta = sum(w * b for w, b in betas) / wsum if wsum else None
        if port_beta is not None and port_beta > 1.5:
            warnings.append({"severity": "warning", "kind": "beta",
                             "message": f"Portfolio beta {port_beta:.2f} — ~{(port_beta-1)*100:.0f}% more volatile than the market"})

    # ---- Total capital at risk (all stops hit simultaneously) ----
    total_risk_pct = sum(w * sp for w, sp in atr_stops) * 100 if atr_stops else None
    if total_risk_pct is not None and total_risk_pct > 12:
        warnings.append({"severity": "warning", "kind": "stop_risk",
                         "message": f"If all 2×ATR stops hit at once: -{total_risk_pct:.1f}% of capital — high aggregate risk"})

    # ---- Correlation clusters ----
    clusters: list[list[str]] = []
    if len(closes) >= 2:
        syms = list(closes.keys())
        # align on common dates
        ret = pd.DataFrame({s: closes[s].pct_change() for s in syms}).dropna()
        if len(ret) > 30:
            corr = ret.corr()
            seen: set[str] = set()
            for i, a in enumerate(syms):
                if a in seen:
                    continue
                grp = [a]
                for b in syms[i + 1:]:
                    try:
                        if abs(corr.loc[a, b]) >= 0.7:
                            grp.append(b)
                            seen.add(b)
                    except Exception:
                        pass
                if len(grp) >= 2:
                    clusters.append(grp)
                    seen.add(a)
            for grp in clusters:
                gw = sum(weights.get(s, 0) for s in grp) * 100
                warnings.append({"severity": "warning", "kind": "correlation",
                                 "message": f"{', '.join(grp)} move together (>0.7 corr) = {gw:.0f}% of book in effectively one bet"})

    # Effective number of independent bets (rough): N holdings minus correlated overlap
    n_effective = len(holdings) - sum(len(g) - 1 for g in clusters)

    if not warnings:
        warnings.append({"severity": "info", "kind": "ok",
                         "message": "No major concentration, beta, or correlation flags. Reasonably balanced book."})

    return {
        "ok": True,
        "n_holdings": len(holdings),
        "n_effective_bets": max(1, n_effective),
        "total_market_value": round(total_mv, 2),
        "largest_position": {"symbol": largest_sym, "weight_pct": round(largest_w * 100, 1)},
        "top3_weight_pct": round(top3_w * 100, 1),
        "sector_weights": {k: round(v * 100, 1) for k, v in sorted(sector_w.items(), key=lambda kv: -kv[1])},
        "portfolio_beta": round(port_beta, 2) if port_beta is not None else None,
        "total_capital_at_risk_pct": round(total_risk_pct, 1) if total_risk_pct is not None else None,
        "correlation_clusters": clusters,
        "warnings": warnings,
    }
