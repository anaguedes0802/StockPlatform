"""Institutional holders (13F-style) + famous-investor lookup.

Two layers:
  - Per-symbol institutional + mutual fund holders from yfinance (live).
  - A curated "famous investors" map: {fund_name → [recent top holdings]} that's
    bundled with the repo so we can show "Buffett holds X, Ackman holds Y" without
    hitting paid 13F APIs. Refresh by editing `apps/api/app/services/famous_investors.py`.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import redis

from app.config import settings
from app.services.famous_investors import FAMOUS_INVESTORS
from app.services.market_data import _yf_ticker  # type: ignore


_redis: redis.Redis | None = None


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _cache_get(key: str) -> Any | None:
    try:
        v = _cache().get(key)
        return json.loads(v) if v else None
    except Exception:
        return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


def _df_to_records(df: Any) -> list[dict[str, Any]]:
    if df is None or not hasattr(df, "to_dict"):
        return []
    try:
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.tz_localize(None).astype(str)
        return d.where(pd.notna(d), None).to_dict(orient="records")
    except Exception:
        return []


def get_institutional_holders(symbol: str) -> list[dict[str, Any]]:
    key = f"inst_holders:{symbol}"
    if (cached := _cache_get(key)) is not None:
        return cached
    t = _yf_ticker(symbol)
    try:
        out = _df_to_records(t.institutional_holders)
    except Exception:
        out = []
    _cache_set(key, out, ttl=24 * 3600)
    return out


def get_mutualfund_holders(symbol: str) -> list[dict[str, Any]]:
    key = f"mf_holders:{symbol}"
    if (cached := _cache_get(key)) is not None:
        return cached
    t = _yf_ticker(symbol)
    try:
        out = _df_to_records(t.mutualfund_holders)
    except Exception:
        out = []
    _cache_set(key, out, ttl=24 * 3600)
    return out


def get_major_holders(symbol: str) -> dict[str, Any]:
    key = f"maj_holders:{symbol}"
    if (cached := _cache_get(key)) is not None:
        return cached
    t = _yf_ticker(symbol)
    try:
        rec = _df_to_records(t.major_holders)
    except Exception:
        rec = []
    out = {"rows": rec}
    _cache_set(key, out, ttl=24 * 3600)
    return out


def famous_investors_for_symbol(symbol: str) -> list[dict[str, Any]]:
    """Which famous investors currently disclose holding this symbol?

    Prefers LIVE 13F-HR data from SEC EDGAR (via edgar_13f.py). Falls back to
    the bundled snapshot if EDGAR is unreachable.
    """
    sym = symbol.upper()
    try:
        from app.services.edgar_13f import famous_investors_for_symbol_live
        live = famous_investors_for_symbol_live(sym)
        if live:
            return live
    except Exception:
        pass

    # Bundled fallback
    out: list[dict[str, Any]] = []
    for inv in FAMOUS_INVESTORS:
        for h in inv.get("holdings", []):
            if h["symbol"] == sym:
                out.append({
                    "investor": inv["name"],
                    "fund": inv.get("fund"),
                    "filing_period": inv.get("filing_period"),
                    "symbol": sym,
                    "weight_pct": h.get("weight_pct"),
                    "change": h.get("change"),
                    "note": h.get("note"),
                    "source": "bundled",
                })
    return out


def list_famous_investors() -> list[dict[str, Any]]:
    """List investors + their top holdings. Prefers live 13F data."""
    try:
        from app.services.edgar_13f import famous_investors_live
        return famous_investors_live()
    except Exception:
        pass
    return [
        {
            "name": inv["name"],
            "fund": inv.get("fund"),
            "philosophy": inv.get("philosophy"),
            "filing_period": inv.get("filing_period"),
            "top_holdings": inv.get("holdings", [])[:10],
        }
        for inv in FAMOUS_INVESTORS
    ]


def institutional_signal(symbol: str) -> dict[str, Any]:
    """Compact signal for the recommendation engine.

    Looks at:
      - Number of famous investors holding it
      - Top institutional holders' aggregate share %
    """
    famous = famous_investors_for_symbol(symbol)
    inst = get_institutional_holders(symbol)
    top_pct = 0.0
    if inst:
        # yfinance returns a `pctHeld` column; sum top 10
        for row in inst[:10]:
            v = row.get("pctHeld") or row.get("% Out") or 0
            try:
                top_pct += float(v)
            except (TypeError, ValueError):
                pass
        if top_pct > 1.5:  # if expressed as percent
            top_pct = top_pct / 100.0

    score = 0.0
    if famous:
        # +0.1 per famous investor, capped at +0.6
        score = min(0.6, 0.1 * len(famous))
    # institutional concentration is generally a quality signal
    if top_pct > 0.5:
        score += 0.15
    score = min(1.0, score)

    summary_parts: list[str] = []
    if famous:
        names = ", ".join(f["investor"] for f in famous[:4])
        summary_parts.append(f"Held by: {names}")
    if top_pct > 0:
        summary_parts.append(f"Top-10 institutional ownership: {top_pct*100:.1f}%")
    summary = "; ".join(summary_parts) or "No notable institutional holders on record."

    return {
        "score": round(float(score), 3),
        "famous_holders": famous,
        "top_institutional_pct": round(top_pct, 4),
        "n_famous": len(famous),
        "summary": summary,
    }
