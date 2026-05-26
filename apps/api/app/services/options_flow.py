"""Options unusual activity (yfinance option_chain).

Heuristic detection of unusual flow without an order-tape (which is paid):
  - per-contract volume / open_interest > 3.0  (the classic UW filter)
  - aggregate call/put dollar volume bias (sweep-like accumulation)
  - put/call ratio vs historical norm

This is not a substitute for real options flow data (Unusual Whales,
FlowAlgo, Polygon options) — but it catches the most obvious unusual
activity for free.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import redis

from app.config import settings
from app.core.logging import log
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


def _fetch_chain(symbol: str, expiration: str | None = None) -> dict[str, Any] | None:
    """Pull a single expiration's chain. yfinance returns calls + puts as dataframes."""
    try:
        t = _yf_ticker(symbol)
        expirations = list(t.options) if hasattr(t, "options") else []
        if not expirations:
            return None
        exp = expiration or expirations[0]
        chain = t.option_chain(exp)
        if chain is None:
            return None
        return {
            "expiration": exp,
            "calls": chain.calls,
            "puts": chain.puts,
        }
    except Exception as e:
        log.warning("yfinance_options_failed", symbol=symbol, err=str(e))
        return None


def unusual_activity(symbol: str, max_expirations: int = 3) -> dict[str, Any]:
    """Scan up to N near-term expirations for unusual volume/OI ratios."""
    sym = symbol.upper()
    key = f"opt_unusual:{sym}:{max_expirations}"
    if (cached := _cache_get(key)) is not None:
        return cached

    t = _yf_ticker(sym)
    try:
        expirations = list(t.options)
    except Exception:
        expirations = []
    if not expirations:
        out = {"score": 0.0, "summary": "No options chain available.",
               "unusual": [], "call_put_ratio": None}
        _cache_set(key, out, ttl=600)
        return out

    rows: list[dict[str, Any]] = []
    call_dollars = 0.0
    put_dollars = 0.0
    for exp in expirations[:max_expirations]:
        chain = _fetch_chain(sym, exp)
        if not chain:
            continue
        for side, df in (("call", chain["calls"]), ("put", chain["puts"])):
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                try:
                    vol = float(row.get("volume") or 0)
                    oi = float(row.get("openInterest") or 0)
                    last = float(row.get("lastPrice") or 0)
                    strike = float(row.get("strike") or 0)
                except (TypeError, ValueError):
                    continue
                dollars = vol * last * 100  # contracts × premium × 100 shares
                if side == "call":
                    call_dollars += dollars
                else:
                    put_dollars += dollars
                if vol >= 100 and oi > 0 and (vol / oi) >= 3.0:
                    rows.append({
                        "side": side,
                        "expiration": exp,
                        "strike": strike,
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "vol_oi_ratio": round(vol / oi, 2),
                        "last_price": last,
                        "dollar_volume": round(dollars, 0),
                        "implied_volatility": float(row.get("impliedVolatility") or 0),
                    })

    # Sort by absolute dollar volume — biggest flow first.
    rows.sort(key=lambda r: -r["dollar_volume"])

    cp_ratio = (call_dollars / put_dollars) if put_dollars > 0.01 else None
    # Score: sum of signed dollar imbalance, squashed
    import math
    if (call_dollars + put_dollars) > 0:
        bias = (call_dollars - put_dollars) / (call_dollars + put_dollars)
        # Boost if there's a notable cluster of unusual rows
        cluster_boost = math.tanh(len(rows) / 5.0)
        score = float(max(-1.0, min(1.0, bias * (0.5 + 0.5 * cluster_boost))))
    else:
        score = 0.0

    summary_parts = []
    if cp_ratio is not None:
        if cp_ratio > 1.5:
            summary_parts.append(f"Call-heavy ({cp_ratio:.1f}× put $-volume)")
        elif cp_ratio < 0.67:
            summary_parts.append(f"Put-heavy ({1/cp_ratio:.1f}× call $-volume)")
        else:
            summary_parts.append(f"Balanced C/P ratio {cp_ratio:.2f}")
    if rows:
        summary_parts.append(f"{len(rows)} unusual contracts (vol/OI > 3)")
        big = rows[0]
        summary_parts.append(
            f"largest: {big['side']} ${big['strike']:.0f} {big['expiration']} "
            f"${big['dollar_volume']/1000:.0f}K notional"
        )

    out = {
        "score": round(score, 3),
        "call_put_ratio": round(cp_ratio, 2) if cp_ratio is not None else None,
        "call_dollar_volume": round(call_dollars, 0),
        "put_dollar_volume": round(put_dollars, 0),
        "unusual": rows[:8],
        "n_unusual": len(rows),
        "summary": "; ".join(summary_parts) if summary_parts else "Limited options activity.",
    }
    _cache_set(key, out, ttl=600)  # 10-minute cache (options vol moves fast)
    return out
