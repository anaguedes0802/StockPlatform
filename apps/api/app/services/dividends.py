"""Dividend tracking for portfolio P&L.

For each (symbol, transactions[]), walk historical dividends from yfinance,
multiply per-share dividend by shares held *on the ex-date*, and sum.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import redis

from app.config import settings
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


def dividend_history(symbol: str) -> pd.Series:
    """yfinance Ticker.dividends — indexed by ex-date, value is per-share $."""
    key = f"div:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            s = pd.Series(cached["values"], index=pd.to_datetime(cached["dates"], utc=True))
            return s
        except Exception:
            pass
    try:
        t = _yf_ticker(symbol)
        s: pd.Series = t.dividends
        if s is None or s.empty:
            _cache_set(key, {"dates": [], "values": []}, ttl=24 * 3600)
            return pd.Series(dtype=float)
        # Normalize tz to UTC
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        _cache_set(key, {"dates": [d.isoformat() for d in s.index], "values": s.tolist()}, ttl=24 * 3600)
        return s
    except Exception:
        return pd.Series(dtype=float)


def dividends_received(
    symbol: str,
    transactions: list[Any],
) -> float:
    """Sum dividends received on this symbol given the transaction history.

    For each dividend ex-date, computes net shares held = sum(buys before ex-date)
    − sum(sells before ex-date). Dividend amount = per_share × shares_held.
    """
    divs = dividend_history(symbol)
    if divs.empty:
        return 0.0
    total = 0.0
    # Pre-build a sorted list of transaction (ts, signed_qty)
    sorted_tx = sorted(
        ((t.occurred_at, Decimal(t.quantity) if t.side == "buy" else -Decimal(t.quantity)) for t in transactions),
        key=lambda r: r[0],
    )
    for ex_date, per_share in divs.items():
        # naive ts comparison — both should be UTC
        ex = ex_date if isinstance(ex_date, datetime) else pd.Timestamp(ex_date).to_pydatetime()
        if ex.tzinfo is None:
            ex = ex.replace(tzinfo=timezone.utc)
        shares = Decimal(0)
        for tx_ts, sq in sorted_tx:
            tts = tx_ts if isinstance(tx_ts, datetime) else pd.Timestamp(tx_ts).to_pydatetime()
            if tts.tzinfo is None:
                tts = tts.replace(tzinfo=timezone.utc)
            if tts <= ex:
                shares += sq
            else:
                break
        if shares > 0:
            total += float(shares) * float(per_share)
    return round(total, 4)
