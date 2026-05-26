"""Insider (SEC Form 4) transactions and ownership snapshots.

Source: yfinance, which wraps Yahoo Finance's insider data (originally from SEC
EDGAR Form 4 filings — CEO/CFO/director buys & sells of their own company stock).

Free, no key, real data. Cached aggressively because insiders file slowly.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import redis

from app.config import settings
from app.services.market_data import _yf_ticker  # type: ignore  (private helper, intentional reuse)


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
    if df is None:
        return []
    try:
        if not hasattr(df, "to_dict"):
            return []
        d = df.copy()
        # Normalize timestamps to ISO strings
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.tz_localize(None).astype(str)
        return d.where(pd.notna(d), None).to_dict(orient="records")
    except Exception:
        return []


def get_insider_transactions(symbol: str, limit: int = 30) -> list[dict[str, Any]]:
    """Recent Form 4 transactions (purchases, sales, gifts, option exercises).

    Primary source: **SEC EDGAR Form 4 filings** — free, authoritative, complete.
    Fallback: yfinance (sparse for many symbols, kept as last-resort).
    """
    key = f"insider_tx_v2:{symbol}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    # Try EDGAR Form 4 first.
    try:
        from app.services.edgar_form4 import fetch_insider_transactions
        edgar_txs = fetch_insider_transactions(symbol, limit_filings=min(50, limit * 2))
        if edgar_txs:
            # Normalize to the legacy yfinance-ish row shape so existing UI keeps working.
            normalized: list[dict[str, Any]] = []
            for t in edgar_txs[:limit]:
                normalized.append({
                    "Insider":       t.get("insider"),
                    "Position":      "; ".join(t.get("titles") or []) or None,
                    "Transaction":   t.get("code_label"),
                    "Action":        t.get("code"),
                    "Start Date":    t.get("date"),
                    "Filed":         t.get("filed"),
                    "Shares":        abs(t.get("shares") or 0),
                    "Value":         t.get("value"),
                    "Source":        "sec_edgar",
                })
            _cache_set(key, normalized, ttl=4 * 3600)
            return normalized
    except Exception:
        pass
    # Fallback to yfinance.
    t = _yf_ticker(symbol)
    out: list[dict[str, Any]] = []
    try:
        df = t.insider_transactions
        out = _df_to_records(df)[:limit]
        for row in out:
            row["Source"] = "yfinance"
    except Exception:
        out = []
    _cache_set(key, out, ttl=4 * 3600)
    return out


def get_insider_purchases(symbol: str) -> dict[str, Any]:
    """Aggregate net buying / selling totals."""
    key = f"insider_purchases:{symbol}"
    if (cached := _cache_get(key)) is not None:
        return cached
    t = _yf_ticker(symbol)
    try:
        df = t.insider_purchases
        rec = _df_to_records(df)
        out = {"rows": rec}
    except Exception:
        out = {"rows": []}
    _cache_set(key, out, ttl=4 * 3600)
    return out


def get_insider_roster(symbol: str) -> list[dict[str, Any]]:
    """Current ownership snapshot for officers/directors."""
    key = f"insider_roster:{symbol}"
    if (cached := _cache_get(key)) is not None:
        return cached
    t = _yf_ticker(symbol)
    out: list[dict[str, Any]] = []
    try:
        df = t.insider_roster_holders
        out = _df_to_records(df)
    except Exception:
        out = []
    _cache_set(key, out, ttl=12 * 3600)
    return out


def insider_signal(symbol: str) -> dict[str, Any]:
    """Compact summary for the recommendation engine.

    Returns:
      {
        "score":      float in [-1, +1]   (net buy bias)
        "n_buys_90d": int
        "n_sells_90d": int
        "net_value_90d": float            (positive = net insider buying)
        "summary":   str
        "source":    "sec_edgar" | "yfinance"
      }
    """
    # Prefer the EDGAR-backed v2 (real, authoritative).
    try:
        from app.services.edgar_form4 import insider_signal_v2
        sig = insider_signal_v2(symbol)
        if sig.get("source") == "sec_edgar" and sig.get("n_buys_90d", 0) + sig.get("n_sells_90d", 0) > 0:
            return sig
    except Exception:
        pass
    txs = get_insider_transactions(symbol, limit=60)
    if not txs:
        return {"score": 0.0, "n_buys_90d": 0, "n_sells_90d": 0, "net_value_90d": 0.0,
                "summary": "No insider activity available."}
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
    n_buys = n_sells = 0
    net_value = 0.0
    for tx in txs:
        # yfinance returns either "Start Date" or "Date" depending on version
        date_str = tx.get("Start Date") or tx.get("Date") or tx.get("Transaction") or ""
        try:
            ts = pd.Timestamp(date_str)
        except Exception:
            continue
        # Some Yahoo timestamps come back tz-naive; coerce both sides to naive UTC.
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        cutoff_naive = cutoff.tz_convert(None) if cutoff.tzinfo is not None else cutoff
        if ts < cutoff_naive:
            continue
        txn_type = (tx.get("Transaction") or tx.get("Action") or "").lower()
        shares = tx.get("Shares") or 0
        value = tx.get("Value") or 0
        try:
            value = float(value) if value is not None else 0.0
            shares = float(shares) if shares is not None else 0.0
        except (TypeError, ValueError):
            value = shares = 0.0

        if "purchase" in txn_type or "buy" in txn_type:
            n_buys += 1; net_value += value
        elif "sale" in txn_type or "sell" in txn_type or "sold" in txn_type:
            n_sells += 1; net_value -= value

    # Normalize to [-1, +1]. Use a saturating curve so very large dollar values don't dominate.
    total = n_buys + n_sells
    if total == 0:
        score = 0.0
    else:
        # buy ratio vs sells (-1..+1)
        ratio = (n_buys - n_sells) / total
        # tilt by sign of net dollar value with diminishing returns
        import math
        dollar_tilt = math.tanh(net_value / 5_000_000)  # $5M is a meaningful cluster
        score = max(-1.0, min(1.0, 0.6 * ratio + 0.4 * dollar_tilt))

    parts: list[str] = []
    if n_buys: parts.append(f"{n_buys} insider buys (90d)")
    if n_sells: parts.append(f"{n_sells} insider sells (90d)")
    if abs(net_value) > 1_000_000: parts.append(f"net ${net_value/1e6:+.1f}M")
    summary = "; ".join(parts) or "No insider activity in last 90d."

    return {
        "score": round(float(score), 3),
        "n_buys_90d": n_buys,
        "n_sells_90d": n_sells,
        "net_value_90d": round(net_value, 2),
        "summary": summary,
    }
