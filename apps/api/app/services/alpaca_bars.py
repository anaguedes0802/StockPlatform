"""Alpaca historical bars adapter.

Alpaca's data REST API exposes daily, minute, and intraday OHLCV bars for US
equities. Free / paper accounts get the IEX feed (real exchange but only IEX's
share of total volume). The rate limits are way more generous than Yahoo's
hammer (~200 req/min vs Yahoo's increasingly-strict throttling).

We use this as the primary source for historical bars whenever Alpaca creds
are configured, and fall back to yfinance for non-US-equity symbols (crypto,
indexes with `^`, FX with `=X`, foreign listings with a `.`).

Docs: https://docs.alpaca.markets/reference/stockbars
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import time

import httpx
import pandas as pd

from app.config import settings
from app.core.logging import log


_BASE_URL = "https://data.alpaca.markets/v2"


def is_configured() -> bool:
    return bool(settings.alpaca_api_key and settings.alpaca_api_secret)


# Alpaca-supported US equity. We skip non-US tickers (Yahoo suffixes like
# `.L`, `.PA`), indexes (`^GSPC`), crypto (`-USD`), and FX (`=X`).
def supports_symbol(symbol: str) -> bool:
    s = symbol.upper()
    if "." in s or "^" in s or "=" in s or "-" in s:
        return False
    return s.isalnum() and 1 <= len(s) <= 6


_RANGE_TO_DAYS = {
    "1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 365, "2y": 730, "3y": 1095, "4y": 1460, "5y": 1825,
    "7y": 2555, "10y": 3650, "ytd": None, "max": 3650 * 4,
}

# Map our interval keys → Alpaca's "timeframe" param.
_INTERVAL_TO_ALPACA = {
    "1m":  "1Min",
    "5m":  "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h":  "1Hour",
    "60m": "1Hour",
    "1d":  "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
}


def _range_to_start(range_: str) -> datetime:
    days = _RANGE_TO_DAYS.get(range_, 365)
    if range_ == "ytd":
        return datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc)
    if days is None:
        days = 365
    return datetime.now(timezone.utc) - timedelta(days=days)


def get_bars(
    symbol: str,
    interval: str = "1d",
    range_: str = "1y",
    feed: str | None = None,
    *,
    extended_hours: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca. Returns a UTC-indexed DataFrame with
    columns [open, high, low, close, volume], or an empty DataFrame on failure.

    `extended_hours=True` includes pre-market + after-hours bars (4am–8pm ET).
    """
    if not is_configured():
        return pd.DataFrame()
    if not supports_symbol(symbol):
        return pd.DataFrame()

    tf = _INTERVAL_TO_ALPACA.get(interval) or _INTERVAL_TO_ALPACA["1d"]
    start = _range_to_start(range_)
    # IEX free tier; pass "sip" only if user has subscribed.
    feed = feed or settings.alpaca_feed

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "accept": "application/json",
    }
    url = f"{_BASE_URL}/stocks/{symbol.upper()}/bars"
    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    pages = 0
    try:
        with httpx.Client(timeout=15.0) as client:
            while True:
                params: dict[str, Any] = {
                    "timeframe": tf,
                    "start": start.isoformat(),
                    "limit": 10000,
                    "adjustment": "all",
                    "feed": feed,
                    # extended-hours flag isn't on this endpoint; bars include
                    # all trades on the exchange, including ext-hours, when
                    # adjustment=all and timeframe is intraday.
                }
                if next_token:
                    params["page_token"] = next_token
                r = client.get(url, headers=headers, params=params)
                if r.status_code == 401:
                    log.error("alpaca_bars_unauthorized", symbol=symbol)
                    return pd.DataFrame()
                if r.status_code == 422:
                    # bad symbol / unsupported feed for plan
                    log.warning("alpaca_bars_422", symbol=symbol, body=r.text[:200])
                    return pd.DataFrame()
                r.raise_for_status()
                data = r.json()
                for b in data.get("bars") or []:
                    rows.append({
                        "ts": b["t"],
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low":  float(b["l"]),
                        "close": float(b["c"]),
                        "volume": int(b.get("v") or 0),
                    })
                next_token = data.get("next_page_token")
                pages += 1
                if not next_token or pages >= 10:  # safety cap
                    break
    except Exception as e:
        log.warning("alpaca_bars_failed", symbol=symbol, err=str(e))
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def get_bars_multi(symbols: list[str], range_: str = "2y", chunk: int = 100) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for many symbols via the multi-symbol endpoint
    (/v2/stocks/bars?symbols=A,B,...): ~100 symbols per request instead of one
    request each, which is what keeps a 900-name universe inside the free
    200 requests/minute quota. Split/dividend adjusted. Backs off on 429."""
    if not is_configured():
        return {}
    start = _range_to_start(range_)
    headers = {"APCA-API-KEY-ID": settings.alpaca_api_key, "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
               "accept": "application/json"}
    rows: dict[str, list[dict[str, Any]]] = {}
    syms = [s.upper() for s in symbols if supports_symbol(s)]
    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(syms), chunk):
            part = syms[i:i + chunk]
            token: str | None = None
            while True:
                params: dict[str, Any] = {"symbols": ",".join(part), "timeframe": "1Day",
                                          "start": start.isoformat(), "limit": 10000,
                                          "adjustment": "all", "feed": settings.alpaca_feed}
                if token:
                    params["page_token"] = token
                for attempt in range(5):
                    r = client.get(f"{_BASE_URL}/stocks/bars", headers=headers, params=params)
                    if r.status_code != 429:
                        break
                    time.sleep(3 * (attempt + 1))
                if r.status_code != 200:
                    log.warning("alpaca_bars_multi_failed", status=r.status_code, body=r.text[:200])
                    break
                data = r.json()
                for sym, bars in (data.get("bars") or {}).items():
                    rows.setdefault(sym, []).extend(
                        {"ts": b["t"], "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
                         "close": float(b["c"]), "volume": int(b.get("v") or 0)} for b in bars)
                token = data.get("next_page_token")
                if not token:
                    break
    out = {}
    for sym, rs in rows.items():
        df = pd.DataFrame(rs)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        out[sym] = df.set_index("ts").sort_index()
    return out


def get_profile(symbol: str) -> dict[str, Any] | None:
    """Look up company name / exchange via Alpaca's assets endpoint.

    Alpaca only returns name + exchange + tradability info — no sector /
    industry / market cap / description. That's a known limitation; we still
    use this as a fallback so symbol pages don't show "null" everywhere when
    Yahoo is rate-limiting yfinance.
    """
    if not is_configured() or not supports_symbol(symbol):
        return None
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            # Trading API is on a different host than the data API.
            r = client.get(
                f"https://paper-api.alpaca.markets/v2/assets/{symbol.upper()}",
                headers=headers,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            a = r.json()
    except Exception as e:
        log.warning("alpaca_profile_failed", symbol=symbol, err=str(e))
        return None
    return {
        "symbol": symbol.upper(),
        "name": a.get("name"),
        "exchange": a.get("exchange"),
        "sector": None,
        "industry": None,
        "country": "United States",
        "currency": "USD",
        "market_cap": None,
        "description": None,
        "website": None,
        "employees": None,
    }


# ---------- Crypto bars (free on Alpaca) ----------

_CRYPTO_BASE = "https://data.alpaca.markets/v1beta3/crypto/us"


def supports_crypto_symbol(symbol: str) -> bool:
    """Recognize yfinance-style crypto symbols (BTC-USD, ETH-USD, ...) so we
    can re-route them to Alpaca's free crypto endpoint."""
    s = symbol.upper()
    return s.endswith("-USD") and "/" not in s and s.replace("-", "").isalnum()


def _crypto_alpaca_pair(symbol: str) -> str:
    # BTC-USD → BTC/USD
    return symbol.upper().replace("-", "/")


def get_crypto_bars(symbol: str, interval: str = "1d", range_: str = "1y") -> pd.DataFrame:
    """Alpaca crypto bars — no auth required for the v1beta3/crypto endpoint
    but we send creds anyway since we have them."""
    if not supports_crypto_symbol(symbol):
        return pd.DataFrame()
    tf = _INTERVAL_TO_ALPACA.get(interval) or "1Day"
    start = _range_to_start(range_)
    pair = _crypto_alpaca_pair(symbol)
    headers = {}
    if settings.alpaca_api_key and settings.alpaca_api_secret:
        headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        }
    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    pages = 0
    try:
        with httpx.Client(timeout=15.0) as client:
            while True:
                params: dict[str, Any] = {
                    "symbols": pair,
                    "timeframe": tf,
                    "start": start.isoformat(),
                    "limit": 10000,
                }
                if next_token:
                    params["page_token"] = next_token
                r = client.get(f"{_CRYPTO_BASE}/bars", headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
                for b in (data.get("bars") or {}).get(pair, []):
                    rows.append({
                        "ts": b["t"],
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low":  float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b.get("v") or 0),
                    })
                next_token = data.get("next_page_token")
                pages += 1
                if not next_token or pages >= 10:
                    break
    except Exception as e:
        log.warning("alpaca_crypto_failed", symbol=symbol, err=str(e))
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def get_crypto_latest_quote(symbol: str) -> dict[str, Any] | None:
    if not supports_crypto_symbol(symbol):
        return None
    pair = _crypto_alpaca_pair(symbol)
    headers = {}
    if settings.alpaca_api_key and settings.alpaca_api_secret:
        headers = {"APCA-API-KEY-ID": settings.alpaca_api_key,
                   "APCA-API-SECRET-KEY": settings.alpaca_api_secret}
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{_CRYPTO_BASE}/latest/trades", headers=headers, params={"symbols": pair})
            r.raise_for_status()
            t = (r.json().get("trades") or {}).get(pair) or {}
            # Prev close via daily bars
            yesterday = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
            br = client.get(f"{_CRYPTO_BASE}/bars", headers=headers,
                            params={"symbols": pair, "timeframe": "1Day", "start": yesterday, "limit": 3})
            br.raise_for_status()
            bars = (br.json().get("bars") or {}).get(pair, [])
            prev_close = float(bars[-2]["c"]) if len(bars) >= 2 else None
    except Exception as e:
        log.warning("alpaca_crypto_quote_failed", symbol=symbol, err=str(e))
        return None
    price = float(t.get("p") or 0.0)
    if price <= 0:
        return None
    change = (price - prev_close) if prev_close else None
    change_pct = (change / prev_close * 100) if prev_close else None
    return {
        "symbol": symbol.upper(),
        "price": price,
        "previous_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": "USD",
        "market_state": None,
        "ts": t.get("t"),
    }


def get_latest_quote(symbol: str) -> dict[str, Any] | None:
    """Latest trade for a symbol (REST, not WS). Used by /stocks/{symbol} as a
    yfinance-rate-limit fallback. Returns None if Alpaca can't satisfy it."""
    if not is_configured() or not supports_symbol(symbol):
        return None
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            # Last trade
            tr = client.get(
                f"{_BASE_URL}/stocks/{symbol.upper()}/trades/latest",
                headers=headers, params={"feed": settings.alpaca_feed},
            )
            tr.raise_for_status()
            t = (tr.json().get("trade") or {})
            # Previous close (one daily bar ago)
            yesterday = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            br = client.get(
                f"{_BASE_URL}/stocks/{symbol.upper()}/bars",
                headers=headers,
                params={"timeframe": "1Day", "start": yesterday, "limit": 7,
                        "adjustment": "all", "feed": settings.alpaca_feed},
            )
            br.raise_for_status()
            bars = br.json().get("bars") or []
            prev_close = float(bars[-2]["c"]) if len(bars) >= 2 else None
    except Exception as e:
        log.warning("alpaca_quote_failed", symbol=symbol, err=str(e))
        return None
    price = float(t.get("p") or 0.0)
    if price <= 0:
        return None
    change = (price - prev_close) if prev_close else None
    change_pct = (change / prev_close * 100) if prev_close else None
    return {
        "symbol": symbol.upper(),
        "price": price,
        "previous_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": "USD",
        "market_state": None,
        "ts": t.get("t"),
    }
