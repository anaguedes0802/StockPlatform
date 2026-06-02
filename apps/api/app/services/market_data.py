"""Market data adapter. v0 backend: yfinance. Pluggable for Polygon/Finnhub later."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd
import redis
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.logging import log
from app.services.universe_data import all_instruments


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


# ---------- search ----------

# Built from `universe_data.py` — ~200 symbols across stocks/ETFs/indexes/crypto/forex.
# In production, replace with a query against the `instruments` table seeded via a provider.
_UNIVERSE: list[dict[str, str | None]] = [
    {
        "symbol": sym,
        "name": name,
        "exchange": exchange,
        "asset_class": asset_class,
        "sector": sector,
    }
    for sym, name, exchange, asset_class, sector in all_instruments()
]

# Original tiny list (kept commented for reference)
_LEGACY_UNIVERSE: list[dict[str, str]] = [
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "AMD", "name": "Advanced Micro Devices, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "INTC", "name": "Intel Corporation", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "NFLX", "name": "Netflix, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "MA", "name": "Mastercard Inc.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "PG", "name": "Procter & Gamble", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "DIS", "name": "The Walt Disney Company", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "BA", "name": "The Boeing Company", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "PEP", "name": "PepsiCo, Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "CRM", "name": "Salesforce, Inc.", "exchange": "NYSE", "asset_class": "stock"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "exchange": "NASDAQ", "asset_class": "stock"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSEARCA", "asset_class": "etf"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "asset_class": "etf"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSEARCA", "asset_class": "etf"},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSEARCA", "asset_class": "etf"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial ETF", "exchange": "NYSEARCA", "asset_class": "etf"},
    {"symbol": "^GSPC", "name": "S&P 500 Index", "exchange": "INDEX", "asset_class": "index"},
    {"symbol": "^IXIC", "name": "Nasdaq Composite", "exchange": "INDEX", "asset_class": "index"},
    {"symbol": "^DJI", "name": "Dow Jones Industrial Average", "exchange": "INDEX", "asset_class": "index"},
    {"symbol": "^VIX", "name": "CBOE Volatility Index", "exchange": "INDEX", "asset_class": "index"},
    {"symbol": "BTC-USD", "name": "Bitcoin USD", "exchange": "CCC", "asset_class": "crypto"},
    {"symbol": "ETH-USD", "name": "Ethereum USD", "exchange": "CCC", "asset_class": "crypto"},
    {"symbol": "SOL-USD", "name": "Solana USD", "exchange": "CCC", "asset_class": "crypto"},
    {"symbol": "EURUSD=X", "name": "EUR / USD", "exchange": "FX", "asset_class": "forex"},
    {"symbol": "GBPUSD=X", "name": "GBP / USD", "exchange": "FX", "asset_class": "forex"},
    {"symbol": "USDJPY=X", "name": "USD / JPY", "exchange": "FX", "asset_class": "forex"},
    # EU
    {"symbol": "ASML.AS", "name": "ASML Holding", "exchange": "AMS", "asset_class": "stock"},
    {"symbol": "SAP.DE", "name": "SAP SE", "exchange": "XETRA", "asset_class": "stock"},
    {"symbol": "MC.PA", "name": "LVMH Moët Hennessy Louis Vuitton", "exchange": "PAR", "asset_class": "stock"},
    {"symbol": "NESN.SW", "name": "Nestlé SA", "exchange": "SWX", "asset_class": "stock"},
    {"symbol": "HSBA.L", "name": "HSBC Holdings plc", "exchange": "LSE", "asset_class": "stock"},
]


_YH_ASSET_CLASS = {
    "EQUITY": "stock",
    "ETF": "etf",
    "INDEX": "index",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY": "forex",
    "MUTUALFUND": "mutual_fund",
    "FUTURE": "future",
}


def _yahoo_search(query: str, limit: int = 10) -> list[dict[str, str | None]]:
    """Live ticker resolver. Catches anything outside the seeded 205 (small-caps,
    new IPOs, OTC, foreign listings) via Yahoo's public autocomplete endpoint.

    Cached 1h per query so a typeahead doesn't hammer Yahoo.
    """
    q = (query or "").strip()
    if not q:
        return []
    key = f"yh_search:{q.lower()}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    try:
        r = httpx.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": limit, "newsCount": 0, "enableFuzzyQuery": "false"},
            headers={"User-Agent": "Mozilla/5.0 StockPlatform/0.1"},
            timeout=4.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("yahoo_search_failed", q=q, err=str(e))
        return []

    out: list[dict[str, str | None]] = []
    for it in (data.get("quotes") or [])[:limit]:
        sym = it.get("symbol")
        if not sym:
            continue
        qt = (it.get("quoteType") or "").upper()
        # Yahoo also returns options/futures for ticker prefixes — skip the noise.
        if qt in {"OPTION", "FUTURE"}:
            continue
        out.append({
            "symbol": sym,
            "name": it.get("shortname") or it.get("longname") or sym,
            "exchange": it.get("exchDisp") or it.get("exchange"),
            "asset_class": _YH_ASSET_CLASS.get(qt) or "stock",
            "sector": it.get("sectorDisp") or it.get("sector"),
        })
    _cache_set(key, out, ttl=3600)
    return out


def search_universe(
    query: str,
    limit: int = 20,
    asset_class: str | None = None,
    sector: str | None = None,
) -> list[dict[str, str | None]]:
    q = (query or "").strip().lower()
    pool = _UNIVERSE
    if asset_class:
        pool = [it for it in pool if (it.get("asset_class") or "").lower() == asset_class.lower()]
    if sector:
        pool = [it for it in pool if (it.get("sector") or "").lower() == sector.lower()]

    if not q:
        return pool[:limit]
    scored: list[tuple[int, dict[str, str | None]]] = []
    for it in pool:
        sym = str(it["symbol"]).lower()
        name = (it.get("name") or "").lower()
        score = 0
        if sym == q:
            score = 100
        elif sym.startswith(q):
            score = 80
        elif q in sym:
            score = 60
        elif name.startswith(q):
            score = 50
        elif q in name:
            score = 30
        if score:
            scored.append((score, it))
    scored.sort(key=lambda t: -t[0])
    local = [it for _, it in scored[:limit]]

    # Remote fallback — only when we still have room AND no filter is set that
    # would clash with the remote results (Yahoo doesn't return our sector taxonomy).
    if len(local) < limit and not sector:
        seen = {str(it["symbol"]).upper() for it in local}
        remote = _yahoo_search(q, limit=limit - len(local) + 5)
        for r in remote:
            sym_u = str(r["symbol"]).upper()
            if sym_u in seen:
                continue
            if asset_class and (r.get("asset_class") or "").lower() != asset_class.lower():
                continue
            local.append(r)
            seen.add(sym_u)
            if len(local) >= limit:
                break
    return local


def list_sectors() -> list[str]:
    return sorted({it.get("sector") or "" for it in _UNIVERSE if it.get("sector")})


def list_asset_classes() -> list[str]:
    return sorted({str(it.get("asset_class")) for it in _UNIVERSE if it.get("asset_class")})


# ---------- quote / profile / history ----------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
def _yf_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _backfill_change(symbol: str, q: dict[str, Any]) -> dict[str, Any]:
    """Ensure a quote always has change + change_pct populated.

    When the live trade endpoint succeeds but the bars endpoint 429s (Alpaca's
    rate limit), `previous_close` ends up None — and the watchlist UI shows
    "—" for change. We can backfill it from the cached daily history (which
    we usually already have from the chart / opinion engine).
    """
    if q.get("change") is not None and q.get("change_pct") is not None:
        return q
    if not q.get("price"):
        return q
    try:
        df = get_history(symbol, interval="1d", range_="5d")
        if not df.empty and len(df) >= 2:
            prev = float(df["close"].iloc[-2])
            last = float(q["price"])
            ch = last - prev
            q["previous_close"] = q.get("previous_close") or prev
            q["change"] = ch
            q["change_pct"] = (ch / prev * 100) if prev else None
    except Exception:
        pass
    return q


# Equity ETFs that closely proxy major indices. When yfinance is rate-limited
# and we can't fetch ^GSPC directly, fetch SPY (which Alpaca handles) and
# derive a synthetic index-style quote so the dashboard doesn't show $0.00.
# These are well-correlated (>99% daily) so the index/ETF % move is identical
# even though the dollar level differs.
_INDEX_PROXY = {
    "^GSPC": "SPY",   # S&P 500 → SPDR S&P 500 ETF
    "^IXIC": "QQQ",   # Nasdaq Composite → ~Nasdaq 100 ETF (close enough)
    "^DJI":  "DIA",   # Dow Jones → SPDR Dow Jones ETF
    "^RUT":  "IWM",   # Russell 2000 → iShares Russell 2000 ETF
    # ^VIX has no clean equity proxy — VIXY is OK-ish but decays. Just hold cache.
}

# Static sector map for common US names. yfinance is the only feed that carries
# GICS sector, and it rate-limits aggressively; Alpaca's assets endpoint has no
# sector field. Without this, portfolio sector-exposure (and the risk panel's
# sector check) collapse to "Unknown" for everyone. This covers the liquid
# names a user is most likely to hold so those features work feed-independent.
_SECTOR_FALLBACK: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CSCO": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "MU": "Technology", "PLTR": "Technology", "SMCI": "Technology",
    "ARM": "Technology", "NOW": "Technology", "PANW": "Technology",
    "SNOW": "Technology", "DELL": "Technology", "IBM": "Technology",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BRK-B": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "SCHW": "Financials", "SOFI": "Financials",
    "AFRM": "Financials", "COIN": "Financials", "PYPL": "Financials",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
    "TMO": "Health Care", "ABT": "Health Care", "AMGN": "Health Care",
    "VKTX": "Health Care", "FEMY": "Health Care", "MRNA": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "OXY": "Energy", "MPC": "Energy",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "DE": "Industrials", "RKLB": "Industrials",
    # Crypto / digital assets (treated as their own bucket for exposure purposes)
    "MARA": "Crypto/Miners", "RIOT": "Crypto/Miners", "CLSK": "Crypto/Miners",
    "MSTR": "Crypto/Miners",
}


def _quote_is_fresh(q: dict[str, Any], max_age_sec: int = 300) -> bool:
    """Is this quote's trade recent enough to trust?

    Alpaca's free IEX feed barely sees extended-hours volume, so its
    `/trades/latest` goes stale outside 09:30–16:00 ET — during pre/post-market
    its "latest" trade can be hours old. A stale Alpaca trade is the signal to
    fall through to Yahoo, which aggregates the SIP tape and carries pre/post
    prints. During regular hours liquid names print well within `max_age_sec`,
    so this never triggers a needless Yahoo call.
    """
    ts = q.get("ts")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return True  # unknown format — don't second-guess, treat as fresh
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    return age <= max_age_sec


def _yahoo_chart_quote(symbol: str) -> dict[str, Any] | None:
    """Extended-hours-aware quote from Yahoo's public chart endpoint.

    `includePrePost=true` makes the 1-minute series span pre-market, regular,
    and after-hours, so the last non-null close is the current extended-hours
    price — matching what Yahoo Finance shows. Change is measured against the
    prior regular-session close (`chartPreviousClose`), which is how pre-market
    moves are conventionally quoted.
    """
    try:
        r = httpx.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"includePrePost": "true", "interval": "1m", "range": "1d"},
            headers={"User-Agent": "Mozilla/5.0 StockPlatform/0.1"},
            timeout=4.0,
        )
        r.raise_for_status()
        result = (r.json().get("chart") or {}).get("result") or []
    except Exception as e:
        log.warning("yahoo_chart_quote_failed", symbol=symbol, err=str(e))
        return None
    if not result:
        return None
    res = result[0]
    meta = res.get("meta") or {}
    closes = (((res.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    stamps = res.get("timestamp") or []
    price: float | None = None
    ts_epoch: int | None = None
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] is not None:
            price = float(closes[i])
            ts_epoch = stamps[i] if i < len(stamps) else None
            break
    if price is None:
        price = meta.get("regularMarketPrice")
    if not price:
        return None
    prev_raw = meta.get("chartPreviousClose") or meta.get("previousClose")
    prev = float(prev_raw) if prev_raw else None
    change = (price - prev) if prev else None
    change_pct = (change / prev * 100) if prev else None
    ts = (datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()
          if ts_epoch else datetime.now(timezone.utc).isoformat())
    return {
        "symbol": symbol.upper(),
        "price": float(price),
        "previous_close": prev,
        "change": change,
        "change_pct": change_pct,
        "currency": meta.get("currency"),
        "market_state": meta.get("marketState"),
        "ts": ts,
    }


def get_quote(symbol: str) -> dict[str, Any]:
    key = f"q:{symbol}"
    if cached := _cache_get(key):
        return cached

    # Try Alpaca REST first for US equities — yfinance is being rate-limited
    # aggressively by Yahoo, and Alpaca's free tier handles 200 req/min.
    from app.services import alpaca_bars as alp
    if alp.supports_crypto_symbol(symbol):
        q = alp.get_crypto_latest_quote(symbol)
        if q is not None and q.get("price"):
            q = _backfill_change(symbol, q)
            _cache_set(key, q, ttl=10)
            return q
    if alp.is_configured() and alp.supports_symbol(symbol):
        q = alp.get_latest_quote(symbol)
        if q is not None and q.get("price") and _quote_is_fresh(q):
            q = _backfill_change(symbol, q)
            _cache_set(key, q, ttl=10)
            return q
        # Alpaca quote is MISSING or STALE — the normal state during pre/post-
        # market, when the free IEX feed is quiet (it often returns nothing at
        # all, not just a stale print). Prefer Yahoo's extended-hours price,
        # which carries the pre/after-market print. Only fall back to a stale
        # Alpaca quote if Yahoo also fails (a few-minutes-old price beats $0).
        # NOTE: this must run even when `q is None`, or premarket shows no price.
        yq = _yahoo_chart_quote(symbol)
        if yq is not None and yq.get("price"):
            _cache_set(key, yq, ttl=10)
            return yq
        if q is not None and q.get("price"):
            q = _backfill_change(symbol, q)
            _cache_set(key, q, ttl=10)
            return q

    # Yahoo-only symbols (indices "^", FX "=X", foreign ".XX") — try yfinance,
    # then fall back to the ETF proxy if yfinance is rate-limited / failing.
    t = _yf_ticker(symbol)
    fi = getattr(t, "fast_info", None) or {}
    last = float(fi.get("last_price") or fi.get("lastPrice") or 0.0)

    if last == 0 and symbol in _INDEX_PROXY:
        proxy = _INDEX_PROXY[symbol]
        proxy_q = get_quote(proxy)
        if proxy_q.get("price"):
            # Synthesize an index-like quote — same % change as the proxy,
            # but we display the proxy's dollar level (so headline is reasonable).
            out = {
                "symbol": symbol,
                "price": proxy_q["price"],
                "previous_close": proxy_q.get("previous_close"),
                "change": proxy_q.get("change"),
                "change_pct": proxy_q.get("change_pct"),
                "currency": "USD",
                "market_state": None,
                "ts": proxy_q.get("ts"),
                "via_proxy": proxy,   # so the UI can label it as a proxy if it wants
            }
            _cache_set(key, out, ttl=60)   # longer cache — indices change slower
            return out
    prev = float(fi.get("previous_close") or fi.get("previousClose") or 0.0)
    change = last - prev if prev else None
    change_pct = (change / prev * 100) if prev else None
    out = {
        "symbol": symbol.upper(),
        "price": last,
        "previous_close": prev or None,
        "change": change,
        "change_pct": change_pct,
        "currency": fi.get("currency"),
        "market_state": fi.get("market_state") or fi.get("marketState"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _cache_set(key, out, ttl=10)
    return out


def get_profile(symbol: str) -> dict[str, Any]:
    key = f"p:{symbol}"
    if cached := _cache_get(key):
        return cached
    t = _yf_ticker(symbol)
    info: dict[str, Any] = {}
    try:
        info = t.info or {}  # may make a network call
    except Exception:
        info = {}
    out = {
        "symbol": symbol.upper(),
        "name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap"),
        "description": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
    }

    # If yfinance gave us nothing (rate-limited / non-Yahoo symbol), fall back
    # to Alpaca's assets endpoint for at least name + exchange + currency.
    # Also fall back to the seeded universe (cheap, no network) as a last resort.
    if not out.get("name"):
        from app.services import alpaca_bars as alp
        if alp.is_configured() and alp.supports_symbol(symbol):
            alp_p = alp.get_profile(symbol)
            if alp_p:
                for k, v in alp_p.items():
                    if v and not out.get(k):
                        out[k] = v
    if not out.get("name"):
        # Check seeded universe (covers ETFs, indexes, crypto, EU)
        for it in _UNIVERSE:
            if it["symbol"].upper() == symbol.upper():
                out["name"] = out.get("name") or it.get("name")
                out["exchange"] = out.get("exchange") or it.get("exchange")
                out["sector"] = out.get("sector") or it.get("sector")
                break
    if not out.get("name"):
        out["name"] = symbol.upper()   # never null — symbol itself is a sane fallback

    # Sector is the field yfinance most often fails to deliver under rate-limit.
    # Fill from the static map for common names so sector-exposure / risk work.
    if not out.get("sector"):
        out["sector"] = _SECTOR_FALLBACK.get(symbol.upper())

    # Don't cache a half-empty profile for 12h — if yfinance was rate-limited
    # (no name from it, only the symbol/Alpaca fallback), keep the TTL short so
    # we re-attempt the richer data soon instead of serving stubs all day.
    ttl = 12 * 3600 if info.get("longName") else 30 * 60
    _cache_set(key, out, ttl=ttl)
    return out


def get_key_stats(symbol: str) -> dict[str, Any]:
    key = f"ks:{symbol}"
    if cached := _cache_get(key):
        return cached
    t = _yf_ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    out = {
        "pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        # Fundamentals for the quality/value stock-picker (all best-effort; yfinance
        # leaves them None for some names). Margins/growth are fractions (0.21 = 21%).
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "debt_to_equity": info.get("debtToEquity"),
        "return_on_equity": info.get("returnOnEquity"),
        "market_cap": info.get("marketCap"),
    }
    _cache_set(key, out, ttl=3600)
    return out


_INTERVAL_TO_YF = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}


def get_history(symbol: str, interval: str = "1d", range_: str = "1y") -> pd.DataFrame:
    yf_interval = _INTERVAL_TO_YF.get(interval, "1d")
    key = f"h:{symbol}:{yf_interval}:{range_}"
    if cached := _cache_get(key):
        df = pd.DataFrame(cached)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df.set_index("ts", inplace=True)
        return df

    df = pd.DataFrame()

    # Primary source: Alpaca (when configured + symbol is a US equity).
    # Yahoo aggressively rate-limits yfinance; Alpaca's 200 req/min is comfortable.
    from app.services import alpaca_bars as alp
    if alp.supports_crypto_symbol(symbol):
        try:
            df = alp.get_crypto_bars(symbol, interval=interval, range_=range_)
        except Exception as e:
            log.warning("alpaca_crypto_bars_failed", symbol=symbol, err=str(e))
            df = pd.DataFrame()
    if df.empty and alp.is_configured() and alp.supports_symbol(symbol):
        try:
            df = alp.get_bars(symbol, interval=interval, range_=range_)
        except Exception as e:
            log.warning("alpaca_bars_failed", symbol=symbol, err=str(e))
            df = pd.DataFrame()

    # Pre / post-market supplement.
    # The free Alpaca IEX feed sees almost no extended-hours volume (most
    # ext-hours trading runs on ECNs IEX doesn't see). Yahoo Finance's bars
    # aggregate from the consolidated SIP tape and include those prints —
    # which is the difference between an empty chart at 22h Lisbon and a
    # populated one. So for INTRADAY timeframes on US equities, we ALSO
    # query yfinance with prepost=True and merge any extra bars on top.
    if (not df.empty
            and alp.supports_symbol(symbol)
            and ("m" in yf_interval or yf_interval == "60m")):
        try:
            t = _yf_ticker(symbol)
            yf_df = t.history(period=range_, interval=yf_interval,
                              auto_adjust=True, prepost=True)
            if not yf_df.empty:
                yf_df = yf_df.rename(columns={"Open": "open", "High": "high",
                                              "Low": "low", "Close": "close",
                                              "Volume": "volume"})
                yf_df = yf_df[["open", "high", "low", "close", "volume"]].copy()
                yf_df.index = pd.to_datetime(yf_df.index, utc=True)
                yf_df.index.name = "ts"
                # Concat: yfinance bars NOT already in Alpaca (timestamp-based)
                missing = yf_df.loc[~yf_df.index.isin(df.index)]
                if not missing.empty:
                    df = pd.concat([df, missing]).sort_index()
                    log.info("yf_extended_hours_supplement",
                             symbol=symbol, added=len(missing))
        except Exception as e:
            log.debug("yf_extended_supplement_skipped", symbol=symbol, err=str(e))

    # Fallback: yfinance — for non-US tickers (indexes, crypto, foreign listings)
    # or when Alpaca returns nothing for a US symbol.
    if df.empty:
        t = _yf_ticker(symbol)
        try:
            df = t.history(
                period=range_, interval=yf_interval,
                auto_adjust=True, prepost=True,  # include pre/post-market bars
            )
        except Exception as e:
            log.warning("yfinance_history_failed", symbol=symbol, interval=yf_interval, range_=range_, err=str(e))
            return pd.DataFrame()
        if df.empty:
            log.info("yfinance_history_empty", symbol=symbol, interval=yf_interval, range_=range_)
            return df
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"

    # cache short for intraday; long for daily
    ttl = 60 if "m" in yf_interval or yf_interval == "60m" else 3600
    serial = df.reset_index().to_dict(orient="records")
    _cache_set(key, serial, ttl=ttl)
    return df


def get_history_bars(symbol: str, interval: str = "1d", range_: str = "1y") -> list[dict[str, Any]]:
    df = get_history(symbol, interval, range_)
    if df.empty:
        return []
    return [
        {
            "ts": ts.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": int(row.volume) if pd.notna(row.volume) else 0,
        }
        for ts, row in df.iterrows()
    ]


def get_news(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    key = f"n:{symbol}:{limit}"
    if cached := _cache_get(key):
        return cached
    t = _yf_ticker(symbol)
    items = []
    try:
        raw = t.news or []
    except Exception:
        raw = []
    for item in raw[:limit]:
        content = item.get("content") or item
        items.append(
            {
                "title": content.get("title"),
                "url": (content.get("canonicalUrl") or {}).get("url")
                       or content.get("clickThroughUrl", {}).get("url")
                       or content.get("link"),
                "source": (content.get("provider") or {}).get("displayName") or item.get("publisher"),
                "published_at": content.get("pubDate") or item.get("providerPublishTime"),
                "summary": content.get("summary"),
                "thumbnail": ((content.get("thumbnail") or {}).get("resolutions") or [{}])[0].get("url"),
            }
        )
    _cache_set(key, items, ttl=300)
    return items


@lru_cache(maxsize=1)
def all_universe() -> list[dict[str, str]]:
    return list(_UNIVERSE)
