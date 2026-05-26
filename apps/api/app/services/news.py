"""News aggregation + sentiment scoring.

v0: yfinance news + Yahoo Finance RSS by symbol.
Pluggable for NewsAPI / Finnhub / SEC EDGAR by adding more `_fetch_*` adapters.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone  # noqa: F401  (re-exported in tests)
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import httpx
import redis

from app.config import settings
from app.services import market_data as md
from app.services import sentiment as sent

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


def _yahoo_rss(symbol: str, limit: int = 25) -> list[dict[str, Any]]:
    """Yahoo Finance per-symbol RSS — free, no key, no rate limit."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}&region=US&lang=en-US"
    try:
        r = httpx.get(url, timeout=8.0, headers={"User-Agent": "StockPlatform/0.1"})
        r.raise_for_status()
    except Exception:
        return []
    items: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(r.text)
        for el in root.iterfind(".//item"):
            title = (el.findtext("title") or "").strip()
            link = (el.findtext("link") or "").strip()
            pub = el.findtext("pubDate") or ""
            summary = (el.findtext("description") or "").strip()
            items.append({"title": title, "url": link, "source": "Yahoo Finance",
                          "published_at": pub, "summary": _strip_html(summary)})
            if len(items) >= limit:
                break
    except ET.ParseError:
        return []
    return items


_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _TAG.sub("", s or "").strip()


def fetch_news(symbol: str, limit: int = 25) -> list[dict[str, Any]]:
    """Combine providers, dedupe by URL, score sentiment, return ordered list."""
    symbol = symbol.upper()
    key = f"news_v2:{symbol}:{limit}"
    if cached := _cache_get(key):
        return cached

    yf_items = md.get_news(symbol, limit=limit) or []
    rss_items = _yahoo_rss(symbol, limit=limit)

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for it in (*yf_items, *rss_items):
        url = it.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        text = " ".join(filter(None, [it.get("title"), it.get("summary")]))
        polarity, confidence = sent.score_text(text)
        merged.append(
            {
                **it,
                "sentiment": round(polarity, 3),
                "sentiment_confidence": round(confidence, 3),
                "impact_score": round(abs(polarity) * confidence, 3),
            }
        )
    # sort newest first if we have parseable dates; else keep order
    def _parse(x: Any) -> float:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(str(x)).timestamp()
            except Exception:
                return 0.0

    merged.sort(key=lambda it: _parse(it.get("published_at")), reverse=True)
    out = merged[:limit]
    _cache_set(key, out, ttl=300)
    return out


def aggregate_sentiment(symbol: str, window_n: int = 25) -> dict[str, float]:
    items = fetch_news(symbol, limit=window_n)
    scored = [(it["sentiment"], it["sentiment_confidence"]) for it in items if it.get("sentiment") is not None]
    return sent.aggregate(scored)


def sentiment_momentum(symbol: str) -> dict[str, Any]:
    """Sentiment momentum — Δ between recent (7d) and broader (30d) windows.

    The change in sentiment is generally a stronger predictor than the level:
    levels get priced in quickly; deltas mark inflection points.
    """
    from datetime import datetime, timedelta, timezone
    items = fetch_news(symbol, limit=50)
    if not items:
        return {"score": 0.0, "summary": "No news to compute momentum."}

    now = datetime.now(timezone.utc)
    bucket_7d: list[tuple[float, float]] = []
    bucket_30d: list[tuple[float, float]] = []
    for it in items:
        ts_str = it.get("published_at")
        if ts_str is None:
            continue
        try:
            if isinstance(ts_str, (int, float)):
                ts = datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
            else:
                # may be e.g. "Sat, 24 May 2025 10:00:00 GMT" or ISO
                from email.utils import parsedate_to_datetime
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                except ValueError:
                    ts = parsedate_to_datetime(str(ts_str))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if it.get("sentiment") is None:
            continue
        s = float(it["sentiment"])
        c = float(it.get("sentiment_confidence") or 0.5)
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days < 0 or age_days > 30:
            continue
        bucket_30d.append((s, c))
        if age_days <= 7:
            bucket_7d.append((s, c))

    if not bucket_30d:
        return {"score": 0.0, "summary": "Not enough dated news."}

    s7 = sent.aggregate(bucket_7d)["weighted"] if bucket_7d else 0.0
    s30 = sent.aggregate(bucket_30d)["weighted"]
    delta = s7 - s30
    # Δ in [-2, +2] possible; squash to [-1, +1]
    import math
    score = math.tanh(delta * 1.5)

    if delta > 0.10:
        summary = f"Sentiment improving: 7d {s7:+.2f} vs 30d {s30:+.2f} (Δ {delta:+.2f})"
    elif delta < -0.10:
        summary = f"Sentiment deteriorating: 7d {s7:+.2f} vs 30d {s30:+.2f} (Δ {delta:+.2f})"
    else:
        summary = f"Sentiment stable: 7d {s7:+.2f} vs 30d {s30:+.2f}"

    return {
        "score": round(float(score), 3),
        "sent_7d": round(float(s7), 3),
        "sent_30d": round(float(s30), 3),
        "delta": round(float(delta), 3),
        "n_7d": len(bucket_7d),
        "n_30d": len(bucket_30d),
        "summary": summary,
    }


def fetch_market_news(limit: int = 30) -> list[dict[str, Any]]:
    """General market news (no symbol) — uses ^GSPC as a proxy via Yahoo RSS."""
    key = f"market_news:{limit}"
    if cached := _cache_get(key):
        return cached
    items = _yahoo_rss("^GSPC", limit=limit)
    for it in items:
        text = " ".join(filter(None, [it.get("title"), it.get("summary")]))
        polarity, confidence = sent.score_text(text)
        it["sentiment"] = round(polarity, 3)
        it["sentiment_confidence"] = round(confidence, 3)
        it["impact_score"] = round(abs(polarity) * confidence, 3)
    _cache_set(key, items, ttl=180)
    return items


def trending_symbols(min_articles: int = 3) -> list[dict[str, Any]]:
    """Compute a tiny 'trending' panel from the seeded universe.
    Picks symbols whose news sentiment dispersion or count is unusual.

    Cached for 10 minutes — the underlying scan is expensive (~3-5s per
    symbol cold). Parallelized across symbols so the wall time stays bounded.
    """
    cache_key = "trending_v2"
    if cached := _cache_get(cache_key):
        return cached

    universe = md.all_universe()[:12]   # smaller scan than the original 20
    out: list[dict[str, Any]] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _one(it):
        try:
            agg = aggregate_sentiment(it["symbol"], window_n=10)
            if agg["n"] >= min_articles:
                return {"symbol": it["symbol"], "name": it.get("name"),
                        "n_articles": int(agg["n"]), "sentiment": round(agg["weighted"], 3)}
        except Exception:
            return None
        return None
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(_one, it) for it in universe]):
            r = f.result()
            if r: out.append(r)

    out.sort(key=lambda x: -abs(x["sentiment"]))
    out = out[:8]
    _cache_set(cache_key, out, ttl=10 * 60)
    return out


def _legacy_trending_symbols(min_articles: int = 3) -> list[dict[str, Any]]:
    """Original serial implementation, kept for reference."""
    universe = md.all_universe()[:20]
    out: list[dict[str, Any]] = []
    for it in universe:
        try:
            agg = aggregate_sentiment(it["symbol"], window_n=15)
            if agg["n"] >= min_articles:
                out.append({
                    "symbol": it["symbol"],
                    "name": it.get("name"),
                    "n_articles": int(agg["n"]),
                    "sentiment": round(agg["weighted"], 3),
                })
        except Exception:
            continue
    out.sort(key=lambda x: -abs(x["sentiment"]))
    return out[:8]
