"""GDELT 2.0 news adapter — free, no key, global news coverage since 2015.

We use the **DOC 2.0 API** for recent news (last 365 days max in a single query)
and the per-article TONE field for sentiment polarity.

  GET https://api.gdeltproject.org/api/v2/doc/doc?query=<q>&format=JSON&mode=ArtList
       &startdatetime=YYYYMMDDHHMMSS&enddatetime=YYYYMMDDHHMMSS&maxrecords=250

Tone in GDELT ranges roughly [-10, +10]; we normalize to [-1, +1].

Notes vs Alpha Vantage NEWS_SENTIMENT:
  - GDELT is free with no daily quota (Alpha Vantage free is 25/day).
  - Tone is computed by GDELT's classifier, not finance-tuned (Alpha Vantage's
    `ticker_sentiment_score` is finance-tuned). Quality trade-off.
  - GDELT doesn't give a per-ticker relevance score; we filter by the symbol's
    company name appearing in the title/snippet.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd
import redis

from app.config import settings
from app.core.logging import log
from app.services.market_data import get_profile


_UA = "StockPlatform/0.1 contact@example.com"
_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

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


def _query_for_symbol(symbol: str) -> str:
    """Build a GDELT query string. Prefer the company name; fall back to ticker."""
    try:
        name = (get_profile(symbol).get("name") or "").strip()
    except Exception:
        name = ""
    if name and len(name) >= 4:
        # Quote it so GDELT treats it as a phrase. Strip Inc./Corp suffixes.
        for suffix in (" Inc.", " Inc", " Corporation", " Corp.", " Corp",
                       " Company", " Co.", " plc", " PLC", " AG", " SA", " SE"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return f'"{name}"'
    return symbol


def fetch_articles(symbol: str, days: int = 30, max_records: int = 100) -> list[dict[str, Any]]:
    """Fetch news articles via GDELT DOC 2.0. Returns a list of dicts:
       {ts, title, url, source, tone_raw, sentiment, language, country}
    Tone is normalized to [-1, +1].
    """
    sym = symbol.upper()
    key = f"gdelt:{sym}:{days}:{max_records}"
    if (cached := _cache_get(key)) is not None:
        return cached

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "query": _query_for_symbol(sym),
        "mode": "ArtList",
        "format": "JSON",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
        "maxrecords":    min(250, max_records),
        "sort": "DateDesc",
    }
    try:
        r = httpx.get(_DOC_URL, params=params,
                      headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=20.0)
        r.raise_for_status()
        body = r.text
        # GDELT sometimes returns HTML on rate-limit; only parse if JSON-ish
        if not body.strip().startswith("{"):
            log.warning("gdelt_non_json", symbol=sym, snippet=body[:200])
            _cache_set(key, [], ttl=600)
            return []
        data = json.loads(body)
    except Exception as e:
        log.warning("gdelt_fetch_failed", symbol=sym, err=str(e))
        return []

    articles = data.get("articles") or []
    out: list[dict[str, Any]] = []
    for a in articles:
        try:
            tone_raw = float(a.get("tone") or 0.0)
        except (TypeError, ValueError):
            tone_raw = 0.0
        sentiment = max(-1.0, min(1.0, tone_raw / 5.0))   # GDELT tone is ~[-10,+10], typically [-5,+5]
        ts = None
        if a.get("seendate"):
            # format: "20250126T143000Z"
            try:
                ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = None
        out.append({
            "ts":   ts.isoformat() if ts else None,
            "title": a.get("title") or "",
            "url":   a.get("url") or "",
            "source": a.get("domain") or "",
            "tone_raw": tone_raw,
            "sentiment": round(sentiment, 3),
            "language": a.get("language") or "",
            "country":  a.get("sourcecountry") or "",
        })
    _cache_set(key, out, ttl=15 * 60)   # 15-min cache for recent news
    return out


def historical_daily_sentiment(symbol: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily-aggregated sentiment + article count from GDELT, aligned to a
    target trading-day index. Used by features.py for PIT-historical join.

    The DOC API allows up to ~1 year of history per query; we batch in 30-day
    chunks to stay safely under any document-count cap.
    """
    if len(target_index) == 0:
        return pd.DataFrame()

    start = target_index.min()
    end = target_index.max() + pd.Timedelta(days=1)
    if start.tz is None:
        start = start.tz_localize("UTC")
        end = end.tz_localize("UTC")

    cache_key = f"gdelt_daily:{symbol}:{start.date()}:{end.date()}"
    if (cached := _cache_get(cache_key)) is not None:
        df = pd.DataFrame(cached)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
        return _align(df, target_index)

    # Batch in 30-day chunks
    all_articles: list[dict[str, Any]] = []
    cur = start
    chunk = pd.Timedelta(days=30)
    while cur < end:
        seg_end = min(cur + chunk, end)
        params = {
            "query": _query_for_symbol(symbol),
            "mode": "ArtList",
            "format": "JSON",
            "startdatetime": cur.strftime("%Y%m%d%H%M%S"),
            "enddatetime":   seg_end.strftime("%Y%m%d%H%M%S"),
            "maxrecords": 250,
            "sort": "DateAsc",
        }
        try:
            r = httpx.get(_DOC_URL, params=params,
                          headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=30.0)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                for a in (r.json().get("articles") or []):
                    try:
                        tone = float(a.get("tone") or 0.0)
                    except (TypeError, ValueError):
                        tone = 0.0
                    sd = a.get("seendate")
                    if not sd: continue
                    try:
                        ts = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    all_articles.append({"ts": ts, "sentiment": tone / 5.0})
        except Exception as e:
            log.warning("gdelt_chunk_failed", symbol=symbol, start=str(cur), err=str(e))
        cur = seg_end

    if not all_articles:
        empty = pd.DataFrame({"news_sent_mean": 0.0, "news_count": 0.0, "news_count_z_30": 0.0},
                             index=target_index)
        empty["news_sent_wmean"] = 0.0
        _cache_set(cache_key, [], ttl=24 * 3600)
        return empty

    df = pd.DataFrame(all_articles)
    df["date"] = df["ts"].dt.normalize()
    agg = df.groupby("date").agg(
        news_sent_mean=("sentiment", "mean"),
        news_count=("sentiment", "count"),
    )
    agg["news_sent_wmean"] = agg["news_sent_mean"]   # GDELT has no per-article relevance
    mean30 = agg["news_count"].rolling(30, min_periods=5).mean()
    std30 = agg["news_count"].rolling(30, min_periods=5).std() + 1e-9
    agg["news_count_z_30"] = ((agg["news_count"] - mean30) / std30).fillna(0)

    serial = agg.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
    _cache_set(cache_key, serial, ttl=24 * 3600)
    return _align(agg, target_index)


def _align(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "news_sent_mean": np.zeros(len(target_index)),
            "news_sent_wmean": np.zeros(len(target_index)),
            "news_count": np.zeros(len(target_index)),
            "news_count_z_30": np.zeros(len(target_index)),
        }, index=target_index)
    out = df.reindex(target_index.tz_convert("UTC").normalize() if target_index.tz else target_index.normalize())
    out.index = target_index
    out["news_count"] = out["news_count"].fillna(0)
    out["news_sent_mean"] = out["news_sent_mean"].fillna(0)
    out["news_sent_wmean"] = out["news_sent_wmean"].fillna(0)
    out["news_count_z_30"] = out.get("news_count_z_30", pd.Series(0, index=out.index)).fillna(0)
    return out
