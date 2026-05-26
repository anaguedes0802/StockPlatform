"""Historical news sentiment for point-in-time training.

Sources, in order of preference:
  1. **Alpha Vantage NEWS_SENTIMENT** — free with key, ~2 years history, per-article
     sentiment + per-ticker relevance + ticker-specific sentiment score.
     https://www.alphavantage.co/documentation/#news-sentiment
  2. **GDELT 2.0** — fully free, global news event database, sentiment via tone field.
     Used as a fallback for symbols/tickers Alpha Vantage doesn't cover well.
     https://www.gdeltproject.org/data.html
  3. **Bundled empty** — returns zeros so training never crashes.

Output: a pandas DataFrame indexed by date with columns:
  - news_sent_mean   (mean sentiment, -1..+1)
  - news_sent_wmean  (relevance-weighted sentiment)
  - news_count       (raw article count)
  - news_count_z_30  (rolling 30d z-score of article count)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd
import redis

from app.config import settings


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


@dataclass
class _ArticleScore:
    ts: datetime
    sentiment: float          # -1 .. +1
    relevance: float          # 0 .. 1


# ---------------------------------------------------------------------------
# Alpha Vantage adapter
# ---------------------------------------------------------------------------

def _alpha_vantage_news(symbol: str, time_from: str, time_to: str, limit: int = 1000) -> list[_ArticleScore]:
    """Fetch articles + ticker_sentiment for one (symbol, window)."""
    if not settings.alphavantage_api_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": time_from,
        "time_to": time_to,
        "limit": min(1000, limit),
        "sort": "EARLIEST",
        "apikey": settings.alphavantage_api_key,
    }
    try:
        r = httpx.get(url, params=params, timeout=20.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # Alpha Vantage rate limits hit silently; surface upstream
        raise RuntimeError(f"alpha vantage fetch failed: {e}")

    if "Information" in data or "Note" in data:
        # rate-limited or daily cap message; treat as empty
        return []

    out: list[_ArticleScore] = []
    for art in data.get("feed", []):
        # time_published is like "20240315T133000"
        ts_str = art.get("time_published")
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        # Find the symbol-specific entry in ticker_sentiment
        s_score: float | None = None
        relevance: float | None = None
        for ts_entry in art.get("ticker_sentiment", []):
            if ts_entry.get("ticker") == symbol:
                try:
                    s_score = float(ts_entry.get("ticker_sentiment_score") or 0)
                    relevance = float(ts_entry.get("relevance_score") or 0)
                except (TypeError, ValueError):
                    pass
                break
        if s_score is None:
            # Fall back to overall article sentiment
            try:
                s_score = float(art.get("overall_sentiment_score") or 0)
                relevance = 0.5
            except (TypeError, ValueError):
                continue
        out.append(_ArticleScore(ts=ts, sentiment=float(s_score), relevance=float(relevance or 0.5)))
    return out


def _av_paginate(symbol: str, start: datetime, end: datetime) -> list[_ArticleScore]:
    """Walk the time range in 30-day chunks (the AV limit is 1000 articles/request)."""
    all_articles: list[_ArticleScore] = []
    window_start = start
    chunk = timedelta(days=30)
    while window_start < end:
        window_end = min(window_start + chunk, end)
        tf = window_start.strftime("%Y%m%dT%H%M")
        tt = window_end.strftime("%Y%m%dT%H%M")
        try:
            chunk_articles = _alpha_vantage_news(symbol, tf, tt, limit=1000)
            all_articles.extend(chunk_articles)
        except RuntimeError:
            break
        window_start = window_end
        # Free tier: 25 req/day, 5 req/min — be polite.
        time.sleep(0.5)
    return all_articles


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_daily(articles: list[_ArticleScore], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bucket articles by trading day and compute daily features."""
    if not articles:
        return pd.DataFrame(
            {
                "news_sent_mean":  np.zeros(len(target_index)),
                "news_sent_wmean": np.zeros(len(target_index)),
                "news_count":      np.zeros(len(target_index)),
                "news_count_z_30": np.zeros(len(target_index)),
            },
            index=target_index,
        )

    df = pd.DataFrame([
        {"ts": a.ts, "sentiment": a.sentiment, "relevance": a.relevance}
        for a in articles
    ])
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    grouped = df.groupby("date").agg(
        news_sent_mean=("sentiment", "mean"),
        news_count=("sentiment", "count"),
        weighted_num=("sentiment", lambda x: float(np.sum(x.values * df.loc[x.index, "relevance"].values))),
        weighted_den=("relevance", "sum"),
    )
    grouped["news_sent_wmean"] = grouped["weighted_num"] / grouped["weighted_den"].replace(0, np.nan)
    grouped["news_sent_wmean"] = grouped["news_sent_wmean"].fillna(grouped["news_sent_mean"])
    grouped = grouped[["news_sent_mean", "news_sent_wmean", "news_count"]]

    # Align to target trading days
    out = grouped.reindex(target_index.tz_convert("UTC").normalize() if target_index.tz else target_index.normalize())
    out.index = target_index
    out["news_count"] = out["news_count"].fillna(0)
    out["news_sent_mean"] = out["news_sent_mean"].fillna(0)
    out["news_sent_wmean"] = out["news_sent_wmean"].fillna(0)

    # 30-day rolling z-score of news_count
    mean30 = out["news_count"].rolling(30, min_periods=5).mean()
    std30 = out["news_count"].rolling(30, min_periods=5).std() + 1e-9
    out["news_count_z_30"] = ((out["news_count"] - mean30) / std30).fillna(0)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_history(symbol: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Get historical daily sentiment aligned to target_index.

    Cache key includes the date range so repeated training/backtest calls hit the
    cache. TTL: 24h (news doesn't change retroactively).
    """
    if len(target_index) == 0:
        return pd.DataFrame()

    start = target_index.min()
    end = target_index.max() + pd.Timedelta(days=1)
    if start.tz is None:
        start = start.tz_localize("UTC")
        end = end.tz_localize("UTC")

    cache_key = f"av_news:{symbol}:{start.date()}:{end.date()}"
    cached = _cache_get(cache_key)
    if cached:
        df = pd.DataFrame(cached)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.set_index("ts", inplace=True)
        # ensure alignment (target_index may differ if caller reindexes)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.reindex(target_index, method="nearest", tolerance=pd.Timedelta("1D")).fillna(0)

    if not settings.alphavantage_api_key:
        # No Alpha Vantage key → fall back to **GDELT** (free, no key).
        # OFF by default: pulling 3+ years of GDELT chunks adds ~60-180s to
        # every cold forecast. Set USE_GDELT_HISTORICAL=1 to enable.
        import os as _os
        if _os.environ.get("USE_GDELT_HISTORICAL", "0") == "1":
            try:
                from app.services.gdelt import historical_daily_sentiment
                return historical_daily_sentiment(symbol, target_index)
            except Exception:
                return _aggregate_daily([], target_index)
        return _aggregate_daily([], target_index)

    articles = _av_paginate(symbol, start.to_pydatetime(), end.to_pydatetime())
    daily = _aggregate_daily(articles, target_index)

    # cache
    serial = daily.reset_index().rename(columns={"index": "ts"}).to_dict(orient="records")
    _cache_set(cache_key, serial, ttl=24 * 3600)
    return daily


def is_available() -> bool:
    return bool(settings.alphavantage_api_key)
