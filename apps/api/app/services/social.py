"""Social sentiment — Reddit + StockTwits.

Sources:
  - **StockTwits public API** — free, no auth. Per-symbol "stream" with up to
    30 most-recent messages including bullish/bearish self-labels.
  - **Reddit (PRAW)** — credential-gated. Free with app registration:
    https://www.reddit.com/prefs/apps
    Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET + REDDIT_USER_AGENT in env.

Both feed into the same composite `social_signal` for use in opinion + screener.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import redis

from app.config import settings
from app.core.logging import log
from app.services import sentiment as sent


_UA_HTTP = "StockPlatform/0.1 (StockTwits adapter)"

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


# -----------------------------------------------------------------------------
# StockTwits
# -----------------------------------------------------------------------------

_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def fetch_stocktwits(symbol: str, limit: int = 30) -> list[dict[str, Any]]:
    """Returns up to 30 recent StockTwits messages for a symbol."""
    key = f"st:{symbol}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    try:
        r = httpx.get(_STOCKTWITS_URL.format(symbol=symbol.upper()),
                      headers={"User-Agent": _UA_HTTP, "Accept": "application/json"},
                      timeout=10.0)
        if r.status_code != 200:
            log.info("stocktwits_non_200", symbol=symbol, status=r.status_code)
            _cache_set(key, [], ttl=300)
            return []
        data = r.json()
    except Exception as e:
        log.warning("stocktwits_fetch_failed", symbol=symbol, err=str(e))
        return []

    out: list[dict[str, Any]] = []
    for m in (data.get("messages") or [])[:limit]:
        body = (m.get("body") or "")[:500]
        # StockTwits' self-labelled sentiment (when the poster marked it)
        st_sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if st_sent == "Bullish":
            label_score = 0.7
        elif st_sent == "Bearish":
            label_score = -0.7
        else:
            label_score = None
        # Fall back to text sentiment when no self-label
        text_pol, text_conf = sent.score_text(body)
        out.append({
            "id": m.get("id"),
            "ts": m.get("created_at"),
            "username": (m.get("user") or {}).get("username"),
            "followers": (m.get("user") or {}).get("followers"),
            "body": body,
            "label_sentiment": label_score,
            "text_sentiment": round(text_pol, 3),
            "text_confidence": round(text_conf, 3),
        })
    _cache_set(key, out, ttl=180)
    return out


# -----------------------------------------------------------------------------
# Reddit (PRAW)
# -----------------------------------------------------------------------------

# Set these via env to enable; otherwise Reddit results are empty (no crash).
_REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
_REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
_REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "StockPlatform/0.1 by stockplatform-app")

# Subreddits to scan
_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "stockmarket"]


def reddit_available() -> bool:
    return bool(_REDDIT_CLIENT_ID and _REDDIT_CLIENT_SECRET)


def fetch_reddit(symbol: str, hours: int = 48, limit: int = 30) -> list[dict[str, Any]]:
    """Search Reddit for cashtag/ticker mentions in the last N hours."""
    if not reddit_available():
        return []
    key = f"reddit:{symbol}:{hours}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached

    try:
        import praw  # type: ignore
    except Exception:
        log.warning("praw_not_installed")
        return []

    try:
        reddit = praw.Reddit(
            client_id=_REDDIT_CLIENT_ID,
            client_secret=_REDDIT_CLIENT_SECRET,
            user_agent=_REDDIT_USER_AGENT,
        )
        reddit.read_only = True
    except Exception as e:
        log.warning("reddit_init_failed", err=str(e))
        return []

    sym = symbol.upper()
    pattern = re.compile(rf"(?i)(?:\${sym}\b|\b{sym}\b)")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: list[dict[str, Any]] = []
    for sub_name in _SUBREDDITS:
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.search(sym, sort="new", limit=15, time_filter="week"):
                ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                if ts < cutoff:
                    continue
                text = f"{post.title} {post.selftext or ''}"
                if not pattern.search(text):
                    continue
                pol, conf = sent.score_text(text)
                out.append({
                    "id": post.id,
                    "ts": ts.isoformat(),
                    "subreddit": sub_name,
                    "title": post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "url": f"https://reddit.com{post.permalink}",
                    "sentiment": round(pol, 3),
                    "sentiment_confidence": round(conf, 3),
                })
        except Exception as e:
            log.warning("reddit_search_failed", sub=sub_name, err=str(e))
            continue
    out.sort(key=lambda r: r["ts"], reverse=True)
    out = out[:limit]
    _cache_set(key, out, ttl=600)
    return out


# -----------------------------------------------------------------------------
# Composite social signal
# -----------------------------------------------------------------------------

def social_signal(symbol: str) -> dict[str, Any]:
    """Combine StockTwits + Reddit into a hype + sentiment score.

    Returns:
      {
        "score": [-1, +1]    bullish/bearish bias
        "hype_score": [0, 1] attention level (high = stock is being talked about)
        "stocktwits": {n, avg_sentiment, bullish_pct, bearish_pct}
        "reddit":     {n, avg_sentiment, sum_score}
        "summary": str
      }
    """
    st_msgs = fetch_stocktwits(symbol, limit=30)
    rd_posts = fetch_reddit(symbol, hours=48, limit=30)

    # StockTwits aggregation
    st_n = len(st_msgs)
    st_sents = []
    bullish = bearish = 0
    for m in st_msgs:
        if m.get("label_sentiment") is not None:
            st_sents.append(float(m["label_sentiment"]))
            if m["label_sentiment"] > 0: bullish += 1
            else: bearish += 1
        elif m.get("text_confidence", 0) > 0.1:
            st_sents.append(float(m["text_sentiment"]))
    st_avg = (sum(st_sents) / len(st_sents)) if st_sents else 0.0

    # Reddit aggregation — weight by post engagement (upvotes + comments)
    rd_n = len(rd_posts)
    rd_weighted_sum = 0.0
    rd_weight_total = 0.0
    rd_sum_score = 0
    for p in rd_posts:
        w = max(1.0, math.log1p(int(p.get("score") or 0) + int(p.get("num_comments") or 0)))
        rd_weighted_sum += float(p.get("sentiment", 0)) * w
        rd_weight_total += w
        rd_sum_score += int(p.get("score") or 0)
    rd_avg = (rd_weighted_sum / rd_weight_total) if rd_weight_total > 0 else 0.0

    # Composite: weight StockTwits 0.6 (more recent, more numerous) and Reddit 0.4
    if st_n and rd_n:
        combined = 0.6 * st_avg + 0.4 * rd_avg
    elif st_n:
        combined = st_avg
    elif rd_n:
        combined = rd_avg
    else:
        combined = 0.0
    score = float(max(-1.0, min(1.0, combined)))

    # Hype score: log-scaled total mentions vs a baseline of ~10
    total_n = st_n + rd_n
    hype = math.tanh(total_n / 25.0)

    parts = []
    if st_n:
        parts.append(f"StockTwits {st_n} msgs (bullish {bullish}, bearish {bearish}, avg {st_avg:+.2f})")
    if rd_n:
        parts.append(f"Reddit {rd_n} posts (avg {rd_avg:+.2f}, ↑{rd_sum_score})")
    if not parts:
        parts.append("No social activity in window.")

    return {
        "score": round(score, 3),
        "hype_score": round(hype, 3),
        "stocktwits": {"n": st_n, "avg_sentiment": round(st_avg, 3),
                       "bullish_pct": round(bullish / st_n * 100, 1) if st_n else 0,
                       "bearish_pct": round(bearish / st_n * 100, 1) if st_n else 0},
        "reddit": {"n": rd_n, "avg_sentiment": round(rd_avg, 3),
                    "sum_score": rd_sum_score,
                    "available": reddit_available()},
        "summary": "; ".join(parts),
    }
