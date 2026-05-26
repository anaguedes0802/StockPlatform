"""Earnings call transcripts — Motley Fool free archive scraper.

Motley Fool publishes lightly-edited earnings call transcripts for major
US-listed companies. They're free, public, and indexed by ticker:
  https://www.fool.com/quote/<exchange>/<ticker>/earnings-call-transcripts/

Limitations:
  - Coverage: most S&P 500 large-caps; spotty for small caps and non-US.
  - Latency: typically published 24-72h after the call.
  - Format: HTML; we extract the body text only, not the structured Q&A.
  - Throttling: aggressive cache (24h) so we don't hammer Fool's CDN.

We don't ship a full transcript-summarization pipeline here — we expose the
metadata (date, URL) and a sentiment + finBERT-scored body excerpt that can
feed back into the opinion engine when an earnings event is recent.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import redis

from app.config import settings
from app.core.logging import log
from app.services import sentiment as sent


_UA = "Mozilla/5.0 (compatible; StockPlatform/0.1) - research/non-commercial"

# We have to guess the exchange for the URL slug. Most US large-caps live on NYSE/NASDAQ.
_EXCHANGE_SLUGS = {
    "NMS": "nasdaq", "NASDAQ": "nasdaq", "NGM": "nasdaq", "NCM": "nasdaq",
    "NYSE": "nyse", "NYQ": "nyse", "NYSEArca": "nyse",
}

_INDEX_URL = "https://www.fool.com/quote/{exch}/{ticker}/earnings-call-transcripts/"
_BASE = "https://www.fool.com"

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


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_LINK_RE = re.compile(
    r'<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)


def _strip_html(html: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def _guess_exchanges(symbol: str) -> list[str]:
    """Try a few exchange slugs in order. We don't have a clean ticker→exchange
    map for every symbol; in practice nasdaq + nyse cover ~95% of US listings."""
    return ["nasdaq", "nyse", "nyseamerican"]


def list_transcripts(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    """Scrape the per-symbol index page; return recent transcript URLs.

    Each entry: {date_iso, title, url, quarter, fiscal_year}
    """
    sym = symbol.upper().lower()
    cache_key = f"transcripts_idx:{sym}:{limit}"
    if (cached := _cache_get(cache_key)) is not None:
        return cached

    html = None
    used_exch = None
    for exch in _guess_exchanges(symbol):
        url = _INDEX_URL.format(exch=exch, ticker=sym)
        try:
            r = httpx.get(url, headers={"User-Agent": _UA}, timeout=12.0,
                          follow_redirects=True)
            if r.status_code == 200 and ("earnings-call-transcript" in r.text.lower()):
                html = r.text
                used_exch = exch
                break
        except Exception as e:
            log.warning("transcripts_index_fetch_failed", symbol=symbol, exch=exch, err=str(e))
            continue
    if html is None:
        _cache_set(cache_key, [], ttl=12 * 3600)
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(html):
        href, text = m.group(1), m.group(2).strip()
        if "earnings-call-transcript" not in href.lower():
            continue
        full = href if href.startswith("http") else (_BASE + href)
        if full in seen:
            continue
        seen.add(full)
        title = text
        # Parse quarter/year from title where possible
        qm = re.search(r"Q(\d)[\s\-]+(\d{4})", title)
        quarter = int(qm.group(1)) if qm else None
        year = int(qm.group(2)) if qm else None
        out.append({
            "title": title,
            "url": full,
            "quarter": quarter,
            "fiscal_year": year,
            "exchange": used_exch,
        })
        if len(out) >= limit:
            break
    _cache_set(cache_key, out, ttl=12 * 3600)
    return out


def fetch_transcript_excerpt(url: str, max_chars: int = 8000) -> dict[str, Any] | None:
    """Fetch a single transcript and extract the body text (capped).

    Returns: {url, char_count, body, published_at?}
    """
    key = f"transcript:{url}"
    if (cached := _cache_get(key)) is not None:
        return cached
    try:
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=20.0,
                      follow_redirects=True)
        if r.status_code != 200:
            return None
        html = r.text
    except Exception as e:
        log.warning("transcript_fetch_failed", url=url, err=str(e))
        return None

    # Pull the publication date out of <time datetime="..."> if present.
    pub_m = re.search(r'<time[^>]+datetime="([^"]+)"', html)
    pub_at = pub_m.group(1) if pub_m else None

    # The transcript body is inside the <article> tag in Fool's template.
    body_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
    if body_match:
        body_html = body_match.group(1)
    else:
        body_html = html
    body = _strip_html(body_html)[:max_chars]
    if len(body) < 500:
        # Likely a paywall or template mismatch — bail
        _cache_set(key, None, ttl=6 * 3600)
        return None
    out = {"url": url, "char_count": len(body), "body": body, "published_at": pub_at}
    _cache_set(key, out, ttl=7 * 24 * 3600)  # transcripts don't change
    return out


def transcript_signal(symbol: str) -> dict[str, Any]:
    """Sentiment + tone summary of the most recent earnings call transcript.

    Scores the body with finBERT (lexicon fallback) and slices into rough
    halves (management discussion vs Q&A) — Q&A sentiment is often more
    revealing than management's prepared remarks.
    """
    entries = list_transcripts(symbol, limit=3)
    if not entries:
        return {"score": 0.0, "summary": "No Motley Fool transcripts found."}
    latest = entries[0]
    excerpt = fetch_transcript_excerpt(latest["url"], max_chars=12000)
    if not excerpt:
        return {"score": 0.0, "summary": "Could not fetch transcript body.",
                 "latest_url": latest.get("url")}
    body = excerpt["body"]
    n = len(body)
    half = n // 2
    md_text = body[:half]    # prepared remarks ("management discussion")
    qa_text = body[half:]   # Q&A
    md_pol, md_conf = sent.score_text(md_text)
    qa_pol, qa_conf = sent.score_text(qa_text)
    # Weight Q&A more — managers tend to spin prepared remarks
    score = 0.4 * md_pol + 0.6 * qa_pol

    return {
        "score": round(float(score), 3),
        "latest_url": latest.get("url"),
        "latest_title": latest.get("title"),
        "published_at": excerpt.get("published_at"),
        "management_sentiment": round(float(md_pol), 3),
        "qa_sentiment": round(float(qa_pol), 3),
        "n_chars": n,
        "summary": f"{latest.get('title')}: management {md_pol:+.2f}, Q&A {qa_pol:+.2f}",
    }
