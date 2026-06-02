"""Discovery API — emerging opportunities + a catalyst news feed.

Two read-only, unauthenticated endpoints (mirroring the other /bot/* GET
endpoints) that surface the names "about to fly" and the news driving them:

  GET /discovery/opportunities   — best emerging-momentum / narrative /
       catalyst names, by running the existing multi-strategy screener
       (rising_stars / breakouts / catalyst_plays / smart_money etc. via the
       opportunities engine) and merging/deduping into one ranked list.

  GET /discovery/catalyst-news   — recent high-impact / catalyst news
       (upgrades, guidance, AI narrative, unusual activity) for the top
       opportunity symbols, classified + scored via news_intel.

Both are SLOW to compute (the screener scans the universe), so the computed
response is cached in Redis for ~15 min keyed by endpoint+limit. Repeat calls
are instant. We fail open if Redis is down and stay defensive per-symbol so a
single bad symbol never sinks the whole response.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis
from fastapi import APIRouter, Query

from app.config import settings
from app.core.logging import log
from app.services import news as news_svc
from app.services import news_intel as ni
from app.services import opportunities as opp_svc

router = APIRouter(prefix="/discovery", tags=["discovery"])

_CACHE_TTL = 900  # 15 minutes

# A few liquid large-caps to backfill the news feed when the opportunity set is
# thin — these reliably carry catalyst-grade headlines (AI narrative, guidance).
_NEWS_FALLBACK_SYMBOLS = ["NVDA", "MSFT", "AMD", "CRM", "AVGO"]


# ---------- Redis cache (same pattern as market_data) ----------

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- opportunities ----------

def _build_opportunities(limit: int) -> dict[str, Any]:
    """Run the multi-strategy opportunity engine and shape it to the discovery
    contract. The engine already merges/dedupes screener strategies, computes a
    ranked score, and emits human-readable rationale — we just project it."""
    try:
        # Pull a bit more than `limit` so dedupe/ranking inside the engine has
        # headroom; favors rising_stars / breakouts / catalyst_plays / smart_money.
        scan = opp_svc.scan(limit=max(limit, 20))
    except Exception as e:
        log.warning("discovery_opportunities_scan_failed", err=str(e))
        scan = {"opportunities": []}

    items: list[dict[str, Any]] = []
    for o in scan.get("opportunities", []):
        try:
            items.append({
                "symbol": o.get("symbol"),
                "name": o.get("name"),
                "score": float(o.get("score") or 0.0),
                "strategy": o.get("best_strategy") or "",
                "last_price": o.get("last_price"),
                "change_pct": o.get("momentum_1m_pct"),
                "reasons": list(o.get("rationale") or [])[:6],
            })
        except Exception:
            continue

    items.sort(key=lambda x: -(x.get("score") or 0.0))
    return {"as_of": _now_iso(), "opportunities": items[:limit]}


# ---------- catalyst news ----------

# Cap how many symbols we classify news for — classify_and_score is heavy
# (fetch + per-article scoring), so we bound it and run the symbols in parallel.
_NEWS_MAX_SYMBOLS = 8


def _news_seed_symbols() -> list[str]:
    """Seed the news feed from the ALREADY-CACHED opportunity list when present.

    Critically we do NOT trigger the full (slow) opportunity scan here — if the
    cache is cold we just use the large-cap fallbacks, so the news endpoint never
    pays the ~2-minute scan cost.
    """
    symbols: list[str] = []
    cached = _cache_get("discovery:opps:12")
    if cached:
        symbols = [o["symbol"] for o in cached.get("opportunities", []) if o.get("symbol")]
    for s in _NEWS_FALLBACK_SYMBOLS:
        if s not in symbols:
            symbols.append(s)
    return symbols[:_NEWS_MAX_SYMBOLS]


def _build_catalyst_news(limit: int) -> dict[str, Any]:
    """Pull recent classified news for the top opportunity symbols (plus a few
    large caps) and keep the higher-impact / catalyst items, newest first.

    Symbols are classified in parallel (the per-symbol fetch+score is the slow
    part) and bounded by `_NEWS_MAX_SYMBOLS` so the endpoint stays responsive.
    """
    from concurrent.futures import ThreadPoolExecutor

    symbols = _news_seed_symbols()

    def _classify(sym: str):
        try:
            return sym, ni.classify_and_score(sym, days_window=14)
        except Exception as e:  # noqa: BLE001
            log.debug("discovery_news_symbol_failed", symbol=sym, err=str(e))
            return sym, None

    items: list[dict[str, Any]] = []
    seen: set[str] = set()  # dedupe on url AND normalized title (same story, diff url)

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_classify, symbols))

    for sym, scored in results:
        if not scored:
            continue
        for art in scored.get("articles") or []:
            try:
                intel = art.get("intel") or {}
                materiality = float(intel.get("materiality") or 0.0)
                category = intel.get("category") or "noise"
                # Keep only catalyst-grade items — drop noise / low materiality.
                if category == "noise" or materiality < 0.4:
                    continue
                url = art.get("url")
                title_key = (art.get("title") or "").strip().lower()
                if (url and url in seen) or (title_key and title_key in seen):
                    continue
                if url:
                    seen.add(url)
                if title_key:
                    seen.add(title_key)
                direction = intel.get("direction")
                items.append({
                    "symbol": sym,
                    "title": art.get("title") or "",
                    "source": art.get("source"),
                    "url": url,
                    "published_at": _as_iso(art.get("published_at")),
                    "sentiment": direction,
                    "summary": intel.get("rationale") or art.get("summary"),
                    "_materiality": materiality,
                    "_ts": _to_epoch(art.get("published_at")),
                })
            except Exception:
                continue

    # Newest first; ties broken by materiality.
    items.sort(key=lambda it: (it.get("_ts") or 0.0, it.get("_materiality") or 0.0), reverse=True)
    cleaned = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items[:limit]]
    return {"as_of": _now_iso(), "items": cleaned}


def _to_epoch(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(str(ts)).timestamp()
        except Exception:
            return 0.0


def _as_iso(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            return None
    return str(ts)


# ---------- routes ----------

@router.get("/opportunities")
def discovery_opportunities(limit: int = Query(12, ge=1, le=50)) -> dict[str, Any]:
    key = f"discovery:opps:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    try:
        out = _build_opportunities(limit)
    except Exception as e:
        log.warning("discovery_opportunities_failed", err=str(e))
        out = {"as_of": _now_iso(), "opportunities": []}
    _cache_set(key, out, _CACHE_TTL)
    return out


@router.get("/catalyst-news")
def discovery_catalyst_news(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    key = f"discovery:news:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    try:
        out = _build_catalyst_news(limit)
    except Exception as e:
        log.warning("discovery_catalyst_news_failed", err=str(e))
        out = {"as_of": _now_iso(), "items": []}
    _cache_set(key, out, _CACHE_TTL)
    return out
