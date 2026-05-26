from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.rate_limit import per_ip_limiter
from app.services import cross_sectional as xs_svc
from app.services import screener as svc

router = APIRouter(prefix="/screener", tags=["screener"])


@router.get("/strategies")
def strategies() -> list[dict]:
    return svc.list_strategies()


@router.get("/cross-sectional/strategies")
def xs_strategies() -> list[dict]:
    return xs_svc.list_strategies()


@router.get(
    "/cross-sectional/{strategy}",
    dependencies=[Depends(per_ip_limiter(5, 0.1))],
)
async def cross_sectional(
    strategy: str,
    top_n: int = Query(10, ge=1, le=50),
    bottom_n: int = Query(10, ge=1, le=50),
    sectors: str | None = Query(None),
) -> dict:
    """Rank the entire universe by a chosen score and return top/bottom deciles
    (longs/shorts). This is what professional quants run.
    """
    try:
        return await asyncio.to_thread(
            xs_svc.rank_universe, strategy, None, top_n, bottom_n,
            [s.strip() for s in sectors.split(",")] if sectors else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# In-memory cache for screener results so the same query doesn't re-run the
# N-symbol scan every time. 5-min TTL because the result depends on live
# quotes that don't change much intraday for a top-N ranking.
_screener_cache: dict[str, tuple[float, list[dict]]] = {}


@router.get(
    "/run/{strategy}",
    dependencies=[Depends(per_ip_limiter(10, 0.2))],
)
async def run(
    strategy: str,
    limit: int = Query(12, ge=1, le=50),
    min_market_cap: float | None = Query(None, description="USD"),
    max_market_cap: float | None = Query(None, description="USD"),
    sectors: str | None = Query(None, description="comma-separated"),
) -> list[dict]:
    """Screening fans out N yfinance calls; run in a worker thread so the
    event loop stays responsive. 5-minute in-memory cache per parameter set."""
    import time as _t
    cache_key = f"{strategy}:{limit}:{min_market_cap}:{max_market_cap}:{sectors}"
    now = _t.time()
    entry = _screener_cache.get(cache_key)
    if entry and entry[0] > now:
        return entry[1]

    try:
        out = await asyncio.to_thread(
            svc.run_screener,
            strategy,
            None,                # universe (use default)
            limit,
            min_market_cap,
            max_market_cap,
            [s.strip() for s in sectors.split(",")] if sectors else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _screener_cache[cache_key] = (now + 300, out)
    # Light eviction.
    if len(_screener_cache) > 64:
        for k in list(_screener_cache.keys()):
            if _screener_cache[k][0] <= now:
                _screener_cache.pop(k, None)
    return out
