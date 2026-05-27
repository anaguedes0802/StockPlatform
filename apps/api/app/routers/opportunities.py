"""Opportunities API — ongoing multi-strategy hunt.

GET  /opportunities         — current best opportunities (cached, max 10min stale)
GET  /opportunities/scan    — force a fresh scan (slow, 30-60s)
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.services import opportunities as opp_svc

router = APIRouter(prefix="/opportunities", tags=["opportunities"])


@router.get("")
def list_opportunities(limit: int = Query(30, ge=1, le=100), max_age_s: int = 600) -> dict:
    """Serve the cached scan if fresh; otherwise run a new one. Never 500 —
    if scan fails (rate limits, etc.) return whatever we last had + an error
    field the UI can surface."""
    cached = opp_svc.get_cached(max_age_s=max_age_s)
    if cached:
        opps = cached.get("opportunities", [])
        return {**cached, "opportunities": opps[:limit], "cached": True}
    # Cache stale or empty — try a fresh scan, but guard against any exception.
    try:
        fresh = opp_svc.scan_and_cache(limit=limit)
        return {**fresh, "cached": False}
    except Exception as e:
        # If we have a stale cache (older than max_age_s), still serve it
        # with a warning rather than 500.
        stale = opp_svc.get_cached(max_age_s=3600 * 24)
        if stale:
            return {**stale, "opportunities": stale.get("opportunities", [])[:limit],
                    "cached": True, "stale": True, "error": str(e)[:200]}
        return {"as_of": None, "n_strategies_run": 0, "n_opportunities": 0,
                "opportunities": [], "cached": False, "error": str(e)[:200]}


@router.post("/scan")
def force_scan(limit: int = Query(30, ge=1, le=100)) -> dict:
    """Force a fresh scan ignoring the cache. Slow (~30-60s)."""
    try:
        return {**opp_svc.scan_and_cache(limit=limit), "cached": False}
    except Exception as e:
        stale = opp_svc.get_cached(max_age_s=3600 * 24)
        if stale:
            return {**stale, "opportunities": stale.get("opportunities", [])[:limit],
                    "cached": True, "stale": True, "error": str(e)[:200]}
        return {"as_of": None, "opportunities": [], "cached": False, "error": str(e)[:200]}
