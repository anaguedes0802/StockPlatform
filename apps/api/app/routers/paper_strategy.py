"""Forward test of the stock-selection strategy on the Alpaca paper account.

Reads are public (like the rest of the research pages). Anything that can
submit orders requires a logged-in user, mirroring the trading-bot router, and
the service itself refuses to run against a real-money endpoint.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db.models import User
from app.deps import get_current_user
from app.services import paper_replay
from app.services import paper_strategy as ps

router = APIRouter(prefix="/paper-strategy", tags=["paper-strategy"])

# Ranking ~900 names takes minutes; cache the dry-run plan.
_plan_cache: tuple[float, dict[str, Any]] | None = None


@router.get("/status")
async def status() -> dict:
    return await asyncio.to_thread(ps.status)


@router.get("/plan")
async def plan(refresh: bool = Query(False)) -> dict:
    """Dry run: this month's target book and the orders a rebalance would send."""
    global _plan_cache
    if not refresh and _plan_cache and time.time() - _plan_cache[0] < 3600:
        return {**_plan_cache[1], "cached": True}
    try:
        p = await asyncio.to_thread(ps.plan)
    except ps.NotPaperError as e:
        raise HTTPException(400, str(e)) from e
    _plan_cache = (time.time(), p)
    return {**p, "cached": False}


@router.post("/rebalance")
async def rebalance(force: bool = False, user: User = Depends(get_current_user)) -> dict:
    """Submit paper orders if a rebalance is due (market hours only)."""
    try:
        out = await asyncio.to_thread(ps.rebalance, True, force)
    except ps.NotPaperError as e:
        raise HTTPException(400, str(e)) from e
    if out.get("executed"):
        await asyncio.to_thread(ps.snapshot)
    return out


@router.get("/replay")
def replay() -> dict:
    """Latest historical replay of the same rules (computed by
    `scripts/paper_strategy.py replay` or POST /replay)."""
    res = paper_replay.cached_result()
    return res or {"available": False,
                   "message": "No replay yet — run scripts/paper_strategy.py replay"}


@router.post("/replay")
async def run_replay(start: str = "2016-01-04", gated: bool = True,
                     user: User = Depends(get_current_user)) -> dict:
    """Recompute the replay (minutes: ranks ~900 stocks daily over the period)."""
    res = await asyncio.to_thread(paper_replay.run, start, 10_000.0, gated)
    return {k: v for k, v in res.items() if k != "holdings_end"}
