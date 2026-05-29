"""Background warmup — keep hot data ready so the UI never has to wait.

Started from `app/main.py` lifespan. Runs an asyncio loop that fires warming
jobs on a schedule. No new dependency — pure asyncio.

Jobs:
  - screeners        : run rising_stars / value_with_catalyst / breakouts /
                       smart_money and store results in the in-memory cache.
                       Triggers `screener_cache` so the routers serve instantly.
  - watched_opinions : for every symbol in any user's portfolio/watchlist,
                       pre-fetch the opinion signal pack so /ai/opinion warms.
                       Skipped if no users yet.
  - notifications    : run the scanners every 15 min so the bell badge stays
                       live without users having to click "Scan".

Cadence:
  - During US regular session (14:30–21:00 UTC, weekdays): every 5 min
  - Pre/post hours (09:00–14:30 / 21:00–01:00 UTC):        every 15 min
  - Otherwise (overnight UTC):                              every 60 min

Tunable via WARMUP_INTERVAL_SECONDS env var (overrides all above).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import log
from app.db.models import Portfolio, User, Watchlist, WatchlistItem
from app.db.session import SessionLocal
from app.services import screener as screener_svc

# We cache screener results in the router module's local dict — import it.
try:
    from app.routers.screener import _screener_cache as SCREENER_CACHE
except Exception:
    SCREENER_CACHE = {}

_STRATEGIES = ("rising_stars", "value_with_catalyst", "breakouts", "smart_money")


def _current_interval_seconds() -> int:
    """Pick a cadence based on US market session in UTC."""
    if (override := os.environ.get("WARMUP_INTERVAL_SECONDS")):
        try:
            return max(60, int(override))
        except ValueError:
            pass
    now = datetime.now(timezone.utc)
    # Sat=5, Sun=6
    if now.weekday() >= 5:
        return 60 * 60                   # 1h on weekends
    minute = now.hour * 60 + now.minute  # UTC minute of day
    # Regular session 14:30–21:00 UTC (during EDT) — keep tight
    if 14 * 60 + 30 <= minute <= 21 * 60:
        return 5 * 60
    # Pre / post hours roughly 9–14:30 and 21–01:00 UTC
    if (9 * 60 <= minute < 14 * 60 + 30) or (21 * 60 < minute or minute < 1 * 60):
        return 15 * 60
    # Overnight quiet
    return 60 * 60


def _warm_one_screener(strat: str) -> dict[str, Any]:
    import time as _t
    try:
        t0 = _t.time()
        results = screener_svc.run_screener(strat, limit=15)
        cache_key = f"{strat}:12:None:None:None"   # matches the router's default param set
        SCREENER_CACHE[cache_key] = (_t.time() + 5 * 60, results)
        return {"n": len(results), "took_ms": int((_t.time() - t0) * 1000)}
    except Exception as e:
        log.warning("warmup_screener_failed", strategy=strat, err=str(e))
        return {"error": str(e)}


def _warm_screeners() -> dict[str, Any]:
    """Run every strategy in PARALLEL and prime the router's in-memory cache.
    Serial → parallel cuts first warmup pass from ~65s to ~25s (= slowest one).
    """
    from concurrent.futures import ThreadPoolExecutor
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(_STRATEGIES)) as ex:
        futures = {ex.submit(_warm_one_screener, s): s for s in _STRATEGIES}
        for fut in futures:
            strat = futures[fut]
            try:
                out[strat] = fut.result(timeout=120)
            except Exception as e:
                out[strat] = {"error": str(e)}
    return out


def _gather_watched_symbols(db: Session, cap: int = 50) -> list[str]:
    """All symbols across every user's portfolios + watchlists, deduped."""
    syms: set[str] = set()
    # Portfolios — gather distinct symbols from all transactions
    for p in db.execute(select(Portfolio)).scalars().all():
        for t in p.transactions:
            syms.add(t.symbol)
    # Watchlists
    for w in db.execute(select(Watchlist)).scalars().all():
        for item in (w.items or []):
            syms.add(item.symbol if hasattr(item, "symbol") else str(item))
    return list(syms)[:cap]


def _warm_opinions(symbols: list[str]) -> dict[str, Any]:
    """Pre-compute the FULL opinion (signals + LLM synthesis) so /ai/opinion
    serves instantly from cache. Uses the conservative risk mode (most common
    default); the other two modes warm on first user click.

    LLM call cost: with Groq free tier (~30 req/min), 30 symbols / 5-min cycle =
    6 LLM calls/min — well within budget. Disable per-symbol LLM by setting
    WARMUP_OPINION_LLM=0.
    """
    from app.services.opinion import get_opinion
    import time as _t, os
    use_llm = os.environ.get("WARMUP_OPINION_LLM", "1") == "1"
    out: dict[str, Any] = {}
    for sym in symbols:
        try:
            t0 = _t.time()
            get_opinion(sym, use_llm=use_llm, risk_mode="conservative")
            out[sym] = {"took_ms": int((_t.time() - t0) * 1000)}
        except Exception as e:
            out[sym] = {"error": str(e)}
    return out


def _warm_notifications() -> dict[str, Any]:
    """Run the notification scanner for every user."""
    from app.services.notifications import scan_all
    out: dict[str, Any] = {"users": 0, "created": 0}
    db = SessionLocal()
    try:
        users = db.execute(select(User)).scalars().all()
        for u in users:
            try:
                r = scan_all(db, u)
                out["users"] += 1
                out["created"] += r.get("n_created", 0)
            except Exception as e:
                log.warning("warmup_scan_user_failed", user=str(u.id), err=str(e))
    finally:
        db.close()
    return out


def _record_and_score_track(scan_result: dict[str, Any]) -> dict[str, Any]:
    """Snapshot freshly-surfaced opportunities and backfill matured returns.

    Two operations against the OpportunityRecord ledger:
      - record_opportunities: one new row per (symbol, strategy, day)
      - score_matured: fill 1w/4w/12w forward returns for rows old enough
    """
    from app.services import track_record as tr
    out: dict[str, Any] = {"recorded": 0, "scored": 0}
    db = SessionLocal()
    try:
        try:
            out["recorded"] = tr.record_opportunities(db, scan_result)
        except Exception as e:
            log.warning("warmup_track_record_record_failed", err=str(e))
        try:
            out["scored"] = tr.score_matured(db)
        except Exception as e:
            log.warning("warmup_track_record_score_failed", err=str(e))
    finally:
        db.close()
    return out


def _warm_triggers() -> dict[str, Any]:
    """Evaluate every active entry trigger; fire notifications for those met."""
    from app.services import triggers as trig_svc
    out: dict[str, Any] = {"fired": 0}
    db = SessionLocal()
    try:
        out["fired"] = trig_svc.evaluate_all(db)
    except Exception as e:
        log.warning("warmup_triggers_failed", err=str(e))
    finally:
        db.close()
    return out


_opinions_warmed_this_session: bool = False


async def _run_once_async() -> None:
    """One warmup pass — runs blocking work in threads so the event loop stays free.

    Cadence policy:
      1. Screeners — every pass (no LLM cost, just market data).
      2. Notifications — every pass (read-only of cached screener results).
      3. Full opinions (LLM-bound) — exactly ONCE per server lifetime, on the
         first warmup pass. After that, opinion synthesis happens on demand
         when the user clicks "Get opinion" — that's the only way to keep the
         free-tier LLM quota (Groq TPM, Gemini RPM) from burning out within
         minutes. Set WARMUP_OPINIONS_FORCE=1 in .env to re-enable periodic
         warming if you're on a paid LLM tier.
    """
    global _opinions_warmed_this_session

    # 1) Screeners (heaviest non-LLM — ~14s cold parallel)
    screen = await asyncio.to_thread(_warm_screeners)
    log.info("warmup_screeners", **{k: v for k, v in screen.items()})

    # 2) Opinions — once per process lifetime (or forced via env)
    if not _opinions_warmed_this_session or os.environ.get("WARMUP_OPINIONS_FORCE", "0") == "1":
        db = SessionLocal()
        try:
            watched = await asyncio.to_thread(_gather_watched_symbols, db, 15)
        finally:
            db.close()
        if watched:
            ops = await asyncio.to_thread(_warm_opinions, watched)
            ok = sum(1 for v in ops.values() if "error" not in v)
            log.info("warmup_opinions", n_symbols=len(watched), n_ok=ok)
            _opinions_warmed_this_session = True

    # 2b) Market regime — cheap cross-asset read, cached 30 min. Warm it so the
    # dashboard gauge and any regime-aware sizing is instant.
    try:
        from app.services import regime as regime_svc
        reg = await asyncio.to_thread(regime_svc.assess, True)
        log.info("warmup_regime", regime=reg.get("regime"), score=reg.get("score"))
    except Exception as e:
        log.warning("warmup_regime_failed", err=str(e))

    # 3) Opportunity Engine — runs all 6 strategies in parallel and caches
    # the unified feed so /opportunities serves instantly. We also snapshot
    # the surfaced names into the track-record ledger (one row per
    # symbol+strategy+day) and backfill forward returns for matured rows so
    # the platform builds an honest, self-auditing batting average over time.
    try:
        from app.services import opportunities as opp_svc
        opps = await asyncio.to_thread(opp_svc.scan_and_cache, 50)
        log.info("warmup_opportunities", n=opps.get("n_opportunities"))
        rec = await asyncio.to_thread(_record_and_score_track, opps)
        log.info("warmup_track_record", **rec)
    except Exception as e:
        log.warning("warmup_opportunities_failed", err=str(e))

    # 4) Notification scan (light, no LLM)
    notif = await asyncio.to_thread(_warm_notifications)
    log.info("warmup_notifications", **notif)

    # 5) Entry triggers — fire user-defined watch conditions into notifications.
    trig = await asyncio.to_thread(_warm_triggers)
    log.info("warmup_triggers", **trig)


_loop_task: asyncio.Task | None = None
_stop = False


async def _scheduler_loop() -> None:
    """Forever loop — wait then warm, picking interval based on session."""
    # First pass: small delay so app finishes starting, then go.
    await asyncio.sleep(5)
    while not _stop:
        try:
            await _run_once_async()
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.error("warmup_pass_failed", err=str(e))
        interval = _current_interval_seconds()
        log.info("warmup_sleeping", seconds=interval)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return


def start_scheduler() -> None:
    """Idempotent. Call from FastAPI lifespan."""
    global _loop_task, _stop
    if os.environ.get("WARMUP_DISABLED", "0") == "1":
        log.info("warmup_disabled")
        return
    if _loop_task and not _loop_task.done():
        return
    _stop = False
    _loop_task = asyncio.create_task(_scheduler_loop())
    log.info("warmup_started")


async def stop_scheduler() -> None:
    global _loop_task, _stop
    _stop = True
    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        try:
            await _loop_task
        except (asyncio.CancelledError, Exception):
            pass
