"""Opt-in daily auto-run for armed trading bots.

Swing systems are decided on the daily close and executed at the next open.
Doing that by hand every morning is where discipline breaks down, so this
loop runs each bot that is armed, active, a "trader", **and has
`execution.autorun` set** once per US trading day at `BOT_AUTORUN_ET`
(default 09:45 New York time, after the opening auction settles). Every
guardrail in `trading_bot.run_live` still applies: arm state, market-hours
check, daily kill-switch, drawdown halt, position caps, PDT.

OFF unless `BOT_AUTORUN=1`, and even then only for bots that opted in. The
Alpaca endpoint (paper vs live) is decided server-side by
`alpaca_trading_base_url`; "sim" bots use their private simulated ledger.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.core.logging import log
from app.db.models import TradingBot
from app.db.session import SessionLocal
from app.services import broker_alpaca as broker
from app.services import trading_bot as botsvc

_ET = ZoneInfo("America/New_York")
_task: asyncio.Task | None = None


def enabled() -> bool:
    return os.environ.get("BOT_AUTORUN", "0").strip() in ("1", "true", "yes")


def run_time_et() -> tuple[int, int]:
    raw = os.environ.get("BOT_AUTORUN_ET", "09:45")
    try:
        h, m = (int(x) for x in raw.split(":", 1))
        return h, m
    except ValueError:
        return 9, 45


def is_due(bot_state: dict[str, Any] | None, now: datetime) -> bool:
    """Weekday, past the run time, and not yet run today (New York date)."""
    et = now.astimezone(_ET)
    if et.weekday() >= 5 or (et.hour, et.minute) < run_time_et():
        return False
    return (bot_state or {}).get("last_autorun_day") != et.date().isoformat()


def run_due(now: datetime | None = None, *, broker_mod=broker) -> list[dict[str, Any]]:
    """Run every due bot once. Returns a summary per bot (for logs/tests)."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(_ET).date().isoformat()
    out: list[dict[str, Any]] = []
    with SessionLocal() as db:
        bots = db.scalars(select(TradingBot).where(
            TradingBot.armed.is_(True), TradingBot.status == "active")).all()
        for b in bots:
            ex = {**botsvc.DEFAULT_EXECUTION, **(b.execution or {})}
            if ((b.bot_type or "trader") != "trader" or not ex.get("autorun")
                    or not is_due(b.run_state, now)):
                continue
            if ex.get("broker") != "sim" and not broker_mod.is_configured():
                continue
            try:
                res = botsvc.execute_pass(universe=b.universe or [], dsl=b.dsl, execution=b.execution,
                                          run_state=b.run_state, alpaca=broker_mod,
                                          context={"source": "autorun", "bot_id": str(b.id)})
            except Exception as e:  # noqa: BLE001 — one bot must not stop the others
                log.warning("bot_autorun_failed", bot=str(b.id), err=str(e))
                out.append({"bot": str(b.id), "error": str(e)})
                continue
            state = dict(res.get("run_state") or {})
            # Mark the day done even when blocked by a market holiday, so the
            # loop doesn't retry every minute; a data/broker outage is retried.
            blocked = res.get("blocked")
            if blocked in (None, "market_closed"):
                state["last_autorun_day"] = today
            b.run_state = state
            b.last_run_at = now
            db.commit()
            acts = [a.get("action") for a in res.get("actions", [])]
            log.info("bot_autorun", bot=str(b.id), blocked=blocked, actions=acts)
            out.append({"bot": str(b.id), "blocked": blocked, "actions": acts})
    return out


_last_score_day: str | None = None


def score_gate_daily(now: datetime | None = None) -> dict[str, int] | None:
    """Backfill AI-gate outcomes once per New York day (after the close)."""
    global _last_score_day
    et = (now or datetime.now(timezone.utc)).astimezone(_ET)
    day = et.date().isoformat()
    if _last_score_day == day or et.hour < 17:
        return None
    from app.services import gate_log
    res = gate_log.score_pending()
    _last_score_day = day
    log.info("gate_scored", **res)
    return res


async def _loop() -> None:
    await asyncio.sleep(10)
    while True:
        try:
            await asyncio.to_thread(run_due)
            await asyncio.to_thread(score_gate_daily)
        except Exception as e:  # noqa: BLE001
            log.warning("bot_autorun_loop_error", err=str(e))
        await asyncio.sleep(60)


def start() -> None:
    global _task
    if not enabled() or _task is not None:
        return
    log.info("bot_autorun_enabled", at_et="%02d:%02d" % run_time_et(), paper=broker.is_paper())
    _task = asyncio.create_task(_loop())


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        _task = None
