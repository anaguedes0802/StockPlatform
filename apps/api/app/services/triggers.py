"""Entry-trigger evaluation — fire a notification only when a user-defined
watch condition is actually met.

Evaluated by the background scanner every cycle. Each trigger has a condition
+ threshold; we fetch the symbol's current quote/indicators and check. When
met, we create a notification and stamp `last_fired_at` so it doesn't spam
(re-arms after 12h or when the condition resets).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import log
from app.db.models import EntryTrigger, User
from app.services import market_data as md


def _rsi(symbol: str) -> float | None:
    try:
        from app.services import indicators as ind
        df = md.get_history(symbol, interval="1d", range_="3mo")
        if df.empty or len(df) < 20:
            return None
        return float(ind.rsi(df["close"], 14).iloc[-1])
    except Exception:
        return None


def _evaluate_one(t: EntryTrigger) -> tuple[bool, str] | None:
    """Return (met, message) if the trigger condition is satisfied, else None."""
    sym = t.symbol
    cond = t.condition
    thr = float(t.threshold)
    try:
        q = md.get_quote(sym)
        price = float(q.get("price") or 0)
        ch_pct = q.get("change_pct")
    except Exception:
        return None
    if price <= 0:
        return None

    if cond == "price_below" and price <= thr:
        return True, f"{sym} hit ${price:.2f} — at/below your ${thr:.2f} level"
    if cond == "price_above" and price >= thr:
        return True, f"{sym} hit ${price:.2f} — at/above your ${thr:.2f} level"
    if cond == "pct_change_day" and ch_pct is not None:
        # threshold can be negative (drop) or positive (spike)
        if (thr < 0 and ch_pct <= thr) or (thr > 0 and ch_pct >= thr):
            return True, f"{sym} moved {ch_pct:+.1f}% today — crossed your {thr:+.0f}% trigger"
    if cond in ("rsi_below", "rsi_above"):
        rsi = _rsi(sym)
        if rsi is None:
            return None
        if cond == "rsi_below" and rsi <= thr:
            return True, f"{sym} RSI {rsi:.0f} — at/below your {thr:.0f} (oversold entry)"
        if cond == "rsi_above" and rsi >= thr:
            return True, f"{sym} RSI {rsi:.0f} — at/above your {thr:.0f}"
    if cond == "breakout_volume" and price >= thr:
        # also require a volume spike to confirm the breakout
        try:
            df = md.get_history(sym, interval="1d", range_="3mo")
            vol = df["volume"]
            vz = (vol.iloc[-1] - vol.rolling(20).mean().iloc[-1]) / (vol.rolling(20).std().iloc[-1] + 1e-9)
            if vz > 1.5:
                return True, f"{sym} broke ${thr:.2f} on {vz:.1f}σ volume — confirmed breakout"
        except Exception:
            return None
    return None


def evaluate_all(db: Session) -> int:
    """Check every active trigger; create notifications for those that fire.
    Returns count fired. Re-arms a trigger 12h after last fire."""
    from app.services.notifications import _create_if_new
    now = datetime.now(timezone.utc)
    rearm_cutoff = now - timedelta(hours=12)
    fired = 0

    triggers = db.execute(select(EntryTrigger).where(EntryTrigger.active.is_(True))).scalars().all()
    for t in triggers:
        # Skip if fired within the re-arm window
        if t.last_fired_at and t.last_fired_at > rearm_cutoff:
            continue
        result = _evaluate_one(t)
        if not result:
            continue
        met, message = result
        if not met:
            continue
        # Look up the owning user for the notification
        user = db.get(User, t.user_id)
        if not user:
            continue
        n = _create_if_new(
            db, user_id=user.id, kind="entry_trigger", severity="warning",
            symbol=t.symbol,
            title=f"🎯 {t.symbol} trigger hit",
            body=message + (f" — {t.note}" if t.note else ""),
            payload={"condition": t.condition, "threshold": float(t.threshold), "trigger_id": str(t.id)},
        )
        if n:
            t.last_fired_at = now
            t.fire_count = (t.fire_count or 0) + 1
            db.commit()
            fired += 1
    return fired
