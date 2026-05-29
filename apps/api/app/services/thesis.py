"""Thesis tracking — evaluate committed theses against live price + signals.

A thesis is the professional discipline of writing down WHY you're in a trade,
what would prove you wrong (invalidation), the target, and the time horizon.
This service evaluates each active thesis and returns a status so the user
reviews against the thesis rather than reacting to noise.

Status logic:
  - invalidated : long below (or short above) invalidation_price
  - target_hit  : reached target_1 (or target_2)
  - expiring    : within 20% of horizon end, thesis not yet played out
  - expired     : past horizon_days, thesis hasn't hit target or invalidation
  - on_track    : none of the above (still working)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services import market_data as md


def evaluate(thesis: dict[str, Any]) -> dict[str, Any]:
    """Given a thesis dict (from the DB row), compute live status + P&L."""
    sym = thesis["symbol"]
    direction = thesis.get("direction", "long")
    try:
        last = float(md.get_quote(sym).get("price") or 0.0)
    except Exception:
        last = 0.0

    entry = _f(thesis.get("entry_price"))
    inval = _f(thesis.get("invalidation_price"))
    t1 = _f(thesis.get("target_1"))
    t2 = _f(thesis.get("target_2"))
    created = thesis.get("created_at")
    horizon = thesis.get("horizon_days")

    # P&L vs entry
    pnl_pct = None
    if entry and last:
        raw = (last - entry) / entry * 100
        pnl_pct = raw if direction == "long" else -raw

    # Days elapsed / remaining
    days_elapsed = None
    days_remaining = None
    if created:
        try:
            c = created if isinstance(created, datetime) else datetime.fromisoformat(str(created))
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            days_elapsed = (datetime.now(timezone.utc) - c).days
            if horizon:
                days_remaining = int(horizon) - days_elapsed
        except Exception:
            pass

    # Status
    status = "on_track"
    note = ""
    if last and inval:
        breached = (direction == "long" and last < inval) or (direction == "short" and last > inval)
        if breached:
            status = "invalidated"
            note = f"Price {last:.2f} breached invalidation {inval:.2f} — thesis is wrong, exit."
    if status == "on_track" and last and t1:
        hit = (direction == "long" and last >= t1) or (direction == "short" and last <= t1)
        if hit:
            if t2 and ((direction == "long" and last >= t2) or (direction == "short" and last <= t2)):
                status = "target_hit"
                note = f"Hit Target 2 ({t2:.2f}). Consider closing the remainder."
            else:
                status = "target_hit"
                note = f"Hit Target 1 ({t1:.2f}). Take partial profit, trail the stop."
    if status == "on_track" and days_remaining is not None:
        if days_remaining < 0:
            status = "expired"
            note = f"Past the {horizon}-day horizon and thesis hasn't played out — time-stop: re-underwrite or exit."
        elif days_remaining <= max(1, int((horizon or 0) * 0.2)):
            status = "expiring"
            note = f"{days_remaining}d left on the {horizon}-day horizon. If the catalyst hasn't materialized, plan an exit."

    if status == "on_track":
        note = "Thesis intact. Hold and monitor; act only if invalidation or target is hit."

    return {
        **thesis,
        "last_price": last or None,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "live_status": status,
        "status_note": note,
        # distance to invalidation / target as % (for risk display)
        "pct_to_invalidation": (round((inval - last) / last * 100, 1) if (last and inval) else None),
        "pct_to_target_1": (round((t1 - last) / last * 100, 1) if (last and t1) else None),
    }


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
