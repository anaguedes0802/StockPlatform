"""Notifications — what surfaces in the bell badge.

Notification kinds:
  - new_rising_star    : symbol surfaced in today's rising_stars vs prior scan
  - underpriced_value  : symbol newly classified as a value play (PE < sector,
                         positive catalyst recent)
  - portfolio_alert    : portfolio review surfaced a critical or warning
  - watchlist_surge    : watchlist symbol moved beyond threshold

The scanner is invoked on-demand via POST /notifications/scan (so we don't
silently burn CPU/credits). A Phase-2 cron job can call the same scan.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from app.core.logging import log
from app.db.models import Notification, ScreenerSnapshot, User, Portfolio, Watchlist
from app.services import market_data as md
from app.services import screener as screener_svc


def _dedupe(user_id: uuid.UUID | None, kind: str, symbol: str | None) -> str:
    raw = f"{user_id}|{kind}|{symbol or ''}|{datetime.now(timezone.utc).date().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _create_if_new(db: Session, *, user_id: uuid.UUID | None, kind: str, severity: str,
                   symbol: str | None, title: str, body: str | None,
                   payload: dict[str, Any] | None) -> Notification | None:
    key = _dedupe(user_id, kind, symbol)
    # Has the same alert already fired today?
    existing = db.scalar(
        select(Notification).where(
            and_(Notification.user_id == user_id, Notification.dedupe_key == key)
        )
    )
    if existing:
        return None
    n = Notification(
        user_id=user_id, kind=kind, severity=severity, symbol=symbol,
        title=title, body=body, payload=payload, dedupe_key=key,
    )
    db.add(n)
    db.commit()
    db.refresh(n)
    return n


# ----------------------------------------------------------------------------
# Scan: new rising stars (diff today vs previous scan)
# ----------------------------------------------------------------------------

def scan_new_rising_stars(db: Session, user_id: uuid.UUID | None = None) -> list[Notification]:
    """Run rising_stars now, diff vs the most recent stored snapshot, emit a
    notification per *newly-surfacing* symbol. Then archive today's snapshot."""
    today = screener_svc.run_screener("rising_stars", limit=15)
    today_syms = [h["symbol"] for h in today]

    # Find most recent prior snapshot (could be from earlier today or earlier)
    prev = db.scalar(
        select(ScreenerSnapshot)
        .where(ScreenerSnapshot.strategy == "rising_stars")
        .order_by(ScreenerSnapshot.captured_at.desc())
    )
    prev_syms = set(prev.symbols or []) if prev else set()
    new_syms = [s for s in today_syms if s not in prev_syms]

    out: list[Notification] = []
    for sym in new_syms:
        hit = next((h for h in today if h["symbol"] == sym), {})
        rationale = (hit.get("rationale") or [])
        body = " · ".join(rationale[:3]) if rationale else "Surfaced in today's rising-stars rank."
        n = _create_if_new(
            db, user_id=user_id, kind="new_rising_star", severity="info",
            symbol=sym,
            title=f"⭐ {sym} surfaced in rising stars",
            body=body,
            payload={"score": hit.get("score"), "rationale": rationale[:5]},
        )
        if n:
            out.append(n)

    # Archive today's snapshot (only if it actually changed)
    if today_syms != list(prev_syms):
        db.add(ScreenerSnapshot(
            strategy="rising_stars",
            symbols=today_syms,
            full_results=[{k: v for k, v in h.items() if k in ("symbol", "score", "name")} for h in today],
        ))
        db.commit()
    return out


# ----------------------------------------------------------------------------
# Scan: underpriced-value symbols
# ----------------------------------------------------------------------------

def scan_underpriced(db: Session, user_id: uuid.UUID | None = None) -> list[Notification]:
    today = screener_svc.run_screener("value_with_catalyst", limit=10)
    today_syms = [h["symbol"] for h in today]

    prev = db.scalar(
        select(ScreenerSnapshot)
        .where(ScreenerSnapshot.strategy == "value_with_catalyst")
        .order_by(ScreenerSnapshot.captured_at.desc())
    )
    prev_syms = set(prev.symbols or []) if prev else set()
    new_syms = [s for s in today_syms if s not in prev_syms]

    out: list[Notification] = []
    for sym in new_syms:
        hit = next((h for h in today if h["symbol"] == sym), {})
        rationale = hit.get("rationale") or []
        n = _create_if_new(
            db, user_id=user_id, kind="underpriced_value", severity="info",
            symbol=sym,
            title=f"💎 {sym} flagged as undervalued",
            body=" · ".join(rationale[:3]) if rationale else "Cheap on PE with a catalyst.",
            payload={"score": hit.get("score"), "rationale": rationale[:5]},
        )
        if n:
            out.append(n)

    if today_syms != list(prev_syms):
        db.add(ScreenerSnapshot(
            strategy="value_with_catalyst",
            symbols=today_syms,
            full_results=[{k: v for k, v in h.items() if k in ("symbol", "score", "name")} for h in today],
        ))
        db.commit()
    return out


# ----------------------------------------------------------------------------
# Scan: portfolio alerts (per-user)
# ----------------------------------------------------------------------------

def scan_portfolio_alerts(db: Session, user: User) -> list[Notification]:
    """Run portfolio_review for each of the user's portfolios; emit a
    notification per critical (or first warning) alert per position."""
    from app.services.portfolio_review import review_portfolio
    from collections import defaultdict
    out: list[Notification] = []

    ports = db.execute(select(Portfolio).where(Portfolio.user_id == user.id)).scalars().all()
    for p in ports:
        # Recreate the position list
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "cost": 0.0})
        by_sym: dict[str, list] = defaultdict(list)
        for t in p.transactions:
            by_sym[t.symbol].append(t)
            qty = float(t.quantity)
            if t.side == "buy":
                agg[t.symbol]["qty"] += qty
                agg[t.symbol]["cost"] += qty * float(t.price) + float(t.fee or 0)
            else:
                if agg[t.symbol]["qty"] > 0:
                    cost_per = agg[t.symbol]["cost"] / agg[t.symbol]["qty"]
                    sell = min(qty, agg[t.symbol]["qty"])
                    agg[t.symbol]["cost"] -= cost_per * sell
                    agg[t.symbol]["qty"] -= sell
        positions = [
            {"symbol": sym, "quantity": v["qty"],
             "avg_buy_price": v["cost"] / v["qty"] if v["qty"] else 0.0}
            for sym, v in agg.items() if v["qty"] > 0
        ]
        if not positions:
            continue
        try:
            reviews = review_portfolio(positions, dict(by_sym))
        except Exception as e:
            log.warning("portfolio_scan_failed", err=str(e))
            continue
        for r in reviews:
            crit = [a for a in (r.get("alerts") or []) if a.get("severity") == "critical"]
            warn = [a for a in (r.get("alerts") or []) if a.get("severity") == "warning"]
            top_alert = crit[0] if crit else (warn[0] if warn else None)
            if not top_alert:
                continue
            sev = top_alert["severity"]
            n = _create_if_new(
                db, user_id=user.id, kind="portfolio_alert", severity=sev,
                symbol=r["symbol"],
                title=f"⚠ {r['symbol']}: {top_alert['kind'].replace('_', ' ')}",
                body=top_alert["message"],
                payload={"verdict": r.get("verdict"), "pnl_pct": r.get("unrealized_pl_pct"),
                         "portfolio_id": str(p.id)},
            )
            if n:
                out.append(n)
    return out


# ----------------------------------------------------------------------------
# Scan: watchlist surge (intraday %-move beyond threshold)
# ----------------------------------------------------------------------------

def scan_watchlist_surges(db: Session, user: User, surge_pct: float = 5.0) -> list[Notification]:
    out: list[Notification] = []
    wls = db.execute(select(Watchlist).where(Watchlist.user_id == user.id)).scalars().all()
    seen: set[str] = set()
    for w in wls:
        for item in (w.items or []):
            sym = item.symbol if hasattr(item, "symbol") else item
            if sym in seen:
                continue
            seen.add(sym)
            try:
                q = md.get_quote(sym)
            except Exception:
                continue
            ch = q.get("change_pct")
            if ch is None or abs(ch) < surge_pct:
                continue
            arrow = "🚀" if ch > 0 else "📉"
            n = _create_if_new(
                db, user_id=user.id, kind="watchlist_surge",
                severity="warning" if abs(ch) > 10 else "info",
                symbol=sym,
                title=f"{arrow} {sym} {ch:+.1f}% today",
                body=f"Watchlist symbol moved past {surge_pct:.0f}% threshold (now ${q.get('price')}).",
                payload={"change_pct": ch, "price": q.get("price")},
            )
            if n:
                out.append(n)
    return out


# ----------------------------------------------------------------------------
# Combined scan
# ----------------------------------------------------------------------------

def scan_all(db: Session, user: User) -> dict[str, Any]:
    """Run every scanner. Returns counts + the new notifications."""
    created: list[Notification] = []
    try:    created.extend(scan_new_rising_stars(db, user_id=user.id))
    except Exception as e: log.warning("scan_rising_stars_failed", err=str(e))
    try:    created.extend(scan_underpriced(db, user_id=user.id))
    except Exception as e: log.warning("scan_underpriced_failed", err=str(e))
    try:    created.extend(scan_portfolio_alerts(db, user))
    except Exception as e: log.warning("scan_portfolio_alerts_failed", err=str(e))
    try:    created.extend(scan_watchlist_surges(db, user))
    except Exception as e: log.warning("scan_watchlist_surges_failed", err=str(e))
    return {
        "n_created": len(created),
        "by_kind": {k: sum(1 for n in created if n.kind == k) for k in set(n.kind for n in created)},
    }
