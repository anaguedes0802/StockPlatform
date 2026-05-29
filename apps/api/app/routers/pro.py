"""Professional-process endpoints: thesis journal, portfolio risk, entry
triggers, and the platform's own track record."""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EntryTrigger, Portfolio, Thesis, Transaction, User
from app.db.session import get_db
from app.deps import get_current_user
from app.services import market_data as md
from app.services import portfolio_risk, regime as regime_svc, thesis as thesis_svc, track_record

router = APIRouter(tags=["pro"])


# ====================================================================== THESIS

class ThesisIn(BaseModel):
    symbol: str
    direction: str = "long"
    thesis: str
    catalyst: str | None = None
    entry_price: float | None = None
    invalidation_price: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    horizon_days: int | None = None
    conviction: int = 3


def _thesis_to_dict(t: Thesis) -> dict[str, Any]:
    return {
        "id": str(t.id), "symbol": t.symbol, "direction": t.direction,
        "thesis": t.thesis, "catalyst": t.catalyst,
        "entry_price": float(t.entry_price) if t.entry_price is not None else None,
        "invalidation_price": float(t.invalidation_price) if t.invalidation_price is not None else None,
        "target_1": float(t.target_1) if t.target_1 is not None else None,
        "target_2": float(t.target_2) if t.target_2 is not None else None,
        "horizon_days": t.horizon_days, "conviction": t.conviction,
        "status": t.status, "created_at": t.created_at.isoformat() if t.created_at else None,
        "close_reason": t.close_reason,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }


@router.get("/theses")
def list_theses(include_closed: bool = False, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)) -> list[dict]:
    q = select(Thesis).where(Thesis.user_id == user.id)
    if not include_closed:
        q = q.where(Thesis.status == "active")
    rows = db.execute(q.order_by(Thesis.created_at.desc())).scalars().all()
    # Evaluate each against live price
    return [thesis_svc.evaluate(_thesis_to_dict(t)) for t in rows]


@router.post("/theses", status_code=201)
def create_thesis(payload: ThesisIn, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> dict:
    t = Thesis(
        user_id=user.id, symbol=payload.symbol.upper(), direction=payload.direction,
        thesis=payload.thesis, catalyst=payload.catalyst,
        entry_price=payload.entry_price, invalidation_price=payload.invalidation_price,
        target_1=payload.target_1, target_2=payload.target_2,
        horizon_days=payload.horizon_days, conviction=max(1, min(5, payload.conviction)),
    )
    db.add(t); db.commit(); db.refresh(t)
    return thesis_svc.evaluate(_thesis_to_dict(t))


@router.post("/theses/{tid}/close", status_code=200)
def close_thesis(tid: uuid.UUID, reason: str = "manual", db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)) -> dict:
    t = db.get(Thesis, tid)
    if not t or t.user_id != user.id:
        raise HTTPException(404, "thesis not found")
    t.status = "closed"
    t.close_reason = reason
    t.closed_at = datetime.now(timezone.utc)
    try:
        t.closed_price = float(md.get_quote(t.symbol).get("price") or 0) or None
    except Exception:
        pass
    db.commit()
    return {"ok": True, "id": str(t.id), "status": t.status}


@router.delete("/theses/{tid}", status_code=204)
def delete_thesis(tid: uuid.UUID, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)) -> None:
    t = db.get(Thesis, tid)
    if not t or t.user_id != user.id:
        raise HTTPException(404, "thesis not found")
    db.delete(t); db.commit()


# =============================================================== PORTFOLIO RISK

@router.get("/portfolios/{pid}/risk")
def portfolio_risk_endpoint(pid: uuid.UUID, db: Session = Depends(get_db),
                            user: User = Depends(get_current_user)) -> dict:
    p = db.get(Portfolio, pid)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "portfolio not found")
    # Aggregate positions (avg-cost)
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"qty": 0.0, "cost": 0.0})
    for t in p.transactions:
        a = agg[t.symbol]; qty = float(t.quantity)
        if t.side == "buy":
            a["qty"] += qty; a["cost"] += qty * float(t.price) + float(t.fee or 0)
        elif a["qty"] > 0:
            cost_per = a["cost"] / a["qty"]; sell = min(qty, a["qty"])
            a["cost"] -= cost_per * sell; a["qty"] -= sell
    positions = []
    for sym, v in agg.items():
        if v["qty"] <= 0:
            continue
        last = float(md.get_quote(sym).get("price") or 0)
        positions.append({"symbol": sym, "quantity": v["qty"],
                          "avg_buy_price": v["cost"] / v["qty"] if v["qty"] else 0,
                          "market_value": v["qty"] * last})
    return portfolio_risk.analyze_risk(positions)


# =============================================================== ENTRY TRIGGERS

class TriggerIn(BaseModel):
    symbol: str
    condition: str   # price_below | price_above | pct_change_day | rsi_below | rsi_above | breakout_volume
    threshold: float
    note: str | None = None


@router.get("/triggers")
def list_triggers(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    rows = db.execute(
        select(EntryTrigger).where(EntryTrigger.user_id == user.id).order_by(EntryTrigger.created_at.desc())
    ).scalars().all()
    return [{
        "id": str(t.id), "symbol": t.symbol, "condition": t.condition,
        "threshold": float(t.threshold), "note": t.note, "active": t.active,
        "fire_count": t.fire_count,
        "last_fired_at": t.last_fired_at.isoformat() if t.last_fired_at else None,
    } for t in rows]


@router.post("/triggers", status_code=201)
def create_trigger(payload: TriggerIn, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> dict:
    valid = {"price_below", "price_above", "pct_change_day", "rsi_below", "rsi_above", "breakout_volume"}
    if payload.condition not in valid:
        raise HTTPException(400, f"condition must be one of {sorted(valid)}")
    t = EntryTrigger(user_id=user.id, symbol=payload.symbol.upper(),
                     condition=payload.condition, threshold=payload.threshold, note=payload.note)
    db.add(t); db.commit(); db.refresh(t)
    return {"id": str(t.id), "symbol": t.symbol, "condition": t.condition,
            "threshold": float(t.threshold), "active": t.active}


@router.delete("/triggers/{trid}", status_code=204)
def delete_trigger(trid: uuid.UUID, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> None:
    t = db.get(EntryTrigger, trid)
    if not t or t.user_id != user.id:
        raise HTTPException(404, "trigger not found")
    db.delete(t); db.commit()


# ================================================================ TRACK RECORD

@router.get("/track-record")
def track_record_endpoint(db: Session = Depends(get_db)) -> dict:
    """Public — the platform's own batting average. No auth needed (anonymized)."""
    return track_record.summary(db)


# ================================================================== REGIME

@router.get("/regime")
def regime_endpoint() -> dict:
    """Public — top-down market regime (risk-on / neutral / risk-off) the user
    should size every trade against. Cached 30 min server-side."""
    return regime_svc.assess()
