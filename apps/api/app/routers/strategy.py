from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtest.engine import run_backtest
from app.db.models import Backtest, Strategy, User
from app.db.session import get_db
from app.deps import get_current_user

router = APIRouter(tags=["strategy"])


class StrategyIn(BaseModel):
    name: str
    dsl: dict


class BacktestIn(BaseModel):
    symbol: str
    start: date | None = None
    end: date | None = None
    initial_cash: float = 10_000.0


class QuickBacktestIn(BaseModel):
    symbol: str
    dsl: dict
    start: date | None = None
    end: date | None = None
    initial_cash: float = 10_000.0


@router.post("/strategies", status_code=201, response_model=dict)
def create_strategy(payload: StrategyIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    s = Strategy(user_id=user.id, name=payload.name, dsl=payload.dsl)
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"id": str(s.id), "name": s.name}


@router.get("/strategies", response_model=list[dict])
def list_strategies(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    rows = db.scalars(select(Strategy).where(Strategy.user_id == user.id)).all()
    return [{"id": str(s.id), "name": s.name, "dsl": s.dsl} for s in rows]


@router.post("/backtest/run", response_model=dict)
def run_quick(payload: QuickBacktestIn) -> dict:
    """Run an ad-hoc backtest without saving — used by the Strategy Lab UI."""
    try:
        return run_backtest(
            payload.symbol.upper(),
            payload.dsl,
            start=payload.start.isoformat() if payload.start else None,
            end=payload.end.isoformat() if payload.end else None,
            initial_cash=payload.initial_cash,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/strategies/{sid}/backtest", response_model=dict)
def run(
    sid: uuid.UUID,
    payload: BacktestIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    s = db.get(Strategy, sid)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "strategy not found")
    try:
        result = run_backtest(
            payload.symbol.upper(),
            s.dsl,
            start=payload.start.isoformat() if payload.start else None,
            end=payload.end.isoformat() if payload.end else None,
            initial_cash=payload.initial_cash,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    bt = Backtest(
        strategy_id=s.id,
        symbol=payload.symbol.upper(),
        start_date=payload.start or date.today(),
        end_date=payload.end or date.today(),
        initial_cash=payload.initial_cash,
        metrics=result["metrics"],
        equity_curve=result["equity_curve"],
        trades=result["trades"],
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)
    return {"id": str(bt.id), **result}
