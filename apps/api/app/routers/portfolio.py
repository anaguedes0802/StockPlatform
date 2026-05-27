from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Portfolio, Transaction, User
from app.db.session import get_db
from app.deps import get_current_user
from app.schemas.portfolio import (
    PortfolioCreate,
    PortfolioDetail,
    PortfolioMetrics,
    Position,
    TransactionCreate,
    TransactionOut,
)
from app.services import market_data as md
from app.services.dividends import dividends_received
from app.services.portfolio_analysis import analyze_portfolio
from app.services.portfolio_review import review_portfolio

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


def _get_owned(db: Session, user: User, pid: uuid.UUID) -> Portfolio:
    p = db.get(Portfolio, pid)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "portfolio not found")
    return p


def _aggregate_positions(transactions: list[Transaction]) -> dict[str, dict[str, Decimal]]:
    """Average-cost. Realized P&L is computed via FIFO in a future iteration."""
    pos: dict[str, dict[str, Decimal]] = defaultdict(lambda: {"qty": Decimal(0), "cost": Decimal(0)})
    for t in transactions:
        p = pos[t.symbol]
        if t.side == "buy":
            p["qty"] += t.quantity
            p["cost"] += t.quantity * t.price + t.fee
        else:
            # reduce avg-cost proportionally
            if p["qty"] > 0:
                cost_per = p["cost"] / p["qty"]
                sell_qty = min(t.quantity, p["qty"])
                p["cost"] -= cost_per * sell_qty
                p["qty"] -= sell_qty
    return {sym: v for sym, v in pos.items() if v["qty"] > 0}


@router.get("", response_model=list[dict])
def list_portfolios(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    rows = db.scalars(select(Portfolio).where(Portfolio.user_id == user.id)).all()
    return [{"id": str(p.id), "name": p.name, "base_currency": p.base_currency} for p in rows]


@router.post("", response_model=dict, status_code=201)
def create_portfolio(
    payload: PortfolioCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    p = Portfolio(user_id=user.id, name=payload.name, base_currency=payload.base_currency)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": str(p.id), "name": p.name, "base_currency": p.base_currency}


@router.get("/{pid}", response_model=PortfolioDetail)
def detail(
    pid: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PortfolioDetail:
    p = _get_owned(db, user, pid)
    agg = _aggregate_positions(p.transactions)

    positions: list[Position] = []
    total_value = 0.0
    total_cost = 0.0
    total_div = 0.0
    sector_alloc: dict[str, float] = defaultdict(float)

    # Group transactions per symbol for dividend calculation
    by_symbol: dict[str, list] = defaultdict(list)
    for t in p.transactions:
        by_symbol[t.symbol].append(t)

    for symbol, data in agg.items():
        qty = data["qty"]
        cost = data["cost"]
        avg = cost / qty if qty else Decimal(0)
        last = float(md.get_quote(symbol).get("price") or 0.0)
        mv = float(qty) * last
        upl = mv - float(cost)
        upl_pct = (upl / float(cost) * 100) if cost else 0.0
        try:
            divs = dividends_received(symbol, by_symbol.get(symbol, []))
        except Exception:
            divs = 0.0
        total_div += divs
        positions.append(
            Position(
                symbol=symbol,
                quantity=qty,
                avg_buy_price=avg.quantize(Decimal("0.00000001")),
                last_price=last,
                market_value=mv,
                unrealized_pl=upl,
                unrealized_pl_pct=upl_pct,
            )
        )
        total_value += mv
        total_cost += float(cost)
        sector = md.get_profile(symbol).get("sector") or "Unknown"
        sector_alloc[sector] += mv

    # normalize sector alloc to share-of-portfolio
    if total_value > 0:
        sector_alloc = {k: round(v / total_value, 4) for k, v in sector_alloc.items()}

    unrealized = total_value - total_cost
    total_return = unrealized + total_div
    metrics = PortfolioMetrics(
        total_value=total_value,
        cost_basis=total_cost,
        unrealized_pl=unrealized,
        unrealized_pl_pct=((unrealized) / total_cost * 100) if total_cost else 0.0,
        realized_pl=0.0,
        total_dividends=round(total_div, 2),
        total_return=round(total_return, 2),
        total_return_pct=((total_return / total_cost * 100) if total_cost else 0.0),
    )

    return PortfolioDetail(
        id=p.id,
        name=p.name,
        base_currency=p.base_currency,
        positions=positions,
        metrics=metrics,
        sector_allocation=dict(sector_alloc),
    )


@router.post("/{pid}/transactions", response_model=TransactionOut, status_code=201)
def add_transaction(
    pid: uuid.UUID,
    payload: TransactionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TransactionOut:
    _get_owned(db, user, pid)
    t = Transaction(
        portfolio_id=pid,
        symbol=payload.symbol.upper(),
        side=payload.side,
        quantity=payload.quantity,
        price=payload.price,
        fee=payload.fee,
        occurred_at=payload.occurred_at,
        note=payload.note,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return TransactionOut.model_validate(t)


@router.get("/{pid}/transactions", response_model=list[TransactionOut])
def list_transactions(
    pid: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[TransactionOut]:
    p = _get_owned(db, user, pid)
    return [TransactionOut.model_validate(t) for t in p.transactions]


@router.post("/{pid}/ai-analysis")
def ai_analysis(
    pid: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Run the AI portfolio analysis on this portfolio's current positions."""
    p = _get_owned(db, user, pid)
    agg = _aggregate_positions(p.transactions)
    positions: list[dict] = []
    for symbol, data in agg.items():
        qty = data["qty"]; cost = data["cost"]
        avg = cost / qty if qty else Decimal(0)
        last = float(md.get_quote(symbol).get("price") or 0.0)
        positions.append({
            "symbol": symbol,
            "quantity": float(qty),
            "avg_buy_price": float(avg),
            "last_price": last,
            "market_value": float(qty) * last,
        })
    return analyze_portfolio(positions, base_currency=p.base_currency)


@router.post("/{pid}/review")
def ai_review(
    pid: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Position-by-position AI review of the portfolio.

    For each holding, surfaces: verdict (HOLD/ADD/TRIM/SELL/STOP_OUT), a
    recommended stop, alerts (drawdown, structure break, earnings, stop-
    approaching, etc.), what's changed since entry, and a short narrative.
    Uses Anthropic (Claude) when ANTHROPIC_API_KEY is set; deterministic
    rule synthesis otherwise. Both paths emit the same shape.
    """
    p = _get_owned(db, user, pid)
    agg = _aggregate_positions(p.transactions)
    by_symbol: dict[str, list] = defaultdict(list)
    for t in p.transactions:
        by_symbol[t.symbol].append(t)
    positions: list[dict] = []
    for symbol, data in agg.items():
        qty = data["qty"]; cost = data["cost"]
        if qty <= 0:
            continue
        avg = cost / qty if qty else Decimal(0)
        positions.append({
            "symbol": symbol,
            "quantity": float(qty),
            "avg_buy_price": float(avg),
        })
    reviews = review_portfolio(positions, dict(by_symbol))
    # Surface top-level alert summary for the UI banner.
    critical = sum(1 for r in reviews for a in r.get("alerts", []) if a.get("severity") == "critical")
    warnings = sum(1 for r in reviews for a in r.get("alerts", []) if a.get("severity") == "warning")
    return {
        "portfolio_id": str(pid),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_positions": len(reviews),
        "n_critical": critical,
        "n_warnings": warnings,
        "reviews": reviews,
    }


@router.delete("/{pid}/transactions/{tid}", status_code=204)
def delete_transaction(
    pid: uuid.UUID,
    tid: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    _get_owned(db, user, pid)
    t = db.get(Transaction, tid)
    if not t or t.portfolio_id != pid:
        raise HTTPException(404, "transaction not found")
    db.delete(t)
    db.commit()
