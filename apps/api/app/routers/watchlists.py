from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User, Watchlist, WatchlistItem
from app.db.session import get_db
from app.deps import get_current_user

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


class WatchlistCreate(BaseModel):
    name: str


class ItemIn(BaseModel):
    symbol: str


def _owned(db: Session, user: User, wid: uuid.UUID) -> Watchlist:
    w = db.get(Watchlist, wid)
    if not w or w.user_id != user.id:
        raise HTTPException(404, "watchlist not found")
    return w


@router.get("", response_model=list[dict])
def list_watchlists(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    rows = db.scalars(select(Watchlist).where(Watchlist.user_id == user.id)).all()
    return [
        {
            "id": str(w.id),
            "name": w.name,
            "symbols": [i.symbol for i in w.items],
        }
        for w in rows
    ]


@router.post("", status_code=201, response_model=dict)
def create(payload: WatchlistCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    w = Watchlist(user_id=user.id, name=payload.name)
    db.add(w)
    db.commit()
    db.refresh(w)
    return {"id": str(w.id), "name": w.name, "symbols": []}


@router.post("/{wid}/items", status_code=201)
def add_item(
    wid: uuid.UUID,
    payload: ItemIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    _owned(db, user, wid)
    item = WatchlistItem(watchlist_id=wid, symbol=payload.symbol.upper())
    db.merge(item)
    db.commit()
    return {"ok": True}


@router.delete("/{wid}/items/{symbol}", status_code=204)
def remove_item(
    wid: uuid.UUID,
    symbol: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    _owned(db, user, wid)
    db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == wid,
        WatchlistItem.symbol == symbol.upper(),
    ).delete()
    db.commit()
