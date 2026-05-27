"""Notifications API — list + mark-read + trigger a scan.

GET  /notifications           — recent for current user (default last 30, includes unread + read)
GET  /notifications/unread    — quick count for the bell badge
POST /notifications/scan      — run all scanners and create new notifications
POST /notifications/{id}/read — mark one as read
POST /notifications/read-all  — mark all as read
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_, update
from sqlalchemy.orm import Session

from app.db.models import Notification, User
from app.db.session import get_db
from app.deps import get_current_user
from app.services import notifications as notif_svc

router = APIRouter(prefix="/notifications", tags=["notifications"])


def _to_dict(n: Notification) -> dict[str, Any]:
    return {
        "id": n.id,
        "kind": n.kind,
        "severity": n.severity,
        "symbol": n.symbol,
        "title": n.title,
        "body": n.body,
        "payload": n.payload,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "read_at": n.read_at.isoformat() if n.read_at else None,
    }


@router.get("")
def list_notifications(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict]:
    rows = db.execute(
        select(Notification)
        .where(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc())
        .limit(min(max(limit, 1), 200))
    ).scalars().all()
    return [_to_dict(n) for n in rows]


@router.get("/unread")
def unread_count(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    n = db.scalar(
        select(Notification)
        .where(and_(Notification.user_id == user.id, Notification.read_at.is_(None)))
        .order_by(Notification.created_at.desc())
    )
    cnt = len(db.execute(
        select(Notification.id)
        .where(and_(Notification.user_id == user.id, Notification.read_at.is_(None)))
    ).scalars().all())
    return {
        "count": cnt,
        "latest_kind": n.kind if n else None,
        "latest_symbol": n.symbol if n else None,
    }


@router.post("/scan")
def scan(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Run all scanners. Idempotent within a day (dedupe key collapses repeats)."""
    return notif_svc.scan_all(db, user)


@router.post("/{nid}/read", status_code=204)
def mark_read(
    nid: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    n = db.get(Notification, nid)
    if not n or n.user_id != user.id:
        raise HTTPException(404, "notification not found")
    if not n.read_at:
        n.read_at = datetime.now(timezone.utc)
        db.commit()


@router.post("/read-all", status_code=204)
def mark_all_read(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    db.execute(
        update(Notification)
        .where(and_(Notification.user_id == user.id, Notification.read_at.is_(None)))
        .values(read_at=datetime.now(timezone.utc))
    )
    db.commit()
