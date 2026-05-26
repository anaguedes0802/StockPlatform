"""Audit logging.

Two paths:
  1. **Middleware** that writes one row per mutating request (POST/PUT/PATCH/DELETE)
     for routes that touch user-scoped resources (/auth/*, /portfolios/*, /watchlists/*,
     /strategies/*). Best-effort; never blocks the response.
  2. **Explicit `audit()`** helper for in-handler events that don't map 1:1 to a
     request line (e.g., login success vs failure inside the same endpoint).

Writes to the existing `audit_log` table from db/models.py.
"""
from __future__ import annotations

import contextlib
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.db.models import AuditLog
from app.db.session import SessionLocal

# Path prefixes we audit. Anything outside these is skipped to keep volume manageable.
_AUDIT_PREFIXES = ("/auth/", "/portfolios", "/watchlists", "/strategies", "/ai/opinion")
_AUDIT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _extract_user_id(request: Request) -> uuid.UUID | None:
    """Best-effort: peek at the JWT to attach a user_id without doing full auth."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1]
    with contextlib.suppress(Exception):
        from app.core.security import decode_token
        payload = decode_token(token)
        if payload.get("type") == "access":
            return uuid.UUID(payload["sub"])
    return None


def audit(
    db: Session,
    *,
    action: str,
    resource: str | None = None,
    user_id: uuid.UUID | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Best-effort write a single audit row. Never raises."""
    try:
        db.add(AuditLog(
            user_id=user_id,
            action=action[:64],
            resource=(resource or "")[:255] or None,
            ip=ip,
            ua=user_agent,
            payload=payload or None,
        ))
        db.commit()
    except Exception:
        with contextlib.suppress(Exception):
            db.rollback()


class AuditMiddleware(BaseHTTPMiddleware):
    """Write one audit row per mutating request to audited prefixes.

    Status code is recorded in `payload.status`. We never log request bodies
    (to avoid capturing passwords / sensitive data); the (method, path,
    status, user_id, ip, ua) tuple is sufficient for forensics.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Filter: only mutating methods on audited prefixes
        if request.method not in _AUDIT_METHODS:
            return response
        path = request.url.path
        if not any(path.startswith(p) for p in _AUDIT_PREFIXES):
            return response

        try:
            user_id = _extract_user_id(request)
            ip = request.client.host if request.client else None
            ua = request.headers.get("user-agent")
            payload = {"status": response.status_code, "method": request.method}
            with SessionLocal() as db:
                audit(
                    db,
                    action=f"{request.method} {path}",
                    resource=path,
                    user_id=user_id,
                    ip=ip,
                    user_agent=ua,
                    payload=payload,
                )
        except Exception:
            pass
        return response
