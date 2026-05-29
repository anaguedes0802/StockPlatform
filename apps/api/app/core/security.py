from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.config import settings

ALGORITHM = "HS256"


# bcrypt has a hard 72-byte input limit. We truncate at the boundary so longer
# passwords don't raise (matching passlib's prior behavior under older bcrypt
# versions). passlib 1.7.x is broken against bcrypt>=4.1 (it probes a missing
# __about__ attribute) — using bcrypt directly avoids that import-time crash.
def _to_bytes(plain: str) -> bytes:
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_to_bytes(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _encode(payload: dict[str, Any], minutes: int) -> str:
    now = datetime.now(timezone.utc)
    # `jti` (a random token id) guarantees every issued token is unique even
    # when two are minted in the same second. Without it, JWT timestamps are
    # second-granularity, so two logins in the same second produced identical
    # refresh tokens → UNIQUE constraint violation on refresh_tokens.token_hash
    # (intermittent HTTP 500 on sign-in).
    payload = {**payload, "iat": now, "exp": now + timedelta(minutes=minutes),
               "jti": secrets.token_urlsafe(12)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def create_access_token(user_id: uuid.UUID) -> str:
    return _encode({"sub": str(user_id), "type": "access"}, settings.jwt_access_minutes)


def create_refresh_token(user_id: uuid.UUID) -> str:
    return _encode(
        {"sub": str(user_id), "type": "refresh"},
        settings.jwt_refresh_days * 24 * 60,
    )


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    except JWTError as e:
        raise ValueError(f"invalid token: {e}") from e
