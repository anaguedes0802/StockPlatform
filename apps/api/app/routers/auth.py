from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import pyotp
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models import RefreshToken, User
from app.db.session import get_db
from app.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------- schemas ----------


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    totp_code: str | None = None


class TokenPair(BaseModel):
    access: str
    refresh: str
    token_type: str = "bearer"


class RefreshIn(BaseModel):
    refresh: str


class MeOut(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    totp_enabled: bool


class TotpSetupOut(BaseModel):
    secret: str
    otpauth_url: str


class TotpVerifyIn(BaseModel):
    code: str


# ---------- helpers ----------


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _store_refresh(db: Session, user_id: uuid.UUID, token: str) -> None:
    expires = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_days)
    db.add(RefreshToken(user_id=user_id, token_hash=_hash(token), expires_at=expires))
    db.commit()


def _issue_pair(db: Session, user_id: uuid.UUID) -> TokenPair:
    access = create_access_token(user_id)
    refresh = create_refresh_token(user_id)
    _store_refresh(db, user_id, refresh)
    return TokenPair(access=access, refresh=refresh)


# ---------- routes ----------


@router.post("/register", response_model=TokenPair, status_code=201)
def register(payload: RegisterIn, db: Session = Depends(get_db)) -> TokenPair:
    if db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(409, "email already registered")
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _issue_pair(db, user.id)


@router.post("/login", response_model=TokenPair)
def login(payload: LoginIn, db: Session = Depends(get_db)) -> TokenPair:
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    if user.totp_enabled:
        if not payload.totp_code:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "totp_required")
        if not pyotp.TOTP(user.totp_secret).verify(payload.totp_code, valid_window=1):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid totp code")
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return _issue_pair(db, user.id)


@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshIn, db: Session = Depends(get_db)) -> TokenPair:
    """Rotating refresh: validate hash + not-revoked, then revoke and issue a new pair."""
    try:
        data = decode_token(payload.refresh)
    except ValueError as e:
        raise HTTPException(401, str(e)) from e
    if data.get("type") != "refresh":
        raise HTTPException(401, "wrong token type")

    rt = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == _hash(payload.refresh)))
    if not rt:
        raise HTTPException(401, "refresh token not recognized")
    if rt.revoked_at is not None:
        raise HTTPException(401, "refresh token already used")
    if rt.expires_at < datetime.now(timezone.utc):
        raise HTTPException(401, "refresh token expired")

    uid = rt.user_id
    rt.revoked_at = datetime.now(timezone.utc)
    db.commit()
    if not db.get(User, uid):
        raise HTTPException(401, "user not found")
    return _issue_pair(db, uid)


@router.post("/logout", status_code=204)
def logout(payload: RefreshIn, db: Session = Depends(get_db)) -> None:
    rt = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == _hash(payload.refresh)))
    if rt and rt.revoked_at is None:
        rt.revoked_at = datetime.now(timezone.utc)
        db.commit()


@router.get("/me", response_model=MeOut)
def me(user: User = Depends(get_current_user)) -> MeOut:
    return MeOut(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        totp_enabled=user.totp_enabled,
    )


# ---------- 2FA ----------


@router.post("/2fa/setup", response_model=TotpSetupOut)
def totp_setup(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TotpSetupOut:
    """Generate a TOTP secret and return its otpauth URL.

    The secret is only pending: logins keep working without a code until
    /2fa/verify confirms the authenticator, so an abandoned setup can't lock the
    user out. Calling setup again replaces a pending secret.
    """
    if user.totp_enabled:
        raise HTTPException(409, "2FA already enabled — disable first to re-provision")
    secret = pyotp.random_base32()
    user.totp_secret = secret
    db.commit()
    otpauth_url = pyotp.TOTP(secret).provisioning_uri(name=user.email, issuer_name="StockPlatform")
    return TotpSetupOut(secret=secret, otpauth_url=otpauth_url)


@router.post("/2fa/verify", status_code=200)
def totp_verify(
    payload: TotpVerifyIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Verify a TOTP code. Activates 2FA; required on subsequent logins."""
    if not user.totp_secret:
        raise HTTPException(400, "2FA not set up — call /2fa/setup first")
    if not pyotp.TOTP(user.totp_secret).verify(payload.code, valid_window=1):
        raise HTTPException(401, "invalid code")
    user.totp_enabled = True
    db.commit()
    return {"ok": True, "totp_enabled": True}


@router.post("/2fa/disable", status_code=200)
def totp_disable(
    payload: TotpVerifyIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not user.totp_enabled:
        # Nothing to disable; drop any pending enrolment.
        user.totp_secret = None
        db.commit()
        return {"ok": True, "totp_enabled": False}
    if not pyotp.TOTP(user.totp_secret).verify(payload.code, valid_window=1):
        raise HTTPException(401, "invalid code")
    user.totp_secret = None
    user.totp_enabled = False
    db.commit()
    return {"ok": True, "totp_enabled": False}
