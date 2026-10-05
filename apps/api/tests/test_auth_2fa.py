"""TOTP 2FA: enrolment only takes effect after a verified code."""
from __future__ import annotations

import pyotp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401  (registers every table)
from app.db.base import Base
from app.db.session import get_db
from app.routers import auth

EMAIL, PASSWORD = "two.factor@stockplatform.dev", "correct horse battery staple"


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)

    def _db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(auth.router)
    app.dependency_overrides[get_db] = _db
    c = TestClient(app)
    tokens = c.post("/auth/register", json={"email": EMAIL, "password": PASSWORD}).json()
    c.headers["Authorization"] = f"Bearer {tokens['access']}"
    yield c
    engine.dispose()


def login(c, code=None):
    return c.post("/auth/login", json={"email": EMAIL, "password": PASSWORD, "totp_code": code})


def test_abandoned_setup_does_not_lock_the_user_out(client):
    assert client.post("/auth/2fa/setup").status_code == 200
    assert client.get("/auth/me").json()["totp_enabled"] is False
    assert login(client).status_code == 200


def test_verify_activates_2fa_and_login_then_needs_a_code(client):
    secret = client.post("/auth/2fa/setup").json()["secret"]
    assert client.post("/auth/2fa/verify", json={"code": "000000"}).status_code == 401
    assert client.get("/auth/me").json()["totp_enabled"] is False

    r = client.post("/auth/2fa/verify", json={"code": pyotp.TOTP(secret).now()})
    assert r.json() == {"ok": True, "totp_enabled": True}
    assert client.get("/auth/me").json()["totp_enabled"] is True

    assert login(client).json()["detail"] == "totp_required"
    assert login(client, "000000").status_code == 401
    assert login(client, pyotp.TOTP(secret).now()).status_code == 200


def test_setup_again_replaces_a_pending_secret_but_not_an_active_one(client):
    first = client.post("/auth/2fa/setup").json()["secret"]
    second = client.post("/auth/2fa/setup").json()["secret"]
    assert first != second
    assert client.post("/auth/2fa/verify", json={"code": pyotp.TOTP(first).now()}).status_code == 401

    client.post("/auth/2fa/verify", json={"code": pyotp.TOTP(second).now()})
    assert client.post("/auth/2fa/setup").status_code == 409


def test_disable_needs_a_valid_code_and_turns_2fa_off(client):
    secret = client.post("/auth/2fa/setup").json()["secret"]
    client.post("/auth/2fa/verify", json={"code": pyotp.TOTP(secret).now()})

    assert client.post("/auth/2fa/disable", json={"code": "000000"}).status_code == 401
    r = client.post("/auth/2fa/disable", json={"code": pyotp.TOTP(secret).now()})
    assert r.json() == {"ok": True, "totp_enabled": False}
    assert login(client).status_code == 200
