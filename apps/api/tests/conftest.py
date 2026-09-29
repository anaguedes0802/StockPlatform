"""Shared test fixtures."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.services import gate_log


@pytest.fixture(autouse=True)
def _isolated_gate_log(monkeypatch):
    """Route the AI-gate forward-test log to a throwaway in-memory DB.

    Any test that exercises the LLM gate would otherwise write rows into the
    developer's real database.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    monkeypatch.setattr(gate_log, "_session_factory", factory)
    monkeypatch.setattr(gate_log, "_ready_binds", set())
    yield factory
    engine.dispose()
