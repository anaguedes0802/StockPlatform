"""Tests for narrative reconciliation — the agentic driver identification +
driver-health + thesis-conflict reconciliation.

Hermetic: the LLM and the shared opinion cache are stubbed so tests exercise
the deterministic reconciliation logic, the offline fallback, and (with a fake
LLM) the agentic path — without any network."""
from __future__ import annotations

import time
import types

import pytest

from app.services import narrative as nv


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No real LLM, no shared cache, no symbol-fetch network in unit tests."""
    monkeypatch.setattr(nv, "_cache_get", lambda k: None)
    monkeypatch.setattr(nv, "_cache_set", lambda k, v, ttl: None)
    monkeypatch.setattr(nv, "_symbol_fetchable", lambda s: True)
    nv._TREND_CACHE.clear()
    nv._SYMBOL_OK.clear()
    yield
    nv._TREND_CACHE.clear()
    nv._SYMBOL_OK.clear()


def _force_offline(monkeypatch):
    monkeypatch.setattr(nv.llm, "is_available", lambda: False)


def _stub_driver(symbol: str, *, ret_30d_pct: float, above_sma50: bool, health: float) -> None:
    nv._TREND_CACHE[symbol] = (
        time.time(),
        {"symbol": symbol, "last": 100.0, "ret_30d_pct": ret_30d_pct,
         "above_sma50": above_sma50, "health": health},
    )


def _fake_llm(monkeypatch, payload: dict):
    """Force the LLM on and make generate() return a canned JSON payload."""
    monkeypatch.setattr(nv.llm, "is_available", lambda: True)
    res = types.SimpleNamespace(json=payload, text="x", provider="fake")
    monkeypatch.setattr(nv.llm, "generate", lambda *a, **k: res)


# --- offline fallback path --------------------------------------------------

def test_unmapped_symbol_is_neutral(monkeypatch) -> None:
    _force_offline(monkeypatch)
    a = nv.assess("AAPL", articles=[])
    assert a.theme is None
    assert a.multiplier == 1.0


def test_broken_driver_haircuts_score(monkeypatch) -> None:
    _force_offline(monkeypatch)
    _stub_driver("BTC-USD", ret_30d_pct=-18.0, above_sma50=False, health=-1.0)
    a = nv.assess("MSTR", articles=[])
    assert a.theme == "Bitcoin"
    assert a.driver_health == -1.0
    assert a.multiplier < 0.5
    assert any("contradicts" in n for n in a.notes)


def test_strong_driver_gives_small_boost(monkeypatch) -> None:
    _force_offline(monkeypatch)
    _stub_driver("BTC-USD", ret_30d_pct=15.0, above_sma50=True, health=1.0)
    a = nv.assess("MARA", articles=[])
    assert 1.0 < a.multiplier <= 1.10


def test_keyword_conflict_flag_and_haircut(monkeypatch) -> None:
    _force_offline(monkeypatch)
    _stub_driver("BTC-USD", ret_30d_pct=-5.0, above_sma50=False, health=-0.3)
    articles = [{"title": "Strategy sold 39 bitcoin last week, filing shows",
                 "summary": "The company trimmed its BTC holdings.", "url": "http://x/1"}]
    a = nv.assess("MSTR", articles=articles)
    assert a.conflicts and a.conflicts[0]["matched"]
    assert a.multiplier < 0.5
    assert any("Narrative conflict" in n for n in a.notes)


def test_supportive_bitcoin_news_does_not_trip_conflict(monkeypatch) -> None:
    _force_offline(monkeypatch)
    _stub_driver("BTC-USD", ret_30d_pct=2.0, above_sma50=True, health=0.2)
    articles = [{"title": "Strategy buys more bitcoin, adds to treasury",
                 "summary": "Company acquired additional BTC.", "url": "http://x/2"}]
    a = nv.assess("MSTR", articles=articles)
    assert not a.conflicts


# --- agentic LLM path -------------------------------------------------------

def test_llm_identifies_driver_for_uncurated_name(monkeypatch) -> None:
    """A name NOT in the fallback map still gets a driver from the LLM agent."""
    _fake_llm(monkeypatch, {
        "has_driver": True, "driver_name": "Bitcoin", "driver_symbol": "BTC-USD",
        "relationship": "proxy", "strength": 0.95,
        "thesis": "Bitcoin treasury proxy", "conflict_criteria": "selling its bitcoin",
    })
    _stub_driver("BTC-USD", ret_30d_pct=-18.0, above_sma50=False, health=-1.0)
    # Conflict pass uses the same fake generate(); return no breaking headlines.
    a = nv.assess("ZZZZ", articles=[])  # symbol not in fallback map
    assert a.source == "llm"
    assert a.driver_label == "Bitcoin"
    assert a.multiplier < 0.5


def test_llm_reports_no_driver(monkeypatch) -> None:
    _fake_llm(monkeypatch, {"has_driver": False, "relationship": "none",
                            "thesis": "diversified software"})
    a = nv.assess("CRM", articles=[])
    assert a.theme is None
    assert a.multiplier == 1.0


def test_input_cost_relationship_inverts(monkeypatch) -> None:
    """For an input-cost driver, a FALLING input cost is bullish (health flips)."""
    _fake_llm(monkeypatch, {
        "has_driver": True, "driver_name": "crude oil", "driver_symbol": "USO",
        "relationship": "input_cost", "strength": 0.6,
        "thesis": "airline; fuel is a major cost", "conflict_criteria": "oil spikes",
    })
    _stub_driver("USO", ret_30d_pct=-20.0, above_sma50=False, health=-1.0)
    a = nv.assess("AAL", articles=[])
    # health -1 input_cost → effective +0.6 → a boost, not a haircut
    assert a.multiplier > 1.0


def test_health_to_multiplier_endpoints() -> None:
    assert nv._health_to_multiplier(0.0) == 1.0
    assert abs(nv._health_to_multiplier(1.0) - 1.10) < 1e-9
    assert abs(nv._health_to_multiplier(-1.0) - 0.45) < 1e-9
