"""Tests for the discovery API (opportunities + catalyst-news).

Fully offline: the screener/opportunity engine and the news_intel classifier
are monkeypatched, and the Redis cache helpers are stubbed to no-ops so we test
the route logic + contract shape without touching the network or Redis.

We mount the discovery router on a bare FastAPI app (no lifespan) so the
background warmup scheduler never starts.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import discovery


@pytest.fixture()
def client(monkeypatch):
    # Disable Redis caching entirely so each request recomputes (and so we
    # never depend on a live Redis in CI).
    monkeypatch.setattr(discovery, "_cache_get", lambda key: None)
    monkeypatch.setattr(discovery, "_cache_set", lambda key, value, ttl: None)

    app = FastAPI()
    app.include_router(discovery.router)
    return TestClient(app)


def _fake_scan(limit: int = 30):
    return {
        "as_of": "2026-06-02T00:00:00+00:00",
        "opportunities": [
            {
                "symbol": "MRVL", "name": "Marvell Technology",
                "score": 0.82, "best_strategy": "catalyst_plays",
                "last_price": 88.5, "momentum_1m_pct": 12.3,
                "rationale": ["Material strategic investment (materiality 0.80)",
                              "1M momentum +12.3%"],
            },
            {
                "symbol": "IREN", "name": "Iris Energy",
                "score": 0.71, "best_strategy": "rising_stars",
                "last_price": 14.2, "momentum_1m_pct": 9.1,
                "rationale": ["3M momentum +30%", "Volume 2.1σ above 20-day avg"],
            },
            {
                "symbol": "BADSYM", "name": None,
                "score": 0.40, "best_strategy": "breakouts",
                "last_price": None, "momentum_1m_pct": None,
                "rationale": [],
            },
        ],
    }


def _fake_classify(symbol: str, days_window: int = 14):
    return {
        "symbol": symbol,
        "articles": [
            {
                "title": f"{symbol} lands marquee AI customer",
                "url": f"https://example.com/{symbol}/1",
                "source": "Reuters",
                "published_at": "2026-06-01T12:00:00+00:00",
                "intel": {"category": "major_customer_win", "materiality": 0.7,
                          "direction": "bullish", "rationale": "anchor customer win"},
            },
            {
                "title": f"{symbol} stocks to watch this week",
                "url": f"https://example.com/{symbol}/2",
                "source": "Blog",
                "published_at": "2026-05-30T12:00:00+00:00",
                # noise — must be filtered out
                "intel": {"category": "noise", "materiality": 0.1,
                          "direction": "neutral", "rationale": "recap"},
            },
        ],
    }


# ---------- opportunities ----------

def test_opportunities_contract_shape(client, monkeypatch):
    monkeypatch.setattr(discovery.opp_svc, "scan", _fake_scan)

    r = client.get("/discovery/opportunities?limit=12")
    assert r.status_code == 200
    body = r.json()
    assert "as_of" in body and isinstance(body["as_of"], str)
    assert isinstance(body["opportunities"], list)
    assert len(body["opportunities"]) == 3

    top = body["opportunities"][0]
    assert set(top.keys()) == {
        "symbol", "name", "score", "strategy",
        "last_price", "change_pct", "reasons",
    }
    # Sorted best-first.
    assert top["symbol"] == "MRVL"
    assert top["strategy"] == "catalyst_plays"
    assert isinstance(top["score"], float)
    assert isinstance(top["reasons"], list) and top["reasons"]
    # Nulls tolerated for the bad symbol row.
    bad = body["opportunities"][2]
    assert bad["last_price"] is None and bad["change_pct"] is None


def test_opportunities_respects_limit(client, monkeypatch):
    monkeypatch.setattr(discovery.opp_svc, "scan", _fake_scan)
    r = client.get("/discovery/opportunities?limit=2")
    assert r.status_code == 200
    assert len(r.json()["opportunities"]) == 2


def test_opportunities_fail_open_on_scan_error(client, monkeypatch):
    def _boom(limit=30):
        raise RuntimeError("universe scan blew up")
    monkeypatch.setattr(discovery.opp_svc, "scan", _boom)

    r = client.get("/discovery/opportunities")
    assert r.status_code == 200
    body = r.json()
    assert body["opportunities"] == []
    assert isinstance(body["as_of"], str)


# ---------- catalyst news ----------

def test_catalyst_news_contract_shape(client, monkeypatch):
    monkeypatch.setattr(discovery.opp_svc, "scan", _fake_scan)
    monkeypatch.setattr(discovery.ni, "classify_and_score", _fake_classify)

    r = client.get("/discovery/catalyst-news?limit=20")
    assert r.status_code == 200
    body = r.json()
    assert "as_of" in body and isinstance(body["as_of"], str)
    assert isinstance(body["items"], list) and body["items"]

    item = body["items"][0]
    assert set(item.keys()) == {
        "symbol", "title", "source", "url",
        "published_at", "sentiment", "summary",
    }
    # Noise items must have been filtered out.
    assert all("stocks to watch" not in (it["title"] or "") for it in body["items"])
    # No duplicate URLs.
    urls = [it["url"] for it in body["items"]]
    assert len(urls) == len(set(urls))


def test_catalyst_news_fail_open_on_classify_error(client, monkeypatch):
    monkeypatch.setattr(discovery.opp_svc, "scan", _fake_scan)

    def _boom(symbol, days_window=14):
        raise RuntimeError("llm down")
    monkeypatch.setattr(discovery.ni, "classify_and_score", _boom)

    r = client.get("/discovery/catalyst-news")
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert isinstance(body["as_of"], str)
