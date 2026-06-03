"""Tests for the catalyst taxonomy — the high-signal 'smart-money validation'
categories and the rumor-vs-fact substantiation discount."""
from __future__ import annotations

from app.services import news_intel as ni


def test_executive_endorsement_classified_and_substantiated() -> None:
    """A peer-CEO endorsement made on stage should be tagged
    executive_endorsement / bullish / official (not 'noise' or 'partnership')."""
    art = {
        "title": "Nvidia CEO Jensen Huang calls Marvell the next trillion-dollar company on stage",
        "summary": "Official remarks at the keynote; Nvidia has invested $2B in Marvell this year.",
        "url": "http://x/1",
        "published_at": None,
    }
    intel = ni.classify_articles("MRVL", [art])[0]["intel"]
    assert intel["category"] == "executive_endorsement"
    assert intel["direction"] == "bullish"
    assert intel["substantiation"] in ("confirmed", "official")


def test_strategic_investment_classified() -> None:
    art = {
        "title": "Nvidia takes a $2B equity stake in chipmaker",
        "summary": "Strategic investment announced.",
        "url": "http://x/2",
        "published_at": None,
    }
    intel = ni.classify_articles("MRVL", [art])[0]["intel"]
    assert intel["category"] == "strategic_investment"
    assert intel["direction"] == "bullish"


def test_rumor_is_discounted_below_confirmed() -> None:
    """The same high-materiality category should contribute less to the
    aggregate when it is an unconfirmed rumor than when it is confirmed."""
    rumor = [{
        "intel": {"category": "acquisition_target", "materiality": 0.9,
                  "direction": "bullish", "substantiation": "rumor"},
        "published_at": None,
    }]
    confirmed = [{
        "intel": {"category": "acquisition_target", "materiality": 0.9,
                  "direction": "bullish", "substantiation": "confirmed"},
        "published_at": None,
    }]
    s_rumor = ni.catalyst_score(rumor)["max_materiality"]
    s_conf = ni.catalyst_score(confirmed)["max_materiality"]
    assert s_rumor < s_conf
    # rumor multiplier is 0.55 → 0.9 * 0.55 ≈ 0.495
    assert abs(s_rumor - 0.495) < 1e-6


def test_noise_recap_stays_noise() -> None:
    art = {"title": "10 stocks to watch this week", "summary": "recap",
           "url": "http://x/3", "published_at": None}
    intel = ni.classify_articles("AAPL", [art])[0]["intel"]
    assert intel["category"] == "noise"
    assert intel["materiality"] <= 0.2


# --- cross-entity routing ---------------------------------------------------

def test_cross_entity_routes_investment_to_target() -> None:
    """'Nvidia invests $2B in Marvell' must route to the TARGET (MRVL), with
    Nvidia as the actor — not filed under NVDA."""
    arts = [{"title": "Nvidia invests $2 billion in Marvell to expand custom chips",
             "url": "u1", "published_at": None}]
    cats = ni.extract_cross_entity_catalysts(arts, use_llm=False)
    assert len(cats) == 1
    c = cats[0]
    assert c["target_symbol"] == "MRVL"
    assert c["actor_symbol"] == "NVDA"
    assert c["category"] == "strategic_investment"
    assert c["direction"] == "bullish"


def test_cross_entity_works_for_any_actor() -> None:
    """Not Nvidia-specific: a Microsoft deal and an Apple supplier pick both route."""
    arts = [
        {"title": "Microsoft signs multi-year cloud deal with CoreWeave", "url": "u2", "published_at": None},
        {"title": "Apple picks Broadcom to supply 5G chips", "url": "u3", "published_at": None},
    ]
    cats = ni.extract_cross_entity_catalysts(arts, use_llm=False)
    routes = {(c["actor_symbol"], c["target_symbol"]) for c in cats}
    assert ("MSFT", "CRWV") in routes
    assert ("AAPL", "AVGO") in routes


def test_cross_entity_acquisition_routes_to_target() -> None:
    arts = [{"title": "Berkshire Hathaway to acquire Oracle", "url": "u4", "published_at": None}]
    cats = ni.extract_cross_entity_catalysts(arts, use_llm=False)
    assert cats[0]["actor_symbol"] == "BRK-B"
    assert cats[0]["target_symbol"] == "ORCL"
    assert cats[0]["category"] == "acquisition_target"


def test_cross_entity_ignores_non_relational_headlines() -> None:
    """A headline mentioning two companies but no deal relation emits nothing."""
    arts = [{"title": "Nvidia and AMD both rallied today on chip optimism",
             "url": "u5", "published_at": None}]
    assert ni.extract_cross_entity_catalysts(arts, use_llm=False) == []


def test_cross_entity_for_symbol_filters_to_beneficiary() -> None:
    arts = [
        {"title": "Nvidia invests $2 billion in Marvell", "url": "u1", "published_at": None},
        {"title": "Apple picks Broadcom to supply chips", "url": "u3", "published_at": None},
    ]
    only_mrvl = ni.cross_entity_for_symbol("MRVL", arts)
    assert len(only_mrvl) == 1
    assert only_mrvl[0]["target_symbol"] == "MRVL"


def test_resolve_company_maps_names_to_tickers() -> None:
    assert ni._resolve_company("Nvidia") == "NVDA"
    assert ni._resolve_company("Marvell Technology, Inc.") == "MRVL"
    assert ni._resolve_company("Berkshire Hathaway") == "BRK-B"
    assert ni._resolve_company("AAPL") == "AAPL"          # already a ticker
    assert ni._resolve_company("a tiny unknown bakery") is None


def test_cross_entity_llm_layer_catches_paraphrase(monkeypatch) -> None:
    """The LLM pass should route deals the regex misses (paraphrased verbs),
    resolving the names it returns to tickers."""
    class _Res:
        provider = "stub"
        json = {"results": [
            {"i": 0, "actor": "Microsoft", "target": "Snowflake",
             "category": "partnership", "substantiation": "confirmed"},
        ]}
        text = "ok"

    monkeypatch.setattr(ni.llm, "is_available", lambda: True)
    monkeypatch.setattr(ni.llm, "generate", lambda *a, **k: _Res())

    # A headline whose phrasing the regex relation patterns do NOT catch.
    arts = [{"title": "Snowflake deepens its alliance, powered by Microsoft Azure",
             "url": "u9", "published_at": None}]
    # regex alone finds nothing here…
    assert ni.extract_cross_entity_catalysts(arts, use_llm=False) == []
    # …but the LLM layer routes MSFT -> SNOW.
    cats = ni.extract_cross_entity_catalysts(arts, use_llm=True)
    routes = {(c["actor_symbol"], c["target_symbol"]) for c in cats}
    assert ("MSFT", "SNOW") in routes
    assert any(c["_source"] == "llm" for c in cats)
