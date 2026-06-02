from __future__ import annotations

from fastapi import APIRouter, Query

from app.services import news as news_svc
from app.services import news_intel as ni

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/market")
def market(limit: int = Query(30, ge=1, le=100)) -> list[dict]:
    return news_svc.fetch_market_news(limit=limit)


@router.get("/symbol/{symbol}")
def by_symbol(symbol: str, limit: int = Query(25, ge=1, le=100)) -> list[dict]:
    return news_svc.fetch_news(symbol.upper(), limit=limit)


@router.get("/symbol/{symbol}/sentiment")
def sentiment_summary(symbol: str, window_n: int = 25) -> dict:
    agg = news_svc.aggregate_sentiment(symbol.upper(), window_n=window_n)
    return {"symbol": symbol.upper(), **agg}


@router.get("/trending")
def trending() -> list[dict]:
    return news_svc.trending_symbols()


@router.get("/catalysts")
def catalysts(limit: int = Query(40, ge=1, le=100)) -> list[dict]:
    """Cross-entity deal/investment feed: 'a heavyweight just did something for
    THIS stock'. Routes 'ACTOR <invests in / acquires / partners with> TARGET'
    headlines to the affected (target) ticker — for any major actor, not just
    Nvidia."""
    return ni.market_cross_entity_catalysts(limit=limit)


@router.get("/symbol/{symbol}/catalysts")
def symbol_catalysts(symbol: str, limit: int = Query(40, ge=1, le=100)) -> list[dict]:
    """Cross-entity catalysts where this symbol is the beneficiary."""
    from app.services.news import fetch_market_news
    arts = fetch_market_news(limit=limit) + news_svc.fetch_news(symbol.upper(), limit=limit)
    return ni.cross_entity_for_symbol(symbol.upper(), arts)
