from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.rate_limit import per_ip_limiter
from app.schemas.stocks import (
    HistoryResponse,
    IndicatorsResponse,
    KeyStats,
    Quote,
    SearchHit,
    StockDetail,
    StockProfile,
)
from app.services import analysts as analyst_svc
from app.services import earnings as earnings_svc
from app.services import indicators as ind
from app.services import market_data as md
from app.services import options_flow as opt_svc
from app.services import price_action as pa

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/search", response_model=list[SearchHit], dependencies=[Depends(per_ip_limiter(120, 2.0))])
def search(
    q: str = Query("", max_length=64),
    limit: int = Query(20, ge=1, le=100),
    asset_class: str | None = Query(None, max_length=16),
    sector: str | None = Query(None, max_length=64),
) -> list[SearchHit]:
    # Search is purely in-memory — sync is fine.
    return [SearchHit(**hit) for hit in md.search_universe(q, limit, asset_class, sector)]


@router.get("/_universe/sectors")
def sectors() -> list[str]:
    return md.list_sectors()


@router.get("/_universe/asset-classes")
def asset_classes() -> list[str]:
    return md.list_asset_classes()


@router.get("/{symbol}", response_model=StockDetail)
async def detail(symbol: str) -> StockDetail:
    """Quote + profile + key stats. Each sub-call is independently fault-tolerant
    so one broken provider doesn't 500 the whole endpoint."""
    symbol = symbol.upper()
    async def _safe(coro_factory):
        try:
            return await asyncio.to_thread(coro_factory, symbol)
        except Exception:
            return None
    profile, quote, stats = await asyncio.gather(
        _safe(md.get_profile), _safe(md.get_quote), _safe(md.get_key_stats),
    )
    profile = profile or {"symbol": symbol, "name": symbol}
    quote   = quote   or {"symbol": symbol, "price": 0.0}
    stats   = stats   or {}
    if (quote.get("price") or 0) == 0 and not profile.get("name"):
        raise HTTPException(404, f"symbol not found: {symbol}")
    return StockDetail(
        profile=StockProfile(**profile),
        quote=Quote(**quote),
        key_stats=KeyStats(**stats),
    )


@router.get("/{symbol}/history", response_model=HistoryResponse)
async def history(
    symbol: str,
    interval: str = Query("1d"),
    range: str = Query("1y", alias="range"),
) -> HistoryResponse:
    symbol = symbol.upper()
    bars = await asyncio.to_thread(md.get_history_bars, symbol, interval, range)
    if not bars:
        raise HTTPException(404, "no data")
    return HistoryResponse(symbol=symbol, interval=interval, bars=bars)


@router.get("/{symbol}/indicators", response_model=IndicatorsResponse)
async def get_indicators(
    symbol: str,
    names: str = Query("sma_20,sma_50,rsi_14,macd,bb_20", description="comma separated"),
    interval: str = Query("1d"),
    range: str = Query("1y", alias="range"),
) -> IndicatorsResponse:
    symbol = symbol.upper()
    df = await asyncio.to_thread(md.get_history, symbol, interval, range)
    if df.empty:
        raise HTTPException(404, "no data")
    requested = [n.strip() for n in names.split(",") if n.strip()]

    def _compute_all() -> dict[str, list[float | None]]:
        out: dict[str, list[float | None]] = {}
        for n in requested:
            try:
                out.update(ind.compute(n, df))
            except KeyError:
                continue
        return out

    out = await asyncio.to_thread(_compute_all)
    return IndicatorsResponse(
        symbol=symbol,
        interval=interval,
        indicators=out,
        ts=[ts.to_pydatetime() for ts in df.index],
    )


@router.get("/{symbol}/news")
async def news(symbol: str, limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    return await asyncio.to_thread(md.get_news, symbol.upper(), limit)


@router.get("/{symbol}/earnings")
async def earnings(symbol: str, limit: int = Query(8, ge=1, le=20)) -> dict:
    sym = symbol.upper()
    dates, nxt, surp = await asyncio.gather(
        asyncio.to_thread(earnings_svc.earnings_dates, sym, limit),
        asyncio.to_thread(earnings_svc.next_earnings, sym),
        asyncio.to_thread(earnings_svc.consensus_surprise_signal, sym),
    )
    return {"symbol": sym, "dates": dates, "next": nxt, "surprise_signal": surp}


@router.get("/{symbol}/analysts")
async def analysts(symbol: str) -> dict:
    """Analyst consensus + revisions: forward EPS/revenue estimates, recent
    upgrades/downgrades, and a combined signal score."""
    sym = symbol.upper()
    estimate, revisions, signal = await asyncio.gather(
        asyncio.to_thread(analyst_svc.earnings_estimate, sym),
        asyncio.to_thread(analyst_svc.revisions_history, sym, 25),
        asyncio.to_thread(analyst_svc.analyst_signal, sym),
    )
    return {"symbol": sym, "estimate": estimate, "revisions": revisions, "signal": signal}


@router.get("/{symbol}/transcripts")
async def transcripts(symbol: str) -> dict:
    """Earnings call transcripts from Motley Fool (free scrape)."""
    from app.services import transcripts as tr_svc
    sym = symbol.upper()
    entries, signal = await asyncio.gather(
        asyncio.to_thread(tr_svc.list_transcripts, sym, 6),
        asyncio.to_thread(tr_svc.transcript_signal, sym),
    )
    return {"symbol": sym, "entries": entries, "signal": signal}


@router.get("/{symbol}/options")
async def options(symbol: str) -> dict:
    """Unusual options activity scan via yfinance (free, no key)."""
    return await asyncio.to_thread(opt_svc.unusual_activity, symbol.upper())


@router.get("/{symbol}/price-action")
async def price_action(
    symbol: str,
    interval: str = Query("1d"),
    range: str = Query("1y", alias="range"),
    swing_window: int = Query(3, ge=1, le=10),
) -> dict:
    """Smart Money Concepts analysis: swings, FVGs, order blocks, BOS/CHOCH,
    liquidity sweeps, demand/supply zones, and a confluence score for the latest bar.
    """
    df = await asyncio.to_thread(md.get_history, symbol.upper(), interval, range)
    if df.empty:
        raise HTTPException(404, "no data")
    return await asyncio.to_thread(pa.analyze, df, swing_window)
