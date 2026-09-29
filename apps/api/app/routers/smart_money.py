from __future__ import annotations

from fastapi import APIRouter, Query

from app.services import insider as insider_svc
from app.services import institutions as inst_svc
from app.services import politicians as pol_svc
from app.services import social as social_svc

router = APIRouter(tags=["smart-money"])


# --- insiders ---

@router.get("/stocks/{symbol}/insider")
def insider(symbol: str, limit: int = Query(30, ge=1, le=100)) -> dict:
    return {
        "transactions": insider_svc.get_insider_transactions(symbol.upper(), limit=limit),
        "purchases": insider_svc.get_insider_purchases(symbol.upper()),
        "signal": insider_svc.insider_signal(symbol.upper()),
    }


# --- institutional + famous ---

@router.get("/stocks/{symbol}/institutional")
def institutional(symbol: str) -> dict:
    sym = symbol.upper()
    return {
        "institutional_holders": inst_svc.get_institutional_holders(sym),
        "mutualfund_holders": inst_svc.get_mutualfund_holders(sym),
        "major_holders": inst_svc.get_major_holders(sym),
        "famous_holders": inst_svc.famous_investors_for_symbol(sym),
        "signal": inst_svc.institutional_signal(sym),
    }


@router.get("/famous-investors")
def famous_investors() -> list[dict]:
    return inst_svc.list_famous_investors()


# --- politicians ---

@router.get("/stocks/{symbol}/politicians")
def politicians_for_symbol(symbol: str, limit: int = Query(25, ge=1, le=100)) -> dict:
    sym = symbol.upper()
    return {
        "trades": pol_svc.trades_for_symbol(sym, limit=limit),
        "signal": pol_svc.political_signal(sym),
        "source": pol_svc.data_source(),
    }


@router.get("/politicians/recent")
def politicians_recent(limit: int = Query(30, ge=1, le=500),
                       days: int | None = Query(None, ge=1, le=3650)) -> list[dict]:
    return pol_svc.recent_trades(limit=limit, days=days)


@router.get("/politicians/tracker")
def politicians_tracker() -> dict:
    """Congress tracker: archive status + copy-the-politician scorecards vs SPY.
    Declared before /politicians/{name} so "tracker" isn't read as a name."""
    return pol_svc.tracker()


@router.get("/politicians/{name}")
def politicians_by_name(name: str, limit: int = Query(30, ge=1, le=500)) -> list[dict]:
    return pol_svc.by_politician(name, limit=limit)


# --- social ---

@router.get("/stocks/{symbol}/social")
def social(symbol: str) -> dict:
    """StockTwits (free, no auth) + Reddit (requires REDDIT_CLIENT_ID/SECRET).

    Returns the composite social signal + raw stream + per-source aggregates.
    """
    sym = symbol.upper()
    return {
        "signal": social_svc.social_signal(sym),
        "stocktwits": social_svc.fetch_stocktwits(sym, limit=20),
        "reddit": social_svc.fetch_reddit(sym, hours=48, limit=20),
    }
