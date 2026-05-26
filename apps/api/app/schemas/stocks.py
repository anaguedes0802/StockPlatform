from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    symbol: str
    name: str | None = None
    exchange: str | None = None
    asset_class: str = "stock"


class Quote(BaseModel):
    symbol: str
    price: float
    previous_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    currency: str | None = None
    market_state: str | None = None
    ts: datetime | None = None


class Bar(BaseModel):
    ts: datetime
    o: float = Field(..., alias="open")
    h: float = Field(..., alias="high")
    l: float = Field(..., alias="low")
    c: float = Field(..., alias="close")
    v: int = Field(..., alias="volume")

    model_config = {"populate_by_name": True}


class HistoryResponse(BaseModel):
    symbol: str
    interval: str
    bars: list[Bar]


class IndicatorsResponse(BaseModel):
    symbol: str
    interval: str
    indicators: dict[str, list[float | None]]
    ts: list[datetime]


class StockProfile(BaseModel):
    symbol: str
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    description: str | None = None
    website: str | None = None
    employees: int | None = None


class KeyStats(BaseModel):
    pe: float | None = None
    forward_pe: float | None = None
    eps: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    avg_volume: int | None = None


class StockDetail(BaseModel):
    profile: StockProfile
    quote: Quote
    key_stats: KeyStats
