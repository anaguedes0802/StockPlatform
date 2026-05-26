from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class PortfolioCreate(BaseModel):
    name: str
    base_currency: str = "USD"


class TransactionCreate(BaseModel):
    symbol: str
    side: str = Field(pattern=r"^(buy|sell)$")
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")
    occurred_at: datetime
    note: str | None = None


class TransactionOut(BaseModel):
    id: uuid.UUID
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    occurred_at: datetime
    note: str | None = None
    model_config = ConfigDict(from_attributes=True)


class Position(BaseModel):
    symbol: str
    quantity: Decimal
    avg_buy_price: Decimal
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None
    unrealized_pl_pct: float | None = None


class PortfolioMetrics(BaseModel):
    total_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float
    realized_pl: float
    cash: float = 0.0
    daily_pl: float | None = None
    total_dividends: float = 0.0
    total_return: float = 0.0       # unrealized_pl + total_dividends
    total_return_pct: float = 0.0   # vs cost_basis


class PortfolioDetail(BaseModel):
    id: uuid.UUID
    name: str
    base_currency: str
    positions: list[Position]
    metrics: PortfolioMetrics
    sector_allocation: dict[str, float] = {}
