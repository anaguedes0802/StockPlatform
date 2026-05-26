from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    symbol: str
    horizons: list[Literal["1d", "5d", "30d", "90d"]] = ["1d", "5d", "30d"]
    models: list[str] = ["ensemble"]


class Interval(BaseModel):
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class Driver(BaseModel):
    name: str
    direction: Literal["up", "down", "neutral"]
    weight: float


class Forecast(BaseModel):
    symbol: str
    horizon: str
    as_of: datetime
    target_date: datetime
    point: float
    intervals: Interval
    direction_prob: dict[str, float]
    expected_volatility_pct: float
    confidence: float
    contributions: dict[str, float]
    drivers: list[Driver]
    model_mix: dict[str, float]
    freshness: dict = {}


class ForecastResponse(BaseModel):
    forecasts: list[Forecast]


class RecommendationRequest(BaseModel):
    symbol: str


class RecommendationReasoning(BaseModel):
    technical: dict
    fundamental: dict
    sentiment: dict
    macro: dict
    price_action: dict = {}
    events: list
    risk: dict


class RecommendationResponse(BaseModel):
    symbol: str
    label: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    score: float
    confidence: float
    reasoning: RecommendationReasoning
