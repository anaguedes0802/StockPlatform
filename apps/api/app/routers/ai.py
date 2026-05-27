from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.rate_limit import per_ip_limiter
from app.db.models import Prediction
from app.db.session import get_db
from app.ml.ensemble import REGIME_MODEL_MIX, ensemble_forecast
from app.schemas.ai import (
    Forecast,
    ForecastRequest,
    ForecastResponse,
    Interval,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services import market_data as md
from app.services.calibration import score_predictions
from app.services.opinion import get_opinion
from app.services.recommendation import recommend as compute_recommendation

router = APIRouter(prefix="/ai", tags=["ai"])


_HORIZON_DAYS = {"1d": 1, "5d": 5, "30d": 21, "90d": 63}


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[Depends(per_ip_limiter(30, 0.5))],
)
async def forecast(payload: ForecastRequest, db: Session = Depends(get_db)) -> ForecastResponse:
    symbol = payload.symbol.upper()
    df = await asyncio.to_thread(md.get_history, symbol, "1d", "3y")
    if df.empty or len(df) < 200:
        raise HTTPException(404, "not enough history to forecast")

    forecasts: list[Forecast] = []
    for h_label in payload.horizons:
        h_days = _HORIZON_DAYS[h_label]
        # Training + inference happens in a worker thread — never block the event loop.
        result, regime, mix, freshness = await asyncio.to_thread(
            ensemble_forecast, df, h_days, symbol,
        )
        target_date = datetime.now(timezone.utc) + timedelta(days=h_days)
        forecasts.append(
            Forecast(
                symbol=symbol,
                horizon=h_label,
                as_of=datetime.now(timezone.utc),
                target_date=target_date,
                point=result.point,
                intervals=Interval(p10=result.p10, p25=result.p25, p50=result.p50, p75=result.p75, p90=result.p90),
                direction_prob={"up": result.direction_prob_up, "down": 1 - result.direction_prob_up},
                expected_volatility_pct=result.expected_volatility_pct,
                confidence=result.confidence,
                contributions=result.contributions,
                drivers=result.drivers,
                model_mix=mix,
                freshness=freshness,
            )
        )

        # Persist for later calibration scoring. Use try/except so a DB hiccup
        # doesn't break the user-facing response.
        try:
            db.add(Prediction(
                symbol=symbol,
                model="ensemble",
                horizon=h_label,
                target_ts=target_date,
                point_estimate=Decimal(str(result.point)),
                lower_80=Decimal(str(result.p10)),
                upper_80=Decimal(str(result.p90)),
                lower_95=Decimal(str(result.p25)),
                upper_95=Decimal(str(result.p75)),
                confidence=float(result.confidence),
                contributions=result.contributions,
                drivers=result.drivers,
                direction_prob_up=float(result.direction_prob_up),
            ))
            db.commit()
        except Exception:
            db.rollback()

    return ForecastResponse(forecasts=forecasts)


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    dependencies=[Depends(per_ip_limiter(30, 0.5))],
)
async def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    symbol = payload.symbol.upper()
    forecast_point: float | None = None
    df = await asyncio.to_thread(md.get_history, symbol, "1d", "3y")
    if not df.empty and len(df) >= 200:
        try:
            result, _, _, _ = await asyncio.to_thread(ensemble_forecast, df, _HORIZON_DAYS["30d"], symbol)
            forecast_point = result.point
        except Exception:
            forecast_point = None
    out = await asyncio.to_thread(compute_recommendation, symbol, forecast_point)
    return RecommendationResponse(**out)


@router.get("/calibration")
def calibration(lookback_days: int = 365, db: Session = Depends(get_db)) -> dict:
    """Live calibration metrics: how accurate have our forecasts been over the
    last N days. Open endpoint so anyone can audit our claims."""
    return score_predictions(db, lookback_days=lookback_days)


@router.post("/calibration/refit")
async def refit_thresholds(lookback_days: int = 730, horizon: str = "30d",
                           db: Session = Depends(get_db)) -> dict:
    """Re-fit recommendation label thresholds from matured predictions.
    Writes apps/api/artifacts/calibration/rec_thresholds.json."""
    from app.ml.threshold_calibration import calibrate
    return await asyncio.to_thread(calibrate, db, lookback_days, horizon)


@router.get(
    "/opinion/{symbol}",
    dependencies=[Depends(per_ip_limiter(10, 0.15))],
)
async def opinion(
    symbol: str,
    use_llm: bool | None = None,
    risk_mode: str = "conservative",
    timeframe: str = "position",
) -> dict:
    """Synthesized analyst opinion.

    `timeframe` ∈ {intraday, swing, position, longterm} — matches the analysis
    to the user's holding period. Different timeframes produce different
    demand/supply zones, different ATR, different structure events:
      - **intraday**  (15m × 1mo, hours–1 day horizon) — day trade view
      - **swing**     (1d × 6mo, 1–3 weeks)            — swing trade view
      - **position**  (1d × 2y, 1–3 months) — DEFAULT  — position trade view
      - **longterm**  (1wk × 10y, 6+ months)           — long-term hold view

    `risk_mode` tunes the trade-plan generator:
      - **conservative** (default): tight zone reachability (3× ATR), 2× ATR stop
      - **balanced**: 5× ATR reachability, 2.5× ATR stop
      - **speculative**: 8× ATR reachability, 3× ATR stop, tiny position cap
    """
    rm = (risk_mode or "conservative").lower()
    if rm not in ("conservative", "balanced", "speculative"):
        rm = "conservative"
    tf = (timeframe or "position").lower()
    if tf not in ("intraday", "swing", "position", "longterm"):
        tf = "position"
    return await asyncio.to_thread(get_opinion, symbol.upper(), use_llm, rm, tf)
