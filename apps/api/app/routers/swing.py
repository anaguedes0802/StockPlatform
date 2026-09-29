"""Swing-trading API — setups, scanner, and portfolio backtests.

Read-only: nothing here touches a broker. Paper/live execution of a swing
setup goes through a saved bot (`POST /bot/bots` with a `swing_*` kind).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import swing

router = APIRouter(prefix="/swing", tags=["swing"])

_BT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_BT_TTL_S = 3600


class ScanIn(BaseModel):
    setup: str = "breakout"
    universe: str | None = "etfs"
    symbols: list[str] | None = None
    params: dict[str, Any] | None = None
    equity: float = Field(default=100_000.0, gt=0)
    risk_pct: float = Field(default=1.0, gt=0, le=5)
    max_position_pct: float = Field(default=20.0, gt=0, le=100)
    include_short: bool | None = None
    ai_filter: bool = False   # score each candidate with the ML signal filter


class BacktestIn(BaseModel):
    setup: str = "breakout"
    universe: str | None = "etfs"
    symbols: list[str] | None = Field(default=None, max_length=80)
    params: dict[str, Any] | None = None
    start: str | None = "2005-01-01"
    end: str | None = None
    use_earnings: bool = True
    # Engine overrides (risk_per_trade_pct, max_positions, slippage_bps, …).
    config: dict[str, Any] | None = None


def _symbols(universe: str | None, symbols: list[str] | None) -> list[str]:
    if symbols:
        return [s.strip().upper() for s in symbols if s.strip()]
    if not universe or universe not in swing.UNIVERSES:
        raise HTTPException(400, f"unknown universe: {universe}")
    return list(swing.UNIVERSES[universe]["symbols"])


@router.get("/config", response_model=dict)
def config() -> dict:
    return {
        "setups": {k: {"label": v["label"], "summary": v["summary"], "params": v["params"]}
                   for k, v in swing.SETUPS.items()},
        "universes": swing.UNIVERSES,
        "engine_defaults": {
            "equity": asdict(swing.config_for("equity")),
            "fx": asdict(swing.config_for("fx")),
        },
        "bot_kinds": swing.BOT_KINDS,
        "disclaimer": (
            "Backtests are hypothetical and use hindsight-chosen universes. A historical "
            "edge is not a forecast. This is not investment advice."
        ),
    }


@router.post("/scan", response_model=dict)
def scan(payload: ScanIn) -> dict:
    if payload.setup not in swing.SETUPS:
        raise HTTPException(400, f"unknown setup: {payload.setup}")
    syms = _symbols(payload.universe, payload.symbols)
    if len(syms) > 80:
        raise HTTPException(400, "scan at most 80 symbols at a time")
    out = swing.scan(
        syms, payload.setup, params=payload.params, equity=payload.equity,
        risk_pct=payload.risk_pct, max_position_pct=payload.max_position_pct,
        include_short=payload.include_short,
    )
    if payload.ai_filter and out["candidates"]:
        from app.ml import signal_filter
        for c in out["candidates"]:
            try:
                c["ai"] = signal_filter.score_latest(c["symbol"], payload.setup, syms, payload.params,
                                                     side=1 if c["side"] == "long" else -1)
            except Exception as e:  # noqa: BLE001
                c["ai"] = {"error": str(e)[:200]}
    return out


@router.post("/backtest", response_model=dict)
def backtest(payload: BacktestIn) -> dict:
    if payload.setup not in swing.SETUPS:
        raise HTTPException(400, f"unknown setup: {payload.setup}")
    key = hashlib.sha1(json.dumps(payload.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    hit = _BT_CACHE.get(key)
    if hit and time.time() - hit[0] < _BT_TTL_S:
        return {**hit[1], "cached": True}
    try:
        res = swing.backtest(
            setup=payload.setup, universe=payload.universe, symbols=payload.symbols,
            params=payload.params, config=payload.config, start=payload.start,
            end=payload.end, use_earnings=payload.use_earnings,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _BT_CACHE[key] = (time.time(), res)
    return {**res, "cached": False}


class FilterEvalIn(BaseModel):
    setup: str = "breakout"
    universe: str | None = "etfs"
    symbols: list[str] | None = Field(default=None, max_length=80)
    params: dict[str, Any] | None = None
    first_test_year: int = Field(default=2010, ge=2008, le=2024)
    use_earnings: bool = True


@router.post("/filter-eval", response_model=dict)
def filter_eval(payload: FilterEvalIn) -> dict:
    """Walk-forward test of the ML signal filter against the unfiltered setup."""
    if payload.setup not in swing.SETUPS:
        raise HTTPException(400, f"unknown setup: {payload.setup}")
    key = "fe:" + hashlib.sha1(json.dumps(payload.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    hit = _BT_CACHE.get(key)
    if hit and time.time() - hit[0] < _BT_TTL_S:
        return {**hit[1], "cached": True}
    from app.ml import signal_filter
    try:
        res = signal_filter.evaluate(
            setup=payload.setup, universe=payload.universe, symbols=payload.symbols,
            params=payload.params, first_test_year=payload.first_test_year,
            use_earnings=payload.use_earnings,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _BT_CACHE[key] = (time.time(), res)
    return {**res, "cached": False}
