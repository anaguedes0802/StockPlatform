"""Minimal backtester. Supports SMA cross, RSI, MACD strategies expressed as DSL.

Production path: replace `_simulate` with vectorbt for vectorized speed + MC.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services import indicators as ind
from app.services import market_data as md
from app.services import risk as risk_mod


def _signals_from_dsl(df: pd.DataFrame, dsl: dict[str, Any]) -> pd.Series:
    """Compile a tiny DSL into a position-size series (1 = long full, 0 = flat).

    Supported:
      {"kind": "sma_cross", "fast": 12, "slow": 26}
      {"kind": "rsi", "buy_below": 30, "sell_above": 70}
      {"kind": "macd_cross"}
      {"kind": "buy_and_hold"}
    """
    kind = dsl.get("kind", "sma_cross")
    close = df["close"]
    if kind == "buy_and_hold":
        return pd.Series(1.0, index=df.index)
    if kind == "sma_cross":
        fast = ind.sma(close, dsl.get("fast", 12))
        slow = ind.sma(close, dsl.get("slow", 26))
        return (fast > slow).astype(float).fillna(0.0)
    if kind == "rsi":
        r = ind.rsi(close, dsl.get("period", 14))
        long = pd.Series(np.nan, index=df.index)
        long[r < dsl.get("buy_below", 30)] = 1.0
        long[r > dsl.get("sell_above", 70)] = 0.0
        return long.ffill().fillna(0.0)
    if kind == "macd_cross":
        m = ind.macd(close)
        return (m["macd"] > m["signal"]).astype(float).fillna(0.0)
    raise ValueError(f"unsupported strategy kind: {kind}")


def _simulate(df: pd.DataFrame, positions: pd.Series, initial_cash: float, commission_bps: float = 1.0) -> dict:
    """Single-asset simulator. Position is fraction of equity (0..1)."""
    close = df["close"].values
    pos = positions.reindex(df.index).fillna(0.0).values
    n = len(df)
    cash = initial_cash
    units = 0.0
    equity = np.zeros(n)
    trades: list[dict] = []
    last_pos = 0.0

    for i in range(n):
        price = float(close[i])
        target_pos = float(pos[i])
        current_value = cash + units * price
        if not np.isclose(target_pos, last_pos):
            # rebalance
            target_value = current_value * target_pos
            target_units = target_value / price if price else 0
            delta_units = target_units - units
            trade_value = abs(delta_units * price)
            fee = trade_value * commission_bps / 10000.0
            cash -= delta_units * price + fee
            units = target_units
            last_pos = target_pos
            trades.append(
                {
                    "ts": df.index[i].isoformat(),
                    "price": price,
                    "delta_units": delta_units,
                    "fee": fee,
                    "position_after": target_pos,
                }
            )
        equity[i] = cash + units * price

    eq = pd.Series(equity, index=df.index)
    rets = eq.pct_change().dropna()
    metrics = {
        "initial_cash": initial_cash,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float((eq.iloc[-1] / initial_cash - 1) * 100),
        "cagr_pct": float(risk_mod.cagr(eq) * 100),
        "sharpe": risk_mod.sharpe(rets),
        "sortino": risk_mod.sortino(rets),
        "max_drawdown_pct": float(risk_mod.max_drawdown(eq) * 100),
        "n_trades": len(trades),
    }
    return {
        "equity_curve": [{"ts": ts.isoformat(), "equity": float(v)} for ts, v in eq.items()],
        "trades": trades,
        "metrics": metrics,
    }


def run_backtest(
    symbol: str,
    dsl: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
    initial_cash: float = 10_000.0,
    range_: str = "5y",
) -> dict:
    df = md.get_history(symbol, interval="1d", range_=range_)
    if df.empty:
        raise ValueError("no data")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    if df.empty:
        raise ValueError("no data in window")
    signals = _signals_from_dsl(df, dsl)
    result = _simulate(df, signals, initial_cash=initial_cash)
    # benchmark: buy & hold
    bh = _simulate(df, pd.Series(1.0, index=df.index), initial_cash=initial_cash)
    result["benchmark"] = {"metrics": bh["metrics"], "equity_curve": bh["equity_curve"]}
    return result


def monte_carlo(metrics_trades: list[dict], n_sim: int = 500) -> dict:
    """Bootstrap resample trade-level returns to estimate terminal-equity dispersion."""
    if not metrics_trades:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    # this implementation is intentionally small; expand by mapping trades to per-trade returns
    raise NotImplementedError("Monte Carlo: see ARCHITECTURE §11 — extend with trade-return resampling.")
