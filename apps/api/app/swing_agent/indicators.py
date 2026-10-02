"""Indicators for the regime layer. All causal: the value at row i uses rows ≤ i.

SMA and Wilder ATR come from `app.services.indicators`; this adds Wilder ADX,
a trailing percentile rank, and session-aware intraday helpers (VWAP,
volume / range relative to the same time of day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.indicators import atr, sma  # noqa: F401 — re-exported for the regime modules


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ADX (trend strength, 0–100, direction-free)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                   axis=1).max(axis=1)
    a = 1 / period
    atr_w = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean()


def trailing_rank(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile (0–1) of today's value within the trailing `window` values,
    today included. No future values enter."""
    mp = min_periods or window // 2
    return x.rolling(window, min_periods=mp).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)


def slope_atr(ma: pd.Series, days: int, atr_s: pd.Series) -> pd.Series:
    """Change of a moving average over `days`, in ATRs per day."""
    return (ma - ma.shift(days)) / (days * atr_s)


# ---------------------------------------------------------------------------
# Intraday (15m bars with a `session` column, regular hours only)
# ---------------------------------------------------------------------------

def slot(m15: pd.DataFrame) -> pd.Series:
    """0-based index of each bar within its session (09:30 bar = 0)."""
    return m15.groupby("session").cumcount()


def session_vwap(m15: pd.DataFrame, price_col: str = "vwap") -> pd.Series:
    """Running VWAP since the open, from each bar's own VWAP × volume."""
    px = m15[price_col].fillna(m15["close"])
    pv = (px * m15["volume"]).groupby(m15["session"]).cumsum()
    v = m15["volume"].groupby(m15["session"]).cumsum()
    return pv / v.replace(0, np.nan)


def same_time_baseline(value: pd.Series, sessions: pd.Series, slots: pd.Series,
                       lookback: int) -> pd.Series:
    """Mean of `value` at the same slot over the previous `lookback` sessions
    (today excluded), aligned back to each bar."""
    wide = pd.DataFrame({"v": value.to_numpy(), "s": sessions.to_numpy(), "k": slots.to_numpy()}) \
        .pivot_table(index="s", columns="k", values="v", aggfunc="last")
    base = wide.rolling(lookback, min_periods=max(5, lookback // 2)).mean().shift(1)
    stacked = base.stack(future_stack=True)
    idx = pd.MultiIndex.from_arrays([sessions.to_numpy(), slots.to_numpy()])
    return pd.Series(stacked.reindex(idx).to_numpy(), index=value.index)
