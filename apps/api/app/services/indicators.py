"""Technical indicators implemented in pandas/numpy — no extra deps required."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    # ema() already applies min_periods=span, so macd_line is NaN until the slow
    # EMA is warmed. Warm the signal EMA too so early bars stay NaN until ready.
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std
    pctb = (close - lower) / (upper - lower)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pctb": pctb})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift()
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    # Real-world VWAP resets each trading session. When we have a DatetimeIndex,
    # group the cumulative sums by calendar day so each session starts fresh.
    # Otherwise fall back to cumulative-from-inception. Past/current-only (no leak).
    typical = (high + low + close) / 3
    pv = typical * volume
    if isinstance(close.index, pd.DatetimeIndex):
        day = close.index.normalize()
        cum_pv = pv.groupby(day).cumsum()
        cum_v = volume.groupby(day).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_v = volume.cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_line = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_line = k_line.rolling(d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def fibonacci_retracement(swing_high: float, swing_low: float) -> dict[str, float]:
    diff = swing_high - swing_low
    return {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    # LOOK-AHEAD: lagging span pulls 26 bars of FUTURE close into the current row.
    # Valid for charting only — never as a model feature. The name "lagging" is the
    # guard: feature_columns() in app.ml.features excludes any column containing it.
    lagging = close.shift(-26)
    return pd.DataFrame({"conv": conv, "base": base, "span_a": span_a, "span_b": span_b, "lagging": lagging})


# ---------- registry-style dispatch ----------

INDICATOR_SPECS: dict[str, tuple[str, dict]] = {
    "sma_20": ("sma", {"period": 20}),
    "sma_50": ("sma", {"period": 50}),
    "sma_200": ("sma", {"period": 200}),
    "ema_12": ("ema", {"period": 12}),
    "ema_26": ("ema", {"period": 26}),
    "rsi_14": ("rsi", {"period": 14}),
    "macd": ("macd", {}),
    "bb_20": ("bollinger", {"period": 20, "k": 2.0}),
    "atr_14": ("atr", {"period": 14}),
    "vwap": ("vwap", {}),
    "stoch_14_3": ("stochastic", {"k": 14, "d": 3}),
    "ichimoku": ("ichimoku", {}),
}


def compute(name: str, df: pd.DataFrame) -> dict[str, list[float | None]]:
    """Compute a named indicator. df must have columns: open, high, low, close, volume."""
    if name not in INDICATOR_SPECS:
        raise KeyError(f"unknown indicator: {name}")
    fn_name, kwargs = INDICATOR_SPECS[name]
    if fn_name in ("sma", "ema", "rsi"):
        s = globals()[fn_name](df["close"], **kwargs)
        return {name: _to_jsonable(s)}
    if fn_name == "macd":
        out = macd(df["close"], **kwargs)
        return {f"{name}.{c}": _to_jsonable(out[c]) for c in out.columns}
    if fn_name == "bollinger":
        out = bollinger(df["close"], **kwargs)
        return {f"{name}.{c}": _to_jsonable(out[c]) for c in out.columns}
    if fn_name == "atr":
        s = atr(df["high"], df["low"], df["close"], **kwargs)
        return {name: _to_jsonable(s)}
    if fn_name == "vwap":
        s = vwap(df["high"], df["low"], df["close"], df["volume"])
        return {name: _to_jsonable(s)}
    if fn_name == "stochastic":
        out = stochastic(df["high"], df["low"], df["close"], **kwargs)
        return {f"{name}.{c}": _to_jsonable(out[c]) for c in out.columns}
    if fn_name == "ichimoku":
        out = ichimoku(df["high"], df["low"], df["close"])
        return {f"{name}.{c}": _to_jsonable(out[c]) for c in out.columns}
    raise KeyError(name)


def _to_jsonable(s: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in s.tolist()]
