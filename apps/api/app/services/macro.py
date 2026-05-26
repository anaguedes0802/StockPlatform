"""Macro feature provider.

Pulls a small set of macro series (VIX, 10Y yield, USD index) and aligns them
by trading-day to a target dataframe. Cached aggressively because these series
are slow-moving and shared across all forecasts.
"""
from __future__ import annotations

import pandas as pd

from app.services import market_data as md

# (yfinance symbol → feature prefix)
MACRO_SERIES: dict[str, str] = {
    "^VIX": "vix",        # equity volatility index
    "^TNX": "yield_10y",  # 10-year US treasury yield (in percent)
    "DX-Y.NYB": "dxy",    # US dollar index
}


def fetch_macro_series(range_: str = "5y") -> pd.DataFrame:
    """Return a wide dataframe of macro series, daily, UTC-indexed."""
    frames = []
    for sym, prefix in MACRO_SERIES.items():
        try:
            df = md.get_history(sym, interval="1d", range_=range_)
            if df.empty:
                continue
            s = df["close"].rename(f"{prefix}")
            frames.append(s)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, sort=False).sort_index()
    # Forward-fill across non-overlapping holidays (DXY/UST close on different days from equities).
    out = out.ffill()
    return out


def macro_features(target_index: pd.DatetimeIndex, range_: str = "5y") -> pd.DataFrame:
    """Build per-bar macro features aligned to `target_index`.

    Features per series X:
      - X_level: current level
      - X_chg_5: 5-day change
      - X_z_60: 60-day z-score (current vs 60-day rolling mean)
    """
    macro = fetch_macro_series(range_=range_)
    if macro.empty:
        return pd.DataFrame(index=target_index)

    feats = pd.DataFrame(index=macro.index)
    for col in macro.columns:
        s = macro[col]
        feats[f"{col}_level"] = s
        feats[f"{col}_chg_5"] = s.pct_change(5)
        mean60 = s.rolling(60, min_periods=20).mean()
        std60 = s.rolling(60, min_periods=20).std() + 1e-9
        feats[f"{col}_z_60"] = (s - mean60) / std60

    # Align to target_index. Forward-fill to handle exchange holiday mismatches.
    feats = feats.reindex(target_index.tz_convert(feats.index.tz) if feats.index.tz else target_index, method="ffill")
    # Replace any remaining NaN at the head with 0 (early bars before macro data starts)
    feats = feats.fillna(0.0)
    return feats
