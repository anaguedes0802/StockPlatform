"""Indicator math sanity checks. Pure pandas — no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import indicators as ind


def _series(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + rng.normal(0, 1, n).cumsum())
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.integers(1_000, 10_000, n).astype(int))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_sma_length():
    df = _series()
    s = ind.sma(df["close"], 20)
    assert s.iloc[:19].isna().all()
    assert not np.isnan(s.iloc[-1])


def test_rsi_bounds():
    df = _series()
    r = ind.rsi(df["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_columns():
    df = _series()
    m = ind.macd(df["close"])
    assert set(m.columns) == {"macd", "signal", "hist"}


def test_compute_dispatch():
    df = _series()
    out = ind.compute("sma_20", df)
    assert "sma_20" in out and len(out["sma_20"]) == len(df)
