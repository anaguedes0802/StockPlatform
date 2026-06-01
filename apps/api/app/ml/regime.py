"""Lightweight market regime detector.

For v0 we use a rule-based detector over realized volatility and trend slope.
Future: 2-state Gaussian HMM on returns + VIX, or a HMM on the
Nixtla NeuralForecast feature set.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Regime = Literal["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "sideways"]


# Hysteresis margins. To *switch* a metric's state we require it to cross the
# threshold by this margin; once switched, it stays until the metric crosses
# back past (threshold - margin). This dead-band stops the regime from
# flip-flopping when vol or slope hovers right at a boundary. Deterministic:
# the result depends only on `close` and the (optional) previous regime.
VOL_THRESHOLD = 0.30
VOL_MARGIN = 0.03          # ~10% of the threshold
SLOPE_THRESHOLD = 0.05
SLOPE_MARGIN = 0.01


def detect(close: pd.Series, lookback: int = 60, prev: Regime | None = None) -> Regime:
    """Rule-based regime detection with hysteresis.

    `prev` is the previously-emitted regime, if any. When supplied, a metric
    only flips its state once it crosses the relevant threshold by a margin,
    avoiding boundary flip-flop. With `prev=None` the bare thresholds apply
    (back-compatible with existing single-shot callers).
    """
    if len(close) < lookback + 5:
        return "sideways"
    window = close.tail(lookback)
    ret = window.pct_change().dropna()
    if ret.empty:
        return "sideways"
    ann_vol = float(ret.std() * np.sqrt(252))
    slope = float((window.iloc[-1] / window.iloc[0] - 1))

    # Derive the prior latched states from the previous regime so we know which
    # side of each dead-band we are currently sitting on.
    prev_high_vol = bool(prev) and prev in ("bull_high_vol", "bear_high_vol")
    prev_bullish = bool(prev) and prev in ("bull_low_vol", "bull_high_vol")
    prev_bearish = bool(prev) and prev in ("bear_low_vol", "bear_high_vol")

    def _latch(value: float, threshold: float, margin: float, currently_on: bool) -> bool:
        # Require crossing above (threshold + margin) to turn on, and dropping
        # below (threshold - margin) to turn off; otherwise hold prior state.
        if currently_on:
            return value > threshold - margin
        return value > threshold + margin

    high_vol = _latch(ann_vol, VOL_THRESHOLD, VOL_MARGIN, prev_high_vol)
    # Trend uses a symmetric dead-band around ±SLOPE_THRESHOLD.
    if prev_bullish:
        bullish = slope > SLOPE_THRESHOLD - SLOPE_MARGIN
    else:
        bullish = slope > SLOPE_THRESHOLD + SLOPE_MARGIN
    if prev_bearish:
        bearish = slope < -(SLOPE_THRESHOLD - SLOPE_MARGIN)
    else:
        bearish = slope < -(SLOPE_THRESHOLD + SLOPE_MARGIN)

    if bullish and high_vol:
        return "bull_high_vol"
    if bullish:
        return "bull_low_vol"
    if bearish and high_vol:
        return "bear_high_vol"
    if bearish:
        return "bear_low_vol"
    return "sideways"


# default per-regime weights for the recommendation engine.
# `price_action` weight reflects how much Smart Money structure matters in that regime.
# In high-vol or bear regimes, structure (liquidity sweeps, demand zones, CHOCHs) signals more.
REGIME_WEIGHTS: dict[Regime, dict[str, float]] = {
    "bull_low_vol":  {"forecast": 0.30, "technical": 0.20, "fundamental": 0.15, "sentiment": 0.10, "macro": 0.10, "price_action": 0.15},
    "bull_high_vol": {"forecast": 0.22, "technical": 0.15, "fundamental": 0.08, "sentiment": 0.20, "macro": 0.10, "price_action": 0.25},
    "bear_low_vol":  {"forecast": 0.28, "technical": 0.20, "fundamental": 0.12, "sentiment": 0.12, "macro": 0.08, "price_action": 0.20},
    "bear_high_vol": {"forecast": 0.20, "technical": 0.12, "fundamental": 0.08, "sentiment": 0.20, "macro": 0.15, "price_action": 0.25},
    "sideways":      {"forecast": 0.20, "technical": 0.20, "fundamental": 0.15, "sentiment": 0.10, "macro": 0.10, "price_action": 0.25},
}
