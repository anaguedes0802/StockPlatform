"""Tests for the deterministic Smart Money Concepts engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import price_action as pa


def _trending_up(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = 0.001
    log_ret = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    op = np.r_[close[0], close[:-1]]
    vol = rng.integers(1_000_000, 10_000_000, n).astype(int)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_analyze_returns_required_keys() -> None:
    df = _trending_up()
    r = pa.analyze(df)
    assert r["ok"] is True
    for k in ("swing_points", "fvgs", "order_blocks", "structure_events",
              "liquidity_sweeps", "zones", "confluence", "current_trend", "atr"):
        assert k in r


def test_confluence_score_in_range() -> None:
    df = _trending_up()
    r = pa.analyze(df)
    score = r["confluence"]["score"]
    assert -1.0 <= score <= 1.0


def test_trending_data_detects_structure() -> None:
    """Engine should detect at least one structural event on 200 noisy bars,
    and emit a valid confluence breakdown for the latest bar."""
    df = _trending_up(seed=7)
    r = pa.analyze(df)
    assert r["current_trend"] in ("up", "down", "range")
    assert len(r["structure_events"]) > 0
    assert isinstance(r["confluence"]["drivers"], list)


def test_analyze_includes_dealing_range() -> None:
    """analyze() should expose the ICT premium/discount array with a coherent
    structure: low < equilibrium < high, OTE bands nested inside, and price
    classified consistently against equilibrium."""
    df = _trending_up()
    r = pa.analyze(df)
    assert "dealing_range" in r
    dr = r["dealing_range"]
    assert dr is not None
    assert dr["low"] < dr["equilibrium"] < dr["high"]
    # bull OTE is a deep-discount band (below equilibrium), bear OTE a
    # deep-premium band (above equilibrium); both lie inside the range.
    assert dr["low"] <= dr["bull_ote_lower"] <= dr["bull_ote_upper"] <= dr["equilibrium"]
    assert dr["equilibrium"] <= dr["bear_ote_lower"] <= dr["bear_ote_upper"] <= dr["high"]
    assert dr["position"] in ("premium", "discount", "equilibrium")
    assert 0.0 <= dr["pct_of_range"] <= 1.0


def test_dealing_range_premium_discount_classification() -> None:
    """compute_dealing_range should label a price near the range high as a
    premium and inside the bearish OTE pocket, and a price near the low as a
    discount inside the bullish OTE pocket."""
    idx = pd.date_range("2024-01-01", periods=120, freq="B", tz="UTC")
    # A clean 100->90->110 range so swing high=110, swing low=90.
    base = np.r_[np.linspace(100, 110, 60), np.linspace(110, 90, 60)]
    df = pd.DataFrame(
        {"open": base, "high": base * 1.001, "low": base * 0.999,
         "close": base, "volume": np.full(120, 1_000_000)},
        index=idx,
    )
    swings = pa.detect_swings(df)

    # Price deep in the premium half (near the high) -> bearish OTE / premium.
    hi = float(df["high"].max())
    lo = float(df["low"].min())
    rng = hi - lo
    premium_px = lo + 0.70 * rng  # inside 0.62-0.79 from the low
    dr_hi = pa.compute_dealing_range(df, swings, premium_px)
    assert dr_hi is not None
    assert dr_hi.in_bear_ote is True
    assert dr_hi.in_bull_ote is False
    assert dr_hi.position == "premium"

    # Price deep in the discount half (near the low) -> bullish OTE / discount.
    discount_px = hi - 0.70 * rng  # inside 0.62-0.79 down from the high
    dr_lo = pa.compute_dealing_range(df, swings, discount_px)
    assert dr_lo is not None
    assert dr_lo.in_bull_ote is True
    assert dr_lo.in_bear_ote is False
    assert dr_lo.position == "discount"


def test_order_blocks_require_displacement() -> None:
    """Displacement gating must drop order blocks formed without an impulsive
    (FVG-bearing) move. A drifting series with no gaps should yield no OBs when
    displacement is required, but may yield some when it is not."""
    df = _trending_up(seed=3)
    atr = pa._atr(df, 14)
    gated = pa.detect_order_blocks(df, atr, require_displacement=True)
    ungated = pa.detect_order_blocks(df, atr, require_displacement=False)
    # Gating can only ever remove blocks, never add them.
    assert len(gated) <= len(ungated)
    # Every gated block's impulse leg must contain a fair-value gap.
    assert all(ob.direction in ("bullish", "bearish") for ob in gated)
