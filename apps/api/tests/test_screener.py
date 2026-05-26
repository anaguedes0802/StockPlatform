"""Tests for the screener scoring strategies on fixture inputs.

We test the pure-scoring functions (_score_rising_star, _score_breakout, etc.)
on hand-built `feature_pack` dicts — no network involved.
"""
from __future__ import annotations

from app.services.screener import (
    _score_breakout,
    _score_rising_star,
    _score_smart_money,
    _score_value_with_catalyst,
)


def _base_pack(**overrides):
    pack = {
        "symbol": "TEST", "name": "Test Co", "sector": "Technology",
        "last_price": 100.0, "ret_1m": 0.03, "ret_3m": 0.10, "ret_1y": 0.30,
        "rsi14": 60.0, "above_sma50": True, "above_sma200": True,
        "atr_pct": 0.02, "vol_z_20": 0.5,
        "pct_off_high": 0.1, "pos_in_range": 0.80,
        "pe": 25.0, "beta": 1.0, "market_cap": 50_000_000_000,
        "n_articles": 10, "sentiment_weighted": 0.20,
        "pa_score": 0.30, "pa_label": "bullish", "pa_drivers": [], "pa_trend": "up",
    }
    pack.update(overrides)
    return pack


def test_rising_star_bullish() -> None:
    score, comps, rationale = _score_rising_star(_base_pack())
    assert -1 <= score <= 1
    assert score > 0
    # Should reference momentum, RSI, or above-SMA in rationale
    assert any("momentum" in r.lower() or "rsi" in r.lower() or "sma" in r.lower() for r in rationale)


def test_rising_star_penalizes_overbought() -> None:
    overheated = _base_pack(rsi14=85)
    normal = _base_pack(rsi14=60)
    s_over, _, _ = _score_rising_star(overheated)
    s_norm, _, _ = _score_rising_star(normal)
    assert s_over < s_norm


def test_smart_money_aligns_with_pa_score() -> None:
    bullish_pa = _base_pack(pa_score=0.7, pa_label="strong_bullish", pa_trend="up")
    bearish_pa = _base_pack(pa_score=-0.7, pa_label="strong_bearish", pa_trend="down",
                              above_sma200=False, sentiment_weighted=-0.2)
    s_b, _, _ = _score_smart_money(bullish_pa)
    s_br, _, _ = _score_smart_money(bearish_pa)
    assert s_b > 0
    assert s_br < s_b


def test_breakout_requires_above_smas() -> None:
    above = _base_pack(above_sma50=True, above_sma200=True, pos_in_range=0.95, vol_z_20=2.0)
    below = _base_pack(above_sma50=False, above_sma200=False, pos_in_range=0.4)
    s_above, _, _ = _score_breakout(above)
    s_below, _, _ = _score_breakout(below)
    assert s_above > s_below


def test_value_catalyst_prefers_low_pe_with_positive_news() -> None:
    cheap_good = _base_pack(pe=12.0, sentiment_weighted=0.3, ret_1m=0.05)
    expensive_bad = _base_pack(pe=80.0, sentiment_weighted=-0.3, ret_1m=-0.05)
    s1, _, _ = _score_value_with_catalyst(cheap_good)
    s2, _, _ = _score_value_with_catalyst(expensive_bad)
    assert s1 > s2
