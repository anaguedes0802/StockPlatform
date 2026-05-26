"""Tests for the recommendation threshold calibration helpers."""
from __future__ import annotations

from app.ml.threshold_calibration import label_from_score


def test_label_from_score_defaults() -> None:
    """Without a calibration file present, falls back to defaults."""
    assert label_from_score(0.7) == "STRONG_BUY"
    assert label_from_score(0.4) == "BUY"
    assert label_from_score(0.0) == "HOLD"
    assert label_from_score(-0.4) == "SELL"
    assert label_from_score(-0.7) == "STRONG_SELL"


def test_label_from_score_monotonic() -> None:
    """The mapping must be monotone non-decreasing in score for a sane label set."""
    rank = {"STRONG_SELL": 0, "SELL": 1, "REDUCE": 2, "HOLD": 3, "ACCUMULATE": 4, "BUY": 5, "STRONG_BUY": 6}
    scores = [-1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    prev_rank = -1
    for s in scores:
        r = rank.get(label_from_score(s), -1)
        assert r >= prev_rank
        prev_rank = r
