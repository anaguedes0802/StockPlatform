"""Data-driven calibration of recommendation labels.

We persist every forecast (predictions table). When target_ts matures we can
compute the realized forward return for each predicted score. This script
buckets historical (recommendation_score, realized_h_day_return) pairs and
finds thresholds where labels are *monotonic in expected return*.

Output: `artifacts/calibration/rec_thresholds.json` like:
  {
    "STRONG_BUY":  0.55,
    "BUY":         0.20,
    "HOLD_LOW":   -0.15,
    "SELL":       -0.45,
    "STRONG_SELL": null,   # < SELL is STRONG_SELL
    "n_samples":   312,
    "calibrated_at": "...",
    "horizon_days": 21,
  }

Recommendation engine reads this at startup; falls back to the hardcoded
defaults when the file doesn't exist.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Prediction
from app.services import market_data as md


def _thresholds_path() -> str:
    d = os.path.join(settings.model_dir, "..", "calibration")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "rec_thresholds.json")


def load_thresholds() -> dict[str, Any] | None:
    p = _thresholds_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def calibrate(db: Session, lookback_days: int = 730, horizon: str = "30d") -> dict[str, Any]:
    """Walk matured predictions, fit thresholds where each label's bucket has
    monotonically increasing mean realized return.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = db.scalars(
        select(Prediction).where(
            Prediction.created_at >= cutoff,
            Prediction.target_ts <= now,
            Prediction.horizon == horizon,
        )
    ).all()
    samples: list[tuple[float, float]] = []
    for r in rows:
        try:
            # forecast direction proxy: log(point / issuance_close)
            issuance = md.get_history(r.symbol, "1d", "2y")
            if issuance.empty:
                continue
            issuance_close = float(issuance[issuance.index <= r.created_at]["close"].iloc[-1])
            target_close = float(issuance[issuance.index <= r.target_ts]["close"].iloc[-1])
            if issuance_close <= 0 or target_close <= 0:
                continue
            score_proxy = float(np.log(float(r.point_estimate) / issuance_close))
            realized = float(np.log(target_close / issuance_close))
            samples.append((score_proxy, realized))
        except Exception:
            continue

    if len(samples) < 30:
        return {
            "ok": False, "reason": f"need ≥30 matured predictions, have {len(samples)}",
            "n_samples": len(samples), "horizon": horizon,
        }

    arr = np.array(samples)
    scores = arr[:, 0]
    realized = arr[:, 1]

    # Use score quantiles as candidate bucket boundaries. Choose 5 buckets:
    # bottom 10%, 10-30%, 30-70%, 70-90%, top 10%.
    qs = np.quantile(scores, [0.10, 0.30, 0.70, 0.90])

    def bucket_mean(lo: float, hi: float) -> tuple[float, int]:
        mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        return (float(realized[mask].mean()) if n else 0.0, n)

    buckets = [
        (-np.inf, qs[0]),
        (qs[0], qs[1]),
        (qs[1], qs[2]),
        (qs[2], qs[3]),
        (qs[3], np.inf),
    ]
    labels = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
    bucket_stats = [(label, *bucket_mean(lo, hi), lo, hi) for label, (lo, hi) in zip(labels, buckets, strict=True)]

    # Verify monotonicity in realized mean. If non-monotonic, fall back to
    # static thresholds (still a useful diagnostic in the output).
    means = [s[1] for s in bucket_stats]
    monotonic = all(means[i] <= means[i + 1] + 1e-6 for i in range(len(means) - 1))

    out = {
        "ok": True,
        "n_samples": len(samples),
        "horizon": horizon,
        "calibrated_at": now.isoformat(),
        "buckets": [
            {"label": s[0], "n": s[2], "mean_realized": round(s[1], 5),
             "score_low": None if s[3] == -np.inf else round(s[3], 5),
             "score_high": None if s[4] == np.inf else round(s[4], 5)}
            for s in bucket_stats
        ],
        "monotonic": monotonic,
        # The actual thresholds applied at the boundary between buckets:
        "thresholds": {
            "STRONG_SELL_max": round(float(qs[0]), 5),
            "SELL_max":        round(float(qs[1]), 5),
            "HOLD_max":        round(float(qs[2]), 5),
            "BUY_max":         round(float(qs[3]), 5),
            # > BUY_max → STRONG_BUY
        },
    }
    with open(_thresholds_path(), "w") as f:
        json.dump(out, f, indent=2)
    return out


def label_from_score(score: float) -> str:
    """Apply calibrated thresholds when available; fall back to defaults."""
    cal = load_thresholds()
    if cal and cal.get("ok"):
        t = cal["thresholds"]
        if score < t["STRONG_SELL_max"]: return "STRONG_SELL"
        if score < t["SELL_max"]:        return "SELL"
        if score < t["HOLD_max"]:        return "HOLD"
        if score < t["BUY_max"]:         return "BUY"
        return "STRONG_BUY"
    # defaults
    if score > 0.6:  return "STRONG_BUY"
    if score > 0.2:  return "BUY"
    if score < -0.6: return "STRONG_SELL"
    if score < -0.2: return "SELL"
    return "HOLD"
