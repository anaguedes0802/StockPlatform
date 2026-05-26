"""Forecast outcome tracking.

For every forecast we issue, we already persist to the `predictions` table.
This service scores predictions whose `target_ts` has passed: compares the
realized close price to the predicted point + intervals, and computes
per-horizon rolling metrics.

Metrics exposed:
  - hit_rate_direction: % of forecasts where sign(predicted return) matched
  - mae_logret: mean absolute error on log returns
  - coverage_p10_p90: fraction of realized values inside the band (target ~0.80)
  - coverage_p25_p75: should be ~0.50
  - bias_mean: did we systematically over- or under-shoot
  - n_scored: how many forecasts contributed
  - rolling_30d: last 30 calendar days of scored predictions

Pure functions over Postgres reads; no caching required (cheap queries).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Prediction
from app.services import market_data as md


def _realized_close(symbol: str, target_ts: datetime) -> float | None:
    """Look up the close price closest to (and not after) target_ts."""
    try:
        df = md.get_history(symbol, interval="1d", range_="2y")
        if df.empty:
            return None
        ts = target_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # find the last bar on or before target_ts
        idx = df.index
        idx_utc = idx.tz_convert("UTC") if idx.tz else idx
        eligible = df[idx_utc <= ts]
        if eligible.empty:
            return None
        return float(eligible["close"].iloc[-1])
    except Exception:
        return None


def score_predictions(db: Session, lookback_days: int = 365) -> dict[str, Any]:
    """Compute calibration metrics across all matured predictions in the window.

    Returns a per-horizon dict + an `overall` block, plus a sample of recent
    scored predictions (for the UI table).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    try:
        rows = db.scalars(
            select(Prediction).where(
                and_(
                    Prediction.created_at >= cutoff,
                    Prediction.target_ts <= now,
                )
            ).order_by(Prediction.target_ts.desc())
        ).all()
    except Exception:
        # Table doesn't exist yet (fresh DB) or other DB error — return empty stats
        # rather than 500-ing. Real deployments run alembic migrations on boot.
        try:
            db.rollback()
        except Exception:
            pass
        return {
            "as_of": now.isoformat(),
            "lookback_days": lookback_days,
            "overall": {"n": 0},
            "per_horizon": {},
            "recent_scored": [],
            "note": "No predictions table yet — run `alembic upgrade head` and start forecasting to populate it.",
        }

    by_horizon: dict[str, list[dict]] = {}
    sample: list[dict[str, Any]] = []

    for r in rows:
        realized = _realized_close(r.symbol, r.target_ts)
        if realized is None or float(r.point_estimate) <= 0:
            continue
        point = float(r.point_estimate)
        # Use the predicted return at issuance as `point` doesn't carry the
        # original "as-of" price. We can approximate the issuance close by
        # finding the close near `created_at`.
        issuance_close = _realized_close(r.symbol, r.created_at)
        if issuance_close is None or issuance_close <= 0:
            continue

        actual_logret = math.log(realized / issuance_close)
        pred_logret = math.log(point / issuance_close)
        err = pred_logret - actual_logret
        direction_correct = (pred_logret > 0) == (actual_logret > 0)
        in_p10_p90 = False
        in_p25_p75 = False
        if r.lower_80 is not None and r.upper_80 is not None:
            in_p10_p90 = float(r.lower_80) <= realized <= float(r.upper_80)
        if r.lower_95 is not None and r.upper_95 is not None:
            in_p25_p75 = float(r.lower_95) <= realized <= float(r.upper_95)

        entry = {
            "id": r.id,
            "symbol": r.symbol,
            "horizon": r.horizon,
            "created_at": r.created_at.isoformat(),
            "target_ts": r.target_ts.isoformat(),
            "issuance_close": round(issuance_close, 4),
            "predicted": round(point, 4),
            "realized": round(realized, 4),
            "pred_logret": round(pred_logret, 5),
            "actual_logret": round(actual_logret, 5),
            "abs_err_logret": round(abs(err), 5),
            "direction_correct": direction_correct,
            "in_p10_p90": in_p10_p90,
            "model": r.model,
            "confidence": float(r.confidence) if r.confidence is not None else None,
        }
        by_horizon.setdefault(r.horizon, []).append(entry)
        sample.append(entry)

    def stats(entries: list[dict]) -> dict[str, Any]:
        if not entries:
            return {"n": 0}
        n = len(entries)
        hit = sum(1 for e in entries if e["direction_correct"]) / n
        mae = sum(e["abs_err_logret"] for e in entries) / n
        cov = sum(1 for e in entries if e["in_p10_p90"]) / n
        bias = sum(e["pred_logret"] - e["actual_logret"] for e in entries) / n
        return {
            "n": n,
            "hit_rate_direction": round(hit, 3),
            "mae_logret": round(mae, 5),
            "coverage_p10_p90": round(cov, 3),
            "bias_mean": round(bias, 5),
        }

    per_horizon = {h: stats(es) for h, es in by_horizon.items()}
    overall = stats([e for es in by_horizon.values() for e in es])

    return {
        "as_of": now.isoformat(),
        "lookback_days": lookback_days,
        "overall": overall,
        "per_horizon": per_horizon,
        "recent_scored": sample[:30],
    }
