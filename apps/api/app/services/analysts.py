"""Analyst consensus + revisions — the "real edge" signals.

Source: yfinance (free, no key). Wraps three datasets:
  - Ticker.earnings_estimate — forward EPS consensus per fiscal period
  - Ticker.revenue_estimate  — same for revenue
  - Ticker.recommendations    — full history of brokerage buy/sell/hold ratings
  - Ticker.upgrades_downgrades — explicit rating change events with firm names

The two signals we extract:
  - `consensus_signal`: latest forward-quarter EPS consensus delta from prior
    quarter's actual + analyst-count quality weighting → score ∈ [-1, +1]
  - `revisions_signal`: net-positive vs net-negative rating actions in the
    last 90 days → score ∈ [-1, +1]
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import redis

from app.config import settings
from app.core.logging import log
from app.services.market_data import _yf_ticker  # type: ignore


_redis: redis.Redis | None = None


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _cache_get(key: str) -> Any | None:
    try:
        v = _cache().get(key)
        return json.loads(v) if v else None
    except Exception:
        return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Earnings consensus
# -----------------------------------------------------------------------------

def earnings_estimate(symbol: str) -> dict[str, Any]:
    """Forward EPS + revenue consensus for the symbol.

    Returns:
      {
        "current_quarter":  {avg_eps, low, high, n_analysts, growth_yoy_pct},
        "next_quarter":     {...},
        "current_year":     {...},
        "next_year":        {...},
        "asof": iso timestamp
      }
    """
    sym = symbol.upper()
    key = f"earn_est:{sym}"
    if (cached := _cache_get(key)) is not None:
        return cached

    t = _yf_ticker(sym)
    out: dict[str, Any] = {"asof": datetime.now(timezone.utc).isoformat()}
    try:
        eps = t.earnings_estimate
        rev = t.revenue_estimate
    except Exception as e:
        log.warning("yfinance_consensus_failed", symbol=sym, err=str(e))
        eps = None
        rev = None

    def _row_to_dict(df: Any, period_key: str, kind: str) -> dict[str, Any] | None:
        if df is None or not hasattr(df, "loc"):
            return None
        try:
            row = df.loc[period_key]
        except (KeyError, AttributeError):
            return None
        d: dict[str, Any] = {}
        for col in ("avg", "low", "high", "numberOfAnalysts", "growth"):
            if col in row.index:
                v = row[col]
                try:
                    d[col] = float(v) if pd.notna(v) else None
                except (TypeError, ValueError):
                    d[col] = None
        return d if d else None

    # yfinance keys these by short-period strings ("0q", "+1q", "0y", "+1y")
    periods = {
        "current_quarter": "0q",
        "next_quarter":    "+1q",
        "current_year":    "0y",
        "next_year":       "+1y",
    }
    for label, key_ in periods.items():
        e = _row_to_dict(eps, key_, "eps")
        r = _row_to_dict(rev, key_, "rev")
        if e or r:
            out[label] = {"eps": e, "revenue": r}

    _cache_set(key, out, ttl=12 * 3600)
    return out


def consensus_signal(symbol: str) -> dict[str, Any]:
    """Score the forward consensus quality.

    Heuristic: next-quarter EPS growth_yoy is the primary signal; weight by
    analyst count (more coverage = more reliable). High forecast growth that's
    backed by many analysts is a tailwind; revisions matter more (see below).
    """
    est = earnings_estimate(symbol)
    nq = (est.get("next_quarter") or {}).get("eps") or {}
    cq = (est.get("current_quarter") or {}).get("eps") or {}
    growth = nq.get("growth") if nq.get("growth") is not None else cq.get("growth")
    n_analysts = nq.get("numberOfAnalysts") or cq.get("numberOfAnalysts") or 0

    if growth is None:
        return {"score": 0.0, "summary": "No forward EPS consensus available.", "n_analysts": 0}

    # Squash: ±100% YoY growth maxes the signal; <=4 analysts derates by half.
    weight = min(1.0, n_analysts / 10.0)
    score = math.tanh(growth * 1.5) * weight

    parts = []
    if growth > 0.05:
        parts.append(f"Consensus EPS growth {growth*100:+.1f}% YoY")
    elif growth < -0.05:
        parts.append(f"Consensus EPS contraction {growth*100:+.1f}% YoY")
    if n_analysts:
        parts.append(f"{int(n_analysts)} analysts covering")

    return {
        "score": round(float(score), 3),
        "growth_yoy": float(growth),
        "n_analysts": int(n_analysts),
        "summary": "; ".join(parts) or "Limited consensus signal.",
    }


# -----------------------------------------------------------------------------
# Analyst revisions
# -----------------------------------------------------------------------------

# Map the most common Buy/Sell/Hold-style strings to a numeric score in [-1, +1]
_RATING_SCORES = {
    "strong buy": 1.0, "buy": 0.7, "outperform": 0.7, "overweight": 0.6,
    "accumulate": 0.5, "add": 0.5, "positive": 0.5,
    "hold": 0.0, "neutral": 0.0, "equal-weight": 0.0, "equal weight": 0.0,
    "in-line": 0.0, "in line": 0.0, "market perform": 0.0, "perform": 0.0,
    "reduce": -0.5, "underweight": -0.6, "underperform": -0.7,
    "sell": -0.7, "strong sell": -1.0, "negative": -0.5,
}


def _rating_score(label: str | None) -> float:
    if not label:
        return 0.0
    return _RATING_SCORES.get(str(label).strip().lower(), 0.0)


def revisions_history(symbol: str, limit: int = 30) -> list[dict[str, Any]]:
    """Recent upgrades/downgrades — newest first.

    Each row: {ts, firm, from_grade, to_grade, action, delta_score}
    """
    sym = symbol.upper()
    key = f"revisions:{sym}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached

    t = _yf_ticker(sym)
    out: list[dict[str, Any]] = []
    try:
        df = t.upgrades_downgrades
    except Exception:
        df = None
    if df is None or not hasattr(df, "iterrows"):
        _cache_set(key, [], ttl=4 * 3600)
        return []

    try:
        for ts, row in df.iterrows():
            firm = row.get("Firm")
            to_grade = row.get("ToGrade")
            from_grade = row.get("FromGrade")
            action = row.get("Action")
            from_s = _rating_score(from_grade)
            to_s = _rating_score(to_grade)
            delta = to_s - from_s if from_grade and to_grade else to_s
            out.append({
                "ts": pd.Timestamp(ts).isoformat(),
                "firm": str(firm) if firm else None,
                "from_grade": str(from_grade) if from_grade else None,
                "to_grade": str(to_grade) if to_grade else None,
                "action": str(action) if action else None,
                "delta_score": round(float(delta), 3),
            })
    except Exception as e:
        log.warning("revisions_parse_failed", symbol=sym, err=str(e))

    out.sort(key=lambda r: r["ts"], reverse=True)
    out = out[:limit]
    _cache_set(key, out, ttl=4 * 3600)
    return out


def revisions_signal(symbol: str, lookback_days: int = 90) -> dict[str, Any]:
    """Net analyst-rating change over the last N days.

    Score is the average delta_score across all revisions in window,
    multiplied by tanh(n_revisions / 5) so a single upgrade carries less
    weight than a cluster of upgrades.
    """
    rows = revisions_history(symbol, limit=50)
    if not rows:
        return {"score": 0.0, "summary": "No recent analyst rating changes.", "n_revisions": 0}
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    window = [r for r in rows if pd.Timestamp(r["ts"]).to_pydatetime().replace(tzinfo=timezone.utc) >= cutoff]
    if not window:
        return {"score": 0.0, "summary": "No rating changes in the lookback window.",
                "n_revisions": 0}
    avg_delta = sum(r["delta_score"] for r in window) / len(window)
    n_factor = math.tanh(len(window) / 5.0)
    score = float(max(-1.0, min(1.0, avg_delta * n_factor * 1.2)))

    n_up = sum(1 for r in window if (r["delta_score"] or 0) > 0)
    n_down = sum(1 for r in window if (r["delta_score"] or 0) < 0)
    summary_parts = []
    if n_up: summary_parts.append(f"{n_up} upgrade{'s' if n_up != 1 else ''}")
    if n_down: summary_parts.append(f"{n_down} downgrade{'s' if n_down != 1 else ''}")
    if window:
        top = sorted(window, key=lambda r: abs(r["delta_score"] or 0), reverse=True)[0]
        if top.get("firm") and top.get("to_grade"):
            summary_parts.append(f"latest: {top['firm']} {top.get('action') or '→'} {top['to_grade']}")
    summary = f"({lookback_days}d) " + "; ".join(summary_parts) if summary_parts else "no notable changes"

    return {
        "score": round(score, 3),
        "n_revisions": len(window),
        "n_upgrades": n_up,
        "n_downgrades": n_down,
        "avg_delta": round(float(avg_delta), 3),
        "summary": summary,
        "recent": window[:5],
    }


def analyst_signal(symbol: str) -> dict[str, Any]:
    """Combined signal: consensus + revisions blended 40/60.

    Revisions get more weight than consensus because they capture *change* —
    the part that hasn't been priced in yet.
    """
    cons = consensus_signal(symbol)
    revs = revisions_signal(symbol)
    score = 0.4 * cons["score"] + 0.6 * revs["score"]
    return {
        "score": round(float(score), 3),
        "consensus": cons,
        "revisions": revs,
        "summary": f"Consensus {cons['score']:+.2f}, revisions {revs['score']:+.2f} → combined {score:+.2f}",
    }
