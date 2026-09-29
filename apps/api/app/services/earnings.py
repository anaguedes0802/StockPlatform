"""Earnings calendar — past + upcoming dates, EPS estimates vs actuals.

Source: yfinance Ticker.earnings_dates (free, ~next/last 4 quarters typical).
Cached aggressively because calendar entries don't change often.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
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


_DISK = Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "earnings"


def _disk_get(key: str, max_age_s: int) -> Any | None:
    """Persistent fallback cache (Redis is optional in dev). Earnings calendars
    change quarterly, so a few days' staleness is harmless."""
    p = _DISK / (key.replace(":", "_") + ".json")
    try:
        if p.exists() and time.time() - p.stat().st_mtime < max_age_s:
            return json.loads(p.read_text())
    except Exception:
        return None
    return None


def _disk_set(key: str, value: Any) -> None:
    try:
        _DISK.mkdir(parents=True, exist_ok=True)
        (_DISK / (key.replace(":", "_") + ".json")).write_text(json.dumps(value, default=str))
    except Exception:
        pass


def earnings_dates(symbol: str, limit: int = 12) -> list[dict[str, Any]]:
    """Recent + upcoming earnings dates with EPS estimates and actuals.

    Returns list of dicts (most recent / upcoming first):
      {ts, eps_estimate, eps_actual, surprise_pct, status: "upcoming"|"reported"}
    """
    key = f"earn:{symbol}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    if (cached := _disk_get(key, max_age_s=3 * 86400)) is not None:
        return cached
    out: list[dict[str, Any]] = []
    t = _yf_ticker(symbol)
    try:
        df = t.earnings_dates
    except Exception as e:
        # Rate limits / scrape failures must NOT be cached as "no earnings":
        # that silently zeroed the earnings factor for hours.
        log.warning("earnings_dates_failed", symbol=symbol, err=str(e)[:120])
        return []
    try:
        if df is None or df.empty:
            _cache_set(key, [], ttl=6 * 3600)
            return []
        # df is indexed by date; columns typically: 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
        for ts, row in df.iterrows():
            try:
                est = row.get("EPS Estimate")
                act = row.get("Reported EPS")
                surp = row.get("Surprise(%)")
                ts_iso = pd.Timestamp(ts).isoformat()
                status = "reported" if (act is not None and pd.notna(act)) else "upcoming"
                out.append({
                    "ts": ts_iso,
                    "eps_estimate": float(est) if est is not None and pd.notna(est) else None,
                    "eps_actual":   float(act) if act is not None and pd.notna(act) else None,
                    "surprise_pct": float(surp) if surp is not None and pd.notna(surp) else None,
                    "status": status,
                })
            except Exception:
                continue
    except Exception:
        pass
    # newest first
    out.sort(key=lambda x: x["ts"], reverse=True)
    out = out[:limit]
    _cache_set(key, out, ttl=6 * 3600)
    _disk_set(key, out)
    return out


def earnings_history(symbol: str, limit: int = 60) -> list[dict[str, Any]]:
    """Up to `limit` past announcements (yfinance goes back ~20 years), oldest
    first: {ts, eps_estimate, eps_actual, surprise_pct}. Used by the
    stock-selection backtest; needs `lxml` (yfinance scrapes this table)."""
    key = f"earn_hist:{symbol}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    if (cached := _disk_get(key, max_age_s=7 * 86400)) is not None:
        return cached
    out: list[dict[str, Any]] = []
    try:
        df = _yf_ticker(symbol).get_earnings_dates(limit=limit)
    except Exception as e:
        log.warning("earnings_history_failed", symbol=symbol, err=str(e)[:120])
        return []   # not cached: a rate limit is not "no history"
    if df is not None and not df.empty:
        for ts, row in df.iterrows():
            act = row.get("Reported EPS")
            if act is None or pd.isna(act):
                continue
            est, surp = row.get("EPS Estimate"), row.get("Surprise(%)")
            out.append({
                "ts": pd.Timestamp(ts).isoformat(),
                "eps_estimate": float(est) if est is not None and pd.notna(est) else None,
                "eps_actual": float(act),
                "surprise_pct": float(surp) if surp is not None and pd.notna(surp) else None,
            })
    out.sort(key=lambda x: x["ts"])
    _cache_set(key, out, ttl=24 * 3600)
    if out:
        _disk_set(key, out)
    return out


def next_earnings(symbol: str) -> dict[str, Any] | None:
    """The single next-upcoming earnings event for a symbol, with days_to_earnings."""
    now = datetime.now(timezone.utc)
    rows = earnings_dates(symbol, limit=8)
    upcoming = sorted([r for r in rows if r["status"] == "upcoming"], key=lambda r: r["ts"])
    if not upcoming:
        return None
    nxt = upcoming[0]
    try:
        when = pd.Timestamp(nxt["ts"])
        if when.tz is None:
            when = when.tz_localize("UTC")
        days = (when - pd.Timestamp(now)).total_seconds() / 86400
        return {**nxt, "days_to_earnings": round(days, 1)}
    except Exception:
        return nxt


def earnings_event_flag(symbol: str, days_window: int = 7) -> float:
    """Returns 1.0 if an earnings event is within ±days_window days, else 0.0.

    Used as a feature in the ML pipeline — markets are systematically more
    volatile around earnings, so the model should know.
    """
    nxt = next_earnings(symbol)
    if not nxt:
        return 0.0
    d = nxt.get("days_to_earnings")
    if d is None:
        return 0.0
    return 1.0 if abs(d) <= days_window else 0.0


def pead_signal(symbol: str) -> dict[str, Any]:
    """Post-Earnings Announcement Drift signal.

    Bernard & Thomas (1989) and subsequent work: stocks that beat consensus
    continue to drift positively for ~60 days; misses drift negatively. The
    effect is strongest in the first 5 days.

    We compute:
      - latest reported surprise % (already in consensus_surprise_signal)
      - days since that earnings event
      - a decaying expectation = sign(surprise) × |surprise|^0.5 × decay(days)
        where decay(d) = max(0, 1 − d/60)

    Score ∈ [-1, +1].
    """
    rows = earnings_dates(symbol, limit=8)
    reported = [r for r in rows if r["status"] == "reported"
                 and r.get("surprise_pct") is not None]
    if not reported:
        return {"score": 0.0, "summary": "No recent reported earnings."}
    latest = sorted(reported, key=lambda r: r["ts"], reverse=True)[0]
    try:
        ts = datetime.fromisoformat(str(latest["ts"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return {"score": 0.0, "summary": "Could not parse earnings date."}

    days_since = (datetime.now(timezone.utc) - ts).days
    if days_since < 0 or days_since > 90:
        return {"score": 0.0, "summary": f"Last earnings {days_since}d ago — outside PEAD window.",
                "days_since_earnings": days_since}

    surp = float(latest["surprise_pct"])  # in %
    # Edge is strongest in first 5d, decays linearly to 0 at 60d
    decay = max(0.0, 1.0 - days_since / 60.0)
    import math
    magnitude = math.sqrt(abs(surp) / 10.0)  # ±10% surprise → magnitude 1
    raw = (1 if surp > 0 else -1) * magnitude * decay
    score = max(-1.0, min(1.0, raw))

    label = "positive drift expected" if surp > 0 else ("negative drift expected" if surp < 0 else "neutral")
    return {
        "score": round(score, 3),
        "latest_surprise_pct": surp,
        "days_since_earnings": days_since,
        "decay_factor": round(decay, 3),
        "summary": f"PEAD: last beat/miss {surp:+.1f}% {days_since}d ago — {label}",
    }


def consensus_surprise_signal(symbol: str) -> dict[str, Any]:
    """Score recent earnings surprises — beats are bullish, misses bearish.

    Returns signal_score in [-1, +1] from the last 4 reports' weighted surprises.
    """
    rows = earnings_dates(symbol, limit=8)
    reported = [r for r in rows if r["status"] == "reported" and r.get("surprise_pct") is not None]
    if not reported:
        return {"score": 0.0, "summary": "No earnings history available."}
    # Take the most recent 4 reports
    recent = sorted(reported, key=lambda r: r["ts"], reverse=True)[:4]
    # Weight: most-recent gets 0.4, then 0.3, 0.2, 0.1
    weights = [0.4, 0.3, 0.2, 0.1][:len(recent)]
    weights = [w / sum(weights) for w in weights]
    import math
    score = sum(math.tanh(r["surprise_pct"] / 10.0) * w for r, w in zip(recent, weights))
    # Build a one-line summary
    parts = []
    for r in recent[:3]:
        parts.append(f"{r['ts'][:10]}: {r['surprise_pct']:+.1f}%")
    return {
        "score": round(float(score), 3),
        "summary": "Recent earnings surprises — " + "; ".join(parts),
        "n_reports": len(recent),
        "latest_surprise_pct": recent[0]["surprise_pct"],
    }
