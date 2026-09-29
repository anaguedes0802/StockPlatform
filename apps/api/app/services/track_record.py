"""Track record — the platform's own honest batting average.

Two halves:
  1. RECORD: every time the opportunity engine surfaces a name, snapshot it
     (symbol, strategy, score, entry_quality, price) — one row per
     (symbol, strategy, day).
  2. SCORE: a backfill that fills in forward returns once 1/4/12 weeks have
     elapsed, by comparing the price then vs the price at surface.

Then `summary()` aggregates by strategy: average forward return + win rate at
each horizon. This is what tells you whether 'catalyst_plays' actually makes
money or just looks clever — and lets the opportunity engine down-weight
strategies that don't perform.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import log
from app.db.models import OpportunityRecord
from app.services import market_data as md

import pandas as pd  # noqa: E402  (type hints in _closes)


def _dedupe_key(symbol: str, strategy: str) -> str:
    raw = f"{symbol}|{strategy}|{datetime.now(timezone.utc).date().isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def record_opportunities(db: Session, scan_result: dict[str, Any]) -> int:
    """Snapshot the opportunities from a scan. One row per (symbol, strategy, day)."""
    created = 0
    for opp in scan_result.get("opportunities", []):
        sym = opp.get("symbol")
        strat = opp.get("best_strategy")
        if not sym or not strat:
            continue
        key = _dedupe_key(sym, strat)
        exists = db.scalar(select(OpportunityRecord).where(OpportunityRecord.dedupe_key == key))
        if exists:
            continue
        db.add(OpportunityRecord(
            symbol=sym, strategy=strat, category=opp.get("best_category"),
            score=opp.get("score"), entry_quality=opp.get("entry_quality"),
            price_at_surface=opp.get("last_price"),
            dedupe_key=key,
        ))
        created += 1
    if created:
        db.commit()
    return created


HORIZONS = {"1w": 7, "4w": 28, "12w": 84}   # calendar days after surfacing


def _closes(symbol: str, cache: dict) -> "pd.Series | None":
    if symbol not in cache:
        try:
            df = md.get_history(symbol, interval="1d", range_="2y")
            s = df["close"].astype(float)
            s.index = s.index.tz_convert("UTC") if s.index.tz is not None else s.index.tz_localize("UTC")
            cache[symbol] = s.sort_index()
        except Exception:
            cache[symbol] = None
    return cache[symbol]


_spy_cache: tuple[float, "pd.Series"] | None = None


def _spy_closes(max_age_s: int = 3600) -> "pd.Series | None":
    """SPY daily closes, cached in-process (the benchmark is read on every
    summary() call; only successful fetches are cached, so a rate-limited
    request is retried next time instead of pinning a null benchmark)."""
    global _spy_cache
    import time as _t
    if _spy_cache and _t.time() - _spy_cache[0] < max_age_s:
        return _spy_cache[1]
    s = _closes("SPY", {})
    if s is not None and not s.empty:
        _spy_cache = (_t.time(), s)
    return s


def _close_at(series, when: datetime) -> float | None:
    """First daily close on or after `when` (the horizon date)."""
    if series is None or series.empty:
        return None
    pos = series.index.searchsorted(when)
    return float(series.iloc[pos]) if pos < len(series) else None


def score_matured(db: Session, max_rows: int = 200) -> int:
    """Backfill forward returns for records whose horizons have elapsed.

    Each horizon uses the close on/after surfaced_at + 7 / 28 / 84 days from
    daily history. (It used to stamp *today's* quote into every horizon that
    had elapsed by the time the job ran, so a record scored late showed the
    same return at 1w, 4w and 12w.) A record is done once its 12w return is set.
    """
    now = datetime.now(timezone.utc)
    rows = db.execute(
        select(OpportunityRecord)
        .where(OpportunityRecord.ret_12w.is_(None))
        .order_by(OpportunityRecord.surfaced_at.asc())
        .limit(max_rows)
    ).scalars().all()
    cache: dict = {}
    updated = 0
    for r in rows:
        surfaced = r.surfaced_at
        if surfaced.tzinfo is None:
            surfaced = surfaced.replace(tzinfo=timezone.utc)
        entry = float(r.price_at_surface) if r.price_at_surface else None
        if not entry:
            continue
        series = _closes(r.symbol, cache)
        touched = False
        for key, days in HORIZONS.items():
            col = f"ret_{key}"
            when = surfaced + timedelta(days=days)
            if getattr(r, col) is not None or when > now:
                continue
            px = _close_at(series, when)
            if px is None or px <= 0:
                continue
            setattr(r, col, (px - entry) / entry * 100)
            touched = True
        if touched:
            r.scored_at = now
            updated += 1
    if updated:
        db.commit()
    return updated


def rescore_all(db: Session) -> int:
    """Clear and recompute every stored horizon return (after the scoring fix)."""
    for r in db.execute(select(OpportunityRecord)).scalars():
        r.ret_1w = r.ret_4w = r.ret_12w = None
    db.commit()
    total = 0
    while True:
        n = score_matured(db, max_rows=5000)
        total += n
        if n == 0:
            return total


def summary(db: Session, days_lookback: int = 365) -> dict[str, Any]:
    """Aggregate performance by strategy. Win rate + avg return at each horizon."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    rows = db.execute(
        select(OpportunityRecord).where(OpportunityRecord.surfaced_at >= cutoff)
    ).scalars().all()

    # SPY over the identical window, so a name that fell 2% while the market
    # fell 5% counts as a win and one that rose 3% in a +8% market doesn't.
    spy = _spy_closes()

    def spy_ret(r, days: int) -> float | None:
        surfaced = r.surfaced_at if r.surfaced_at.tzinfo else r.surfaced_at.replace(tzinfo=timezone.utc)
        a, b = _close_at(spy, surfaced), _close_at(spy, surfaced + timedelta(days=days))
        return (b - a) / a * 100 if a and b else None

    by_strat: dict[str, dict[str, list[tuple[float, float | None]]]] = {}
    for r in rows:
        s = by_strat.setdefault(r.strategy, {"1w": [], "4w": [], "12w": []})
        for key, days in HORIZONS.items():
            v = getattr(r, f"ret_{key}")
            if v is not None:
                s[key].append((float(v), spy_ret(r, days)))

    def _stats(vals: list[tuple[float, float | None]]) -> dict[str, Any]:
        if not vals:
            return {"n": 0, "avg": None, "win_rate": None, "avg_vs_spy": None, "beat_spy_rate": None}
        rets = [v for v, _ in vals]
        ex = [v - b for v, b in vals if b is not None]
        return {"n": len(rets), "avg": round(sum(rets) / len(rets), 2),
                "win_rate": round(sum(1 for v in rets if v > 0) / len(rets) * 100, 0),
                "avg_vs_spy": round(sum(ex) / len(ex), 2) if ex else None,
                "beat_spy_rate": round(sum(1 for v in ex if v > 0) / len(ex) * 100, 0) if ex else None}

    out = []
    for strat, h in sorted(by_strat.items()):
        out.append({
            "strategy": strat,
            "1w": _stats(h["1w"]), "4w": _stats(h["4w"]), "12w": _stats(h["12w"]),
            "n_surfaced": db.scalar(
                select(OpportunityRecord).where(OpportunityRecord.strategy == strat).limit(1)
            ) is not None and len([r for r in rows if r.strategy == strat]),
        })
    total = len(rows)
    matured = sum(1 for r in rows if r.ret_1w is not None)
    return {
        "n_records": total,
        "n_matured_1w": matured,
        "by_strategy": out,
        "note": "Each horizon uses the close 7 / 28 / 84 days after the name was surfaced. "
                "avg_vs_spy = average return minus SPY over the same window; "
                "beat_spy_rate = % of surfaced names that beat SPY. Names surfaced repeatedly "
                "on consecutive days overlap, so treat small differences as noise.",
    }


# ---------------------------------------------------------------------------
# Learned strategy weights for the opportunity engine
# ---------------------------------------------------------------------------
# The engine's per-strategy weights were hand-set priors ("catalyst_plays =
# biggest alpha") that no data supported. These multipliers move each strategy
# toward what its own surfaced names actually did versus SPY, in proportion to
# how much *independent* evidence exists. Names re-surface daily, so evidence
# is counted in distinct surfacing weeks, not rows.

EVIDENCE_PRIOR_WEEKS = 12      # weeks of data at which evidence gets half the say
SENSITIVITY_PER_PCT = 0.08     # multiplier change per 1% avg excess vs SPY, at full evidence
MULT_BOUNDS = (0.5, 1.3)
VALIDATED_AFTER_WEEKS = 26     # ~6 months of distinct surfacing weeks

_weights_cache: tuple[float, dict[str, Any]] | None = None


def strategy_weights(db: Session, max_age_s: int = 3600) -> dict[str, Any]:
    """{"strategies": {name: {...multiplier...}}, "weeks_of_evidence", "validated", "message"}."""
    global _weights_cache
    import time as _t
    if _weights_cache and _t.time() - _weights_cache[0] < max_age_s:
        return _weights_cache[1]

    summ = summary(db, days_lookback=730)
    rows = db.execute(select(OpportunityRecord.strategy, OpportunityRecord.surfaced_at)
                      .where(OpportunityRecord.ret_4w.is_not(None))).all()
    weeks: dict[str, set] = {}
    for strat, ts in rows:
        weeks.setdefault(strat, set()).add(tuple(ts.isocalendar()[:2]))
    out: dict[str, Any] = {}
    for s in summ["by_strategy"]:
        name = s["strategy"]
        ex = [s[h]["avg_vs_spy"] for h in ("4w", "12w") if s[h].get("avg_vs_spy") is not None]
        n_weeks = len(weeks.get(name, ()))
        excess = sum(ex) / len(ex) if ex else 0.0
        evidence = n_weeks / (n_weeks + EVIDENCE_PRIOR_WEEKS)
        mult = min(MULT_BOUNDS[1], max(MULT_BOUNDS[0], 1 + SENSITIVITY_PER_PCT * evidence * excess))
        out[name] = {
            "multiplier": round(mult, 3),
            "avg_vs_spy_4w": s["4w"].get("avg_vs_spy"),
            "avg_vs_spy_12w": s["12w"].get("avg_vs_spy"),
            "beat_spy_rate_12w": s["12w"].get("beat_spy_rate"),
            "weeks_of_evidence": n_weeks,
            "evidence_weight": round(evidence, 2),
        }
    all_weeks = len(set().union(*weeks.values())) if weeks else 0
    validated = all_weeks >= VALIDATED_AFTER_WEEKS
    # overall record: surfaced-name-weighted average excess vs SPY at 12w
    n12 = [(s["12w"]["n"], s["12w"]["avg_vs_spy"]) for s in summ["by_strategy"]
           if s["12w"].get("avg_vs_spy") is not None]
    overall = (sum(n * v for n, v in n12) / sum(n for n, _ in n12)) if n12 else None
    if overall is None:
        verdict = "no matured outcomes yet"
    elif overall > 0:
        verdict = f"so far its picks beat SPY by {overall:.1f}% on average at 12 weeks"
    else:
        verdict = f"so far its picks trailed SPY by {-overall:.1f}% on average at 12 weeks"
    result = {
        "strategies": out,
        "weeks_of_evidence": all_weeks,
        "validated": validated,
        "overall_avg_vs_spy_12w": round(overall, 2) if overall is not None else None,
        "message": (f"Weights learned from {all_weeks} weeks of the list's own results; {verdict}."
                    if validated else
                    f"Not validated: only {all_weeks} distinct weeks of outcomes (need {VALIDATED_AFTER_WEEKS}); "
                    f"{verdict}. Treat the list as an idea screen, not buy signals."),
    }
    # don't pin neutral weights for an hour just because the SPY fetch failed
    if overall is not None or summ["n_matured_1w"] == 0:
        _weights_cache = (_t.time(), result)
    return result
