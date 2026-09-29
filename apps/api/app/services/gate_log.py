"""Forward-test log for the LLM risk-gate.

An LLM's trading judgement can't be backtested honestly: it has read about the
very events a backtest would replay. The only valid evidence is prospective,
so every verdict is written here the moment it is made, and scored later
against what the market actually did.

Scoring (`score_pending`) replays the *hypothetical* trade the quant signal
would have taken, with the strategy's own exit rules (same swing engine as the
backtests, costs included), and also records plain forward returns at 5/10/20
sessions from the next open. The report then asks one question: did the
trades the gate vetoed or downsized do worse than the ones it approved, by
more than chance?

Gate value per decision, in R (positive = the gate saved money):
    VETO      → −R            (you skipped a trade that made R)
    DOWNSIZE  → −(1 − m) · R  (you took a fraction m of it)
    APPROVE   → 0

Fail-open verdicts (provider unavailable / errored → APPROVE by default) are
logged but kept out of the evaluation: they are not the model's judgement.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.core.logging import log
from app.db.models import AiGateDecision

FAIL_OPEN_PROVIDERS = {"unavailable", "error", "stub", None, ""}
KIND_TO_SETUP = {"rsi2_meanrev": "rsi2", "swing_breakout": "breakout", "swing_pullback": "pullback"}
MIN_GATED_FOR_VERDICT = 20
HORIZONS = (5, 10, 20)

# Overridable in tests.
_session_factory = None
_ready_binds: set[int] = set()


def _sessions():
    if _session_factory is not None:
        return _session_factory
    from app.db.session import SessionLocal
    return SessionLocal


def ensure_table(db) -> None:
    bind = db.get_bind()
    if id(bind) in _ready_binds:
        return
    AiGateDecision.__table__.create(bind=bind, checkfirst=True)
    _ready_binds.add(id(bind))


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def record(symbol: str, signal: dict[str, Any], verdict: dict[str, Any], *,
           kind: str, source: str = "signal", bot_id: str | None = None) -> int | None:
    """Persist one gate verdict. Never raises — logging must not block trading."""
    try:
        as_of = str(signal.get("as_of") or datetime.now(timezone.utc).date().isoformat())[:10]
        row = AiGateDecision(
            symbol=symbol.upper(), as_of=as_of, kind=kind or "unknown", source=source,
            bot_id=bot_id, decision=str(verdict.get("decision", "APPROVE")),
            size_multiplier=_f(verdict.get("size_multiplier")), conviction=_f(verdict.get("conviction")),
            provider=verdict.get("provider"), rationale=(verdict.get("rationale") or "")[:4000],
            key_risks=list(verdict.get("key_risks") or [])[:20],
            signal_price=_f(signal.get("price")), stop_dist=_f(signal.get("stop_dist")),
        )
        with _sessions()() as db:
            ensure_table(db)
            db.add(row)
            db.commit()
            return int(row.id)
    except Exception as e:  # noqa: BLE001
        log.warning("gate_log_record_failed", symbol=symbol, err=str(e)[:200])
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def outcome(symbol: str, as_of: str, kind: str, df: pd.DataFrame,
            earnings: list[pd.Timestamp] | None = None) -> dict[str, Any] | None:
    """Forward returns + hypothetical trade result for a signal on `as_of`.

    `df` must be completed daily bars reaching back ≥ 1 year before `as_of`
    (indicator warm-up). Returns None when the next session hasn't happened.
    `trade_*` keys are absent while the hypothetical trade is still open.
    """
    from app.backtest import swing_engine as eng
    from app.ml.signal_filter import _label_config
    from app.services import swing

    if df.empty:
        return None
    ts = pd.Timestamp(as_of, tz="UTC")
    i = int(df.index.searchsorted(ts, side="right")) - 1   # last bar at or before as_of
    if i < 0 or i + 1 >= len(df):
        return None
    entry = float(df["open"].iloc[i + 1])
    out: dict[str, Any] = {"entry_price": entry}
    for h in HORIZONS:
        j = i + h
        out[f"ret_{h}"] = (float(df["close"].iloc[j]) / entry - 1) if j < len(df) else None

    setup = KIND_TO_SETUP.get(kind, "rsi2")
    sig = swing.build_signals(df, setup)
    if not np.isfinite(sig["stop_dist_long"].iloc[i]):
        return out  # indicators not warm → returns only
    sig.loc[:, ["long_entry", "short_entry"]] = False
    sig.iloc[i, sig.columns.get_loc("long_entry")] = True
    cfg = _label_config(swing.config_for("equity"))
    cfg.earnings_blackout_days = 0  # the entry already happened; exits still respect reports
    res = eng.simulate({symbol: df}, {symbol: sig}, swing.setup_rules(setup), cfg,
                       earnings={symbol: earnings}, mc_sims=0)
    if not res.trades:
        return out
    t = res.trades[0]
    if t.exit_reason == "end_of_data":
        return out  # still open
    out.update({"trade_r": t.r_multiple, "trade_return_pct": t.return_pct,
                "trade_exit_reason": t.exit_reason, "trade_bars": t.bars_held})
    return out


def score_pending(*, limit: int = 300, now: datetime | None = None, loader=None,
                  earnings_loader=None) -> dict[str, int]:
    """Backfill outcomes for decisions whose trade may have resolved."""
    from app.services import swing
    from app.services import swing_data as sd

    now = now or datetime.now(timezone.utc)
    loader = loader or (lambda sym, start: swing.completed_bars(sd.daily_bars(sym, start=start, max_age_s=6 * 3600), sym))
    earnings_loader = earnings_loader or sd.earnings_dates
    scored = partial = 0
    with _sessions()() as db:
        ensure_table(db)
        rows = db.scalars(
            select(AiGateDecision)
            .where(AiGateDecision.scored_at.is_(None),
                   AiGateDecision.created_at < now - timedelta(hours=20))
            .order_by(AiGateDecision.created_at).limit(limit)
        ).all()
        by_sym: dict[str, list[AiGateDecision]] = {}
        for r in rows:
            by_sym.setdefault(r.symbol, []).append(r)
        for sym, rs in by_sym.items():
            start = (pd.Timestamp(min(r.as_of for r in rs), tz="UTC") - pd.Timedelta(days=420)).date().isoformat()
            try:
                df = loader(sym, start)
                earn = earnings_loader(sym)
            except Exception as e:  # noqa: BLE001
                log.warning("gate_score_load_failed", symbol=sym, err=str(e)[:200])
                continue
            for r in rs:
                o = outcome(sym, r.as_of, r.kind, df, earn)
                if not o:
                    continue
                for k, v in o.items():
                    setattr(r, k, v)
                if o.get("trade_r") is not None:
                    r.scored_at = now
                    scored += 1
                else:
                    partial += 1
        db.commit()
    return {"scored": scored, "partial": partial}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _row_dict(r: AiGateDecision) -> dict[str, Any]:
    return {
        "id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None,
        "symbol": r.symbol, "as_of": r.as_of, "kind": r.kind, "source": r.source,
        "decision": r.decision, "size_multiplier": _f(r.size_multiplier),
        "conviction": _f(r.conviction), "provider": r.provider, "rationale": r.rationale,
        "key_risks": r.key_risks or [], "signal_price": _f(r.signal_price),
        "entry_price": _f(r.entry_price), "ret_5": _f(r.ret_5), "ret_10": _f(r.ret_10),
        "ret_20": _f(r.ret_20), "trade_r": _f(r.trade_r), "trade_return_pct": _f(r.trade_return_pct),
        "trade_exit_reason": r.trade_exit_reason, "trade_bars": r.trade_bars,
        "scored": r.scored_at is not None,
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One judged decision per (symbol, signal bar, strategy): the first."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda x: (x["created_at"] or "", x["id"])):
        k = (r["symbol"], r["as_of"], r["kind"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def gate_value(r: dict[str, Any]) -> float:
    R = r.get("trade_r") or 0.0
    if r["decision"] == "VETO":
        return -R
    if r["decision"] == "DOWNSIZE":
        m = r.get("size_multiplier")
        m = 1.0 if m is None else min(max(m, 0.0), 1.0)
        return -(1.0 - m) * R
    return 0.0


def summarize(rows: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 3) -> dict[str, Any]:
    """Aggregate decisions into the forward-test verdict (pure; testable)."""
    judged = [r for r in rows if r.get("provider") not in FAIL_OPEN_PROVIDERS]
    fail_open = len(rows) - len(judged)
    judged = _dedupe(judged)
    scored = [r for r in judged if r.get("trade_r") is not None]

    def stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        R = [r["trade_r"] for r in rs]
        r10 = [r["ret_10"] for r in rs if r.get("ret_10") is not None]
        return {"n": len(rs),
                "win_rate_pct": round(float(np.mean([x > 0 for x in R]) * 100), 1) if R else None,
                "mean_r": round(float(np.mean(R)), 3) if R else None,
                "mean_ret_10_pct": round(float(np.mean(r10) * 100), 2) if r10 else None}

    by = {d: [r for r in scored if r["decision"] == d] for d in ("APPROVE", "DOWNSIZE", "VETO")}
    gated = by["DOWNSIZE"] + by["VETO"]
    vals = np.array([gate_value(r) for r in gated])
    value = {"n": int(len(vals)), "total_r": round(float(vals.sum()), 2) if len(vals) else 0.0,
             "mean_r": round(float(vals.mean()), 3) if len(vals) else None, "lo": None, "hi": None}
    if len(vals) >= 5:
        rng = np.random.default_rng(seed)
        boots = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(axis=1)
        value["lo"], value["hi"] = (round(float(x), 3) for x in np.percentile(boots, [5, 95]))

    if len(gated) < MIN_GATED_FOR_VERDICT:
        verdict = {"label": "collecting", "summary": (
            f"{len(gated)} scored vetoes/downsizes so far; {MIN_GATED_FOR_VERDICT} are needed before the "
            "numbers mean anything. Until then the gate is an unproven opinion.")}
    elif value["lo"] is not None and value["lo"] > 0:
        verdict = {"label": "helping", "summary": "Trades the gate cut did worse than chance would explain: it is saving money."}
    elif value["hi"] is not None and value["hi"] < 0:
        verdict = {"label": "hurting", "summary": "Trades the gate cut went on to make money: it is costing you. Turn it off."}
    else:
        verdict = {"label": "inconclusive", "summary": "No detectable effect yet: the gate's cuts look like random skips."}

    return {
        "counts": {"logged": len(rows), "fail_open": fail_open, "judged": len(judged),
                   "scored": len(scored), "pending": len(judged) - len(scored),
                   **{d.lower(): sum(1 for r in judged if r["decision"] == d)
                      for d in ("APPROVE", "DOWNSIZE", "VETO")}},
        "outcomes": {d.lower(): stats(rs) for d, rs in by.items()},
        "gate_value": value,
        "verdict": verdict,
    }


def report(*, limit_recent: int = 50, score: bool = True) -> dict[str, Any]:
    if score:
        try:
            score_pending()
        except Exception as e:  # noqa: BLE001 — a scoring hiccup shouldn't hide the log
            log.warning("gate_score_failed", err=str(e)[:200])
    with _sessions()() as db:
        ensure_table(db)
        rows = [_row_dict(r) for r in db.scalars(select(AiGateDecision).order_by(AiGateDecision.id)).all()]
    out = summarize(rows)
    out["recent"] = list(reversed(rows[-limit_recent:]))
    out["method"] = ("Each verdict is logged when made. After the fact, the trade the quant signal "
                     "would have taken is replayed with the strategy's own exits (costs included) to get "
                     "its R-multiple. Gate value = R avoided by vetoes and downsizes.")
    return out
