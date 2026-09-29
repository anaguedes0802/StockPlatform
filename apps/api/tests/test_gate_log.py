"""AI risk-gate forward-test log: recording, scoring, verdict."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from app.db.models import AiGateDecision
from app.services import gate_log
from app.services import trading_bot as bot


def _rows(factory) -> list[AiGateDecision]:
    with factory() as db:
        gate_log.ensure_table(db)
        return db.scalars(select(AiGateDecision).order_by(AiGateDecision.id)).all()


def test_record_persists_verdict(_isolated_gate_log) -> None:
    rid = gate_log.record("aaa", {"as_of": "2026-09-28T00:00:00", "price": 50.0, "stop_dist": 2.0},
                          {"decision": "VETO", "size_multiplier": 0.0, "conviction": 0.8,
                           "rationale": "fraud headline", "key_risks": ["legal"], "provider": "groq:x"},
                          kind="swing_breakout", source="run", bot_id="b1")
    assert rid is not None
    r = _rows(_isolated_gate_log)[0]
    assert (r.symbol, r.as_of, r.decision, r.source, r.kind) == ("AAA", "2026-09-28", "VETO", "run", "swing_breakout")
    assert float(r.stop_dist) == 2.0 and r.key_risks == ["legal"]


def test_record_never_raises(monkeypatch) -> None:
    def broken():
        raise RuntimeError("db down")
    monkeypatch.setattr(gate_log, "_session_factory", broken)
    assert gate_log.record("AAA", {}, {"decision": "APPROVE"}, kind="x") is None


def test_llm_gate_verdicts_are_logged(monkeypatch, _isolated_gate_log) -> None:
    from tests.test_trading_bot import _uptrend_with_final_dip
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    monkeypatch.setattr(bot, "_market_uptrend", lambda *a, **k: None)
    monkeypatch.setattr(bot.bot_intel, "llm_review", lambda s, o: {
        "decision": "DOWNSIZE", "size_multiplier": 0.5, "conviction": 0.6,
        "rationale": "r", "key_risks": [], "provider": "gemini:x"})
    sig = bot.latest_signal("FAKE", llm_gate=True, context={"source": "scan", "bot_id": "b9"})
    assert sig["action"] == "BUY"
    r = _rows(_isolated_gate_log)[0]
    assert (r.decision, r.source, r.bot_id, r.kind) == ("DOWNSIZE", "scan", "b9", "rsi2_meanrev")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _uptrend_then(n=320, jump_at=None) -> pd.DataFrame:
    # Small alternating wiggle: a perfectly monotonic series has no down days,
    # which leaves RSI undefined (real data always has them).
    c = np.linspace(80, 100, n) + np.where(np.arange(n) % 2, 0.05, -0.05)
    o = np.r_[c[0], c[:-1]]
    h, l = c * 1.001, c * 0.999
    if jump_at is not None:
        h[jump_at] = c[jump_at] * 1.05          # target (+1%) touched the day after entry
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6}, index=idx)


def test_outcome_replays_the_hypothetical_trade() -> None:
    df = _uptrend_then(jump_at=301)             # target touched on the entry bar
    as_of = df.index[300].date().isoformat()
    o = gate_log.outcome("AAA", as_of, "rsi2_meanrev", df)
    assert o["entry_price"] == pytest.approx(float(df["open"].iloc[301]))
    assert o["ret_5"] == pytest.approx(float(df["close"].iloc[305]) / o["entry_price"] - 1)
    assert o["trade_exit_reason"] == "target" and o["trade_r"] > 0


def test_outcome_leaves_open_trades_unscored() -> None:
    df = _uptrend_then()
    o = gate_log.outcome("AAA", df.index[-3].date().isoformat(), "swing_breakout", df)
    assert "trade_r" not in o          # trailing trend trade still open at the last bar
    assert gate_log.outcome("AAA", df.index[-1].date().isoformat(), "swing_breakout", df) is None


def test_score_pending_backfills(_isolated_gate_log) -> None:
    df = _uptrend_then(jump_at=301)
    gate_log.record("AAA", {"as_of": df.index[300].date().isoformat(), "price": 99.0},
                    {"decision": "VETO", "provider": "groq:x"}, kind="rsi2_meanrev")
    later = datetime.now(timezone.utc) + timedelta(days=2)
    res = gate_log.score_pending(now=later, loader=lambda s, start: df, earnings_loader=lambda s: None)
    assert res == {"scored": 1, "partial": 0}
    r = _rows(_isolated_gate_log)[0]
    assert r.scored_at is not None and float(r.trade_r) > 0


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def _d(i, decision, r, provider="groq:x", sym=None, m=None):
    return {"id": i, "created_at": f"2026-01-{1 + i % 28:02d}T00:00:{i % 60:02d}", "symbol": sym or f"S{i}",
            "as_of": "2026-01-01", "kind": "rsi2_meanrev", "decision": decision, "provider": provider,
            "trade_r": r, "ret_10": r / 100 if r is not None else None, "size_multiplier": m}


def test_gate_value_signs() -> None:
    assert gate_log.gate_value(_d(1, "VETO", -1.0)) == 1.0          # avoided a loser
    assert gate_log.gate_value(_d(1, "VETO", 2.0)) == -2.0          # skipped a winner
    assert gate_log.gate_value(_d(1, "DOWNSIZE", -1.0, m=0.5)) == 0.5
    assert gate_log.gate_value(_d(1, "APPROVE", -1.0)) == 0.0


def test_summary_collects_until_enough_evidence() -> None:
    rows = [_d(i, "VETO", -1.0) for i in range(5)] + [_d(100 + i, "APPROVE", 0.5) for i in range(20)]
    s = gate_log.summarize(rows)
    assert s["verdict"]["label"] == "collecting"
    assert s["counts"]["veto"] == 5


def test_summary_detects_helpful_and_harmful_gates() -> None:
    helpful = [_d(i, "VETO", -1.0 - (i % 3) * 0.1) for i in range(30)]
    assert gate_log.summarize(helpful)["verdict"]["label"] == "helping"
    harmful = [_d(i, "VETO", 1.0 + (i % 3) * 0.1) for i in range(30)]
    assert gate_log.summarize(harmful)["verdict"]["label"] == "hurting"


def test_summary_excludes_fail_open_and_duplicates() -> None:
    rows = [_d(1, "APPROVE", 0.3, provider="unavailable"),
            _d(2, "VETO", -1.0, sym="X"), _d(3, "APPROVE", -1.0, sym="X")]   # same signal re-reviewed
    s = gate_log.summarize(rows)
    assert s["counts"]["fail_open"] == 1
    assert s["counts"]["judged"] == 1 and s["counts"]["veto"] == 1
