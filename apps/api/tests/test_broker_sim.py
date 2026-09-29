"""Simulated paper broker + per-bot broker selection for A/B paper tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.services import bot_autorun, broker_sim
from app.services import trading_bot as bot

T0 = datetime(2026, 10, 1, 13, 45, tzinfo=timezone.utc)   # 09:45 ET


def _sim(prices: dict[str, float], bars: dict[str, pd.DataFrame] | None = None, now=T0, ledger=None,
         **kw) -> broker_sim.SimBroker:
    return broker_sim.SimBroker(
        ledger, quote_fn=lambda s: prices[s], clock_fn=lambda: True, now_fn=lambda: now,
        intraday_fn=lambda s, since: (bars or {}).get(s, pd.DataFrame()), **kw)


def _five_min(start: datetime, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# ledger mechanics
# ---------------------------------------------------------------------------

def test_market_buy_and_close_books_costs_and_pl() -> None:
    s = _sim({"AAA": 100.0}, slippage_bps=10, commission_bps=10)
    s.submit_market_order("AAA", qty=10)
    pos = s.ledger["positions"]["AAA"]
    assert pos["avg_entry_price"] == pytest.approx(100.1)               # paid the slippage
    assert s.ledger["cash"] == pytest.approx(100_000 - 10 * 100.1 * 1.001)
    s._px_cache["AAA"] = 110.0
    out = s.close_position("AAA")
    expected = 10 * 110 * 0.999 * 0.999 - 10 * 100.1
    assert out["realized_pl"] == pytest.approx(expected, abs=0.01)
    assert s.ledger["positions"] == {}


def test_insufficient_cash_is_rejected() -> None:
    s = _sim({"AAA": 100.0}, initial_cash=500.0)
    with pytest.raises(ValueError):
        s.submit_market_order("AAA", qty=10)


def test_account_and_positions_mark_to_market() -> None:
    s = _sim({"AAA": 50.0})
    s.submit_market_order("AAA", notional=5_000)
    s._px_cache["AAA"] = 55.0
    acct = s.get_account()
    pos = s.list_positions()[0]
    assert pos["unrealized_pl"] > 0 and pos["current_price"] == 55.0
    assert acct["equity"] == pytest.approx(s.ledger["cash"] + pos["qty"] * 55.0, abs=0.01)


# ---------------------------------------------------------------------------
# resting bracket legs replayed on 5-minute bars
# ---------------------------------------------------------------------------

def test_target_leg_fills_on_later_bar() -> None:
    s = _sim({"AAA": 100.0}, slippage_bps=0, commission_bps=0)
    s.submit_bracket_order("AAA", qty=10, stop_loss_price=75.0, take_profit_price=101.0)
    later = T0 + timedelta(days=1)
    bars = _five_min(T0 + timedelta(minutes=5), [(100, 100.5, 99.8, 100.2), (100.2, 101.4, 100.1, 101.2)])
    s2 = _sim({"AAA": 101.0}, {"AAA": bars}, now=later, ledger=s.ledger, slippage_bps=0, commission_bps=0)
    fills = s2.process_resting_orders()
    assert [f["reason"] for f in fills] == ["target"]
    assert fills[0]["price"] == pytest.approx(101.0)
    assert fills[0]["realized_pl"] == pytest.approx(10.0)


def test_stop_gap_fills_at_open_and_stop_wins_ties() -> None:
    s = _sim({"AAA": 100.0}, slippage_bps=0, commission_bps=0)
    s.submit_bracket_order("AAA", qty=10, stop_loss_price=95.0, take_profit_price=101.0)
    bars = _five_min(T0 + timedelta(minutes=5), [(93.0, 102.0, 92.0, 94.0)])   # gaps below stop, also tags target
    s2 = _sim({"AAA": 94.0}, {"AAA": bars}, now=T0 + timedelta(days=1), ledger=s.ledger,
              slippage_bps=0, commission_bps=0)
    fills = s2.process_resting_orders()
    assert fills[0]["reason"] == "stop" and fills[0]["price"] == pytest.approx(93.0)


def test_bars_before_fill_and_forming_bars_are_ignored() -> None:
    s = _sim({"AAA": 100.0}, slippage_bps=0, commission_bps=0)
    s.submit_bracket_order("AAA", qty=10, stop_loss_price=95.0, take_profit_price=None)
    before = _five_min(T0 - timedelta(minutes=10), [(96, 96, 90, 96)])            # dipped before we owned it
    forming = _five_min(T0 + timedelta(minutes=5), [(99, 99, 94, 94)])            # still open at "now"
    bars = pd.concat([before, forming])
    s2 = _sim({"AAA": 94.0}, {"AAA": bars}, now=T0 + timedelta(minutes=7), ledger=s.ledger)
    assert s2.process_resting_orders() == []
    assert "AAA" in s2.ledger["positions"]


def test_replace_and_cancel_legs() -> None:
    s = _sim({"AAA": 100.0})
    o = s.submit_bracket_order("AAA", qty=5, stop_loss_price=90.0, take_profit_price=110.0)
    stop_id = next(g["id"] for g in o["legs"] if g["type"] == "stop")
    s.replace_stop(stop_id, 97.0)
    assert s.ledger["positions"]["AAA"]["stop"] == 97.0
    assert s.cancel_open_orders("AAA") == 2
    assert s.ledger["positions"]["AAA"]["stop"] is None


def test_summary_uses_stored_prices_only() -> None:
    s = _sim({"AAA": 100.0})
    s.submit_market_order("AAA", qty=10)
    s.mark()
    sm = broker_sim.summary(s.ledger)
    assert sm["positions"][0]["symbol"] == "AAA"
    assert sm["equity_history"][0]["date"] == "2026-10-01"
    assert sm["closed_trades"] == 0


# ---------------------------------------------------------------------------
# production path: run_live on a sim ledger
# ---------------------------------------------------------------------------

SIM_EX = {"broker": "sim", "max_position_usd": 12_500.0, "max_open_positions": 8,
          "max_gross_exposure_usd": 100_000.0, "max_daily_loss_usd": 5_000.0, "max_daily_orders": 20}


def test_execute_pass_on_sim_opens_bracket_and_persists_ledger(monkeypatch) -> None:
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 100.0, "reason": "test"})
    factory = lambda ledger, initial_cash: _sim({"AAA": 100.0}, ledger=ledger, initial_cash=initial_cash)
    res = bot.execute_pass(universe=["AAA"], dsl={"llm_gate": False}, execution=SIM_EX,
                           run_state=None, sim_factory=factory)
    assert res["mode"] == "sim" and res["blocked"] is None
    led = res["run_state"]["sim_ledger"]
    pos = led["positions"]["AAA"]
    assert pos["qty"] == 125                          # floor($12.5k cap / $100 signal price)
    assert pos["target"] == pytest.approx(101.0) and pos["stop"] == pytest.approx(75.0)
    assert res["run_state"]["positions"]["AAA"]["protective"] == "broker"
    assert led["equity_history"]


def test_two_sim_bots_have_independent_ledgers(monkeypatch) -> None:
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 100.0, "reason": "test"})
    factory = lambda ledger, initial_cash: _sim({"AAA": 100.0}, ledger=ledger, initial_cash=initial_cash)
    a = bot.execute_pass(universe=["AAA"], dsl=None, execution=SIM_EX, run_state=None, sim_factory=factory)
    b = bot.execute_pass(universe=["AAA"], dsl=None, execution=SIM_EX, run_state=None, sim_factory=factory)
    assert "AAA" in a["run_state"]["sim_ledger"]["positions"]
    assert "AAA" in b["run_state"]["sim_ledger"]["positions"]   # no collision on the same symbol


def test_broker_side_target_is_reconciled_next_pass(monkeypatch) -> None:
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 100.0, "reason": "test"})
    first = bot.execute_pass(universe=["AAA"], dsl=None, execution=SIM_EX, run_state=None,
                             sim_factory=lambda l, initial_cash: _sim({"AAA": 100.0}, ledger=l))
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {"symbol": sym, "action": "FLAT"})
    bars = _five_min(T0 + timedelta(minutes=5), [(100.2, 101.5, 100.1, 101.3)])
    second = bot.execute_pass(
        universe=["AAA"], dsl=None, execution=SIM_EX, run_state=first["run_state"],
        sim_factory=lambda l, initial_cash: _sim({"AAA": 101.3}, {"AAA": bars}, now=T0 + timedelta(days=1), ledger=l))
    assert [f["reason"] for f in second["resting_fills"]] == ["target"]
    assert any(a["action"] == "CLOSED_BY_BROKER" for a in second["actions"])
    assert second["run_state"]["sim_ledger"]["positions"] == {}


def test_alpaca_bots_unchanged(monkeypatch) -> None:
    from tests.test_trading_bot import FakeBroker
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 100.0, "reason": "test"})
    res = bot.execute_pass(universe=["AAA"], dsl=None, execution={"max_position_usd": 500.0},
                           run_state=None, alpaca=fb)
    assert res["mode"] == "paper" and len(fb.orders) == 1


# ---------------------------------------------------------------------------
# completed-bar signals
# ---------------------------------------------------------------------------

def test_completed_bars_signal_uses_swing_data_and_live_price(monkeypatch) -> None:
    from tests.test_trading_bot import _uptrend_with_final_dip
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.swing_mod, "_recent_bars", lambda s: df)
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: (_ for _ in ()).throw(AssertionError("IEX path used")))
    monkeypatch.setattr(bot.md, "get_quote", lambda s: {"price": 171.0})
    sig = bot.latest_signal("FAKE", {"completed_bars": True, "market_filter": False}, llm_gate=False)
    assert sig["action"] == "BUY"
    assert sig["price"] == 171.0 and sig["signal_close"] == pytest.approx(float(df["close"].iloc[-1]), abs=1e-3)


# ---------------------------------------------------------------------------
# auto-run is per-bot opt-in
# ---------------------------------------------------------------------------

def test_autorun_requires_bot_opt_in(monkeypatch) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.models import TradingBot, User

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    User.__table__.create(eng)
    TradingBot.__table__.create(eng)
    factory = sessionmaker(bind=eng, expire_on_commit=False, class_=Session)
    monkeypatch.setattr(bot_autorun, "SessionLocal", factory)
    ran = []
    monkeypatch.setattr(bot_autorun.botsvc, "execute_pass",
                        lambda **k: ran.append(k["context"]["bot_id"]) or {"run_state": {}, "blocked": None, "actions": []})
    with factory() as db:
        u = User(email="t@example.com", password_hash="x")
        db.add(u)
        db.flush()
        for name, ex in (("opted", {"broker": "sim", "autorun": True}), ("not", {"broker": "sim"})):
            db.add(TradingBot(user_id=u.id, name=name, bot_type="trader", dsl={}, universe=["AAA"],
                              status="active", mode="paper", armed=True, execution=ex, run_state={}))
        db.commit()
        ids = {b.name: str(b.id) for b in db.query(TradingBot).all()}
    monkeypatch.setenv("BOT_AUTORUN_ET", "09:45")
    bot_autorun.run_due(datetime(2026, 9, 29, 13, 50, tzinfo=timezone.utc))
    assert ran == [ids["opted"]]
