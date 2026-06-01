"""Tests for the automated trading bot — trade-level engine + strategy signals.

All offline: the simulator takes a DataFrame directly, and the network-bound
helpers are exercised with a monkeypatched price history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.bot_engine import ExitRules, simulate
from app.services import trading_bot as bot


def _bars(closes: list[float], *, highs=None, lows=None, opens=None) -> pd.DataFrame:
    n = len(closes)
    close = np.array(closes, dtype=float)
    op = np.array(opens, dtype=float) if opens else np.r_[close[0], close[:-1]]
    high = np.array(highs, dtype=float) if highs else close * 1.001
    low = np.array(lows, dtype=float) if lows else close * 0.999
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": np.full(n, 1_000_000)}, index=idx)


def test_winning_trade_counts_as_win() -> None:
    # Enter at bar 1 open (=100), exit on signal at bar 2 close (=110).
    df = _bars([100, 100, 110, 110], opens=[100, 100, 110, 110])
    entries = pd.Series([True, False, False, False], index=df.index)
    exit_sig = pd.Series([False, False, True, True], index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=exit_sig), commission_bps=0.0)
    assert rep.metrics["n_trades"] == 1
    assert rep.metrics["n_wins"] == 1
    assert rep.metrics["win_rate_pct"] == 100.0
    assert rep.trades[0].return_pct == pytest.approx(10.0, abs=0.01)


def test_losing_trade_counts_as_loss() -> None:
    df = _bars([100, 100, 90, 90], opens=[100, 100, 90, 90])
    entries = pd.Series([True, False, False, False], index=df.index)
    exit_sig = pd.Series([False, False, True, True], index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=exit_sig))
    assert rep.metrics["n_trades"] == 1
    assert rep.metrics["n_losses"] == 1
    assert rep.metrics["win_rate_pct"] == 0.0
    assert rep.trades[0].win is False


def test_stop_loss_takes_priority_and_caps_loss() -> None:
    # Entry at 100; bar 2 low pierces the 10% stop. Exit must be the stop, a loss.
    df = _bars([100, 100, 100, 100],
               opens=[100, 100, 95, 100], highs=[100, 100, 96, 100], lows=[100, 100, 85, 100])
    entries = pd.Series([True, False, False, False], index=df.index)
    exit_sig = pd.Series(False, index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=exit_sig, stop_loss_pct=0.10), commission_bps=0.0)
    assert rep.trades[0].exit_reason == "stop_loss"
    assert rep.trades[0].return_pct == pytest.approx(-10.0, abs=0.01)


def test_take_profit_target_fills_at_target() -> None:
    df = _bars([100, 100, 100],
               opens=[100, 100, 100], highs=[100, 100, 105], lows=[100, 100, 100])
    entries = pd.Series([True, False, False], index=df.index)
    exit_sig = pd.Series(False, index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=exit_sig, take_profit_pct=0.02), commission_bps=0.0)
    assert rep.trades[0].exit_reason == "take_profit"
    assert rep.trades[0].return_pct == pytest.approx(2.0, abs=0.01)
    assert rep.trades[0].win is True


def test_commission_reduces_net_return() -> None:
    # Same +10% price move, but a round-trip commission must make the *net*
    # per-trade return strictly less than the gross 10% (and win-rate metrics
    # are computed from this net number).
    df = _bars([100, 100, 110, 110], opens=[100, 100, 110, 110])
    entries = pd.Series([True, False, False, False], index=df.index)
    exit_sig = pd.Series([False, False, True, True], index=df.index)
    gross = simulate(df, entries, ExitRules(exit_signal=exit_sig), commission_bps=0.0)
    net = simulate(df, entries, ExitRules(exit_signal=exit_sig), commission_bps=10.0)
    assert net.trades[0].return_pct < gross.trades[0].return_pct
    # 10 bps each leg ≈ (1-0.001)^2 * 1.10 - 1 ≈ 9.78%
    assert net.trades[0].return_pct == pytest.approx(9.78, abs=0.05)


def test_time_stop_closes_position() -> None:
    df = _bars([100] * 6, opens=[100] * 6)
    entries = pd.Series([True] + [False] * 5, index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=pd.Series(False, index=df.index), max_hold_bars=3))
    assert rep.trades[0].exit_reason == "time_stop"
    assert rep.trades[0].bars_held == 3


def test_only_one_position_at_a_time() -> None:
    # Two consecutive entry signals must not stack — second is ignored while in pos.
    df = _bars([100, 100, 100, 110], opens=[100, 100, 100, 110])
    entries = pd.Series([True, True, False, False], index=df.index)
    exit_sig = pd.Series([False, False, False, True], index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=exit_sig))
    assert rep.metrics["n_trades"] == 1


def test_metrics_keys_present() -> None:
    df = _bars([100, 100, 110, 110], opens=[100, 100, 110, 110])
    entries = pd.Series([True, False, False, False], index=df.index)
    rep = simulate(df, entries, ExitRules(exit_signal=pd.Series([False, False, True, True], index=df.index)))
    for k in ("win_rate_pct", "profit_factor", "expectancy_pct", "max_drawdown_pct",
              "sharpe", "total_return_pct", "avg_win_pct", "avg_loss_pct"):
        assert k in rep.metrics


def test_build_signals_rsi2_returns_bool_entries() -> None:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300)))
    df = _bars(list(close))
    entries, rules = bot._build_signals(df, bot.DEFAULT_STRATEGY)
    assert entries.dtype == bool
    assert len(entries) == len(df)
    assert isinstance(rules, ExitRules)


def test_default_strategy_is_high_win_config() -> None:
    s = bot.DEFAULT_STRATEGY
    assert s["kind"] == "rsi2_meanrev"
    assert s["take_profit_pct"] == 0.01
    assert s["trend_sma"] == 200


def _synthetic_meanrev_history() -> pd.DataFrame:
    """An uptrend with periodic sharp dips — the regime the bot is built for."""
    rng = np.random.default_rng(7)
    n = 400
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 1.5, n)
    dips = np.zeros(n)
    dips[50::40] = -12  # periodic sharp oversold dips
    close = trend + noise + dips
    return _bars(list(close))


def test_backtest_symbol_uses_history(monkeypatch) -> None:
    df = _synthetic_meanrev_history()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    rep = bot.backtest_symbol("FAKE")
    assert rep.symbol == "FAKE"
    assert rep.metrics["n_trades"] >= 1


def test_latest_signal_shape(monkeypatch) -> None:
    df = _synthetic_meanrev_history()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    sig = bot.latest_signal("FAKE", llm_gate=False)  # pure quant — no network
    assert sig["symbol"] == "FAKE"
    assert sig["action"] in ("BUY", "FLAT")
    assert "reason" in sig
    assert sig["size_multiplier"] == 1.0


def test_backtest_portfolio_pools_trades(monkeypatch) -> None:
    df = _synthetic_meanrev_history()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    res = bot.backtest_portfolio(["AAA", "BBB"])
    assert res["pooled"]["n_trades"] >= 1
    assert set(res["per_symbol"].keys()) == {"AAA", "BBB"}
    assert 0.0 <= res["pooled"]["win_rate_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Live execution — a fake broker keeps everything offline.
# ---------------------------------------------------------------------------

class FakeBroker:
    """In-memory stand-in for app.services.broker_alpaca."""

    def __init__(self, *, configured=True, market_open=True, buying_power=10_000.0,
                 positions=None):
        self._configured = configured
        self._market_open = market_open
        self._buying_power = buying_power
        self._positions = positions or []
        self.orders: list[dict] = []
        self.closed: list[str] = []

    def is_configured(self):
        return self._configured

    def is_paper(self):
        return True

    def is_market_open(self):
        return self._market_open

    def get_account(self):
        return {"buying_power": self._buying_power, "equity": self._buying_power,
                "cash": self._buying_power, "is_paper": True}

    def list_positions(self):
        return list(self._positions)

    def submit_market_order(self, symbol, *, notional=None, qty=None, side="buy"):
        o = {"id": f"o{len(self.orders)}", "symbol": symbol.upper(), "side": side,
             "notional": notional, "status": "accepted"}
        self.orders.append(o)
        return o

    def close_position(self, symbol):
        self.closed.append(symbol.upper())
        return {"id": f"c{len(self.closed)}", "symbol": symbol.upper(), "status": "accepted"}


def test_run_live_blocked_when_market_closed() -> None:
    fb = FakeBroker(market_open=False)
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert res["blocked"] == "market_closed"
    assert fb.orders == []


def test_run_live_blocked_when_broker_unconfigured() -> None:
    fb = FakeBroker(configured=False)
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert res["blocked"] == "broker_not_configured"


def test_run_live_opens_buy_on_signal(monkeypatch) -> None:
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "BUY",
                                                    "reason": "test", "price": 100.0})
    res = bot.run_live(universe=["AAA"], dsl=None,
                       execution={"max_position_usd": 500.0}, run_state=None, broker_mod=fb)
    assert res["blocked"] is None
    assert len(fb.orders) == 1
    assert fb.orders[0]["notional"] == 500.0
    assert res["run_state"]["orders_today"] == 1


def test_run_live_skips_already_held(monkeypatch) -> None:
    fb = FakeBroker(positions=[{"symbol": "AAA", "qty": 5, "avg_entry_price": 90.0,
                                "current_price": 95.0, "unrealized_pl": 25.0, "side": "long"}])
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "BUY",
                                                    "reason": "test", "price": 100.0})
    monkeypatch.setattr(bot, "evaluate_position_exit",
                        lambda *a, **k: {"exit": False, "reason": "hold"})
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert fb.orders == []
    assert fb.closed == []


def test_run_live_closes_on_exit_signal(monkeypatch) -> None:
    fb = FakeBroker(positions=[{"symbol": "AAA", "qty": 5, "avg_entry_price": 90.0,
                                "current_price": 99.0, "unrealized_pl": 45.0, "side": "long"}])
    monkeypatch.setattr(bot, "evaluate_position_exit",
                        lambda *a, **k: {"exit": True, "reason": "take_profit"})
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "FLAT"})
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert fb.closed == ["AAA"]
    assert res["run_state"]["orders_today"] == 1


def test_kill_switch_halts_after_max_daily_orders(monkeypatch) -> None:
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "BUY",
                                                    "reason": "test", "price": 100.0})
    res = bot.run_live(universe=["AAA", "BBB", "CCC"], dsl=None,
                       execution={"max_daily_orders": 2, "max_position_usd": 100.0},
                       run_state=None, broker_mod=fb)
    assert len(fb.orders) == 2  # third is skipped by the kill-switch
    skipped = [a for a in res["actions"] if a["action"] == "BUY_SKIPPED"]
    assert skipped and "max_daily_orders" in skipped[0]["reason"]


def test_kill_switch_halts_after_max_daily_loss(monkeypatch) -> None:
    fb = FakeBroker(positions=[{"symbol": "AAA", "qty": 5, "avg_entry_price": 100.0,
                                "current_price": 80.0, "unrealized_pl": -100.0, "side": "long"}])
    monkeypatch.setattr(bot, "evaluate_position_exit",
                        lambda *a, **k: {"exit": True, "reason": "stop_loss"})
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "BUY",
                                                    "reason": "test", "price": 100.0})
    res = bot.run_live(universe=["AAA", "BBB"], dsl=None,
                       execution={"max_daily_loss_usd": 50.0, "max_position_usd": 100.0},
                       run_state=None, broker_mod=fb)
    assert fb.closed == ["AAA"]
    assert fb.orders == []
    assert res["run_state"]["realized_loss_today"] == 100.0


def test_daily_counters_reset_on_new_day(monkeypatch) -> None:
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "FLAT"})
    stale = {"trading_day": "2000-01-01", "orders_today": 99,
             "realized_loss_today": 9999.0, "log": [{"x": 1}]}
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None,
                       run_state=stale, broker_mod=fb)
    assert res["run_state"]["trading_day"] == bot._today_str()
    assert res["run_state"]["orders_today"] == 0


def test_evaluate_position_exit_take_profit() -> None:
    d = bot.evaluate_position_exit("AAA", {"take_profit_pct": 0.01, "stop_loss_pct": 0.25},
                                   entry_price=100.0, current_price=101.5)
    assert d["exit"] is True
    assert d["reason"] == "take_profit"


def test_evaluate_position_exit_stop_loss() -> None:
    d = bot.evaluate_position_exit("AAA", {"take_profit_pct": 0.01, "stop_loss_pct": 0.10},
                                   entry_price=100.0, current_price=85.0)
    assert d["exit"] is True
    assert d["reason"] == "stop_loss"


# ---------------------------------------------------------------------------
# LLM risk-gate
# ---------------------------------------------------------------------------

from app.services import bot_intel  # noqa: E402


def _uptrend_with_final_dip() -> pd.DataFrame:
    """Rising series with a sharp two-bar dip at the end → fires the RSI(2)
    oversold-in-uptrend entry on the last bar, deterministically."""
    closes = list(np.linspace(100, 200, 250))
    closes[-2] = closes[-3] * 0.95
    closes[-1] = closes[-2] * 0.95
    return _bars(closes)


def test_signal_fires_on_final_dip(monkeypatch) -> None:
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    sig = bot.latest_signal("FAKE", llm_gate=False)
    assert sig["action"] == "BUY"  # sanity: the gate tests below rely on this


def test_llm_veto_downgrades_buy_to_flat(monkeypatch) -> None:
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    monkeypatch.setattr(bot.bot_intel, "llm_review",
                        lambda sym, sig, ctx=None: {"decision": "VETO", "conviction": 0.9,
                                                    "size_multiplier": 0.0, "rationale": "fraud headline",
                                                    "key_risks": ["sec probe"], "provider": "test"})
    sig = bot.latest_signal("FAKE", llm_gate=True)
    assert sig["action"] == "FLAT"
    assert sig["size_multiplier"] == 0.0
    assert sig["llm"]["decision"] == "VETO"


def test_llm_downsize_scales_size(monkeypatch) -> None:
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    monkeypatch.setattr(bot.bot_intel, "llm_review",
                        lambda *a, **k: {"decision": "DOWNSIZE", "conviction": 0.6,
                                         "size_multiplier": 0.5, "rationale": "elevated risk",
                                         "key_risks": [], "provider": "test"})
    sig = bot.latest_signal("FAKE", llm_gate=True)
    assert sig["action"] == "BUY"
    assert sig["size_multiplier"] == 0.5


def test_llm_gate_off_skips_review(monkeypatch) -> None:
    df = _uptrend_with_final_dip()
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1
        return {}
    monkeypatch.setattr(bot.bot_intel, "llm_review", boom)
    bot.latest_signal("FAKE", llm_gate=False)
    assert calls["n"] == 0


def test_run_live_scales_notional_by_size_multiplier(monkeypatch) -> None:
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "BUY",
                                                    "reason": "test", "price": 100.0,
                                                    "size_multiplier": 0.5,
                                                    "llm": {"decision": "DOWNSIZE"}})
    res = bot.run_live(universe=["AAA"], dsl=None,
                       execution={"max_position_usd": 1000.0}, run_state=None, broker_mod=fb)
    assert fb.orders[0]["notional"] == 500.0  # 1000 * 0.5
    assert res["actions"][0]["llm"]["decision"] == "DOWNSIZE"


def test_run_live_reports_veto(monkeypatch) -> None:
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal",
                        lambda sym, dsl=None, **k: {"symbol": sym.upper(), "action": "FLAT",
                                                    "reason": "LLM veto: bad news",
                                                    "size_multiplier": 0.0,
                                                    "llm": {"decision": "VETO", "rationale": "bad news"}})
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert fb.orders == []
    assert res["actions"][0]["action"] == "VETOED"


def test_llm_review_fail_open_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(bot_intel.llm, "is_available", lambda: False)
    v = bot_intel.llm_review("AAA", {"price": 100.0})
    assert v["decision"] == "APPROVE"
    assert v["size_multiplier"] == 1.0
    assert v["provider"] == "unavailable"


def test_llm_review_parses_and_clamps(monkeypatch) -> None:
    monkeypatch.setattr(bot_intel.llm, "is_available", lambda: True)
    monkeypatch.setattr(bot_intel, "gather_context", lambda sym: {"news": {}})

    class _Res:
        provider = "groq"; model = "llama"
        json = {"decision": "DOWNSIZE", "conviction": 2.0,  # out-of-range → clamped
                "size_multiplier": 9.0, "rationale": "x", "key_risks": ["a", "b"]}
    monkeypatch.setattr(bot_intel.llm, "generate", lambda **k: _Res())
    v = bot_intel.llm_review("AAA", {"price": 100.0})
    assert v["decision"] == "DOWNSIZE"
    assert v["conviction"] == 1.0          # clamped 2.0 → 1.0
    assert 0.0 < v["size_multiplier"] < 1.0  # DOWNSIZE forced to actually reduce


def test_llm_review_veto_zeroes_size(monkeypatch) -> None:
    monkeypatch.setattr(bot_intel.llm, "is_available", lambda: True)
    monkeypatch.setattr(bot_intel, "gather_context", lambda sym: {})

    class _Res:
        provider = "groq"; model = "llama"
        json = {"decision": "VETO", "conviction": 0.8, "size_multiplier": 0.9,
                "rationale": "fraud", "key_risks": []}
    monkeypatch.setattr(bot_intel.llm, "generate", lambda **k: _Res())
    v = bot_intel.llm_review("AAA", {"price": 100.0})
    assert v["decision"] == "VETO"
    assert v["size_multiplier"] == 0.0
