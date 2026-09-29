"""Swing trading: portfolio engine, setups, live execution path, auto-run.

All offline. The engine takes hand-built bars + signal frames so each rule
(fills, gaps, targets, trailing, sizing, earnings, costs) is pinned exactly.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.backtest import swing_engine as eng
from app.services import bot_autorun
from app.services import swing
from app.services import trading_bot as bot


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bars(opens, highs, lows, closes, start="2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": 1e6}, index=idx, dtype=float)


def _flat(n=10, px=100.0, start="2024-01-01") -> pd.DataFrame:
    return _bars([px] * n, [px + 0.5] * n, [px - 0.5] * n, [px] * n, start=start)


def _sig(df: pd.DataFrame, *, long_at=(), short_at=(), dist=2.0, exit_long_at=(),
         atr=1.0, score=0.0) -> pd.DataFrame:
    n = len(df)
    s = pd.DataFrame({
        "long_entry": [i in long_at for i in range(n)],
        "short_entry": [i in short_at for i in range(n)],
        "stop_dist_long": dist, "stop_dist_short": dist,
        "exit_long": [i in exit_long_at for i in range(n)],
        "exit_short": False, "atr": atr, "score_long": score, "score_short": score,
    }, index=df.index)
    return s


def _cfg(**kw) -> eng.EngineConfig:
    base = dict(commission_bps=0.0, slippage_bps=0.0, max_position_pct=100.0,
                max_heat_pct=100.0, max_gross_leverage=10.0, whole_units=False,
                earnings_blackout_days=0, min_notional=0.0)
    base.update(kw)
    return eng.EngineConfig(**base)


def _run(df, sig, rules=None, cfg=None, **kw):
    return eng.simulate({"AAA": df}, {"AAA": sig}, rules or eng.SetupRules(), cfg or _cfg(),
                        mc_sims=0, **kw)


# ---------------------------------------------------------------------------
# engine: fills, stops, targets
# ---------------------------------------------------------------------------

def test_entry_fills_next_open_and_stop_costs_one_r() -> None:
    df = _flat(8)
    df.iloc[3, df.columns.get_loc("low")] = 97.0   # entry bar 2 open=100, stop=98
    res = _run(df, _sig(df, long_at={1}, dist=2.0))
    t = res.trades[0]
    assert t.entry_date == df.index[2].date().isoformat()
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(98.0)
    assert t.r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open_and_loses_more_than_one_r() -> None:
    df = _flat(8)
    df.iloc[3] = [95.0, 96.0, 94.0, 95.0, 1e6]      # opens below the 98 stop
    res = _run(df, _sig(df, long_at={1}, dist=2.0))
    t = res.trades[0]
    assert t.exit_price == pytest.approx(95.0)
    assert t.r_multiple == pytest.approx(-2.5)


def test_target_fills_at_limit() -> None:
    df = _flat(8)
    df.iloc[4, df.columns.get_loc("high")] = 105.0  # 2R target = 104
    res = _run(df, _sig(df, long_at={1}, dist=2.0), rules=eng.SetupRules(target_r=2.0))
    t = res.trades[0]
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(104.0)
    assert t.r_multiple == pytest.approx(2.0)


def test_stop_assumed_first_when_bar_touches_both() -> None:
    df = _flat(8)
    df.iloc[3, df.columns.get_loc("high")] = 110.0
    df.iloc[3, df.columns.get_loc("low")] = 90.0
    res = _run(df, _sig(df, long_at={1}, dist=2.0), rules=eng.SetupRules(target_r=2.0))
    assert res.trades[0].exit_reason == "stop"


def test_signal_exit_fills_next_open() -> None:
    df = _flat(8)
    df.iloc[5, df.columns.get_loc("open")] = 103.0
    res = _run(df, _sig(df, long_at={1}, exit_long_at={4}))
    t = res.trades[0]
    assert t.exit_reason == "signal"
    assert t.exit_date == df.index[5].date().isoformat()
    assert t.exit_price == pytest.approx(103.0)


def test_time_stop() -> None:
    df = _flat(12)
    res = _run(df, _sig(df, long_at={1}), rules=eng.SetupRules(max_hold_bars=3))
    t = res.trades[0]
    assert t.exit_reason == "time"
    assert t.bars_held == 3


def test_chandelier_trailing_stop_ratchets_and_exits() -> None:
    closes = [100, 100, 100, 104, 108, 112, 112, 106, 106, 106]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    lows[7] = 105.0                                    # 112.5 - 3*1 = 109.5 trail → hit
    df = _bars(closes, highs, lows, closes)
    df.iloc[7, df.columns.get_loc("open")] = 108.0
    res = _run(df, _sig(df, long_at={1}, dist=2.0, atr=1.0),
               rules=eng.SetupRules(trail_atr_mult=3.0))
    t = res.trades[0]
    assert t.exit_reason == "trail_stop"
    assert t.exit_price == pytest.approx(108.0)      # gapped below 109.5 → open
    assert t.r_multiple > 3.0


def test_short_trade_profits_when_price_falls() -> None:
    closes = [100, 100, 100, 96, 92, 92, 92, 92]
    df = _bars(closes, [c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    res = _run(df, _sig(df, short_at={1}, dist=2.0), rules=eng.SetupRules(target_r=2.0),
               cfg=_cfg(allow_short=True))
    t = res.trades[0]
    assert t.side == -1
    assert t.exit_reason == "target"
    assert t.pnl > 0
    assert t.r_multiple == pytest.approx(2.0)


def test_shorts_ignored_unless_allowed() -> None:
    df = _flat(8)
    res = _run(df, _sig(df, short_at={1}), cfg=_cfg(allow_short=False))
    assert res.trades == []


# ---------------------------------------------------------------------------
# engine: sizing, capacity, costs, earnings
# ---------------------------------------------------------------------------

def test_risk_sizing_whole_shares() -> None:
    df = _flat(6)
    res = _run(df, _sig(df, long_at={1}, dist=3.0),
               cfg=_cfg(initial_equity=10_000.0, risk_per_trade_pct=1.0, whole_units=True))
    assert res.trades[0].units == 33          # floor(100 / 3)
    assert res.trades[0].initial_risk == pytest.approx(99.0)


def test_position_cap_binds_before_risk_budget() -> None:
    df = _flat(6)
    res = _run(df, _sig(df, long_at={1}, dist=0.5),
               cfg=_cfg(initial_equity=10_000.0, risk_per_trade_pct=1.0,
                        max_position_pct=20.0, whole_units=True))
    assert res.trades[0].units == 20          # 20% of 10k at $100, not 200 shares
    assert res.diagnostics["skipped_at_fill"]["max_positions"] == 0


def test_max_positions_and_ranking() -> None:
    data = {s: _flat(6) for s in ("AAA", "BBB", "CCC")}
    sigs = {s: _sig(data[s], long_at={1}, score=sc) for s, sc in (("AAA", 1), ("BBB", 3), ("CCC", 2))}
    res = eng.simulate(data, sigs, eng.SetupRules(), _cfg(max_positions=2), mc_sims=0)
    assert sorted(t.symbol for t in res.trades) == ["BBB", "CCC"]
    assert res.diagnostics["skipped_at_fill"]["max_positions"] == 1


def test_costs_and_financing_reduce_pnl() -> None:
    df = _flat(10)
    df.iloc[6, df.columns.get_loc("open")] = 102.0
    free = _run(df, _sig(df, long_at={1}, exit_long_at={5}))
    costly = _run(df, _sig(df, long_at={1}, exit_long_at={5}),
                  cfg=_cfg(commission_bps=5.0, slippage_bps=5.0, financing_bps_per_year=300.0))
    assert costly.trades[0].pnl < free.trades[0].pnl
    assert costly.trades[0].entry_price > 100.0   # bought above the open
    assert costly.trades[0].exit_price < 102.0    # sold below the open


def test_earnings_exit_at_close_before_report() -> None:
    df = _flat(10)
    report = df.index[5]
    res = _run(df, _sig(df, long_at={1}), earnings={"AAA": [report]},
               cfg=_cfg(earnings_blackout_days=0))
    t = res.trades[0]
    assert t.exit_reason == "earnings"
    assert t.exit_date == df.index[4].date().isoformat()


def test_earnings_blackout_skips_entry() -> None:
    df = _flat(10)
    res = _run(df, _sig(df, long_at={1}), earnings={"AAA": [df.index[4]]},
               cfg=_cfg(earnings_blackout_days=7))
    assert res.trades == []
    assert res.diagnostics["skipped_earnings"] == 1


def test_equity_curve_matches_trade_pnl() -> None:
    df = _flat(10)
    df.iloc[6, df.columns.get_loc("open")] = 110.0
    res = _run(df, _sig(df, long_at={1}, exit_long_at={5}),
               cfg=_cfg(initial_equity=10_000.0, commission_bps=2.0, slippage_bps=3.0))
    final = res.equity_curve[-1]["equity"]
    assert final == pytest.approx(10_000.0 + sum(t.pnl for t in res.trades), abs=0.02)


# ---------------------------------------------------------------------------
# verdict & reporting
# ---------------------------------------------------------------------------

def test_verdict_no_edge_on_negative_expectancy() -> None:
    v = eng.verdict({"n_trades": 300, "expectancy_r": -0.05, "sharpe_t_stat": -0.5, "sharpe": -0.1},
                    None, [], {"prob_total_r_le_0_pct": 80.0})
    assert v["label"] == "no_edge"


def test_verdict_flags_edge_below_benchmark() -> None:
    halves = [{"expectancy_r": 0.2, "cagr_pct": 4.0}, {"expectancy_r": 0.1, "cagr_pct": 2.0}]
    v = eng.verdict({"n_trades": 500, "expectancy_r": 0.15, "sharpe_t_stat": 2.5, "sharpe": 0.5,
                     "cagr_pct": 3.0, "max_drawdown_pct": -20.0},
                    {"sharpe": 0.7, "cagr_pct": 10.0, "max_drawdown_pct": -50.0},
                    halves, {"prob_total_r_le_0_pct": 0.0})
    assert v["label"] == "edge_below_benchmark"


# ---------------------------------------------------------------------------
# setups
# ---------------------------------------------------------------------------

def _random_walk(n=600, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return _bars(o, h, l, c, start="2020-01-01")


@pytest.mark.parametrize("setup", ["breakout", "pullback", "rsi2"])
def test_signals_have_no_lookahead(setup) -> None:
    df = _random_walk()
    full = swing.build_signals(df, setup)
    cut = swing.build_signals(df.iloc[:450], setup)
    cols = ["long_entry", "short_entry", "exit_long", "exit_short"]
    pd.testing.assert_frame_equal(full.iloc[:450][cols], cut[cols])
    np.testing.assert_allclose(full.iloc[:450]["stop_dist_long"], cut["stop_dist_long"], equal_nan=True)


@pytest.mark.parametrize("setup", ["breakout", "pullback", "rsi2"])
def test_setups_run_through_engine(setup) -> None:
    data = {"AAA": _random_walk(seed=2), "BBB": _random_walk(seed=3)}
    sigs = {k: swing.build_signals(v, setup) for k, v in data.items()}
    res = eng.simulate(data, sigs, swing.setup_rules(setup), swing.config_for("equity"), mc_sims=50)
    out = res.as_dict()
    assert set(SIG for SIG in eng.SIGNAL_COLUMNS) <= set(sigs["AAA"].columns)
    assert out["metrics"]["n_trades"] >= 1
    assert out["verdict"]["label"] in {"no_edge", "inconclusive", "edge_below_benchmark", "historical_edge"}
    assert all(t["r_multiple"] > -5 for t in out["trades"])  # no absurd sizing artefacts


def test_breakout_entry_requires_new_high_above_trend() -> None:
    df = _random_walk(seed=4)
    s = swing.build_signals(df, "breakout")
    fired = s.index[s["long_entry"]]
    for ts in fired[:20]:
        i = df.index.get_loc(ts)
        assert df["close"].iloc[i] > df["high"].iloc[i - 55:i].max()


def test_completed_bars_drops_forming_equity_bar() -> None:
    df = _flat(5, start="2026-09-23")
    now_open = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)   # 11:00 ET
    df.index = pd.date_range("2026-09-23", periods=5, freq="B", tz="UTC")
    assert df.index[-1].date().isoformat() == "2026-09-29"
    assert len(swing.completed_bars(df, "SPY", now_open)) == 4
    after_close = datetime(2026, 9, 29, 21, 0, tzinfo=timezone.utc)  # 17:00 ET
    assert len(swing.completed_bars(df, "SPY", after_close)) == 5


# ---------------------------------------------------------------------------
# live execution path (FakeBroker)
# ---------------------------------------------------------------------------

class SwingBroker:
    def __init__(self, *, equity=100_000.0, positions=None):
        self.equity = equity
        self._positions = positions or []
        self.orders, self.brackets, self.closed, self.cancelled, self.replaced = [], [], [], [], []

    def is_configured(self): return True
    def is_paper(self): return True
    def is_market_open(self): return True

    def get_account(self):
        return {"equity": self.equity, "buying_power": self.equity, "cash": self.equity}

    def list_positions(self): return list(self._positions)

    def submit_market_order(self, symbol, *, notional=None, qty=None, side="buy"):
        o = {"id": f"m{len(self.orders)}", "symbol": symbol, "qty": qty, "notional": notional}
        self.orders.append(o)
        return o

    def submit_bracket_order(self, symbol, *, qty=None, notional=None, side="buy",
                             stop_loss_price=None, take_profit_price=None, time_in_force="gtc"):
        o = {"id": f"b{len(self.brackets)}", "symbol": symbol, "qty": qty, "notional": notional,
             "time_in_force": time_in_force,
             "legs": [{"id": f"stop{len(self.brackets)}", "type": "stop", "stop_price": stop_loss_price}]}
        self.brackets.append({**o, "stop": stop_loss_price, "tp": take_profit_price})
        return o

    def cancel_open_orders(self, symbol):
        self.cancelled.append(symbol)
        return 1

    def replace_stop(self, order_id, stop_price):
        self.replaced.append((order_id, stop_price))
        return {"id": order_id, "stop_price": stop_price}

    def close_position(self, symbol):
        self.closed.append(symbol)
        return {"id": "c", "symbol": symbol}


SWING_DSL = {"kind": "swing_breakout"}


def test_swing_entry_sizes_by_risk_and_attaches_gtc_stop(monkeypatch) -> None:
    fb = SwingBroker(equity=100_000.0)
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 50.0, "stop_dist": 2.5, "stop": 47.5,
        "target": None, "reason": "test"})
    res = bot.run_live(universe=["AAA"], dsl=SWING_DSL,
                       execution={"max_position_usd": 20_000.0, "risk_per_trade_pct": 1.0,
                                  "max_gross_exposure_usd": 100_000.0},
                       run_state=None, broker_mod=fb)
    assert res["blocked"] is None
    b = fb.brackets[0]
    assert b["qty"] == 400                      # risk 1% = $1000 / $2.50 = 400 sh ($20k cap → 400)
    assert b["time_in_force"] == "gtc"
    assert b["stop"] == pytest.approx(47.5)
    meta = res["run_state"]["positions"]["AAA"]
    assert meta["stop_order_id"] == "stop0"
    assert meta["protective"] == "broker"


def test_swing_entry_capped_by_position_limit(monkeypatch) -> None:
    fb = SwingBroker(equity=100_000.0)
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 50.0, "stop_dist": 0.5, "stop": 49.5})
    bot.run_live(universe=["AAA"], dsl=SWING_DSL,
                 execution={"max_position_usd": 5_000.0, "max_gross_exposure_usd": 100_000.0},
                 run_state=None, broker_mod=fb)
    assert fb.brackets[0]["qty"] == 100         # $5k cap / $50, not 2000 shares


def test_broker_side_close_is_reconciled(monkeypatch) -> None:
    fb = SwingBroker(positions=[])
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {"symbol": sym, "action": "FLAT"})
    state = {"trading_day": bot._today_str(), "orders_today": 0, "realized_loss_today": 0.0,
             "equity_hwm": 0.0, "log": [], "positions": {"AAA": {"stop": 47.5, "entry_date": "2026-09-01"}}}
    res = bot.run_live(universe=["AAA"], dsl=SWING_DSL, execution=None, run_state=state, broker_mod=fb)
    assert "AAA" not in res["run_state"]["positions"]
    assert any(a["action"] == "CLOSED_BY_BROKER" for a in res["actions"])
    assert fb.cancelled == ["AAA"]


def test_exit_cancels_legs_before_closing(monkeypatch) -> None:
    fb = SwingBroker(positions=[{"symbol": "AAA", "qty": 10, "avg_entry_price": 50.0,
                                 "current_price": 55.0, "unrealized_pl": 50.0, "side": "long"}])
    monkeypatch.setattr(bot, "evaluate_position_exit", lambda *a, **k: {"exit": True, "reason": "signal"})
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {"symbol": sym, "action": "FLAT"})
    state = {"positions": {"AAA": {"stop": 47.5}}}
    res = bot.run_live(universe=["AAA"], dsl=SWING_DSL, execution=None, run_state=state, broker_mod=fb)
    assert fb.cancelled == ["AAA"] and fb.closed == ["AAA"]
    assert "AAA" not in res["run_state"]["positions"]


def test_trailing_stop_moves_broker_leg_up_only(monkeypatch) -> None:
    fb = SwingBroker(positions=[{"symbol": "AAA", "qty": 10, "avg_entry_price": 50.0,
                                 "current_price": 60.0, "unrealized_pl": 100.0, "side": "long"}])
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {"symbol": sym, "action": "FLAT"})
    state = {"positions": {"AAA": {"stop": 47.5, "stop_order_id": "leg1"}}}
    monkeypatch.setattr(bot, "evaluate_position_exit",
                        lambda *a, **k: {"exit": False, "reason": "hold", "new_stop": 55.0})
    res = bot.run_live(universe=["AAA"], dsl=SWING_DSL, execution=None, run_state=state, broker_mod=fb)
    assert fb.replaced == [("leg1", 55.0)]
    assert res["run_state"]["positions"]["AAA"]["stop"] == 55.0
    # a lower proposal must not loosen the stop
    monkeypatch.setattr(bot, "evaluate_position_exit",
                        lambda *a, **k: {"exit": False, "reason": "hold", "new_stop": 52.0})
    res2 = bot.run_live(universe=["AAA"], dsl=SWING_DSL, execution=None,
                        run_state=res["run_state"], broker_mod=fb)
    assert fb.replaced == [("leg1", 55.0)]
    assert res2["run_state"]["positions"]["AAA"]["stop"] == 55.0


def test_rsi2_bracket_uses_whole_shares_not_notional(monkeypatch) -> None:
    fb = SwingBroker()
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "BUY", "price": 100.0, "reason": "test"})
    bot.run_live(universe=["AAA"], dsl=None, execution={"max_position_usd": 1_050.0},
                 run_state=None, broker_mod=fb)
    b = fb.brackets[0]
    assert b["qty"] == 10 and b["notional"] is None
    assert b["time_in_force"] == "gtc"
    assert b["stop"] == pytest.approx(75.0)     # −25% default stop


def test_rsi2_time_stop_enforced_with_entry_record(monkeypatch) -> None:
    df = _flat(60, start="2024-01-01")
    monkeypatch.setattr(bot.md, "get_history", lambda *a, **k: df)
    d = bot.evaluate_position_exit("AAA", {"max_hold_bars": 40, "take_profit_pct": None,
                                           "exit_sma": 500},
                                   entry_price=100.0, current_price=100.0,
                                   meta={"entry_date": df.index[5].date().isoformat()})
    assert d == {"exit": True, "reason": "time_stop"}


def test_swing_bot_rejects_fx_universe() -> None:
    from fastapi import HTTPException
    from app.routers import bot as bot_router
    with pytest.raises(HTTPException):
        bot_router._check_tradable(["SPY", "EURUSD=X"])


# ---------------------------------------------------------------------------
# auto-run scheduling
# ---------------------------------------------------------------------------

def test_autorun_due_window(monkeypatch) -> None:
    monkeypatch.setenv("BOT_AUTORUN_ET", "09:45")
    tue_0940 = datetime(2026, 9, 29, 13, 40, tzinfo=timezone.utc)   # 09:40 ET
    tue_0950 = datetime(2026, 9, 29, 13, 50, tzinfo=timezone.utc)
    sat = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)
    assert not bot_autorun.is_due({}, tue_0940)
    assert bot_autorun.is_due({}, tue_0950)
    assert not bot_autorun.is_due({"last_autorun_day": "2026-09-29"}, tue_0950)
    assert not bot_autorun.is_due({}, sat)


def test_autorun_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BOT_AUTORUN", raising=False)
    assert not bot_autorun.enabled()


# ---------------------------------------------------------------------------
# earnings fallback (Yahoo calendar down → SEC-history estimate)
# ---------------------------------------------------------------------------

def test_next_report_estimated_from_last_year(monkeypatch) -> None:
    hist = [pd.Timestamp(d, tz="UTC") for d in
            ("2025-07-31", "2025-10-30", "2026-01-29", "2026-04-30", "2026-07-30")]
    monkeypatch.setattr(swing.sd, "earnings_dates", lambda s: hist)
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    est = swing._estimate_next_report("AAA", now)
    assert est.date().isoformat() == "2026-10-29"      # 2025-10-30 + 364d, same weekday


def test_estimated_earnings_blocks_entries_conservatively(monkeypatch) -> None:
    monkeypatch.setattr(swing, "_next_earnings",
                        lambda s: {"known": True, "estimated": True, "date": "2026-10-10", "days": 12.0})
    df = _random_walk(seed=5)
    monkeypatch.setattr(swing, "_recent_bars", lambda s: df)
    sig = swing.build_signals(df, "breakout")
    monkeypatch.setattr(swing, "build_signals",
                        lambda d, st, p=None: sig.assign(long_entry=True))
    out = swing.latest_signal("AAA", {"kind": "swing_breakout", "earnings_blackout_days": 7})
    assert out["action"] == "FLAT" and "estimated" in out["reason"]   # 12 − 7 ≤ 7
