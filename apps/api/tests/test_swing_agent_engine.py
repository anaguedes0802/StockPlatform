"""Swing agent, phase 3: the portfolio engine's execution rules, pinned on hand-built bars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.swing_agent import costs as cx
from app.swing_agent import engine as en

C = {"commission_per_share": 0.0, "sec_fee_per_million": 0.0, "finra_taf_per_share": 0.0,
     "finra_taf_max": 8.3, "half_spread_bps_min": 1.0, "half_spread_cents": 0.5,
     "slippage_market_bps": 2.0, "slippage_stop_bps": 5.0, "slippage_moc_bps": 1.0}
ZERO = {k: 0.0 for k in C} | {"finra_taf_max": 0.0}
EX = {"risk_per_trade": 0.01, "max_positions": 5, "max_position_pct": 0.20, "max_gross": 1.0,
      "max_order_pct_volume": 1.0, "earnings_block_sessions": 5, "exit_before_earnings": True}
N_SESS, N_SLOT = 12, 26


def book(key="X", bars=None, signals=(), stop=1.0, target=np.nan, trail_after=np.nan, trail=np.nan,
         max_sessions=10, sf=1.0, div=None, react=(), exit_moc=(), is_etf=False, last=10**6):
    """`bars`: {(session, slot): (o, h, l, c)}; every other bar is flat at 100."""
    rows = []
    for s in range(N_SESS):
        for k in range(N_SLOT):
            o, h, l, c = (bars or {}).get((s, k), (100.0, 100.0, 100.0, 100.0))
            rows.append((s, k, 585 + 15 * k, o, h, l, c))   # bar close, minutes ET
    a = np.array(rows, dtype=float)
    n = len(a)
    ent = np.zeros(n, bool)
    for s, k in signals:
        ent[s * N_SLOT + k] = True
    em = np.zeros(n, bool)
    for s, k in exit_moc:
        em[s * N_SLOT + k] = True
    sig = {"S": {"entry": ent, "setup_entry": ent, "stop_dist": np.full(n, stop),
                 "target_dist": np.full(n, target), "trail_after": np.full(n, trail_after),
                 "trail_dist": np.full(n, trail), "max_sessions": np.full(n, max_sessions, int),
                 "score": np.ones(n), "exit_moc": em}}
    closes = np.array([a[(a[:, 0] == s), 6][-1] for s in range(N_SESS)])
    return en.Book(key=key, is_etf=is_etf, sess=a[:, 0].astype(int), slot=a[:, 1].astype(int),
                   mod=a[:, 2].astype(int), o=a[:, 3], h=a[:, 4], l=a[:, 5], c=a[:, 6],
                   sf=np.full(n, sf), med_vol=np.full(n, 1e9), micro=np.array(["range"] * n, object),
                   stock_regime=np.array(["bull"] * n, object), sig=sig, daily_close=closes,
                   daily_sf=np.full(N_SESS, sf), div_amt=np.zeros(N_SESS) if div is None else div,
                   last_session=last, react=list(react))


def ctx(keys=("X",), fomc=(), late=None):
    days = list(pd.bdate_range("2024-01-08", periods=N_SESS))
    return en.Context(sessions=days, n_slots=[N_SLOT] * N_SESS,
                      universe={"2024-01": set(keys)}, member={}, refs=set(), market_regime={},
                      fomc=set(fomc), late_start=late or {})


def run(books, c=ZERO, cap=10_000.0, ex=EX, context=None):
    b = {x.key: x for x in books}
    return en.run(b, context or ctx(tuple(b)), {"S": "entry"}, ex, c, cap, 0, N_SESS - 1)


def test_signal_fills_next_bar_open_with_costs():
    b = book(bars={(1, 3): (101.0, 101.0, 101.0, 101.0)}, signals=[(1, 2)], stop=2.0)
    r = run([b], c=C)
    t = r["trades"].iloc[0]
    half = max(0.005, 101 * 1e-4)
    assert t["entry"] == pytest.approx(101 + half + 101 * 2e-4)
    assert t["entry_date"] == pd.Timestamp("2024-01-09")
    # 1% of 10k risk / 2.0 stop = 50 shares, capped at 20% of equity at the signal close = 20
    assert t["shares"] == 20


def test_gap_through_stop_fills_at_the_open():
    bars = {(1, 3): (100.0, 100.0, 100.0, 100.0), (2, 0): (95.0, 96.0, 94.0, 95.5)}
    r = run([book(bars=bars, signals=[(1, 2)], stop=2.0)])
    t = r["trades"].iloc[0]
    assert t["reason"] == "stop_gap" and t["exit"] == pytest.approx(95.0)
    assert t["r"] == pytest.approx(-2.5)                 # a gap can lose more than 1R


def test_stop_beats_target_in_the_same_bar_and_target_needs_a_trade_through():
    bars = {(1, 5): (100.0, 103.5, 97.5, 101.0)}
    r = run([book(bars=bars, signals=[(1, 2)], stop=2.0, target=3.0)])
    assert r["trades"].iloc[0]["reason"] == "stop"
    bars = {(1, 5): (100.0, 103.0, 99.5, 101.0), (1, 6): (101.0, 103.01, 100.5, 102.0)}
    t = run([book(bars=bars, signals=[(1, 2)], stop=2.0, target=3.0)])["trades"].iloc[0]
    assert t["reason"] == "target" and t["exit"] == pytest.approx(103.0) and t["exit_mod"] == 585 + 15 * 6


def test_entry_bar_checks_stop_only():
    bars = {(1, 3): (100.0, 104.0, 97.0, 99.0)}           # fill at 100, same bar reaches both
    t = run([book(bars=bars, signals=[(1, 2)], stop=2.0, target=3.0)])["trades"].iloc[0]
    assert t["reason"] == "stop" and t["exit_date"] == pd.Timestamp("2024-01-09")


def test_trailing_stop_ratchets_and_never_falls():
    bars = {(1, 5): (100, 104, 100, 104.0), (1, 6): (104, 104, 102.5, 102.5),
            (1, 7): (102.4, 102.4, 101.9, 102.0)}
    t = run([book(bars=bars, signals=[(1, 2)], stop=5.0, trail_after=3.0, trail=2.0)])["trades"].iloc[0]
    # activated at 104 close → stop 102; the 102 close does not lower it; 101.9 low hits it
    assert t["reason"] == "trail" and t["exit"] == pytest.approx(102.0)


def test_time_exit_in_the_closing_auction():
    t = run([book(signals=[(1, 2)], stop=2.0, max_sessions=3)], c=C)["trades"].iloc[0]
    assert t["reason"] == "time" and t["exit_date"] == pd.Timestamp("2024-01-12") and t["exit_mod"] == 960
    assert t["exit"] == pytest.approx(100 * (1 - 1e-4))   # auction: 1 bp, no spread


def test_earnings_block_and_exit_before_report():
    blocked = run([book(signals=[(1, 2)], react=[5])])
    assert blocked["trades"].empty                         # report within 5 sessions → no entry
    t = run([book(signals=[(1, 2)], react=[8], max_sessions=20)])["trades"].iloc[0]
    assert t["reason"] == "earnings" and t["exit_date"] == pd.Timestamp("2024-01-17")  # session 7


def test_dividend_credited_to_overnight_holders():
    div = np.zeros(N_SESS)
    div[3] = 0.5
    t = run([book(signals=[(1, 2)], stop=2.0, max_sessions=4, div=div)])["trades"].iloc[0]
    assert t["divs"] == pytest.approx(0.5 * t["shares"]) and t["pnl"] == pytest.approx(t["divs"])


def test_whole_raw_shares_after_a_split():
    t = run([book(signals=[(1, 2)], stop=2.0, sf=4.0)])["trades"].iloc[0]
    raw_px = 100 * 4
    assert t["shares"] / 4 == int(t["shares"] / 4)          # whole raw shares
    assert t["shares"] / 4 == int(0.20 * 10_000 // raw_px)  # 20% cap in raw terms = 5 shares


def test_max_positions_and_cash():
    books = [book(key=f"K{i}", signals=[(1, 2)], stop=2.0) for i in range(7)]
    r = run(books)
    assert len(r["trades"]) == 5
    assert (r["equity"]["positions"].max()) == 5
    assert (r["trades"]["shares"] * 100 <= 0.2 * 10_000 + 1e-6).all()


def test_universe_fomc_and_late_start_filters():
    b = book(signals=[(1, 2)])
    assert run([b], context=ctx(keys=("OTHER",)))["trades"].empty
    assert run([b], context=ctx(fomc={1}))["trades"].empty
    assert run([b], context=ctx(late={1: 10 * 60 + 30}))["trades"].empty   # 10:15 signal before 10:30
    assert len(run([b], context=ctx(late={1: 10 * 60}))["trades"]) == 1


def test_costs_model():
    px, fee = cx.fill(100.0, +1, "market", 10, C)
    assert px == pytest.approx(100 + 0.01 + 0.02)
    px, fee = cx.fill(20.0, -1, "stop", 10, C | {"sec_fee_per_million": 27.8, "finra_taf_per_share": 0.000166})
    assert px == pytest.approx(20 - 0.005 - 20 * 5e-4)
    assert fee == pytest.approx(px * 10 * 27.8e-6 + 10 * 0.000166)
    assert cx.fill(50.0, -1, "limit", 10, C)[0] == 50.0


def test_strategy_signals_never_peek():
    from app.swing_agent import regime_intraday as ri
    from app.swing_agent import store as st
    from app.swing_agent import strategies as sg
    from app.swing_agent.calendar import NY, TradingCalendar

    rng = np.random.default_rng(7)
    days = pd.bdate_range("2023-01-02", periods=300)
    cal = TradingCalendar.from_rows([{"date": str(d.date()), "open": "09:30", "close": "16:00"} for d in days])
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(days)))
    daily = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.003, len(days))), "high": close * 1.01,
                          "low": close * 0.99, "close": close, "tr_factor": 1.0},
                         index=pd.DatetimeIndex(days, name="session"))
    rows = []
    for d, c0 in zip(days[-40:], close[-41:-1]):
        start = pd.Timestamp(f"{d.date()} 09:30", tz=NY).tz_convert("UTC")
        px = c0 * np.cumprod(1 + rng.normal(0, 0.003, 26))
        for j, c in enumerate(px):
            o = px[j - 1] if j else c0 * (1 + rng.normal(0, 0.01))
            rows.append((start + pd.Timedelta(minutes=15 * j), o, max(o, c) * 1.001, min(o, c) * 0.999, c,
                         1000.0 * (1 + rng.random()), c))
    m = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "vwap"]).set_index("ts")
    m["trades"] = 1.0
    rth, _ = st.split_sessions(m, cal)
    rth["tr_factor"] = 1.0
    sreg = pd.DataFrame({"regime": "bull"}, index=daily.index)
    pin = {"gap_atr": 0.5, "relvol_lookback": 20, "gap_go_relvol": 1.5, "trend_side_share": 0.75,
           "trend_dist_atr": 0.25, "reversal_ext_atr": 0.5, "vol_expand": 1.3, "vol_compress": 0.7}
    import tomllib
    from app.swing_agent import config
    cfg = tomllib.loads(config.CONFIG_PATH.read_text())

    def all_signals(m15, d):
        x = sg.frame(ri.features(m15, d, pin), d, sreg)
        return {n: sg.signals(n, x, cfg["strategy"][n], cfg["execution"]) for n in sg.NAMES}

    base = all_signals(rth, daily)
    cut = rth.index[-26 + 12]                       # 12:30 bar of the last session
    poisoned = rth.copy()
    later = poisoned.index > cut
    poisoned.loc[later, ["open", "high", "low", "close", "vwap"]] *= 0.7
    poisoned.loc[later, "volume"] *= 20
    d2 = daily.copy()
    d2.iloc[-1, :4] = [1.0, 1.0, 1.0, 1.0]          # today's daily bar is not known intraday
    again = all_signals(poisoned, d2)
    for n in sg.NAMES:
        a = base[n].loc[:cut].drop(columns=["exit_moc"])
        b = again[n].loc[:cut].drop(columns=["exit_moc"])
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
