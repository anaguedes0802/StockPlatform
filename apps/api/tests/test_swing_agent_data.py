"""Swing agent, phase 1: calendar, bar shaping, no-look-ahead views, universe.

All offline: hand-built calendars and bars pin each convention exactly.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.swing_agent import config, events, survivorship
from app.swing_agent import store as st
from app.swing_agent.calendar import NY, TradingCalendar
from app.swing_agent.universe import Membership, build_universe, by_ticker, rank_liquidity

# 2025-11-26 normal, 11-27 Thanksgiving (closed), 11-28 half day; DST ended 11-02.
ROWS = [
    {"date": "2025-10-31", "open": "09:30", "close": "16:00"},
    {"date": "2025-11-03", "open": "09:30", "close": "16:00"},
    {"date": "2025-11-25", "open": "09:30", "close": "16:00"},
    {"date": "2025-11-26", "open": "09:30", "close": "16:00"},
    {"date": "2025-11-28", "open": "09:30", "close": "13:00"},
    {"date": "2025-12-01", "open": "09:30", "close": "16:00"},
]


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar.from_rows(ROWS)


def et(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=NY).tz_convert("UTC")


def m15_bars(day: str, start="04:00", end="20:00", px=100.0) -> pd.DataFrame:
    idx = pd.date_range(et(day, start), et(day, end), freq="15min", inclusive="left")
    n = len(idx)
    close = px + np.arange(n) * 0.01
    return pd.DataFrame({"open": close - 0.005, "high": close + 0.02, "low": close - 0.02,
                         "close": close, "volume": 1000.0, "trades": 10.0, "vwap": close},
                        index=pd.DatetimeIndex(idx, name="ts"))


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def test_calendar_dst_early_close_and_holidays(cal):
    assert cal.session(date(2025, 10, 31)).open == pd.Timestamp("2025-10-31 13:30", tz="UTC")  # EDT
    assert cal.session(date(2025, 11, 3)).open == pd.Timestamp("2025-11-03 14:30", tz="UTC")   # EST
    half = cal.session(date(2025, 11, 28))
    assert half.early_close and half.minutes == 210
    assert not cal.is_session(date(2025, 11, 27))
    assert cal.next(date(2025, 11, 26)).day == date(2025, 11, 28)
    assert cal.previous(date(2025, 11, 28)).day == date(2025, 11, 26)
    assert [s.day for s in cal.month_starts(date(2025, 10, 1), date(2025, 12, 31))] == \
        [date(2025, 10, 31), date(2025, 11, 3), date(2025, 12, 1)]


def test_session_lookup_uses_new_york_date(cal):
    # 19:30 ET on 11-26 is 00:30 UTC on 11-27, still the 11-26 trading date
    late = et("2025-11-26", "19:30")
    assert late.date() == date(2025, 11, 27)
    assert cal.last_closed(late).day == date(2025, 11, 26)
    assert cal.session_at(et("2025-11-26", "15:59")).day == date(2025, 11, 26)
    assert cal.session_at(et("2025-11-26", "16:00")) is None


# ---------------------------------------------------------------------------
# bar shaping
# ---------------------------------------------------------------------------

def test_split_sessions_keeps_only_regular_hours(cal):
    raw = pd.concat([m15_bars("2025-11-26"), m15_bars("2025-11-28", end="17:00"),
                     m15_bars("2025-11-27")])  # stray prints on the holiday
    rth, pm = st.split_sessions(raw, cal)
    per = rth.groupby("session").size()
    assert per[pd.Timestamp("2025-11-26")] == 26
    assert per[pd.Timestamp("2025-11-28")] == 14          # 09:30 → 13:00
    assert pd.Timestamp("2025-11-27") not in per.index
    assert (rth["t_close"] - rth.index == pd.Timedelta(minutes=15)).all()
    assert rth.index.min() == et("2025-11-26", "09:30")
    assert rth[rth.session == "2025-11-28"]["t_close"].max() == et("2025-11-28", "13:00")
    assert pm.loc["2025-11-26", "pm_volume"] == 22 * 1000.0  # 04:00 → 09:30


def test_hourly_bars_align_to_open_and_never_peek(cal):
    rth, _ = st.split_sessions(pd.concat([m15_bars("2025-11-26"), m15_bars("2025-11-28")]), cal)
    h = st.resample_hourly(rth, cal)
    d1 = h[h.session == "2025-11-26"]
    assert list(d1.index.tz_convert(NY).strftime("%H:%M")) == \
        ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert d1["t_close"].iloc[-1] == et("2025-11-26", "16:00")
    assert d1["n_bars"].tolist() == [4, 4, 4, 4, 4, 4, 2]
    d2 = h[h.session == "2025-11-28"]
    assert d2["t_close"].iloc[-1] == et("2025-11-28", "13:00") and d2["n_bars"].iloc[-1] == 2
    # every 15m bar inside an hour closes no later than the hour does
    for ts, row in h.iterrows():
        members = rth[(rth.index >= ts) & (rth.index < row["t_close"])]
        assert (members["t_close"] <= row["t_close"]).all()
        assert row["close"] == members["close"].iloc[-1]
        assert row["high"] == members["high"].max() and row["volume"] == members["volume"].sum()


def test_hour_with_missing_15m_bar_still_closes_on_the_clock(cal):
    rth, _ = st.split_sessions(m15_bars("2025-11-26"), cal)
    rth = rth.drop(et("2025-11-26", "10:15"))  # no trades in the last quarter of the first hour
    h = st.resample_hourly(rth, cal)
    first = h.iloc[0]
    assert first["n_bars"] == 3 and first["t_close"] == et("2025-11-26", "10:30")


def _daily(cal: TradingCalendar, closes: list[float]) -> pd.DataFrame:
    days = [s.day for s in cal.sessions][: len(closes)]
    idx = pd.DatetimeIndex(pd.to_datetime(days), name="session")
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                       "volume": 1e6, "trades": 1.0, "vwap": closes}, index=idx)
    df["t_close"] = pd.DatetimeIndex([cal.session(d).close for d in days])
    return df


def test_daily_regime_only_sees_closed_days(cal):
    d = _daily(cal, [10, 11, 12, 13, 14, 15])
    # during 11-28 (half day) the newest usable daily bar is 11-26
    assert st.known_at(d, et("2025-11-28", "12:45")).index.max() == pd.Timestamp("2025-11-26")
    # at the 13:00 close the 11-28 bar becomes usable
    assert st.known_at(d, et("2025-11-28", "13:00")).index.max() == pd.Timestamp("2025-11-28")
    # the morning of 12-01 still sees 11-28, not 12-01
    assert st.known_at(d, et("2025-12-01", "09:45")).index.max() == pd.Timestamp("2025-11-28")


def test_daily_frame_derives_split_factor_and_dividend(cal):
    idx = pd.DatetimeIndex([pd.Timestamp(f"{r['date']} 00:00", tz=NY).tz_convert("UTC")
                            for r in ROWS[:4]], name="ts")

    def f(c: list[float], v: float = 100.0) -> pd.DataFrame:
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v,
                             "trades": 1.0, "vwap": c}, index=idx)

    # 2:1 split effective on day 3; $1 dividend ex on day 4 (raw close 50 → 49)
    raw = f([100.0, 100.0, 50.0, 49.0])
    split = f([50.0, 50.0, 50.0, 49.0], 200.0)
    total = f([49.0, 49.0, 49.0, 49.0], 200.0)   # dividend back-adjusted
    d = st.daily_frame(raw, split, total, cal)
    assert d["split_factor"].tolist() == [2.0, 2.0, 1.0, 1.0]
    assert d["div_ret"].iloc[:3].tolist() == [0.0, 0.0, 0.0]
    assert d["div_ret"].iloc[3] == pytest.approx(1 / 49.0, rel=1e-6)
    assert d["dollar_volume"].iloc[0] == 100.0 * 100.0
    assert d["t_close"].iloc[0] == et("2025-10-31", "16:00")


# ---------------------------------------------------------------------------
# splits / final-test lock
# ---------------------------------------------------------------------------

def test_research_reads_exclude_final_test_and_lock_is_one_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "root", lambda cfg=None: tmp_path)
    idx = pd.DatetimeIndex(pd.to_datetime(["2023-12-29", "2024-01-02", "2026-09-30"]), name="session")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    assert st.clip_segment(df, "research").index.max() == pd.Timestamp("2023-12-29")
    assert st.clip_segment(df, "train").empty
    with pytest.raises(PermissionError):
        st.clip_segment(df, "final_test")
    lock = st.FinalTestLock(tmp_path)
    lock.open("phase 5 final evaluation")
    assert len(st.clip_segment(df, "final_test")) == 3
    with pytest.raises(PermissionError):
        lock.open("second look")


# ---------------------------------------------------------------------------
# point-in-time universe
# ---------------------------------------------------------------------------

CSV = """date,tickers
2015-12-01,"AAA,FB,OLD"
2016-03-01,"AAA,FB,NEW"
2016-06-01,"AAA,META,NEW"
"""


def test_membership_intervals_and_keys():
    m = Membership.from_csv_text(CSV)
    assert m.members_on(date(2016, 2, 29)) == {"AAA", "FB", "OLD"}
    assert m.members_on(date(2016, 3, 1)) == {"AAA", "FB", "NEW"}
    ivs = m.intervals(date(2016, 1, 1))
    keys = {iv.ticker: iv.key for iv in ivs}
    assert keys == {"AAA": "AAA", "FB": "FB@2016-05-31", "META": "META",
                    "NEW": "NEW", "OLD": "OLD@2016-02-29"}
    idx = by_ticker(ivs)
    assert Membership.key_on(idx, "FB", date(2016, 5, 31)) == "FB@2016-05-31"
    assert Membership.key_on(idx, "FB", date(2016, 6, 1)) is None


def _liq_cal() -> TradingCalendar:
    days = pd.bdate_range("2016-01-04", "2016-07-29")
    return TradingCalendar.from_rows([{"date": str(d.date()), "open": "09:30", "close": "16:00"}
                                      for d in days])


def _liq_daily(cal: TradingCalendar, dv: float, start="2016-01-04", end="2016-07-29") -> pd.DataFrame:
    days = pd.DatetimeIndex([pd.Timestamp(s.day) for s in cal.between(date.fromisoformat(start),
                                                                      date.fromisoformat(end))])
    return pd.DataFrame({"close": 20.0, "close_raw": 20.0, "volume": dv / 20, "dollar_volume": dv,
                         "tr_close": 20.0}, index=days)


def test_liquidity_rank_ignores_the_rebalance_day_and_after():
    cal = _liq_cal()
    daily = {"A": _liq_daily(cal, 2e8), "B": _liq_daily(cal, 1e8)}
    on = date(2016, 6, 1)
    base = rank_liquidity(daily, ["A", "B"], cal, on, lookback=63, min_price=5,
                          min_dollar_volume=5e7, min_coverage=0.9)
    assert base["key"].tolist() == ["A", "B"]
    poisoned = {k: v.copy() for k, v in daily.items()}
    poisoned["B"].loc[poisoned["B"].index >= pd.Timestamp(on), "dollar_volume"] = 1e12
    again = rank_liquidity(poisoned, ["A", "B"], cal, on, lookback=63, min_price=5,
                           min_dollar_volume=5e7, min_coverage=0.9)
    pd.testing.assert_frame_equal(base, again)


def test_universe_uses_members_of_that_month_only():
    cal = _liq_cal()
    m = Membership.from_csv_text(CSV)
    ivs = m.intervals(date(2016, 1, 1))
    daily = {"AAA": _liq_daily(cal, 1e8), "FB@2016-05-31": _liq_daily(cal, 9e8),
             "META": _liq_daily(cal, 9e8), "NEW": _liq_daily(cal, 3e8),
             "OLD@2016-02-29": _liq_daily(cal, 5e8)}
    u = build_universe(m, ivs, daily, cal, date(2016, 4, 1), date(2016, 7, 29), top_n=2,
                       lookback=63, min_price=5, min_dollar_volume=5e7, min_coverage=0.9)
    by = u.groupby("rebalance")["key"].apply(list)
    assert by[pd.Timestamp("2016-04-01")] == ["FB@2016-05-31", "NEW"]   # OLD already gone
    assert by[pd.Timestamp("2016-06-01")] == ["META", "NEW"]           # FB renamed → META key
    survivors_only = build_universe(m, ivs, daily, cal, date(2016, 4, 1), date(2016, 4, 30),
                                    top_n=2, lookback=63, min_price=5, min_dollar_volume=5e7,
                                    min_coverage=0.9, restrict_to={"AAA", "META", "NEW"})
    assert survivors_only["key"].tolist() == ["NEW", "AAA"]


def test_survivorship_classification():
    cal = _liq_cal()
    m = Membership.from_csv_text(CSV)
    ivs = m.intervals(date(2016, 1, 1))
    meta = _liq_daily(cal, 9e8)
    crash = _liq_daily(cal, 5e8, end="2016-02-26")
    crash.loc[crash.index[-3:], "tr_close"] = [9.0, 6.0, 4.0]
    daily = {"AAA": _liq_daily(cal, 1e8), "FB@2016-05-31": meta.copy(), "META": meta,
             "NEW": _liq_daily(cal, 3e8), "OLD@2016-02-29": crash}
    cls = survivorship.classify(ivs, daily, cal, date(2016, 7, 29))
    assert cls == {"AAA": "current", "FB@2016-05-31": "renamed_member", "META": "current",
                   "NEW": "current", "OLD@2016-02-29": "delisted_distressed"}


def test_survivorship_prices_cash_out_or_zero():
    cal = _liq_cal()
    d = _liq_daily(cal, 1e8, end="2016-03-15")
    px = survivorship._prices({"X": d}, ["X"], [date(2016, 3, 1), date(2016, 4, 1)], set())
    assert px["X"].tolist() == [20.0, 20.0]
    px0 = survivorship._prices({"X": d}, ["X"], [date(2016, 3, 1), date(2016, 4, 1)], {"X"})
    assert px0["X"].tolist() == [20.0, 0.0]


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def test_earnings_reaction_session(cal):
    earn = pd.DataFrame({"symbol": ["A", "B", "C", "D"],
                         "date": pd.to_datetime(["2025-11-26", "2025-11-26", "2025-11-26",
                                                 "2025-11-27"]),
                         "timing": ["bmo", "amc", "unknown", "bmo"]})
    r = events.reaction_sessions(earn, cal)
    got = r.groupby("symbol")["session"].apply(lambda s: [x.date().isoformat() for x in s]).to_dict()
    assert got == {"A": ["2025-11-26"], "B": ["2025-11-28"],
                   "C": ["2025-11-26", "2025-11-28"], "D": ["2025-11-28"]}


def test_share_classes_count_once():
    cal = _liq_cal()
    rng = np.random.default_rng(0)
    base = _liq_daily(cal, 9e8)
    base["close"] = 20 * np.cumprod(1 + rng.normal(0, 0.01, len(base)))
    clone = base.copy()
    clone["close"] = base["close"] * 1.003 * (1 + rng.normal(0, 0.0005, len(base)))
    other = base.copy()
    other["close"] = 20 * np.cumprod(1 + rng.normal(0, 0.01, len(base)))
    daily = {"GOOGL": base, "GOOG": clone, "XOM": other}
    lo, hi = pd.Timestamp("2016-03-01"), pd.Timestamp("2016-05-31")
    from app.swing_agent.universe import dedupe_same_company
    assert dedupe_same_company(["GOOGL", "GOOG", "XOM"], daily, lo, hi, 2, 0.95) == ["GOOGL", "XOM"]


def test_same_cik_counts_once_even_when_returns_diverge():
    cal = _liq_cal()
    rng = np.random.default_rng(1)
    a = _liq_daily(cal, 9e8)
    a["close"] = 20 * np.cumprod(1 + rng.normal(0, 0.01, len(a)))
    b = a.copy()
    b["close"] = 20 * np.cumprod(1 + rng.normal(0, 0.01, len(a)))  # uncorrelated path
    lo, hi = pd.Timestamp("2016-03-01"), pd.Timestamp("2016-05-31")
    from app.swing_agent.universe import dedupe_same_company
    daily = {"GOOGL": a, "GOOG": b}
    assert dedupe_same_company(["GOOGL", "GOOG"], daily, lo, hi, 2, 0.95) == ["GOOGL", "GOOG"]
    assert dedupe_same_company(["GOOGL", "GOOG"], daily, lo, hi, 2, 0.95,
                               {"GOOGL": "1652044", "GOOG": "1652044"}) == ["GOOGL"]


def test_merge_earnings_is_conservative():
    d = pd.Timestamp
    edgar = pd.DataFrame({"symbol": ["A", "B", "C"], "date": [d("2024-01-25"), d("2024-02-01"),
                                                              d("2024-03-05")],
                          "timing": ["unknown", "bmo", "amc"]})
    nasdaq = pd.DataFrame({"symbol": ["A", "B", "C"], "date": [d("2024-01-25"), d("2024-02-01"),
                                                               d("2024-03-04")],
                           "timing": ["amc", "amc", "amc"]})
    m = events.merge_earnings(edgar, nasdaq).set_index(["symbol", "date"])
    assert m.loc[("A", d("2024-01-25")), "timing"] == "amc"       # known fills unknown
    assert m.loc[("B", d("2024-02-01")), "timing"] == "unknown"   # disagreement → both sessions
    assert len(m.loc["C"]) == 2                                   # 8-K a day late: keep both dates


# ---------------------------------------------------------------------------
# corporate actions
# ---------------------------------------------------------------------------

def _src(raw: list[float], adj_factor: list[float]) -> pd.DataFrame:
    """Alpaca-like daily source: adjusted = raw / factor."""
    idx = pd.DatetimeIndex(pd.bdate_range("2016-05-02", periods=len(raw)), name="session")
    raw_a = np.asarray(raw, float)
    adj = raw_a / np.asarray(adj_factor, float)
    return pd.DataFrame({"open": adj, "high": adj * 1.01, "low": adj * 0.99, "close": adj,
                         "volume": 1e6 * np.asarray(adj_factor), "trades": 1.0, "vwap": adj,
                         "close_raw": raw_a}, index=idx)


def test_misrecorded_spinoff_is_undone_and_valued():
    from app.swing_agent import corporate as co
    # WestRock 2016-05-16: raw 42.56 → 39.63 (Ingevity spun off), but Alpaca
    # applied a 1→0.1666 "reverse split", inflating earlier adjusted prices 6x.
    raw = [42.0] * 10 + [39.6] * 10
    f = [0.1666] * 10 + [1.0] * 10
    src = _src(raw, f)
    spl = co.validate_splits(co.applied_splits(src), src["close_raw"])
    assert len(spl) == 1 and spl.loc[0, "state"] == "not_split"
    assert spl.loc[0, "ratio"] == pytest.approx(0.1666, rel=1e-3)
    fixed, spl2, dist = co.build(src, [("WRK", pd.Timestamp.min, pd.Timestamp.max)], {}, lambda *_: None)
    assert fixed["close"].iloc[0] == pytest.approx(42.0)          # no fake 6x level
    assert (fixed["split_factor"] == 1.0).all()
    assert dist["kind"].tolist() == ["misrecorded_split_gap_estimate"]
    assert dist["amount_raw"].iloc[0] == pytest.approx(42.0 - 39.6)
    # total return is continuous across the ex-date, execution price is not
    r = fixed["tr_close"].pct_change().iloc[10]
    assert abs(r) < 1e-9 and fixed["close"].pct_change().iloc[10] < -0.05


def test_real_split_is_kept():
    from app.swing_agent import corporate as co
    raw = [400.0] * 10 + [100.5] * 10          # 4:1 split, raw price really drops 4x
    f = [4.0] * 10 + [1.0] * 10
    src = _src(raw, f)
    fixed, spl, dist = co.build(src, [("AAPL", pd.Timestamp.min, pd.Timestamp.max)], {}, lambda *_: None)
    assert spl.loc[0, "state"] == "split" and dist.empty
    assert fixed["close"].iloc[0] == pytest.approx(100.0)
    assert fixed["split_factor"].iloc[0] == pytest.approx(4.0) and fixed["split_factor"].iloc[-1] == 1.0
    assert (fixed["close"] * fixed["split_factor"] - fixed["close_raw"]).abs().max() < 1e-6


def test_dividend_is_a_cash_distribution_and_tr_factor_has_no_lookahead():
    from app.swing_agent import corporate as co
    raw = [50.0] * 10 + [49.0] * 10
    src = _src(raw, [1.0] * 20)
    acts = {"cash_dividends": [{"symbol": "X", "ex_date": str(src.index[10].date()), "rate": 1.0}]}
    fixed, _, dist = co.build(src, [("X", pd.Timestamp.min, pd.Timestamp.max)], acts, lambda *_: None)
    assert fixed["div_ret"].iloc[10] == pytest.approx(1 / 50)
    assert fixed["tr_close"].iloc[10] == pytest.approx(fixed["tr_close"].iloc[9])
    assert fixed["tr_factor"].iloc[0] == 1.0 and (fixed["tr_factor"].iloc[:10] == 1.0).all()


def test_symbol_periods_follow_renames_without_reuse():
    from app.swing_agent import corporate as co
    acts = {"name_changes": [
        {"old_symbol": "FB", "new_symbol": "META", "process_date": "2022-06-09"},
        {"old_symbol": "OLDCO", "new_symbol": "FB", "process_date": "2030-01-01"},  # later reuse of FB
    ]}
    p = co.symbol_periods("META", acts)
    syms = {(s, lo.date() if lo.year > 1900 else None, hi.date() if hi.year < 2200 else None)
            for s, lo, hi in p}
    assert ("META", date(2022, 6, 9), None) in syms
    assert ("FB", None, date(2022, 6, 8)) in syms
    assert not any(s == "OLDCO" for s, _, _ in syms)
    assert co._in_periods("FB", pd.Timestamp("2020-01-02"), p)
    assert not co._in_periods("FB", pd.Timestamp("2031-01-02"), p)


def test_symbol_periods_anchor_ignores_later_reuse():
    from app.swing_agent import corporate as co
    acts = {"name_changes": [
        {"old_symbol": "CBS", "new_symbol": "VIAC", "process_date": "2020-02-13"},
        {"old_symbol": "VIAC", "new_symbol": "PARA", "process_date": "2022-02-17"},
        {"old_symbol": "BNZI", "new_symbol": "PARA", "process_date": "2026-08-07"},  # reuse
    ]}
    p = co.symbol_periods("PARA", acts, anchor=pd.Timestamp("2025-08-07"))
    names = [s for s, _, _ in p]
    assert names == ["CBS", "VIAC", "PARA"] and "BNZI" not in names
    assert p[2][2] == pd.Timestamp("2026-08-06")       # ends where the reuse starts
    today = co.symbol_periods("PARA", acts, anchor=pd.Timestamp("2026-09-01"))
    assert [s for s, _, _ in today] == ["BNZI", "PARA"]
    viac = co.symbol_periods("VIAC", acts, anchor=pd.Timestamp("2022-02-16"))
    assert [s for s, _, _ in viac] == ["CBS", "VIAC", "PARA"]


def test_market_data_applies_fixes_to_daily_and_15m(tmp_path, monkeypatch):
    import json as _json

    from app.swing_agent import corporate as co
    from app.swing_agent.data import MarketData
    monkeypatch.setattr(config, "root", lambda cfg=None: tmp_path)
    days = pd.bdate_range("2016-05-02", periods=20)
    (tmp_path / "calendar.json").write_text(_json.dumps(
        [{"date": str(d.date()), "open": "09:30", "close": "16:00"} for d in days]))
    cal = TradingCalendar.load(tmp_path / "calendar.json")
    src = _src([42.0] * 10 + [39.6] * 10, [0.1666] * 10 + [1.0] * 10)  # the WRK case
    src.index = pd.DatetimeIndex(days, name="session")
    src["t_close"] = pd.DatetimeIndex([cal.session(d.date()).close for d in days])
    bs = st.BarStore(tmp_path)
    bs.write("daily", "WRK", src)
    m15 = pd.concat([m15_bars(str(d.date()), "09:30", "10:00", px=float(src["close"].iloc[i]))
                     for i, d in enumerate(days)])
    rth, _ = st.split_sessions(m15, cal)
    bs.write("m15", "WRK", rth)
    _, spl, dist = co.build(src, [("WRK", pd.Timestamp.min, pd.Timestamp.max)], {}, lambda *_: None)
    (tmp_path / "adjustments.json").write_text(_json.dumps({"WRK": {
        "splits": spl.assign(ex_date=spl["ex_date"].astype(str)).to_dict("records"),
        "distributions": dist.assign(ex_date=dist["ex_date"].astype(str)).to_dict("records")}}))
    md = MarketData(tmp_path, segment=None)
    d = md.daily("WRK")
    assert d["close"].iloc[0] == pytest.approx(42.0)
    i = md.m15("WRK")
    assert i["close"].iloc[0] == pytest.approx(42.0, rel=1e-3)      # 6x fake level undone
    assert i["tr_factor"].iloc[0] == 1.0 and i["tr_factor"].iloc[-1] > 1.0
    assert md.h1("WRK")["t_close"].iloc[0] == et(str(days[0].date()), "10:30")  # closes on the clock


def test_split_ratio_snaps_to_exact_fractions():
    from app.swing_agent.corporate import snap_ratio
    assert snap_ratio(3.99996) == 4.0 and snap_ratio(0.1000004) == 0.1
    assert snap_ratio(1.50002) == 1.5 and snap_ratio(0.66668) == pytest.approx(2 / 3)
    assert snap_ratio(1.0816) == 1.0816 or abs(snap_ratio(1.0816) - 1.0816) / 1.0816 < 0.005
