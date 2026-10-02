"""Swing agent, phase 2: regime rules and their no-look-ahead guarantees. Offline."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.swing_agent import experiments, regime_daily as rd, regime_eval as ev
from app.swing_agent import regime_intraday as ri
from app.swing_agent import store as st
from app.swing_agent.calendar import NY, TradingCalendar
from app.swing_agent.indicators import adx, same_time_baseline, trailing_rank
from app.swing_agent.universe import Interval

DAILY_P = {"sma_fast": 50, "sma_slow": 200, "slope_fast_days": 10, "adx_period": 14,
           "adx_trend": 20, "atr_period": 14, "atr_rank_window": 252, "atr_rank_high": 0.9,
           "vix_high_enter": 25.0, "vix_high_exit": 20.0, "breadth_bull_min": 0.5,
           "confirm_days": 2, "structure": "two_axis"}
INTRA_P = {"gap_atr": 0.5, "relvol_lookback": 20, "gap_go_relvol": 1.5, "trend_side_share": 0.75,
           "trend_dist_atr": 0.25, "reversal_ext_atr": 0.5, "vol_expand": 1.3, "vol_compress": 0.7}


def cal_for(days: pd.DatetimeIndex) -> TradingCalendar:
    return TradingCalendar.from_rows([{"date": str(d.date()), "open": "09:30", "close": "16:00"}
                                      for d in days])


def daily_from_close(close: np.ndarray, start="2016-01-04", spread=0.01) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(close))
    cal = cal_for(days)
    c = pd.Series(close, index=pd.DatetimeIndex(days, name="session"))
    df = pd.DataFrame({"open": c.shift(1).fillna(c), "high": c * (1 + spread), "low": c * (1 - spread),
                       "close": c, "volume": 1e6, "tr_factor": 1.0})
    df["t_close"] = pd.DatetimeIndex([cal.session(d.date()).close for d in days])
    return df


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def test_adx_high_in_trend_low_in_noise():
    n = 200
    rng = np.random.default_rng(0)
    trend = pd.Series(100 + np.arange(n) * 0.5)
    noise = pd.Series(100 + rng.normal(0, 0.5, n))
    a_t = adx(trend * 1.005, trend * 0.995, trend).iloc[-1]
    a_n = adx(noise + 0.3, noise - 0.3, noise).iloc[-1]
    assert a_t > 40 and a_n < 25


def test_trailing_rank_ignores_the_future():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=400))
    r = trailing_rank(x, 252)
    y = x.copy()
    y.iloc[300:] = 99.0
    pd.testing.assert_series_equal(r.iloc[:300], trailing_rank(y, 252).iloc[:300])


def test_same_time_baseline_excludes_today():
    sessions = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=30), 4))
    slots = pd.Series(np.tile(np.arange(4), 30))
    v = pd.Series(np.ones(120))
    base = same_time_baseline(v, sessions, slots, 20)
    v2 = v.copy()
    v2.iloc[-4:] = 1000.0                        # today's volume explodes
    base2 = same_time_baseline(v2, sessions, slots, 20)
    pd.testing.assert_series_equal(base, base2)  # today's baseline unchanged
    assert base.iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# daily rules
# ---------------------------------------------------------------------------

def test_latch_and_confirm():
    on = pd.Series([False, True, False, False, False, False])
    off = pd.Series([False, False, False, True, False, False])
    assert rd.latch(on, off).tolist() == [False, True, True, False, False, False]
    raw = pd.Series(["bull", "bull", "range", "bull", "range", "range", "high_vol", "bull"], dtype=object)
    got = rd.confirm(raw, 2).tolist()
    # one-day 'range' blips are ignored; high_vol is adopted at once; bull needs 2 days back
    assert got == ["bull", "bull", "bull", "bull", "bull", "range", "high_vol", "high_vol"]


def test_two_axis_keeps_a_bull_trend_in_high_vol():
    f = pd.DataFrame({"close": [110.0], "sma_fast": [105.0], "sma_slow": [100.0], "slope_fast": [0.1],
                      "adx": [30.0], "atr_rank": [0.5]}, index=pd.DatetimeIndex(["2020-06-12"]))
    vix = pd.Series([36.0], index=f.index)
    breadth = pd.Series([0.8], index=f.index)
    two = rd.raw_labels(f, DAILY_P, vix=vix, breadth_s=breadth)
    one = rd.raw_labels(f, DAILY_P | {"structure": "priority"}, vix=vix, breadth_s=breadth)
    assert two.iloc[0] == "bull" and one.iloc[0] == "high_vol"
    f2 = f.assign(close=[95.0], slope_fast=[-0.1])          # same VIX, no bull trend → stress
    assert rd.raw_labels(f2, DAILY_P, vix=vix, breadth_s=breadth).iloc[0] == "high_vol"


def test_market_regime_has_no_lookahead_and_is_usable_next_day():
    rng = np.random.default_rng(2)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 600))
    bench = daily_from_close(close)
    days = pd.DatetimeIndex(bench.index)
    cal = cal_for(days)
    vix = pd.DataFrame({"close": 15 + 5 * rng.random(600)}, index=days)
    br = pd.DataFrame({"breadth": 0.6 + 0.1 * rng.random(600)}, index=days)
    base = rd.market_regime(bench, vix, br, cal, DAILY_P)
    k = 450
    crash = bench.copy()
    crash.iloc[k:, :4] *= 0.5                          # the future collapses...
    vix2 = vix.copy()
    vix2.iloc[k:] = 80.0
    br2 = br.copy()
    br2.iloc[k:] = 0.05
    again = rd.market_regime(crash, vix2, br2, cal, DAILY_P)
    pd.testing.assert_series_equal(base["regime"].iloc[:k], again["regime"].iloc[:k])  # ...the past doesn't move
    assert again["regime"].iloc[k] == "high_vol"
    d = days[k]
    at_close = pd.Timestamp(f"{d.date()} 16:00", tz=NY).tz_convert("UTC")
    assert st.known_at(again, at_close).index.max() == days[k - 1]   # VIX not settled yet
    next_open = pd.Timestamp(f"{days[k + 1].date()} 09:45", tz=NY).tz_convert("UTC")
    assert st.known_at(again, next_open).index.max() == d


def test_breadth_counts_members_only_on_their_dates():
    days = pd.DatetimeIndex(pd.bdate_range("2016-01-04", periods=80), name="session")
    up = daily_from_close(np.linspace(100, 200, 80))
    dn = daily_from_close(np.linspace(200, 100, 80))
    b = rd.breadth([Interval("UP", date(2016, 1, 4), None)], {"UP": up}, days, ma=50)
    assert b["breadth"].dropna().eq(1.0).all()
    both = rd.breadth([Interval("UP", date(2016, 1, 4), None),
                       Interval("DN", date(2016, 1, 4), date(2016, 3, 31))],
                      {"UP": up, "DN@2016-03-31": dn}, days, ma=50)
    after = both.loc[both.index > pd.Timestamp("2016-03-31"), "breadth"].dropna()
    before = both.loc[(both.index <= pd.Timestamp("2016-03-31")) & both["breadth"].notna(), "breadth"]
    assert (before == 0.5).all() and (after == 1.0).all()   # DN stops counting after it leaves


# ---------------------------------------------------------------------------
# intraday rules
# ---------------------------------------------------------------------------

def _intraday(n_days=30, last_day_path=None, gap=0.0):
    """30 quiet sessions, then a custom last session; daily ATR ≈ 2."""
    days = pd.bdate_range("2024-01-02", periods=n_days)
    cal = cal_for(days)
    rows = []
    for i, d in enumerate(days):
        path = last_day_path if (i == n_days - 1 and last_day_path is not None) else \
            100 + 0.05 * np.sin(np.arange(26))
        start = pd.Timestamp(f"{d.date()} 09:30", tz=NY).tz_convert("UTC")
        for j, c in enumerate(path):
            ts = start + pd.Timedelta(minutes=15 * j)
            o = path[j - 1] if j else path[0]
            rows.append((ts, o, max(o, c) + 0.05, min(o, c) - 0.05, c,
                         (5000.0 if i == n_days - 1 else 1000.0), c))
    m = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "vwap"]).set_index("ts")
    m["trades"] = 1.0
    rth, _ = st.split_sessions(m, cal)
    rth["tr_factor"] = 1.0
    daily = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 - gap,
                          "tr_factor": 1.0}, index=pd.DatetimeIndex(days, name="session"))
    daily.iloc[-1] = [100, 101, 99, 100, 1.0]
    return rth, daily


def test_gap_and_go_and_gap_fade_labels():
    up = 102 + np.linspace(0, 1.5, 26)                     # gaps up ~1 ATR and keeps going
    rth, daily = _intraday(last_day_path=up)
    lab = ri.micro_regime(rth, daily, INTRA_P)
    last = lab[lab["session"] == lab["session"].max()]
    assert last["micro"].iloc[1] == "gap_go_up"
    fade = np.r_[102.0, 102.3, 101.5, np.linspace(101, 99.5, 23)]   # gives the gap back
    rth, daily = _intraday(last_day_path=fade)
    lab = ri.micro_regime(rth, daily, INTRA_P)
    last = lab[lab["session"] == lab["session"].max()]
    assert last["micro"].iloc[-1] == "reversal_down"


def test_intraday_features_never_peek():
    rng = np.random.default_rng(3)
    path = 100 + np.cumsum(rng.normal(0, 0.2, 26))
    rth, daily = _intraday(last_day_path=path)
    base = ri.micro_regime(rth, daily, INTRA_P, hourly_only=False)
    cut = rth.index[-26 + 9]                               # 11:45 bar of the last session
    poisoned = rth.copy()
    later = poisoned.index > cut
    poisoned.loc[later, ["open", "high", "low", "close", "vwap"]] *= 1.3
    poisoned.loc[later, "volume"] *= 50
    daily2 = daily.copy()
    daily2.iloc[-1] = [130, 140, 120, 135, 1.0]            # today's daily bar (unknown intraday)
    again = ri.micro_regime(poisoned, daily2, INTRA_P, hourly_only=False)
    cols = ["gap", "dist", "side_share", "relvol", "range_ratio", "atr_prev", "micro", "vol_state"]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], again.loc[:cut, cols])


def test_hour_close_flags_and_forward_drops_last_bar():
    rth, daily = _intraday()
    lab = ri.micro_regime(rth, daily, INTRA_P)
    one = lab[lab["session"] == lab["session"].max()]
    assert one["slot"].tolist() == [3, 7, 11, 15, 19, 23, 25]
    last_close = rth.groupby("session")["close"].last()
    fwd = ev.forward_intraday(lab, last_close)
    assert 25 not in set(fwd["slot"])


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def test_drawdown_detection_lag():
    idx = pd.bdate_range("2020-01-01", periods=12)
    px = pd.Series([100, 105, 110, 100, 90, 80, 85, 95, 105, 110, 112, 115.0], index=idx)
    eps = ev.drawdowns(px, 0.10)
    assert len(eps) == 1 and eps[0]["peak"] == idx[2] and eps[0]["trough"] == idx[5]
    reg = pd.Series(["bull", "bull", "bull", "bull", "high_vol", "high_vol", "high_vol", "range",
                     "bull", "bull", "bull", "bull"], index=idx)
    d = ev.detection(eps, reg, px)[0]
    assert d["lag_sessions"] == 2 and d["drawdown_at_detection_pct"] == pytest.approx(-18.2, abs=0.1)
    assert d["back_to_bull"] == str(idx[8].date()) and d["rebound_missed_pct"] == pytest.approx(31.2, abs=0.1)


def test_experiment_registry_counts_distinct_variants(tmp_path):
    experiments.log(2, "x", "a", {"p": 1}, "train", {"m": 1}, root=tmp_path)
    experiments.log(2, "x", "a", {"p": 1}, "validation", {"m": 2}, root=tmp_path)
    experiments.log(2, "x", "b", {"p": 2}, "train", {"m": 3}, root=tmp_path)
    assert experiments.count("x", root=tmp_path) == 2 and len(experiments.read(tmp_path)) == 3
