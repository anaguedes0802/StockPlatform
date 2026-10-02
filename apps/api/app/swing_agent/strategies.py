"""The four individual strategies (long only), as per-15m-bar signal frames.

Each strategy has a daily **setup** (known before the open: it uses daily
bars up to yesterday) and an intraday **trigger** (a 15m bar of today that
closes inside the signal window). The engine fills a signal at the next
bar's open. The *daily-only twin* of a strategy enters at the first bar of
the window whenever the setup holds, with no trigger, so phase 3 can measure
whether the 15m timing adds anything.

| Strategy | Setup (yesterday) | Trigger (today, 15m) | Exits |
|---|---|---|---|
| pullback | stock regime bull, RSI(3) ≤ 30 | closes back above the session VWAP after trading below it | stop 1.5 ATR, target 3 ATR, trail 1.5 ATR after +1.5 ATR, 10 sessions |
| vwap_reversion | above SMA200, RSI(2) ≤ 10 | closes ≥ 0.5 ATR below the session VWAP | close above its 5-day average (auction), stop 2.5 ATR, 5 sessions |
| compression_breakout | Bollinger width in bottom 20% of 6 months, above SMA50 | closes above the 20-day high on ≥ 1.2× volume | stop 1.5 ATR, target 3 ATR, trail 2 ATR after +1.5 ATR, 10 sessions |
| gap_go | above a rising SMA50; today gaps ≥ 0.5 ATR | above the open and VWAP on ≥ 1.5× volume, by 11:30 | stop under the session low (0.5–1.5 ATR), target 2 ATR, trail 1 ATR after +1 ATR, 5 sessions |

Distances come out in execution prices (split-adjusted), converted from the
signal prices (× tr_factor) the features use.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.indicators import rsi
from app.swing_agent.calendar import NY
from app.swing_agent.indicators import sma, trailing_rank
from app.swing_agent.regime_daily import signal_prices

NAMES = ("pullback", "vwap_reversion", "compression_breakout", "gap_go")
INTRADAY_NAMES = ("gap_fade_down", "liquidity_sweep", "last_half_hour", "orb30")
SIGNAL_COLS = ["entry", "setup_entry", "stop_dist", "target_dist", "trail_after", "trail_dist",
               "max_sessions", "score", "exit_moc"]


def minute_of_day(ts: pd.Series) -> pd.Series:
    t = pd.DatetimeIndex(ts).tz_convert(NY)
    return pd.Series(t.hour * 60 + t.minute, index=ts.index)


def hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily indicators on signal prices, SHIFTED so the row for session D
    holds values as of D−1's close."""
    s = signal_prices(daily)
    c = s["close"]
    f = pd.DataFrame(index=daily.index)
    f["close"] = c
    f["rsi2"] = rsi(c, 2)
    f["rsi3"] = rsi(c, 3)
    f["sma50"] = sma(c, 50)
    f["sma200"] = sma(c, 200)
    f["sma50_slope"] = f["sma50"] - f["sma50"].shift(10)
    mid = sma(c, 20)
    sd = c.rolling(20, min_periods=20).std()
    f["bw"] = (4 * sd) / mid
    f["bw_rank"] = trailing_rank(f["bw"], 126)
    f["donchian20"] = s["high"].rolling(20, min_periods=20).max()
    f["low"] = s["low"]
    f["high"] = s["high"]
    f["sum4"] = c.rolling(4, min_periods=4).sum()   # for an SMA5 that includes today's price
    return f.shift(1).add_suffix("_y")


def frame(m15f: pd.DataFrame, daily: pd.DataFrame, stock_reg: pd.DataFrame | None) -> pd.DataFrame:
    """Join intraday features (regime_intraday.features) with yesterday's daily
    features and stock regime, and add the time helpers the strategies use."""
    x = m15f.join(daily_features(daily), on="session")
    if stock_reg is not None:
        x = x.join(stock_reg[["regime"]].shift(1).rename(columns={"regime": "stock_regime_y"}), on="session")
    else:
        x["stock_regime_y"] = None
    x["mod"] = minute_of_day(x["t_close"])
    last_slot = x.groupby("session")["slot"].transform("max")
    x["moc_bar"] = x["slot"] == last_slot - 1          # closes 15 min before the close
    x["trf"] = m15f["tr_factor_bar"] if "tr_factor_bar" in m15f else 1.0
    x["atr_exec"] = x["atr_prev"] / x["trf"]
    below = (x["close"] < x["vwap_s"]).astype(int)
    x["was_below"] = below.groupby(x["session"]).cummax().groupby(x["session"]).shift(1).fillna(0) > 0
    return x


def _first_per_session(mask: pd.Series, session: pd.Series) -> pd.Series:
    return mask & (mask.astype(int).groupby(session).cumsum() == 1)


def _out(x: pd.DataFrame, trigger: pd.Series, setup: pd.Series, window: pd.Series, first_bar: pd.Series,
         stop_dist: pd.Series, p: dict[str, Any], score: pd.Series,
         exit_moc: pd.Series | None = None) -> pd.DataFrame:
    a = x["atr_exec"]
    o = pd.DataFrame(index=x.index)
    o["entry"] = _first_per_session((trigger & setup & window).fillna(False), x["session"])
    o["setup_entry"] = (setup & first_bar).fillna(False)
    o["stop_dist"] = stop_dist
    o["target_dist"] = p.get("target_atr", np.nan) * a
    o["trail_after"] = p.get("trail_after_atr", np.nan) * a
    o["trail_dist"] = p.get("trail_atr", np.nan) * a
    o["max_sessions"] = p["max_sessions"]
    o["score"] = score
    o["exit_moc"] = exit_moc.fillna(False) if exit_moc is not None else False
    return o


def signals(name: str, x: pd.DataFrame, p: dict[str, Any], ex: dict[str, Any]) -> pd.DataFrame:
    lo, hi = (hhmm(t) for t in ex["signal_window"])
    window = (x["mod"] >= lo) & (x["mod"] <= hi)
    first_bar = x["mod"] == lo                           # the daily-only twin's entry bar
    a = x["atr_exec"]
    if name == "pullback":
        setup = (x["stock_regime_y"] == "bull") & (x["rsi3_y"] <= p["rsi_max"])
        trig = (x["close"] > x["vwap_s"]) & x["was_below"]
        return _out(x, trig, setup, window, first_bar, p["stop_atr"] * a, p, -x["rsi3_y"])
    if name == "vwap_reversion":
        setup = (x["close_y"] > x["sma200_y"]) & (x["rsi2_y"] <= p["rsi2_max"])
        trig = x["dist"] <= -p["stretch_atr"]
        sma5_today = (x["sum4_y"] + x["close"]) / 5        # 4 past closes + this bar's price
        exit_moc = x["moc_bar"] & (x["close"] > sma5_today)
        return _out(x, trig, setup, window, first_bar, p["stop_atr"] * a, p, -x["rsi2_y"], exit_moc)
    if name == "compression_breakout":
        setup = (x["bw_rank_y"] <= p["bw_rank_max"]) & (x["close_y"] > x["sma50_y"])
        trig = (x["close"] > x["donchian20_y"]) & (x["relvol"] >= p["relvol_min"])
        return _out(x, trig, setup, window, first_bar, p["stop_atr"] * a, p, x["relvol"])
    if name == "gap_go":
        setup = (x["close_y"] > x["sma50_y"]) & (x["sma50_slope_y"] > 0) & (x["gap"] >= p["gap_atr"])
        trig = (x["close"] > x["open_d"]) & (x["close"] > x["vwap_s"]) & (x["relvol"] >= p["relvol_min"]) \
            & (x["mod"] <= hhmm(p["latest_signal"]))
        to_low = (x["close"] - x["sess_low"]) / x["trf"]
        stop = to_low.clip(lower=p["stop_min_atr"] * a, upper=p["stop_max_atr"] * a)
        return _out(x, trig, setup, window, first_bar, stop, p, x["gap"])
    # ---- intraday-only (flat at the close) --------------------------------
    if name in INTRADAY_NAMES:
        f_lo = hhmm(p.get("first_signal", ex["signal_window"][0]))
        f_hi = hhmm(p.get("latest_signal", ex["signal_window"][1]))
        win = (x["mod"] >= f_lo) & (x["mod"] <= f_hi)
        none = pd.Series(False, index=x.index)
    if name == "gap_fade_down":
        trig = (x["gap"] <= -p["gap_atr"]) & (x["close"] < x["open_d"]) & (x["close"] < x["vwap_s"]) \
            & (x["relvol"] >= p["relvol_min"])
        return _out(x, trig, win, win, none, p["stop_atr"] * a, p, -x["gap"])
    if name == "liquidity_sweep":
        swept = (x["sess_low"] < x["low_y"])
        trig = swept & (x["close"] > x["low_y"])
        to_low = (x["close"] - x["sess_low"]) / x["trf"] + p["stop_buffer_atr"] * a
        stop = to_low.clip(lower=p["stop_min_atr"] * a, upper=p["stop_max_atr"] * a)
        return _out(x, trig, win, win, none, stop, p, -(x["sess_low"] - x["low_y"]) / x["atr_prev"])
    if name == "last_half_hour":
        first_close = x["close"].where(x["mod"] == hhmm("10:00")).groupby(x["session"]).transform("max")
        up = first_close > x["prev_close"]
        trig = up & (x["mod"] == hhmm("15:30"))
        is_sym = pd.Series(bool(x.attrs.get("key") in p["symbols"]), index=x.index)
        return _out(x, trig & is_sym, trig & is_sym, trig & is_sym, none, p["stop_atr"] * a, p,
                    (first_close / x["prev_close"] - 1))
    if name == "orb30":
        or_hi = x["high"].where(x["slot"] <= 1).groupby(x["session"]).transform("max")
        or_lo = x["low"].where(x["slot"] <= 1).groupby(x["session"]).transform("min")
        trig = (x["slot"] >= 2) & (x["close"] > or_hi) & (x["relvol"] >= p["relvol_min"])
        stop = ((x["close"] - or_lo) / x["trf"]).clip(lower=p["stop_min_atr"] * a, upper=p["stop_max_atr"] * a)
        win2 = (x["mod"] >= hhmm(ex["signal_window"][0])) & (x["mod"] <= hhmm(p["latest_signal"]))
        return _out(x, trig, win2, win2, none, stop, p, x["relvol"])
    raise KeyError(name)
