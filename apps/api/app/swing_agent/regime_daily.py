"""Daily regimes: the market (macro) regime and each stock's own trend regime.

Market regime, decided at each close from SPY, the VIX and breadth, on two
axes (rules and thresholds in `swing_agent.toml` [regime.daily]):

* **trend** — *bull*: SPY above its 200-day average, the 50-day average
  rising, ADX ≥ 20 (a real trend, not drift) and at least half the S&P 500
  above their own 50-day averages (broad participation); *bear*: below the
  200-day average with the 50-day average falling; *range*: anything else.
* **vol** — *high*: VIX ≥ 25 (staying on until it is back under 20), or SPY's
  ATR% in the top decile of its trailing year while below its 50-day average.

The single label is the trend, except that high vol without a bull trend is
**high_vol** (stress / panic). A bull trend in high vol stays *bull* and
carries `vol_state = "high"`, which the risk layer uses to shrink positions.

Why two axes (structure "two_axis"): the first version (structure
"priority", high_vol overriding everything) kept the 2020 recovery labelled
high_vol for 218 sessions, because the VIX stayed above 20 until 2021, and
left *bear* almost unused (1–2% of days), since most declines lift the VIX.
Both versions are in the experiment log.

A switch to a calmer regime must hold for `confirm_days` closes; a switch
into high_vol is immediate (get out fast, come back slowly).

Prices are signal prices (close × tr_factor), so dividends and spin-offs
never look like drops. The VIX settles at 16:15, so a day's regime is
stamped `t_close` = 16:15 and, through `store.known_at`, is first usable
the next morning.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent.calendar import TradingCalendar
from app.swing_agent.indicators import adx, atr, slope_atr, sma, trailing_rank
from app.swing_agent.universe import Interval

LABELS = ("bull", "range", "bear", "high_vol")
RISK_OFF = ("bear", "high_vol")


def signal_prices(df: pd.DataFrame) -> pd.DataFrame:
    f = df["tr_factor"] if "tr_factor" in df else 1.0
    return pd.DataFrame({c: df[c] * f for c in ("open", "high", "low", "close")}, index=df.index)


# ---------------------------------------------------------------------------
# Breadth
# ---------------------------------------------------------------------------

def breadth(intervals: list[Interval], daily: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex,
            ma: int = 50) -> pd.DataFrame:
    """Share of index members above their own `ma`-day average, each session.

    Membership is point-in-time: a stock counts only on days it was in the
    index. A member without `ma` bars of history yet is left out of both the
    numerator and the denominator.
    """
    cols_a: dict[str, pd.Series] = {}
    cols_m: dict[str, np.ndarray] = {}
    for iv in intervals:
        df = daily.get(iv.key)
        if df is None or df.empty:
            continue
        px = signal_prices(df)["close"]
        avg = sma(px, ma)
        lo = pd.Timestamp(iv.start)
        hi = pd.Timestamp(iv.end) if iv.end else sessions[-1]
        cols_a[iv.key] = (px > avg).where(avg.notna()).reindex(sessions)
        cols_m[iv.key] = cols_m.get(iv.key, np.zeros(len(sessions), bool)) | \
            ((sessions >= lo) & (sessions <= hi))
    above = pd.DataFrame(cols_a, index=sessions)
    member = pd.DataFrame(cols_m, index=sessions)[above.columns]
    valid = member & above.notna()
    n = valid.sum(axis=1)
    share = above.where(valid).astype(float).sum(axis=1) / n.replace(0, np.nan)
    return pd.DataFrame({"breadth": share, "n_members": n})


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def trend_features(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    s = signal_prices(df)
    a = atr(s["high"], s["low"], s["close"], p["atr_period"])
    f = pd.DataFrame(index=df.index)
    f["close"] = s["close"]
    f["sma_fast"] = sma(s["close"], p["sma_fast"])
    f["sma_slow"] = sma(s["close"], p["sma_slow"])
    f["atr"] = a
    f["slope_fast"] = slope_atr(f["sma_fast"], p["slope_fast_days"], a)
    f["adx"] = adx(s["high"], s["low"], s["close"], p["adx_period"])
    f["atr_pct"] = a / s["close"]
    f["atr_rank"] = trailing_rank(f["atr_pct"], p["atr_rank_window"])
    return f


def latch(on: pd.Series, off: pd.Series) -> pd.Series:
    """Hysteresis: switch on when `on`, stay on until `off`."""
    state, out = False, []
    for a, b in zip(on.to_numpy(), off.to_numpy()):
        if not state and a:
            state = True
        elif state and b:
            state = False
        out.append(state)
    return pd.Series(out, index=on.index)


def trend_raw(f: pd.DataFrame, p: dict[str, Any], breadth_s: pd.Series | None = None) -> pd.Series:
    bear = (f["close"] < f["sma_slow"]) & (f["slope_fast"] < 0)
    bull = (f["close"] > f["sma_slow"]) & (f["slope_fast"] > 0) & (f["adx"] >= p["adx_trend"])
    if breadth_s is not None and p.get("breadth_bull_min", 0) > 0:
        bull = bull & (breadth_s.reindex(f.index) >= p["breadth_bull_min"])
    lab = np.select([bear, bull], ["bear", "bull"], default="range").astype(object)
    ready = f[["sma_slow", "adx", "slope_fast"]].notna().all(axis=1).to_numpy()
    lab[~ready] = None
    return pd.Series(lab, index=f.index, name="trend")


def vol_high(f: pd.DataFrame, p: dict[str, Any], vix: pd.Series | None = None) -> pd.Series:
    hv = (f["atr_rank"] >= p["atr_rank_high"]) & (f["close"] < f["sma_fast"])
    if vix is not None:
        v = vix.reindex(f.index)
        hv = hv | latch(v >= p["vix_high_enter"], v < p["vix_high_exit"])
    return hv.rename("vol_high")


def raw_labels(f: pd.DataFrame, p: dict[str, Any], *, vix: pd.Series | None = None,
               breadth_s: pd.Series | None = None) -> pd.Series:
    trend = trend_raw(f, p, breadth_s)
    hv = vol_high(f, p, vix)
    if p.get("structure", "two_axis") == "priority":   # version 1, kept for the record
        lab = np.where(hv, "high_vol", trend.to_numpy()).astype(object)
    else:
        lab = np.where(hv & (trend != "bull"), "high_vol", trend.to_numpy()).astype(object)
    ready = trend.notna() & f["atr_rank"].notna()
    if vix is not None:
        ready &= vix.reindex(f.index).notna()
    lab[~ready.to_numpy()] = None
    return pd.Series(lab, index=f.index, name="raw")


def confirm(raw: pd.Series, days: int, immediate: tuple[str, ...] = ("high_vol",)) -> pd.Series:
    """Adopt a new label only after `days` consecutive raw closes agree,
    except labels in `immediate`, adopted at once."""
    cur, run_lab, run, out = None, None, 0, []
    for x in raw.to_numpy():
        if x is None or (isinstance(x, float) and np.isnan(x)):
            out.append(cur)
            continue
        run = run + 1 if x == run_lab else 1
        run_lab = x
        if cur is None or x in immediate or (x != cur and run >= days):
            cur = x
        out.append(cur)
    return pd.Series(out, index=raw.index, name="regime")


def market_regime(bench: pd.DataFrame, vix: pd.DataFrame, breadth_df: pd.DataFrame,
                  cal: TradingCalendar, p: dict[str, Any]) -> pd.DataFrame:
    f = trend_features(bench, p)
    f["vix"] = vix["close"].reindex(f.index)
    f["breadth"] = breadth_df["breadth"].reindex(f.index)
    f["trend"] = trend_raw(f, p, f["breadth"])
    f["vol_state"] = np.where(vol_high(f, p, f["vix"]), "high", "normal")
    f["raw"] = raw_labels(f, p, vix=f["vix"], breadth_s=f["breadth"])
    f["regime"] = confirm(f["raw"], p["confirm_days"])
    # usable once the VIX has settled (16:15 ET) — i.e. from the next session
    cf = cal.frame()
    f["t_close"] = pd.DatetimeIndex(cf["close"].reindex(f.index)) + pd.Timedelta(minutes=15)
    return f


def stock_regime(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    f = trend_features(df, p)
    f["trend"] = trend_raw(f, p)
    f["vol_state"] = np.where(vol_high(f, p), "high", "normal")
    f["raw"] = raw_labels(f, p)
    f["regime"] = confirm(f["raw"], p["confirm_days"])
    f["t_close"] = df["t_close"]
    return f


def spells(regime: pd.Series) -> pd.DataFrame:
    """Consecutive runs of one label: (label, start, end, sessions)."""
    r = regime.dropna()
    if r.empty:
        return pd.DataFrame(columns=["label", "start", "end", "sessions"])
    grp = (r != r.shift()).cumsum()
    g = r.groupby(grp)
    return pd.DataFrame({"label": g.first(), "start": g.apply(lambda x: x.index[0]),
                         "end": g.apply(lambda x: x.index[-1]), "sessions": g.size()}).reset_index(drop=True)


def date_range_index(cal: TradingCalendar, start: date, end: date) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(s.day) for s in cal.between(start, end)], name="session")
