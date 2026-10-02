"""Turn MarketData + regimes + events into engine Books and a Context.

Shared by the phase-3 strategy runs, the phase-4 agent and the phase-6 paper
loop, so the backtest and the live path build signals with the same code.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent import config, events, regime_intraday as ri, strategies as sg
from app.swing_agent.calendar import TradingCalendar
from app.swing_agent.data import MarketData
from app.swing_agent.engine import Book, Context
from app.swing_agent.strategies import hhmm
from app.swing_agent.universe import Membership


def session_axis(cal: TradingCalendar, start: date, end: date) -> tuple[list[pd.Timestamp], list[int], dict]:
    ss = cal.between(start, end)
    days = [pd.Timestamp(s.day) for s in ss]
    return days, [s.minutes // 15 for s in ss], {d: i for i, d in enumerate(days)}


def known_micro(lab: pd.DataFrame, f: pd.DataFrame) -> np.ndarray:
    """Intraday regime known at each bar's close: the latest completed hour's
    label (a bar that closes an hour knows that hour). Before 10:30 there is
    no completed hour: 'first_hour'."""
    m = lab["micro"].where(f["hour_close"])
    m = m.groupby(f["session"]).ffill()
    return m.fillna("first_hour").to_numpy(dtype=object)


def build_book(key: str, md: MarketData, idx: dict, react: list[int], cfg: dict[str, Any],
               stock_reg: pd.DataFrame | None, is_etf: bool, full_last_day: pd.Timestamp,
               names: tuple[str, ...] = sg.NAMES) -> Book | None:
    m15 = md.m15(key)
    daily = md.daily(key)
    if m15.empty or daily.empty:
        return None
    pin = cfg["regime"]["intraday"]
    f = ri.features(m15, daily, pin)
    lab = ri.label(f, pin)
    x = sg.frame(f, daily, stock_reg)
    x.attrs["key"] = key
    sig = {}
    for n in names:
        s = sg.signals(n, x, cfg["strategy"][n], cfg["execution"])
        sig[n] = {c: s[c].to_numpy() for c in sg.SIGNAL_COLS}
        for c in ("entry", "setup_entry", "exit_moc"):
            sig[n][c] = sig[n][c].astype(bool)
        sig[n]["max_sessions"] = sig[n]["max_sessions"].astype(int)
    sess = m15["session"].map(idx)
    keep = sess.notna().to_numpy()
    sess_i = sess[keep].astype(int).to_numpy()
    n_sess = max(idx.values()) + 1
    d_close = daily["close"].reindex(list(idx)).to_numpy()
    d_sf = daily["split_factor"].reindex(list(idx)).to_numpy()
    div = (daily["div_ret"] * daily["close"].shift(1)).reindex(list(idx)).fillna(0.0).to_numpy()
    last_si = max(idx.values()) + 10**6 if full_last_day > max(idx) else idx.get(daily.index.max(), n_sess - 1)
    vol = m15["volume"]
    med_vol = vol.rolling(26 * 20, min_periods=26 * 5).median().shift(1).to_numpy()
    stock = x["stock_regime_y"].to_numpy(dtype=object)
    return Book(
        key=key, is_etf=is_etf, sess=sess_i, slot=x["slot"].to_numpy()[keep].astype(int),
        mod=x["mod"].to_numpy()[keep].astype(int),
        o=m15["open"].to_numpy()[keep], h=m15["high"].to_numpy()[keep],
        l=m15["low"].to_numpy()[keep], c=m15["close"].to_numpy()[keep],
        sf=m15["split_factor"].to_numpy()[keep], med_vol=med_vol[keep],
        micro=known_micro(lab, f)[keep], stock_regime=stock[keep],
        sig={n: {c: v[keep] for c, v in d.items()} for n, d in sig.items()},
        daily_close=d_close, daily_sf=d_sf, div_amt=div, last_session=int(last_si), react=react,
        cash_merger=md.cash_merger(key))


def reaction_index(cal: TradingCalendar, idx: dict, root: Path) -> dict[str, list[int]]:
    e = pd.read_parquet(root / "events" / "earnings.parquet").rename(columns={"key": "symbol"})
    rs = events.reaction_sessions(e, cal)
    out: dict[str, list[int]] = {}
    for sym, g in rs.groupby("symbol"):
        out[sym] = sorted({idx[d] for d in g["session"] if d in idx})
    return out


def context(cfg: dict[str, Any], cal: TradingCalendar, days: list[pd.Timestamp], n_slots: list[int],
            idx: dict, root: Path) -> Context:
    pit = pd.read_parquet(root / "universe_pit.parquet")
    uni: dict[str, set[str]] = {}
    for reb, g in pit.groupby("rebalance"):
        uni[f"{reb.year:04d}-{reb.month:02d}"] = set(g["key"])
    mem = Membership.load(root / "membership" / "sp500_hist.csv")
    member = {iv.key: (pd.Timestamp(iv.start), pd.Timestamp(iv.end) if iv.end else None)
              for iv in mem.intervals(date.fromisoformat(cfg["data"]["history_start"]))}
    mr = pd.read_parquet(root / "regimes" / "market_daily.parquet")["regime"]
    known = mr.shift(1)            # yesterday's regime is what the morning knows
    market = {idx[d]: v for d, v in known.items() if d in idx and isinstance(v, str)}
    mac = pd.read_parquet(root / "events" / "macro.parquet")
    ex = cfg["execution"]
    fomc = {idx[d] for d in mac.loc[mac["kind"].isin(ex["macro_no_entry"]), "date"] if d in idx}
    late = {}
    for kind, t in ex["macro_late_start"].items():
        for d in mac.loc[mac["kind"] == kind, "date"]:
            if d in idx:
                late[idx[d]] = max(late.get(idx[d], 0), hhmm(t))
    return Context(sessions=days, n_slots=n_slots, universe=uni, member=member,
                   refs=set(cfg["universe"]["references"]), market_regime=market, fomc=fomc,
                   late_start=late)


def eurusd_at(day: pd.Timestamp, fallback: float) -> float:
    try:
        from app.services import swing_data

        fx = swing_data.daily_bars("EURUSD=X")
        fx.index = fx.index.tz_convert(None).normalize()
        v = fx["close"].loc[:day]
        return float(v.iloc[-1]) if len(v) else fallback
    except Exception:  # noqa: BLE001
        return fallback


def build_all(md: MarketData, cfg: dict[str, Any], days: list[pd.Timestamp], idx: dict,
              names: tuple[str, ...] = sg.NAMES, progress: Any = None) -> dict[str, Book]:
    root = config.root()
    pit = pd.read_parquet(root / "universe_pit.parquet")
    refs = list(cfg["universe"]["references"])
    keys = sorted(set(pit["key"]) | set(refs))
    react = reaction_index(md.calendar, idx, root)
    books = {}
    for i, k in enumerate(keys, 1):
        sp = root / "regimes" / "stock_daily" / f"{k}.parquet"
        sr = pd.read_parquet(sp) if sp.exists() else None
        full = md.store.read("daily", k, segment=None)
        last = full.index.max() if len(full) else pd.Timestamp.min
        b = build_book(k, md, idx, react.get(k, []), cfg, sr, k in refs, last, names)
        if b is not None:
            books[k] = b
        if progress and i % 20 == 0:
            progress(i, len(keys))
    return books
