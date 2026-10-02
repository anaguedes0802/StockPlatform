"""Data-quality checks for phase 1. Each function returns a JSON-able summary.

What is checked, and why it matters for the backtest:

* membership coverage — a member-day without a bar silently shrinks the
  universe, which is survivorship bias by another name;
* OHLC consistency, non-positive prices, extreme moves — bad prints trigger
  fake stops and fake breakouts;
* missing 15m slots — an illiquid name has holes, and holes distort ATR,
  VWAP and relative volume;
* daily ↔ intraday reconciliation — the first 15m open must match the daily
  open and the RTH volume must be a stable share of daily volume, otherwise
  the two timeframes describe different markets;
* earnings coverage — a stock-month with no report on file is a stock-month
  in which the "avoid earnings" rule cannot work.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent.calendar import TradingCalendar
from app.swing_agent.store import parse_key
from app.swing_agent.universe import Interval, Membership


def _q(x: pd.Series | np.ndarray, qs=(0.5, 0.9, 0.99)) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {f"p{int(q * 100)}": round(float(np.quantile(a, q)), 4) for q in qs}


def ohlc_violations(df: pd.DataFrame) -> int:
    lo = df[["open", "close"]].min(axis=1)
    hi = df[["open", "close"]].max(axis=1)
    eps = 1e-9
    return int(((df["low"] > lo + eps) | (df["high"] < hi - eps) | (df["low"] <= 0)).sum())


def membership_coverage(m: Membership, ivs: list[Interval], daily: dict[str, pd.DataFrame],
                        cal: TradingCalendar, start: date) -> dict[str, Any]:
    last = max((df.index.max() for df in daily.values()), default=pd.Timestamp(start)).date()
    tot = have = 0
    per_gone, per_cur = [], []
    missing, partial, ended_early = [], [], []
    for iv in ivs:
        lo = max(iv.start, start)
        hi = min(iv.end or last, last)
        days = [s.day for s in cal.between(lo, hi)]
        if not days:
            continue
        df = daily.get(iv.key)
        got = 0 if df is None else int(pd.DatetimeIndex(df.index).isin(pd.to_datetime(days)).sum())
        tot += len(days)
        have += got
        cov = got / len(days)
        (per_gone if iv.end else per_cur).append(cov)
        if got == 0:
            missing.append(iv.key)
        elif cov < 0.95:
            partial.append((iv.key, round(cov, 3)))
        if df is not None and len(df) and iv.end and df.index.max().date() < cal.previous(iv.end, 5).day:
            ended_early.append((iv.key, str(df.index.max().date())))
    return {
        "member_sessions": tot, "member_sessions_with_bar": have,
        "coverage": round(have / tot, 5) if tot else None,
        "intervals_current": len(per_cur), "intervals_gone": len(per_gone),
        "coverage_current_median": round(float(np.median(per_cur)), 4) if per_cur else None,
        "coverage_gone_median": round(float(np.median(per_gone)), 4) if per_gone else None,
        "no_data": sorted(missing), "partial_lt95": sorted(partial),
        "data_ends_before_index_exit": sorted(ended_early),
    }


def daily_summary(daily: dict[str, pd.DataFrame]) -> dict[str, Any]:
    viol, nonpos, zero_vol, splits, divs = 0, 0, 0, 0, 0
    big = []
    for k, df in daily.items():
        viol += ohlc_violations(df)
        nonpos += int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        zero_vol += int((df["volume"] <= 0).sum())
        sf = df["split_factor"].round(3)
        splits += int((sf.diff().abs() > 1e-3).sum())
        divs += int((df["div_ret"] > 0).sum())
        r = df["close"].pct_change()
        for ts, v in r[r.abs() > 0.4].items():
            big.append((k, str(ts.date()), round(float(v), 3)))
    n = sum(len(df) for df in daily.values())
    return {"keys": len(daily), "bars": n, "ohlc_violations": viol, "non_positive": nonpos,
            "zero_volume_days": zero_vol, "split_events": splits, "dividend_exdates": divs,
            "abs_daily_move_gt_40pct": sorted(big, key=lambda x: -abs(x[2]))[:40],
            "n_abs_daily_move_gt_40pct": len(big)}


def intraday_summary(m15: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame],
                     cal: TradingCalendar) -> dict[str, Any]:
    cf = cal.frame()
    slots = (cf["close"] - cf["open"]).dt.total_seconds() // 900
    rows = []
    for k, df in m15.items():
        if df.empty:
            rows.append({"key": k, "bars": 0})
            continue
        d = daily.get(k)
        per = df.groupby("session").agg(n=("open", "size"), o=("open", "first"),
                                        c=("close", "last"), v=("volume", "sum"))
        exp = slots.reindex(per.index)
        rec: dict[str, Any] = {
            "key": k, "bars": len(df), "sessions": len(per),
            "missing_slot_share": round(float(1 - per["n"].sum() / exp.sum()), 5),
            "ohlc_violations": ohlc_violations(df),
            # within-session bar-to-bar moves (overnight gaps excluded)
            "abs_15m_move_gt_10pct": int((df.groupby("session")["close"].pct_change().abs() > 0.10).sum()),
        }
        if d is not None and len(d):
            j = per.join(d[["open", "close", "volume"]], how="inner", rsuffix="_d")
            span = d.loc[(d.index >= per.index.min()) & (d.index <= per.index.max())]
            rec["daily_sessions_without_15m"] = int((~span.index.isin(per.index)).sum())
            rec["open_gap_bps_median"] = round(float(((j["o"] / j["open"] - 1).abs() * 1e4).median()), 2)
            rec["open_gap_bps_p99"] = round(float(((j["o"] / j["open"] - 1).abs() * 1e4).quantile(0.99)), 2)
            rec["close_gap_bps_median"] = round(float(((j["c"] / j["close"] - 1).abs() * 1e4).median()), 2)
            rec["rth_volume_share_median"] = round(float((j["v"] / j["volume"]).median()), 4)
        rows.append(rec)
    t = pd.DataFrame(rows)
    agg = {"keys": len(t), "bars": int(t["bars"].sum())}
    for c in ("missing_slot_share", "open_gap_bps_median", "open_gap_bps_p99",
              "close_gap_bps_median", "rth_volume_share_median", "daily_sessions_without_15m",
              "abs_15m_move_gt_10pct"):
        if c in t:
            agg[c] = _q(t[c])
    agg["ohlc_violations"] = int(t.get("ohlc_violations", pd.Series(dtype=float)).sum())
    agg["worst_missing_slots"] = t.nlargest(8, "missing_slot_share")[["key", "missing_slot_share"]] \
        .to_dict("records") if "missing_slot_share" in t else []
    agg["per_key"] = t.to_dict("records")
    return agg


def earnings_coverage(earn: pd.DataFrame, pit: pd.DataFrame, symbol_of: dict[str, str] | None = None,
                      etfs: set[str] | None = None) -> dict[str, Any]:
    """Share of universe stock-months with an earnings report within ±70 days.

    Every listed US company reports quarterly, so a stock-month without one
    nearby means the calendar is missing that name (often a ticker change).
    """
    by_sym = {s: np.sort(g["date"].to_numpy()) for s, g in earn.groupby("symbol")}
    ok = tot = 0
    holes: dict[str, int] = {}
    for key, reb in pit[["key", "rebalance"]].itertuples(index=False):
        t, _ = parse_key((symbol_of or {}).get(key, key))
        if etfs and t in etfs:
            continue
        tot += 1
        d = by_sym.get(key)
        if d is None:
            d = by_sym.get(t, by_sym.get(t.replace(".", "/"), by_sym.get(t.replace(".", ""))))
        hit = d is not None and np.any(np.abs(d - np.datetime64(reb)) <= np.timedelta64(70, "D"))
        ok += bool(hit)
        if not hit:
            holes[key] = holes.get(key, 0) + 1
    timing = earn["timing"].value_counts(normalize=True).round(3).to_dict()
    by_year = earn.assign(y=earn["date"].dt.year).groupby("y")["timing"] \
        .apply(lambda s: round(float((s == "unknown").mean()), 3)).to_dict()
    return {"stock_months": tot, "covered": ok, "coverage": round(ok / tot, 4) if tot else None,
            "worst_holes": sorted(holes.items(), key=lambda x: -x[1])[:25],
            "timing_mix": timing, "unknown_timing_share_by_year": by_year}
