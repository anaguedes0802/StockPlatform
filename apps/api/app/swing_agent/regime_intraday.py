"""Intraday (micro) regime per stock, on each 1h close of the session.

Built from the session's 15m bars up to that moment. Distances are in daily
ATRs as of the previous close (the daily ATR of today is not known yet).
Labels, in priority order (thresholds in `swing_agent.toml` [regime.intraday]):

1. **gap_go_up / gap_go_down** — opened ≥ 0.5 ATR away from yesterday's close,
   is still beyond the open and on the gap side of the session VWAP, on
   volume ≥ 1.5× normal for that time of day.
2. **reversal_down / reversal_up** — either a gap that has been half filled
   and is back through VWAP, or a ≥ 0.5 ATR extension from the open that has
   come back through both the VWAP and the open.
3. **trend_up / trend_down** — ≥ 75% of the 15m closes so far on one side of
   VWAP and the price ≥ 0.25 ATR away from it.
4. **range** — anything else.

Separately, **vol_state**: the session's range so far vs its normal range at
the same time of day (expanding ≥ 1.3×, compressing ≤ 0.7×).

Every feature at bar i uses bars ≤ i of the same session plus daily bars up
to yesterday; the same-time-of-day baselines use previous sessions only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent.indicators import atr, same_time_baseline, session_vwap, slot
from app.swing_agent.regime_daily import signal_prices

LABELS = ("gap_go_up", "gap_go_down", "reversal_up", "reversal_down", "trend_up", "trend_down",
          "range")
DIRECTION = {"gap_go_up": 1, "reversal_up": 1, "trend_up": 1,
             "gap_go_down": -1, "reversal_down": -1, "trend_down": -1, "range": 0}


def features(m15: pd.DataFrame, daily: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    """Per-15m-bar features. `m15` from MarketData (has session, t_close,
    tr_factor); `daily` the same key's corrected daily frame."""
    trf = m15["tr_factor"] if "tr_factor" in m15 else 1.0
    b = pd.DataFrame({c: m15[c] * trf for c in ("open", "high", "low", "close", "vwap")},
                     index=m15.index)
    b["volume"] = m15["volume"]
    b["tr_factor_bar"] = trf
    b["session"] = m15["session"]
    b["t_close"] = m15["t_close"]
    sess = b["session"]

    d = signal_prices(daily)
    d_atr = atr(d["high"], d["low"], d["close"], 14)
    prev = pd.DataFrame({"prev_close": d["close"], "atr_prev": d_atr}).shift(1)  # as of yesterday
    b = b.join(prev, on="session")

    g = b.groupby("session")
    b["slot"] = slot(b)
    b["open_d"] = g["open"].transform("first")
    b["gap"] = (b["open_d"] - b["prev_close"]) / b["atr_prev"]
    b["vwap_s"] = session_vwap(b)
    b["dist"] = (b["close"] - b["vwap_s"]) / b["atr_prev"]
    above = (b["close"] > b["vwap_s"]).astype(float)
    b["side_share"] = above.groupby(sess).cumsum() / (b["slot"] + 1)
    sign = np.sign(b["close"] - b["vwap_s"])
    flips = (sign != sign.groupby(sess).shift()) & sign.groupby(sess).shift().notna() & (sign != 0)
    b["vwap_crosses"] = flips.astype(int).groupby(sess).cumsum()
    b["sess_high"] = g["high"].cummax()
    b["sess_low"] = g["low"].cummin()
    b["ext_up"] = (b["sess_high"] - b["open_d"]) / b["atr_prev"]
    b["ext_down"] = (b["open_d"] - b["sess_low"]) / b["atr_prev"]
    b["move"] = (b["close"] - b["open_d"]) / b["atr_prev"]
    cumvol = b["volume"].groupby(sess).cumsum()
    rng = b["sess_high"] - b["sess_low"]
    lb = p["relvol_lookback"]
    b["relvol"] = cumvol / same_time_baseline(cumvol, sess, b["slot"], lb)
    b["range_ratio"] = rng / same_time_baseline(rng, sess, b["slot"], lb)
    last_slot = g["slot"].transform("max")
    b["hour_close"] = ((b["slot"] % 4) == 3) | (b["slot"] == last_slot)
    return b


def label(f: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    g = p["gap_atr"]
    up_gap, dn_gap = f["gap"] >= g, f["gap"] <= -g
    above, below = f["close"] > f["vwap_s"], f["close"] < f["vwap_s"]
    rv = f["relvol"] >= p["gap_go_relvol"]
    gap_go_up = up_gap & (f["close"] > f["open_d"]) & above & rv
    gap_go_dn = dn_gap & (f["close"] < f["open_d"]) & below & rv
    half_back = (f["close"] - f["prev_close"]) / (f["open_d"] - f["prev_close"]) <= 0.5
    fade_up = up_gap & below & half_back          # up-gap given back → reversal down
    fade_dn = dn_gap & above & half_back
    r = p["reversal_ext_atr"]
    rev_dn = fade_up | ((f["ext_up"] >= r) & below & (f["close"] < f["open_d"]))
    rev_up = fade_dn | ((f["ext_down"] >= r) & above & (f["close"] > f["open_d"]))
    s, dist = p["trend_side_share"], p["trend_dist_atr"]
    tr_up = (f["side_share"] >= s) & (f["dist"] >= dist)
    tr_dn = (f["side_share"] <= 1 - s) & (f["dist"] <= -dist)
    lab = np.select([gap_go_up, gap_go_dn, rev_dn, rev_up, tr_up, tr_dn],
                    ["gap_go_up", "gap_go_down", "reversal_down", "reversal_up", "trend_up",
                     "trend_down"], default="range").astype(object)
    ready = f[["atr_prev", "prev_close", "vwap_s"]].notna().all(axis=1).to_numpy()
    lab[~ready] = None
    vs = np.select([f["range_ratio"] >= p["vol_expand"], f["range_ratio"] <= p["vol_compress"]],
                   ["expanding", "compressing"], default="normal").astype(object)
    vs[f["range_ratio"].isna().to_numpy()] = None
    return pd.DataFrame({"micro": lab, "vol_state": vs}, index=f.index)


def micro_regime(m15: pd.DataFrame, daily: pd.DataFrame, p: dict[str, Any],
                 hourly_only: bool = True) -> pd.DataFrame:
    f = features(m15, daily, p)
    out = f.join(label(f, p))
    return out[out["hour_close"]] if hourly_only else out
