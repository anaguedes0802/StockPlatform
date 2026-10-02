"""How good is a regime classification, before any strategy uses it?

* **Shape:** share of time per label, switches per year, spell lengths,
  transition matrix. A regime that flips every few days is noise for a
  2–10 day swing system.
* **Information:** forward return and realized volatility after each label
  (daily), forward drift in the label's direction (intraday). Labels that
  don't separate the future are only descriptions.
* **Lag:** for each benchmark drawdown ≥ 10%, how many sessions after the peak
  the regime turned risk-off, how much of the drawdown had happened by then,
  and how much of the rebound passed before it turned bull again.

Intraday t-stats are clustered by session: the 40 stocks of one day share
the market's move, so they are not 40 independent observations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent.regime_daily import RISK_OFF, spells
from app.swing_agent.regime_intraday import DIRECTION


def shape(regime: pd.Series, per_year: float = 252.0) -> dict[str, Any]:
    r = regime.dropna()
    sp = spells(r)
    n = len(r)
    switches = int((r != r.shift()).sum() - 1) if n else 0
    out = {
        "sessions": n,
        "share": r.value_counts(normalize=True).round(4).to_dict(),
        "switches": switches,
        "switches_per_year": round(switches / (n / per_year), 2) if n else None,
        "spell_median": sp.groupby("label")["sessions"].median().to_dict(),
        "spell_mean": sp.groupby("label")["sessions"].mean().round(1).to_dict(),
        "spells": sp.groupby("label").size().to_dict(),
    }
    nxt = r.shift(-1)
    tm = pd.crosstab(r[:-1], nxt[:-1], normalize="index").round(3) if n > 1 else pd.DataFrame()
    out["transition"] = tm.to_dict("index")
    return out


def forward_daily(bench: pd.DataFrame, regime: pd.Series, horizon: int) -> pd.DataFrame:
    """Forward return from the NEXT open (the regime is known after the close)
    to the close `horizon` sessions later, and realized vol over that window."""
    px = bench["close"] * bench.get("tr_factor", 1.0)
    op = bench["open"] * bench.get("tr_factor", 1.0)
    ret = px.pct_change()
    entry = op.shift(-1)
    exit_ = px.shift(-horizon)
    fwd = exit_ / entry - 1
    vol = ret[::-1].rolling(horizon).std()[::-1].shift(-1) * np.sqrt(252)
    return pd.DataFrame({"regime": regime, "fwd_ret": fwd, "fwd_vol": vol}).dropna()


def forward_table(fd: pd.DataFrame, horizon: int) -> dict[str, Any]:
    g = fd.groupby("regime")
    t = pd.DataFrame({
        "days": g.size(),
        "independent_obs": (g.size() / horizon).round(0),
        "fwd_ret_mean_pct": (100 * g["fwd_ret"].mean()).round(2),
        "fwd_ret_median_pct": (100 * g["fwd_ret"].median()).round(2),
        "fwd_up_share": g["fwd_ret"].apply(lambda x: (x > 0).mean()).round(3),
        "fwd_vol_mean_pct": (100 * g["fwd_vol"].mean()).round(1),
    })
    return t.to_dict("index")


def vol_separation(fd: pd.DataFrame, risk_off: tuple[str, ...] = RISK_OFF) -> float | None:
    off = fd["regime"].isin(risk_off)
    if not off.any() or off.all():
        return None
    return round(float(fd.loc[off, "fwd_vol"].mean() / fd.loc[~off, "fwd_vol"].mean()), 3)


def drawdowns(px: pd.Series, min_dd: float) -> list[dict[str, Any]]:
    """Peak → trough → recovery episodes deeper than `min_dd`."""
    px = px.dropna()
    out, peak_i, i = [], 0, 0
    vals, idx = px.to_numpy(), px.index
    while i < len(vals):
        if vals[i] >= vals[peak_i]:
            peak_i = i
            i += 1
            continue
        # in a drawdown from peak_i: find trough until recovery
        j = i
        trough_i = i
        while j < len(vals) and vals[j] < vals[peak_i]:
            if vals[j] < vals[trough_i]:
                trough_i = j
            j += 1
        depth = vals[trough_i] / vals[peak_i] - 1
        if depth <= -min_dd:
            out.append({"peak": idx[peak_i], "trough": idx[trough_i],
                        "recovered": idx[j] if j < len(vals) else None, "depth": round(float(depth), 4)})
        peak_i = j if j < len(vals) else peak_i
        i = j
    return out


def detection(episodes: list[dict[str, Any]], regime: pd.Series, px: pd.Series,
              risk_off: tuple[str, ...] = RISK_OFF) -> list[dict[str, Any]]:
    out = []
    r = regime.dropna()
    for e in episodes:
        after = r[r.index > e["peak"]]
        hit = after[after.isin(risk_off)]
        rec: dict[str, Any] = {"peak": str(e["peak"].date()), "trough": str(e["trough"].date()),
                               "depth_pct": round(100 * e["depth"], 1)}
        already = r.loc[:e["peak"]].iloc[-1] if len(r.loc[:e["peak"]]) else None
        rec["risk_off_at_peak"] = already in risk_off if already else None
        if len(hit):
            d = hit.index[0]
            rec["detected"] = str(d.date())
            rec["lag_sessions"] = int(((r.index > e["peak"]) & (r.index <= d)).sum())
            dd_then = px.loc[d] / px.loc[e["peak"]] - 1
            rec["drawdown_at_detection_pct"] = round(100 * float(dd_then), 1)
            rec["share_of_drawdown_before_detection"] = round(float(dd_then / e["depth"]), 2)
            if d > e["trough"]:
                rec["note"] = "detected after the trough"
        bull = r[(r.index > e["trough"]) & (r == "bull")]
        if len(bull):
            b = bull.index[0]
            rec["back_to_bull"] = str(b.date())
            rec["rebound_missed_pct"] = round(100 * float(px.loc[b] / px.loc[e["trough"]] - 1), 1)
            rec["sessions_trough_to_bull"] = int(((r.index > e["trough"]) & (r.index <= b)).sum())
        out.append(rec)
    return out


def agreement(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1).dropna()
    return round(float((j.iloc[:, 0] == j.iloc[:, 1]).mean()), 4) if len(j) else float("nan")


# ---------------------------------------------------------------------------
# Intraday
# ---------------------------------------------------------------------------

def forward_intraday(micro: pd.DataFrame, session_close: pd.Series) -> pd.DataFrame:
    """Forward move from each hourly close to the session's last close, in
    daily ATRs, and the label's direction (+1 / −1; `range` gets the
    reversion sign, −sign of the distance to VWAP). The session's last bar
    has no forward and is dropped."""
    last = micro["session"].map(session_close)
    fwd = (last - micro["close"]) / micro["atr_prev"]
    d = micro["micro"].map(DIRECTION).astype(float)
    rng = micro["micro"] == "range"
    d[rng] = -np.sign(micro.loc[rng, "dist"])
    final = micro["slot"] == micro.groupby("session")["slot"].transform("max")
    out = pd.DataFrame({"session": micro["session"], "micro": micro["micro"],
                        "vol_state": micro["vol_state"], "fwd_atr": fwd, "dir": d,
                        "slot": micro["slot"]})
    return out[~final].dropna(subset=["micro", "fwd_atr"])


def add_relative(x: pd.DataFrame) -> pd.DataFrame:
    """fwd relative to the average stock at the same session and hour, so the
    market's own afternoon move doesn't masquerade as label information."""
    x = x.copy()
    x["fwd_rel_atr"] = x["fwd_atr"] - x.groupby(["session", "slot"])["fwd_atr"].transform("mean")
    x["signed_fwd_atr"] = x["fwd_atr"] * x["dir"]
    x["signed_rel_atr"] = x["fwd_rel_atr"] * x["dir"]
    return x


def clustered(x: pd.DataFrame, value: str, by: str) -> dict[str, Any]:
    """Mean of `value` per `by` group with a t-stat clustered by session."""
    out = {}
    for lab, g in x.groupby(by):
        per_day = g.groupby("session")[value].mean()
        n = len(per_day)
        se = per_day.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out[lab] = {"obs": len(g), "sessions": n,
                    "mean_per_day": round(float(per_day.mean()), 4),   # what the t-stat tests
                    "mean_pooled": round(float(g[value].mean()), 4),
                    "t_clustered": round(float(per_day.mean() / se), 2) if se and se > 0 else None,
                    "share": round(len(g) / len(x), 4)}
    return out
