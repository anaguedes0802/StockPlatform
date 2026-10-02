"""Performance metrics for equity curves and trade lists.

Conventions: daily returns from the session-close equity; Sharpe and Sortino
annualised with √252 and a zero risk-free rate (idle cash earns nothing in
the engine either, so both sides of the ratio are consistent); max drawdown
on the close-to-close curve. "Costs as % of gross profit" = all spread,
slippage and fees ÷ the P&L the trades would have made without them.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def curve(eq: pd.Series) -> dict[str, Any]:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252)
    down = r[r < 0]
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under == 0).cumsum()).sum().max()) if under.any() else 0
    return {"start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
            "total_return_pct": round(100 * (eq.iloc[-1] / eq.iloc[0] - 1), 2),
            "cagr_pct": round(100 * cagr, 2), "vol_pct": round(100 * vol, 2),
            "sharpe": round(float(r.mean() / r.std(ddof=1) * np.sqrt(252)), 3) if r.std() > 0 else None,
            "sortino": round(float(r.mean() / down.std(ddof=1) * np.sqrt(252)), 3) if len(down) > 1 else None,
            "max_dd_pct": round(100 * float(dd.min()), 2), "longest_dd_sessions": longest}


def trades(t: pd.DataFrame) -> dict[str, Any]:
    if t.empty:
        return {"n": 0}
    wins, losses = t.loc[t["pnl"] > 0, "pnl"], t.loc[t["pnl"] <= 0, "pnl"]
    gross = t["pnl"] + t["costs"]
    r = t["r"].dropna()
    return {
        "n": int(len(t)),
        "win_rate": round(float((t["pnl"] > 0).mean()), 3),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
        "mean_r": round(float(r.mean()), 3), "median_r": round(float(r.median()), 3),
        "t_mean_r": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 1 else None,
        "mean_sessions": round(float(t["sessions"].mean()), 2),
        "net_pnl": round(float(t["pnl"].sum()), 2),
        "costs": round(float(t["costs"].sum()), 2),
        "costs_pct_of_gross_profit": round(float(100 * t["costs"].sum() / gross.sum()), 1)
        if gross.sum() > 0 else None,
        "costs_r_per_trade": round(float((t["costs"] / t["risk"]).mean()), 3),
        "exit_reasons": t["reason"].value_counts().to_dict(),
    }


def by_group(t: pd.DataFrame, cols: list[str], min_n: int = 1) -> list[dict[str, Any]]:
    out = []
    if t.empty:
        return out
    for keys, g in t.groupby(cols, dropna=False):
        if len(g) < min_n:
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = g["r"].dropna()
        wins, losses = g.loc[g["pnl"] > 0, "pnl"], g.loc[g["pnl"] <= 0, "pnl"]
        out.append({**{c: (None if pd.isna(k) else k) for c, k in zip(cols, keys)},
                    "n": int(len(g)), "mean_r": round(float(r.mean()), 3),
                    "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else None,
                    "win_rate": round(float((g["pnl"] > 0).mean()), 3),
                    "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
                    "total_r": round(float(r.sum()), 1)})
    return sorted(out, key=lambda x: -x["n"])


def yearly(eq: pd.Series) -> dict[int, float]:
    y = eq.groupby(eq.index.year).last()
    first = eq.iloc[0]
    prev = pd.concat([pd.Series([first]), y.iloc[:-1]]).to_numpy()
    return {int(k): round(100 * (v / p - 1), 2) for k, v, p in zip(y.index, y.to_numpy(), prev)}
