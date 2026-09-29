"""Historical replay of the paper forward test — the exact live rules, run on the past.

`paper_strategy` trades forward, which is the only unfakeable test but takes
months. This replays the same rules day by day over history so the Live test
page has something to compare against from day one:

  - rebalance when >= REBALANCE_EVERY_DAYS calendar days have passed
  - SPY above its 10-month SMA -> top 10% of the composite, else SHY
  - orders from the SAME `plan_orders` as the live test (drift band, cash
    buffer, budget cap), decided at the close and filled at the NEXT open with
    SLIPPAGE_BPS + COMMISSION_BPS per side
  - holdings are share counts, so weights drift between rebalances exactly as
    they would in the account

It is still a backtest: the universe is today's S&P 500 + 400, so companies
that failed or left the index are missing (survivorship bias flatters the
result, most before ~2014). Treat SPY-relative numbers as an upper bound.

Data: Yahoo daily bars (split/dividend adjusted, consolidated volume) and the
earnings history, both cached on disk.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import log
from app.ml import stock_selection as ss
from app.services import paper_strategy as ps

_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "paper_strategy"
PANEL_PATH = _DIR / "replay_panel.pkl"
RESULT_PATH = _DIR / "replay.json"


def load_panel(start: str = "2005-01-01", max_age_days: int = 7) -> dict[str, Any]:
    """Daily open/close/volume for the S&P 500+400 plus SPY and SHY."""
    if PANEL_PATH.exists() and time.time() - PANEL_PATH.stat().st_mtime < max_age_days * 86400:
        with open(PANEL_PATH, "rb") as f:
            return pickle.load(f)
    import yfinance as yf

    symbols = ps._universe() + ["SPY", ps.DEFENSIVE]
    parts = []
    for i in range(0, len(symbols), 100):   # Yahoo drops very large batch requests
        for attempt in range(4):
            df = yf.download(symbols[i:i + 100], start=start, auto_adjust=True, progress=False,
                             threads=False, group_by="column")
            if len(df):
                break
            time.sleep(20 * (attempt + 1))
        parts.append(df)
    panel: dict[str, Any] = {}
    for fld in ("Open", "Close", "Volume"):
        frame = pd.concat([p[fld] for p in parts if len(p)], axis=1)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        panel[fld.lower()] = frame.sort_index()
    panel["sectors"] = ps.universe_sectors()
    _DIR.mkdir(parents=True, exist_ok=True)
    with open(PANEL_PATH, "wb") as f:
        pickle.dump(panel, f)
    return panel


def load_events(symbols: list[str], pause_s: float = 0.6) -> dict[str, list[ss.EarningsEvent]]:
    from app.services import earnings as earnings_svc
    out = {}
    for s in symbols:
        cached = earnings_svc._disk_get(f"earn_hist:{s}:60", max_age_s=7 * 86400) is not None
        rows = earnings_svc.earnings_history(s, limit=60)
        if not cached:
            time.sleep(pause_s)
        out[s] = [ss.EarningsEvent(pd.Timestamp(r["ts"]), r["surprise_pct"]) for r in rows]
    return out


def run(start: str = "2016-01-04", capital: float = 10_000.0, gated: bool = True,
        save: bool = True) -> dict[str, Any]:
    panel = load_panel()
    close_all, open_all, vol_all = panel["close"], panel["open"], panel["volume"]
    stocks = [s for s in close_all.columns if s not in ("SPY", ps.DEFENSIVE)
              and close_all[s].notna().sum() > 300]
    events = load_events(stocks)
    close = close_all[stocks]
    factors = ss.compute_factors(close, close_all["SPY"], panel.get("sectors") or {}, events)
    mask = ss.universe_mask(close, vol_all[stocks])
    score, _ = ss.composite(factors, mask)
    risk_on = ss.market_risk_on(close_all["SPY"])

    # price arrays for fast daily marking (carry last price through gaps)
    tradables = stocks + [ps.DEFENSIVE]
    cl = close_all[tradables].ffill()
    op = open_all[tradables]
    col = {s: i for i, s in enumerate(tradables)}
    dates = cl.index[cl.index >= start]
    slip, comm = ps.SLIPPAGE_BPS / 1e4, ps.COMMISSION_BPS / 1e4

    cash = capital
    qty = np.zeros(len(tradables))
    last_reb = None
    curve, rebs = [], []
    n_orders = 0
    for i, d in enumerate(dates[:-1]):
        px = cl.loc[d].values
        held = np.nan_to_num(qty * px)
        value = cash + held.sum()
        curve.append((d, value, float(close_all.at[d, "SPY"])))
        if last_reb is not None and (d - last_reb).days < ps.REBALANCE_EVERY_DAYS:
            continue
        on = bool(risk_on.get(d, True)) if gated else True
        if on:
            row = score.loc[d].dropna()
            n = max(5, int(len(row) * ss.TOP_FRACTION))
            targets = list(row.sort_values(ascending=False).index[:n])
        else:
            targets = [ps.DEFENSIVE]
        positions = {s: {"market_value": float(held[j]), "price": float(px[j])}
                     for s, j in col.items() if qty[j] > 1e-9}
        q_now = {s: float(qty[j]) for s, j in col.items() if qty[j] > 1e-9}
        sells, buys, _ = ps.plan_orders(targets, positions, q_now, cash, value)

        nxt = dates[i + 1]
        o = op.loc[nxt]
        turnover = 0.0
        for x in sells:
            j = col[x["symbol"]]
            p = o.iat[j]
            if not (p > 0):
                p = px[j]   # no open print (halt / data gap): use last close
            q = min(x["qty"], qty[j])
            proceeds = q * p * (1 - slip) * (1 - comm)
            qty[j] -= q
            cash += proceeds
            turnover += proceeds
            n_orders += 1
        spend_total = sum(b["notional"] for b in buys)
        scale = min(1.0, max(0.0, cash) / spend_total) if spend_total > 0 else 1.0
        for b in buys:
            j = col[b["symbol"]]
            p = o.iat[j]
            if not (p > 0):
                continue
            notional = b["notional"] * scale
            q = notional / (1 + comm) / (p * (1 + slip))
            qty[j] += q
            cash -= notional
            turnover += notional
            n_orders += 1
        last_reb = d
        rebs.append({"date": d.date().isoformat(), "risk_on": on, "n_targets": len(targets),
                     "n_sells": len(sells), "n_buys": len(buys),
                     "turnover_pct": round(turnover / value * 100, 1) if value else 0.0})

    curve_df = pd.DataFrame(curve, columns=["date", "value", "spy"]).set_index("date")
    result = _summarize(curve_df, rebs, capital, n_orders, gated, start)
    result["holdings_end"] = sorted(
        [{"symbol": s, "value": round(float(qty[j] * cl.iloc[-1].values[j]), 2)}
         for s, j in col.items() if qty[j] > 1e-9], key=lambda h: -h["value"])[:120]
    if save:
        _DIR.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, default=str))
    return result


def _stats(v: pd.Series) -> dict[str, float]:
    r = np.log(v).diff().dropna()
    yrs = len(r) / 252
    if yrs <= 0 or r.std() == 0:
        return {}
    eq = v / v.iloc[0]
    return {"cagr_pct": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1)}


def _summarize(c: pd.DataFrame, rebs: list[dict], capital: float, n_orders: int,
               gated: bool, start: str) -> dict[str, Any]:
    segs = {}
    for lab, lo, hi in (("full", None, None), ("2016-2019", "2016", "2019-12-31"),
                        ("2020-2022", "2020", "2022-12-31"), ("2023+", "2023", None)):
        part = c.loc[lo:hi] if lo or hi else c
        if len(part) > 60:
            segs[lab] = {"strategy": _stats(part["value"]), "spy": _stats(part["spy"])}
    yearly = []
    for y, g in c.groupby(c.index.year):
        if len(g) > 20:
            yearly.append({"year": int(y), "strategy_pct": round((g["value"].iloc[-1] / g["value"].iloc[0] - 1) * 100, 1),
                           "spy_pct": round((g["spy"].iloc[-1] / g["spy"].iloc[0] - 1) * 100, 1)})
    weekly = c.resample("W-FRI").last().dropna()
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "start": start, "end": c.index[-1].date().isoformat(), "capital": capital, "gated": gated,
        "final_value": round(float(c["value"].iloc[-1]), 2),
        "segments": segs, "yearly": yearly,
        "n_rebalances": len(rebs), "n_orders": n_orders,
        "time_risk_on_pct": round(100 * sum(r["risk_on"] for r in rebs) / max(1, len(rebs)), 1),
        "avg_turnover_pct": round(float(np.mean([r["turnover_pct"] for r in rebs[1:]])) if len(rebs) > 1 else 0.0, 1),
        "rebalances": rebs[-36:],
        "curve": [{"day": d.date().isoformat(),
                   "strategy_pct": round((row.value / capital - 1) * 100, 2),
                   "spy_pct": round((row.spy / weekly["spy"].iloc[0] - 1) * 100, 2)}
                  for d, row in weekly.iterrows()],
        "caveats": [
            "Backtest, not live: the universe is today's S&P 500+400, so failed/delisted companies are missing "
            "(survivorship bias) — treat the gap to SPY as an upper bound.",
            "Fills at the next open with 5 bps slippage + 1 bp commission per side; no taxes.",
            "The forward (paper) test is the real check; this replay shows what the same rules did historically.",
        ],
    }


def cached_result() -> dict[str, Any] | None:
    try:
        return json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else None
    except Exception as e:
        log.warning("paper_replay.read_failed", err=str(e)[:120])
        return None
