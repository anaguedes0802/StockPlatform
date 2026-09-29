"""Insider cluster-buy strategy: event study with placebo controls + portfolio.

Pre-declared design (fixed before any result was seen)
-----------------------------------------------------
* **Cluster buy**: ≥ 2 distinct officers/directors file open-market purchases
  (Form 4 code P, acquired) of the same company within 30 calendar days,
  totalling ≥ $100k. Purchases flagged as pre-scheduled (Rule 10b5-1,
  reported since 2023) are excluded. The event fires on the FILING date that
  completes the cluster: that is when the market could first know.
  One event per company per 6 months.
* **Trade**: buy at the open of the first session after the filing date, sell
  at the open 126 sessions (~6 months) later. 10 bps cost per side.
* **Universe**: companies whose SEC CIK still maps to a listed ticker on a US
  exchange today (NYSE / Nasdaq / NYSE American / Cboe), price ≥ $5 and median
  daily dollar volume ≥ $5M over the prior 60 sessions at the event.
  **Survivorship**: firms that were later delisted or acquired have no
  current ticker and are dropped. This is counted and reported.
* **Primary test**: 6-month return minus SPY ("abnormal return", AR) of the
  events, minus the same measure for the *same stocks on random dates* with no
  insider purchases within ±6 months (the placebo). The placebo absorbs the
  survivorship and stock-selection bias; what is left is the timing value of
  following insiders. Verdict "edge" only if the 90% interval (bootstrap
  clustered by calendar month) is above zero.
* **Secondary**: a placebo matched on the prior 6-month return (insiders
  often buy after drops, so "is it just buying losers?"); split halves;
  liquidity terciles; cluster size; and a calendar-time portfolio (max 20
  equal slots) against SPY and against 20 placebo portfolios.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.core.logging import log
from app.services import insider_bulk as ib
from app.services import swing_data as sd

WINDOW_DAYS = 30
MIN_INSIDERS = 2
MIN_VALUE = 100_000.0
COOLDOWN_DAYS = 182
HOLD = 126
COST = 0.0010
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 5_000_000.0
LIQ_LOOKBACK = 60
PLACEBO_K = 5
EXCHANGES = {"NYSE", "Nasdaq", "NYSE American", "NYSE MKT", "CBOE", "Cboe", "BATS"}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_CACHE = Path(__file__).resolve().parents[2] / "artifacts" / "insider_bulk"
_BARS_MAX_AGE = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def qualifying_purchases(tx: pd.DataFrame) -> pd.DataFrame:
    p = tx[(tx["code"] == "P") & (tx["acq_disp"] == "A") & ib.is_officer_or_director(tx["relationship"])
           & ~tx["planned_10b5_1"] & (tx["value"] > 0) & (tx["price"] > 0)]
    return p


def detect_clusters(tx: pd.DataFrame, *, window_days: int = WINDOW_DAYS, min_insiders: int = MIN_INSIDERS,
                    min_value: float = MIN_VALUE, cooldown_days: int = COOLDOWN_DAYS) -> pd.DataFrame:
    """Cluster-buy events, keyed to the filing date that completes the cluster."""
    p = qualifying_purchases(tx)
    if p.empty:
        return pd.DataFrame(columns=["issuer_cik", "signal_date", "n_insiders", "total_value",
                                     "ceo_cfo", "issuer_symbol", "issuer_name"])
    agg = (p.groupby(["issuer_cik", "filing_date", "owner_cik"], as_index=False)
             .agg(value=("value", "sum"), title=("title", "first"),
                  symbol=("issuer_symbol", "last"), name=("issuer_name", "last")))
    events = []
    win = pd.Timedelta(days=window_days)
    cool = pd.Timedelta(days=cooldown_days)
    for cik, g in agg.groupby("issuer_cik", sort=False):
        g = g.sort_values("filing_date")
        dates = g["filing_date"].to_numpy()
        last_event = None
        for d in np.unique(dates):
            d = pd.Timestamp(d)
            if last_event is not None and d - last_event < cool:
                continue
            w = g[(g["filing_date"] > d - win) & (g["filing_date"] <= d)]
            n = w["owner_cik"].nunique()
            v = float(w["value"].sum())
            if n >= min_insiders and v >= min_value:
                titles = " ".join(w["title"].fillna("").str.lower())
                events.append({
                    "issuer_cik": int(cik), "signal_date": d, "n_insiders": int(n), "total_value": v,
                    "ceo_cfo": any(k in titles for k in ("ceo", "chief executive", "cfo", "chief financial")),
                    "issuer_symbol": w["symbol"].iloc[-1], "issuer_name": w["name"].iloc[-1],
                })
                last_event = d
    cols = ["issuer_cik", "signal_date", "n_insiders", "total_value", "ceo_cfo", "issuer_symbol", "issuer_name"]
    if not events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(events)[cols].sort_values("signal_date").reset_index(drop=True)


def current_listings(max_age_s: int = _BARS_MAX_AGE) -> pd.DataFrame:
    """SEC's current CIK → ticker/exchange map (first ticker per CIK)."""
    path = _CACHE / "company_tickers_exchange.json"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_s:
        r = httpx.get(_TICKERS_URL, headers={"User-Agent": "StockPlatform/0.1 contact@example.com"}, timeout=30)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw["data"], columns=raw["fields"])
    df = df.drop_duplicates(subset=["cik"], keep="first")
    return df.rename(columns={"cik": "issuer_cik"})


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_bars(symbols: list[str], *, verbose: bool = False, workers: int = 4) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Daily bars for many symbols (a few parallel requests; cached on disk)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(s: str) -> tuple[str, pd.DataFrame]:
        df = sd.daily_bars(s, start="2004-06-01", max_age_s=_BARS_MAX_AGE)
        if df.empty:
            time.sleep(2.0)  # one polite retry: Yahoo occasionally 429s a burst
            df = sd.daily_bars(s, start="2004-06-01", max_age_s=_BARS_MAX_AGE, refresh=True)
        return s, df

    bars, missing = {}, []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (s, df) in enumerate(pool.map(one, symbols)):
            if df.empty:
                missing.append(s)
            else:
                bars[s] = df
            if verbose and (i + 1) % 250 == 0:
                print(f"  prices {i + 1}/{len(symbols)} ({len(missing)} missing)", flush=True)
    return bars, missing


@dataclass
class _Series:
    dates: np.ndarray
    open: np.ndarray
    close: np.ndarray
    dollar_vol_med: np.ndarray
    ret_prior: np.ndarray


def _series(df: pd.DataFrame) -> _Series:
    dv = (df["close"] * df["volume"]).rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    prior = df["close"] / df["close"].shift(HOLD) - 1
    return _Series(df.index.tz_localize(None).to_numpy(), df["open"].to_numpy(float), df["close"].to_numpy(float),
                   dv.to_numpy(float), prior.to_numpy(float))


def _trade(s: _Series, spy: _Series, i: int, *, hold: int = HOLD, cost: float = COST,
           asof_end: np.datetime64) -> dict[str, Any] | None:
    """Return / abnormal return for a signal on session index i (entry at i+1)."""
    if i + 1 >= len(s.dates):
        return None
    entry_d = s.dates[i + 1]
    j = i + 1 + hold
    if j < len(s.dates):
        exit_px, exit_d, how = s.open[j], s.dates[j], "time"
    else:
        # Data ended early. If it ended well before today the stock stopped
        # trading under this ticker: exit at its last close. Otherwise the
        # trade simply hasn't matured yet.
        if (asof_end - s.dates[-1]) / np.timedelta64(1, "D") < 10:
            return None
        exit_px, exit_d, how = s.close[-1], s.dates[-1], "data_end"
    entry_px = s.open[i + 1]
    if not (entry_px > 0 and exit_px > 0):
        return None
    r = exit_px * (1 - cost) / (entry_px * (1 + cost)) - 1
    a = np.searchsorted(spy.dates, entry_d, side="left")
    b = np.searchsorted(spy.dates, exit_d, side="left")
    if a >= len(spy.dates) or b >= len(spy.dates):
        return None
    spy_r = spy.open[b] / spy.open[a] - 1
    return {"entry_date": pd.Timestamp(entry_d), "exit_date": pd.Timestamp(exit_d), "ret": r,
            "spy_ret": spy_r, "ar": r - spy_r, "exit_how": how}


def _eligible(s: _Series, i: int) -> bool:
    return (i >= LIQ_LOOKBACK and np.isfinite(s.dollar_vol_med[i]) and s.dollar_vol_med[i] >= MIN_DOLLAR_VOL
            and s.close[i] >= MIN_PRICE)


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def _cluster_boot(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Mean with a 90% bootstrap interval, resampling whole calendar months."""
    df = pd.DataFrame({"v": values, "g": groups})
    by = [g["v"].to_numpy() for _, g in df.groupby("g")]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, len(by), len(by))
        allv = np.concatenate([by[p] for p in pick])
        means[k] = allv.mean()
    lo, hi = np.percentile(means, [5, 95])
    return float(values.mean()), float(lo), float(hi)


def run_study(tx: pd.DataFrame, bars: dict[str, pd.DataFrame], spy: pd.DataFrame, listings: pd.DataFrame,
              *, asof: pd.Timestamp | None = None, seed: int = 7) -> dict[str, Any]:
    asof_end = np.datetime64(pd.Timestamp(asof or pd.Timestamp.now(tz="UTC")).tz_localize(None))
    events = detect_clusters(tx)
    n_all = len(events)
    ev = events.merge(listings[["issuer_cik", "ticker", "exchange"]], on="issuer_cik", how="left")
    counts = {"cluster_events": n_all, "no_current_listing": int(ev["ticker"].isna().sum())}
    ev = ev[ev["ticker"].notna()]
    counts["not_major_exchange"] = int((~ev["exchange"].isin(EXCHANGES)).sum())
    ev = ev[ev["exchange"].isin(EXCHANGES)]
    spy_s = _series(spy)
    series = {k: _series(v) for k, v in bars.items()}
    purchases = qualifying_purchases(tx)
    buy_dates = {int(c): g["filing_date"].to_numpy() for c, g in purchases.groupby("issuer_cik")}

    rows, no_price, illiquid, immature = [], 0, 0, 0
    for e in ev.itertuples(index=False):
        s = series.get(e.ticker)
        if s is None:
            no_price += 1
            continue
        i = int(np.searchsorted(s.dates, np.datetime64(e.signal_date), side="right")) - 1
        if i < 0 or not _eligible(s, i):
            illiquid += 1
            continue
        t = _trade(s, spy_s, i, asof_end=asof_end)
        if t is None:
            immature += 1
            continue
        rows.append({"ticker": e.ticker, "issuer_cik": e.issuer_cik, "signal_date": e.signal_date,
                     "n_insiders": e.n_insiders, "total_value": e.total_value, "ceo_cfo": e.ceo_cfo,
                     "dollar_vol": s.dollar_vol_med[i], "prior_ret": s.ret_prior[i], "i": i, **t})
    counts.update({"no_price_data": no_price, "failed_liquidity": illiquid, "not_matured": immature,
                   "studied": len(rows)})
    E = pd.DataFrame(rows)
    if E.empty:
        return {"counts": counts, "error": "no events survived the filters"}

    # ---- placebos: same stocks, random dates with no insider buying nearby ----
    rng = np.random.default_rng(seed)
    P, M = [], []
    for tick, g in E.groupby("ticker"):
        s = series[tick]
        cik = int(g["issuer_cik"].iloc[0])
        bd = buy_dates.get(cik, np.array([], dtype="datetime64[ns]"))
        hi = len(s.dates) - HOLD - 2
        if hi <= LIQ_LOOKBACK:
            continue
        cand = np.arange(LIQ_LOOKBACK, hi)
        # exclude sessions within ±HOLD sessions of any qualifying insider purchase filing
        if len(bd):
            pos = np.searchsorted(s.dates, bd)
            bad = np.zeros(len(s.dates), dtype=bool)
            for p_ in pos:
                bad[max(0, p_ - HOLD):min(len(s.dates), p_ + HOLD + 1)] = True
            cand = cand[~bad[cand]]
        if len(cand):
            dv, cl = s.dollar_vol_med[cand], s.close[cand]
            cand = cand[np.isfinite(dv) & (dv >= MIN_DOLLAR_VOL) & (cl >= MIN_PRICE)]
        cand = cand[s.dates[cand] >= np.datetime64("2006-01-01")] if len(cand) else cand
        if len(cand) == 0:
            continue
        for e in g.itertuples(index=False):
            for k in rng.choice(cand, size=min(PLACEBO_K, len(cand)), replace=False):
                t = _trade(s, spy_s, int(k), asof_end=asof_end)
                if t:
                    P.append({"ticker": tick, "signal_date": pd.Timestamp(s.dates[k]), **t})
            # momentum-matched: prior 6-month return within ±10 points of the event's
            if np.isfinite(e.prior_ret):
                pr = s.ret_prior[cand]
                near = cand[np.isfinite(pr) & (np.abs(pr - e.prior_ret) <= 0.10)]
                for k in rng.choice(near, size=min(PLACEBO_K, len(near)), replace=False) if len(near) else []:
                    t = _trade(s, spy_s, int(k), asof_end=asof_end)
                    if t:
                        M.append({"ticker": tick, "signal_date": pd.Timestamp(s.dates[k]), **t})
    P, M = pd.DataFrame(P), pd.DataFrame(M)

    def month(d: pd.Series) -> np.ndarray:
        return d.dt.to_period("M").astype(str).to_numpy()

    ev_mean, ev_lo, ev_hi = _cluster_boot(E["ar"].to_numpy(), month(E["signal_date"]))
    pl_mean = float(P["ar"].mean()) if len(P) else float("nan")
    mm_mean = float(M["ar"].mean()) if len(M) else float("nan")
    # Primary: event AR minus the placebo mean for the same stock, per event.
    pl_by = P.groupby("ticker")["ar"].mean() if len(P) else pd.Series(dtype=float)
    mm_by = M.groupby("ticker")["ar"].mean() if len(M) else pd.Series(dtype=float)
    E["ar_minus_placebo"] = E["ar"] - E["ticker"].map(pl_by)
    E["ar_minus_matched"] = E["ar"] - E["ticker"].map(mm_by)
    prim = E.dropna(subset=["ar_minus_placebo"])
    d_mean, d_lo, d_hi = _cluster_boot(prim["ar_minus_placebo"].to_numpy(), month(prim["signal_date"]))
    sec = E.dropna(subset=["ar_minus_matched"])
    m_mean, m_lo, m_hi = (_cluster_boot(sec["ar_minus_matched"].to_numpy(), month(sec["signal_date"]))
                          if len(sec) >= 30 else (float("nan"),) * 3)

    def split(frame: pd.DataFrame, col: str) -> dict[str, Any]:
        f = frame.dropna(subset=[col])
        return {"n": int(len(f)), "mean_pct": round(float(f[col].mean() * 100), 2) if len(f) else None,
                "win_rate_pct": round(float((f[col] > 0).mean() * 100), 1) if len(f) else None}

    halves = {
        "2006-2015": split(E[E["signal_date"] < "2016-01-01"], "ar_minus_placebo"),
        "2016-2026": split(E[E["signal_date"] >= "2016-01-01"], "ar_minus_placebo"),
    }
    terc = pd.qcut(E["dollar_vol"].rank(method="first"), 3, labels=["smaller", "middle", "larger"])
    by_liq = {str(k): split(g, "ar_minus_placebo") for k, g in E.groupby(terc, observed=True)}
    by_size = {"2 insiders": split(E[E["n_insiders"] == 2], "ar_minus_placebo"),
               "3+ insiders": split(E[E["n_insiders"] >= 3], "ar_minus_placebo"),
               "CEO/CFO involved": split(E[E["ceo_cfo"]], "ar_minus_placebo")}

    pct = lambda x: round(x * 100, 2) if x == x else None  # noqa: E731 — NaN-safe
    verdict = ("edge" if d_lo > 0 else "no_edge" if d_hi < 0 else "not_distinguishable")
    return {
        "counts": counts,
        "event_ar": {"mean_pct": pct(ev_mean), "lo_pct": pct(ev_lo), "hi_pct": pct(ev_hi),
                     "win_rate_pct": round(float((E["ar"] > 0).mean() * 100), 1),
                     "mean_raw_ret_pct": pct(float(E["ret"].mean()))},
        "placebo_ar_mean_pct": pct(pl_mean), "matched_placebo_ar_mean_pct": pct(mm_mean),
        "primary": {"mean_pct": pct(d_mean), "lo_pct": pct(d_lo), "hi_pct": pct(d_hi), "n": int(len(prim))},
        "momentum_matched": {"mean_pct": pct(m_mean), "lo_pct": pct(m_lo), "hi_pct": pct(m_hi), "n": int(len(sec))},
        "halves": halves, "by_liquidity": by_liq, "by_cluster": by_size,
        "verdict": verdict,
        "_events": E, "_placebo": P,
    }


# ---------------------------------------------------------------------------
# Calendar-time portfolio
# ---------------------------------------------------------------------------

def simulate_portfolio(events: pd.DataFrame, bars: dict[str, pd.DataFrame], spy: pd.DataFrame, *,
                       max_positions: int = 20, cost: float = COST, initial: float = 100_000.0,
                       start: str = "2006-01-01") -> pd.Series:
    """Equal-slot portfolio: each event gets equity/max_positions at its entry
    open (if a slot is free) and is sold at its exit. Returns daily equity."""
    cal = spy.index[spy.index >= pd.Timestamp(start, tz="UTC")]
    ev = events.sort_values(["entry_date", "total_value"], ascending=[True, False]) if "total_value" in events \
        else events.sort_values("entry_date")
    by_entry: dict[pd.Timestamp, list] = {}
    for e in ev.itertuples(index=False):
        by_entry.setdefault(pd.Timestamp(e.entry_date).tz_localize("UTC"), []).append(e)
    px = {k: (v["open"], v["close"]) for k, v in bars.items()}
    cash, held = initial, {}   # key → [ticker, units, exit_date]
    last_close: dict[str, float] = {}
    eq = []
    for d in cal:
        # exits at the open
        for key in [k for k, h in held.items() if h[2] <= d]:
            tick, units, _ = held.pop(key)
            o = px[tick][0].get(d)
            p = o if (o is not None and o == o and o > 0) else last_close.get(tick, 0.0)
            cash += units * p * (1 - cost)
        # entries at the open
        for e in by_entry.get(d, []):
            if len(held) >= max_positions:
                continue
            o = px[e.ticker][0].get(d)
            if o is None or not o > 0:
                continue
            equity = cash + sum(h[1] * last_close.get(h[0], 0.0) for h in held.values())
            alloc = min(equity / max_positions, cash)
            if alloc <= 0:
                continue
            units = alloc / (o * (1 + cost))
            cash -= alloc
            held[(e.ticker, d)] = [e.ticker, units, pd.Timestamp(e.exit_date).tz_localize("UTC")]
        for h in held.values():
            c = px[h[0]][1].get(d)
            if c is not None and c == c:
                last_close[h[0]] = c
        eq.append(cash + sum(h[1] * last_close.get(h[0], 0.0) for h in held.values()))
    return pd.Series(eq, index=cal)
