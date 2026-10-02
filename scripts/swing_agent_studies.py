"""Three studies asked for after phase 3, rules in swing_agent.toml [study.*].

    apps/api/.venv/bin/python scripts/swing_agent_studies.py selection   # A: stock-selection model, survivorship-free
    apps/api/.venv/bin/python scripts/swing_agent_studies.py ipos        # B: buying new IPOs
    apps/api/.venv/bin/python scripts/swing_agent_studies.py spinoffs    # C: buying spun-off companies

Research window only (data ≤ 2023-12-31). Results → apps/api/artifacts/swing_agent/studies/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml import stock_selection as ss  # noqa: E402
from app.swing_agent import config, experiments, survivorship  # noqa: E402
from app.swing_agent import store as st  # noqa: E402
from app.swing_agent.alpaca_data import AlpacaData  # noqa: E402
from app.swing_agent.data import MarketData  # noqa: E402
from app.swing_agent.universe import Membership  # noqa: E402

CFG = config.load()
ROOT = config.root()
OUT = ROOT / "studies"
OUT.mkdir(parents=True, exist_ok=True)
RESEARCH_END = config.segment("research")[1]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def say(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str))


# ---------------------------------------------------------------------------
# A. stock selection
# ---------------------------------------------------------------------------

def step_selection() -> None:
    p = CFG["study"]["selection"]
    md = MarketData(segment="research")
    cal = md.calendar
    mem = Membership.load(ROOT / "membership" / "sp500_hist.csv")
    ivs = mem.intervals(date.fromisoformat(CFG["data"]["history_start"]))
    daily = {}
    for iv in ivs:
        try:
            d = md.daily(iv.key)
        except KeyError:
            continue
        if not d.empty:
            daily[iv.key] = d
    keys = sorted(daily)
    say(f"selection: {len(keys)} point-in-time S&P 500 keys")
    close = pd.DataFrame({k: daily[k]["close"] * daily[k]["tr_factor"] for k in keys})
    open_ = pd.DataFrame({k: daily[k]["open"] * daily[k]["tr_factor"] for k in keys})
    vol = pd.DataFrame({k: daily[k]["volume"] for k in keys})
    spy = md.daily("SPY")
    spy_c, spy_o = spy["close"] * spy["tr_factor"], spy["open"] * spy["tr_factor"]
    idx = close.index
    # point-in-time membership mask
    member = pd.DataFrame(False, index=idx, columns=keys)
    for iv in ivs:
        if iv.key in member:
            hi = pd.Timestamp(iv.end) if iv.end else idx[-1]
            member.loc[(idx >= pd.Timestamp(iv.start)) & (idx <= hi), iv.key] = True
    # sectors: the sector ETF each stock correlates with most
    etf = pd.DataFrame({e: (lambda d: d["close"] * d["tr_factor"])(md.daily(e)) for e in SECTOR_ETFS})
    er = etf.pct_change()
    sectors = {}
    for k in keys:
        r = close[k].pct_change()
        ok = r.notna()
        cors = er[ok].corrwith(r[ok]) if ok.sum() >= 120 else pd.Series(dtype=float)
        sectors[k] = cors.idxmax() if cors.notna().any() else None
    # earnings surprises (yfinance, cached): survivors only; renamed keys use today's ticker
    full_daily = {k: md.store.read("daily", k, segment=None) for k in keys}
    today = max(d.index.max() for d in full_daily.values()).date()
    cls = survivorship.classify(ivs, full_daily, cal, today)
    ren = survivorship.renamed_to(ivs, full_daily, cal, today)
    from app.services import earnings as earn_svc

    events, have = {}, 0
    for k in keys:
        if cls.get(k) not in ("current", "renamed_member"):
            continue
        tkr = st.parse_key(ren.get(k, k))[0].replace(".", "-")
        hist = earn_svc._disk_get(f"earn_hist:{tkr}:60", max_age_s=30 * 86400) or []
        if hist:
            have += 1
        events[k] = [ss.EarningsEvent(pd.Timestamp(e["ts"]), e["surprise_pct"]) for e in hist]
    say(f"selection: earnings history for {have} keys (survivors only; delisted count as neutral)")
    survivors = {k for k, c in cls.items() if c in ("current", "renamed_member")}
    mask_all = ss.universe_mask(close, vol) & member
    out = {"n_keys": len(keys), "n_survivors": len(survivors & set(keys)), "earnings_keys": have, "runs": {}}
    segs = {"train": ("2017-01-01", "2021-12-31"), "validation": ("2022-01-01", "2023-12-31")}
    for variant in p["variants"]:
        w = dict(ss.WEIGHTS)
        ev = events if variant == "full_composite" else None
        if variant == "momentum_only":
            w = {k: v for k, v in w.items() if not k.startswith("earn_")}
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        factors = ss.compute_factors(close, spy_c, sectors, ev)
        for uni in ("point_in_time", "survivors_only"):
            mask = mask_all.copy()
            if uni == "survivors_only":
                mask.loc[:, [k for k in keys if k not in survivors]] = False
            score, _ = ss.composite(factors, mask, w)
            days = idx[idx >= "2017-01-01"]
            runs = []
            for off in range(p["offsets"]):
                r = ss.backtest(score, open_, mask, hold=p["hold"], top=p["top"], start=str(days[off].date()))
                r["benchmark"] = np.log(spy_o.shift(-(1 + p["hold"])) / spy_o.shift(-1)).reindex(r.index)
                runs.append(r)
            res = {}
            for seg, (lo, hi) in segs.items():
                rows = [ss.summarize(r.loc[lo:hi], p["hold"]) for r in runs]
                med = lambda f: float(np.median([f(x) for x in rows]))  # noqa: E731
                rng = lambda f: [float(min(f(x) for x in rows)), float(max(f(x) for x in rows))]  # noqa: E731
                res[seg] = {
                    "long_cagr_median": med(lambda x: x["long"]["cagr_pct"]),
                    "long_cagr_range": rng(lambda x: x["long"]["cagr_pct"]),
                    "long_sharpe_median": med(lambda x: x["long"]["sharpe"]),
                    "long_maxdd_median": med(lambda x: x["long"]["max_drawdown_pct"]),
                    "equal_weight_cagr_median": med(lambda x: x["equal_weight"]["cagr_pct"]),
                    "spy_cagr_median": med(lambda x: x["spy"]["cagr_pct"]),
                    "spy_sharpe_median": med(lambda x: x["spy"]["sharpe"]),
                    "excess_vs_ew_median": med(lambda x: x["excess_vs_equal_weight_pct_per_year"]),
                    "excess_t_median": med(lambda x: x["excess_t_stat"]),
                    "rank_ic_median": med(lambda x: x["rank_ic_mean"]),
                }
                experiments.log(9, "study_selection", f"{variant}__{uni}", {"weights": w, **p}, seg, res[seg])
            out["runs"][f"{variant}__{uni}"] = res
            say(f"selection {variant:15s} {uni:15s} | " + " | ".join(
                f"{s}: book {v['long_cagr_median']}% (Sharpe {v['long_sharpe_median']}) vs EW "
                f"{v['equal_weight_cagr_median']}% vs SPY {v['spy_cagr_median']}%, excess {v['excess_vs_ew_median']}%/yr "
                f"t {v['excess_t_median']}" for s, v in res.items()))
    jdump("selection.json", out)


# ---------------------------------------------------------------------------
# B. IPOs
# ---------------------------------------------------------------------------

def _nasdaq_ipo_month(ym: str) -> list[dict]:
    path = OUT / "ipo_calendar" / f"{ym}.json"
    if path.exists():
        return json.loads(path.read_text())
    from app.swing_agent.events import _browser_get

    js = _browser_get("https://api.nasdaq.com/api/ipo/calendar", {"date": ym}).json()
    rows = (((js.get("data") or {}).get("priced") or {}).get("rows")) or []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    time.sleep(0.4)
    return rows


def _money(s: str | None) -> float:
    try:
        return float(re.sub(r"[^0-9.]", "", s or "") or "nan")
    except ValueError:
        return float("nan")


def _bars(client: AlpacaData, sym: str, start: pd.Timestamp, end: pd.Timestamp, asof: str,
          tag: str = "") -> pd.DataFrame:
    path = OUT / "bars" / f"{sym}@{asof}{'_' + tag if tag else ''}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    try:
        b = client.bars([sym], "1Day", start.strftime("%Y-%m-%d"), end, adjustment="all", asof=asof).get(sym)
    except Exception as e:  # noqa: BLE001
        say(f"bars failed {sym}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame() if b is None else b
    if len(df):
        df.index = st.session_days(df.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def event_returns(px: pd.Series, spy: pd.Series, entry_i: int, horizon: int, cost: float,
                  data_end: pd.Timestamp) -> dict | None:
    """Buy at the close `entry_i` sessions after the first close, sell `horizon`
    sessions later (or at the last close if the stock stops trading first:
    `ended_early`, i.e. delisted/acquired before the window and before the
    end of the data requested)."""
    if entry_i >= len(px):
        return None
    e_day = px.index[entry_i]
    x_i = min(entry_i + horizon, len(px) - 1)
    x_day = px.index[x_i]
    if x_day > RESEARCH_END:
        x_day = px.index[px.index <= RESEARCH_END][-1]
        x_i = px.index.get_loc(x_day)
    r = px.iloc[x_i] / px.iloc[entry_i] - 1 - cost
    s0, s1 = spy.asof(e_day), spy.asof(x_day)
    rs = s1 / s0 - 1
    return {"entry": e_day, "exit": x_day, "ret": r, "spy": rs, "bhar": r - rs,
            "full_horizon": x_i - entry_i >= horizon, "ended_early": (x_i == len(px) - 1) and
            (x_i - entry_i < horizon) and px.index[-1] < min(data_end, RESEARCH_END) - pd.Timedelta(days=10)}


def calendar_portfolio(events: list[tuple[pd.Series, pd.Timestamp, pd.Timestamp]], spy: pd.Series) -> dict:
    """Equal-weight daily portfolio of every event inside its holding window
    (Fama's calendar-time method: one return per day, so overlapping events
    are not counted as independent)."""
    rets = []
    for px, a, b in events:
        r = px.pct_change().loc[(px.index > a) & (px.index <= b)]
        rets.append(r)
    if not rets:
        return {}
    panel = pd.concat(rets, axis=1)
    port = panel.mean(axis=1).dropna()
    port = port[port.index <= RESEARCH_END]
    sp = spy.pct_change().reindex(port.index)
    ex = port - sp
    m = (1 + ex).groupby(ex.index.to_period("M")).prod() - 1
    eq = (1 + port).cumprod()
    yrs = len(port) / 252
    return {"days": len(port), "avg_names": round(float(panel.notna().sum(axis=1).reindex(port.index).mean()), 1),
            "cagr_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 2),
            "spy_cagr_pct": round(100 * ((1 + sp).prod() ** (1 / yrs) - 1), 2),
            "sharpe": round(float(port.mean() / port.std() * math.sqrt(252)), 2),
            "spy_sharpe": round(float(sp.mean() / sp.std() * math.sqrt(252)), 2),
            "max_dd_pct": round(100 * float((eq / eq.cummax() - 1).min()), 1),
            "excess_monthly_mean_pct": round(100 * float(m.mean()), 2),
            "excess_monthly_t": round(float(m.mean() / (m.std() / math.sqrt(len(m)))), 2)}


def summarize_events(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    b = df["bhar"]
    return {"n": int(len(df)), "mean_ret_pct": round(100 * df["ret"].mean(), 1),
            "median_ret_pct": round(100 * df["ret"].median(), 1),
            "mean_bhar_pct": round(100 * b.mean(), 1), "median_bhar_pct": round(100 * b.median(), 1),
            "share_beating_spy": round(float((b > 0).mean()), 3),
            "share_lost_half": round(float((df["ret"] < -0.5).mean()), 3),
            "t_bhar": round(float(b.mean() / (b.std() / math.sqrt(len(b)))), 2) if len(b) > 2 else None,
            "ended_early": int(df["ended_early"].sum())}


def step_ipos() -> None:
    p = CFG["study"]["ipo"]
    months = pd.period_range(p["first_month"], p["last_month"], freq="M")
    rows = []
    for m in months:
        for r in _nasdaq_ipo_month(str(m)):
            rows.append(r)
    cal = pd.DataFrame(rows)
    cal["price"] = cal["proposedSharePrice"].map(_money)
    cal["offer_usd"] = cal["dollarValueOfSharesOffered"].map(_money)
    cal["date"] = pd.to_datetime(cal["pricedDate"], format="%m/%d/%Y", errors="coerce")
    cal["sym"] = cal["proposedTickerSymbol"].str.strip().str.upper()
    n0 = len(cal)
    spac = cal["companyName"].str.contains(p["exclude_name_regex"], regex=True, na=False)
    unit = cal["sym"].str.len().ge(5) & cal["sym"].str[-1].isin(p["exclude_symbol_suffix"])
    keep = ~spac & ~unit & (cal["price"] >= p["min_price"]) & (cal["offer_usd"] >= p["min_offer_usd"]) \
        & cal["date"].notna() & cal["proposedExchange"].str.contains("NASDAQ|NYSE", case=False, na=False)
    cal = cal[keep].drop_duplicates("sym", keep="first").sort_values("date")
    say(f"ipos: {n0} priced deals → {len(cal)} after excluding SPACs, units, < ${p['min_offer_usd'] / 1e6:.0f}M, < ${p['min_price']}")
    client = AlpacaData(max_rpm=CFG["data"]["max_requests_per_min"])
    spy = MarketData(segment=None).daily("SPY")["tr_close"]
    recs, missing = [], 0
    for i, r in enumerate(cal.itertuples(), 1):
        asof = (r.date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end = min(r.date + pd.Timedelta(days=620), RESEARCH_END + pd.Timedelta(days=1))   # 126 + 252 sessions
        df = _bars(client, r.sym, r.date, pd.Timestamp(end, tz="UTC"), asof, tag="w620")
        if df.empty or df.index.min() > r.date + pd.Timedelta(days=10):
            missing += 1
            continue
        px = df["close"]
        for e in p["entries"]:
            ev = event_returns(px, spy, e, p["horizon"], p["cost_round_trip"], pd.Timestamp(end))
            if ev and ev["entry"] <= RESEARCH_END:
                recs.append({"sym": r.sym, "name": r.companyName, "ipo_date": r.date, "offer_usd": r.offer_usd,
                             "ipo_price": r.price, "first_close": float(px.iloc[0]),
                             "day1_pop": float(px.iloc[0] / r.price - 1) if r.price else np.nan,
                             "entry_rule": e, **ev})
        if i % 100 == 0:
            say(f"ipos: {i}/{len(cal)} ({client.requests} requests)")
    t = pd.DataFrame(recs)
    t.to_parquet(OUT / "ipo_events.parquet")
    split = pd.Timestamp(p["train_until"])
    res = {"deals_priced": n0, "deals_kept": int(len(cal)), "no_price_data": missing,
           "day1_pop_median_pct": round(100 * t.loc[t["entry_rule"] == 0, "day1_pop"].median(), 1), "by_entry": {}}
    for e in p["entries"]:
        g = t[t["entry_rule"] == e]
        full = g[g["full_horizon"] | g["ended_early"]]
        res["by_entry"][str(e)] = {"train": summarize_events(full[full["ipo_date"] <= split]),
                                   "validation": summarize_events(full[full["ipo_date"] > split])}
        evs = []
        for sym, gg in g.groupby("sym"):
            path = next((OUT / "bars").glob(f"{sym}@*_w620.parquet"), None)
            if path is None:
                continue
            px = pd.read_parquet(path)["close"]
            evs.append((px, gg["entry"].iloc[0], gg["exit"].iloc[0]))
        res["by_entry"][str(e)]["calendar_time_portfolio"] = calendar_portfolio(evs, spy)
        for seg in ("train", "validation"):
            experiments.log(9, "study_ipo", f"entry_{e}", p, seg, res["by_entry"][str(e)][seg])
        v = res["by_entry"][str(e)]
        say(f"ipos entry +{e:3d} sessions | train {v['train']} | validation {v['validation']} | "
            f"calendar-time {v['calendar_time_portfolio']}")
    jdump("ipos.json", res)


# ---------------------------------------------------------------------------
# C. spin-offs
# ---------------------------------------------------------------------------

def step_spinoffs() -> None:
    p = CFG["study"]["spinoff"]
    acts = json.loads((ROOT / "corporate_actions.json").read_text())
    mem = Membership.load(ROOT / "membership" / "sp500_hist.csv")
    rows = []
    for r in acts.get("spin_offs", []):
        d = pd.Timestamp(r["ex_date"])
        if not (pd.Timestamp(p["first"]) <= d <= pd.Timestamp(p["last"])):
            continue
        parent = r.get("source_symbol")
        if parent not in mem.members_on(d.date()):
            continue
        rows.append({"parent": parent, "new": r.get("new_symbol"), "ex_date": d,
                     "ratio": float(r.get("new_rate") or 0) / float(r.get("source_rate") or 1)})
    sp = pd.DataFrame(rows).drop_duplicates(["new", "ex_date"]).sort_values("ex_date")
    say(f"spin-offs by S&P 500 members {p['first'][:4]}–{p['last'][:4]}: {len(sp)}")
    client = AlpacaData(max_rpm=CFG["data"]["max_requests_per_min"])
    spy = MarketData(segment=None).daily("SPY")["tr_close"]
    recs, evs = [], {h: [] for h in p["horizons"]}
    for r in sp.itertuples():
        if not r.new:
            continue
        sym = r.new.replace(".WI", "").split(".")[0]
        df = _bars(client, sym, r.ex_date - pd.Timedelta(days=5),
                   pd.Timestamp(min(r.ex_date + pd.Timedelta(days=420), RESEARCH_END + pd.Timedelta(days=1)), tz="UTC"),
                   (r.ex_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
        if df.empty:
            recs.append({"parent": r.parent, "new": sym, "ex_date": r.ex_date, "note": "no data"})
            continue
        px = df["close"][df.index >= r.ex_date]
        for h in p["horizons"]:
            ev = event_returns(px, spy, 0, h, p["cost_round_trip"],
                               min(r.ex_date + pd.Timedelta(days=420), RESEARCH_END))
            if ev:
                recs.append({"parent": r.parent, "new": sym, "ex_date": r.ex_date, "horizon": h, **ev})
                evs[h].append((px, ev["entry"], ev["exit"]))
    t = pd.DataFrame(recs)
    t.to_parquet(OUT / "spinoff_events.parquet")
    split = pd.Timestamp(p["train_until"])
    res = {"spinoffs": int(len(sp)), "no_data": int((t.get("note") == "no data").sum()) if "note" in t else 0,
           "list": sp.assign(ex_date=sp["ex_date"].dt.date.astype(str)).to_dict("records"), "by_horizon": {}}
    for h in p["horizons"]:
        g = t[t.get("horizon") == h] if "horizon" in t else pd.DataFrame()
        res["by_horizon"][str(h)] = {"train": summarize_events(g[g["ex_date"] <= split]),
                                     "validation": summarize_events(g[g["ex_date"] > split]),
                                     "calendar_time_portfolio": calendar_portfolio(evs[h], spy)}
        for seg in ("train", "validation"):
            experiments.log(9, "study_spinoff", f"horizon_{h}", p, seg, res["by_horizon"][str(h)][seg])
        v = res["by_horizon"][str(h)]
        say(f"spin-offs {h} sessions | train {v['train']} | validation {v['validation']} | "
            f"calendar-time {v['calendar_time_portfolio']}")
    jdump("spinoffs.json", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["selection", "ipos", "spinoffs"])
    a = ap.parse_args()
    for s in a.steps:
        say(f"== {s}")
        {"selection": step_selection, "ipos": step_ipos, "spinoffs": step_spinoffs}[s]()


if __name__ == "__main__":
    main()
