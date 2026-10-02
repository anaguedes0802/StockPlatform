"""Swing agent — phase 1: data and point-in-time universe.

Usage (from the repo root; steps are resumable, finished keys are skipped):
    apps/api/.venv/bin/python scripts/swing_agent_data.py calendar membership
    apps/api/.venv/bin/python scripts/swing_agent_data.py daily        # ~770 keys x raw/split/all
    apps/api/.venv/bin/python scripts/swing_agent_data.py universe     # monthly PIT top-N
    apps/api/.venv/bin/python scripts/swing_agent_data.py intraday     # 15m for every key ever in the universe
    apps/api/.venv/bin/python scripts/swing_agent_data.py earnings macro
    apps/api/.venv/bin/python scripts/swing_agent_data.py report       # quality + survivorship-bias report
    apps/api/.venv/bin/python scripts/swing_agent_data.py all

Everything lands in apps/api/artifacts/swing_agent/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

import pandas as pd  # noqa: E402

from app.swing_agent import config, corporate, events, quality, reference, survivorship  # noqa: E402
from app.swing_agent import store as st  # noqa: E402
from app.swing_agent.alpaca_data import AlpacaData  # noqa: E402
from app.swing_agent.calendar import TradingCalendar  # noqa: E402
from app.swing_agent.universe import Membership  # noqa: E402

CFG = config.load()
ROOT = config.root()
START = CFG["data"]["history_start"]
TODAY = datetime.now(timezone.utc).date()
END_TS = pd.Timestamp(TODAY, tz="UTC")  # bars up to yesterday's close
MEMBERSHIP_CSV = ROOT / "membership" / "sp500_hist.csv"


def say(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def client() -> AlpacaData:
    return AlpacaData(feed=CFG["data"]["feed"], max_rpm=CFG["data"]["max_requests_per_min"])


def load_cal() -> TradingCalendar:
    return TradingCalendar.load(ROOT / "calendar.json")


def load_membership() -> tuple[Membership, list]:
    m = Membership.load(MEMBERSHIP_CSV)
    return m, m.intervals(date.fromisoformat(START))


def all_daily_keys() -> list[str]:
    m, ivs = load_membership()
    refs = CFG["universe"]["references"] + CFG["universe"]["context_etfs"]
    return sorted({iv.key for iv in ivs} | set(refs))


# ---------------------------------------------------------------------------

def step_calendar() -> None:
    cal = TradingCalendar.fetch(client(), "2015-01-01", "2027-12-31", ROOT / "calendar.json")
    say(f"calendar: {len(cal.sessions)} sessions {cal.first} → {cal.last}, "
        f"{sum(s.early_close for s in cal.sessions)} early closes")


def step_membership() -> None:
    Membership.download(CFG["universe"]["membership_url"], MEMBERSHIP_CSV)
    m, ivs = load_membership()
    meta = {"url": CFG["universe"]["membership_url"], "sha256": st.file_sha256(MEMBERSHIP_CSV),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_change": m.last_update.isoformat(), "n_current": len(m.current),
            "n_intervals_since_start": len(ivs),
            "n_gone_intervals": sum(1 for iv in ivs if iv.end)}
    (ROOT / "membership" / "meta.json").write_text(json.dumps(meta, indent=1))
    say("membership:", meta)


def step_daily(refresh: bool = False) -> None:
    c, cal, bs = client(), load_cal(), st.BarStore()
    keys = [k for k in all_daily_keys() if refresh or not bs.has("daily", k)]
    groups: dict[str | None, list[str]] = defaultdict(list)
    for k in keys:
        t, asof = st.parse_key(k)
        groups[asof].append(t)
    batches = []
    for asof, tickers in groups.items():
        for i in range(0, len(tickers), 100):
            batches.append((asof, tickers[i:i + 100]))
    say(f"daily: {len(keys)} keys to fetch in {len(batches)} batches")
    missing = []
    for bi, (asof, tickers) in enumerate(batches, 1):
        got = {a: c.bars(tickers, "1Day", START, END_TS, adjustment=a, asof=asof)
               for a in ("raw", "split", "all")}
        for t in tickers:
            k = st.store_key(t, asof)
            if t not in got["split"]:
                missing.append(k)
                continue
            df = st.daily_frame(got["raw"][t], got["split"][t], got["all"][t], cal)
            bs.write("daily", k, df, symbol=t, asof=asof)
        bs.save_manifest()
        if bi % 10 == 0 or bi == len(batches):
            say(f"daily: batch {bi}/{len(batches)}, requests so far {c.requests}")
    (ROOT / "daily_missing.json").write_text(json.dumps(sorted(missing)))
    say(f"daily done; no Alpaca data for {len(missing)} keys: {missing[:30]}")


def step_cik() -> None:
    """SEC CIK per key: identifies share classes of one company (GOOG/GOOGL)
    and is the handle for 8-K earnings dates. ETFs have none.

    Resolution order: EDGAR's legacy ticker lookup (knows some delisted
    tickers, e.g. TWTR) → SEC's current ticker file → the current key a
    renamed key maps to (FB@… → META) → for delisted names, Alpaca's asset
    name searched on EDGAR, accepted only if that CIK filed an earnings 8-K
    while the key was in the index.
    """
    path = ROOT / "cik_map.json"
    cmap: dict[str, int | None] = json.loads(path.read_text()) if path.exists() else {}
    src: dict[str, str] = json.loads((ROOT / "cik_source.json").read_text()) \
        if (ROOT / "cik_source.json").exists() else {}
    etfs = set(CFG["universe"]["references"]) | set(CFG["universe"]["context_etfs"])
    keys = [k for k in all_daily_keys() if st.parse_key(k)[0] not in etfs and not k.startswith("^")]
    todo = [k for k in keys if k not in cmap]
    say(f"cik: {len(todo)} keys for the legacy lookup")
    for i, k in enumerate(todo, 1):
        cmap[k] = events.cik_for_ticker(st.parse_key(k)[0])
        if cmap[k]:
            src[k] = "edgar_legacy_lookup"
        if i % 100 == 0:
            say(f"cik: {i}/{len(todo)}")

    from app.services import edgar

    for k in keys:
        if not cmap.get(k) and "@" not in k and (c := edgar.cik_for(k.replace(".", "-"))):
            cmap[k], src[k] = c, "sec_company_tickers"
    cal = load_cal()
    m, ivs = load_membership()
    daily = load_daily([iv.key for iv in ivs], raw=True)
    last_day = max(df.index.max() for df in daily.values()).date()
    for k, cur in survivorship.renamed_to(ivs, daily, cal, last_day).items():
        if not cmap.get(k) and cmap.get(cur):
            cmap[k], src[k] = cmap[cur], f"renamed_to:{cur}"

    from app.services import broker_alpaca

    ivk = {iv.key: iv for iv in ivs}
    for k in keys:
        if cmap.get(k) or "@" not in k:
            continue
        try:
            asset = broker_alpaca.get_asset(st.parse_key(k)[0])
        except Exception:  # noqa: BLE001 — unknown symbol
            continue
        if asset.get("status") != "inactive" or not asset.get("name"):
            continue  # symbol reused by another company today: name is not ours
        hit = events.cik_by_name(asset["name"])
        if not hit:
            continue
        iv = ivk[k]
        since = max(str(iv.start), START)
        dates = [d for d, _ in events.edgar_earnings(hit[0], since=since) if d.date() <= iv.end]
        if dates:
            cmap[k], src[k] = hit[0], f"alpaca_name:{asset['name']} -> {hit[1]}"
    # same entity through name changes (CBS → VIAC → PARA) shares one CIK
    adjp = ROOT / "adjustments.json"
    adj = json.loads(adjp.read_text()) if adjp.exists() else {}
    for k in keys:
        if cmap.get(k):
            continue
        t, _ = st.parse_key(k)
        end = str(ivk[k].end or TODAY)
        for k2, a in adj.items():
            if k2 != k and cmap.get(k2) and any(
                    sym == t and (lo is None or lo <= end) and (hi is None or end <= hi)
                    for sym, lo, hi in a.get("periods", [])):
                cmap[k], src[k] = cmap[k2], f"same_entity:{k2}"
                break
    for k, ov in CFG["universe"].get("cik_overrides", {}).items():
        r = events._sec_get(events._SUBMISSIONS.format(cik=int(ov["cik"])))
        name = r.json().get("name", "") if r is not None else ""
        iv = ivk.get(k)
        dates = [d for d, _ in events.edgar_earnings(int(ov["cik"]), since=max(str(iv.start), START))
                 if d.date() <= iv.end] if iv else []
        if ov["name"].upper() in name.upper() and dates:
            cmap[k], src[k] = int(ov["cik"]), f"override verified: {name}, {len(dates)} 8-K 2.02"
        else:
            say(f"cik: override for {k} REJECTED (SEC name {name!r}, {len(dates)} earnings 8-Ks)")
    path.write_text(json.dumps(cmap, indent=0, sort_keys=True))
    (ROOT / "cik_source.json").write_text(json.dumps(src, indent=0, sort_keys=True))
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    miss_u = sorted(k for k in set(pit["key"]) if k in cmap and not cmap[k])
    say(f"cik: {sum(not v for v in cmap.values())} of {len(cmap)} keys without a CIK; "
        f"in the universe: {miss_u}")


def company_map() -> dict[str, str]:
    p = ROOT / "cik_map.json"
    return {k: str(v) for k, v in json.loads(p.read_text()).items() if v} if p.exists() else {}


def load_daily(keys: list[str], segment: str | None = None, raw: bool = False) -> dict[str, pd.DataFrame]:
    """Daily bars per key: corrected (MarketData) once the 'corporate' step
    has run, Alpaca's stored bars before that (raw=True forces those)."""
    if raw or not (ROOT / "adjustments.json").exists():
        bs = st.BarStore()
        return {k: df for k in keys if not (df := bs.read("daily", k, segment=segment)).empty}
    from app.swing_agent.data import MarketData

    md = MarketData(segment=segment)
    out = {}
    for k in keys:
        try:
            df = md.daily(k)
        except KeyError:
            continue
        if not df.empty:
            out[k] = df
    return out


def step_universe() -> None:
    cal = load_cal()
    m, ivs = load_membership()
    u = CFG["universe"]
    daily = load_daily([iv.key for iv in ivs])
    kw = dict(top_n=u["top_n"], lookback=u["rank_lookback_sessions"], min_price=u["min_price"],
              min_dollar_volume=u["min_median_dollar_volume"], min_coverage=u["min_coverage"],
              same_company_corr=u["same_company_corr"])
    first = cal.next(date.fromisoformat(START), u["rank_lookback_sessions"]).day
    last = cal.last_closed(END_TS).day
    from app.swing_agent.universe import build_universe

    pit = build_universe(m, ivs, daily, cal, first, last, company_of=company_map(), **kw)
    pit.to_parquet(ROOT / "universe_pit.parquet")
    say(f"universe: {pit['rebalance'].nunique()} months, {pit['key'].nunique()} distinct keys "
        f"({first} → {last})")


def intraday_spans() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """15m fetch window per key: from 3 months before its first month in the
    universe to 2 months after its last (indicator warm-up + positions that
    outlive a universe exit). Reference ETFs get the full history."""
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    lo0 = pd.Timestamp(START, tz="UTC")
    spans = {}
    for k, g in pit.groupby("key")["rebalance"]:
        lo = max(lo0, pd.Timestamp(g.min(), tz="UTC") - pd.Timedelta(days=92))
        hi = min(END_TS, pd.Timestamp(g.max(), tz="UTC") + pd.Timedelta(days=92))
        spans[k] = (lo, hi)
    for k in [*CFG["universe"]["references"], "IWM"]:
        spans[k] = (lo0, END_TS)
    return spans


def intraday_keys() -> list[str]:
    return sorted(intraday_spans())


def step_intraday(refresh: bool = False) -> None:
    c, cal, bs = client(), load_cal(), st.BarStore()
    spans = intraday_spans()
    keys = [k for k in sorted(spans) if refresh or not bs.has("m15", k)]
    months = sum((spans[k][1] - spans[k][0]).days / 30.4 for k in keys)
    say(f"intraday: {len(keys)} keys, ~{months:.0f} key-months to fetch (15Min, split-adjusted); "
        f"Alpaca pages are ~1 month, so ~{months / CFG['data']['max_requests_per_min']:.0f} min")
    t0, done = time.time(), 0.0
    for i, k in enumerate(keys, 1):
        t, asof = st.parse_key(k)
        lo, hi = spans[k]
        raw = c.bars([t], CFG["data"]["intraday_timeframe"], lo, hi,
                     adjustment="split", asof=asof).get(t)
        done += (hi - lo).days / 30.4
        if raw is None or raw.empty:
            say(f"intraday: no bars for {k}")
            continue
        rth, pm = st.split_sessions(raw, cal)
        bs.write("m15", k, rth, symbol=t, asof=asof, raw_rows=len(raw),
                 span=[str(lo.date()), str(hi.date())])
        bs.write("premarket", k, pm, symbol=t, asof=asof)
        bs.save_manifest()
        el = time.time() - t0
        say(f"intraday: {i}/{len(keys)} {k}: {len(rth)} RTH bars, {c.requests} req, "
            f"eta {el / done * (months - done) / 60:.1f} min")


def step_earnings() -> None:
    cal = load_cal()
    d0, d1 = date.fromisoformat(START) - timedelta(days=7), TODAY - timedelta(days=1)
    n = events.sync_earnings(cal, d0, d1, ROOT / "events" / "earnings",
                             progress=lambda n, tot: say(f"earnings: {n}/{tot} days"))
    nas = events.earnings_table(ROOT / "events" / "earnings")
    nas.to_parquet(ROOT / "events" / "earnings_nasdaq.parquet")
    say(f"earnings/nasdaq: {n} fetched; {len(nas)} reports, {nas['symbol'].nunique()} symbols")

    # SEC 8-K Item 2.02 for every stock that is ever in the universe
    m, ivs = load_membership()
    cmap = json.loads((ROOT / "cik_map.json").read_text())
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    daily = load_daily([iv.key for iv in ivs])
    last_day = max(df.index.max() for df in daily.values()).date()
    ren = survivorship.renamed_to(ivs, daily, cal, last_day)
    rows, by_cik = [], {}
    for k in sorted(pit["key"].unique()):
        cik = cmap.get(k)
        if not cik:
            continue
        if cik not in by_cik:
            by_cik[cik] = events.edgar_earnings(int(cik), since="2015-06-01")
        span = daily[k].index
        rows += [(k, d, t) for d, t in by_cik[cik] if span.min() <= d <= span.max()]
    edgar = pd.DataFrame(rows, columns=["symbol", "date", "timing"])
    # Nasdaq rows carry today's ticker; relabel to the key they belong to
    sym_of = {k: st.parse_key(ren.get(k, k))[0] for k in pit["key"].unique()}
    nas_k = pd.concat([nas[nas["symbol"] == s].assign(symbol=k) for k, s in sym_of.items()],
                      ignore_index=True)
    nas_k = nas_k[nas_k.apply(lambda r: daily[r["symbol"]].index.min() <= r["date"]
                              <= daily[r["symbol"]].index.max(), axis=1)]
    merged = events.merge_earnings(edgar, nas_k[["symbol", "date", "timing"]])
    merged = merged.rename(columns={"symbol": "key"})
    merged.to_parquet(ROOT / "events" / "earnings.parquet")
    say(f"earnings merged: {len(merged)} reports for {merged['key'].nunique()} keys; "
        f"sources {merged['source'].value_counts().to_dict()}; "
        f"timing {merged['timing'].value_counts().to_dict()}")


def step_macro() -> None:
    tbl = events.macro_table(int(START[:4]), TODAY.year)
    tbl.to_parquet(ROOT / "events" / "macro.parquet")
    say("macro:", tbl.groupby("kind").size().to_dict(), "first", tbl["date"].min().date(),
        "last", tbl["date"].max().date())


def step_corporate() -> None:
    """Fetch corporate actions, validate splits, value spin-offs, save per-key adjustments."""
    c, bs = client(), st.BarStore()
    keys = all_daily_keys()
    tickers = sorted({st.parse_key(k)[0] for k in keys})
    path = ROOT / "corporate_actions.json"
    acts = corporate.fetch(c, tickers, "2015-06-01", str(TODAY), path)
    extra = sorted({x for r in acts.get("name_changes", [])
                    for x in (r.get("old_symbol"), r.get("new_symbol")) if x} - set(tickers))
    if extra:
        more = corporate.fetch(c, extra, "2015-06-01", str(TODAY), ROOT / "corporate_actions_extra.json")
        for kind, rows in more.items():
            ids = {r["id"] for r in acts.get(kind, [])}
            acts.setdefault(kind, []).extend(r for r in rows if r["id"] not in ids)
        path.write_text(json.dumps(acts))
    say("corporate:", {k: len(v) for k, v in acts.items()})

    # spin-off values need the new company's raw close on the ex-date
    spins = acts.get("spin_offs", [])
    px: dict[tuple[str, str], float] = {}
    for r in spins:
        sym, d = r.get("new_symbol"), r.get("ex_date")
        if not sym or not d:
            continue
        end = min(pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=7), END_TS)
        b = c.bars([sym], "1Day", d, end, adjustment="raw", asof=d).get(sym)
        if b is not None and len(b):
            px[(sym, d)] = float(b["close"].iloc[0])

    def price_of(sym: str | None, d: pd.Timestamp) -> float | None:
        return px.get((sym, d.strftime("%Y-%m-%d"))) if sym else None

    adj, invalid, estimated, mismatch = {}, [], [], []
    for k in keys:
        src = bs.read("daily", k, segment=None)
        if src.empty or k.startswith("^"):
            continue
        t, asof = st.parse_key(k)
        periods = corporate.symbol_periods(t, acts, pd.Timestamp(asof) if asof else None)
        fixed, spl, dist = corporate.build(src, periods, acts, price_of)
        invalid += [(k, str(r.ex_date.date()), round(r.ratio, 4), r.observed_ratio)
                    for r in spl.itertuples() if r.state == "not_split"]
        estimated += [(k, str(d.date()), kind, round(a, 2)) for d, a, kind, _ in dist.itertuples(index=False)
                      if "estimate" in kind]
        off = (fixed["close"] * fixed["split_factor"] / fixed["close_raw"] - 1).abs()
        if (off > 0.02).any():
            mismatch.append((k, int((off > 0.02).sum()), str(off.idxmax().date())))
        adj[k] = {"splits": spl.assign(ex_date=spl["ex_date"].astype(str)).to_dict("records"),
                  "distributions": dist.assign(ex_date=dist["ex_date"].astype(str)).to_dict("records"),
                  "cash_merger": corporate.cash_merger_price(periods, acts, src.index.max()),
                  "periods": [(a, str(b.date()) if b.year > 1900 else None,
                               str(c.date()) if c.year < 2200 else None) for a, b, c in periods]}
    (ROOT / "adjustments.json").write_text(json.dumps(adj, default=str))
    rep = {"invalid_splits_undone": invalid, "distributions_estimated_from_gap": estimated,
           "raw_vs_adjusted_mismatch": mismatch,
           "n_splits_checked": sum(len(v["splits"]) for v in adj.values()),
           "n_distributions": sum(len(v["distributions"]) for v in adj.values())}
    (ROOT / "corporate_report.json").write_text(json.dumps(rep, indent=1, default=str))
    say("corporate report:", json.dumps(rep, default=str)[:3000])


def step_vix() -> None:
    vix, source = reference.vix_daily(load_cal())
    bs = st.BarStore()
    bs.write("daily", "^VIX", vix, source=source)
    bs.save_manifest()
    say(f"vix: {len(vix)} sessions from {source}, {vix.index.min().date()} → {vix.index.max().date()}")


def step_report() -> None:
    cal = load_cal()
    m, ivs = load_membership()
    rep = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config_hash": config.config_hash()}
    daily = load_daily([iv.key for iv in ivs] + CFG["universe"]["references"])
    rep["coverage"] = quality.membership_coverage(m, ivs, daily, cal, date.fromisoformat(START))
    rep["daily_quality"] = quality.daily_summary(daily)
    keys = intraday_keys()
    from app.swing_agent.data import MarketData

    md = MarketData(segment=None)
    rep["intraday_quality"] = quality.intraday_summary({k: md.m15(k) for k in keys}, daily, cal)
    rep["survivorship"] = survivorship.measure(m, ivs, daily, cal, CFG, company_map())
    earn_p = ROOT / "events" / "earnings.parquet"
    if earn_p.exists():
        pit = pd.read_parquet(ROOT / "universe_pit.parquet")
        earn = pd.read_parquet(earn_p).rename(columns={"key": "symbol"})
        etfs = set(CFG["universe"]["references"]) | set(CFG["universe"]["context_etfs"])
        rep["earnings_coverage"] = quality.earnings_coverage(earn, pit, None, etfs)
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    rep["universe"] = {"months": int(pit["rebalance"].nunique()), "distinct_keys": int(pit["key"].nunique()),
                       "first": str(pit["rebalance"].min().date()), "last": str(pit["rebalance"].max().date()),
                       "keys": sorted(pit["key"].unique())}
    for name in ("corporate_report.json", "cik_source.json"):
        if (ROOT / name).exists():
            rep[name.removesuffix(".json")] = json.loads((ROOT / name).read_text())
    if "cik_source" in rep:
        src = rep.pop("cik_source")
        rep["cik_sources"] = pd.Series([v.split(":")[0] for v in src.values()]).value_counts().to_dict()
    mp = ROOT / "events" / "macro.parquet"
    if mp.exists():
        mac = pd.read_parquet(mp)
        rep["macro_counts"] = mac.groupby("kind").size().to_dict()
    out = ROOT / "phase1_report.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    say(f"report → {out}")
    print(json.dumps(rep, indent=1, default=str)[:6000])


STEPS = {"calendar": step_calendar, "membership": step_membership, "daily": step_daily,
         "cik": step_cik, "universe": step_universe, "intraday": step_intraday, "earnings": step_earnings,
         "macro": step_macro, "vix": step_vix, "corporate": step_corporate,
         "report": step_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    steps = list(STEPS) if a.steps == ["all"] else a.steps
    for s in steps:
        say(f"== {s}")
        fn = STEPS[s]
        fn(refresh=a.refresh) if s in ("daily", "intraday") else fn()


if __name__ == "__main__":
    main()
