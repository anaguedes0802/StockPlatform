"""Measure survivorship bias by building the universe three ways on the same data.

* **PIT** — the real rule: point-in-time S&P 500 members, monthly top-N by
  dollar volume, delisted names included.
* **Survivors only** — the same rule, but only entities still in today's
  index are allowed. This is what you get from a data source that only has
  current tickers (Yahoo, most free feeds).
* **Hindsight list** — today's N most liquid members, held through the whole
  window. This is what "pick 40 liquid tickers" means in practice: it adds
  look-ahead selection on top of survivorship (today's most liquid names are
  mostly the ones that went up).

Each is held equal-weight, rebalanced monthly, on total-return closes. The
gaps between them, in percentage points per year, are the bias.

Delisting returns: when an entity's data stops (acquisition, failure), the
default treats the last price as the exit (right for cash takeovers). Names
whose data ends after a collapse (−50% in their last 20 sessions, e.g. SIVB,
FRC) are also run with a −100% delisting return, as a bound.

Only the research window (train + validation) is used, so the final test
stays untouched.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent import config
from app.swing_agent.calendar import TradingCalendar
from app.swing_agent.universe import Interval, Membership, build_universe, by_ticker


def classify(ivs: list[Interval], daily: dict[str, pd.DataFrame], cal: TradingCalendar,
             last_day: date) -> dict[str, str]:
    """key → current | renamed_member | left_index_trading | delisted | delisted_distressed | no_data."""
    return _classify(ivs, daily, cal, last_day)[0]


def renamed_to(ivs: list[Interval], daily: dict[str, pd.DataFrame], cal: TradingCalendar,
               last_day: date) -> dict[str, str]:
    """Gone key → today's key for the same entity (FB@2022-06-08 → META)."""
    return _classify(ivs, daily, cal, last_day)[1]


def _classify(ivs: list[Interval], daily: dict[str, pd.DataFrame], cal: TradingCalendar,
              last_day: date) -> tuple[dict[str, str], dict[str, str]]:
    tail_cut = cal.previous(last_day, 5).day
    fp: dict[tuple, str] = {}
    for iv in ivs:
        df = daily.get(iv.key)
        if iv.end is None and df is not None and len(df) >= 10:
            fp[_fingerprint(df)] = iv.key
    out: dict[str, str] = {}
    ren: dict[str, str] = {}
    for iv in ivs:
        df = daily.get(iv.key)
        if df is None or df.empty:
            out[iv.key] = "no_data"
        elif iv.end is None:
            out[iv.key] = "current"
        elif df.index.max().date() >= tail_cut:
            cur = fp.get(_fingerprint(df))
            out[iv.key] = "renamed_member" if cur else "left_index_trading"
            if cur:
                ren[iv.key] = cur
        else:
            last20 = df["tr_close"].iloc[-20:]
            crash = len(last20) > 1 and last20.iloc[-1] / last20.max() - 1 < -0.5
            out[iv.key] = "delisted_distressed" if crash else "delisted"
    return out, ren


def _fingerprint(df: pd.DataFrame) -> tuple:
    t = df.iloc[-10:]
    return tuple(np.round(t["close"].to_numpy(), 2)) + tuple(np.round(t["volume"].to_numpy(), -2))


def _prices(daily: dict[str, pd.DataFrame], keys: list[str], anchors: list[date],
            zero_after_end: set[str]) -> pd.DataFrame:
    """tr_close at each anchor (last close ≤ anchor). After a key's data ends
    the price is carried (cash at the last price), or set to 0 for keys in
    `zero_after_end` (−100% delisting return)."""
    idx = pd.DatetimeIndex(pd.to_datetime(anchors))
    cols = {}
    for k in keys:
        df = daily.get(k)
        if df is None or df.empty:
            continue
        s = df["tr_close"].dropna()
        p = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        p[idx < s.index.min()] = np.nan
        if k in zero_after_end:
            p[idx > s.index.max()] = 0.0
        cols[k] = p
    return pd.DataFrame(cols, index=idx)


def ew_returns(holdings: dict[pd.Timestamp, list[str]], prices: pd.DataFrame) -> pd.Series:
    """Equal-weight return of each month's holdings, anchor i → i+1."""
    anchors = list(prices.index)
    out = {}
    for i, a in enumerate(anchors[:-1]):
        keys = [k for k in holdings.get(a, []) if k in prices.columns]
        p0, p1 = prices.loc[a, keys], prices.loc[anchors[i + 1], keys]
        ok = p0.notna() & (p0 > 0) & p1.notna()
        if ok.any():
            out[anchors[i + 1]] = float((p1[ok] / p0[ok] - 1).mean())
    return pd.Series(out).sort_index()


def stats(r: pd.Series) -> dict[str, float]:
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    return {"cagr_pct": round(100 * cagr, 2), "vol_pct": round(100 * vol, 2),
            "sharpe_rf0": round(float(r.mean() * 12 / vol), 3) if vol > 0 else None,
            "max_dd_pct": round(100 * float((eq / eq.cummax() - 1).min()), 2), "months": len(r)}


def measure(m: Membership, ivs: list[Interval], daily: dict[str, pd.DataFrame],
            cal: TradingCalendar, cfg: dict[str, Any] | None = None,
            company_of: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = cfg or config.load()
    u = cfg["universe"]
    lo, hi = (x.date() for x in config.segment("research", cfg))
    last_day = max(df.index.max() for df in daily.values()).date()
    cls = classify(ivs, daily, cal, last_day)
    survivors = {k for k, c in cls.items() if c in ("current", "renamed_member")}
    distressed = {k for k, c in cls.items() if c == "delisted_distressed"}
    kw = dict(top_n=u["top_n"], lookback=u["rank_lookback_sessions"], min_price=u["min_price"],
              min_dollar_volume=u["min_median_dollar_volume"], min_coverage=u["min_coverage"],
              same_company_corr=u["same_company_corr"], company_of=company_of)

    reb = [s.day for s in cal.month_starts(lo, hi)]
    anchors = [cal.previous(d).day for d in reb] + [cal.between(lo, hi)[-1].day]
    a_ts = pd.to_datetime(anchors)

    pit = build_universe(m, ivs, daily, cal, lo, hi, **kw)
    surv = build_universe(m, ivs, daily, cal, lo, hi, restrict_to=survivors, **kw)

    # hindsight: today's top-N current members, ranked on the latest 63 sessions
    from app.swing_agent.universe import dedupe_same_company, rank_liquidity

    cur_keys = [iv.key for iv in ivs if iv.end is None]
    on = cal.next(last_day).day
    ranked = rank_liquidity(daily, cur_keys, cal, on, lookback=u["rank_lookback_sessions"],
                            min_price=u["min_price"], min_dollar_volume=u["min_median_dollar_volume"],
                            min_coverage=u["min_coverage"])["key"].tolist()
    today_top = dedupe_same_company(
        ranked, daily, pd.Timestamp(cal.previous(on, u["rank_lookback_sessions"]).day),
        pd.Timestamp(last_day), u["top_n"], u["same_company_corr"], company_of)

    def hold(univ: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
        g = univ.groupby("rebalance")["key"].apply(list)
        return {a_ts[reb.index(d.date())]: v for d, v in g.items()}

    idx = by_ticker(ivs)
    all_members = {a_ts[i]: [k for t in m.members_on(d) if (k := Membership.key_on(idx, t, d))]
                   for i, d in enumerate(reb)}
    all_survivors = {a: [k for k in v if k in survivors] for a, v in all_members.items()}
    keys = sorted({k for v in all_members.values() for k in v} | set(today_top))

    out: dict[str, Any] = {"window": [str(lo), str(hi)], "top_n": u["top_n"],
                           "classification_counts": pd.Series(cls).value_counts().to_dict()}
    for scen, zero in (("last_price", set()), ("distressed_minus100", distressed)):
        px = _prices(daily, keys, anchors, zero)
        series = {
            "pit_topN": ew_returns(hold(pit), px),
            "survivors_topN": ew_returns(hold(surv), px),
            "hindsight_today_topN": ew_returns({a: today_top for a in a_ts}, px),
            "pit_all_members": ew_returns(all_members, px),
            "survivors_all_members": ew_returns(all_survivors, px),
        }
        st = {k: stats(v) for k, v in series.items()}
        base, base_all = st["pit_topN"]["cagr_pct"], st["pit_all_members"]["cagr_pct"]
        st["bias_pp_per_year"] = {
            "survivorship_topN": round(st["survivors_topN"]["cagr_pct"] - base, 2),
            "survivorship_plus_hindsight_topN": round(st["hindsight_today_topN"]["cagr_pct"] - base, 2),
            "survivorship_all_members": round(st["survivors_all_members"]["cagr_pct"] - base_all, 2),
        }
        yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in series.items()})
        st["yearly_pct"] = (100 * yearly).round(2).to_dict("index")
        out[scen] = st

    gone_in_pit = pit[pit["key"].map(lambda k: k not in survivors)]
    out["non_survivor_months_in_pit_topN"] = {
        "share": round(len(gone_in_pit) / len(pit), 4),
        "by_key": gone_in_pit.groupby("key").size().sort_values(ascending=False).head(30).to_dict(),
        "classes": {k: cls.get(k) for k in gone_in_pit["key"].unique()},
    }
    out["hindsight_list"] = today_top
    return out
