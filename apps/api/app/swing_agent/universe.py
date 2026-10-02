"""Point-in-time universe: S&P 500 members on each date, ranked by liquidity.

Membership comes from the free fja05680/sp500 dataset (one row per change
date, tickers as they were on that date, 1996 → present). Tickers that later
changed or disappeared keep their old names there, e.g. FB until 2022-06,
TWTR until 2022-11, SIVB until 2023-03.

Each membership interval maps to an Alpaca request key:

* still a member today → plain ticker, today's symbol mapping;
* gone → `TICKER@<last member date>`, so Alpaca's `asof` resolves the
  entity that carried the ticker then (FB@2022-06-08 is Meta's series, a
  reused ticker would not be).

Monthly rule (fixed before any test, see swing_agent.toml):
on the first session of each month, take that day's members, drop those with
a raw close below `min_price` or too few bars, rank by median dollar volume
over the `lookback` sessions that ended BEFORE that day, keep `top_n`,
skipping a second share class of a company already kept (same SEC CIK:
GOOG vs GOOGL, FOX vs FOXA), or a name whose daily returns over the same
window correlate above `same_company_corr` with one already kept. Two share
classes are one bet, not two. Correlation alone is not enough: the GOOG/GOOGL
spread collapsed on 2021-07-28, which drags their 63-day correlation to 0.90.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from app.swing_agent.calendar import TradingCalendar
from app.swing_agent.store import store_key


@dataclass(frozen=True)
class Interval:
    ticker: str
    start: date
    end: date | None  # last date the ticker was a member; None = still a member

    @property
    def key(self) -> str:
        return store_key(self.ticker, self.end.isoformat() if self.end else None)


class Membership:
    def __init__(self, rows: list[tuple[date, frozenset[str]]]) -> None:
        self.rows = sorted(rows)
        self._dates = [d for d, _ in self.rows]

    @classmethod
    def from_csv_text(cls, text: str) -> Membership:
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            tickers = frozenset(t.strip() for t in r["tickers"].split(",") if t.strip())
            rows.append((date.fromisoformat(r["date"]), tickers))
        return cls(rows)

    @classmethod
    def load(cls, path: Path) -> Membership:
        return cls.from_csv_text(path.read_text())

    @staticmethod
    def download(url: str, path: Path) -> Path:
        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return path

    @property
    def last_update(self) -> date:
        return self._dates[-1]

    @property
    def current(self) -> frozenset[str]:
        return self.rows[-1][1]

    def members_on(self, d: date) -> frozenset[str]:
        """Members effective on date d (the latest change row dated ≤ d)."""
        import bisect

        i = bisect.bisect_right(self._dates, d) - 1
        if i < 0:
            raise KeyError(f"membership starts {self._dates[0]}, asked {d}")
        return self.rows[i][1]

    def intervals(self, since: date) -> list[Interval]:
        """Every continuous membership stretch that overlaps [since, today]."""
        open_: dict[str, date] = {}
        out: list[Interval] = []
        prev: frozenset[str] = frozenset()
        prev_day: date | None = None
        for d, members in self.rows:
            for t in members - prev:
                open_[t] = d
            for t in prev - members:
                assert prev_day is not None
                # last member day = the day before this change row took effect
                end = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
                if end >= since:
                    out.append(Interval(t, open_.pop(t), end))
                else:
                    open_.pop(t, None)
            prev, prev_day = members, d
        out.extend(Interval(t, s, None) for t, s in open_.items())
        return sorted(out, key=lambda iv: (iv.ticker, iv.start))

    @staticmethod
    def key_on(by_ticker: dict[str, list[Interval]], ticker: str, d: date) -> str | None:
        """Store key of `ticker`'s membership interval that contains d."""
        for iv in by_ticker.get(ticker, ()):
            if iv.start <= d and (iv.end is None or d <= iv.end):
                return iv.key
        return None


def by_ticker(intervals: list[Interval]) -> dict[str, list[Interval]]:
    out: dict[str, list[Interval]] = {}
    for iv in intervals:
        out.setdefault(iv.ticker, []).append(iv)
    return out


def rank_liquidity(daily: dict[str, pd.DataFrame], keys: list[str], cal: TradingCalendar,
                   on: date, *, lookback: int, min_price: float, min_dollar_volume: float,
                   min_coverage: float) -> pd.DataFrame:
    """Rank candidate keys on rebalance date `on` using only bars before `on`.

    Returns a frame (key, median_dv, last_raw_close, coverage) sorted by
    median_dv desc, already filtered by price / liquidity / coverage.
    """
    window = cal.previous(on, lookback).day, cal.previous(on, 1).day
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    rows = []
    for k in keys:
        df = daily.get(k)
        if df is None or df.empty:
            continue
        w = df.loc[(df.index >= lo) & (df.index <= hi)]
        cov = len(w) / lookback
        if cov < min_coverage:
            continue
        last_px = float(w["close_raw"].iloc[-1])
        mdv = float(np.nanmedian(w["dollar_volume"].to_numpy()))
        if last_px < min_price or not np.isfinite(mdv) or mdv < min_dollar_volume:
            continue
        rows.append((k, mdv, last_px, cov))
    out = pd.DataFrame(rows, columns=["key", "median_dv", "last_raw_close", "coverage"])
    return out.sort_values("median_dv", ascending=False, ignore_index=True)


def dedupe_same_company(ranked: list[str], daily: dict[str, pd.DataFrame], lo: pd.Timestamp,
                        hi: pd.Timestamp, top_n: int, max_corr: float,
                        company_of: dict[str, str] | None = None) -> list[str]:
    """Greedy pass down the liquidity ranking that drops second share classes."""
    kept: list[str] = []
    companies: set[str] = set()
    rets: dict[str, pd.Series] = {}
    for k in ranked:
        co = (company_of or {}).get(k)
        if co and co in companies:
            continue
        c = daily[k]["close"]
        r = c.loc[(c.index >= lo) & (c.index <= hi)].pct_change().dropna()
        if any(r.corr(rets[j]) > max_corr for j in kept):
            continue
        kept.append(k)
        rets[k] = r
        if co:
            companies.add(co)
        if len(kept) == top_n:
            break
    return kept


def build_universe(membership: Membership, intervals: list[Interval],
                   daily: dict[str, pd.DataFrame], cal: TradingCalendar, start: date, end: date,
                   *, top_n: int, lookback: int, min_price: float, min_dollar_volume: float,
                   min_coverage: float, same_company_corr: float = 0.95,
                   company_of: dict[str, str] | None = None,
                   restrict_to: set[str] | None = None) -> pd.DataFrame:
    """Monthly point-in-time top-N list.

    `restrict_to` (a set of keys) builds the counterfactual survivor-only
    universe for the bias measurement; leave None for the real one.
    """
    idx = by_ticker(intervals)
    out = []
    for s in cal.month_starts(start, end):
        members = membership.members_on(s.day)
        keys = [k for t in members if (k := Membership.key_on(idx, t, s.day))]
        if restrict_to is not None:
            keys = [k for k in keys if k in restrict_to]
        ranked = rank_liquidity(daily, keys, cal, s.day, lookback=lookback, min_price=min_price,
                                min_dollar_volume=min_dollar_volume, min_coverage=min_coverage)
        lo, hi = pd.Timestamp(cal.previous(s.day, lookback).day), pd.Timestamp(cal.previous(s.day).day)
        keep = dedupe_same_company(ranked["key"].tolist(), daily, lo, hi, top_n, same_company_corr,
                                   company_of)
        top = ranked.set_index("key").loc[keep].reset_index()
        top = top.assign(rebalance=pd.Timestamp(s.day), rank=lambda x: x.index + 1)
        top["n_members"] = len(members)
        top["n_ranked"] = len(ranked)
        out.append(top)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
