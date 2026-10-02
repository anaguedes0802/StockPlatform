"""Corporate actions: validated splits, cash dividends, spin-offs, cash mergers.

Alpaca's adjusted bars are not taken on trust. Two errors found in phase 1:

* WestRock's 2016-05-16 Ingevity spin-off (1 NGVT per 6 WRK) is recorded as a
  1→0.1666 *reverse split*, so split-adjusted WRK prices before that day are
  6× too high and show a fake −84% day.
* Corteva's 2026-10-01 spin-off is not adjusted at all (77.65 → 12.57).

Alpaca's corporate-action list records many spin-offs as reverse splits
(UTX→Carrier/Otis, EQT→Equitrans, TGNA→Cars.com, WRK→Ingevity) and, through
reused tickers, attaches some actions to the wrong company. So the list is
not used to decide what was applied. Instead:

* The splits Alpaca **applied** are read from the data: raw close / adjusted
  close is a step function that jumps on each applied split.
* An applied split is **valid** only if the raw price actually moved by its
  ratio across the ex-date — log(prev raw close / raw close) closer to
  log(ratio) than to 0. A step under 10% is never a split (the smallest
  common one is 5:4); Alpaca uses such steps for spin-offs. Every non-split
  step is undone, so execution prices show the real ex-date drop and the
  distribution is credited separately.
* Cash dividends (regular and special) and spin-off values (ratio × the new
  share's ex-date close) are *distributions*: the backtest credits them as
  cash to held positions, and `tr_factor` adjusts prices for signals so
  indicators never see a distribution as a crash. It is anchored forward
  (1 on the first day) so no future distribution leaks into a past level.
* Cash mergers give the exact take-out price for a delisted stock.

Dividend / spin-off / merger records are keyed by the symbol in force on the
action date. An entity's symbol history (FB→META, WRK→SW) comes from the
name-change records, each symbol only for the period the entity used it.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TYPES = "cash_dividend,forward_split,reverse_split,spin_off,cash_merger,stock_merger,name_change"


def fetch(client: Any, symbols: list[str], start: str, end: str, path: Path,
          batch: int = 100) -> dict[str, list[dict[str, Any]]]:
    """All action types for `symbols`, merged across batches, cached as JSON."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    syms = sorted(set(symbols))
    for i in range(0, len(syms), batch):
        for kind, rows in client.corporate_actions(syms[i:i + batch], start, end, types=TYPES).items():
            out[kind].extend(rows)
    dedup = {k: list({r["id"]: r for r in v}.values()) for k, v in out.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dedup))
    return dedup


def load(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text())


def symbol_periods(ticker: str, actions: dict[str, list[dict[str, Any]]],
                   anchor: pd.Timestamp | None = None) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(symbol, first day, last day) the entity that carried `ticker` on
    `anchor` (default: today) used each symbol, following name changes
    backwards and forwards from that date. A symbol reused by another company
    before or after (PARA went to a different issuer on 2026-08-07) is not
    attached to this one."""
    anchor = anchor or pd.Timestamp.today().normalize()
    ren = sorted(((pd.Timestamp(r["process_date"]), r["old_symbol"], r["new_symbol"])
                  for r in actions.get("name_changes", [])
                  if r.get("old_symbol") and r.get("new_symbol") and r.get("process_date")),
                 key=lambda x: x[0])
    lo, hi = pd.Timestamp.min, pd.Timestamp.max
    # the ticker's own period around the anchor
    into = [d for d, _, new in ren if new == ticker and d <= anchor]
    out_ = [d for d, old, _ in ren if old == ticker and d > anchor]
    reuse = [d for d, _, new in ren if new == ticker and d > anchor]  # someone else takes it
    t_lo = max(into) if into else lo
    t_hi = min(out_ + reuse) - pd.Timedelta(days=1) if (out_ or reuse) else hi
    periods = [(ticker, t_lo, t_hi)]
    cur, cur_lo = ticker, t_lo                 # backwards: older names
    while cur_lo != lo:
        prev = [(d, old) for d, old, new in ren if new == cur and d == cur_lo]
        if not prev:
            break
        d, old = prev[0]
        older = [x for x, _, new in ren if new == old and x < d]
        o_lo = max(older) if older else lo
        periods.append((old, o_lo, d - pd.Timedelta(days=1)))
        cur, cur_lo = old, o_lo
    cur, cur_hi = ticker, t_hi                 # forwards: later names
    while cur_hi != hi:
        nxt = [(d, new) for d, old, new in ren if old == cur and d == cur_hi + pd.Timedelta(days=1)]
        if not nxt:
            break
        d, new = nxt[0]
        later = [x for x, old, _ in ren if old == new and x > d]
        n_hi = min(later) - pd.Timedelta(days=1) if later else hi
        periods.append((new, d, n_hi))
        cur, cur_hi = new, n_hi
    return sorted(set(periods), key=lambda p: p[1])


def _in_periods(sym: str | None, d: pd.Timestamp, periods: list[tuple]) -> bool:
    return any(sym == s and lo <= d <= hi for s, lo, hi in periods)


def snap_ratio(r: float, tol: float = 0.005) -> float:
    """Nearest small fraction n/d (n, d ≤ 50) within 0.5%: 3.99996 → 4, 0.6667 → 2/3.
    Ratios read from rounded adjusted prices are never exact."""
    best, err = r, tol
    for d in range(1, 51):
        n = round(r * d)
        if 1 <= n <= 50 * d:
            e = abs(n / d / r - 1)
            if e < err - 1e-12:
                best, err = n / d, e
    return best


def applied_splits(src: pd.DataFrame, min_step: float = 0.02) -> pd.DataFrame:
    """Steps Alpaca applied, read from the step function raw/adjusted close.

    Adjusted prices are rounded, so the ratio is noisy (≈1% for a stock whose
    old adjusted price is under a dollar). A step counts when the day's jump
    exceeds 1% and the 10-day medians on either side differ by more than
    `min_step` in log terms.
    """
    f = np.log((src["close_raw"] / src["close"]).replace([np.inf, -np.inf], np.nan)).dropna()
    rows = []
    if len(f) >= 3:
        for i in np.flatnonzero(f.diff().abs().to_numpy() > 0.01):
            before = f.iloc[max(0, i - 10):i].median()
            after = f.iloc[i:i + 10].median()
            if abs(before - after) > min_step:
                rows.append((f.index[i], float(np.exp(before - after))))
    df = pd.DataFrame(rows, columns=["ex_date", "ratio"])
    if len(df) > 1:  # a noisy stretch can flag neighbours of one step: keep the biggest
        grp = (df["ex_date"].diff() > pd.Timedelta(days=7)).cumsum()
        df = df.loc[df.assign(m=np.log(df["ratio"]).abs()).groupby(grp)["m"].idxmax()]
    return df.reset_index(drop=True)


def validate_splits(splits: pd.DataFrame, close_raw: pd.Series) -> pd.DataFrame:
    """Add `state` to each applied step, from the raw price across the
    ex-date (first bar on/after it vs the bar before):

    * "split" — the raw price moved by the ratio: keep;
    * "not_split" — it didn't, or the step is under 10%: undo;
    * "unchecked" — ex-date outside the data: nothing contradicts it, keep.
    """
    s = splits.copy()
    state, obs = [], []
    idx = close_raw.index
    for d, r in s[["ex_date", "ratio"]].itertuples(index=False):
        i = idx.searchsorted(d)
        if i <= 0 or i >= len(idx):
            state.append("unchecked")
            obs.append(None)
            continue
        o = float(close_raw.iloc[i - 1] / close_raw.iloc[i])
        lo = math.log(o)
        ok = abs(math.log(r)) >= 0.1 and abs(lo - math.log(r)) < abs(lo)
        state.append("split" if ok else "not_split")
        obs.append(round(o, 4))
    s["observed_ratio"] = obs
    s["state"] = state
    # a real split has an exact ratio (4:1, 3:2); a non-split step is undone by
    # exactly what was applied, so only real ones are snapped
    if len(s):
        s["ratio"] = [snap_ratio(r) if st_ == "split" else r for r, st_ in zip(s["ratio"], s["state"])]
    return s


def factor_after(dates: pd.DatetimeIndex, splits: pd.DataFrame) -> np.ndarray:
    """Π ratio of the given splits whose ex-date is after each date."""
    f = np.ones(len(dates))
    for d, r in splits[["ex_date", "ratio"]].itertuples(index=False):
        f[np.asarray(dates < d)] *= r
    return f


def _invalid(splits: pd.DataFrame) -> pd.DataFrame:
    return splits[splits["state"] == "not_split"] if len(splits) else splits


def _kept(splits: pd.DataFrame) -> pd.DataFrame:
    return splits[splits["state"] != "not_split"] if len(splits) else splits


def distributions(periods: list[tuple], actions: dict[str, list[dict[str, Any]]],
                  price_of: Any) -> pd.DataFrame:
    """Cash dividends + spin-off values per raw share, by ex-date.

    `price_of(symbol, date)` returns the raw close of a spun-off company on
    its ex-date (None when unknown, in which case the spin-off is flagged and
    valued at 0 — the parent's drop then stays in the price, conservatively)."""
    rows = []
    for r in actions.get("cash_dividends", []):
        if r.get("ex_date") and r.get("rate") and \
                _in_periods(r.get("symbol"), pd.Timestamp(r["ex_date"]), periods):
            rows.append((pd.Timestamp(r["ex_date"]), float(r["rate"]),
                         "special" if r.get("special") else "dividend", None))
    for r in actions.get("spin_offs", []):
        if r.get("ex_date") and _in_periods(r.get("source_symbol"), pd.Timestamp(r["ex_date"]), periods):
            ratio = float(r.get("new_rate") or 0) / float(r.get("source_rate") or 1)
            px = price_of(r.get("new_symbol"), pd.Timestamp(r["ex_date"]))
            rows.append((pd.Timestamp(r["ex_date"]), ratio * px if px else 0.0, "spin_off",
                         r.get("new_symbol")))
    df = pd.DataFrame(rows, columns=["ex_date", "amount_raw", "kind", "new_symbol"])
    return df.sort_values("ex_date", ignore_index=True)


def cash_merger_price(periods: list[tuple], actions: dict[str, list[dict[str, Any]]],
                      last_day: pd.Timestamp) -> float | None:
    """Cash take-out price if a cash merger ended the series (within 10 days)."""
    for r in actions.get("cash_mergers", []):
        d = pd.Timestamp(r.get("effective_date") or r.get("process_date") or "1900-01-01")
        if r.get("rate") and abs((d - last_day).days) <= 10 and \
                _in_periods(r.get("acquiree_symbol"), last_day, periods):
            return float(r["rate"])
    return None


def gap_value(src: pd.DataFrame, d: pd.Timestamp) -> float | None:
    """Raw value distributed on ex-date d, inferred from the overnight gap
    (previous raw close − raw open). Used for spin-offs Alpaca misrecorded as
    reverse splits or whose new shares have no price. It also absorbs that
    night's market move, so it is an estimate, flagged as such."""
    idx = src.index
    i = idx.searchsorted(d)
    if not 0 < i < len(idx):
        return None
    raw_open = src["open"].iloc[i] * src["close_raw"].iloc[i] / src["close"].iloc[i]
    v = float(src["close_raw"].iloc[i - 1] - raw_open)
    return v if v > 0 else None


def build(src: pd.DataFrame, periods: list[tuple], actions: dict[str, list[dict[str, Any]]],
          price_of: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(adjusted daily, splits with validity, distributions) for one key."""
    spl = validate_splits(applied_splits(src), src["close_raw"])
    if spl.empty:
        spl = pd.DataFrame(columns=["ex_date", "ratio", "observed_ratio", "state"])
    lo, hi = src.index.min(), src.index.max()
    dist = distributions(periods, actions, price_of)
    dist = dist[(dist["ex_date"] > lo) & (dist["ex_date"] <= hi)].copy()
    rows = []
    for d, a, kind, new in dist.itertuples(index=False):
        if kind == "spin_off" and not a:
            a, kind = gap_value(src, d) or 0.0, "spin_off_gap_estimate"
        rows.append((d, a, kind, new))
    have = [r[0] for r in rows if r[2] in ("spin_off", "special", "spin_off_gap_estimate")]
    for r in spl.itertuples():
        near = any(abs((r.ex_date - d).days) <= 3 for d in have)
        if r.state == "not_split" and (r.observed_ratio or 0) > 1.02 and not near:
            rows.append((r.ex_date, gap_value(src, r.ex_date) or 0.0, "misrecorded_split_gap_estimate", None))
    dist = pd.DataFrame(rows, columns=["ex_date", "amount_raw", "kind", "new_symbol"]) \
        .sort_values("ex_date", ignore_index=True)
    return adjust_daily(src, spl, dist), spl, dist


def adjust_daily(src: pd.DataFrame, splits: pd.DataFrame, dists: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a daily frame from Alpaca's split-adjusted source.

    Alpaca divided prices by every recorded split; an invalid split is undone
    by multiplying back its ratio before its ex-date. split_factor, div_ret,
    tr_close and tr_factor are then recomputed from the kept splits and the
    distributions.
    """
    df = src.copy()
    idx = pd.DatetimeIndex(df.index)
    fix = factor_after(idx, _invalid(splits))
    for c in ("open", "high", "low", "close", "vwap"):
        df[c] = df[c] * fix
    df["volume"] = df["volume"] / fix
    df["split_factor"] = factor_after(idx, _kept(splits))
    df["split_fix"] = fix
    amt = pd.Series(0.0, index=idx)
    for d, a in dists[["ex_date", "amount_raw"]].itertuples(index=False):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            amt.iloc[i] += a
    amt_adj = amt / df["split_factor"]
    prev = df["close"].shift(1)
    df["div_ret"] = (amt_adj / prev).fillna(0.0)
    gross = ((df["close"] + amt_adj) / prev).fillna(1.0)
    df["tr_close"] = gross.cumprod() * (df["close"].iloc[0] if len(df) else 1.0)
    # signal prices = price × tr_factor. Forward-anchored (1 on the first
    # day), so the level on day t only depends on distributions up to t.
    trf = df["tr_close"] / df["close"]
    df["tr_factor"] = trf / trf.iloc[0] if len(df) else trf
    return df


def intraday_fix(sessions: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """Per-row price multiplier for 15m bars (same rule as adjust_daily)."""
    return factor_after(pd.DatetimeIndex(sessions), _invalid(splits))
