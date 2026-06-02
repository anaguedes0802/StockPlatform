"""Honest point-in-time backtest of a quality stock-picker vs SPY.

The question: can picking individual stocks by fundamentals + momentum actually
beat just owning the index? We answer it WITHOUT cheating:

  * At each quarterly rebalance date `t`, every stock is scored using ONLY data
    knowable on `t`: point-in-time fundamentals (fundamentals_pit, which keys off
    real EDGAR filing dates), trailing-price momentum, and earnings yield from the
    last *filed* EPS. No future data leaks in.
  * Score = cross-sectional z-sum of value (earnings yield), quality (net margin,
    low debt), growth (YoY revenue/EPS), and 12-1 momentum.
  * Hold the top-N equal-weight until the next rebalance; compound.
  * Compare to SPY buy-and-hold over the identical window.

KNOWN CAVEAT (stated, not hidden): the universe is today's survivors, so results
are survivorship-biased UPWARD. If the picker can't beat SPY even *with* that
tailwind, that's damning; if it wins only slightly, treat it as a wash.

Run:  PYTHONPATH=. .venv/bin/python scripts/eval_quality.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from app.services import fundamentals_pit as fpit
from app.services.investing import _full_history_close

# Liquid large caps across sectors, long-lived (survivorship caveat applies).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "ORCL", "CSCO", "ADBE", "INTC",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
    "JNJ", "UNH", "PFE", "MRK", "ABBV",
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "DIS",
    "XOM", "CVX", "CAT", "BA", "HON", "GE",
]
TOP_N = 10
START = "2013-01-01"


def _zscore(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    sd = v.std(ddof=0)
    return (v - v.mean()) / sd if sd and not np.isnan(sd) else v * 0.0


def main() -> None:
    print(f"Loading prices + PIT fundamentals for {len(UNIVERSE)} names...", file=sys.stderr)
    closes, funds = {}, {}
    for s in UNIVERSE + ["SPY"]:
        px = _full_history_close(s)
        if px is None or px.empty:
            continue
        closes[s] = px
    common = None
    for s, px in closes.items():
        common = px.index if common is None else common.union(px.index)
    cal = pd.DatetimeIndex(sorted(common))
    # PIT fundamentals aligned to the daily calendar (uses available_at internally).
    for s in UNIVERSE:
        if s not in closes:
            continue
        try:
            f = fpit.features_for_index(s, cal)
            if f is not None and not f.empty:
                funds[s] = f
        except Exception as e:  # noqa: BLE001
            print(f"  no fundamentals for {s}: {e}", file=sys.stderr)

    names = [s for s in UNIVERSE if s in closes and s in funds]
    print(f"Usable names: {len(names)}", file=sys.stderr)

    # Quarterly rebalance dates from START.
    px_all = pd.DataFrame({s: closes[s] for s in names + (["SPY"] if "SPY" in closes else [])}).ffill()
    px_all = px_all[px_all.index >= pd.Timestamp(START, tz="UTC")]
    rebal = px_all.groupby([px_all.index.year, px_all.index.quarter]).head(1).index

    def score_at(t: pd.Timestamp) -> dict[str, float]:
        rows = {}
        for s in names:
            f = funds[s]
            fa = f[f.index <= t]
            if fa.empty:
                continue
            r = fa.iloc[-1]
            px = closes[s]
            p_now = float(px[px.index <= t].iloc[-1]) if (px.index <= t).any() else np.nan
            hist = px[px.index <= t]
            if len(hist) < 252 or np.isnan(p_now) or p_now <= 0:
                continue
            mom = p_now / float(hist.iloc[-252]) - 1 - (p_now / float(hist.iloc[-21]) - 1)  # 12-1
            eps_ttm = r.get("eps_ttm", np.nan)
            rows[s] = {
                "ey": (eps_ttm / p_now) if pd.notna(eps_ttm) else np.nan,   # earnings yield
                "margin": r.get("net_margin", np.nan),
                "rev_g": r.get("revenue_growth_yoy", np.nan),
                "eps_g": r.get("eps_growth_yoy", np.nan),
                "low_debt": -r.get("debt_to_assets", np.nan) if pd.notna(r.get("debt_to_assets", np.nan)) else np.nan,
                "mom": mom,
            }
        if len(rows) < TOP_N + 2:
            return {}
        df = pd.DataFrame(rows).T
        z = pd.DataFrame(index=df.index)
        for col in ["ey", "margin", "rev_g", "eps_g", "low_debt", "mom"]:
            z[col] = _zscore(df[col].fillna(df[col].median()))
        composite = z.sum(axis=1)
        return composite.sort_values(ascending=False).head(TOP_N).to_dict()

    # Walk the rebalances, equal-weight the picks, compound.
    eq, spy_eq = 1.0, 1.0
    eq_curve, spy_curve = [], []
    picks_log = []
    rebal = [t for t in rebal if (px_all.index <= t).any()]
    for i, t in enumerate(rebal):
        picks = score_at(t)
        if not picks:
            continue
        nxt = rebal[i + 1] if i + 1 < len(rebal) else px_all.index[-1]
        held = list(picks.keys())
        seg = px_all.loc[(px_all.index >= t) & (px_all.index <= nxt)]
        if len(seg) < 2:
            continue
        # equal-weight return of the picks over the segment
        rets = (seg[held].iloc[-1] / seg[held].iloc[0] - 1).mean()
        eq *= (1 + rets)
        if "SPY" in seg:
            spy_eq *= (seg["SPY"].iloc[-1] / seg["SPY"].iloc[0])
        eq_curve.append((t, eq)); spy_curve.append((t, spy_eq))
        picks_log.append((t.date().isoformat(), held))

    if not eq_curve:
        print("no results"); return
    yrs = (eq_curve[-1][0] - eq_curve[0][0]).days / 365.25
    def cagr(x): return (x ** (1 / yrs) - 1) * 100
    def mdd(curve):
        s = pd.Series([v for _, v in curve]); peak = s.cummax()
        return float(((s - peak) / peak).min() * 100)

    print(f"\nPIT quality picker — top {TOP_N}, quarterly, {eq_curve[0][0].date()} → {eq_curve[-1][0].date()} ({yrs:.1f}y)")
    print(f"  Picker:  {(eq-1)*100:>7.0f}% total   {cagr(eq):>5.1f}%/yr   maxDD {mdd(eq_curve):>6.1f}%")
    print(f"  SPY:     {(spy_eq-1)*100:>7.0f}% total   {cagr(spy_eq):>5.1f}%/yr   maxDD {mdd(spy_curve):>6.1f}%")
    print(f"  Edge:    {cagr(eq)-cagr(spy_eq):+.1f}%/yr")
    print("\n  Sample recent picks:")
    for d, held in picks_log[-3:]:
        print(f"    {d}: {', '.join(held)}")


if __name__ == "__main__":
    main()
