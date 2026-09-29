"""Insider cluster-buy backtest: event study with placebo controls + portfolio.

Usage (repo root; first run downloads ~1 GB of SEC files and a few thousand
price histories, later runs use the caches):
    apps/api/.venv/bin/python scripts/backtest_insider_clusters.py --sync
    apps/api/.venv/bin/python scripts/backtest_insider_clusters.py

Design is fixed in apps/api/app/ml/insider_study.py (read its docstring).
Writes apps/api/artifacts/insider_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("WARMUP_DISABLED", "1")
_API = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "api")
sys.path.insert(0, os.path.abspath(_API))
os.chdir(_API)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.backtest.swing_engine import _curve_metrics  # noqa: E402
from app.ml import insider_study as st  # noqa: E402
from app.services import insider_bulk as ib  # noqa: E402
from app.services import swing_data as sd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="download any missing SEC quarters first")
    ap.add_argument("--placebo-portfolios", type=int, default=20)
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--out", default="artifacts/insider_results.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.sync:
        print(ib.sync(verbose=True))
    tx = ib.load_transactions()
    print(f"insider P/S rows: {len(tx):,}  filings {tx['filing_date'].min().date()} → {tx['filing_date'].max().date()}")
    listings = st.current_listings()
    events = st.detect_clusters(tx)
    print(f"cluster-buy events: {len(events):,}  ({time.time() - t0:.0f}s)")
    ev = events.merge(listings[["issuer_cik", "ticker", "exchange"]], on="issuer_cik", how="inner")
    tickers = sorted(set(ev.loc[ev["exchange"].isin(st.EXCHANGES), "ticker"]))
    print(f"loading prices for {len(tickers):,} listed tickers…", flush=True)
    bars, missing = st.load_bars(tickers, verbose=True)
    spy = sd.daily_bars("SPY", start="2004-06-01")
    print(f"prices: {len(bars):,} ok, {len(missing):,} missing  ({time.time() - t0:.0f}s)")

    out = st.run_study(tx, bars, spy, listings)
    E, P = out.pop("_events"), out.pop("_placebo")

    # ---- calendar-time portfolio vs SPY and vs placebo portfolios ----
    eq = st.simulate_portfolio(E, bars, spy, max_positions=a.max_positions)
    spy_eq = spy["close"].reindex(eq.index).ffill()
    spy_eq = spy_eq / spy_eq.iloc[0] * eq.iloc[0]
    rng = np.random.default_rng(11)
    placebo_metrics = []
    by_tick = {k: g for k, g in P.groupby("ticker")}
    for k in range(a.placebo_portfolios):
        rows = []
        for tick, g in E.groupby("ticker"):
            pool = by_tick.get(tick)
            if pool is None or pool.empty:
                continue
            pick = pool.iloc[rng.integers(0, len(pool), len(g))]
            rows.append(pick.assign(total_value=0.0))
        pe = pd.concat(rows, ignore_index=True)
        m = _curve_metrics(st.simulate_portfolio(pe, bars, spy, max_positions=a.max_positions))
        placebo_metrics.append({"cagr_pct": m.get("cagr_pct"), "sharpe": m.get("sharpe")})
    strat = _curve_metrics(eq)
    bench = _curve_metrics(spy_eq)
    ps = sorted(x["sharpe"] for x in placebo_metrics)
    pc = sorted(x["cagr_pct"] for x in placebo_metrics)
    out["portfolio"] = {
        "strategy": strat, "spy": bench,
        "correlation_to_spy": round(float(eq.pct_change().corr(spy_eq.pct_change())), 3),
        "placebo_sharpe_median": ps[len(ps) // 2] if ps else None,
        "placebo_sharpe_range": [ps[0], ps[-1]] if ps else None,
        "placebo_cagr_median": pc[len(pc) // 2] if pc else None,
        "strategy_beats_placebos": sum(strat["sharpe"] > x for x in ps),
        "n_placebo_portfolios": len(ps),
        "yearly": [{"year": int(y), "strategy_pct": round(float(g.iloc[-1] / g.iloc[0] - 1) * 100, 1)}
                   for y, g in eq.groupby(eq.index.year)],
    }
    out["events_per_year"] = {int(k): int(v) for k, v in E.groupby(E["signal_date"].dt.year).size().items()}
    out["runtime_s"] = round(time.time() - t0)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    c, e, pr, mm = out["counts"], out["event_ar"], out["primary"], out["momentum_matched"]
    print("\ncoverage:", c)
    print(f"event 6-month return vs SPY: {e['mean_pct']}% [{e['lo_pct']}, {e['hi_pct']}]  "
          f"(raw {e['mean_raw_ret_pct']}%, beat SPY {e['win_rate_pct']}% of the time)")
    print(f"same stocks, random dates: {out['placebo_ar_mean_pct']}%   momentum-matched: {out['matched_placebo_ar_mean_pct']}%")
    print(f"PRIMARY  event − placebo: {pr['mean_pct']}% [{pr['lo_pct']}, {pr['hi_pct']}]  n={pr['n']}  → {out['verdict']}")
    print(f"secondary event − momentum-matched: {mm['mean_pct']}% [{mm['lo_pct']}, {mm['hi_pct']}]")
    print("halves:", out["halves"])
    print("by liquidity:", out["by_liquidity"])
    print("by cluster:", out["by_cluster"])
    pf = out["portfolio"]
    print(f"portfolio: CAGR {strat['cagr_pct']}% Sharpe {strat['sharpe']} MDD {strat['max_drawdown_pct']}%  | "
          f"SPY {bench['cagr_pct']}% / {bench['sharpe']} / {bench['max_drawdown_pct']}%  | placebo Sharpe median "
          f"{pf['placebo_sharpe_median']} range {pf['placebo_sharpe_range']} (strategy beats {pf['strategy_beats_placebos']}/"
          f"{pf['n_placebo_portfolios']})")
    print(f"wrote {a.out} in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
