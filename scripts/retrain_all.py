"""Retrain XGBoost forecasters for a list of symbols and horizons.

Designed to run on a schedule (cron, systemd timer, Celery beat, GitHub Actions).

Recommended cadence:
  - Daily (after US close, ~22:00 UTC) for frequently-traded symbols and short horizons.
  - Weekly (Saturday morning UTC) for the full universe.

Usage:
    # Default: top 30 symbols, horizons 1/5/21 days
    python scripts/retrain_all.py

    # Custom
    python scripts/retrain_all.py --symbols AAPL MSFT NVDA --horizons 1 5 21 63 --range 5y
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "apps/api")

from app.ml.xgboost_model import XGBForecaster  # noqa: E402
from app.services import market_data as md  # noqa: E402

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
    "AVGO", "ORCL", "CRM", "ADBE", "NFLX",
    "JPM", "V", "MA", "BAC", "GS",
    "JNJ", "LLY", "UNH", "PFE",
    "XOM", "CVX",
    "WMT", "COST", "HD",
    "SPY", "QQQ", "DIA",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])
    parser.add_argument("--range", default="3y", dest="range_")
    args = parser.parse_args()

    start = datetime.now(timezone.utc)
    print(f"[{start.isoformat()}] retrain start: {len(args.symbols)} symbols × {len(args.horizons)} horizons")

    ok = fail = 0
    for symbol in args.symbols:
        df = md.get_history(symbol, interval="1d", range_=args.range_)
        if df.empty:
            print(f"  [skip] {symbol}: no data")
            fail += 1
            continue
        for h in args.horizons:
            t0 = time.time()
            try:
                model = XGBForecaster()
                model.fit(df, h)
                path = model.save(symbol, h)
                ok += 1
                print(f"  [ok]   {symbol} h={h:>2}d  rows={model.bundle.n_train_rows}  → {path}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                fail += 1
                print(f"  [fail] {symbol} h={h:>2}d: {e}")

    dt = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"done: ok={ok} fail={fail} elapsed={dt:.0f}s")


if __name__ == "__main__":
    main()
