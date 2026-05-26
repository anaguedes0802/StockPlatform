"""Run per-symbol Optuna hyperparameter tuning for the XGBoost forecaster.

Usage:
    python scripts/tune_hyperparams.py AAPL --horizons 5 21
    python scripts/tune_hyperparams.py AAPL NVDA SPY --horizons 5 --trials 30

Best params are saved to apps/api/artifacts/tuning/{symbol}_{horizon}d.json.
XGBForecaster.fit() picks them up automatically on next training.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "apps/api")

from app.ml.tuning import tune  # noqa: E402
from app.services import market_data as md  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="+")
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 21])
    p.add_argument("--range", default="5y", dest="range_")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--val-size", type=int, default=120)
    p.add_argument("--timeout", type=int, default=600, help="seconds per (symbol,horizon)")
    args = p.parse_args()

    for symbol in args.symbols:
        df = md.get_history(symbol, interval="1d", range_=args.range_)
        if df.empty:
            print(f"[skip] {symbol}: no data")
            continue
        for h in args.horizons:
            print(f"\n=== Tuning {symbol} h={h}d  ({args.trials} trials, {args.splits} CV splits) ===")
            try:
                summary = tune(
                    symbol, df, h,
                    n_trials=args.trials,
                    n_splits=args.splits,
                    val_size=args.val_size,
                    timeout_seconds=args.timeout,
                )
                print(json.dumps(
                    {k: v for k, v in summary.items() if k != "symbol"},
                    indent=2,
                ))
            except Exception as e:
                print(f"[fail] {symbol} h={h}: {e}")


if __name__ == "__main__":
    main()
