"""Train baseline XGBoost forecasters for a list of symbols and save them.

Usage (from apps/api):
    python ../../scripts/train_baseline.py AAPL MSFT NVDA --horizons 1 5 21
"""
from __future__ import annotations

import argparse
import sys

# Allow running from repo root.
sys.path.insert(0, "apps/api")

from app.ml.xgboost_model import XGBForecaster  # noqa: E402
from app.services import market_data as md  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])
    parser.add_argument("--range", default="3y")
    args = parser.parse_args()

    for symbol in args.symbols:
        df = md.get_history(symbol, interval="1d", range_=args.range)
        if df.empty:
            print(f"[skip] {symbol}: no data")
            continue
        for h in args.horizons:
            model = XGBForecaster()
            try:
                model.fit(df, h)
                path = model.save(symbol, h)
                print(f"[ok]   {symbol} h={h} -> {path}")
            except Exception as e:
                print(f"[fail] {symbol} h={h}: {e}")


if __name__ == "__main__":
    main()
