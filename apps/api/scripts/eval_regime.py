"""Test a market-regime filter (economically motivated, not curve-fit).

Hypothesis: most of the strategy's big losers happen when the *whole market* is
falling — an oversold dip in a downtrend often keeps falling. Gating entries on
"SPY above its own 200-day SMA" should cut those tail losses and lift the profit
factor, at the cost of fewer trades. We judge it purely out-of-sample.

Run from apps/api:  PYTHONPATH=. .venv/bin/python scripts/eval_regime.py
"""
from __future__ import annotations

import sys

import pandas as pd

from app.backtest.bot_engine import simulate
from app.services import indicators as ind
from app.services import market_data as md
from app.services import trading_bot as tb
from scripts.eval_strategy import (COMMISSION_BPS, SLIP, TRAIN_END, WIDE_UNIVERSE,
                                    _fetch, _row, _stats)


def _spy_uptrend(range_="10y") -> pd.Series:
    spy = md.get_history("SPY", interval="1d", range_=range_)
    return (spy["close"] > ind.sma(spy["close"], 200)).fillna(False)


def _pool(frames, dsl, regime: pd.Series | None, *, lo=None, hi=None) -> dict:
    all_trades = []
    for sym, df in frames.items():
        d = df
        if lo is not None:
            d = d[d.index >= lo]
        if hi is not None:
            d = d[d.index < hi]
        if len(d) < int(dsl.get("trend_sma", 200)) + 10:
            continue
        entries, rules = tb._build_signals(d, dsl)
        if regime is not None:
            mask = regime.reindex(d.index).ffill().fillna(False)
            entries = entries & mask
        rep = simulate(d, entries, rules, symbol=sym, slippage_bps=SLIP,
                       commission_bps=COMMISSION_BPS)
        all_trades.extend(rep.trades)
    return _stats(all_trades)


def main() -> None:
    print(f"Fetching {len(WIDE_UNIVERSE)} symbols...", file=sys.stderr)
    frames = _fetch(WIDE_UNIVERSE)
    regime = _spy_uptrend()
    base = dict(tb.DEFAULT_STRATEGY)
    print(f"Got {len(frames)} symbols. OOS test window: > {TRAIN_END}\n")

    print("=== Effect of the SPY>SMA200 market-regime filter (net of costs) ===\n")
    for label, lo, hi in [("full 10y", None, None),
                          ("TEST (out-of-sample)", TRAIN_END, None)]:
        no = _pool(frames, base, None, lo=lo, hi=hi)
        yes = _pool(frames, base, regime, lo=lo, hi=hi)
        print(f"[{label}]")
        print(_row("  baseline (no filter)", no))
        print(_row("  + regime filter", yes))
        print()


if __name__ == "__main__":
    main()
