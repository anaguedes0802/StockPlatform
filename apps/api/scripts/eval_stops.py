"""Sweep stop-loss + time-stop to shrink the average loss (out-of-sample).

The strategy's weak point is asymmetry: small wins, larger losses. Losers
rarely hit the wide 25% stop — they bleed and exit on a weak bounce or time out.
So the real levers on the loss are (a) a tighter hard stop and (b) a shorter
time stop. This sweeps both on the broad, less survivor-biased universe, OUT OF
SAMPLE, and reports the full trade-off: win rate vs avg loss vs expectancy vs
the worst single trade. No curve-fitting claim — we just look at the curve.

Run from apps/api:  PYTHONPATH=. .venv/bin/python scripts/eval_stops.py
"""
from __future__ import annotations

import sys

from app.backtest.bot_engine import simulate
from app.services import trading_bot as tb
from scripts.eval_regime import _pool as _pool_reg, _spy_uptrend
from scripts.eval_strategy import (COMMISSION_BPS, SLIP, TRAIN_END,
                                   WIDE_UNIVERSE, _fetch)


def _pool_full(frames, dsl, regime, *, lo=None, hi=None) -> dict:
    """Like eval_regime._pool but also returns worst single-trade return."""
    import pandas as pd
    trades = []
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
            entries = entries & regime.reindex(d.index).ffill().fillna(False)
        rep = simulate(d, entries, rules, symbol=sym, slippage_bps=SLIP,
                       commission_bps=COMMISSION_BPS)
        trades.extend(rep.trades)
    n = len(trades)
    if not n:
        return {}
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    gw = sum(t.return_pct for t in wins)
    gl = abs(sum(t.return_pct for t in losses))
    worst = min((t.return_pct for t in trades), default=0.0)
    return {
        "n": n,
        "win": round(len(wins) / n * 100, 1),
        "pf": round(gw / gl, 2) if gl else None,
        "exp": round(sum(t.return_pct for t in trades) / n, 4),
        "avg_w": round(gw / len(wins), 3) if wins else 0.0,
        "avg_l": round(-gl / len(losses), 3) if losses else 0.0,
        "worst": round(worst, 1),
    }


def main() -> None:
    print(f"Fetching {len(WIDE_UNIVERSE)} symbols...", file=sys.stderr)
    frames = _fetch(WIDE_UNIVERSE)
    regime = _spy_uptrend()
    print(f"Got {len(frames)} symbols. Window: OOS (> {TRAIN_END}), regime filter ON, net of costs.\n")

    hdr = (f"{'stop':>5} {'hold':>5} | {'n':>4} {'win%':>6} {'PF':>5} "
           f"{'exp%':>8} {'avgW':>6} {'avgL':>7} {'worst%':>7}")
    print(hdr)
    print("-" * len(hdr))
    base = dict(tb.DEFAULT_STRATEGY)
    for stop in [0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.25]:
        for hold in [10, 20, 40]:
            dsl = dict(base); dsl["stop_loss_pct"] = stop; dsl["max_hold_bars"] = hold
            s = _pool_full(frames, dsl, regime, lo=TRAIN_END)
            if not s:
                continue
            star = "  <- current" if (stop == 0.25 and hold == 40) else ""
            print(f"{stop:>5.2f} {hold:>5} | {s['n']:>4} {s['win']:>6} {str(s['pf']):>5} "
                  f"{s['exp']:>8} {s['avg_w']:>6} {s['avg_l']:>7} {s['worst']:>7}{star}")


if __name__ == "__main__":
    main()
