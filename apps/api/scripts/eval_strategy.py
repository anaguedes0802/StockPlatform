"""Offline strategy evaluation: wide universe, costs, and out-of-sample search.

Run from apps/api:  .venv/bin/python scripts/eval_strategy.py

What it does, honestly:
  1. Fetches a broad, *deliberately less survivor-biased* universe (sector ETFs,
     cyclicals, value names, laggards — not just mega-cap tech winners).
  2. Splits each symbol's history into an in-sample (train) window and a later
     out-of-sample (test) window by date — no peeking.
  3. Reports the current DEFAULT_STRATEGY net of slippage + commission.
  4. Grid-searches a few key parameters on TRAIN only, picks the best by train
     expectancy (subject to a win-rate floor + trade-count floor), then reports
     that config's TEST performance. The test number is the only one that counts.
"""
from __future__ import annotations

import sys
from itertools import product

import pandas as pd

from app.backtest.bot_engine import simulate
from app.services import market_data as md
from app.services import trading_bot as tb

# A broader, more honest universe. Includes laggards/value/cyclicals + sector
# ETFs so the read isn't dominated by the decade's biggest winners.
WIDE_UNIVERSE = [
    # broad market
    "SPY", "QQQ", "IWM", "DIA",
    # sectors
    "XLF", "XLE", "XLV", "XLI", "XLK", "XLP", "XLY", "XLU", "XLB",
    # mega-cap tech (the original survivor set)
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META",
    # value / cyclicals / laggards (reduce survivorship overstatement)
    "JPM", "BAC", "WFC", "XOM", "CVX", "KO", "PG", "JNJ", "PFE",
    "WMT", "HD", "DIS", "BA", "GE", "INTC", "CSCO", "T", "VZ", "F", "MCD",
]

SLIP = tb.DEFAULT_SLIPPAGE_BPS  # 5 bps/side
COMMISSION_BPS = 1.0            # 1 bp round-trip (matches simulate default)
TRAIN_END = "2022-01-01"       # train: <= this; test: > this  (true OOS)
RANGE = "10y"


def _fetch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = md.get_history(s, interval="1d", range_=RANGE)
            if not df.empty and len(df) > 260:
                out[s] = df
            else:
                print(f"  skip {s}: insufficient data ({len(df)} bars)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {s}: {e}", file=sys.stderr)
    return out


def _pool(frames: dict[str, pd.DataFrame], dsl: dict, *, lo=None, hi=None) -> dict:
    """Backtest dsl across all frames on the [lo, hi) date slice; pool trades."""
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
        rep = simulate(d, entries, rules, symbol=sym, slippage_bps=SLIP,
                       commission_bps=COMMISSION_BPS)
        all_trades.extend(rep.trades)
    return _stats(all_trades)


def _stats(trades) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0, "win": 0.0, "pf": None, "exp": 0.0, "avg_w": 0.0, "avg_l": 0.0}
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    gw = sum(t.return_pct for t in wins)
    gl = abs(sum(t.return_pct for t in losses))
    return {
        "n": n,
        "win": round(len(wins) / n * 100, 1),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "exp": round(sum(t.return_pct for t in trades) / n, 4),
        "avg_w": round(gw / len(wins), 3) if wins else 0.0,
        "avg_l": round(-gl / len(losses), 3) if losses else 0.0,
    }


def _row(label: str, s: dict) -> str:
    return (f"{label:<34} n={s['n']:>4}  win={s['win']:>5}%  pf={str(s['pf']):>5}  "
            f"exp={s['exp']:>8}%  avgW={s['avg_w']:>6}  avgL={s['avg_l']:>7}")


def main() -> None:
    print(f"Fetching {len(WIDE_UNIVERSE)} symbols ({RANGE})...", file=sys.stderr)
    frames = _fetch(WIDE_UNIVERSE)
    print(f"Got {len(frames)} usable symbols. Train <= {TRAIN_END} < Test\n")

    base = dict(tb.DEFAULT_STRATEGY)
    print("=== BASELINE (current DEFAULT_STRATEGY), net of slippage+commission ===")
    print(_row("  full 10y", _pool(frames, base)))
    print(_row("  train (in-sample)", _pool(frames, base, hi=TRAIN_END)))
    print(_row("  TEST (out-of-sample)", _pool(frames, base, lo=TRAIN_END)))
    print()

    # ---- Grid search on TRAIN only ----
    grid = {
        "rsi_entry":      [5.0, 10.0, 15.0],
        "stop_loss_pct":  [0.05, 0.08, 0.10, 0.25],
        "take_profit_pct":[0.01, 0.02, 0.03],
        "exit_mode":      ["sma", "first_up"],
        "exit_sma":       [5, 10, 15],
    }
    keys = list(grid)
    combos = list(product(*[grid[k] for k in keys]))
    print(f"=== GRID SEARCH on TRAIN ({len(combos)} configs) ===", file=sys.stderr)

    WIN_FLOOR = 78.0   # keep the high-hit-rate character
    N_FLOOR = 150      # enough train trades to trust the number
    scored = []
    for combo in combos:
        dsl = dict(base)
        dsl.update(dict(zip(keys, combo)))
        # exit_sma only matters when exit_mode == "sma"; dedup the rest
        if dsl["exit_mode"] != "sma" and dsl["exit_sma"] != grid["exit_sma"][0]:
            continue
        tr = _pool(frames, dsl, hi=TRAIN_END)
        if tr["n"] < N_FLOOR or tr["win"] < WIN_FLOOR or tr["pf"] is None:
            continue
        scored.append((tr["exp"], dsl, tr))

    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"\n{len(scored)} configs cleared the floors (win>={WIN_FLOOR}%, n>={N_FLOOR}).")
    print("\nTop 5 by TRAIN expectancy — with their TEST (OOS) result:\n")
    for exp, dsl, tr in scored[:5]:
        te = _pool(frames, dsl, lo=TRAIN_END)
        tag = (f"rsi<{dsl['rsi_entry']:g} stop{dsl['stop_loss_pct']:g} "
               f"tp{dsl['take_profit_pct']:g} {dsl['exit_mode']}"
               + (f"{dsl['exit_sma']}" if dsl['exit_mode'] == 'sma' else ""))
        print(_row(f"TRAIN {tag}", tr))
        print(_row(f"  -> TEST", te))
        print()

    if scored:
        _, best_dsl, _ = scored[0]
        print("=== Best-by-train config (full 10y, net) ===")
        print(_row("  full 10y", _pool(frames, best_dsl)))
        import json
        tuned = {k: best_dsl[k] for k in ("rsi_entry", "stop_loss_pct",
                 "take_profit_pct", "exit_mode", "exit_sma")}
        print("\nTuned params:", json.dumps(tuned))


if __name__ == "__main__":
    main()
