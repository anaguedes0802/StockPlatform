"""Single-account profit sim for the trading bot — a real dollar figure.

The per-symbol backtest puts 100% of capital into each trade, which massively
overstates a real account (you can't be all-in on 8 names at once, and a dip-
buyer sits in CASH most of the time). This simulates ONE shared cash account
trading the whole universe:

  * one position per symbol at a time, each sized to equity / N_SLOTS,
  * entries fill on the next bar's open (no look-ahead), exits per the strategy
    rules (stop / take-profit / SMA bounce / time-stop),
  * net of slippage + commission, current DEFAULT_STRATEGY (regime filter on),
  * cash earns nothing while idle.

Reports the compounded equity, money-weighted CAGR, max drawdown, and crucially
the % of days actually invested — the honest context for the return.

Run:  PYTHONPATH=. .venv/bin/python scripts/eval_account.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from app.services import trading_bot as tb
from app.services.market_data import get_history

START_CASH = 10_000.0
SLIP = tb.DEFAULT_SLIPPAGE_BPS / 10_000.0
FEE = 1.0 / 10_000.0


def _load(symbols):
    out = {}
    for s in symbols:
        try:
            df = get_history(s, interval="1d", range_="10y")
            if not df.empty and len(df) > 260:
                out[s] = df
        except Exception as e:  # noqa: BLE001
            print(f"  skip {s}: {e}", file=sys.stderr)
    return out


def simulate_account(symbols, *, n_slots: int) -> dict:
    frames = _load(symbols)
    if not frames:
        raise SystemExit("no data")
    # Precompute entry/exit signals (with the regime filter) per symbol.
    sig = {}
    for s, df in frames.items():
        entries, rules = tb._entries_with_regime(df, tb.DEFAULT_STRATEGY)
        sig[s] = {
            "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
            "entry": entries.reindex(df.index).fillna(False),
            "exit": rules.exit_signal.reindex(df.index).fillna(False),
            "stop": tb.DEFAULT_STRATEGY.get("stop_loss_pct"),
            "tp": tb.DEFAULT_STRATEGY.get("take_profit_pct"),
            "hold": tb.DEFAULT_STRATEGY.get("max_hold_bars"),
        }
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()]))

    cash = START_CASH
    pos: dict[str, dict] = {}          # sym -> {units, entry_px, entry_date}
    equity_curve, invested_days, trades, wins = [], 0, 0, 0

    def price(s, d):  # close on date d, else None
        c = sig[s]["close"]
        return float(c.loc[d]) if d in c.index else None

    for d in all_dates:
        # ---- exits first (free up capital) ----
        for s in list(pos.keys()):
            row = sig[s]
            if d not in row["close"].index:
                continue
            p = pos[s]
            held = (d - p["entry_date"]).days
            o, hi, lo, c = (float(row["open"].loc[d]), float(row["high"].loc[d]),
                            float(row["low"].loc[d]), float(row["close"].loc[d]))
            stop_px = p["entry_px"] * (1 - row["stop"]) if row["stop"] else None
            tp_px = p["entry_px"] * (1 + row["tp"]) if row["tp"] else None
            exit_px = None
            if stop_px is not None and lo <= stop_px:
                exit_px = min(o, stop_px) if o < stop_px else stop_px
            elif tp_px is not None and hi >= tp_px:
                exit_px = max(o, tp_px) if o > tp_px else tp_px
            elif bool(row["exit"].loc[d]):
                exit_px = c
            elif row["hold"] and held >= row["hold"] * 1.5:  # ~bars→days fudge
                exit_px = c
            if exit_px is not None:
                fill = exit_px * (1 - SLIP)
                proceeds = p["units"] * fill * (1 - FEE)
                cash += proceeds
                trades += 1
                if fill > p["entry_px"]:
                    wins += 1
                del pos[s]

        # ---- entries (fill at today's open on yesterday's signal) ----
        idx_pos = all_dates.index(d)
        if idx_pos > 0:
            prev = all_dates[idx_pos - 1]
            for s, row in sig.items():
                if len(pos) >= n_slots:
                    break
                if s in pos or prev not in row["entry"].index or d not in row["open"].index:
                    continue
                if not bool(row["entry"].loc[prev]):
                    continue
                equity_now = cash + sum(
                    pp["units"] * (price(ss, d) or pp["entry_px"]) for ss, pp in pos.items())
                alloc = min(equity_now / n_slots, cash)
                if alloc < 1:
                    continue
                entry_px = float(row["open"].loc[d]) * (1 + SLIP)
                units = (alloc * (1 - FEE)) / entry_px
                cash -= alloc
                pos[s] = {"units": units, "entry_px": entry_px, "entry_date": d}

        mv = sum(pp["units"] * (price(ss, d) or pp["entry_px"]) for ss, pp in pos.items())
        equity_curve.append(cash + mv)
        if pos:
            invested_days += 1

    eq = pd.Series(equity_curve, index=pd.DatetimeIndex(all_dates))
    final = float(eq.iloc[-1])
    yrs = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = (final / START_CASH) ** (1 / yrs) - 1
    dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    return {
        "final": final, "ret_pct": (final / START_CASH - 1) * 100, "cagr_pct": cagr * 100,
        "maxdd_pct": dd, "trades": trades, "win_pct": (wins / trades * 100) if trades else 0,
        "invested_pct": invested_days / len(all_dates) * 100, "years": yrs,
        "start": all_dates[0].date().isoformat(), "end": all_dates[-1].date().isoformat(),
    }


def main() -> None:
    for name, uni, slots in [("Default 8-name universe", tb.DEFAULT_UNIVERSE, 8)]:
        r = simulate_account(uni, n_slots=slots)
        print(f"\n[{name}]  one account, ≤{slots} concurrent, sized equity/{slots}, net of costs")
        print(f"  {r['start']} → {r['end']} ({r['years']:.1f}y)")
        print(f"  ${START_CASH:,.0f} → ${r['final']:,.0f}   ({r['ret_pct']:.0f}% total, "
              f"{r['cagr_pct']:.1f}%/yr CAGR)")
        print(f"  max drawdown {r['maxdd_pct']:.1f}% · {r['trades']} trades · "
              f"{r['win_pct']:.1f}% win · invested only {r['invested_pct']:.0f}% of days")


if __name__ == "__main__":
    main()
