"""Trade-level backtester for the automated trading bot.

Unlike `engine.py` (which simulates a continuously-rebalanced position fraction
and reports an equity curve), this engine pairs every entry with an exit and
reports **per-trade** outcomes — so it can compute a true win rate, profit
factor, and expectancy. That is what the bot is graded on.

Entries come from a strategy signal series (see `app.services.trading_bot`).
Exits are resolved by the simulator from a set of exit rules: a mean-reversion
target, a hard stop-loss, and a time stop. Fills are modelled on the *next*
bar's open to avoid look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExitRules:
    """How an open position is closed.

    exit_signal:  a boolean Series (aligned to the bars) that, when True, closes
                  the position at that bar's close (the mean-reversion target).
    stop_loss_pct: hard stop as a fraction below entry (e.g. 0.08 = -8%). The
                  fill is the stop price (or the bar's open if it gapped through).
    max_hold_bars: time stop — force-close after this many bars in the trade.
    """

    exit_signal: pd.Series
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_hold_bars: int | None = None


@dataclass
class Trade:
    entry_ts: str
    entry_price: float
    exit_ts: str
    exit_price: float
    bars_held: int
    return_pct: float
    exit_reason: str
    win: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_ts": self.entry_ts,
            "entry_price": round(self.entry_price, 4),
            "exit_ts": self.exit_ts,
            "exit_price": round(self.exit_price, 4),
            "bars_held": self.bars_held,
            "return_pct": round(self.return_pct, 4),
            "exit_reason": self.exit_reason,
            "win": self.win,
        }


@dataclass
class BacktestReport:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trades": [t.as_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
            "metrics": self.metrics,
        }


def _compute_metrics(trades: list[Trade], equity: pd.Series, initial_cash: float) -> dict[str, Any]:
    n = len(trades)
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    win_rate = (len(wins) / n * 100) if n else 0.0
    gross_win = sum(t.return_pct for t in wins)
    gross_loss = abs(sum(t.return_pct for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (-gross_loss / len(losses)) if losses else 0.0
    expectancy = (sum(t.return_pct for t in trades) / n) if n else 0.0

    final_equity = float(equity.iloc[-1]) if len(equity) else initial_cash
    rets = equity.pct_change().dropna() if len(equity) else pd.Series(dtype=float)
    running_max = equity.cummax() if len(equity) else pd.Series(dtype=float)
    mdd = float(((equity - running_max) / running_max).min() * 100) if len(equity) else 0.0
    sharpe = 0.0
    if not rets.empty and rets.std() not in (0, np.nan):
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std())

    return {
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "expectancy_pct": round(expectancy, 4),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 2) if trades else 0.0,
        "initial_cash": initial_cash,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / initial_cash - 1) * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "sharpe": round(sharpe, 3),
    }


def simulate(
    df: pd.DataFrame,
    entries: pd.Series,
    rules: ExitRules,
    *,
    symbol: str = "",
    initial_cash: float = 10_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 0.0,
) -> BacktestReport:
    """Event-driven, one-position-at-a-time long-only simulator.

    A position is opened at the open of the bar *after* an entry signal. Each
    subsequent bar, exits are checked in priority order: stop-loss, take-profit,
    exit-signal, time-stop. The whole equity is allocated to each trade.

    `slippage_bps` is the adverse price move per fill (slippage + half the
    bid-ask spread): buys fill that much above the quoted price and sells that
    much below it. Reported per-trade returns are net of both slippage and the
    `commission_bps` round-trip fee — gross fills materially overstate edge for
    a strategy that takes many small mean-reversion pops.
    """
    idx = df.index
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    entry_sig = entries.reindex(idx).fillna(False).to_numpy(dtype=bool)
    exit_sig = rules.exit_signal.reindex(idx).fillna(False).to_numpy(dtype=bool)
    n = len(df)

    cash = initial_cash
    equity = np.full(n, initial_cash, dtype=float)
    trades: list[Trade] = []

    in_pos = False
    entry_price = 0.0
    entry_i = 0
    units = 0.0
    fee_rate = commission_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    for i in range(n):
        price = close[i]

        if in_pos:
            bars_held = i - entry_i
            exit_price: float | None = None
            reason = ""

            stop_px = entry_price * (1 - rules.stop_loss_pct) if rules.stop_loss_pct else None
            tp_px = entry_price * (1 + rules.take_profit_pct) if rules.take_profit_pct else None

            # Priority: protective stop, then target, then mean-reversion signal,
            # then time stop. Intrabar stop/target use the bar's low/high.
            if stop_px is not None and low[i] <= stop_px:
                exit_price = min(open_[i], stop_px) if open_[i] < stop_px else stop_px
                reason = "stop_loss"
            elif tp_px is not None and high[i] >= tp_px:
                exit_price = max(open_[i], tp_px) if open_[i] > tp_px else tp_px
                reason = "take_profit"
            elif exit_sig[i]:
                exit_price = price
                reason = "exit_signal"
            elif rules.max_hold_bars is not None and bars_held >= rules.max_hold_bars:
                exit_price = price
                reason = "time_stop"

            if exit_price is not None:
                fill_px = exit_price * (1 - slip_rate)  # sell below the quote
                proceeds = units * fill_px
                proceeds -= proceeds * fee_rate
                cash = proceeds
                # Net per-trade return, inclusive of BOTH commission legs: a buy
                # pays fee_rate on entry and the sell pays it again on exit, so the
                # round-trip multiplies the gross price move by (1 - fee_rate)**2.
                # entry_price/fill_px already include slippage, so this return is
                # net of every modelled cost — which is what the headline win-rate,
                # expectancy and profit-factor are computed from.
                ret_pct = ((1 - fee_rate) ** 2 * (fill_px / entry_price) - 1) * 100
                exit_price = fill_px
                trades.append(
                    Trade(
                        entry_ts=idx[entry_i].isoformat(),
                        entry_price=float(entry_price),
                        exit_ts=idx[i].isoformat(),
                        exit_price=float(exit_price),
                        bars_held=int(bars_held),
                        return_pct=float(ret_pct),
                        exit_reason=reason,
                        win=bool(ret_pct > 0),  # a net win, after costs
                    )
                )
                in_pos = False
                units = 0.0

        # Enter on the bar following a signal (fill at this bar's open).
        if not in_pos and i > 0 and entry_sig[i - 1]:
            entry_price = open_[i] * (1 + slip_rate)  # buy above the quote
            if entry_price > 0:
                units = (cash * (1 - fee_rate)) / entry_price
                cash = 0.0
                in_pos = True
                entry_i = i

        equity[i] = cash + units * price

    eq = pd.Series(equity, index=idx)
    metrics = _compute_metrics(trades, eq, initial_cash)
    return BacktestReport(
        symbol=symbol,
        trades=trades,
        equity_curve=[{"ts": ts.isoformat(), "equity": round(float(v), 2)} for ts, v in eq.items()],
        metrics=metrics,
    )
