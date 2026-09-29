"""Automated trading bot — strategy signals, backtesting, and paper execution.

The bot's default strategy is a trend-filtered short-term mean-reversion system
(a Connors RSI-2 variant). Mean reversion in an uptrend produces a high *win
rate* because the edge is "buy a brief, sharp dip in something that is
structurally rising, then sell the bounce back to the short MA." The trade-offs
are small average wins and occasional larger losses — which is why the engine
also reports profit factor and expectancy, not just win rate.

Strategy DSL (all keys optional, defaults shown):
    {
        "kind": "rsi2_meanrev",
        "rsi_period": 2,
        "rsi_entry": 10.0,      # enter long when RSI(2) closes below this
        "trend_sma": 200,       # only trade longs while close > SMA(trend_sma)
        "exit_sma": 5,          # exit when close rises back above SMA(exit_sma)
        "stop_loss_pct": 0.10,  # hard protective stop below entry
        "max_hold_bars": 20     # time stop
    }
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from app.backtest.bot_engine import BacktestReport, ExitRules, simulate
from app.services import bot_intel
from app.services import gate_log
from app.services import broker_alpaca as broker
from app.services import indicators as ind
from app.services import market_data as md
from app.services import regime as regime_mod
from app.services import swing as swing_mod

# Tuned config. Two honest reads, both NET of slippage (5bps/side) + commission:
#   * Default 8-name universe (survivor-biased — overstates live edge):
#       ~85% win rate, profit factor ~3.1, expectancy ~+0.86%/trade.
#   * Broad 39-name universe (sectors/cyclicals/laggards, out-of-sample 2022+):
#       ~82% win rate, profit factor ~2.15, expectancy ~+0.55%/trade.
# The broad/OOS number is the one to trust. An out-of-sample parameter search
# found no robust improvement on the entry/exit thresholds (looser settings
# overfit and degraded OOS), so those defaults are kept; the only validated
# improvement was the market-regime filter below.
DEFAULT_STRATEGY: dict[str, Any] = {
    "kind": "rsi2_meanrev",
    "rsi_period": 2,
    "rsi_entry": 10.0,
    "trend_sma": 200,
    "exit_mode": "sma",
    # Exit when price closes back above SMA(exit_sma). 20 beat the old 15 on
    # every axis, in- AND out-of-sample, on both universes: higher profit factor
    # and expectancy, same ~82% win rate, and a SMALLER average loss (a slightly
    # slower exit lets the bounce develop and closes losers on a firmer rebound).
    "exit_sma": 20,
    "take_profit_pct": 0.01,
    "stop_loss_pct": 0.25,
    "max_hold_bars": 40,
    # Market-regime filter: only take dips while a broad-market proxy is itself
    # in an uptrend (proxy close > its SMA). An oversold dip *during a market-
    # wide downtrend* is where the strategy's worst losers cluster — the dip
    # keeps falling. Gating on regime holds the win rate (~82%) but lifts the
    # profit factor and trims the average loss (validated out-of-sample on a
    # wide, less survivor-biased universe). Fail-open: if the proxy can't be
    # fetched, the filter is skipped rather than blocking the signal.
    "market_filter": True,
    "market_proxy": "SPY",
    "market_sma": 200,
    # When true, a live BUY trigger is routed through the LLM risk-gate
    # (news + smart-money review) before any order is placed. Backtests ignore
    # this flag — it only affects live/scan entries via `latest_signal`.
    "llm_gate": True,
}

# Swing bots (kind "swing_breakout" / "swing_pullback") delegate signals and
# exit management to app.services.swing. Risk-based sizing: each entry risks
# `execution.risk_per_trade_pct` of equity between the fill and the stop, in
# whole shares so a broker-side GTC stop can be attached. The LLM gate is OFF
# by default here because it cannot be backtested — the swing results in
# BACKTEST_RESULTS.md are for the systematic rules alone.
DEFAULT_SWING_STRATEGY: dict[str, Any] = {
    "kind": "swing_breakout",
    "params": {},                  # overrides for swing.SETUPS[setup]["params"]
    "earnings_blackout_days": 7,   # skip entries with a report this close
    "llm_gate": False,
}
SWING_DEFAULT_UNIVERSE = list(swing_mod.UNIVERSES["etfs"]["symbols"])


def is_swing(dsl: dict[str, Any] | None) -> bool:
    return bool(dsl) and str(dsl.get("kind", "")) in swing_mod.BOT_KINDS


def merge_dsl(dsl: dict[str, Any] | None) -> dict[str, Any]:
    """Fill a (possibly partial) DSL with the defaults of its strategy family."""
    if is_swing(dsl):
        return {**DEFAULT_SWING_STRATEGY, **(dsl or {})}
    return {**DEFAULT_STRATEGY, **(dsl or {})}


# Liquid, structurally-uptrending names the mean-reversion edge works best on.
# NOTE: this is a survivor set — backtests on it overstate the live edge because
# the decade's losers/delistings are absent. Treat pooled stats as illustrative.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META"]

# Per-fill execution cost (slippage + half the bid-ask spread) applied in
# backtests so reported win rate / expectancy are net, not gross. 5bps/side is
# realistic-to-conservative for these large, liquid names; raise it for less
# liquid tickers. Gross (0bps) fills materially flatter a many-small-wins
# mean-reversion strategy.
DEFAULT_SLIPPAGE_BPS = 5.0

# Live-execution safety defaults. Conservative on purpose — a fresh live bot
# can only ever risk a small, capped slice of the account per name, and the
# daily kill-switch halts trading after a few orders or a small loss.
DEFAULT_EXECUTION: dict[str, Any] = {
    "max_position_usd": 1_000.0,   # hard ceiling on each new position
    "max_daily_orders": 10,        # stop opening/closing after this many orders/day
    "max_daily_loss_usd": 500.0,   # stop trading once realized losses hit this
    "market_hours_only": True,     # only submit during regular US market hours
    # --- portfolio-level risk (conservative defaults) ---
    "max_gross_exposure_usd": 5_000.0,  # cap total long market value across all names
    "max_open_positions": 5,            # cap number of concurrent positions
    "max_total_drawdown_pct": 0.15,     # halt NEW entries if equity is down >15% from HWM
    # --- PDT protection ---
    # On a sub-$25k margin account, opening a same-day round-trip risks a
    # pattern-day-trader flag. When True, refuse new entries if the account is
    # flagged PDT or under the equity floor.
    "pdt_protect": True,
    "pdt_equity_floor": 25_000.0,
    # Use a broker-side protective bracket (stop + take-profit) on entries when
    # the broker supports it; falls back to a plain market order otherwise.
    "use_broker_bracket": True,
    # Consume the top-down market-regime overlay to scale entry sizing (a
    # hostile tape down-sizes every BUY). Off by default because it makes a
    # network call to assess the regime; flip on for live operation.
    "regime_sizing": False,
    # Swing bots only: % of equity at risk between entry and initial stop.
    "risk_per_trade_pct": 1.0,
    # "alpaca" (the configured Alpaca account, paper by default) or "sim" (a
    # private simulated paper ledger per bot — see broker_sim).
    "broker": "alpaca",
    "sim_initial_cash": 100_000.0,
    # Opt-in to the daily auto-run (also needs BOT_AUTORUN=1 on the server).
    "autorun": False,
}


def _build_signals(df: pd.DataFrame, dsl: dict[str, Any]) -> tuple[pd.Series, ExitRules]:
    """Compile a strategy DSL into (entry_signal, exit_rules) for the simulator."""
    kind = dsl.get("kind", "rsi2_meanrev")
    close = df["close"]

    if kind == "rsi2_meanrev":
        rsi_period = int(dsl.get("rsi_period", 2))
        rsi_entry = float(dsl.get("rsi_entry", 10.0))
        trend_sma = int(dsl.get("trend_sma", 200))
        exit_sma = int(dsl.get("exit_sma", 5))
        exit_mode = dsl.get("exit_mode", "sma")

        rsi = ind.rsi(close, rsi_period)
        trend = ind.sma(close, trend_sma)

        in_uptrend = close > trend
        oversold = rsi < rsi_entry
        entries = (in_uptrend & oversold).fillna(False)

        # Exit mode controls how aggressively the bounce is taken — the faster
        # the exit, the higher the win rate (mean-reversion pops are brief).
        if exit_mode == "first_up":
            # Close the trade on the first up-close (close > prior close).
            exit_signal = (close > close.shift(1)).fillna(False)
        elif exit_mode == "rsi":
            exit_rsi = float(dsl.get("exit_rsi", 65.0))
            exit_signal = (rsi > exit_rsi).fillna(False)
        else:  # "sma"
            exit_ma = ind.sma(close, exit_sma)
            exit_signal = (close > exit_ma).fillna(False)

        rules = ExitRules(
            exit_signal=exit_signal,
            stop_loss_pct=dsl.get("stop_loss_pct"),
            take_profit_pct=dsl.get("take_profit_pct"),
            max_hold_bars=dsl.get("max_hold_bars"),
        )
        return entries, rules

    if kind == "bollinger_meanrev":
        period = int(dsl.get("bb_period", 20))
        k = float(dsl.get("bb_k", 2.0))
        trend_sma = int(dsl.get("trend_sma", 200))
        bb = ind.bollinger(close, period, k)
        trend = ind.sma(close, trend_sma)
        entries = ((close < bb["lower"]) & (close > trend)).fillna(False)
        exit_signal = (close > bb["mid"]).fillna(False)
        rules = ExitRules(
            exit_signal=exit_signal,
            stop_loss_pct=dsl.get("stop_loss_pct"),
            take_profit_pct=dsl.get("take_profit_pct"),
            max_hold_bars=dsl.get("max_hold_bars"),
        )
        return entries, rules

    raise ValueError(f"unsupported bot strategy kind: {kind}")


def _market_uptrend(index: pd.Index, dsl: dict[str, Any]) -> pd.Series | None:
    """Boolean 'broad market is in an uptrend' series aligned to `index`.

    Returns None (→ caller skips the filter) when the market proxy can't be
    fetched, so a data hiccup never blocks an otherwise-valid signal. No
    look-ahead: the regime is read at each bar's close, and entries only ever
    fill on the *next* bar's open.
    """
    if not dsl.get("market_filter", False):
        return None
    proxy = str(dsl.get("market_proxy", "SPY"))
    sma_n = int(dsl.get("market_sma", 200))
    try:
        mkt = (swing_mod._recent_bars(proxy) if dsl.get("completed_bars")
               else md.get_history(proxy, interval="1d", range_="max"))
        if mkt.empty:
            return None
        regime = (mkt["close"] > ind.sma(mkt["close"], sma_n)).fillna(False)
        return regime.reindex(index).ffill().fillna(False)
    except Exception:  # noqa: BLE001 — fail open, don't block the strategy
        return None


def _entries_with_regime(df: pd.DataFrame, dsl: dict[str, Any]) -> tuple[pd.Series, ExitRules]:
    """`_build_signals` + the optional market-regime gate on entries."""
    entries, rules = _build_signals(df, dsl)
    regime = _market_uptrend(df.index, dsl)
    if regime is not None:
        entries = (entries & regime).fillna(False)
    return entries, rules


def backtest_symbol(
    symbol: str,
    dsl: dict[str, Any] | None = None,
    *,
    range_: str = "10y",
    initial_cash: float = 10_000.0,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestReport:
    if is_swing(dsl):
        raise ValueError("swing strategies backtest at the portfolio level: use POST /swing/backtest")
    dsl = {**DEFAULT_STRATEGY, **(dsl or {})}
    df = md.get_history(symbol, interval="1d", range_=range_)
    if df.empty or len(df) < int(dsl.get("trend_sma", 200)) + 10:
        raise ValueError(f"insufficient data for {symbol}")
    entries, rules = _entries_with_regime(df, dsl)
    return simulate(
        df, entries, rules, symbol=symbol.upper(),
        initial_cash=initial_cash, slippage_bps=slippage_bps,
    )


def backtest_portfolio(
    symbols: list[str] | None = None,
    dsl: dict[str, Any] | None = None,
    *,
    range_: str = "10y",
    initial_cash: float = 10_000.0,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Backtest the strategy across a basket and aggregate a pooled win rate.

    The pooled win rate (all trades across all symbols) is the headline number
    the bot is graded on — it reflects the strategy, not one lucky ticker.
    Net of `slippage_bps` per fill; see DEFAULT_SLIPPAGE_BPS.
    """
    if is_swing(dsl):
        raise ValueError("swing strategies backtest at the portfolio level: use POST /swing/backtest")
    symbols = symbols or DEFAULT_UNIVERSE
    dsl = {**DEFAULT_STRATEGY, **(dsl or {})}
    per_symbol: dict[str, Any] = {}
    all_trades = []
    errors: dict[str, str] = {}

    for sym in symbols:
        try:
            rep = backtest_symbol(
                sym, dsl, range_=range_, initial_cash=initial_cash, slippage_bps=slippage_bps,
            )
        except Exception as e:  # noqa: BLE001 — one bad symbol shouldn't sink the run
            errors[sym] = str(e)
            continue
        per_symbol[sym] = rep.metrics
        all_trades.extend(rep.trades)

    n = len(all_trades)
    wins = [t for t in all_trades if t.win]
    losses = [t for t in all_trades if not t.win]
    gross_win = sum(t.return_pct for t in wins)
    gross_loss = abs(sum(t.return_pct for t in losses))
    pooled = {
        "n_trades": n,
        "n_wins": len(wins),
        "win_rate_pct": round(len(wins) / n * 100, 2) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "avg_win_pct": round(gross_win / len(wins), 3) if wins else 0.0,
        "avg_loss_pct": round(-gross_loss / len(losses), 3) if losses else 0.0,
        "expectancy_pct": round(sum(t.return_pct for t in all_trades) / n, 4) if n else 0.0,
    }
    return {
        "strategy": dsl,
        "universe": symbols,
        "pooled": pooled,
        "per_symbol": per_symbol,
        "errors": errors,
    }


def latest_signal(symbol: str, dsl: dict[str, Any] | None = None, *, range_: str = "2y",
                  llm_gate: bool | None = None, universe: list[str] | None = None,
                  context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate the strategy on the most recent bar — the live bot decision.

    Returns the action the bot would take *today* for `symbol`: BUY when a fresh
    entry signal fires, else FLAT. Paper/live execution layers consume this.

    When `llm_gate` is True (or None and the DSL enables it) AND the quant
    trigger fires, the BUY is routed through `bot_intel.llm_review` — a news +
    smart-money pressure-test that may VETO (→ FLAT) or DOWNSIZE the entry. The
    LLM never creates a BUY that the systematic model didn't fire, and never
    upsizes; it can only confirm or reduce risk. `size_multiplier` rides along
    so the executor can scale the order. Fail-open: if the LLM is unavailable,
    the systematic BUY stands.
    """
    dsl = merge_dsl(dsl)
    if llm_gate is None:
        llm_gate = bool(dsl.get("llm_gate", False))
    if is_swing(dsl):
        out = swing_mod.latest_signal(symbol, dsl)
        if out.get("action") == "BUY" and dsl.get("ml_filter"):
            out = _apply_ml_filter(symbol, out, dsl, universe)
        if out.get("action") == "BUY" and llm_gate:
            out = _apply_llm_gate(symbol, out, dsl=dsl, context=context)
        return out
    # `completed_bars`: decide on the last *finished* session from the same
    # consolidated daily data the backtests use (signal at the close, act at
    # the next open). Without it the bot reads today's still-forming bar.
    if dsl.get("completed_bars"):
        df = swing_mod._recent_bars(symbol)
    else:
        df = md.get_history(symbol, interval="1d", range_=range_)
    if df.empty:
        return {"symbol": symbol.upper(), "action": "FLAT", "reason": "no data"}
    entries, rules = _entries_with_regime(df, dsl)
    close = df["close"]
    rsi = ind.rsi(close, int(dsl.get("rsi_period", 2)))
    fired = bool(entries.iloc[-1])
    out: dict[str, Any] = {
        "symbol": symbol.upper(),
        "action": "BUY" if fired else "FLAT",
        "as_of": df.index[-1].isoformat(),
        "price": round(float(close.iloc[-1]), 4),
        "rsi": round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else None,
        "exit_target": "close > SMA(%d)" % int(dsl.get("exit_sma", 5)),
        "reason": "RSI(%d) oversold in uptrend" % int(dsl.get("rsi_period", 2)) if fired else "no entry condition met",
        "size_multiplier": 1.0,
    }
    if fired and dsl.get("completed_bars"):
        # Size and place brackets off the price we'd actually pay now.
        out["signal_close"] = out["price"]
        try:
            live = float(md.get_quote(symbol).get("price") or 0.0)
            if live > 0:
                out["price"] = round(live, 4)
        except Exception:  # noqa: BLE001 — fall back to the signal close
            pass

    # Optional ML signal filter, then the LLM risk-gate — both only on a live
    # quant BUY, and both can only decline or shrink it.
    if fired and dsl.get("ml_filter"):
        out = _apply_ml_filter(symbol, out, dsl, universe)
    if out.get("action") == "BUY" and llm_gate:
        out = _apply_llm_gate(symbol, out, dsl=dsl, context=context)
    return out


def _apply_ml_filter(symbol: str, out: dict[str, Any], dsl: dict[str, Any],
                     universe: list[str] | None) -> dict[str, Any]:
    """Skip the BUY when the ML signal filter rates it below average.

    Off by default: in walk-forward tests it hurt the ETF breakout and helped
    only RSI(2) on survivor-biased large caps (see BACKTEST_RESULTS.md).
    Fail-open, like the LLM gate: if the model can't be trained or scored,
    the systematic signal stands and the reason says so.
    """
    from app.ml import signal_filter

    setup = swing_mod.BOT_KINDS.get(str(dsl.get("kind")), "rsi2")
    train_on = [s.upper() for s in (universe or [])] or (
        SWING_DEFAULT_UNIVERSE if is_swing(dsl) else DEFAULT_UNIVERSE)
    try:
        res = signal_filter.score_latest(symbol, setup, train_on,
                                         dsl.get("params") if is_swing(dsl) else None)
    except Exception as e:  # noqa: BLE001
        out["ml_filter"] = {"error": str(e)[:200]}
        out["reason"] += " · ML filter unavailable"
        return out
    out["ml_filter"] = res
    if not res["pass"]:
        out["action"] = "FLAT"
        out["size_multiplier"] = 0.0
        out["reason"] = f"ML filter skip: P(win) {res['p']:.2f} < {res['threshold']:.2f}"
    return out


def _apply_llm_gate(symbol: str, out: dict[str, Any], *, dsl: dict[str, Any] | None = None,
                    context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route a quant BUY through the LLM risk-gate (veto / downsize only).

    Every verdict is written to the forward-test log (gate_log) so the gate's
    value can be measured on trades it had no way of knowing the outcome of.
    """
    try:
        verdict = bot_intel.llm_review(symbol, out)
    except Exception as e:  # noqa: BLE001 — gate failure must not block a proven signal
        verdict = {"decision": "APPROVE", "conviction": 0.5, "size_multiplier": 1.0,
                   "rationale": f"gate error, deferring to signal: {e}",
                   "key_risks": [], "provider": "error"}
    ctx = context or {}
    gate_log.record(symbol, out, verdict, kind=str((dsl or {}).get("kind") or "rsi2_meanrev"),
                    source=ctx.get("source", "signal"), bot_id=ctx.get("bot_id"))
    out["llm"] = verdict
    if verdict["decision"] == "VETO":
        out["action"] = "FLAT"
        out["size_multiplier"] = 0.0
        out["reason"] = f"LLM veto: {verdict.get('rationale', '')}"
    else:
        out["size_multiplier"] = float(verdict.get("size_multiplier", 1.0))
        if verdict["decision"] == "DOWNSIZE":
            out["reason"] += f" · LLM downsize ×{out['size_multiplier']:.2f}"
    return out


# ---------------------------------------------------------------------------
# Live execution
# ---------------------------------------------------------------------------

def evaluate_position_exit(
    symbol: str,
    dsl: dict[str, Any] | None,
    *,
    entry_price: float,
    current_price: float,
    range_: str = "2y",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether an *open* position should be closed right now.

    Mirrors the backtest exit priority (stop-loss → take-profit → signal →
    time stop) against the broker's real fill price and the latest bar.
    `meta` is the entry record `run_live` persisted when it opened the
    position; with it the time stop is enforced live (positions opened before
    this record existed simply skip the time stop). Swing kinds delegate to
    `swing.evaluate_exit`, which may also return a raised trailing stop.
    """
    dsl = merge_dsl(dsl)
    if is_swing(dsl):
        return swing_mod.evaluate_exit(symbol, dsl, meta, entry_price=entry_price,
                                       current_price=current_price)
    stop = dsl.get("stop_loss_pct")
    tp = dsl.get("take_profit_pct")
    if entry_price > 0:
        if stop and current_price <= entry_price * (1 - float(stop)):
            return {"exit": True, "reason": "stop_loss"}
        if tp and current_price >= entry_price * (1 + float(tp)):
            return {"exit": True, "reason": "take_profit"}
    try:
        df = (swing_mod._recent_bars(symbol) if dsl.get("completed_bars")
              else md.get_history(symbol, interval="1d", range_=range_))
        if not df.empty:
            _entries, rules = _build_signals(df, dsl)
            if bool(rules.exit_signal.reindex(df.index).fillna(False).iloc[-1]):
                return {"exit": True, "reason": "exit_signal"}
            max_hold = dsl.get("max_hold_bars")
            if max_hold and meta and meta.get("entry_date"):
                done = swing_mod.completed_bars(df, symbol)
                held = int((done.index.normalize() >= pd.Timestamp(meta["entry_date"], tz="UTC")).sum())
                if held >= int(max_hold):
                    return {"exit": True, "reason": "time_stop"}
    except Exception as e:  # noqa: BLE001 — never let a data hiccup force a bad close
        return {"exit": False, "reason": f"hold (signal check failed: {e})"}
    return {"exit": False, "reason": "hold"}


def _today_str() -> str:
    return date.today().isoformat()


def _fresh_run_state() -> dict[str, Any]:
    # `equity_hwm` (high-water mark) intentionally persists across day rollovers
    # via the caller, so the drawdown kill-switch tracks a real peak; a fresh
    # state seeds it to 0.0 and the first run sets it to current equity.
    return {"trading_day": _today_str(), "orders_today": 0, "realized_loss_today": 0.0,
            "equity_hwm": 0.0, "log": []}


def run_live(
    *,
    universe: list[str],
    dsl: dict[str, Any] | None,
    execution: dict[str, Any] | None,
    run_state: dict[str, Any] | None,
    broker_mod=broker,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one live pass for a bot: close exits, open new entries.

    Pure orchestration with an injectable broker (so it's testable offline):
      1. Refuse if the broker isn't configured.
      2. Reset the daily counters when the trading day rolls over.
      3. Enforce market-hours-only (if set) — otherwise report `blocked`.
      4. Close any held position whose exit condition has fired.
      5. Open BUY signals for un-held universe symbols, sized to
         `max_position_usd`, subject to the daily order/loss kill-switch and
         available buying power.

    Returns the actions taken, an order log, the (possibly reset) run_state,
    and an account snapshot. Never raises on a single-symbol error.
    """
    dsl = merge_dsl(dsl)
    execution = {**DEFAULT_EXECUTION, **(execution or {})}
    state = dict(run_state or {})
    if state.get("trading_day") != _today_str():
        prior_hwm = float(state.get("equity_hwm", 0.0) or 0.0)
        state = _fresh_run_state()
        state["equity_hwm"] = prior_hwm  # HWM is a peak — survives day rollovers
    state.setdefault("log", [])
    state.setdefault("equity_hwm", 0.0)
    # Per-position entry records (entry date, stop, target, stop order id).
    # They survive day rollovers: a swing position lives for days or weeks.
    state["positions"] = dict((run_state or {}).get("positions") or {})

    if not broker_mod.is_configured():
        return {"blocked": "broker_not_configured", "actions": [], "run_state": state, "account": None}

    try:
        market_open = broker_mod.is_market_open()
    except Exception as e:  # noqa: BLE001
        return {"blocked": f"clock_unavailable: {e}", "actions": [], "run_state": state, "account": None}

    if execution.get("market_hours_only", True) and not market_open:
        return {"blocked": "market_closed", "actions": [], "run_state": state,
                "account": _safe_account(broker_mod), "market_open": False}

    account = _safe_account(broker_mod)
    actions: list[dict[str, Any]] = []
    try:
        positions = {p["symbol"]: p for p in broker_mod.list_positions()}
    except Exception as e:  # noqa: BLE001
        return {"blocked": f"positions_unavailable: {e}", "actions": [], "run_state": state, "account": account}

    max_orders = int(execution.get("max_daily_orders", 10))
    max_loss = float(execution.get("max_daily_loss_usd", 500.0))
    max_pos = float(execution.get("max_position_usd", 1_000.0))
    max_gross = float(execution.get("max_gross_exposure_usd", 0.0) or 0.0)
    max_open = int(execution.get("max_open_positions", 0) or 0)
    max_dd_pct = float(execution.get("max_total_drawdown_pct", 0.0) or 0.0)

    # --- Track equity high-water mark and evaluate the drawdown kill-switch ---
    equity = float(account.get("equity", 0.0)) if account else 0.0
    if equity > state.get("equity_hwm", 0.0):
        state["equity_hwm"] = equity
    hwm = float(state.get("equity_hwm", 0.0) or 0.0)
    drawdown_halt = (
        max_dd_pct > 0.0 and hwm > 0.0 and equity < hwm * (1.0 - max_dd_pct)
    )

    # --- Top-down regime size multiplier (gap 5). Defaults to 1.0 if disabled
    #     or if the regime overlay is unavailable; never blocks trading. ---
    regime_mult = 1.0
    if execution.get("regime_sizing", False):
        try:
            regime_mult = float(regime_mod.assess().get("position_sizing_mult", 1.0) or 1.0)
        except Exception:  # noqa: BLE001 — regime is an overlay, never fatal
            regime_mult = 1.0

    # Current gross long exposure and open-position count (held + opened this run).
    open_count = sum(1 for p in positions.values()
                     if p.get("side") == "long" and p.get("qty", 0) > 0)
    gross_exposure = sum(float(p.get("market_value", 0.0) or 0.0)
                         for p in positions.values()
                         if p.get("side") == "long" and p.get("qty", 0) > 0)

    def kill_switch_hit() -> str | None:
        if state["orders_today"] >= max_orders:
            return "max_daily_orders reached"
        if state["realized_loss_today"] >= max_loss:
            return "max_daily_loss reached"
        return None

    def entry_blocked(notional: float) -> str | None:
        """Portfolio-level guards that only gate NEW entries (not exits)."""
        if drawdown_halt:
            return (f"max_total_drawdown reached (equity {equity:.0f} "
                    f"< {(1.0 - max_dd_pct) * 100:.0f}% of HWM {hwm:.0f})")
        if max_open > 0 and open_count >= max_open:
            return f"max_open_positions reached ({open_count}/{max_open})"
        if max_gross > 0 and (gross_exposure + notional) > max_gross:
            return (f"max_gross_exposure reached "
                    f"({gross_exposure:.0f}+{notional:.0f} > {max_gross:.0f})")
        return None

    # PDT: a sub-$25k *real-money margin* account that opens (and may same-day
    # close) round-trips risks a pattern-day-trader flag. Block new entries when
    # flagged or below the equity floor. Paper accounts are exempt (no real PDT
    # enforcement), and we only gate when the broker actually reports being live.
    pdt_protect = bool(execution.get("pdt_protect", True))
    pdt_floor = float(execution.get("pdt_equity_floor", 25_000.0))
    pdt_flagged = bool(account.get("pattern_day_trader", False)) if account else False
    is_paper = True
    try:
        if hasattr(broker_mod, "is_paper"):
            is_paper = bool(broker_mod.is_paper())
    except Exception:  # noqa: BLE001
        is_paper = True
    pdt_block_reason = None
    if pdt_protect and not is_paper and (pdt_flagged or (equity > 0.0 and equity < pdt_floor)):
        pdt_block_reason = (
            f"PDT_BLOCKED: account flagged={pdt_flagged}, equity {equity:.0f} "
            f"< floor {pdt_floor:.0f} — opening a round-trip risks a PDT violation"
        )

    # --- 0. Reconcile entry records with the broker. A record without a
    #        position means a broker-side stop/target leg closed it. ---
    for sym in list(state["positions"]):
        if sym in positions:
            continue
        meta = state["positions"].pop(sym)
        _cancel_leftovers(broker_mod, sym)
        rec = {"symbol": sym, "action": "CLOSED_BY_BROKER",
               "reason": "protective stop/target filled at the broker",
               "stop": meta.get("stop"), "target": meta.get("target"),
               "ts": datetime.now(timezone.utc).isoformat()}
        actions.append(rec)
        state["log"].append(rec)

    # --- 1. Exits on currently-held positions ---
    for sym, pos in positions.items():
        if pos.get("side") != "long" or pos.get("qty", 0) <= 0:
            continue
        meta = state["positions"].get(sym)
        decision = evaluate_position_exit(
            sym, dsl, entry_price=pos["avg_entry_price"], current_price=pos["current_price"],
            meta=meta,
        )
        if not decision["exit"]:
            new_stop = decision.get("new_stop")
            if new_stop and meta is not None:
                actions.append(_raise_stop(broker_mod, sym, meta, float(new_stop)))
            continue
        stop = kill_switch_hit()
        if stop:
            actions.append({"symbol": sym, "action": "EXIT_SKIPPED", "reason": stop})
            continue
        try:
            _cancel_leftovers(broker_mod, sym)
            order = broker_mod.close_position(sym)
            state["positions"].pop(sym, None)
            state["orders_today"] += 1
            # Prefer the broker's reported realized P/L on the close; fall back
            # to the position's unrealized P/L at decision time as an ESTIMATE
            # (the actual fill may differ from the last quote).
            realized = order.get("realized_pl") if isinstance(order, dict) else None
            if realized is None:
                realized = pos["unrealized_pl"]  # estimate — see comment above
            if realized < 0:
                state["realized_loss_today"] += abs(realized)
            rec = {"symbol": sym, "action": "SELL", "reason": decision["reason"],
                   "unrealized_pl": round(pos["unrealized_pl"], 2),
                   "realized_pl": round(float(realized), 2), "order": order,
                   "ts": datetime.now(timezone.utc).isoformat()}
            actions.append(rec)
            state["log"].append(rec)
        except Exception as e:  # noqa: BLE001
            actions.append({"symbol": sym, "action": "SELL_FAILED", "reason": str(e)})

    # --- 2. Entries on un-held universe symbols ---
    for sym in universe:
        if sym.upper() in positions:
            continue
        stop = kill_switch_hit()
        if stop:
            actions.append({"symbol": sym.upper(), "action": "BUY_SKIPPED", "reason": stop})
            continue
        try:
            sig = latest_signal(sym, dsl, universe=universe, context={"source": "run", **(context or {})})
        except Exception as e:  # noqa: BLE001
            actions.append({"symbol": sym.upper(), "action": "ERROR", "reason": str(e)})
            continue
        if sig.get("action") != "BUY":
            # Surface an LLM veto / ML skip explicitly so the user sees *why* a
            # quant trigger didn't fire an order.
            if sig.get("llm", {}).get("decision") == "VETO":
                actions.append({"symbol": sym.upper(), "action": "VETOED",
                                "reason": sig.get("reason"), "llm": sig.get("llm")})
            elif sig.get("ml_filter", {}).get("pass") is False:
                actions.append({"symbol": sym.upper(), "action": "ML_SKIPPED",
                                "reason": sig.get("reason"), "ml_filter": sig.get("ml_filter")})
            continue
        # PDT guard — refuse new round-trip-eligible entries on a flagged or
        # sub-floor account before sizing anything.
        if pdt_block_reason:
            actions.append({"symbol": sym.upper(), "action": "PDT_BLOCKED",
                            "reason": pdt_block_reason})
            continue
        buying_power = account.get("buying_power", 0.0) if account else 0.0
        # Stack the LLM size multiplier with the top-down regime multiplier so a
        # hostile tape down-sizes every entry (defaults to 1.0 if unavailable).
        size_mult = float(sig.get("size_multiplier", 1.0) or 1.0) * regime_mult
        notional = min(max_pos * size_mult, buying_power)
        qty: int | None = None
        price = float(sig.get("price") or 0.0)
        if sig.get("stop_dist") and price > 0:
            # Swing: risk-based size in whole shares (brackets need whole shares).
            risk_pct = float(execution.get("risk_per_trade_pct", 1.0) or 1.0)
            by_risk = equity * risk_pct / 100.0 / float(sig["stop_dist"]) * size_mult
            qty = int(min(by_risk, notional / price))
            if qty < 1:
                actions.append({"symbol": sym.upper(), "action": "BUY_SKIPPED",
                                "reason": "risk budget / position cap buys less than one share"})
                continue
            notional = qty * price
        if notional < 1.0:
            actions.append({"symbol": sym.upper(), "action": "BUY_SKIPPED",
                            "reason": "insufficient buying power"})
            continue
        # Portfolio-level guards (gross exposure, open-position count, drawdown).
        pf_block = entry_blocked(notional)
        if pf_block:
            actions.append({"symbol": sym.upper(), "action": "BUY_SKIPPED", "reason": pf_block})
            continue
        try:
            order = _open_entry(broker_mod, sym, notional=notional, sig=sig,
                                dsl=dsl, execution=execution, qty=qty)
            state["orders_today"] += 1
            open_count += 1
            gross_exposure += notional
            state["positions"][sym.upper()] = _entry_record(sig, dsl, order)
            rec = {"symbol": sym.upper(), "action": "BUY", "reason": sig.get("reason"),
                   "notional": round(notional, 2), "signal_price": sig.get("price"),
                   "regime_mult": round(regime_mult, 3),
                   "llm": sig.get("llm"), "order": order,
                   "ts": datetime.now(timezone.utc).isoformat()}
            actions.append(rec)
            state["log"].append(rec)
        except Exception as e:  # noqa: BLE001
            actions.append({"symbol": sym.upper(), "action": "BUY_FAILED", "reason": str(e)})

    # Keep the persisted log bounded.
    state["log"] = state["log"][-100:]
    return {"blocked": None, "market_open": market_open, "actions": actions,
            "run_state": state, "account": account}


def execute_pass(*, universe: list[str], dsl: dict[str, Any] | None, execution: dict[str, Any] | None,
                 run_state: dict[str, Any] | None, context: dict[str, Any] | None = None,
                 alpaca=broker, sim_factory=None) -> dict[str, Any]:
    """One live pass on the bot's broker: Alpaca, or its private sim ledger.

    For a sim bot, resting stop/target legs are first settled against the
    intraday bars since the last run, then `run_live` runs unchanged against
    the ledger, which is written back into `run_state["sim_ledger"]`.
    """
    ex = {**DEFAULT_EXECUTION, **(execution or {})}
    if ex.get("broker") == "sim":
        from app.services import broker_sim
        state = dict(run_state or {})
        factory = sim_factory or broker_sim.SimBroker
        sim = factory(state.get("sim_ledger"), initial_cash=float(ex.get("sim_initial_cash") or broker_sim.DEFAULT_CASH))
        fills = sim.process_resting_orders()
        res = run_live(universe=universe, dsl=dsl, execution=execution, run_state=state,
                       broker_mod=sim, context=context)
        try:
            sim.mark()
        except Exception:  # noqa: BLE001 — a missing quote shouldn't lose the ledger
            pass
        res["run_state"]["sim_ledger"] = sim.ledger
        res["resting_fills"] = fills
        res["mode"] = "sim"
        return res
    res = run_live(universe=universe, dsl=dsl, execution=execution, run_state=run_state,
                   broker_mod=alpaca, context=context)
    res["mode"] = "paper" if alpaca.is_paper() else "live"
    return res


def _protective_prices(sig: dict[str, Any], dsl: dict[str, Any]) -> tuple[float | None, float | None]:
    """(stop, take_profit) absolute prices for a new long entry, if defined."""
    if sig.get("stop") is not None:  # swing signals carry absolute levels
        return float(sig["stop"]), (float(sig["target"]) if sig.get("target") else None)
    price = sig.get("price") or sig.get("signal_price")
    if not price or float(price) <= 0:
        return None, None
    stop_pct, tp_pct = dsl.get("stop_loss_pct"), dsl.get("take_profit_pct")
    stop = float(price) * (1.0 - float(stop_pct)) if stop_pct else None
    tp = float(price) * (1.0 + float(tp_pct)) if tp_pct else None
    return stop, tp


def _open_entry(broker_mod, sym: str, *, notional: float, sig: dict[str, Any],
                dsl: dict[str, Any], execution: dict[str, Any],
                qty: int | None = None) -> dict[str, Any]:
    """Open a long entry, attaching a broker-side protective bracket when possible.

    When the broker exposes `submit_bracket_order` and the strategy defines a
    stop, submit a market entry with a real broker-side stop-loss (+ take-
    profit), good-til-cancelled, so the stop is enforced even if the bot never
    re-runs, overnight, or through a gap. Alpaca only accepts child legs on a
    whole-share quantity, so a notional budget is converted to whole shares.
    Falls back to a plain market order — relying on the software exit check
    in `evaluate_position_exit` — when the broker lacks brackets, the config
    disables them, the budget buys < 1 share, or the bracket is rejected.
    The returned order carries `protective: "broker" | "software"`.
    """
    use_bracket = bool(execution.get("use_broker_bracket", True))
    stop_price, tp_price = _protective_prices(sig, dsl)
    price = float(sig.get("price") or sig.get("signal_price") or 0.0)
    whole = qty if qty is not None else (int(notional // price) if price > 0 else 0)
    if (use_bracket and stop_price and whole >= 1
            and hasattr(broker_mod, "submit_bracket_order")):
        try:
            order = broker_mod.submit_bracket_order(
                sym, qty=whole, side="buy",
                stop_loss_price=stop_price, take_profit_price=tp_price,
            )
            return {**order, "protective": "broker"}
        except Exception as e:  # noqa: BLE001 — degrade gracefully to a plain order
            fallback_reason = f"bracket rejected: {e}"
    else:
        fallback_reason = "no broker-side stop (brackets off, unsupported, or < 1 whole share)"
    if qty is not None:
        order = broker_mod.submit_market_order(sym, qty=qty, side="buy")
    else:
        order = broker_mod.submit_market_order(sym, notional=notional, side="buy")
    return {**order, "protective": "software", "protective_note": fallback_reason}


def _entry_record(sig: dict[str, Any], dsl: dict[str, Any], order: dict[str, Any]) -> dict[str, Any]:
    """What run_live remembers about a position it opened (JSON-safe)."""
    stop, target = _protective_prices(sig, dsl)
    stop_leg = next((g.get("id") for g in (order.get("legs") or [])
                     if str(g.get("type", "")).startswith("stop")), None)
    return {
        "entry_date": date.today().isoformat(),
        "kind": dsl.get("kind"),
        "signal_price": sig.get("price"),
        "stop": round(stop, 4) if stop else None,
        "initial_stop": round(stop, 4) if stop else None,
        "target": round(target, 4) if target else None,
        "stop_dist": sig.get("stop_dist"),
        "stop_order_id": stop_leg,
        "protective": order.get("protective"),
        "order_id": order.get("id"),
    }


def _cancel_leftovers(broker_mod, sym: str) -> None:
    """Cancel resting bracket legs so a close isn't blocked and no orphaned
    GTC stop can fire later and open a short. Best-effort."""
    if hasattr(broker_mod, "cancel_open_orders"):
        try:
            broker_mod.cancel_open_orders(sym)
        except Exception:  # noqa: BLE001
            pass


def _raise_stop(broker_mod, sym: str, meta: dict[str, Any], new_stop: float) -> dict[str, Any]:
    """Ratchet a position's stop up (never down); move the broker leg if any."""
    old = meta.get("stop")
    if old is not None and new_stop <= float(old):
        return {"symbol": sym, "action": "HOLD", "reason": "stop unchanged"}
    moved = "software"
    if meta.get("stop_order_id") and hasattr(broker_mod, "replace_stop"):
        try:
            broker_mod.replace_stop(meta["stop_order_id"], new_stop)
            moved = "broker"
        except Exception as e:  # noqa: BLE001 — keep the software stop authoritative
            moved = f"software (broker replace failed: {e})"
    meta["stop"] = round(new_stop, 4)
    return {"symbol": sym, "action": "STOP_RAISED", "reason": f"trailing stop → {new_stop:.2f}",
            "old_stop": old, "new_stop": round(new_stop, 4), "where": moved,
            "ts": datetime.now(timezone.utc).isoformat()}


def _safe_account(broker_mod) -> dict[str, Any] | None:
    try:
        return broker_mod.get_account()
    except Exception:  # noqa: BLE001
        return None
