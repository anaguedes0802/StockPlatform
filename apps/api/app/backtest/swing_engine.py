"""Portfolio-level swing-trading backtester.

What makes this different from `bot_engine.py` (one symbol, whole equity per
trade, trades pooled after the fact):

* **One shared account across a universe.** Positions compete for capital,
  slots and risk budget exactly as they would in a real account.
* **Risk-based sizing.** Each trade risks a fixed % of *current* equity between
  the fill and its initial stop (or equal-weight sizing for setups whose stop
  is not a risk definition). Capped by max position size, gross leverage and
  total open risk ("portfolio heat").
* **Daily-bar execution model, no look-ahead.** Signals are read at the close;
  entries fill at the next open. Stops and targets are resting orders checked
  against each bar's range. A bar that *opens* through a stop fills at the
  open (the overnight gap a stop cannot protect against). If a bar touches
  both the stop and the target, the stop is assumed to have hit first.
* **Costs.** Commission + slippage on every market fill (stops are market
  orders; targets are limit orders and fill at the limit). Optional overnight
  financing on notional for leveraged products (FX / CFDs).
* **Earnings.** With earnings dates supplied, positions are closed at the
  close of the last session before a report and new entries are refused when
  a report falls inside the blackout window.
* **Honest reporting.** R-multiples, a buy-and-hold benchmark, split-half
  stability, a trade-bootstrap Monte Carlo, and an explicit verdict with the
  checks it is based on.

P&L accounting is return-on-notional: `units × (exit − entry)` in the
instrument's price. That is exact for USD-priced assets and USD-quoted FX
pairs (EURUSD). For pairs quoted in another currency (USDJPY, EURGBP) the
error is the quote currency's move over the holding period — second-order for
multi-day holds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Signal frame columns every setup must provide (see app.services.swing).
SIGNAL_COLUMNS = (
    "long_entry", "short_entry",          # bool, evaluated at the bar's close
    "stop_dist_long", "stop_dist_short",  # price distance from fill to initial stop
    "exit_long", "exit_short",            # bool close-based exit conditions
    "atr",                                # for trailing stops
    "score_long", "score_short",          # ranking when signals exceed free slots
)


@dataclass
class SetupRules:
    """Exit/sizing behaviour shared by every symbol traded with one setup."""
    target_r: float | None = None          # take-profit at N × initial risk
    target_pct: float | None = None        # or at a fixed % from the fill
    trail_atr_mult: float | None = None    # chandelier: extreme since entry ∓ k·ATR
    max_hold_bars: int | None = None       # time stop
    signal_exit_fill: str = "next_open"    # "next_open" | "close"
    sizing: str = "risk"                   # "risk" | "equal"


@dataclass
class EngineConfig:
    initial_equity: float = 100_000.0
    risk_per_trade_pct: float = 1.0        # % of equity lost if the initial stop hits
    max_positions: int = 8
    max_position_pct: float = 20.0         # max notional per position, % of equity
    max_gross_leverage: float = 1.0        # Σ notional / equity (1.0 = cash account)
    max_heat_pct: float = 8.0              # Σ open risk to current stops, % of equity
    commission_bps: float = 1.0            # per side, on notional
    slippage_bps: float = 5.0              # per market fill, adverse
    financing_bps_per_year: float = 0.0    # on notional, per calendar day held
    allow_short: bool = False
    whole_units: bool = True               # whole shares (needed for broker brackets)
    min_notional: float = 100.0
    earnings_exit: bool = True
    earnings_blackout_days: int = 7        # calendar days after the signal
    start: str | None = None               # no entries before this date
    end: str | None = None


@dataclass
class Trade:
    symbol: str
    side: int
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    units: float
    bars_held: int
    pnl: float
    r_multiple: float
    return_pct: float
    exit_reason: str
    initial_risk: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("entry_price", "exit_price"):
            d[k] = round(d[k], 6)
        for k in ("pnl", "initial_risk"):
            d[k] = round(d[k], 2)
        d["r_multiple"] = round(d["r_multiple"], 3)
        d["return_pct"] = round(d["return_pct"], 3)
        d["units"] = round(d["units"], 4)
        d["side"] = "long" if self.side > 0 else "short"
        return d


@dataclass
class _Pos:
    symbol: str
    side: int
    units: float
    entry_price: float
    entry_t: int
    entry_date: pd.Timestamp
    stop: float
    initial_stop: float
    target: float | None
    extreme: float                 # highest high (long) / lowest low (short) since entry
    last_close: float
    last_t: int
    bars_held: int = 0
    costs: float = 0.0             # commissions + financing paid so far

    @property
    def init_risk(self) -> float:
        return abs(self.entry_price - self.initial_stop) * self.units


@dataclass
class SwingResult:
    equity_curve: list[dict[str, Any]]
    benchmark_curve: list[dict[str, Any]]
    trades: list[Trade]
    metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any] | None
    yearly: list[dict[str, Any]]
    halves: list[dict[str, Any]]
    monte_carlo: dict[str, Any]
    verdict: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Full-resolution daily equity (not serialised) for paired comparisons.
    daily_equity: pd.Series | None = None

    def as_dict(self, *, max_trades: int = 500) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "equity_curve": self.equity_curve,
            "benchmark_curve": self.benchmark_curve,
            "yearly": self.yearly,
            "halves": self.halves,
            "monte_carlo": self.monte_carlo,
            "verdict": self.verdict,
            "diagnostics": self.diagnostics,
            "trades": [t.as_dict() for t in self.trades[-max_trades:]],
            "n_trades_total": len(self.trades),
        }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _align(df: pd.DataFrame, sig: pd.DataFrame, cal: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    d = df.reindex(cal)
    s = sig.reindex(cal)
    out = {c: d[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")}
    out["has"] = ~np.isnan(out["close"])
    for c in SIGNAL_COLUMNS:
        if c.startswith(("long_", "short_", "exit_")):
            out[c] = s[c].astype(float).fillna(0.0).to_numpy() > 0
        else:
            out[c] = s[c].to_numpy(dtype=float)
    # Next session date of this symbol's own calendar (for "report before the
    # next session?" checks). Holidays are known in advance, so this isn't
    # look-ahead in any meaningful sense.
    own = df.index
    pos = np.searchsorted(own.values, cal.values, side="right")
    nxt = np.full(len(cal), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(own)
    nxt[ok] = own.values[pos[ok]]
    out["next_date"] = nxt
    return out


def _report_between(dates: np.ndarray | None, lo: np.datetime64, hi: np.datetime64) -> bool:
    """True if an earnings date falls in (lo, hi]."""
    if dates is None or len(dates) == 0 or np.isnat(hi):
        return False
    i = np.searchsorted(dates, lo, side="right")
    return i < len(dates) and dates[i] <= hi


def simulate(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    rules: SetupRules,
    cfg: EngineConfig,
    *,
    earnings: dict[str, list[pd.Timestamp] | None] | None = None,
    long_regime: pd.Series | None = None,
    benchmark: pd.DataFrame | None = None,
    benchmark_symbol: str | None = None,
    entry_filter: dict[str, pd.Series] | None = None,
    mc_sims: int = 2000,
    seed: int = 7,
) -> SwingResult:
    syms = [s for s in data if s in signals and not data[s].empty]
    if not syms:
        raise ValueError("no symbols with data")
    cal = pd.DatetimeIndex(sorted(set().union(*[data[s].index for s in syms])))
    if cfg.end:
        cal = cal[cal <= pd.Timestamp(cfg.end, tz="UTC")]
    A = {s: _align(data[s], signals[s], cal) for s in syms}
    earn = {}
    for s in syms:
        dts = (earnings or {}).get(s)
        earn[s] = (np.array(sorted(pd.DatetimeIndex(dts).tz_convert("UTC").tz_localize(None)),
                            dtype="datetime64[ns]") if dts else None)
    cal_np = cal.tz_localize(None).values
    regime = (long_regime.reindex(cal).ffill().fillna(False).to_numpy(dtype=bool)
              if long_regime is not None else None)
    # Optional per-signal veto (e.g. the ML signal filter): True = allowed.
    # A signal with no entry in the filter is allowed but counted, so a
    # coverage gap shows up in diagnostics instead of silently changing trades.
    allow: dict[str, np.ndarray] = {}
    allow_known: dict[str, np.ndarray] = {}
    if entry_filter is not None:
        for s in syms:
            f = entry_filter.get(s)
            if f is None or f.empty:
                allow_known[s] = np.zeros(len(cal), dtype=bool)
                allow[s] = np.ones(len(cal), dtype=bool)
                continue
            f = f[~f.index.duplicated(keep="last")].reindex(cal)
            allow_known[s] = f.notna().to_numpy()
            allow[s] = f.astype(float).fillna(1.0).to_numpy() > 0

    # First bar on which any symbol has a usable signal row (indicators warm).
    ready = np.zeros(len(cal), dtype=bool)
    for s in syms:
        ready |= np.isfinite(A[s]["stop_dist_long"]) | np.isfinite(A[s]["stop_dist_short"])
    if not ready.any():
        raise ValueError("not enough history to warm up the indicators")
    t0 = int(np.argmax(ready))
    if cfg.start:
        t0 = max(t0, int(np.searchsorted(cal_np, np.datetime64(pd.Timestamp(cfg.start).tz_localize(None)))))
    if t0 >= len(cal) - 2:
        raise ValueError("backtest window is empty after warm-up")

    comm = cfg.commission_bps / 1e4
    slip = cfg.slippage_bps / 1e4
    fin_daily = cfg.financing_bps_per_year / 1e4 / 365.0

    cash = cfg.initial_equity          # realised equity
    positions: dict[str, _Pos] = {}
    pending_entries: list[tuple[str, int, float, float]] = []   # sym, side, stop_dist, score
    pending_exits: dict[str, str] = {}
    trades: list[Trade] = []
    equity = np.full(len(cal), np.nan)
    gross_hist = np.zeros(len(cal))
    # Why queued signals did not become trades. "max_positions" etc. name the
    # limit that bound; persistent setups (a breakout stays valid for days)
    # re-queue daily, so these count signal-days, not distinct opportunities.
    skipped = {"max_positions": 0, "gross_leverage": 0, "portfolio_heat": 0,
               "max_position_size": 0, "min_size": 0}
    diag = {"signals": 0, "skipped_earnings": 0, "skipped_regime": 0, "earnings_exits": 0,
            "skipped_filter": 0, "filter_missing": 0,
            "skipped_at_fill": skipped,
            "symbols": len(syms),
            "earnings_coverage": sum(1 for s in syms if earn[s] is not None)}

    def unrealized() -> float:
        return sum(p.side * p.units * (p.last_close - p.entry_price) for p in positions.values())

    def close_pos(p: _Pos, px: float, t: int, reason: str, *, market: bool) -> None:
        nonlocal cash
        fill = px * (1 - p.side * slip) if market else px
        c = abs(p.units * fill) * comm
        gross = p.side * p.units * (fill - p.entry_price)
        cash += gross - c
        p.costs += c
        pnl = gross - p.costs
        risk = p.init_risk
        trades.append(Trade(
            symbol=p.symbol, side=p.side,
            entry_date=p.entry_date.date().isoformat(), entry_price=p.entry_price,
            exit_date=cal[t].date().isoformat(), exit_price=fill, units=p.units,
            bars_held=p.bars_held, pnl=pnl,
            r_multiple=(pnl / risk) if risk > 0 else 0.0,
            return_pct=pnl / abs(p.units * p.entry_price) * 100 if p.units else 0.0,
            exit_reason=reason, initial_risk=risk,
        ))
        positions.pop(p.symbol, None)
        pending_exits.pop(p.symbol, None)

    for t in range(t0, len(cal)):
        eq_prev = cash + unrealized()

        # ---- 1. exits queued at the previous close fill at this open ----
        for sym in list(pending_exits):
            a = A[sym]
            if a["has"][t] and sym in positions:
                close_pos(positions[sym], a["open"][t], t, pending_exits[sym], market=True)

        # ---- 2. entries queued at the previous close fill at this open ----
        if pending_entries:
            eq_now = cash + unrealized()
            for sym, side, dist, _score in sorted(pending_entries, key=lambda x: -x[3]):
                a = A[sym]
                if sym in positions or not a["has"][t]:
                    continue
                if len(positions) >= cfg.max_positions:
                    skipped["max_positions"] += 1
                    continue
                fill = a["open"][t] * (1 + side * slip)
                if not (np.isfinite(fill) and fill > 0 and np.isfinite(dist) and dist > 0):
                    continue
                if side < 0 and dist >= fill * 0.9:
                    continue
                gross_now = sum(abs(p.units * p.last_close) for p in positions.values())
                heat_now = sum(max(0.0, p.side * (p.last_close - p.stop)) * p.units
                               for p in positions.values())
                limits = {
                    "target": (eq_now / cfg.max_positions / fill if rules.sizing == "equal"
                               else eq_now * cfg.risk_per_trade_pct / 100.0 / dist),
                    "max_position_size": eq_now * cfg.max_position_pct / 100.0 / fill,
                    "gross_leverage": max(0.0, cfg.max_gross_leverage * eq_now - gross_now) / fill,
                    "portfolio_heat": max(0.0, eq_now * cfg.max_heat_pct / 100.0 - heat_now) / dist,
                }
                binding = min(limits, key=limits.get)
                units = limits[binding]
                if cfg.whole_units:
                    units = math.floor(units)
                if units <= 0 or units * fill < cfg.min_notional:
                    skipped[binding if binding != "target" else "min_size"] += 1
                    continue
                stop = fill - side * dist
                if rules.target_r:
                    target = fill + side * rules.target_r * dist
                elif rules.target_pct:
                    target = fill * (1 + side * rules.target_pct)
                else:
                    target = None
                c = units * fill * comm
                cash -= c
                positions[sym] = _Pos(
                    symbol=sym, side=side, units=units, entry_price=fill, entry_t=t,
                    entry_date=cal[t], stop=stop, initial_stop=stop, target=target,
                    extreme=fill, last_close=fill, last_t=t, costs=c,
                )
            pending_entries = []

        # ---- 3. resting stop / target orders against this bar's range ----
        for sym, p in list(positions.items()):
            a = A[sym]
            if not a["has"][t]:
                continue
            o, h, l = a["open"][t], a["high"][t], a["low"][t]
            trailed = p.stop != p.initial_stop
            if p.side > 0:
                if l <= p.stop:
                    close_pos(p, min(o, p.stop), t, "trail_stop" if trailed else "stop", market=True)
                elif p.target is not None and h >= p.target:
                    close_pos(p, max(o, p.target), t, "target", market=False)
            else:
                if h >= p.stop:
                    close_pos(p, max(o, p.stop), t, "trail_stop" if trailed else "stop", market=True)
                elif p.target is not None and l <= p.target:
                    close_pos(p, min(o, p.target), t, "target", market=False)

        # ---- 4. at the close: bookkeeping, close-based exits, trailing ----
        for sym, p in list(positions.items()):
            a = A[sym]
            if not a["has"][t]:
                continue
            c_px = a["close"][t]
            if fin_daily:
                days = max(1, int((cal_np[t] - cal_np[p.last_t]) / np.timedelta64(1, "D")))
                f = abs(p.units * p.last_close) * fin_daily * days
                cash -= f
                p.costs += f
            p.last_close, p.last_t = c_px, t
            p.bars_held += 1
            if (cfg.earnings_exit
                    and _report_between(earn[sym], cal_np[t], a["next_date"][t])):
                diag["earnings_exits"] += 1
                close_pos(p, c_px, t, "earnings", market=True)
                continue
            exit_now = a["exit_long"][t] if p.side > 0 else a["exit_short"][t]
            timed_out = rules.max_hold_bars is not None and p.bars_held >= rules.max_hold_bars
            if exit_now or timed_out:
                reason = "signal" if exit_now else "time"
                if rules.signal_exit_fill == "close":
                    close_pos(p, c_px, t, reason, market=True)
                    continue
                pending_exits[sym] = reason
            if rules.trail_atr_mult and np.isfinite(a["atr"][t]):
                if p.side > 0:
                    p.extreme = max(p.extreme, a["high"][t])
                    p.stop = max(p.stop, p.extreme - rules.trail_atr_mult * a["atr"][t])
                else:
                    p.extreme = min(p.extreme, a["low"][t])
                    p.stop = min(p.stop, p.extreme + rules.trail_atr_mult * a["atr"][t])

        equity[t] = cash + unrealized()
        gross_hist[t] = sum(abs(p.units * p.last_close) for p in positions.values())

        # ---- 5. new signals at the close → fill next open ----
        if t == len(cal) - 1:
            break
        for sym in syms:
            a = A[sym]
            if not a["has"][t] or sym in positions or sym in pending_exits:
                continue
            side = 0
            if a["long_entry"][t]:
                side = 1
            elif cfg.allow_short and a["short_entry"][t]:
                side = -1
            if not side:
                continue
            diag["signals"] += 1
            if side > 0 and regime is not None and not regime[t]:
                diag["skipped_regime"] += 1
                continue
            if entry_filter is not None:
                if not allow_known[sym][t]:
                    diag["filter_missing"] += 1
                elif not allow[sym][t]:
                    diag["skipped_filter"] += 1
                    continue
            if cfg.earnings_blackout_days and _report_between(
                earn[sym], cal_np[t], cal_np[t] + np.timedelta64(cfg.earnings_blackout_days, "D")
            ):
                diag["skipped_earnings"] += 1
                continue
            dist = a["stop_dist_long"][t] if side > 0 else a["stop_dist_short"][t]
            score = a["score_long"][t] if side > 0 else a["score_short"][t]
            pending_entries.append((sym, side, float(dist), float(score) if np.isfinite(score) else -1e9))

    # Close anything still open at the final close so every trade is counted.
    last_t = len(cal) - 1
    for sym, p in list(positions.items()):
        close_pos(p, p.last_close, p.last_t if p.last_t <= last_t else last_t, "end_of_data", market=True)
    equity[last_t] = cash

    eq = pd.Series(equity[t0:], index=cal[t0:]).ffill()
    gross = pd.Series(gross_hist[t0:], index=cal[t0:])
    diag["avg_gross_leverage"] = round(float((gross / eq).mean()), 3)
    diag["time_in_market_pct"] = round(float((gross > 0).mean() * 100), 1)

    bench_eq = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark["close"].reindex(cal[t0:]).ffill().bfill()
        if b.notna().all() and b.iloc[0] > 0:
            bench_eq = b / b.iloc[0] * cfg.initial_equity

    res = _report(eq, bench_eq, trades, cfg, diag, benchmark_symbol, mc_sims, seed)
    res.daily_equity = eq
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _curve_metrics(eq: pd.Series) -> dict[str, Any]:
    eq = eq.dropna()
    if len(eq) < 3:
        return {}
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    sd = rets.std()
    sd = float(sd)
    sharpe = float(np.sqrt(252) * rets.mean() / sd) if sd > 0 else 0.0
    dn = rets[rets < 0].std()
    sortino = float(np.sqrt(252) * rets.mean() / dn) if dn and dn > 0 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    return {
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "years": round(years, 2),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_vol_pct": round(float(sd * np.sqrt(252) * 100), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else None,
        "sharpe_t_stat": round(sharpe * math.sqrt(years), 2),
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


def _trade_metrics(trades: list[Trade]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    sd = r.std(ddof=1) if n > 1 else 0.0
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {
        "n_trades": n,
        "n_long": int(sum(1 for t in trades if t.side > 0)),
        "n_short": int(sum(1 for t in trades if t.side < 0)),
        "win_rate_pct": round(len(wins) / n * 100, 2),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "expectancy_r": round(float(r.mean()), 4),
        "median_r": round(float(np.median(r)), 3),
        "best_r": round(float(r.max()), 2),
        "worst_r": round(float(r.min()), 2),
        "trade_t_stat": round(float(r.mean() / sd * math.sqrt(n)), 2) if sd > 0 else None,
        "profit_factor": round(float(gw / gl), 3) if gl > 0 else None,
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 1),
        "exit_reasons": reasons,
    }


def _monte_carlo(trades: list[Trade], sims: int, seed: int) -> dict[str, Any]:
    """Bootstrap the trade R-multiples: how much of the result is sequence luck?"""
    r = np.array([t.r_multiple for t in trades])
    n = len(r)
    if n < 20 or sims <= 0:
        return {"n_sims": 0, "note": "fewer than 20 trades — not enough to resample"}
    rng = np.random.default_rng(seed)
    draws = r[rng.integers(0, n, size=(sims, n))]
    total = draws.sum(axis=1)
    cum = np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((sims, 1)), cum], axis=1), axis=1)[:, 1:]
    mdd = (cum - peak).min(axis=1)
    return {
        "n_sims": sims,
        "n_trades": n,
        "total_r_p5": round(float(np.percentile(total, 5)), 1),
        "total_r_p50": round(float(np.percentile(total, 50)), 1),
        "total_r_p95": round(float(np.percentile(total, 95)), 1),
        "max_dd_r_p50": round(float(np.percentile(mdd, 50)), 1),
        "max_dd_r_p95": round(float(np.percentile(mdd, 5)), 1),  # 95th-worst
        "prob_total_r_le_0_pct": round(float((total <= 0).mean() * 100), 1),
    }


def _yearly(eq: pd.Series, bench: pd.Series | None) -> list[dict[str, Any]]:
    out = []
    ye = eq.groupby(eq.index.year).agg(["first", "last"])
    prev_last = None
    yb = bench.groupby(bench.index.year).agg(["first", "last"]) if bench is not None else None
    prev_b = None
    for yr, row in ye.iterrows():
        base = prev_last if prev_last is not None else row["first"]
        rec = {"year": int(yr), "strategy_pct": round(float(row["last"] / base - 1) * 100, 2)}
        if yb is not None and yr in yb.index:
            bbase = prev_b if prev_b is not None else yb.loc[yr, "first"]
            rec["benchmark_pct"] = round(float(yb.loc[yr, "last"] / bbase - 1) * 100, 2)
            prev_b = yb.loc[yr, "last"]
        out.append(rec)
        prev_last = row["last"]
    return out


def _halves(eq: pd.Series, trades: list[Trade]) -> list[dict[str, Any]]:
    mid = eq.index[0] + (eq.index[-1] - eq.index[0]) / 2
    out = []
    for label, part in (("first_half", eq[eq.index <= mid]), ("second_half", eq[eq.index > mid])):
        if len(part) < 3:
            continue
        lo, hi = part.index[0].date().isoformat(), part.index[-1].date().isoformat()
        tr = [t for t in trades if lo <= t.exit_date <= hi]
        cm = _curve_metrics(part)
        tm = _trade_metrics(tr)
        out.append({
            "label": label, "start": lo, "end": hi,
            "cagr_pct": cm.get("cagr_pct"), "sharpe": cm.get("sharpe"),
            "max_drawdown_pct": cm.get("max_drawdown_pct"),
            "n_trades": tm.get("n_trades", 0), "expectancy_r": tm.get("expectancy_r"),
        })
    return out


def verdict(metrics: dict[str, Any], bench: dict[str, Any] | None,
            halves: list[dict[str, Any]], mc: dict[str, Any]) -> dict[str, Any]:
    """Plain-language edge assessment from explicit, pre-declared checks.

    A historical edge is necessary, not sufficient, for future profit: the
    universe is chosen with hindsight, costs drift, and every extra variant
    tested raises the odds that a good-looking result is luck.
    """
    checks = []

    def add(key: str, ok: bool | None, detail: str) -> None:
        checks.append({"key": key, "pass": ok, "detail": detail})

    n = metrics.get("n_trades", 0)
    exp_r = metrics.get("expectancy_r")
    add("sample_size", n >= 100,
        f"{n} trades (want ≥ 100 for the averages to mean much)")
    add("positive_expectancy", exp_r is not None and exp_r > 0,
        f"average trade {exp_r:+.3f}R after costs" if exp_r is not None else "no trades")
    st = metrics.get("sharpe_t_stat")
    add("significant", st is not None and st >= 2.0,
        f"Sharpe t-stat {st} (≥ 2 ≈ unlikely to be pure noise)" if st is not None else "n/a")
    stable = bool(halves) and all((h.get("expectancy_r") or 0) > 0 and (h.get("cagr_pct") or 0) > 0
                                   for h in halves)
    add("stable_across_halves", stable,
        "profitable in both halves of the sample" if stable else
        "at least one half of the sample lost money or had negative expectancy")
    p_loss = mc.get("prob_total_r_le_0_pct")
    add("robust_to_sequence", p_loss is not None and p_loss < 5.0,
        f"{p_loss}% of resampled trade sequences end ≤ 0R" if p_loss is not None else "too few trades")
    if bench:
        s, b = metrics.get("sharpe") or 0.0, bench.get("sharpe") or 0.0
        add("beats_benchmark_risk_adjusted", s >= b,
            f"Sharpe {s:.2f} vs buy-and-hold {b:.2f}; CAGR {metrics.get('cagr_pct')}% vs "
            f"{bench.get('cagr_pct')}%, max DD {metrics.get('max_drawdown_pct')}% vs "
            f"{bench.get('max_drawdown_pct')}%")

    core = {c["key"]: c["pass"] for c in checks}
    robust = all(core.get(k) for k in ("sample_size", "positive_expectancy", "significant",
                                       "stable_across_halves", "robust_to_sequence"))
    if not core.get("positive_expectancy") or (st is not None and st < 1.0):
        label, summary = "no_edge", "No evidence of an edge after costs."
    elif not robust:
        label, summary = "inconclusive", (
            "Positive, but not convincingly distinguishable from luck.")
    elif bench and core.get("beats_benchmark_risk_adjusted") is False:
        label, summary = "edge_below_benchmark", (
            "A real but weak edge: it made money consistently, yet simply holding the "
            "benchmark earned more per unit of risk. Its value, if any, is as a "
            "lower-drawdown or diversifying sleeve, not as a replacement.")
    else:
        label, summary = "historical_edge", (
            "Historically profitable, statistically distinguishable from noise, and at "
            "least as good as the benchmark per unit of risk. Necessary, not sufficient: "
            "paper-trade it before trusting it.")
    return {"label": label, "summary": summary, "checks": checks}


def _report(eq: pd.Series, bench_eq: pd.Series | None, trades: list[Trade], cfg: EngineConfig,
            diag: dict[str, Any], benchmark_symbol: str | None, mc_sims: int, seed: int) -> SwingResult:
    cm = _curve_metrics(eq)
    tm = _trade_metrics(trades)
    metrics = {**cm, **tm}
    bm = _curve_metrics(bench_eq) if bench_eq is not None else None
    if bm is not None:
        bm["symbol"] = benchmark_symbol
        corr = eq.pct_change().corr(bench_eq.pct_change())
        metrics["correlation_to_benchmark"] = round(float(corr), 3) if np.isfinite(corr) else None
    halves = _halves(eq, trades)
    mc = _monte_carlo(trades, mc_sims, seed)

    def thin(s: pd.Series | None) -> list[dict[str, Any]]:
        if s is None:
            return []
        step = max(1, len(s) // 600)  # keep payloads chart-sized
        idx = list(range(0, len(s), step))
        if idx[-1] != len(s) - 1:
            idx.append(len(s) - 1)
        return [{"ts": s.index[i].date().isoformat(), "equity": round(float(s.iloc[i]), 2)} for i in idx]

    return SwingResult(
        equity_curve=thin(eq),
        benchmark_curve=thin(bench_eq),
        trades=trades,
        metrics=metrics,
        benchmark_metrics=bm,
        yearly=_yearly(eq, bench_eq),
        halves=halves,
        monte_carlo=mc,
        verdict=verdict(metrics, bm, halves, mc),
        diagnostics={**diag, "config": asdict(cfg)},
    )
