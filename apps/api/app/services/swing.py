"""Swing trading: setups, universes, portfolio backtests, scanner, live helpers.

Holding period: days to a few weeks, decided on daily closes, executed at the
next session's open. Every setup here is a textbook rule set with **fixed,
pre-declared parameters** — none were tuned on the data they are reported on.
That matters more than any single number: a strategy searched into shape on
one history usually fails on the next.

Setups
------
* ``breakout`` — Donchian trend breakout (Turtle "System 2" family). Enter on a
  close beyond the prior 55-day extreme in the direction of the 200-day trend;
  initial stop 2×ATR; exit on a 20-day opposite extreme or a 3×ATR chandelier
  trailing stop. Low win rate, fat right tail.
* ``pullback`` — trend pullback continuation. In an established trend
  (close and SMA50 on the same side of SMA200), wait for a dip into the
  20-EMA, then enter when price resumes (close beyond the prior bar's extreme).
  Stop under the 5-bar swing extreme (≥ 1×ATR); target 2R; exit on a close
  back through SMA50 or after 15 bars.
* ``rsi2`` — the platform's existing trend-filtered RSI(2) mean reversion,
  with its own (bot) parameters, run through the same portfolio engine so it
  can be compared on equal terms.

Live trading is long-only US equities/ETFs through Alpaca (the platform's
broker). FX and shorts are research-only here: Alpaca does not offer spot FX.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.backtest import swing_engine as eng
from app.services import indicators as ind
from app.services import swing_data as sd

# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------

SETUPS: dict[str, dict[str, Any]] = {
    "breakout": {
        "label": "Trend breakout (Donchian 55/20)",
        "summary": "Buy a close above the 55-day high in a 200-day uptrend. 2×ATR initial "
                   "stop, 3×ATR chandelier trail, exit on a 20-day low. Few winners, big ones.",
        "params": {"entry_lookback": 55, "exit_lookback": 20, "trend_sma": 200,
                   "atr_period": 20, "stop_atr": 2.0, "trail_atr": 3.0, "rank_lookback": 126},
    },
    "pullback": {
        "label": "Trend pullback (20-EMA dip, 2R target)",
        "summary": "In an uptrend (close and SMA50 above SMA200), buy the first close above the "
                   "prior high after a dip into the 20-EMA. Stop under the 5-bar low, target 2R, "
                   "exit on a close below SMA50 or after 15 bars.",
        "params": {"fast_sma": 50, "trend_sma": 200, "ema": 20, "touch_lookback": 3,
                   "swing_lookback": 5, "atr_period": 14, "stop_buffer_atr": 0.25,
                   "min_stop_atr": 1.0, "target_r": 2.0, "max_hold_bars": 15,
                   "rank_lookback": 126},
    },
    "rsi2": {
        "label": "RSI(2) mean reversion (current bot rules)",
        "summary": "Buy RSI(2) < 10 above SMA200 while SPY is above its SMA200. +1% target, "
                   "exit on a close above SMA20, −25% stop, 40-bar time stop. Equal-weight "
                   "sizing because a −25% stop is not a risk budget.",
        "params": {"rsi_period": 2, "rsi_entry": 10.0, "trend_sma": 200, "exit_sma": 20,
                   "take_profit_pct": 0.01, "stop_loss_pct": 0.25, "max_hold_bars": 40,
                   "atr_period": 14, "market_filter": True, "market_proxy": "SPY",
                   "market_sma": 200},
    },
}

# Bot DSL kinds that route to this module (see trading_bot.is_swing).
BOT_KINDS = {"swing_breakout": "breakout", "swing_pullback": "pullback"}


def _p(setup: str, params: dict[str, Any] | None) -> dict[str, Any]:
    if setup not in SETUPS:
        raise ValueError(f"unknown swing setup: {setup}")
    return {**SETUPS[setup]["params"], **(params or {})}


def build_signals(df: pd.DataFrame, setup: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-bar signal frame (columns = swing_engine.SIGNAL_COLUMNS).

    Everything is computed from bars up to and including the current close —
    rolling extremes are shifted one bar so "breaks the prior 55-day high"
    never compares a bar against itself.
    """
    p = _p(setup, params)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atr = ind.atr(h, l, c, int(p["atr_period"]))
    nan = pd.Series(np.nan, index=df.index)
    false = pd.Series(False, index=df.index)

    if setup == "breakout":
        n_in, n_out = int(p["entry_lookback"]), int(p["exit_lookback"])
        trend = ind.sma(c, int(p["trend_sma"]))
        hh_in, ll_in = h.rolling(n_in).max().shift(1), l.rolling(n_in).min().shift(1)
        hh_out, ll_out = h.rolling(n_out).max().shift(1), l.rolling(n_out).min().shift(1)
        roc = c / c.shift(int(p["rank_lookback"])) - 1
        out = pd.DataFrame({
            "long_entry": (c > hh_in) & (c > trend),
            "short_entry": (c < ll_in) & (c < trend),
            "stop_dist_long": float(p["stop_atr"]) * atr,
            "stop_dist_short": float(p["stop_atr"]) * atr,
            "exit_long": c < ll_out,
            "exit_short": c > hh_out,
            "atr": atr,
            "score_long": roc,
            "score_short": -roc,
        })
        ready = trend.notna() & hh_in.notna() & atr.notna()
    elif setup == "pullback":
        fast, slow = ind.sma(c, int(p["fast_sma"])), ind.sma(c, int(p["trend_sma"]))
        ema = ind.ema(c, int(p["ema"]))
        k, sw = int(p["touch_lookback"]), int(p["swing_lookback"])
        buf, min_stop = float(p["stop_buffer_atr"]), float(p["min_stop_atr"])
        up = (c > slow) & (fast > slow)
        down = (c < slow) & (fast < slow)
        long_trig = up & (l.rolling(k).min() <= ema) & (c > h.shift(1)) & (c > o)
        short_trig = down & (h.rolling(k).max() >= ema) & (c < l.shift(1)) & (c < o)
        dist_l = np.maximum(c - (l.rolling(sw).min() - buf * atr), min_stop * atr)
        dist_s = np.maximum((h.rolling(sw).max() + buf * atr) - c, min_stop * atr)
        roc = c / c.shift(int(p["rank_lookback"])) - 1
        out = pd.DataFrame({
            "long_entry": long_trig, "short_entry": short_trig,
            "stop_dist_long": dist_l, "stop_dist_short": dist_s,
            "exit_long": c < fast, "exit_short": c > fast,
            "atr": atr, "score_long": roc, "score_short": -roc,
        })
        ready = slow.notna() & fast.notna() & atr.notna()
    elif setup == "rsi2":
        r = ind.rsi(c, int(p["rsi_period"]))
        trend = ind.sma(c, int(p["trend_sma"]))
        out = pd.DataFrame({
            "long_entry": (r < float(p["rsi_entry"])) & (c > trend),
            "short_entry": false,
            "stop_dist_long": float(p["stop_loss_pct"]) * c,
            "stop_dist_short": nan,
            "exit_long": c > ind.sma(c, int(p["exit_sma"])),
            "exit_short": false,
            "atr": atr, "score_long": -r, "score_short": nan,
        })
        ready = trend.notna() & r.notna()
    else:  # pragma: no cover — _p already validated
        raise ValueError(setup)

    out.loc[~ready, ["long_entry", "short_entry"]] = False
    out.loc[~ready, ["stop_dist_long", "stop_dist_short"]] = np.nan
    for col in ("long_entry", "short_entry", "exit_long", "exit_short"):
        out[col] = out[col].fillna(False).astype(bool)
    return out


def setup_rules(setup: str, params: dict[str, Any] | None = None) -> eng.SetupRules:
    p = _p(setup, params)
    if setup == "breakout":
        return eng.SetupRules(trail_atr_mult=float(p["trail_atr"]))
    if setup == "pullback":
        return eng.SetupRules(target_r=float(p["target_r"]), max_hold_bars=int(p["max_hold_bars"]))
    return eng.SetupRules(target_pct=float(p["take_profit_pct"]),
                          max_hold_bars=int(p["max_hold_bars"]),
                          signal_exit_fill="close", sizing="equal")


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

UNIVERSES: dict[str, dict[str, Any]] = {
    "etfs": {
        "label": "Index, sector & cross-asset ETFs",
        "note": "Low survivorship bias: broad funds rarely die. Starts when enough funds exist.",
        "asset_class": "equity",
        "benchmark": "SPY",
        "symbols": ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLB", "XLE", "XLF", "XLI", "XLK",
                    "XLP", "XLU", "XLV", "XLY", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ",
                    "TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ", "HYG"],
    },
    "us_large_caps": {
        "label": "US large caps (today's leaders)",
        "note": "SURVIVORSHIP-BIASED: these are today's large caps, so the backtest never "
                "holds the ones that shrank or died. Treat results as an upper bound.",
        "asset_class": "equity",
        "benchmark": "SPY",
        "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
                    "JNJ", "PFE", "MRK", "UNH", "ABT", "XOM", "CVX", "COP", "PG", "KO",
                    "PEP", "WMT", "HD", "MCD", "NKE", "DIS", "CMCSA", "VZ", "T", "INTC",
                    "CSCO", "ORCL", "IBM", "TXN", "QCOM", "ADBE", "CRM", "CAT", "DE", "BA",
                    "HON", "GE", "MMM", "UPS", "LMT", "UNP", "AMGN", "GILD", "LLY", "COST"],
    },
    "fx_majors": {
        "label": "FX majors & crosses (research only)",
        "note": "Yahoo indicative daily bars. Costs: 2 bps/side spread+slippage and 1%/yr "
                "financing markup; the interest-rate carry itself is not modelled. Not "
                "tradable through this platform's broker.",
        "asset_class": "fx",
        "benchmark": None,
        "symbols": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
                    "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X", "AUDJPY=X", "EURCHF=X"],
    },
}


def config_for(asset_class: str, overrides: dict[str, Any] | None = None) -> eng.EngineConfig:
    """Engine defaults per asset class; `overrides` wins."""
    if asset_class == "fx":
        base = eng.EngineConfig(
            commission_bps=0.0, slippage_bps=2.0, financing_bps_per_year=100.0,
            allow_short=True, whole_units=False, max_gross_leverage=4.0,
            max_position_pct=150.0, earnings_exit=False, earnings_blackout_days=0,
        )
    else:
        base = eng.EngineConfig()
    for k, v in (overrides or {}).items():
        if v is not None and hasattr(base, k):
            setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# Backtest orchestration
# ---------------------------------------------------------------------------

def resolve_universe(universe: str | None, symbols: list[str] | None) -> tuple[list[str], str, str | None, dict]:
    """(symbols, asset_class, benchmark_symbol, universe_meta) for a request."""
    if symbols:
        syms = [s.upper() for s in symbols]
        classes = {sd.asset_class(s) for s in syms}
        if len(classes) > 1:
            raise ValueError("mix of asset classes — backtest FX and equities separately")
        aclass = classes.pop()
        return syms, aclass, ("SPY" if aclass == "equity" else None), {
            "label": "custom", "note": "custom symbol list", "asset_class": aclass}
    key = universe or "etfs"
    if key not in UNIVERSES:
        raise ValueError(f"unknown universe: {key}")
    meta = UNIVERSES[key]
    return list(meta["symbols"]), meta["asset_class"], meta.get("benchmark"), meta


def prepare(
    *,
    setup: str,
    universe: str | None = None,
    symbols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    start: str | None = "2005-01-01",
    end: str | None = None,
    use_earnings: bool = True,
    max_age_s: int | None = None,
) -> dict[str, Any]:
    """Load data and build everything a simulation needs (shared by the plain
    backtest and the ML signal-filter evaluation so both see identical inputs)."""
    syms, aclass, bench_sym, uni_meta = resolve_universe(universe, symbols)
    p = _p(setup, params)
    # Load a year of extra history so the 200-day indicators are warm at `start`.
    load_from = (pd.Timestamp(start) - pd.Timedelta(days=400)).date().isoformat() if start else None
    if max_age_s is None:
        data, errors = sd.load_universe(syms, start=load_from)
    else:
        data, errors = {}, {}
        for s in syms:
            df = sd.daily_bars(s, start=load_from, max_age_s=max_age_s)
            if df.empty:
                errors[s] = "no data"
            else:
                data[s] = df
    if not data:
        raise ValueError("no market data could be loaded for this universe")
    # Only finished sessions: today's still-forming bar changes minute to
    # minute and would make two runs on the same day disagree.
    data = {s: completed_bars(df, s) for s, df in data.items()}
    signals = {s: build_signals(df, setup, p) for s, df in data.items()}
    cfg = config_for(aclass, {**(config or {}), "start": start, "end": end})
    if setup == "rsi2":
        cfg.allow_short = False  # mean-reversion rules are long-only

    def _aux(sym: str) -> pd.DataFrame:
        return data[sym] if sym in data else completed_bars(sd.daily_bars(sym, start=load_from), sym)

    bench_df = _aux(bench_sym) if bench_sym else None
    regime = None
    if setup == "rsi2" and p.get("market_filter") and aclass == "equity":
        proxy = _aux(p["market_proxy"])
        if proxy is not None and not proxy.empty:
            regime = proxy["close"] > ind.sma(proxy["close"], int(p["market_sma"]))
    earnings = sd.earnings_for(list(data)) if (use_earnings and aclass == "equity") else None
    return {
        "setup": setup, "params": p, "rules": setup_rules(setup, p), "cfg": cfg,
        "data": data, "errors": errors, "signals": signals, "earnings": earnings,
        "regime": regime, "benchmark": bench_df, "benchmark_symbol": bench_sym,
        "asset_class": aclass, "universe": universe if not symbols else "custom",
        "universe_meta": uni_meta, "load_from": load_from,
    }


def backtest(
    *,
    setup: str,
    universe: str | None = None,
    symbols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    start: str | None = "2005-01-01",
    end: str | None = None,
    use_earnings: bool = True,
    max_trades: int = 500,
) -> dict[str, Any]:
    """Run a portfolio swing backtest and return a JSON-ready report."""
    ctx = prepare(setup=setup, universe=universe, symbols=symbols, params=params, config=config,
                  start=start, end=end, use_earnings=use_earnings)
    res = eng.simulate(
        ctx["data"], ctx["signals"], ctx["rules"], ctx["cfg"],
        earnings=ctx["earnings"], long_regime=ctx["regime"],
        benchmark=ctx["benchmark"], benchmark_symbol=ctx["benchmark_symbol"],
    )
    return {**res.as_dict(max_trades=max_trades), **report_meta(ctx)}


def jitter_context(ctx: dict[str, Any], seed: int, eps: float = 1e-5) -> dict[str, Any]:
    """Copy of `ctx` with prices nudged by ~eps (default 0.001%, far below a
    tick) and signals rebuilt. Robustness probe: a portfolio with limited
    slots is path-dependent, so a result worth trusting must survive this."""
    rng = np.random.default_rng(seed)
    data = {}
    for sym, df in ctx["data"].items():
        d = df.copy()
        cols = ["open", "high", "low", "close"]
        d[cols] = df[cols].to_numpy() * (1 + eps * rng.standard_normal((len(df), 4)))
        d["high"] = d[["high", "open", "close"]].max(axis=1)
        d["low"] = d[["low", "open", "close"]].min(axis=1)
        data[sym] = d
    signals = {sym: build_signals(d, ctx["setup"], ctx["params"]) for sym, d in data.items()}
    return {**ctx, "data": data, "signals": signals}


def report_meta(ctx: dict[str, Any]) -> dict[str, Any]:
    um = ctx["universe_meta"]
    return {
        "setup": ctx["setup"], "setup_label": SETUPS[ctx["setup"]]["label"], "params": ctx["params"],
        "universe": ctx["universe"], "universe_label": um["label"], "universe_note": um["note"],
        "asset_class": ctx["asset_class"], "symbols": sorted(ctx["data"]), "errors": ctx["errors"],
    }


# ---------------------------------------------------------------------------
# Live: completed bars, scanner, entry signal, exit management
# ---------------------------------------------------------------------------

def completed_bars(df: pd.DataFrame, symbol: str, now: datetime | None = None) -> pd.DataFrame:
    """Drop today's still-forming bar so signals don't flicker intraday.

    Bars are keyed to the session date (UTC midnight). A US equity session is
    final ~16:15 ET; an FX/crypto "day" is final 24h after it starts.
    """
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    last = df.index[-1]
    if sd.asset_class(symbol) == "equity":
        et = pd.Timestamp(now).tz_convert("America/New_York")
        if last.date() >= et.date() and (et.hour, et.minute) < (16, 15):
            return df.iloc[:-1]
        return df
    if pd.Timestamp(now) < last + pd.Timedelta(hours=24):
        return df.iloc[:-1]
    return df


def _recent_bars(symbol: str) -> pd.DataFrame:
    """Recent consolidated daily bars (fresh within ~30 min) for live decisions."""
    start = (datetime.now(timezone.utc) - timedelta(days=800)).date().isoformat()
    return completed_bars(sd.daily_bars(symbol, start=start, max_age_s=30 * 60), symbol)


EARNINGS_ESTIMATE_UNCERTAINTY_DAYS = 7


def _estimate_next_report(symbol: str, now: datetime | None = None) -> pd.Timestamp | None:
    """Next report estimated from SEC 8-K Item 2.02 history.

    Companies keep a steady annual rhythm, so "same quarter last year + 364
    days" (same weekday) is usually within a few days of the real date.
    """
    hist = sd.earnings_dates(symbol)
    if not hist:
        return None
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    floor = now_ts - pd.Timedelta(days=3)
    cands = sorted(d + pd.Timedelta(days=364) for d in hist)
    cands += [hist[-1] + pd.Timedelta(days=91)]
    future = [c for c in cands if c >= floor and c > hist[-1] + pd.Timedelta(days=45)]
    return min(future) if future else None


def _next_earnings(symbol: str) -> dict[str, Any] | None:
    if sd.asset_class(symbol) != "equity":
        return {"known": True, "date": None, "days": None, "note": "no earnings (not a stock)"}
    try:
        from app.services import earnings as earn
        nxt = earn.next_earnings(symbol)
    except Exception:  # noqa: BLE001
        nxt = None
    if nxt is not None:
        return {"known": True, "estimated": False, "date": str(nxt.get("ts", ""))[:10],
                "days": nxt.get("days_to_earnings"), "note": None}
    try:
        est = _estimate_next_report(symbol)
    except Exception:  # noqa: BLE001
        est = None
    if est is not None:
        days = (est - pd.Timestamp(datetime.now(timezone.utc))).total_seconds() / 86400
        return {"known": True, "estimated": True, "date": est.date().isoformat(),
                "days": round(days, 1),
                "note": f"estimated from last year's SEC filing (±{EARNINGS_ESTIMATE_UNCERTAINTY_DAYS}d) "
                        "— confirm the date"}
    # Unknown ≠ none: ETFs legitimately have none, but a stock whose
    # calendar failed to load must be checked by hand.
    return {"known": False, "date": None, "days": None,
            "note": "earnings date unavailable — check before trading"}


def _conservative_days(earn: dict[str, Any] | None) -> float | None:
    """Days until the report, pulled earlier by the estimate's uncertainty."""
    if not earn or earn.get("days") is None:
        return None
    d = float(earn["days"])
    return d - EARNINGS_ESTIMATE_UNCERTAINTY_DAYS if earn.get("estimated") else d


def plan_trade(last: pd.Series, *, side: int, price: float, rules: eng.SetupRules,
               equity: float, risk_pct: float, max_position_pct: float,
               whole_units: bool) -> dict[str, Any]:
    """Entry/stop/target/size for a signal on the latest completed bar.

    `price` is the reference (last close); the real fill is the next open, so
    the stop is carried as a *distance* and re-anchored to the actual fill.
    """
    dist = float(last["stop_dist_long"] if side > 0 else last["stop_dist_short"])
    stop = price - side * dist
    if rules.target_r:
        target = price + side * rules.target_r * dist
    elif rules.target_pct:
        target = price * (1 + side * rules.target_pct)
    else:
        target = None
    if rules.sizing == "equal":
        units = equity * max_position_pct / 100.0 / price
    else:
        units = min(equity * risk_pct / 100.0 / dist, equity * max_position_pct / 100.0 / price)
    if whole_units:
        units = float(np.floor(units))
    return {
        "entry_ref": round(price, 6), "stop": round(stop, 6), "stop_dist": round(dist, 6),
        "stop_pct": round(dist / price * 100, 2),
        "target": round(target, 6) if target is not None else None,
        "units": round(units, 4), "notional": round(units * price, 2),
        "risk_usd": round(units * dist, 2),
        "pct_of_equity": round(units * price / equity * 100, 2) if equity else None,
    }


def scan(
    symbols: list[str],
    setup: str = "breakout",
    *,
    params: dict[str, Any] | None = None,
    equity: float = 100_000.0,
    risk_pct: float = 1.0,
    max_position_pct: float = 20.0,
    include_short: bool | None = None,
) -> dict[str, Any]:
    """Today's actionable setups across `symbols` from the last completed bar."""
    p = _p(setup, params)
    rules = setup_rules(setup, p)
    out, errors = [], {}
    for raw in symbols:
        sym = raw.upper()
        try:
            df = _recent_bars(sym)
            if len(df) < 260:
                errors[sym] = "not enough history"
                continue
            sig = build_signals(df, setup, p)
            last = sig.iloc[-1]
            aclass = sd.asset_class(sym)
            shorts = include_short if include_short is not None else (aclass == "fx")
            side = 1 if last["long_entry"] else (-1 if (shorts and setup != "rsi2" and last["short_entry"]) else 0)
            if not side:
                continue
            price = float(df["close"].iloc[-1])
            plan = plan_trade(last, side=side, price=price, rules=rules, equity=equity,
                              risk_pct=risk_pct,
                              max_position_pct=max_position_pct if aclass != "fx" else 150.0,
                              whole_units=aclass == "equity")
            earn = _next_earnings(sym)
            warn = []
            cd = _conservative_days(earn)
            if cd is not None and cd <= 7 and float(earn["days"]) >= -1:
                warn.append(f"earnings in ~{float(earn['days']):.0f} days — the bot skips these entries")
            if earn and earn.get("note") and sd.asset_class(sym) == "equity":
                warn.append(earn["note"])
            if aclass != "equity":
                warn.append("research only — not tradable through Alpaca")
            score = last["score_long"] if side > 0 else last["score_short"]
            out.append({
                "symbol": sym, "asset_class": aclass, "setup": setup,
                "side": "long" if side > 0 else "short",
                "as_of": df.index[-1].date().isoformat(),
                "close": round(price, 6), "atr": round(float(last["atr"]), 6),
                "score": round(float(score), 4) if np.isfinite(score) else None,
                **plan, "earnings": earn, "warnings": warn,
            })
        except Exception as e:  # noqa: BLE001 — one bad symbol shouldn't sink the scan
            errors[sym] = str(e)
    out.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1e9))
    return {
        "setup": setup, "setup_label": SETUPS[setup]["label"], "params": p,
        "equity": equity, "risk_pct": risk_pct,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "candidates": out, "errors": errors,
        "note": "Signals use the last completed daily bar; orders fill at the next open, "
                "so re-anchor the stop to the actual fill price.",
    }


def latest_signal(symbol: str, dsl: dict[str, Any]) -> dict[str, Any]:
    """Live entry decision for a swing bot (long-only, equities)."""
    setup = BOT_KINDS[dsl["kind"]]
    p = _p(setup, dsl.get("params"))
    sym = symbol.upper()
    if sd.asset_class(sym) != "equity":
        return {"symbol": sym, "action": "FLAT", "reason": "not tradable via Alpaca (non-equity)"}
    df = _recent_bars(sym)
    if len(df) < 260:
        return {"symbol": sym, "action": "FLAT", "reason": "not enough history"}
    sig = build_signals(df, setup, p)
    last = sig.iloc[-1]
    price = float(df["close"].iloc[-1])
    out: dict[str, Any] = {
        "symbol": sym, "action": "FLAT", "as_of": df.index[-1].date().isoformat(),
        "price": round(price, 4), "setup": setup, "reason": "no entry condition met",
        "size_multiplier": 1.0,
    }
    if not bool(last["long_entry"]):
        return out
    blackout = int(dsl.get("earnings_blackout_days", 7))
    earn = _next_earnings(sym)
    cd = _conservative_days(earn)
    if cd is not None and cd <= blackout and float(earn["days"]) >= -1:
        out["reason"] = (f"{setup} signal skipped: earnings in ~{float(earn['days']):.1f} days"
                         + (" (estimated)" if earn.get("estimated") else ""))
        out["earnings"] = earn
        return out
    rules = setup_rules(setup, p)
    dist = float(last["stop_dist_long"])
    out.update({
        "action": "BUY", "reason": f"{SETUPS[setup]['label']} entry",
        "stop_dist": round(dist, 4), "stop": round(price - dist, 4),
        "target": round(price + rules.target_r * dist, 4) if rules.target_r else None,
        "score": float(last["score_long"]) if np.isfinite(last["score_long"]) else None,
        "earnings": earn,
    })
    return out


def evaluate_exit(symbol: str, dsl: dict[str, Any], meta: dict[str, Any] | None,
                  *, entry_price: float, current_price: float) -> dict[str, Any]:
    """Decide whether a held swing position should close, and where its stop goes.

    Returns {"exit": bool, "reason": str, "new_stop": float | None}. `meta` is
    the entry record the bot persisted (stop, entry date, …); without it the
    position is managed with a stop re-derived from today's ATR.
    """
    setup = BOT_KINDS[dsl["kind"]]
    p = _p(setup, dsl.get("params"))
    rules = setup_rules(setup, p)
    meta = meta or {}
    stop = meta.get("stop")
    target = meta.get("target")
    if stop is not None and current_price <= float(stop):
        return {"exit": True, "reason": "stop", "new_stop": None}
    if target is not None and current_price >= float(target):
        return {"exit": True, "reason": "target", "new_stop": None}

    earn = _next_earnings(symbol)
    cd = _conservative_days(earn)
    if cd is not None and cd <= 1.5 and float(earn["days"]) >= -1:
        return {"exit": True, "reason": "earnings_ahead" + ("_estimated" if earn.get("estimated") else ""),
                "new_stop": None}

    try:
        df = _recent_bars(symbol)
    except Exception as e:  # noqa: BLE001 — never force a close on a data hiccup
        return {"exit": False, "reason": f"hold (data unavailable: {e})", "new_stop": None}
    if df.empty:
        return {"exit": False, "reason": "hold (no bars)", "new_stop": None}
    sig = build_signals(df, setup, p)
    if bool(sig["exit_long"].iloc[-1]):
        return {"exit": True, "reason": "signal", "new_stop": None}

    new_stop = None
    since = None
    if meta.get("entry_date"):
        since = df[df.index >= pd.Timestamp(meta["entry_date"], tz="UTC")]
        if rules.max_hold_bars and len(since) >= rules.max_hold_bars:
            return {"exit": True, "reason": "time", "new_stop": None}
    if rules.trail_atr_mult:
        window = since if since is not None and not since.empty else df.iloc[-1:]
        atr_now = float(sig["atr"].iloc[-1])
        if np.isfinite(atr_now):
            chand = float(max(window["high"].max(), entry_price)) - rules.trail_atr_mult * atr_now
            if stop is None or chand > float(stop):
                new_stop = round(chand, 2)
                if current_price <= new_stop:
                    return {"exit": True, "reason": "trail_stop", "new_stop": None}
    return {"exit": False, "reason": "hold", "new_stop": new_stop}
