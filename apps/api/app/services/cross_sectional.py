"""Cross-sectional ranking — the way professional quants actually screen.

Absolute thresholds (e.g. "score > 0.6 = STRONG_BUY") have a known weakness:
in a strong tape, almost every symbol scores > 0.6, so the label loses
discrimination. Cross-sectional ranking solves this — we rank the **universe**
today by a chosen score, then surface the top decile as longs and the bottom
decile as shorts. The labels become *relative*, not absolute.

This is what's behind quant equity long/short portfolios.

To make the ranking comparable across symbols and unbiased by sector
composition, raw scores are turned into **cross-sectional z-scores**:

  z_i = (x_i - mean(x)) / std(x)

and, when sector metadata is available, the standardization is done *within
sector groups* (sector-neutral / demeaned-by-sector), so e.g. a cheap utility
is ranked against other utilities rather than against expensive tech names.
Symbols whose sector is unknown fall into a shared "_unknown" bucket.

We rank by:
  - **composite**: z-score of the opinion engine's blended score
  - **momentum**: z-score of 3-month return
  - **value**: z-score of earnings yield (E/P = 1 / PE, forward PE preferred);
    cheap stocks (high earnings yield / low PE) rank higher
  - **smart_money**: z-score of insider + institutional + political + options
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from app.core.logging import log
from app.services import market_data as md
from app.services.opinion import collect_signals


# ----------------------------------------------------------------------------
# Score extractors
# ----------------------------------------------------------------------------

def _composite_score(sym: str) -> dict[str, Any] | None:
    """Fast composite scorer — uses cheap signals only (no model training).

    Pulls technicals + price action + analyst + sector RS + PEAD directly,
    skipping the ensemble forecast (which would require training models per
    symbol — prohibitively slow for a universe scan).
    """
    try:
        from app.services import analysts as analyst_svc
        from app.services import earnings as earnings_svc
        from app.services import market_data as md_svc
        from app.services import price_action as pa_svc
        from app.services import sector_rs as sector_svc

        df = md_svc.get_history(sym, interval="1d", range_="1y")
        if df.empty or len(df) < 60:
            return None

        # Price action (deterministic, ~30ms)
        pa_result = pa_svc.analyze(df)
        pa_score = float(((pa_result.get("confluence") or {}).get("score")) or 0.0) if pa_result.get("ok") else 0.0

        # Sector relative strength (~1 yfinance call shared cache)
        srs = float(sector_svc.sector_relative_strength(sym).get("score") or 0.0)

        # Analyst signal (yfinance cached)
        analyst_score = float(analyst_svc.analyst_signal(sym).get("score") or 0.0)

        # PEAD
        pead = float(earnings_svc.pead_signal(sym).get("score") or 0.0)

        # Simple technical (RSI + above-SMA200)
        from app.services import indicators as ind
        close = df["close"]
        rsi = float(ind.rsi(close, 14).iloc[-1])
        last = float(close.iloc[-1])
        sma200 = float(ind.sma(close, 200).iloc[-1]) if len(close) >= 200 else last
        tech = 0.3 * (1.0 if last > sma200 else -1.0) + 0.2 * (
            1.0 if 45 <= rsi <= 70 else (-0.5 if rsi > 80 else (-0.3 if rsi < 30 else 0.0)))

        score = (
            0.25 * pa_score
            + 0.20 * analyst_score
            + 0.20 * srs
            + 0.15 * pead
            + 0.20 * tech
        )
        return {
            "symbol": sym,
            "score": float(max(-1.0, min(1.0, score))),
            "components": {
                "price_action": round(pa_score, 3),
                "analyst": round(analyst_score, 3),
                "sector_rs": round(srs, 3),
                "pead": round(pead, 3),
                "technical": round(tech, 3),
            },
            "last_price": last,
        }
    except Exception as e:
        log.warning("xs_composite_failed", symbol=sym, err=str(e))
        return None


def _momentum_score(sym: str) -> dict[str, Any] | None:
    try:
        df = md.get_history(sym, interval="1d", range_="6mo")
        if df.empty or len(df) < 63:
            return None
        ret_3m = float(df["close"].iloc[-1] / df["close"].iloc[-63] - 1)
        return {
            "symbol": sym, "score": float(np.tanh(ret_3m * 2.0)),
            "components": {"ret_3m": ret_3m},
            "last_price": float(df["close"].iloc[-1]),
        }
    except Exception:
        return None


def _smart_money_only(sym: str) -> dict[str, Any] | None:
    try:
        signals = collect_signals(sym, use_cache=True)
        if not signals.get("last_price"):
            return None
        ins = float((signals.get("insider") or {}).get("score") or 0.0)
        inst = float((signals.get("institutional") or {}).get("score") or 0.0)
        pol = float((signals.get("politicians") or {}).get("score") or 0.0)
        opt = float((signals.get("options") or {}).get("score") or 0.0)
        score = 0.35 * ins + 0.25 * inst + 0.20 * pol + 0.20 * opt
        return {
            "symbol": sym, "score": float(max(-1.0, min(1.0, score))),
            "components": {"insider": ins, "institutional": inst, "political": pol, "options": opt},
            "last_price": signals.get("last_price"),
        }
    except Exception:
        return None


def _value_score(sym: str) -> dict[str, Any] | None:
    """Value strategy: rank by earnings yield (E/P = 1 / PE).

    Forward PE is preferred (uses analyst-estimated forward earnings); we fall
    back to trailing PE. A higher earnings yield (i.e. lower PE = cheaper) gets a
    higher raw score. Non-positive or missing PE (loss-making / no estimate) is
    not a valid value signal, so those symbols are skipped. The raw earnings
    yield is later standardized into a cross-sectional z-score by rank_universe.
    """
    try:
        ks = md.get_key_stats(sym)
        pe = ks.get("forward_pe")
        if pe is None or (isinstance(pe, (int, float)) and pe <= 0):
            pe = ks.get("pe")
        if pe is None:
            return None
        pe = float(pe)
        if pe <= 0:
            return None
        earnings_yield = 1.0 / pe
        q = md.get_quote(sym)
        return {
            "symbol": sym,
            # Raw score = earnings yield; standardized downstream. Higher = cheaper.
            "score": float(earnings_yield),
            "components": {"earnings_yield": round(earnings_yield, 4),
                           "pe": round(pe, 2)},
            "last_price": q.get("price"),
        }
    except Exception:
        return None


_STRATEGIES: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "composite": _composite_score,
    "momentum": _momentum_score,
    "value": _value_score,
    "smart_money": _smart_money_only,
}


def _rank_momentum_pead(universe: list[str], top_n: int, bottom_n: int) -> dict[str, Any]:
    """The backtested stock-selection composite (app/ml/stock_selection.py).

    Unlike the per-symbol scorers above this is inherently cross-sectional —
    sector-relative momentum needs the whole panel — so it builds a
    (date x symbol) panel and runs exactly the code the backtest runs, then
    reads off the latest bar. No sector z-scoring on top: the composite is
    already a blend of cross-sectional ranks, as in the backtest.
    """
    import pandas as pd

    from app.ml import stock_selection as ss
    from app.services import earnings as earnings_svc
    from app.services.universe_data import all_instruments

    def hist(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            return sym, md.get_history(sym, interval="1d", range_="2y")
        except Exception:
            return sym, pd.DataFrame()

    frames: dict[str, pd.DataFrame] = {}
    if len(universe) > 60:
        # one request per ~100 symbols instead of one each (rate limits)
        from app.services import alpaca_bars
        frames = alpaca_bars.get_bars_multi(list(universe) + ["SPY"], range_="2y")
    missing = [s for s in list(universe) + ["SPY"] if s not in frames]
    if missing:
        with ThreadPoolExecutor(max_workers=4) as ex:
            frames.update(dict(ex.map(hist, missing)))
    frames = {k: v for k, v in frames.items() if not v.empty}
    if "SPY" not in frames:
        return {"strategy": "momentum_pead", "longs": [], "shorts": [], "n_scored": 0,
                "error": "benchmark history unavailable"}

    def panel(col: str) -> pd.DataFrame:
        cols = {s: df[col] for s, df in frames.items()}
        out = pd.DataFrame(cols)
        out.index = pd.to_datetime(out.index, utc=True).tz_convert(None).normalize()
        return out.sort_index()

    close, volume = panel("close"), panel("volume")
    spy = close.pop("SPY")
    volume = volume.drop(columns="SPY")

    def events(sym: str) -> tuple[str, list[ss.EarningsEvent]]:
        try:
            rows = earnings_svc.earnings_dates(sym, limit=8)
        except Exception:
            rows = []
        return sym, [ss.EarningsEvent(pd.Timestamp(r["ts"]), r.get("surprise_pct"))
                     for r in rows if r.get("status") == "reported"]

    with ThreadPoolExecutor(max_workers=3) as ex:   # Yahoo rate-limits bursts
        ev = dict(ex.map(events, list(close.columns)))
    earnings_coverage = sum(1 for v in ev.values() if v) / max(1, len(ev))
    sectors = {sym: sec for sym, _n, _e, _a, sec in all_instruments()}

    factors = ss.compute_factors(close, spy, sectors, ev)
    # Alpaca's free IEX feed sees only ~2-5% of consolidated volume, so the
    # backtest's $5M/day liquidity floor would wrongly drop most mid-caps.
    from app.config import settings as _settings
    min_dv = 5e6 * (0.02 if (_settings.alpaca_feed or "iex") == "iex" else 1.0)
    mask = ss.universe_mask(close, volume, min_dollar_volume=min_dv)
    score, ranks = ss.composite(factors, mask)
    # Rank as of the latest bar most of the universe has. Mixed sources (Alpaca
    # first, yfinance fallback) can leave a few symbols a bar behind; ranking on
    # the very last row would silently drop them.
    have = close.notna().sum(axis=1)
    last = have[have >= 0.8 * have.max()].index[-1]
    latest = score.loc[last].dropna().sort_values(ascending=False)
    if latest.empty:
        return {"strategy": "momentum_pead", "longs": [], "shorts": [], "n_scored": 0}
    n = len(latest)
    z = (latest - latest.mean()) / (latest.std() or 1.0)
    sma = spy.rolling(ss.TREND_SMA_BARS).mean()
    risk_on = bool(spy.loc[last] > sma.loc[last]) if pd.notna(sma.loc[last]) else True
    results = [{
        "symbol": sym,
        "score": round(float(z[sym]), 3),
        "raw_score": round(float(latest[sym]), 4),
        "components": {k: round(float(r.loc[last, sym]), 3) if pd.notna(r.loc[last, sym]) else None
                       for k, r in ranks.items()},
        "last_price": float(close.loc[last, sym]),
        "percentile": round(100 * (1 - i / max(n - 1, 1)), 1),
        "in_portfolio": i < max(5, int(n * ss.TOP_FRACTION)),
    } for i, sym in enumerate(latest.index)]
    return {
        "strategy": "momentum_pead",
        "as_of": last.date().isoformat(),
        "n_scored": n,
        "universe_size": len(universe),
        "hold_days": ss.HOLD_BARS,
        "portfolio_size": max(5, int(n * ss.TOP_FRACTION)),
        "earnings_coverage": round(earnings_coverage, 3),
        "market_regime": {
            "risk_on": risk_on,
            "spy_close": round(float(spy.loc[last]), 2),
            "spy_sma_10m": round(float(sma.loc[last]), 2) if pd.notna(sma.loc[last]) else None,
            "allocation": ("top-ranked stocks, equal weight" if risk_on
                           else "short-term Treasuries (SHY) until SPY closes back above its 10-month average"),
            "note": "Trend gate is optional: ungated = higher return, ~2x deeper drawdowns (backtest 2006-2026).",
        },
        "longs": results[:top_n],
        "shorts": list(reversed(results[-bottom_n:])),
    }


# ----------------------------------------------------------------------------
# Cross-sectional standardization (z-scores, sector-neutral)
# ----------------------------------------------------------------------------

def _zscore(values: list[float]) -> list[float]:
    """Cross-sectional z-score of a list. Returns zeros if std is ~0 or n<2."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return [0.0] * arr.size
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd < 1e-12:
        return [0.0] * arr.size
    return [float((v - mu) / sd) for v in arr]


def _standardize(results: list[dict[str, Any]], sector_of: dict[str, str | None]) -> None:
    """Replace each result's raw `score` with a cross-sectional z-score, in
    place. When sector metadata is available the z-score is computed WITHIN each
    sector group (sector-neutral); symbols with unknown sector share an
    "_unknown" bucket. The pre-standardization value is preserved as `raw_score`.
    """
    # Bucket indices by sector (sector-neutral grouping).
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        sec = sector_of.get(r["symbol"]) or "_unknown"
        groups[sec].append(i)

    for r in results:
        r["raw_score"] = r["score"]

    for _sec, idxs in groups.items():
        zs = _zscore([results[i]["score"] for i in idxs])
        for i, z in zip(idxs, zs):
            results[i]["score"] = z


# ----------------------------------------------------------------------------
# Ranking entry point
# ----------------------------------------------------------------------------

def rank_universe(
    strategy: str = "composite",
    universe: list[str] | None = None,
    top_n: int = 10,
    bottom_n: int = 10,
    sectors: list[str] | None = None,
) -> dict[str, Any]:
    """Score every symbol in the universe, return top-N longs + bottom-N shorts."""
    if strategy not in _STRATEGIES and strategy != "momentum_pead":
        raise ValueError(f"unknown strategy: {strategy}")

    if universe is None:
        all_u = md.all_universe()
        universe = [it["symbol"] for it in all_u if it.get("asset_class") == "stock"]
        if sectors:
            universe = [s for s in universe
                        if md.get_profile(s).get("sector") in set(sectors)]

    if strategy == "momentum_pead":
        return _rank_momentum_pead(universe, top_n, bottom_n)

    scorer = _STRATEGIES[strategy]
    results: list[dict[str, Any]] = []

    # Parallelize per-symbol scoring. Composite involves the opinion pack
    # which is cached, so subsequent runs are fast.
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(scorer, s): s for s in universe}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        return {"strategy": strategy, "longs": [], "shorts": [], "n_scored": 0}

    # Standardize raw scores into cross-sectional z-scores, sector-neutral when
    # we can resolve sector metadata. This makes scores comparable across the
    # universe and removes systematic sector tilts from the ranking.
    sector_of: dict[str, str | None] = {}
    for r in results:
        try:
            sector_of[r["symbol"]] = md.get_profile(r["symbol"]).get("sector")
        except Exception:
            sector_of[r["symbol"]] = None
    _standardize(results, sector_of)

    results.sort(key=lambda r: -r["score"])
    # Add percentile rank
    n = len(results)
    for i, r in enumerate(results):
        r["percentile"] = round(100 * (1 - i / max(n - 1, 1)), 1)

    return {
        "strategy": strategy,
        "n_scored": n,
        "universe_size": len(universe),
        "longs":  results[:top_n],
        "shorts": list(reversed(results[-bottom_n:])),
    }


def list_strategies() -> list[dict[str, str]]:
    return [
        {"key": "momentum_pead", "label": "Earnings surprise + momentum (backtested)",
         "description": "50% latest EPS surprise vs consensus, 50% 12-1 / residual / "
                        "sector-relative momentum; rebalance monthly, hold the top 10%. "
                        "S&P 500+400 backtest 2006-2026 (median of all rebalance days, after "
                        "costs): ~17%/yr vs 14% equal-weight and 11% SPY, max drawdown ~-52%. "
                        "With the market_regime trend gate (SHY when SPY < 10-month SMA): "
                        "~14%/yr, drawdown ~-27%. Did not beat equal weight in 2006-2016; "
                        "survivorship bias flatters all numbers. Paper-trade before trusting."},
        {"key": "composite",   "label": "Composite (full opinion blend)",
         "description": "Rank by the full opinion-engine blended score across every available signal."},
        {"key": "momentum",    "label": "Pure momentum",
         "description": "3-month return only. The simplest CTA-style ranking."},
        {"key": "value",       "label": "Value (earnings yield)",
         "description": "Rank by earnings yield (1/PE, forward PE preferred) "
                        "as a sector-neutral z-score — cheaper stocks rank higher."},
        {"key": "smart_money", "label": "Smart money only",
         "description": "Insider + institutional + political + options-flow signals."},
    ]
