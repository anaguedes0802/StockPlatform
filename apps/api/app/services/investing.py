"""Long-horizon investment bot — the buy-and-hold counterpart to `trading_bot`.

Where the trading bot harvests brief oversold dips and SELLS the bounce, this
module ACCUMULATES a diversified target allocation and HOLDS it. It never takes
profit. Its three jobs map onto what the user asked for:

  1. **Hold a diversified target allocation** — broad ETFs spanning US large-cap,
     growth, international, a quality/dividend sleeve, gold and bonds (a classic
     robo-advisor mix). The user can override the weights or add individual names.

  2. **Tell the user WHEN it's an attractive time to add money.** For a long-term
     accumulator the logic is the *opposite* of the trading bot: a fearful,
     beaten-down tape is a BETTER entry, not a worse one ("be greedy when others
     are fearful"). `invest_timing()` blends the market regime with how deep the
     current pullback is into a NORMAL / ACCUMULATE / STRONG_BUY call. The API
     layer turns a fresh ACCUMULATE/STRONG_BUY into a notification.

  3. **Deploy cash toward the target weights.** `rebalance_plan()` does tax-
     friendly *cash-flow* rebalancing: available cash + any new contribution the
     user chooses to add buys the most UNDERWEIGHT sleeves, rather than selling
     winners. On a strong dip it tilts new money toward equities.

Like the trading bot, execution is broker-injectable and paper-by-default; the
bot only places orders while `armed`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import redis

from app.config import settings
from app.core.logging import log
from app.services import broker_alpaca as broker
from app.services import llm
from app.services import market_data as md
from app.services import regime as regime_svc

# ---------------------------------------------------------------------------
# Redis result cache (same fail-open pattern as market_data._cache_*)
# ---------------------------------------------------------------------------
# `quality_picks` scores a whole universe (~68 names) by fetching history +
# key-stats per symbol over the network. Even with the per-symbol 1h caches in
# market_data, the cross-sectional scan is slow on a cold path. We cache the
# *final* result keyed by (top_n, universe signature) for 30 minutes so repeat
# calls — the bot preview/run and the /bot/invest/quality-picks endpoint —
# return instantly. Fail-open: any Redis error just falls through to compute.

_redis: redis.Redis | None = None

# How long a computed picks result stays warm (seconds).
QUALITY_PICKS_TTL = 1800  # 30 minutes


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _cache_get(key: str) -> Any | None:
    try:
        v = _cache().get(key)
        return json.loads(v) if v else None
    except Exception:
        return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass

# A diversified, low-cost default. Weights sum to 1.0. This *is* the "quality
# screen" for v1 — a curated set of broad, liquid sleeves rather than a single
# index — but every weight is user-overridable via the bot's dsl.
DEFAULT_ALLOCATION: dict[str, float] = {
    "SPY": 0.35,   # US large-cap core
    "QQQ": 0.15,   # US growth / tech
    "VXUS": 0.10,  # international (ex-US)
    "SCHD": 0.10,  # US quality / dividend tilt
    "GLD": 0.10,   # gold — inflation / crisis diversifier
    "BND": 0.20,   # US aggregate bonds — ballast
}

DEFAULT_INVEST_CONFIG: dict[str, Any] = {
    # How the bot decides WHAT to hold:
    #   "allocation" — the fixed diversified ETF basket below (default; safe).
    #   "quality"    — dynamically pick the top-N individual stocks by the
    #                  cross-sectional composite (value+quality+growth+momentum+
    #                  smart-money), equal-weight, and rotate on each run. Higher
    #                  potential return, higher risk; see the PIT backtest.
    "pick_mode": "allocation",
    "top_n": 10,
    "pick_strategy": "composite",
    "pick_universe": None,   # None → the platform's full liquid stock universe
    "allocation": DEFAULT_ALLOCATION,
    # Only rebalance a sleeve once it drifts more than this far from target
    # (fraction of portfolio), to avoid churning on tiny gaps.
    "rebalance_band": 0.05,
    # On a dip, shift up to this fraction of NEW money out of bonds/gold and into
    # equities (buy more stock when it's cheap). 0 disables the tilt.
    "dip_equity_tilt": 0.20,
    # Sleeves treated as "equity" for the dip tilt.
    "equity_sleeves": ["SPY", "QQQ", "VXUS", "SCHD"],
    # Sleeves the dip tilt pulls FROM (defensive ballast).
    "ballast_sleeves": ["BND", "GLD"],
    # Per-order ceiling so a single run can't dump everything into one name.
    "max_order_usd": 5_000.0,
    # Don't bother placing dust orders.
    "min_order_usd": 1.0,
}

# Drawdown (from the trailing 1y high) at which a pullback is treated as a
# maximum-strength long-term buying opportunity.
_MAX_DD = 0.20


def _closes(symbol: str):
    try:
        df = md.get_history(symbol, interval="1d", range_="1y")
        return df["close"] if not df.empty else None
    except Exception:  # noqa: BLE001
        return None


def _drawdown_from_high(symbol: str = "SPY") -> float:
    """Current drawdown from the trailing-year high, as a positive fraction.

    0.0 = at a new high; 0.15 = 15% below the year's peak.
    """
    s = _closes(symbol)
    if s is None or s.empty:
        return 0.0
    peak = float(s.max())
    last = float(s.iloc[-1])
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - last) / peak)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_weights(alloc: dict[str, float]) -> dict[str, float]:
    """Return weights that sum to 1.0 (drop non-positive, renormalise)."""
    clean = {str(k).upper(): float(v) for k, v in (alloc or {}).items() if float(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        return dict(DEFAULT_ALLOCATION)
    return {k: v / total for k, v in clean.items()}


# ---------------------------------------------------------------------------
# Timing: "is now a good time to add money?"
# ---------------------------------------------------------------------------

_SYSTEM_INVEST = (
    "You advise a LONG-TERM, buy-and-hold investment bot that dollar-cost-averages "
    "into a diversified ETF portfolio and never sells for profit. You are NOT timing "
    "short trades. A market that is down and fearful is a BETTER place to add long-term "
    "money, not worse. Given the regime read and current drawdown, write ONE or TWO "
    "calm sentences telling the investor whether now is a normal time to invest or an "
    "unusually good time to add extra, and why. No hype, no predictions, no JSON."
)


def invest_timing(*, use_llm: bool = True) -> dict[str, Any]:
    """Assess how attractive *now* is for adding long-term money.

    Combines the market regime (risk-off = cheaper = better for an accumulator)
    with the current drawdown from the year's high. Returns a level —
    NORMAL / ACCUMULATE / STRONG_BUY — plus an opportunity score and an optional
    one-line LLM read. Never raises: degrades to a regime-free heuristic.
    """
    try:
        reg = regime_svc.assess()
    except Exception:  # noqa: BLE001
        reg = {"score": 0, "regime": "neutral", "headline": "", "spy": {}}

    dd = _drawdown_from_high("SPY")
    reg_score = float(reg.get("score", 0) or 0)  # [-100, +100]

    # Opportunity rises as the market falls and fear rises. Drawdown carries the
    # most weight (it's the concrete "things are on sale" signal); the regime's
    # risk-off reading adds to it.
    dd_score = _clamp01(dd / _MAX_DD)
    fear_score = _clamp01(-reg_score / 100.0)
    opportunity = round(_clamp01(0.65 * dd_score + 0.35 * fear_score), 3)

    if opportunity >= 0.60:
        level, headline = "STRONG_BUY", "Unusually good time to add — the market is well off its highs."
    elif opportunity >= 0.30:
        level, headline = "ACCUMULATE", "A decent pullback — a good moment to add a bit extra."
    else:
        level, headline = "NORMAL", "No special discount — invest on your normal schedule."

    cfg_tilt = float(DEFAULT_INVEST_CONFIG["dip_equity_tilt"])
    # Scale the equity tilt with the opportunity (only kicks in past ACCUMULATE).
    equity_tilt = round(cfg_tilt * _clamp01((opportunity - 0.30) / 0.70), 3) if opportunity >= 0.30 else 0.0

    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "headline": headline,
        "opportunity": opportunity,
        "drawdown_from_high_pct": round(dd * 100, 2),
        "suggested_equity_tilt": equity_tilt,
        "regime": {"score": int(reg_score), "label": reg.get("regime"),
                   "headline": reg.get("headline")},
        "rationale": None,
    }

    if use_llm and llm.is_available():
        try:
            user = (
                f"Regime score {int(reg_score)} ({reg.get('regime')}). "
                f"SPY is {out['drawdown_from_high_pct']}% below its 1-year high. "
                f"Internal call: {level}. Give the investor a calm one-liner."
            )
            res = llm.generate(system=_SYSTEM_INVEST, user=user, max_tokens=120,
                               temperature=0.3, tier="quality")
            if res and res.text:
                out["rationale"] = res.text.strip()[:400]
                out["provider"] = f"{res.provider}:{res.model}"
        except Exception as e:  # noqa: BLE001
            log.warning("invest_timing_llm_failed", err=str(e)[:200])

    return out


# ---------------------------------------------------------------------------
# Allocation math + cash-flow rebalancing
# ---------------------------------------------------------------------------

def current_weights(positions: list[dict[str, Any]], cash: float) -> tuple[dict[str, float], float]:
    """Current portfolio weights (by market value) and total invested+cash value."""
    held = {str(p["symbol"]).upper(): float(p.get("market_value", 0.0))
            for p in positions if float(p.get("market_value", 0.0)) > 0}
    total = sum(held.values()) + max(0.0, cash)
    if total <= 0:
        return {}, 0.0
    return {k: v / total for k, v in held.items()}, total


def _apply_dip_tilt(weights: dict[str, float], tilt: float, dsl: dict[str, Any]) -> dict[str, float]:
    """Shift `tilt` fraction of weight from ballast sleeves into equity sleeves.

    Used only for *deploying new money* on a dip — the long-term target itself is
    untouched. Returns a fresh normalised weight map.
    """
    if tilt <= 0:
        return weights
    eq = [s for s in dsl.get("equity_sleeves", DEFAULT_INVEST_CONFIG["equity_sleeves"]) if s in weights]
    bal = [s for s in dsl.get("ballast_sleeves", DEFAULT_INVEST_CONFIG["ballast_sleeves"]) if s in weights]
    if not eq or not bal:
        return weights
    w = dict(weights)
    moved = sum(w[s] for s in bal) * tilt
    for s in bal:
        w[s] *= (1 - tilt)
    eq_total = sum(w[s] for s in eq) or 1.0
    for s in eq:
        w[s] += moved * (w[s] / eq_total)
    return normalize_weights(w)


# A liquid large/mid-cap candidate universe for the picker. Bounded so a live
# scan stays tractable; survivor-biased like any current list (stated honestly).
DEFAULT_PICK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "ORCL", "CSCO", "ADBE", "CRM",
    "INTC", "AMD", "QCOM", "TXN", "AVGO", "IBM", "NOW", "INTU",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK", "SCHW",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "LOW", "SBUX", "TGT",
    "DIS", "CMCSA", "VZ", "T", "NFLX",
    "XOM", "CVX", "COP", "CAT", "BA", "HON", "GE", "DE", "UNP", "LMT", "RTX", "MMM",
]
# Secondary share classes to drop so we don't double-count one company.
_SECONDARY_CLASSES = {"GOOG", "BRK.A", "FOX", "NWS", "UA"}


def _zscore_fill(values: list[float]) -> list[float]:
    """Cross-sectional z-score; missing values imputed to the median (z≈0)."""
    arr = pd.Series(values, dtype=float)
    med = arr.median()
    arr = arr.fillna(med)
    sd = arr.std(ddof=0)
    if not sd or pd.isna(sd):
        return [0.0] * len(arr)
    return list((arr - arr.mean()) / sd)


def _pick_factors(sym: str) -> dict[str, Any] | None:
    """Current factor inputs for one symbol (value / quality / growth / momentum).

    Mirrors the factors the PIT backtest validated, using current data: earnings
    yield (value), profit margin + low leverage + ROE (quality), revenue & EPS
    growth (growth), and 12-1-month price momentum. Returns raw factor values
    (z-scored across the universe by the caller). None if too little to score.
    """
    try:
        df = md.get_history(sym, interval="1d", range_="1y")
        if df.empty or len(df) < 200:
            return None
        close = df["close"]
        last = float(close.iloc[-1])
        mom = last / float(close.iloc[-252]) - 1 - (last / float(close.iloc[-21]) - 1) \
            if len(close) >= 252 else last / float(close.iloc[0]) - 1
        ks = md.get_key_stats(sym)
        pe = ks.get("forward_pe") or ks.get("pe")
        ey = (1.0 / float(pe)) if pe and float(pe) > 0 else None
        d2e = ks.get("debt_to_equity")
        return {
            "symbol": sym,
            "earnings_yield": ey,
            "margin": ks.get("profit_margin"),
            "roe": ks.get("return_on_equity"),
            "low_debt": (-float(d2e)) if d2e is not None else None,
            "rev_growth": ks.get("revenue_growth"),
            "eps_growth": ks.get("earnings_growth"),
            "momentum": mom,
            "last_price": last,
        }
    except Exception:  # noqa: BLE001
        return None


def _picks_cache_key(deduped: list[str], top_n: int) -> str:
    """Stable cache key from top_n + a hash of the resolved universe.

    The universe is hashed (after upper-casing + de-duping) so a long symbol
    list collapses to a short, deterministic key. Order-independent: the same
    set of symbols always maps to the same signature.
    """
    sig = hashlib.sha1(",".join(sorted(deduped)).encode()).hexdigest()[:16]
    return f"qpicks:{top_n}:{sig}"


def quality_picks(*, universe: list[str] | None = None, top_n: int = 10,
                  strategy: str = "quality") -> dict[str, Any]:
    """Top-N stocks by a multi-factor QUALITY/VALUE/GROWTH/MOMENTUM rank.

    This is the SAME factor recipe validated in the point-in-time backtest (not
    the momentum-only `composite`): each factor is z-scored across the universe
    and summed, equal-weighting the winners. Share classes are de-duplicated so
    one company (e.g. Alphabet's GOOGL/GOOG) can't take two slots. Returns
    {weights, picks}; empty on failure so the caller falls back to the ETF mix.

    The final result is cached in Redis for `QUALITY_PICKS_TTL` (30 min) keyed
    by top_n + a hash of the universe, so repeated previews/runs return without
    re-scanning the whole universe over the network. Fail-open: a cache miss or
    any Redis error just recomputes.
    """
    from concurrent.futures import ThreadPoolExecutor

    uni = [s.upper() for s in (universe or DEFAULT_PICK_UNIVERSE)]
    # De-dupe share classes (drop known secondary classes; keep first seen).
    seen, deduped = set(), []
    for s in uni:
        if s in _SECONDARY_CLASSES or s in seen:
            continue
        seen.add(s)
        deduped.append(s)

    cache_key = _picks_cache_key(deduped, top_n)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = [r for r in ex.map(_pick_factors, deduped) if r is not None]
    except Exception as e:  # noqa: BLE001
        log.warning("quality_picks_failed", err=str(e)[:200])
        return {"weights": {}, "picks": []}
    if len(rows) < max(top_n, 3):
        return {"weights": {}, "picks": []}

    factors = ["earnings_yield", "margin", "roe", "low_debt", "rev_growth", "eps_growth", "momentum"]
    z = {f: _zscore_fill([r.get(f) for r in rows]) for f in factors}
    scored = []
    for i, r in enumerate(rows):
        composite = sum(z[f][i] for f in factors)
        scored.append({"symbol": r["symbol"], "score": round(composite, 3),
                       "earnings_yield": r.get("earnings_yield"),
                       "margin": r.get("margin"), "momentum": round(r.get("momentum") or 0.0, 3)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    n = len(scored)
    for i, s in enumerate(scored):
        s["percentile"] = round(100 * (1 - i / max(n - 1, 1)), 1)
    top = scored[:top_n]
    w = 1.0 / len(top)
    result = {"weights": {p["symbol"]: w for p in top}, "picks": top, "n_scored": n}
    # Only cache a real (non-empty) result — empty/failure paths return early
    # above without caching so a transient data outage is retried, not pinned.
    _cache_set(cache_key, result, ttl=QUALITY_PICKS_TTL)
    return result


def _resolve_target(dsl: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
    """Resolve the bot's target weights and (for quality mode) the picks detail."""
    if dsl.get("pick_mode") == "quality":
        qp = quality_picks(universe=dsl.get("pick_universe"),
                           top_n=int(dsl.get("top_n", 10)),
                           strategy=str(dsl.get("pick_strategy", "composite")))
        if qp["weights"]:
            return normalize_weights(qp["weights"]), qp["picks"]
    return normalize_weights(dsl.get("allocation", DEFAULT_ALLOCATION)), None


def rebalance_plan(
    positions: list[dict[str, Any]],
    cash: float,
    dsl: dict[str, Any] | None = None,
    *,
    contribution: float = 0.0,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute buy orders to move the portfolio toward its target allocation.

    Tax-friendly *cash-flow* rebalancing: the deployable amount (idle cash +
    the user's optional new contribution) is allocated to whichever sleeves are
    the most UNDERWEIGHT versus target — we never sell winners to rebalance. On
    a dip (per `timing`) new money is tilted toward equities.

    Returns the target/current weights, the per-sleeve buy plan, and leftover.
    """
    dsl = {**DEFAULT_INVEST_CONFIG, **(dsl or {})}
    target, picks = _resolve_target(dsl)

    # Optional dip tilt on the *deployment* target (only meaningful for the ETF
    # allocation — for stock picks there are no ballast sleeves so it's a no-op).
    tilt = float((timing or {}).get("suggested_equity_tilt", 0.0) or 0.0)
    deploy_target = _apply_dip_tilt(target, tilt, dsl)

    cur_w, total_now = current_weights(positions, cash)
    deployable = max(0.0, cash) + max(0.0, float(contribution or 0.0))
    min_order = float(dsl.get("min_order_usd", 1.0))
    max_order = float(dsl.get("max_order_usd", 5_000.0))

    # Post-deployment portfolio value (what weights are measured against).
    held_mv = {str(p["symbol"]).upper(): float(p.get("market_value", 0.0)) for p in positions}
    future_total = sum(held_mv.values()) + deployable
    if future_total <= 0:
        return {"deployable": 0.0, "orders": [], "target": target, "current": cur_w,
                "leftover": deployable, "note": "nothing to deploy"}

    # Dollar gap to target for each sleeve; only underweights are buy candidates.
    gaps: dict[str, float] = {}
    for sym, w in deploy_target.items():
        desired_mv = w * future_total
        gap = desired_mv - held_mv.get(sym, 0.0)
        if gap > 0:
            gaps[sym] = gap
    total_gap = sum(gaps.values())

    orders: list[dict[str, Any]] = []
    if total_gap > 0 and deployable >= min_order:
        # Distribute the deployable cash across underweight sleeves in proportion
        # to how underweight each is.
        for sym, gap in sorted(gaps.items(), key=lambda kv: kv[1], reverse=True):
            amt = min(deployable * (gap / total_gap), gap, max_order)
            if amt >= min_order:
                orders.append({"symbol": sym, "notional": round(amt, 2),
                               "target_pct": round(deploy_target[sym] * 100, 1)})
    spent = sum(o["notional"] for o in orders)
    return {
        "deployable": round(deployable, 2),
        "contribution": round(float(contribution or 0.0), 2),
        "idle_cash": round(max(0.0, cash), 2),
        "orders": orders,
        "pick_mode": dsl.get("pick_mode", "allocation"),
        "picks": picks,  # populated only in quality mode
        "target": {k: round(v, 4) for k, v in target.items()},
        "deploy_target": {k: round(v, 4) for k, v in deploy_target.items()},
        "current": {k: round(v, 4) for k, v in cur_w.items()},
        "equity_tilt_applied": tilt,
        "leftover": round(deployable - spent, 2),
    }


# ---------------------------------------------------------------------------
# Live execution
# ---------------------------------------------------------------------------

def run_invest(
    *,
    dsl: dict[str, Any] | None,
    execution: dict[str, Any] | None,
    contribution: float = 0.0,
    broker_mod=broker,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Execute one investment pass: deploy idle cash (+ contribution) to targets.

    Buy-only cash-flow rebalancing — never sells. Honours the manual arm guard
    upstream (the router checks `armed`); here we only require a configured
    broker and an open market (unless overridden). Returns the timing read, the
    plan, and the orders actually placed.
    """
    dsl = {**DEFAULT_INVEST_CONFIG, **(dsl or {})}
    execution = execution or {}

    if not broker_mod.is_configured():
        return {"blocked": "broker_not_configured", "actions": [], "plan": None, "account": None}

    try:
        market_open = broker_mod.is_market_open()
    except Exception as e:  # noqa: BLE001
        return {"blocked": f"clock_unavailable: {e}", "actions": [], "plan": None, "account": None}

    if execution.get("market_hours_only", True) and not market_open:
        return {"blocked": "market_closed", "actions": [], "plan": None,
                "account": _safe_account(broker_mod), "market_open": False}

    account = _safe_account(broker_mod)
    try:
        positions = broker_mod.list_positions()
    except Exception as e:  # noqa: BLE001
        return {"blocked": f"positions_unavailable: {e}", "actions": [], "plan": None, "account": account}

    cash = float(account.get("cash", 0.0)) if account else 0.0
    timing = invest_timing(use_llm=use_llm)
    plan = rebalance_plan(positions, cash, dsl, contribution=contribution, timing=timing)

    actions: list[dict[str, Any]] = []
    buying_power = float(account.get("buying_power", 0.0)) if account else 0.0
    spent = 0.0
    for o in plan["orders"]:
        if spent + o["notional"] > buying_power:
            actions.append({"symbol": o["symbol"], "action": "BUY_SKIPPED",
                            "reason": "insufficient buying power"})
            continue
        try:
            order = broker_mod.submit_market_order(o["symbol"], notional=o["notional"], side="buy")
            spent += o["notional"]
            actions.append({"symbol": o["symbol"], "action": "BUY",
                            "notional": o["notional"], "target_pct": o["target_pct"],
                            "order": order, "ts": datetime.now(timezone.utc).isoformat()})
        except Exception as e:  # noqa: BLE001
            actions.append({"symbol": o["symbol"], "action": "BUY_FAILED", "reason": str(e)})

    return {"blocked": None, "market_open": market_open, "timing": timing,
            "plan": plan, "actions": actions, "account": account}


def _safe_account(broker_mod) -> dict[str, Any] | None:
    try:
        return broker_mod.get_account()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Dollar-cost-averaging backtest — the long-term growth story
# ---------------------------------------------------------------------------

def _full_history_close(symbol: str) -> pd.Series | None:
    """Full dividend-adjusted daily close back to inception, via yfinance.

    Alpaca's free tier only reaches ~2016/2020 for some ETFs, which truncates a
    long-term DCA backtest. yfinance (auto_adjust=True) returns total-return
    closes to inception. Best-effort: returns None on any failure so the caller
    falls back to the normal data path.
    """
    try:
        t = md._yf_ticker(symbol)
        df = t.history(period="max", interval="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index, utc=True)
        s.name = symbol
        return s.dropna()
    except Exception:  # noqa: BLE001
        return None


def _annualised_irr(cashflows: list[float]) -> float | None:
    """Money-weighted annual return from a series of MONTHLY cashflows.

    cashflows[i] is the net flow at month i (contributions negative, the final
    portfolio value positive at the end). Solves NPV(r)=0 by bisection on the
    monthly rate, then annualises. Returns a fraction (0.08 = 8%/yr) or None.
    """
    def npv(r: float) -> float:
        # Running discount factor avoids (1+r)**i, which underflows to 0.0 for
        # r near -1 over long horizons and triggers a divide-by-zero.
        total, df, factor = 0.0, 1.0, 1.0 / (1.0 + r)
        for cf in cashflows:
            total += cf * df
            df *= factor
        return total
    lo, hi = -0.9999, 1.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:  # no sign change → can't bracket a root
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    monthly = (lo + hi) / 2
    return (1 + monthly) ** 12 - 1


def _simulate_dca(px: pd.DataFrame, weights_fn, monthly: float) -> dict[str, Any]:
    """Buy `monthly` dollars of the target weights on the 1st trading day of each
    month, holding everything (never selling). `weights_fn(date)` returns the
    deployment weights for that month (lets the caller apply a dip tilt).

    Returns total contributed, final value, money-weighted IRR, a time-weighted
    growth index (contribution-neutral) for CAGR + max drawdown.
    """
    months = px.groupby([px.index.year, px.index.month]).head(1).index  # first bar each month
    month_set = set(months)
    shares: dict[str, float] = {s: 0.0 for s in px.columns}
    contributed = 0.0
    flows: list[float] = []          # monthly cashflows for IRR
    twr = 1.0                        # contribution-neutral growth index
    twr_peak, max_dd = 1.0, 0.0
    v_prev = 0.0

    for ts in px.index:
        row = px.loc[ts]
        v_pre = float(sum(shares[s] * row[s] for s in px.columns))  # before today's flow
        if v_prev > 0:
            twr *= v_pre / v_prev
            twr_peak = max(twr_peak, twr)
            max_dd = min(max_dd, twr / twr_peak - 1.0)
        if ts in month_set:
            w = weights_fn(ts)
            for s in px.columns:
                if w.get(s, 0) > 0 and row[s] > 0:
                    shares[s] += (monthly * w[s]) / row[s]
            contributed += monthly
            flows.append(-monthly)
            v_post = float(sum(shares[s] * row[s] for s in px.columns))
        else:
            v_post = v_pre
        v_prev = v_post

    final_value = v_prev
    flows.append(final_value)  # terminal inflow
    irr = _annualised_irr(flows)
    n_years = max(1e-9, (px.index[-1] - px.index[0]).days / 365.25)
    twr_cagr = twr ** (1 / n_years) - 1
    return {
        "contributed": round(contributed, 2),
        "final_value": round(final_value, 2),
        "profit": round(final_value - contributed, 2),
        "total_return_pct": round((final_value / contributed - 1) * 100, 2) if contributed else 0.0,
        "irr_annual_pct": round(irr * 100, 2) if irr is not None else None,
        "twr_cagr_pct": round(twr_cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_contributions": len(flows) - 1,
    }


def _level_from_drawdown(dd: float) -> tuple[str, float]:
    """Historical timing proxy from drawdown alone (the live regime composite
    isn't available for past dates). Mirrors invest_timing's thresholds using
    only the drawdown component, so a backtest can label what the bot WOULD have
    called on a given day and tilt accordingly."""
    opp = _clamp01(0.65 * _clamp01(dd / _MAX_DD))
    if opp >= 0.60:
        level = "STRONG_BUY"
    elif opp >= 0.30:
        level = "ACCUMULATE"
    else:
        level = "NORMAL"
    tilt = float(DEFAULT_INVEST_CONFIG["dip_equity_tilt"]) * _clamp01((opp - 0.30) / 0.70) if opp >= 0.30 else 0.0
    return level, tilt


def backtest_picks(
    start_date: str,
    *,
    amount: float = 10_000.0,
    dsl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Point-in-time buy list → growth-to-today.

    "If on `start_date` you'd followed the bot's recommendation and invested
    `amount` across its allocation, what would it have said to buy, and how much
    has each holding grown since?" Applies the dip-tilt the bot WOULD have used
    given how far the market was off its highs on that date, labels the timing
    call (NORMAL/ACCUMULATE/STRONG_BUY), and compares the blended result to
    putting the whole amount in SPY. Dividend-adjusted; no look-ahead (weights
    use only data available on start_date).
    """
    dsl = {**DEFAULT_INVEST_CONFIG, **(dsl or {})}
    target = normalize_weights(dsl.get("allocation", DEFAULT_ALLOCATION))
    symbols = list(target)

    closes: dict[str, pd.Series] = {}
    for s in set(symbols) | {"SPY"}:
        ser = _full_history_close(s)
        if ser is None:
            try:
                df = md.get_history(s, interval="1d", range_="max")
                ser = df["close"] if not df.empty else None
            except Exception:  # noqa: BLE001
                ser = None
        if ser is not None and not ser.empty:
            closes[s] = ser

    have = [s for s in symbols if s in closes]
    if not have or "SPY" not in closes:
        raise ValueError("insufficient price history")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    # Drawdown of SPY as of start_date (trailing-1y high) → timing + tilt.
    spy = closes["SPY"]
    spy_hist = spy[spy.index <= start_ts]
    if len(spy_hist) < 20:
        raise ValueError(f"not enough history before {start_date}")
    trail_high = float(spy_hist.iloc[-252:].max())
    spy_at = float(spy_hist.iloc[-1])
    dd = max(0.0, (trail_high - spy_at) / trail_high) if trail_high > 0 else 0.0
    level, tilt = _level_from_drawdown(dd)

    w_target = normalize_weights({s: target[s] for s in have})
    deploy_w = _apply_dip_tilt(w_target, tilt, dsl) if tilt > 0 else w_target

    def price_on_or_after(s: str, ts: pd.Timestamp) -> tuple[float, str] | None:
        ser = closes[s]
        sub = ser[ser.index >= ts]
        if sub.empty:
            return None
        return float(sub.iloc[0]), sub.index[0].date().isoformat()

    holdings = []
    total_now = 0.0
    actual_start = None
    for s in have:
        buy = price_on_or_after(s, start_ts)
        if buy is None:
            continue
        buy_px, buy_day = buy
        now_px = float(closes[s].iloc[-1])
        invested = amount * deploy_w[s]
        shares = invested / buy_px if buy_px > 0 else 0.0
        cur_val = shares * now_px
        total_now += cur_val
        actual_start = actual_start or buy_day
        holdings.append({
            "symbol": s,
            "weight_pct": round(deploy_w[s] * 100, 1),
            "invested": round(invested, 2),
            "buy_price": round(buy_px, 2),
            "now_price": round(now_px, 2),
            "growth_pct": round((now_px / buy_px - 1) * 100, 1) if buy_px > 0 else 0.0,
            "current_value": round(cur_val, 2),
        })
    holdings.sort(key=lambda h: h["growth_pct"], reverse=True)

    # SPY-only comparison for the same amount/date.
    spy_buy = price_on_or_after("SPY", start_ts)
    spy_now = float(closes["SPY"].iloc[-1])
    spy_growth = (spy_now / spy_buy[0] - 1) * 100 if spy_buy else 0.0
    end_day = closes["SPY"].index[-1].date().isoformat()
    yrs = max(1e-9, (pd.Timestamp(end_day, tz="UTC") - pd.Timestamp(actual_start, tz="UTC")).days / 365.25)

    return {
        "requested_start": start_date,
        "actual_start": actual_start,
        "end": end_day,
        "years": round(yrs, 1),
        "amount": amount,
        "timing_at_start": {
            "level": level,
            "spy_drawdown_from_high_pct": round(dd * 100, 2),
            "equity_tilt_applied": round(tilt, 3),
        },
        "holdings": holdings,
        "portfolio": {
            "invested": round(amount, 2),
            "current_value": round(total_now, 2),
            "growth_pct": round((total_now / amount - 1) * 100, 1),
            "cagr_pct": round(((total_now / amount) ** (1 / yrs) - 1) * 100, 1),
        },
        "spy_only": {
            "growth_pct": round(spy_growth, 1),
            "current_value": round(amount * (1 + spy_growth / 100), 2),
        },
    }


def backtest_dca(
    dsl: dict[str, Any] | None = None,
    *,
    monthly_contribution: float = 1_000.0,
    range_: str = "max",
) -> dict[str, Any]:
    """Backtest dollar-cost-averaging into the bot's allocation, three ways:

      * DCA  — fixed monthly buys into the target weights.
      * DCA + dip-tilt — same, but shift new money toward equities when SPY is
        below its trailing-year high (the bot's dip-accumulation behaviour). The
        tilt uses only past data at each date — no look-ahead.
      * Benchmark — the same monthly cash put 100% into SPY.

    Dividend-adjusted closes. Limited to the window where every holding has data
    (so the start date is the latest holding's inception); the actual span is
    reported. This is the long-term analogue of the trading bot's win-rate report.
    """
    dsl = {**DEFAULT_INVEST_CONFIG, **(dsl or {})}
    target = normalize_weights(dsl.get("allocation", DEFAULT_ALLOCATION))
    symbols = list(target)

    closes: dict[str, pd.Series] = {}
    for s in set(symbols) | {"SPY"}:
        ser = _full_history_close(s)
        if ser is None:
            try:
                df = md.get_history(s, interval="1d", range_=range_)
                ser = df["close"] if not df.empty else None
            except Exception:  # noqa: BLE001
                ser = None
        if ser is not None and not ser.empty:
            closes[s] = ser
    missing = [s for s in symbols if s not in closes]
    have = [s for s in symbols if s in closes]
    if not have or "SPY" not in closes:
        raise ValueError("insufficient price history for a DCA backtest")

    # Align on the common window (all holdings present) — daily.
    px = pd.DataFrame({s: closes[s] for s in have}).dropna()
    if len(px) < 252:
        raise ValueError("not enough overlapping history (need ~1y)")
    w_target = normalize_weights({s: target[s] for s in have})

    # Trailing-1y-high drawdown of SPY at each date, for the dip tilt (past-only).
    spy = closes["SPY"].reindex(px.index).ffill()
    roll_high = spy.rolling(252, min_periods=20).max()
    dd_series = (1.0 - spy / roll_high).clip(lower=0.0).fillna(0.0)
    tilt_cfg = float(dsl.get("dip_equity_tilt", 0.20))

    def flat_w(_ts):
        return w_target

    def tilt_w(ts):
        dd = float(dd_series.loc[ts]) if ts in dd_series.index else 0.0
        opp = _clamp01(0.65 * _clamp01(dd / _MAX_DD))  # drawdown-only opportunity
        tilt = tilt_cfg * _clamp01((opp - 0.30) / 0.70) if opp >= 0.30 else 0.0
        return _apply_dip_tilt(w_target, tilt, dsl) if tilt > 0 else w_target

    spy_px = pd.DataFrame({"SPY": closes["SPY"]}).reindex(px.index).dropna()

    plain = _simulate_dca(px, flat_w, monthly_contribution)
    tilted = _simulate_dca(px, tilt_w, monthly_contribution)
    bench = _simulate_dca(spy_px, lambda _ts: {"SPY": 1.0}, monthly_contribution)

    return {
        "monthly_contribution": monthly_contribution,
        "start": px.index[0].date().isoformat(),
        "end": px.index[-1].date().isoformat(),
        "years": round((px.index[-1] - px.index[0]).days / 365.25, 1),
        "allocation": {k: round(v, 4) for k, v in w_target.items()},
        "excluded_no_history": missing,
        "dca": plain,
        "dca_dip_tilt": tilted,
        "benchmark_spy": bench,
    }
