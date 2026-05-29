"""Synthesizing opinion agent.

Orchestrates every signal in the platform into one structured analyst-style opinion:

  forecast (ensemble, 30d) + recommendation (regime-weighted blend) +
  price action (Smart Money confluence) + insider Form 4 + institutional 13F +
  famous-investor holdings + politician STOCK Act disclosures.

Two paths:

  - **LLM path** (when ANTHROPIC_API_KEY is set): Claude is given the entire
    structured signal pack as context and asked for a synthesized opinion with
    bull/bear/risk framing. Streams via the chat router; this module also
    exposes a non-streaming JSON-only synthesis.

  - **Deterministic path** (no API key): rule-based synthesis from the same
    signals — produces the same structured shape so the UI doesn't need to
    branch.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
import redis

from app.config import settings
from app.ml.ensemble import ensemble_forecast
from app.ml.regime import REGIME_WEIGHTS, detect
from app.services import analysts as analyst_svc
from app.services import earnings as earnings_svc
from app.services import indicators as ind
from app.services import insider as insider_svc
from app.services import institutions as inst_svc
from app.services import market_data as md
from app.services import politicians as pol_svc
from app.services import price_action as pa_svc
from app.services import sector_rs as sector_svc
from app.services.fundamentals_pit import _fetch_quarterly
from app.services.recommendation import recommend as compute_recommendation


_redis: redis.Redis | None = None
# In-process fallback when Redis is disabled / unreachable. Without this,
# every /ai/opinion call re-runs the 20+ sub-signal pipeline because Redis
# raises on every operation, so cache reads return None forever.
_mem_cache: dict[str, tuple[float, Any]] = {}


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _cache_get(key: str) -> Any | None:
    # Try Redis first (cheap when it's up).
    try:
        v = _cache().get(key)
        if v:
            return json.loads(v)
    except Exception:
        pass
    # In-process fallback.
    import time as _t
    entry = _mem_cache.get(key)
    if not entry:
        return None
    expires, value = entry
    if _t.time() > expires:
        _mem_cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any, ttl: int) -> None:
    import time as _t
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass
    # Always populate the in-process fallback too (cheap, GC'd on expiry).
    _mem_cache[key] = (_t.time() + ttl, value)
    # Light eviction so this can't grow unbounded.
    if len(_mem_cache) > 256:
        now = _t.time()
        for k in list(_mem_cache.keys()):
            if _mem_cache[k][0] <= now:
                _mem_cache.pop(k, None)


# -----------------------------------------------------------------------------
# Timeframe profiles — match the analysis to the user's holding period.
# A pro analyst would never run a swing-trade plan on weekly bars or a
# position-trade plan on 5-minute pivots. Each profile drives:
#   - which interval × range to fetch for the price-action engine
#   - which forecast horizon makes sense to cite
#   - human-readable horizon label that shows up in the opinion narrative
# -----------------------------------------------------------------------------
_TIMEFRAMES = {
    "intraday": {
        "label":          "Intraday (day trade)",
        "interval":       "15m",
        "range":          "1mo",
        "forecast_label": "next session",
        "horizon_text":   "hours to 1 day",
        "swing_window":   3,
    },
    "swing": {
        "label":          "Swing (1–3 weeks)",
        "interval":       "1d",
        "range":          "6mo",
        "forecast_label": "5–10 day",
        "horizon_text":   "1–3 weeks",
        "swing_window":   3,
    },
    "position": {
        "label":          "Position (1–3 months)",
        "interval":       "1d",
        "range":          "2y",
        "forecast_label": "30 day",
        "horizon_text":   "1–3 months",
        "swing_window":   5,
    },
    "longterm": {
        "label":          "Long-term (6m+)",
        "interval":       "1wk",
        "range":          "10y",
        "forecast_label": "quarterly",
        "horizon_text":   "6+ months",
        "swing_window":   3,
    },
}


# -----------------------------------------------------------------------------
# Signal collection — synchronous, ~2-5s for a fresh symbol
# -----------------------------------------------------------------------------

def collect_signals(symbol: str, use_cache: bool = True, timeframe: str = "position") -> dict[str, Any]:
    """Gather every signal we have. Each block is independently fault-tolerant.

    `timeframe` ∈ {intraday, swing, position, longterm} controls which chart
    bars feed the price-action engine (and the structure events, FVGs, zones,
    ATR — everything that's bar-derived). Other signals (fundamentals,
    insider, institutional, sentiment) are timeframe-agnostic by nature.

    Cached separately per `(symbol, timeframe)` so the chart view doesn't
    invalidate the position-trade analysis (or vice versa).
    """
    sym = symbol.upper()
    tf = _TIMEFRAMES.get(timeframe) or _TIMEFRAMES["position"]
    cache_key = f"opinion_signals:v3:{sym}:{timeframe}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached:
            return cached

    pack: dict[str, Any] = {
        "symbol": sym,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "timeframe_label": tf["label"],
        "horizon_text": tf["horizon_text"],
    }

    # price + history. The provider (Alpaca/yfinance) rate-limits under load
    # (e.g. while the background warmer runs), and a single failed 3y pull used
    # to abort the entire opinion with "no price data". Degrade gracefully:
    # try progressively shorter ranges — a 1y/6mo pull is smaller, more likely
    # cached, and still enough for last_price + regime. Only give up if every
    # range fails.
    df = None
    for _rng in ("3y", "2y", "1y", "6mo"):
        try:
            cand = md.get_history(sym, interval="1d", range_=_rng)
            if cand is not None and not cand.empty and "close" in cand.columns:
                df = cand
                break
        except Exception:
            continue
    if df is None or df.empty:
        return {"symbol": sym, "error": "no price data (provider rate-limited or unknown symbol)"}
    pack["last_price"] = float(df["close"].iloc[-1])
    try:
        pack["regime"] = detect(df["close"])
    except Exception:
        pack["regime"] = "unknown"

    pack["profile"] = md.get_profile(sym)
    pack["key_stats"] = md.get_key_stats(sym)

    # forecast (30d ensemble)
    try:
        if len(df) >= 200:
            r, regime, mix, fresh = ensemble_forecast(df, 21, sym)
            pack["forecast_30d"] = {
                "point": r.point,
                "p10": r.p10, "p90": r.p90,
                "direction_prob_up": r.direction_prob_up,
                "confidence": r.confidence,
                "regime": regime,
                "model_mix": mix,
                "drivers": r.drivers[:5],
                "contributions": r.contributions,
            }
    except Exception as e:
        pack["forecast_30d"] = {"error": str(e)}

    # recommendation engine (already blends most signals)
    try:
        pack["recommendation"] = compute_recommendation(sym, forecast_point=pack.get("forecast_30d", {}).get("point"))
    except Exception as e:
        pack["recommendation"] = {"error": str(e)}

    # price action (1y)
    try:
        # Price-action analysis is timeframe-aware — different intervals
        # produce different swings, FVGs, zones, structure events.
        df_pa = md.get_history(sym, interval=tf["interval"], range_=tf["range"])
        pa_result = pa_svc.analyze(df_pa, swing_window=tf["swing_window"])
        if pa_result.get("ok"):
            pack["price_action"] = {
                "trend": pa_result["current_trend"],
                "confluence": pa_result["confluence"],
                # Polarity-aware zones (NEW). active_* are the only ones a
                # trader actually plans entries around. flipped_* are former
                # S/R that price has already broken — kept as context, not as
                # actionable entry levels.
                "active_demand_zones": (pa_result.get("active_demand") or [])[:3],
                "active_supply_zones": (pa_result.get("active_supply") or [])[:3],
                "flipped_resistance":  (pa_result.get("flipped_resistance") or [])[:2],
                "flipped_support":     (pa_result.get("flipped_support") or [])[:2],
                "price_in_discovery":  bool(pa_result.get("price_in_discovery")),
                "nearest_demand_pct_below": pa_result.get("nearest_demand_pct_below"),
                "nearest_supply_pct_above": pa_result.get("nearest_supply_pct_above"),
                "recent_structure": pa_result["structure_events"][-3:],
                "active_fvgs": [f for f in pa_result["fvgs"] if not f["filled"]][-5:],
                "unmitigated_obs": [o for o in pa_result["order_blocks"] if not o["mitigated"]][-5:],
                "recent_sweeps": pa_result["liquidity_sweeps"][-3:],
            }
    except Exception as e:
        pack["price_action"] = {"error": str(e)}

    # insider (Form 4)
    try:
        pack["insider"] = insider_svc.insider_signal(sym)
        pack["insider"]["recent_transactions"] = insider_svc.get_insider_transactions(sym, limit=10)
    except Exception as e:
        pack["insider"] = {"error": str(e)}

    # institutional + famous
    try:
        pack["institutional"] = inst_svc.institutional_signal(sym)
        pack["institutional"]["top_institutional_holders"] = inst_svc.get_institutional_holders(sym)[:8]
    except Exception as e:
        pack["institutional"] = {"error": str(e)}

    # politicians
    try:
        pack["politicians"] = pol_svc.political_signal(sym)
        pack["politicians"]["recent_trades"] = pol_svc.trades_for_symbol(sym, limit=10)
    except Exception as e:
        pack["politicians"] = {"error": str(e)}

    # earnings — next event + recent-surprise signal + PEAD drift
    try:
        nxt = earnings_svc.next_earnings(sym)
        surp = earnings_svc.consensus_surprise_signal(sym)
        pead = earnings_svc.pead_signal(sym)
        pack["earnings"] = {
            "next": nxt,
            "surprise_signal": surp,
            "pead_signal": pead,
        }
    except Exception as e:
        pack["earnings"] = {"error": str(e)}

    # analyst consensus + revisions
    try:
        pack["analysts"] = analyst_svc.analyst_signal(sym)
    except Exception as e:
        pack["analysts"] = {"error": str(e)}

    # sector relative strength
    try:
        pack["sector_rs"] = sector_svc.sector_relative_strength(sym)
    except Exception as e:
        pack["sector_rs"] = {"error": str(e)}

    # sentiment momentum (Δ between 7d and 30d)
    try:
        from app.services.news import sentiment_momentum
        pack["sentiment_momentum"] = sentiment_momentum(sym)
    except Exception as e:
        pack["sentiment_momentum"] = {"error": str(e)}

    # options unusual activity (free yfinance heuristic)
    try:
        from app.services.options_flow import unusual_activity
        pack["options"] = unusual_activity(sym)
    except Exception as e:
        pack["options"] = {"error": str(e)}

    # News topic classification (zero-shot BART) — only runs if model loaded.
    # Disable by setting USE_TOPIC_CLASSIFIER=0 (avoids the ~1.6GB model download).
    try:
        from app.services import news as news_svc
        from app.services.news_topics import topic_signal
        articles = news_svc.fetch_news(sym, limit=15)
        pack["news_topics"] = topic_signal(sym, articles)
    except Exception as e:
        pack["news_topics"] = {"error": str(e)}

    # Social sentiment — StockTwits (always free, no auth) + Reddit (key-gated)
    try:
        from app.services.social import social_signal
        pack["social"] = social_signal(sym)
    except Exception as e:
        pack["social"] = {"error": str(e)}

    # Most-recent earnings call transcript sentiment (Motley Fool scrape)
    try:
        from app.services.transcripts import transcript_signal
        pack["transcript"] = transcript_signal(sym)
    except Exception as e:
        pack["transcript"] = {"error": str(e)}

    # PIT fundamentals (EDGAR — full history, real filing dates)
    try:
        qf = _fetch_quarterly(sym)
        if qf is not None and not qf.empty:
            latest = qf.dropna(subset=["revenue"]).tail(1)
            ttm = qf.dropna(subset=["eps_ttm"]).tail(1)
            yoy = qf.dropna(subset=["revenue_growth_yoy"]).tail(1)
            fpack: dict[str, Any] = {}
            if not latest.empty:
                r = latest.iloc[0]
                fpack["latest_quarter"] = r["period_end"].date().isoformat()
                fpack["filed_on"] = r["available_at"].date().isoformat()
                fpack["revenue_latest"] = float(r["revenue"]) if pd.notna(r.get("revenue")) else None
                fpack["net_income_latest"] = float(r["net_income"]) if pd.notna(r.get("net_income")) else None
                fpack["eps_latest"] = float(r["eps"]) if pd.notna(r.get("eps")) else None
                # Sanity-flag ratios derived from a tiny denominator. Femasys-style
                # microcaps with $1-2M quarterly revenue can produce absurd margins
                # (e.g. one-time non-cash gains divided by near-zero revenue → "199%
                # net margin"). Mark them unreliable so the case-builder can skip.
                _rev = fpack.get("revenue_latest") or 0
                _nm = float(r["net_margin"]) if pd.notna(r.get("net_margin")) else None
                fpack["net_margin"] = _nm
                fpack["net_margin_reliable"] = bool(_nm is not None and abs(_rev) > 10_000_000)
                fpack["debt_to_assets"] = float(r["debt_to_assets"]) if pd.notna(r.get("debt_to_assets")) else None
            if not yoy.empty:
                r = yoy.iloc[0]
                fpack["revenue_growth_yoy"] = float(r["revenue_growth_yoy"]) if pd.notna(r.get("revenue_growth_yoy")) else None
                fpack["eps_growth_yoy"] = float(r["eps_growth_yoy"]) if pd.notna(r.get("eps_growth_yoy")) else None
            if not ttm.empty:
                fpack["eps_ttm"] = float(ttm.iloc[0]["eps_ttm"])
                # PE on TTM if we have a price
                if pack.get("last_price") and fpack["eps_ttm"] > 0:
                    fpack["pe_ttm"] = round(pack["last_price"] / fpack["eps_ttm"], 2)
            pack["fundamentals_pit"] = fpack
    except Exception as e:
        pack["fundamentals_pit"] = {"error": str(e)}

    # technical snapshot
    try:
        close = df["close"]; high = df["high"]; low = df["low"]
        pack["technical"] = {
            "rsi_14": float(ind.rsi(close, 14).iloc[-1]),
            "macd_hist": float(ind.macd(close)["hist"].iloc[-1]),
            "atr_pct": float(ind.atr(high, low, close, 14).iloc[-1] / pack["last_price"]),
            "above_sma_50": float(close.iloc[-1]) > float(ind.sma(close, 50).iloc[-1]),
            "above_sma_200": float(close.iloc[-1]) > float(ind.sma(close, 200).iloc[-1])
                              if len(close) >= 200 else None,
            "ret_3m": float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0.0,
            "ret_1y": float(close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else 0.0,
        }
    except Exception as e:
        pack["technical"] = {"error": str(e)}

    # Cache the assembled pack for 5 minutes. Concurrent /ai/opinion calls on
    # the same symbol then hit Redis instead of re-running 13 sub-fetches.
    if use_cache:
        _cache_set(cache_key, pack, ttl=300)
    return pack


# -----------------------------------------------------------------------------
# Structured opinion shape (both paths return this)
# -----------------------------------------------------------------------------

OpinionVerdict = Literal["STRONG_BUY", "BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL", "STRONG_SELL"]


def _empty_opinion(symbol: str, error: str | None = None) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "HOLD",
        "score": 0.0,
        "confidence": 0.0,
        "time_horizon": "30 days",
        "thesis": error or "Insufficient data to form an opinion.",
        "bull_case": [],
        "bear_case": [],
        "key_risks": [],
        "entry_zone": None,
        "stop": None,
        "target_1": None,
        "target_2": None,
        "suggested_position_pct": 0.0,
        "signal_summary": {},
        "generator": "deterministic",
    }


# -----------------------------------------------------------------------------
# Deterministic synthesizer (used when no LLM key)
# -----------------------------------------------------------------------------

def _verdict_from_score(score: float) -> OpinionVerdict:
    if score > 0.55: return "STRONG_BUY"
    if score > 0.25: return "BUY"
    if score > 0.10: return "ACCUMULATE"
    if score < -0.55: return "STRONG_SELL"
    if score < -0.25: return "SELL"
    if score < -0.10: return "REDUCE"
    return "HOLD"


# Risk-profile thresholds.
#
# Literature anchors (read these before touching values):
#   - Stop multiple in ATR units: LeBeau (1992) Chandelier Exit uses 2.5–3× ATR.
#     Wilder (1978) "New Concepts in Technical Trading Systems" introduced ATR.
#     Van Tharp (1998) "Trade Your Way to Financial Freedom" uses 2–3× as the
#     standard band. → 2× / 2.5× / 3× spans Tharp's accepted range.
#   - Per-trade risk budget: Van Tharp (1998) recommends 0.5–1% portfolio risk
#     for active traders; 2% is the upper bound before drawdowns explode.
#     → 1% / 0.5% / 0.25% scales within his band; speculative goes lowest
#     because microcaps have fatter tails than the model assumes.
#   - Zone reachability in ATR units (not raw %): a fixed % threshold is wrong
#     for cross-asset use — 10% is a normal day for FEMY but a 3σ event for
#     AAPL. Using N×ATR self-scales. Empirical Wyckoff/SMC literature treats a
#     setup as "reachable" if the demand zone is within ~3× recent ATR for
#     swing trades (3–6 weeks). I chose 3 / 5 / 8 ATR-multiples by analogy to
#     Bollinger's 2σ envelope (Bollinger 1980) and ATR-stretch heuristics in
#     CompuTrac / TradeStation user manuals. Honest grade: structure motivated,
#     specific numbers calibrated by hand.
#   - Zone max width: regardless of profile, a zone wider than ~2× ATR is
#     noise — you can't place a meaningful entry inside it. This is universal,
#     not profile-dependent.
#   - Minimum |score| to act: no direct literature; this is a confidence-gating
#     heuristic. Lower thresholds for speculative reflect that you tolerate
#     more false-positives when expected gains are larger.
_RISK_PROFILES = {
    "conservative": {
        "min_score":             0.10,   # heuristic — conviction floor
        "zone_max_dist_atr":     3.0,    # ≤ 3× ATR from current price
        "zone_max_width_atr":    2.0,    # ≤ 2× ATR wide
        "stop_atr_mult":         2.0,    # Wilder/Tharp lower band
        "trade_risk_pct":        1.0,    # 1% of capital per trade (Van Tharp)
        "pos_size_min":          0.5,    # never trivially small
        "pos_size_max":          10.0,
        "fallback_targets":      False,  # require forecast for targets
    },
    "balanced": {
        "min_score":             0.07,
        "zone_max_dist_atr":     5.0,
        "zone_max_width_atr":    2.0,
        "stop_atr_mult":         2.5,    # LeBeau midpoint
        "trade_risk_pct":        0.5,    # Van Tharp default
        "pos_size_min":          0.25,
        "pos_size_max":          5.0,
        "fallback_targets":      True,   # PA-derived targets ok
    },
    "speculative": {
        "min_score":             0.04,
        "zone_max_dist_atr":     8.0,    # microcaps move 5-10× ATR in a session
        "zone_max_width_atr":    2.0,
        "stop_atr_mult":         3.0,    # LeBeau Chandelier upper / Tharp upper
        "trade_risk_pct":        0.25,   # cut risk because tail fatness ↑
        "pos_size_min":          0.10,
        "pos_size_max":          2.0,    # max 2% in a lottery-ticket name
        "fallback_targets":      True,
    },
}


def synthesize_deterministic(symbol: str, signals: dict[str, Any], risk_mode: str = "conservative") -> dict[str, Any]:
    if signals.get("error") or not signals.get("last_price"):
        return _empty_opinion(symbol, error=signals.get("error"))

    risk_p = _RISK_PROFILES.get(risk_mode, _RISK_PROFILES["conservative"])

    sym = signals["symbol"]
    last_price = signals["last_price"]
    regime = signals.get("regime", "sideways")

    # Component scores in [-1, +1]
    rec = signals.get("recommendation") or {}
    rec_score = float(rec.get("score") or 0.0)

    pa = signals.get("price_action") or {}
    pa_score = float((pa.get("confluence") or {}).get("score") or 0.0)

    insider_score = float((signals.get("insider") or {}).get("score") or 0.0)
    inst_score = float((signals.get("institutional") or {}).get("score") or 0.0)
    pol_score = float((signals.get("politicians") or {}).get("score") or 0.0)
    analyst_score = float((signals.get("analysts") or {}).get("score") or 0.0)

    fc = signals.get("forecast_30d") or {}
    if "point" in fc:
        fc_move = (fc["point"] - last_price) / last_price
    else:
        fc_move = 0.0

    # Weighted blend. The regime weights table already covers
    # forecast/technical/fundamental/sentiment/macro/price_action — we layer
    # insider/institutional/political on top as "smart-money tilt".
    w = REGIME_WEIGHTS[regime]
    base_weights = {
        "recommendation": 0.40,   # already a blend (technical+fundamental+sentiment+macro+PA)
        "forecast_move":   0.10,
        "price_action":    w.get("price_action", 0.20) * 0.5,
        "analysts":        0.12,  # NEW — consensus + revisions, the "real edge" item
        "insider":         0.08,
        "institutional":   0.06,
        "political":       0.05,
    }
    s = sum(base_weights.values())
    base_weights = {k: v / s for k, v in base_weights.items()}

    score = (
        base_weights["recommendation"] * rec_score
        + base_weights["forecast_move"] * max(-1.0, min(1.0, fc_move * 6.0))  # ±17% saturates
        + base_weights["price_action"] * pa_score
        + base_weights["analysts"] * analyst_score
        + base_weights["insider"] * insider_score
        + base_weights["institutional"] * inst_score
        + base_weights["political"] * pol_score
    )
    score = max(-1.0, min(1.0, score))

    # Confidence: signal agreement + recommendation confidence
    signs = [s for s in (rec_score, fc_move, pa_score, analyst_score, insider_score, inst_score, pol_score) if abs(s) > 0.05]
    agreement = (max(sum(1 for x in signs if x > 0), sum(1 for x in signs if x < 0)) / len(signs)) if signs else 0.5
    confidence = max(0.1, min(0.95, 0.35 + 0.55 * agreement)) * max(0.5, float(rec.get("confidence") or 0.6))

    verdict = _verdict_from_score(score)

    # Build bull/bear/risks narrative from drivers.
    # `risks` is initialized here (and not later in the function) so the social
    # and forecast blocks above can append to it (microcap pump detection,
    # forecast-band-too-wide warning, etc.).
    bull: list[str] = []
    bear: list[str] = []
    risks: list[str] = []

    if rec.get("reasoning", {}).get("technical", {}).get("bullish"):
        bull.extend(rec["reasoning"]["technical"]["bullish"])
    if rec.get("reasoning", {}).get("technical", {}).get("bearish"):
        bear.extend(rec["reasoning"]["technical"]["bearish"])
    # Split fundamental highlights into individual bullets and route each one
    # to bull/bear by its content — the original code dumped the whole
    # semicolon-joined summary into one bucket, producing absurdities like
    # "PE 94.5 (expensive); EPS positive (1.35); High beta" as a bull point.
    fund_h = (rec.get("reasoning", {}).get("fundamental") or {}).get("highlights") or []
    for h in fund_h:
        hl = h.lower()
        if "expensive" in hl or "eps negative" in hl:
            bear.append(h)
        elif "cheap" in hl or "eps positive" in hl:
            bull.append(h)
        elif "high beta" in hl:
            risks.append(h)
        elif "low beta" in hl:
            # Defensive characteristic — bullish in risk-off, neutral otherwise.
            bull.append(h)

    # PIT fundamentals — cite real numbers in the case construction
    fp = signals.get("fundamentals_pit") or {}
    if fp.get("revenue_growth_yoy") is not None:
        g = fp["revenue_growth_yoy"] * 100
        if g > 10:
            bull.append(f"Revenue +{g:.1f}% YoY in Q ending {fp.get('latest_quarter')}")
        elif g < -5:
            bear.append(f"Revenue {g:+.1f}% YoY in Q ending {fp.get('latest_quarter')}")
    if fp.get("eps_growth_yoy") is not None:
        g = fp["eps_growth_yoy"] * 100
        if g > 15:
            bull.append(f"EPS +{g:.1f}% YoY (TTM EPS ${fp.get('eps_ttm', 0):.2f}, PE_TTM {fp.get('pe_ttm', 'n/a')})")
        elif g < -10:
            bear.append(f"EPS {g:+.1f}% YoY")
    # Only cite margin when the denominator (revenue) is big enough to make it
    # a meaningful business-quality signal — see net_margin_reliable above.
    if fp.get("net_margin") is not None and fp.get("net_margin_reliable"):
        nm = fp["net_margin"]
        if 0.18 < nm < 1.0:
            bull.append(f"Net margin {nm*100:.1f}% (high-quality business)")
        elif nm < 0.0:
            bear.append(f"Operating at a loss (net margin {nm*100:.1f}%)")
    if fp.get("debt_to_assets") is not None and fp["debt_to_assets"] > 0.45:
        bear.append(f"Elevated leverage (debt/assets {fp['debt_to_assets']*100:.0f}%)")

    # Analyst consensus + revisions (the "real edge" item)
    a = signals.get("analysts") or {}
    rev = a.get("revisions") or {}
    cons = a.get("consensus") or {}
    if rev.get("n_upgrades", 0) > rev.get("n_downgrades", 0) and rev.get("score", 0) > 0.10:
        bull.append(f"Analyst revisions: {rev['summary']}")
    elif rev.get("n_downgrades", 0) > rev.get("n_upgrades", 0) and rev.get("score", 0) < -0.10:
        bear.append(f"Analyst revisions: {rev['summary']}")
    if cons.get("growth_yoy") is not None:
        g = cons["growth_yoy"] * 100
        if g > 10 and (cons.get("n_analysts") or 0) >= 3:
            bull.append(f"Forward consensus EPS growth {g:+.1f}% YoY ({int(cons['n_analysts'])} analysts)")
        elif g < -5 and (cons.get("n_analysts") or 0) >= 3:
            bear.append(f"Forward consensus EPS contraction {g:+.1f}% YoY ({int(cons['n_analysts'])} analysts)")

    # Sector relative strength
    srs = signals.get("sector_rs") or {}
    if srs.get("score", 0) > 0.20:
        bull.append(f"Sector RS: {srs['summary']}")
    elif srs.get("score", 0) < -0.20:
        bear.append(f"Sector RS: {srs['summary']}")

    # Sentiment momentum (Δ-sentiment is a stronger signal than the level)
    sm = signals.get("sentiment_momentum") or {}
    if sm.get("score", 0) > 0.15:
        bull.append(f"News sentiment improving (Δ {sm.get('delta', 0):+.2f})")
    elif sm.get("score", 0) < -0.15:
        bear.append(f"News sentiment deteriorating (Δ {sm.get('delta', 0):+.2f})")

    # Options unusual activity (free yfinance heuristic; not real flow data)
    opt = signals.get("options") or {}
    if opt.get("score", 0) > 0.25:
        bull.append(f"Options: {opt.get('summary')}")
    elif opt.get("score", 0) < -0.25:
        bear.append(f"Options: {opt.get('summary')}")

    # News topic mix (only when classifier is active)
    nt = signals.get("news_topics") or {}
    if nt.get("score", 0) > 0.20:
        bull.append(f"News flow: {nt.get('summary')}")
    elif nt.get("score", 0) < -0.20:
        bear.append(f"News flow: {nt.get('summary')}")

    # Social sentiment (StockTwits + Reddit). On microcap / penny names a
    # one-sided bull skew with no bears is almost always a pump — flag as risk
    # instead of crediting the bull case. Use market cap as the sniff test.
    soc = signals.get("social") or {}
    _mcap = (signals.get("profile") or {}).get("market_cap") or 0
    # Market-cap cutoffs: MSCI Global Industry Classification — micro <$300M,
    # small $300M–$2B. Russell uses $250M / $2B. We use MSCI numbers.
    # The pump-pattern detection on one-sided social activity follows the
    # spirit of Sabherwal, Sarkar, Zhang (2011) "Do Internet Stock Message
    # Boards Influence Trading?" — manipulation correlates with one-sided
    # message bursts in low-float names. We don't have message-velocity
    # tracking yet, so we use static count thresholds as a proxy.
    _is_microcap = bool(_mcap and _mcap < 300_000_000)
    _is_smallcap = bool(_mcap and _mcap < 2_000_000_000)
    _st = soc.get("stocktwits") or {}
    _n = int(_st.get("n") or 0)
    _bull_count = int(round(_n * (float(_st.get("bullish_pct") or 0) / 100)))
    _bear_count = int(round(_n * (float(_st.get("bearish_pct") or 0) / 100)))
    # Pump pattern: enough messages to be meaningful, zero dissent, no
    # institutional / insider corroboration. Severity scales with market cap.
    _suspicious_strong = _is_microcap   and _bull_count >= 10 and _bear_count == 0
    _suspicious_soft   = (not _is_microcap) and _bull_count >= 10 and _bear_count == 0
    if _suspicious_strong:
        risks.append("Microcap with one-sided bullish social activity (no dissent) — possible pump; do not size up on this signal")
    elif _suspicious_soft:
        # Mid/large cap with 30 bulls and 0 bears — unusual; flag softly, don't kill the bull case.
        risks.append(f"Social buzz is one-sided ({_bull_count} bullish, 0 bearish) — sanity-check before sizing up")
        if soc.get("hype_score", 0) > 0.50 and soc.get("score", 0) > 0.20:
            bull.append(f"Retail buzz: {soc.get('summary')}")
    elif soc.get("hype_score", 0) > 0.50 and soc.get("score", 0) > 0.20:
        bull.append(f"Retail buzz: {soc.get('summary')}")
    elif soc.get("hype_score", 0) > 0.50 and soc.get("score", 0) < -0.20:
        bear.append(f"Retail negativity: {soc.get('summary')}")

    # Earnings call transcript sentiment (Q&A weighted)
    tr = signals.get("transcript") or {}
    if tr.get("score", 0) > 0.20:
        bull.append(f"Earnings call: {tr.get('summary')}")
    elif tr.get("score", 0) < -0.20:
        bear.append(f"Earnings call: {tr.get('summary')}")

    # earnings catalyst
    earn = signals.get("earnings") or {}
    surp = earn.get("surprise_signal") or {}
    nxt = earn.get("next")
    if surp.get("score", 0) > 0.15:
        bull.append(f"Earnings surprises trend positive (latest {surp.get('latest_surprise_pct', 0):+.1f}%)")
    elif surp.get("score", 0) < -0.15:
        bear.append(f"Earnings surprises trend negative (latest {surp.get('latest_surprise_pct', 0):+.1f}%)")
    # PEAD drift expectation
    pead = earn.get("pead_signal") or {}
    if pead.get("score", 0) > 0.10:
        bull.append(f"PEAD: {pead.get('summary')}")
    elif pead.get("score", 0) < -0.10:
        bear.append(f"PEAD: {pead.get('summary')}")
    if nxt and nxt.get("days_to_earnings") is not None:
        d = nxt["days_to_earnings"]
        if 0 < d <= 14:
            # near-term earnings is a binary-event risk regardless of direction
            pass  # handled in risks below

    if pa_score > 0.15:
        for d in (pa.get("confluence") or {}).get("drivers", [])[:3]:
            bull.append(f"Price action: {d}")
    elif pa_score < -0.15:
        for d in (pa.get("confluence") or {}).get("drivers", [])[:3]:
            bear.append(f"Price action: {d}")

    ins = signals.get("insider") or {}
    if ins.get("score", 0) > 0.1:
        bull.append(f"Insider buying: {ins.get('summary')}")
    elif ins.get("score", 0) < -0.1:
        bear.append(f"Insider selling: {ins.get('summary')}")

    inst = signals.get("institutional") or {}
    if inst.get("n_famous", 0) > 0:
        names = [h["investor"] for h in inst.get("famous_holders", [])[:3]]
        bull.append(f"Held by notable investors: {', '.join(names)}")

    pol = signals.get("politicians") or {}
    if pol.get("score", 0) > 0.1:
        bull.append(f"Political accumulation: {pol.get('summary')}")
    elif pol.get("score", 0) < -0.1:
        bear.append(f"Political selling: {pol.get('summary')}")

    if fc and "point" in fc and fc.get("p10") is not None and fc.get("p90") is not None:
        band_pct = (fc["p90"] - fc["p10"]) / last_price if last_price else 1.0
        if band_pct <= 0.50:
            bull_or_bear = bull if fc_move > 0 else bear
            bull_or_bear.append(
                f"30d forecast {fc_move*100:+.1f}% (range "
                f"{(fc['p10'] - last_price)/last_price*100:+.1f}% to "
                f"{(fc['p90'] - last_price)/last_price*100:+.1f}%, "
                f"P(up)={fc['direction_prob_up']*100:.0f}%)"
            )
        else:
            # Forecast interval wider than 50% — model has no edge on this name.
            # Cite the uncertainty as a risk rather than printing a point estimate.
            risks.append(
                f"Forecast interval too wide to be actionable "
                f"(±{band_pct*50:.0f}% spread); model has no edge here"
            )

    # Risks (the list was initialized higher so social/forecast blocks could append)
    tech = signals.get("technical") or {}
    atr_pct = tech.get("atr_pct")
    if atr_pct and atr_pct > 0.04:
        risks.append(f"Elevated volatility (ATR {atr_pct*100:.1f}% of price)")
    if regime in ("bear_high_vol", "bull_high_vol"):
        risks.append(f"High-vol regime ({regime}) — wider stops needed")
    if rec.get("reasoning", {}).get("risk", {}).get("max_drawdown_90d_pct", 0) < -15:
        risks.append(f"Recent drawdown {rec['reasoning']['risk']['max_drawdown_90d_pct']:.1f}% in last 90d")
    if nxt and nxt.get("days_to_earnings") is not None and 0 < nxt["days_to_earnings"] <= 14:
        risks.append(f"Earnings in {nxt['days_to_earnings']:.0f} days — binary event risk (consider smaller size or post-event entry)")
    if not risks:
        risks.append("Standard market risk; respect position sizing.")

    # Forecast usefulness gate. Two criteria, BOTH must hold for the point
    # estimate to drive trade-plan targets:
    #   (i) Interval not pathologically wide: p90-p10 ≤ 50% of price.
    #       Heuristic — no specific literature, but motivated by Henriksson-
    #       Merton (1981) "On Market Timing and Investment Performance" — at
    #       extreme uncertainty the directional bet is uninformative.
    #  (ii) Directional probability has actual edge: |P(up) − 0.5| > 0.10.
    #       Source: Granger & Pesaran (2000) "Economic and Statistical Measures
    #       of Forecast Accuracy", JoF — directional accuracy must be
    #       significantly different from 0.5 to be economically actionable.
    _band_ok = bool(
        fc and "point" in fc and last_price > 0
        and (fc.get("p90") is not None and fc.get("p10") is not None)
        and ((fc["p90"] - fc["p10"]) / last_price) <= 0.50
    )
    _dir_ok = bool(fc and abs((fc.get("direction_prob_up") or 0.5) - 0.5) > 0.10)
    fc_usable = bool(_band_ok and _dir_ok)

    # Entry / stop / target — only emit when verdict has conviction.
    # HOLDs (|score| < 0.10) deliberately get no trade plan: handing out
    # entry/stop/target numbers for a "no opinion" verdict is what produced
    # the contradictory FEMY output (supply-zone entry with long-side stop).
    rec_risk = (rec.get("reasoning") or {}).get("risk") or {}
    entry_zone = None
    suggested_stop = None
    target_1 = target_2 = None
    direction: str | None = None

    # A zone is only useful as an entry if it's *reachable from current price*.
    # Reachability gauged in ATR units (Wilder 1978) rather than raw %, so the
    # gate self-scales — a "3 ATR away" zone is normal-distance regardless of
    # whether the ticker is AAPL or FEMY. This is the correct cross-asset
    # generalization; previous version used fixed % thresholds that broke on
    # low-priced high-vol microcaps.
    _zone_dist_atr = risk_p["zone_max_dist_atr"]
    _zone_width_atr = risk_p["zone_max_width_atr"]
    # Convert to raw-price units using last_price × atr_pct.
    _abs_atr = (atr_pct or 0.0) * last_price
    def _pickable_zone(zones, direction: str):
        if not zones or last_price <= 0 or _abs_atr <= 0:
            return None
        for z in zones:
            mid = (z["lower"] + z["upper"]) / 2
            dist_atr = (mid - last_price) / _abs_atr
            width_atr = (z["upper"] - z["lower"]) / _abs_atr
            if width_atr > _zone_width_atr:
                continue
            if direction == "long":
                # Demand zone at-or-below price, within N×ATR drawdown distance.
                # Allow up to +0.5 ATR above price (already-broken-through level).
                if -_zone_dist_atr <= dist_atr <= 0.5:
                    return z
            else:
                if -0.5 <= dist_atr <= _zone_dist_atr:
                    return z
        return None

    if abs(score) >= risk_p["min_score"] and atr_pct > 0:
        direction = "long" if score > 0 else "short"
        stop_mult = risk_p["stop_atr_mult"]
        if direction == "long":
            # Long: entry at demand, stop below, targets above.
            z = _pickable_zone(pa.get("active_demand_zones") or [], "long")
            if z is not None:
                entry_zone = {"lower": z["lower"], "upper": z["upper"], "type": "demand_zone"}
                suggested_stop = rec_risk.get("suggested_stop") or last_price * (1 - stop_mult * atr_pct)
                if fc_usable:
                    target_1 = round(fc["point"], 2)
                    target_2 = round(fc["p90"], 2)
                elif risk_p["fallback_targets"]:
                    # Forecast useless; use price-action geometry — next supply
                    # zone above, then a 2R extension. Lets balanced/speculative
                    # still produce actionable plans on noisy names.
                    sups = sorted(pa.get("active_supply_zones") or [],
                                  key=lambda s: ((s["lower"] + s["upper"]) / 2))
                    above = [s for s in sups if (s["lower"] + s["upper"]) / 2 > last_price]
                    if above:
                        target_1 = round((above[0]["lower"] + above[0]["upper"]) / 2, 2)
                        # 2R extension on the ATR risk
                        target_2 = round(last_price * (1 + 4 * atr_pct), 2)
        else:
            # Short: entry at supply, stop ABOVE, targets below.
            z = _pickable_zone(pa.get("active_supply_zones") or [], "short")
            if z is not None:
                entry_zone = {"lower": z["lower"], "upper": z["upper"], "type": "supply_zone"}
                suggested_stop = last_price * (1 + stop_mult * atr_pct)
                if fc_usable:
                    target_1 = round(fc["point"], 2)
                    target_2 = round(fc["p10"], 2)
                elif risk_p["fallback_targets"]:
                    dems = sorted(pa.get("active_demand_zones") or [],
                                  key=lambda d: ((d["lower"] + d["upper"]) / 2), reverse=True)
                    below = [d for d in dems if (d["lower"] + d["upper"]) / 2 < last_price]
                    if below:
                        target_1 = round((below[0]["lower"] + below[0]["upper"]) / 2, 2)
                        target_2 = round(last_price * (1 - 4 * atr_pct), 2)

        # Don't emit half-baked trade plans. If we couldn't find a reachable
        # entry zone, or no targets, drop the whole plan and tell the user WHY.
        if entry_zone is None:
            switch_hint = "" if risk_mode == "speculative" else \
                f" Try risk_mode={'balanced' if risk_mode == 'conservative' else 'speculative'} for wider tolerance."
            msg = (f"Price extended above nearest demand zone (>{int(_zone_dist_atr)}× ATR away) — "
                   f"wait for pullback rather than chase.{switch_hint}") if direction == "long" else \
                  (f"Price already below nearest supply zone (>{int(_zone_dist_atr)}× ATR away) — "
                   f"no clean short entry.{switch_hint}")
            risks.insert(0, msg)
            suggested_stop = None
            target_1 = target_2 = None
            direction = None
        elif target_1 is None and target_2 is None:
            risks.insert(0, "Trade plan suppressed: forecast has no edge, no fallback targets available")
            entry_zone = None
            suggested_stop = None
            direction = None

        # Sanity check: if any target is on the wrong side of entry, drop it
        # rather than print a contradictory plan.
        if entry_zone and target_1 is not None:
            entry_mid = (entry_zone["lower"] + entry_zone["upper"]) / 2
            wrong_side = (direction == "long"  and target_1 <= entry_mid) or \
                         (direction == "short" and target_1 >= entry_mid)
            if wrong_side:
                target_1 = None
        if entry_zone and target_2 is not None:
            entry_mid = (entry_zone["lower"] + entry_zone["upper"]) / 2
            wrong_side = (direction == "long"  and target_2 <= entry_mid) or \
                         (direction == "short" and target_2 >= entry_mid)
            if wrong_side:
                target_2 = None

    # Position sizing: volatility-targeted + capped at quarter-Kelly.
    #
    # 1) **Vol-targeted base**: pick a target portfolio risk budget per position
    #    (we use 0.5% portfolio risk per trade by default — Larry Connors / TF
    #    rule-of-thumb). Position size = target_risk_pct / (ATR_pct × stop_mult).
    #    A 60% IV name and a 15% IV name will get very different sizes.
    #
    # 2) **Quarter-Kelly cap**: Kelly fraction f* = edge / odds.
    #    We approximate edge ≈ |score| × confidence × 0.05  (5% expected per
    #    point of conviction) and odds ≈ ATR_pct × √horizon. Full Kelly is
    #    famously aggressive — we cap at 25% Kelly which is the empirically
    #    safer choice (Thorp 2006, MacLean/Thorp/Ziemba 2010).
    #
    # 3) **Floor + ceiling**: never larger than 10% of capital, never smaller
    #    than 0.5% of capital (otherwise the trade can't move the needle).
    # Position sizing — two methods, take the smaller (more conservative):
    #
    # (a) Vol-targeted risk budget: size such that a stop-out loses exactly
    #     `trade_risk_pct` of capital. Position % = trade_risk_pct / (ATR_pct × stop_mult).
    #     Source: Van Tharp (1998) "Trade Your Way to Financial Freedom" Ch. 12 —
    #     fixed-fractional risk model, the foundation of professional sizing.
    #
    # (b) ¼-Kelly cap: Kelly fraction f* = edge / variance. We approximate
    #     edge ≈ |score| × confidence × 5%  (a unit-conviction trade returns ~5%
    #     in expectation — calibrated from our backtest's average winning trade,
    #     NOT pulled from thin air) and variance via ATR over the horizon.
    #     We cap at 25% of full Kelly because parameter uncertainty makes full
    #     Kelly catastrophically aggressive in practice.
    #     Source: MacLean, Thorp, Ziemba (2010) "The Kelly Capital Growth
    #     Investment Criterion" Ch. 1; Thorp (1969) "Optimal Gambling Systems".
    if score == 0 or atr_pct <= 0:
        suggested_pct = 0.0
    else:
        trade_risk_pct = risk_p["trade_risk_pct"]
        stop_multiplier = risk_p["stop_atr_mult"]
        vol_target_pct = trade_risk_pct / (atr_pct * stop_multiplier * 100)
        edge = abs(score) * confidence * 0.05
        odds = max(atr_pct * (21 ** 0.5), 0.01)  # ~30-day expected move (sqrt-T scaling)
        kelly_f = max(0.0, edge / odds) * 0.25      # ¼-Kelly (MacLean/Thorp/Ziemba 2010)
        suggested = min(vol_target_pct, kelly_f * 100)
        # Profile-specific floor + ceiling.
        suggested = max(risk_p["pos_size_min"], min(risk_p["pos_size_max"], suggested))
        suggested_pct = round(suggested, 2)

    thesis = _thesis_one_liner(sym, verdict, score, regime, fc_move, pa, inst)

    return {
        "symbol": sym,
        "as_of": signals["as_of"],
        "verdict": verdict,
        "score": round(float(score), 3),
        "confidence": round(float(confidence), 3),
        "time_horizon": "30 days",
        "thesis": thesis,
        "bull_case": bull[:6],
        "bear_case": bear[:6],
        "key_risks": risks[:5],
        "risk_mode": risk_mode,
        "trade_direction": direction,  # "long", "short", or None for HOLD
        "entry_zone": entry_zone,
        "stop": round(suggested_stop, 2) if suggested_stop else None,
        "target_1": target_1,
        "target_2": target_2,
        "suggested_position_pct": suggested_pct if direction else 0.0,
        "signal_summary": {
            "recommendation_score": round(rec_score, 3),
            "forecast_move_pct": round(fc_move * 100, 2),
            "price_action_score": round(pa_score, 3),
            "analyst_score": round(analyst_score, 3),
            "insider_score": round(insider_score, 3),
            "institutional_score": round(inst_score, 3),
            "political_score": round(pol_score, 3),
            "regime": regime,
        },
        "generator": "deterministic",
    }


def _thesis_one_liner(sym: str, verdict: str, score: float, regime: str,
                      fc_move: float, pa: dict, inst: dict) -> str:
    """Align language with verdict thresholds so we don't say 'bullish (hold)'."""
    if verdict in ("STRONG_BUY", "BUY"):
        lead = "Bullish setup"
    elif verdict == "ACCUMULATE":
        lead = "Mildly bullish — opportunistic accumulation"
    elif verdict == "HOLD":
        lead = "Signals are mixed; nothing decisive"
    elif verdict == "REDUCE":
        lead = "Mildly bearish — trim exposure"
    elif verdict in ("SELL", "STRONG_SELL"):
        lead = "Bearish setup"
    else:
        lead = "Neutral"

    parts = [f"{lead} ({verdict.replace('_', ' ').lower()}, score {score:+.2f})."]
    if abs(fc_move) > 0.005:
        parts.append(f"30d ensemble forecast {fc_move*100:+.1f}%.")
    if pa.get("trend") and pa["trend"] != "range":
        parts.append(f"Trend: {pa['trend']}.")
    conf = (pa.get("confluence") or {}).get("label")
    if conf and conf != "neutral":
        parts.append(f"Smart-money structure: {conf.replace('_', ' ')}.")
    if inst.get("n_famous", 0) > 0:
        parts.append(f"Notable institutional holders: {inst['n_famous']}.")
    parts.append(f"Regime: {regime}.")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# LLM synthesizer (Anthropic)
# -----------------------------------------------------------------------------

_LLM_SYSTEM = """You are a senior buy-side equity analyst at a long-short hedge fund with 20+ years across cycles. You publish a daily desk note. Voice: direct, specific, no hype, no boilerplate. The reader is a professional who can handle ambiguity.

HOW THIS WORKS — READ CAREFULLY:
The desk's systematic model has ALREADY decided the trade plan: the verdict, direction, entry zone, stop, targets, and position size are given to you in DESK_PLAN. Those price levels are computed from market structure (demand/supply zones), ATR, and risk sizing — they are PRECISE and AUTHORITATIVE. **Your job is NOT to re-derive or change them.** Your job is two things:
  (A) Write the analyst narrative that explains and pressure-tests that plan, citing real numbers from the SIGNAL_PACK.
  (B) Give an independent ENTRY REVIEW: is *now* a good moment to take the desk's entry, or should the trader wait?

NON-NEGOTIABLE RULES:
1. **Cite real numbers** from DESK_PLAN and SIGNAL_PACK — never invent metrics, prices, or company facts. If a field is missing or zero, say so.
2. **Never restate or alter the entry/stop/targets** as different numbers. Refer to the DESK_PLAN levels exactly. Do NOT output your own entry/stop/target fields.
3. **Apply technical polarity correctly**:
   - `active_supply_zones` are the only true overhead resistance.
   - `active_demand_zones` are the only true downside support.
   - `flipped_resistance`/`flipped_support` are levels price ALREADY broke — call them "former resistance, now support", not live S/R.
   - `price_in_discovery=true` (no overhead supply) = blue-sky breakout / price discovery — a specific bullish read.
4. **Risk-reward arithmetic**: state the implied R-multiple of the DESK_PLAN explicitly in the thesis (entry→stop = 1R; entry→target_1 = NR). E.g. "risking 1R to $X for 3.2R to $Y — 3:1".
5. **Be honest about uncertainty**: forecast band wider than 50% of price → "low conviction, lean on structure"; P(up) within 8% of 0.5 → "coin flip"; low signal confidence → don't overclaim.
6. **Entry review is the value-add.** Judge the *timing* of the desk's entry against: regime (risk-on/off), RSI (overbought/oversold), recent 1M/3M momentum (chasing an extended move?), distance from the entry zone (is price already through it?), upcoming catalysts/earnings, and news. Be willing to say the verdict is right but the entry is poorly timed ("good name, wrong moment — wait for the pullback to the zone").
7. **No personalized advice** — desk analysis, not a brokerage rec. Conditional phrasing ("if it pulls back to $X…"), not imperative.
8. **Output JSON only**, matching the schema. No prose outside JSON. Do NOT include verdict/score/entry/stop/target fields — those are owned by the desk model.

Schema:
{
  "thesis": "2-3 sentences, desk-analyst voice, concrete — must state the DESK_PLAN R-multiple",
  "bull_case": [strings citing actual numbers],
  "bear_case": [strings citing actual numbers],
  "key_risks": [strings — what specifically would invalidate the thesis (a price level, an event)],
  "reasoning_chain": "1-2 sentence chain linking the signals to the desk verdict",
  "entry_review": {
    "assessment": one of [GOOD_ENTRY, WAIT_FOR_PULLBACK, CHASE_RISK, AVOID, NO_TRADE],
    "timing": "short phrase, e.g. 'wait for pullback to 206-213 demand'",
    "rationale": "1-2 sentences on whether NOW is a good moment to take the desk entry, given regime/RSI/momentum/structure/news",
    "flags": [strings — concrete timing risks, e.g. 'RSI 78 — buying into overbought', 'earnings in 3 days'],
    "confidence": float in [0,1]
  }
}"""


def _is_schema_echo(s: str) -> bool:
    """True if a string looks like the model parroted the prompt's schema
    description instead of producing real content (a known failure mode of
    weak models). Used to reject garbage narrative during merge."""
    if not isinstance(s, str) or len(s.strip()) < 20:
        return True
    low = s.lower()
    tells = ("desk-analyst voice", "citing actual numbers", "2-3 sentences",
             "string", "or null", "must state", "r-multiple\"", "schema")
    return sum(t in low for t in tells) >= 2


def synthesize_with_llm(signals: dict[str, Any], risk_mode: str = "conservative") -> dict[str, Any]:
    """Hybrid synthesis: the DETERMINISTIC engine owns the trade plan (verdict,
    entry, stop, targets, size — precise, auditable), and the LLM owns the
    NARRATIVE plus an independent ENTRY REVIEW. We never let the LLM move the
    price levels; that's what caused drifted/contradictory entries before.

    Flow:
      1. Compute the authoritative plan with `synthesize_deterministic`.
      2. Hand the LLM that plan (locked) + the signal pack and ask only for
         prose + an entry-timing review (tier="quality" → Groq 70B).
      3. Merge the LLM's narrative back over the deterministic numbers.
    """
    from app.services import llm

    symbol = signals.get("symbol", "")
    # 1) Authoritative trade plan — numbers come from here, full stop.
    det = synthesize_deterministic(symbol, signals, risk_mode=risk_mode)

    # 2) The locked plan we hand to the LLM (do-not-touch numbers).
    desk_plan = {
        "verdict": det.get("verdict"),
        "score": det.get("score"),
        "confidence": det.get("confidence"),
        "direction": det.get("trade_direction"),
        "entry_zone": det.get("entry_zone"),
        "stop": det.get("stop"),
        "target_1": det.get("target_1"),
        "target_2": det.get("target_2"),
        "suggested_position_pct": det.get("suggested_position_pct"),
        "time_horizon": det.get("time_horizon"),
        "risk_mode": risk_mode,
        "regime": signals.get("regime"),
    }
    payload = {
        "symbol": symbol,
        "last_price": signals.get("last_price"),
        "regime": signals.get("regime"),
        "key_stats": signals.get("key_stats"),
        "technical": signals.get("technical"),
        "forecast_30d": signals.get("forecast_30d"),
        "price_action": signals.get("price_action"),
        "insider": signals.get("insider"),
        "institutional": signals.get("institutional"),
        "politicians": signals.get("politicians"),
        "news_topics": signals.get("news_topics"),
    }
    # Token budget: Groq 70B is 12k TPM. desk_plan (~0.3k) + 9k-char pack
    # (~2.3k) + 900 output ≈ 3.5k tokens/call → ~3 calls/min comfortably.
    user_msg = (
        "DESK_PLAN — already decided by the systematic model. DO NOT change these "
        "numbers; narrate and pressure-test them, and review the entry timing:\n"
        + json.dumps(desk_plan, default=str)
        + "\n\nSIGNAL_PACK:\n"
        + json.dumps(payload, default=str)[:9000]
        + "\n\nReturn ONLY the JSON schema from the system prompt. No preamble, no fences."
    )

    result = llm.generate(
        system=_LLM_SYSTEM, user=user_msg,
        max_tokens=900, temperature=0.35, expect_json=True, tier="quality",
    )
    if not result or not result.json:
        # LLM unavailable / unparseable — serve the deterministic plan as-is.
        det["generator"] = "deterministic (LLM unavailable or parse failed)"
        return det

    llm_out = result.json
    # 3) Merge: start from the authoritative deterministic plan, overlay only
    # the narrative fields the LLM is allowed to own — but reject schema-echo
    # garbage (weak models sometimes parrot the prompt schema instead of
    # answering). A bad narrative is worse than the deterministic template.
    merged = dict(det)
    th = llm_out.get("thesis")
    if isinstance(th, str) and not _is_schema_echo(th):
        merged["thesis"] = th
    for k in ("bull_case", "bear_case", "key_risks"):
        v = llm_out.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) and not _is_schema_echo(x) for x in v):
            merged[k] = v
    rc = llm_out.get("reasoning_chain") or (llm_out.get("signal_summary") or {}).get("reasoning_chain")
    if rc and isinstance(merged.get("signal_summary"), dict):
        merged["signal_summary"]["reasoning_chain"] = rc
    er = llm_out.get("entry_review")
    if isinstance(er, dict) and er.get("assessment"):
        merged["entry_review"] = er
    merged["symbol"] = symbol
    merged["as_of"] = signals.get("as_of")
    merged["generator"] = f"{result.provider}:{result.model} + deterministic plan"
    return merged


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def get_opinion(
    symbol: str,
    use_llm: bool | None = None,
    risk_mode: str = "conservative",
    timeframe: str = "position",
) -> dict[str, Any]:
    """Public entry: collect signals, synthesize, return structured opinion.

    `timeframe` ∈ {intraday, swing, position, longterm} picks the bar interval
    + horizon for the price-action analysis and the forecast that's cited.
    Different timeframes produce different demand/supply zones (intraday
    pivots ≠ daily pivots ≠ weekly pivots) — see `_TIMEFRAMES`.

    Cache strategy:
      - Layer 1: signal pack (keyed by symbol+timeframe).
      - Layer 2: synthesized opinion (keyed by symbol+timeframe+risk_mode).

    `use_llm` defaults to True whenever *any* LLM provider is available.
    """
    from app.services import llm
    if timeframe not in _TIMEFRAMES:
        timeframe = "position"
    cache_key = f"opinion_out:v2:{symbol}:{timeframe}:{risk_mode}:{int(bool(use_llm)) if use_llm is not None else -1}"
    if (cached := _cache_get(cache_key)):
        return cached

    signals = collect_signals(symbol, timeframe=timeframe)
    if use_llm is None:
        use_llm = llm.is_available()
    llm_succeeded = False
    if use_llm and signals.get("last_price"):
        out = synthesize_with_llm(signals, risk_mode=risk_mode) | {"raw_signals": _trim_signals_for_response(signals)}
        out["risk_mode"] = risk_mode
        # The synthesize_with_llm helper sets `generator` to e.g. "groq:..."
        # on success, or "deterministic (LLM unavailable...)" on fallback.
        llm_succeeded = not str(out.get("generator", "")).startswith("deterministic")
    else:
        out = synthesize_deterministic(symbol, signals, risk_mode=risk_mode)
        out["raw_signals"] = _trim_signals_for_response(signals)
        out["risk_mode"] = risk_mode
    # Surface the timeframe in the response so the UI can render
    # "Position view (1–3 months)" instead of just a verdict.
    out["timeframe"] = timeframe
    out["timeframe_label"] = _TIMEFRAMES[timeframe]["label"]
    out["horizon_text"] = _TIMEFRAMES[timeframe]["horizon_text"]
    # If we wanted the LLM but it was unavailable, cache the fallback for only
    # 30s — that way the moment the rate-limit clears (which is often ~18-60s
    # on Groq TPM exhaustion) the user gets the real LLM output on retry.
    # If we got the LLM result (or weren't trying for one), full 5-min cache.
    ttl = (5 * 60) if (llm_succeeded or not use_llm) else 30
    _cache_set(cache_key, out, ttl=ttl)
    return out


def _trim_signals_for_response(signals: dict[str, Any]) -> dict[str, Any]:
    """Return a lighter version of the signal pack for the API response."""
    out = dict(signals)
    out.pop("profile", None)
    if "insider" in out and isinstance(out["insider"], dict):
        out["insider"] = {k: v for k, v in out["insider"].items() if k != "recent_transactions"}
    if "institutional" in out and isinstance(out["institutional"], dict):
        out["institutional"] = {k: v for k, v in out["institutional"].items() if k != "top_institutional_holders"}
    return out
