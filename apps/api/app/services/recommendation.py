"""Hybrid recommendation engine. Blends forecast, technical, fundamental,
sentiment, macro, events into a single labeled recommendation with explanations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ml.regime import REGIME_WEIGHTS, detect
from app.services import earnings as earnings_svc
from app.services import indicators as ind
from app.services import market_data as md
from app.services import price_action as pa
from app.services import sentiment as sent


def _technical_signal(df: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    close = df["close"]
    rsi = ind.rsi(close, 14).iloc[-1]
    macd_df = ind.macd(close)
    macd_hist = macd_df["hist"].iloc[-1]
    sma50 = ind.sma(close, 50).iloc[-1]
    sma200 = ind.sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    last = float(close.iloc[-1])

    bullish, bearish = [], []
    score = 0.0
    if not np.isnan(rsi):
        if rsi < 30: bullish.append(f"RSI {rsi:.0f} (oversold)"); score += 0.3
        elif rsi > 70: bearish.append(f"RSI {rsi:.0f} (overbought)"); score -= 0.3
    if not np.isnan(macd_hist):
        if macd_hist > 0: bullish.append("MACD histogram positive"); score += 0.15
        else: bearish.append("MACD histogram negative"); score -= 0.15
    if not np.isnan(sma50):
        if last > sma50: bullish.append("Above 50-day SMA"); score += 0.1
        else: bearish.append("Below 50-day SMA"); score -= 0.1
    if not np.isnan(sma200):
        if last > sma200: bullish.append("Above 200-day SMA (long-term bullish)"); score += 0.1
        else: bearish.append("Below 200-day SMA (long-term bearish)"); score -= 0.1

    return float(np.clip(score, -1, 1)), {
        "summary": _summarize(bullish, bearish),
        "bullish": bullish,
        "bearish": bearish,
        "indicators": {
            "rsi_14": None if np.isnan(rsi) else float(rsi),
            "macd_hist": None if np.isnan(macd_hist) else float(macd_hist),
        },
    }


def _fundamental_signal(symbol: str) -> tuple[float, dict[str, Any]]:
    stats = md.get_key_stats(symbol)
    profile = md.get_profile(symbol)
    pe = stats.get("pe")
    eps = stats.get("eps")
    beta = stats.get("beta")
    highlights: list[str] = []
    score = 0.0

    # PE cutoffs.
    # The "right" PE is regime-dependent (Shiller CAPE; Fama-French value).
    # 15 ≈ S&P long-run median earnings-yield reciprocal (~6.7%); 40 is loosely
    # a "growth premium" cutoff. Honest grade: heuristic — a rigorous version
    # would compare to sector median or to CAPE.
    if pe:
        if 0 < pe < 15: highlights.append(f"PE {pe:.1f} (cheap)"); score += 0.25
        elif pe > 40: highlights.append(f"PE {pe:.1f} (expensive)"); score -= 0.15
        else: highlights.append(f"PE {pe:.1f}")
    # EPS sign as a simple profitability filter — Novy-Marx (2013) "The Other
    # Side of Value: The Gross Profitability Premium" RFS, shows positive
    # profitability predicts cross-sectional returns. EPS is a noisier proxy.
    if eps and eps > 0:
        highlights.append(f"EPS positive ({eps:.2f})"); score += 0.05
    elif eps and eps < 0:
        highlights.append("EPS negative"); score -= 0.1
    # Beta context only (does not move the score). Standard CAPM
    # interpretation — Sharpe (1964) JoF. Beta >1.5 = high systemic vol;
    # <0.7 = defensive.
    if beta:
        if beta > 1.5: highlights.append(f"High beta ({beta:.2f})")
        elif beta < 0.7: highlights.append(f"Low beta ({beta:.2f})")

    return float(np.clip(score, -1, 1)), {
        "summary": "; ".join(highlights) or "Limited fundamental data.",
        "highlights": highlights,
        "stats": {"pe": pe, "eps": eps, "beta": beta, "sector": profile.get("sector")},
    }


def _sentiment_signal(symbol: str) -> tuple[float, dict[str, Any]]:
    news = md.get_news(symbol, limit=15)
    scores: list[tuple[float, float]] = []
    for n in news:
        text = " ".join(filter(None, [n.get("title"), n.get("summary")]))
        scores.append(sent.score_text(text))
    agg = sent.aggregate(scores)
    polarity = agg["weighted"]
    return float(np.clip(polarity, -1, 1)), {
        "summary": f"News sentiment {polarity:+.2f} from {int(agg['n'])} articles.",
        "polarity": polarity,
        "n_articles": int(agg["n"]),
    }


def _direction_signal_from_forecast(forecast_point: float, last_price: float) -> float:
    if last_price <= 0:
        return 0.0
    move_pct = (forecast_point - last_price) / last_price
    # squash to [-1, 1] gently
    return float(np.tanh(move_pct * 10))


def _summarize(bullish: list[str], bearish: list[str]) -> str:
    parts = []
    if bullish: parts.append("Bullish: " + ", ".join(bullish))
    if bearish: parts.append("Bearish: " + ", ".join(bearish))
    return "; ".join(parts) if parts else "Mixed / neutral signals."


def _label_from_score(score: float) -> str:
    """Use data-driven thresholds when calibration has been run; defaults otherwise."""
    from app.ml.threshold_calibration import label_from_score
    return label_from_score(score)


def recommend(symbol: str, forecast_point: float | None = None) -> dict[str, Any]:
    df = md.get_history(symbol, interval="1d", range_="2y")
    if df.empty:
        return {
            "symbol": symbol,
            "label": "HOLD",
            "score": 0.0,
            "confidence": 0.1,
            "reasoning": {"technical": {}, "fundamental": {}, "sentiment": {}, "macro": {}, "events": [], "risk": {}},
        }
    last_price = float(df["close"].iloc[-1])
    regime = detect(df["close"])
    w = REGIME_WEIGHTS[regime]

    tech_score, tech_detail = _technical_signal(df)
    fund_score, fund_detail = _fundamental_signal(symbol)
    sent_score, sent_detail = _sentiment_signal(symbol)
    forecast_score = (
        _direction_signal_from_forecast(forecast_point, last_price) if forecast_point else 0.0
    )

    # Smart Money price-action confluence (already in [-1, +1])
    try:
        pa_result = pa.analyze(df)
        if pa_result.get("ok"):
            pa_score = float(pa_result["confluence"]["score"])
            pa_detail = {
                "summary": _humanize_pa_label(pa_result["confluence"]["label"]),
                "score": pa_score,
                "label": pa_result["confluence"]["label"],
                "drivers": pa_result["confluence"]["drivers"],
                "current_trend": pa_result["current_trend"],
                "active_demand_zones": [z for z in pa_result["zones"] if z["direction"] == "bullish"][:3],
                "active_supply_zones": [z for z in pa_result["zones"] if z["direction"] == "bearish"][:3],
                "recent_structure": pa_result["structure_events"][-3:],
            }
        else:
            pa_score, pa_detail = 0.0, {"summary": "Insufficient data for price-action analysis."}
    except Exception as e:
        pa_score, pa_detail = 0.0, {"summary": f"Price action unavailable: {e}"}

    score = (
        w["forecast"] * forecast_score
        + w["technical"] * tech_score
        + w["fundamental"] * fund_score
        + w["sentiment"] * sent_score
        + w["price_action"] * pa_score
    )

    # risk metrics
    atr_pct = float(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1] / last_price)
    rolling_max = df["close"].cummax()
    dd_90 = float((df["close"].tail(90) / rolling_max.tail(90) - 1).min())
    suggested_stop = last_price * (1 - 2 * atr_pct) if atr_pct > 0 else last_price * 0.95

    risk = {
        "atr_pct_of_price": round(atr_pct * 100, 2),
        "max_drawdown_90d_pct": round(dd_90 * 100, 2),
        "suggested_stop": round(suggested_stop, 2),
        "regime": regime,
    }

    # confidence: stronger when signals agree
    signals = [forecast_score, tech_score, fund_score, sent_score, pa_score]
    signs = [np.sign(s) for s in signals if abs(s) > 0.05]
    agreement = (max(signs.count(1), signs.count(-1)) / len(signs)) if signs else 0.5
    confidence = float(np.clip(0.4 + 0.5 * agreement, 0.1, 0.95))

    return {
        "symbol": symbol,
        "label": _label_from_score(score),
        "score": round(float(score), 4),
        "confidence": round(confidence, 4),
        "reasoning": {
            "technical": tech_detail,
            "fundamental": fund_detail,
            "sentiment": sent_detail,
            "macro": {"summary": f"Market regime: {regime}.", "regime": regime, "weights": w},
            "price_action": pa_detail,
            "events": [],
            "risk": risk,
        },
    }


def _humanize_pa_label(label: str) -> str:
    return {
        "strong_bullish": "Strong bullish structure (BOS/CHOCH, demand zone, FVG aligned).",
        "bullish": "Bullish structure with at least one confluent signal.",
        "neutral": "No clear structural setup.",
        "bearish": "Bearish structure with at least one confluent signal.",
        "strong_bearish": "Strong bearish structure (BOS/CHOCH, supply zone, FVG aligned).",
    }.get(label, label)
