"""Composite screeners.

Three opinionated screens, each with per-symbol score breakdown:

  - **rising_stars**: emerging-growth + momentum + positive news.
       Targets mid/small-caps not yet at 52w highs, breaking out, with
       positive news velocity. The classic "early but not too early" screen.

  - **breakouts**: pure technical breakouts above 50/200 SMA with rising volume.

  - **value_with_catalyst**: cheap (low PE) + positive sentiment + tightening.

All return a ranked list of dicts:
  {
    "symbol": "...", "name": "...", "score": 0.71,
    "components": {"momentum": 0.6, "news": 0.8, ...},
    "rationale": ["positive 3M momentum +12%", "news sentiment +0.4 across 14 articles", ...]
  }
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from app.services import indicators as ind
from app.services import market_data as md
from app.services import news as news_svc
from app.services import price_action as pa_svc


# --- per-symbol feature pull ----------------------------------------------------

def _feature_pack(symbol: str) -> dict[str, Any] | None:
    """Pull everything we need for one symbol. Returns None on failure."""
    try:
        df = md.get_history(symbol, interval="1d", range_="1y")
        if df.empty or len(df) < 200:
            return None
        close = df["close"]
        last = float(close.iloc[-1])

        # technical
        ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0.0
        ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
        ret_1y = float(close.iloc[-1] / close.iloc[0] - 1)
        rsi14 = float(ind.rsi(close, 14).iloc[-1])
        sma50 = float(ind.sma(close, 50).iloc[-1])
        sma200 = float(ind.sma(close, 200).iloc[-1])
        atr_pct = float(ind.atr(df["high"], df["low"], close, 14).iloc[-1] / last)

        # volume
        vol = df["volume"]
        vol_z_20 = float((vol.iloc[-1] - vol.rolling(20).mean().iloc[-1]) / (vol.rolling(20).std().iloc[-1] + 1e-9))

        # 52w distance
        high_52 = float(close.tail(252).max() if len(close) >= 252 else close.max())
        low_52 = float(close.tail(252).min() if len(close) >= 252 else close.min())
        pct_off_high = float((high_52 - last) / high_52) if high_52 > 0 else 0.0
        pos_in_range = (last - low_52) / (high_52 - low_52 + 1e-9)

        # fundamentals (current snapshot)
        try:
            stats = md.get_key_stats(symbol)
            profile = md.get_profile(symbol)
        except Exception:
            stats, profile = {}, {}
        pe = stats.get("pe")
        beta = stats.get("beta")
        mkt_cap = profile.get("market_cap")
        sector = profile.get("sector")
        name = profile.get("name") or symbol

        # news + sentiment (small fetch)
        try:
            articles = news_svc.fetch_news(symbol, limit=20)
        except Exception:
            articles = []
        n_articles = len(articles)
        sent_vals = [a.get("sentiment") for a in articles if a.get("sentiment") is not None]
        sent_weights = [a.get("sentiment_confidence") or 0.0 for a in articles]
        if sent_vals:
            ws = float(sum(s * w for s, w in zip(sent_vals, sent_weights, strict=True)) / (sum(sent_weights) or 1.0))
        else:
            ws = 0.0

        # price action
        try:
            pa_result = pa_svc.analyze(df)
            pa_score = pa_result.get("confluence", {}).get("score", 0.0) if pa_result.get("ok") else 0.0
            pa_label = pa_result.get("confluence", {}).get("label", "neutral") if pa_result.get("ok") else "neutral"
            pa_drivers = pa_result.get("confluence", {}).get("drivers", []) if pa_result.get("ok") else []
            pa_trend = pa_result.get("current_trend", "range") if pa_result.get("ok") else "range"
        except Exception:
            pa_score, pa_label, pa_drivers, pa_trend = 0.0, "neutral", [], "range"

        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "last_price": last,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_1y": ret_1y,
            "rsi14": rsi14,
            "above_sma50": last > sma50,
            "above_sma200": last > sma200,
            "atr_pct": atr_pct,
            "vol_z_20": vol_z_20,
            "pct_off_high": pct_off_high,
            "pos_in_range": float(np.clip(pos_in_range, 0, 1)),
            "pe": pe, "beta": beta, "market_cap": mkt_cap,
            "n_articles": n_articles,
            "sentiment_weighted": ws,
            "pa_score": pa_score,
            "pa_label": pa_label,
            "pa_drivers": pa_drivers,
            "pa_trend": pa_trend,
        }
    except Exception:
        return None


# --- scoring strategies --------------------------------------------------------


def _score_rising_star(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    """Mid/small-cap growth + momentum + positive news, not at peak.

    Returns (composite_score, component_scores, rationale_lines).
    """
    components: dict[str, float] = {}
    rationale: list[str] = []

    # momentum: 3m and 1m positive, not too overheated
    r3 = f["ret_3m"]
    momentum = _clip(r3 * 2.5, -1, 1)  # ±40% maps to ±1
    if r3 > 0:
        rationale.append(f"3M momentum +{r3*100:.1f}%")
    components["momentum_3m"] = momentum

    r1 = f["ret_1m"]
    components["momentum_1m"] = _clip(r1 * 5, -1, 1)
    if r1 > 0.05: rationale.append(f"1M momentum +{r1*100:.1f}%")

    # not overheated: RSI between 50 and 75 is ideal; above 80 penalize
    rsi = f["rsi14"]
    if 50 <= rsi <= 75:
        components["rsi_zone"] = 1.0
        rationale.append(f"RSI {rsi:.0f} (constructive zone)")
    elif rsi > 80:
        components["rsi_zone"] = -0.6
        rationale.append(f"RSI {rsi:.0f} (overbought — caution)")
    elif rsi < 40:
        components["rsi_zone"] = -0.3
    else:
        components["rsi_zone"] = 0.3

    # above SMAs (long-term trend)
    components["above_sma200"] = 0.8 if f["above_sma200"] else -0.5
    if f["above_sma200"]: rationale.append("Above 200-day SMA")

    # early-stage proxy: between 70% and 92% of 52w high — past breakout, not exhausted
    pir = f["pos_in_range"]
    if 0.7 <= pir <= 0.92:
        components["range_position"] = 1.0
        rationale.append(f"At {pir*100:.0f}% of 52w range (constructive)")
    elif pir > 0.97:
        components["range_position"] = -0.3
    elif pir < 0.4:
        components["range_position"] = -0.2
    else:
        components["range_position"] = 0.4

    # volume spike
    vz = f["vol_z_20"]
    components["volume_spike"] = _clip(vz * 0.5, -1, 1)
    if vz > 1.5: rationale.append(f"Volume {vz:.1f}σ above 20-day avg")

    # news velocity + sentiment
    n = f["n_articles"]
    news_count_score = _clip((n - 8) / 15, -1, 1)
    sent = f["sentiment_weighted"]
    sent_score = _clip(sent * 2, -1, 1)
    components["news_velocity"] = news_count_score
    components["news_sentiment"] = sent_score
    if n >= 12: rationale.append(f"{n} recent articles (elevated coverage)")
    if sent > 0.15: rationale.append(f"Positive news sentiment {sent:+.2f}")
    if sent < -0.15: rationale.append(f"Negative news sentiment {sent:+.2f}")

    # market-cap preference: penalize mega-caps (>$200B) — rising stars are usually smaller
    mc = f.get("market_cap") or 0
    if mc:
        if mc < 50_000_000_000:           # < $50B → mid/small cap
            components["size_preference"] = 0.5
            rationale.append(f"Market cap ${mc/1e9:.1f}B (mid/small-cap)")
        elif mc < 200_000_000_000:        # 50-200B → mid-large
            components["size_preference"] = 0.1
        else:
            components["size_preference"] = -0.3   # mega-cap — not "rising"
    else:
        components["size_preference"] = 0.0

    # composite: weighted mean of components
    weights = {
        "momentum_3m": 0.18,
        "momentum_1m": 0.10,
        "rsi_zone": 0.08,
        "above_sma200": 0.08,
        "range_position": 0.14,
        "volume_spike": 0.10,
        "news_velocity": 0.08,
        "news_sentiment": 0.16,
        "size_preference": 0.08,
    }
    score = sum(components[k] * weights[k] for k in weights)
    return float(_clip(score, -1, 1)), components, rationale


def _score_breakout(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    components, rationale = {}, []
    components["above_sma50"] = 0.5 if f["above_sma50"] else -0.5
    components["above_sma200"] = 0.5 if f["above_sma200"] else -0.5
    if f["above_sma50"] and f["above_sma200"]: rationale.append("Above both 50d and 200d SMAs")
    components["near_high"] = 1.0 if f["pos_in_range"] > 0.85 else (-0.3 if f["pos_in_range"] < 0.5 else 0.3)
    if f["pos_in_range"] > 0.9: rationale.append(f"At {f['pos_in_range']*100:.0f}% of 52w range")
    components["volume_spike"] = _clip(f["vol_z_20"] * 0.6, -1, 1)
    if f["vol_z_20"] > 1.5: rationale.append(f"Volume spike {f['vol_z_20']:.1f}σ")
    components["rsi"] = 0.6 if 55 < f["rsi14"] < 78 else -0.3
    score = sum(components.values()) / max(len(components), 1)
    return float(_clip(score, -1, 1)), components, rationale


def _score_value_with_catalyst(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    components, rationale = {}, []
    pe = f.get("pe")
    if pe and 0 < pe < 18:
        components["valuation"] = 0.8
        rationale.append(f"PE {pe:.1f} (cheap)")
    elif pe and pe > 40:
        components["valuation"] = -0.4
    else:
        components["valuation"] = 0.1

    components["news_sentiment"] = _clip(f["sentiment_weighted"] * 2, -1, 1)
    if f["sentiment_weighted"] > 0.2: rationale.append(f"Positive news catalyst {f['sentiment_weighted']:+.2f}")
    components["momentum"] = _clip(f["ret_1m"] * 5, -1, 1)
    if f["ret_1m"] > 0: rationale.append(f"1M momentum +{f['ret_1m']*100:.1f}%")
    components["above_sma50"] = 0.5 if f["above_sma50"] else -0.3

    score = sum(components.values()) / max(len(components), 1)
    return float(_clip(score, -1, 1)), components, rationale


def _score_smart_money(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    """Symbols currently sitting at a high-confluence smart-money setup:
    BOS/CHOCH, demand zone proximity, fresh FVG, or recent liquidity sweep.
    """
    components, rationale = {}, []
    pa = f.get("pa_score", 0.0)
    components["pa_confluence"] = pa
    if abs(pa) >= 0.15:
        rationale.append(f"Price action: {f.get('pa_label')} (score {pa:+.2f})")
    for d in (f.get("pa_drivers") or [])[:4]:
        rationale.append(d)

    # require alignment between PA and short-term trend
    trend = f.get("pa_trend", "range")
    if trend == "up" and pa > 0:
        components["trend_align"] = 0.5
    elif trend == "down" and pa < 0:
        components["trend_align"] = 0.5
    elif trend == "range":
        components["trend_align"] = 0.1
    else:
        components["trend_align"] = -0.4

    # confirmation: positive sentiment + above 200d SMA for longs
    if pa > 0:
        components["news_confirm"] = _clip(f["sentiment_weighted"] * 1.5, -1, 1)
        components["above_sma200"] = 0.4 if f["above_sma200"] else -0.5
    else:
        components["news_confirm"] = -_clip(f["sentiment_weighted"] * 1.5, -1, 1)
        components["above_sma200"] = -0.4 if f["above_sma200"] else 0.4

    if f["above_sma200"] and pa > 0:
        rationale.append("Above 200-day SMA (long-term bullish)")

    weights = {
        "pa_confluence": 0.55,
        "trend_align": 0.20,
        "news_confirm": 0.15,
        "above_sma200": 0.10,
    }
    score = sum(components[k] * weights[k] for k in weights)
    return float(_clip(score, -1, 1)), components, rationale


def _score_underpriced_mover(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    """Small/mid-cap + low PE + recent positive catalyst + early momentum.

    The "value but moving" play: a company that's still cheap by traditional
    metrics but where the market is starting to wake up to a catalyst. This
    is where you find 2-3x runners early.

    Filters:
      - Market cap < $5B (avoid the value-trap large-caps)
      - PE < 15 OR EPS > 0 AND PE NOT extreme
      - Recent material catalyst (LLM-classified, materiality >= 0.5)
      - 1M momentum positive (the market is starting to notice)
    """
    from app.services import news_intel as ni
    components: dict[str, float] = {}
    rationale: list[str] = []

    mc = f.get("market_cap") or 0
    if mc > 5_000_000_000:
        components["size"] = -1.0   # too big to be "undiscovered"
    elif mc < 2_000_000_000:
        components["size"] = 1.0
        rationale.append(f"Small-cap ${mc/1e9:.1f}B (often less analyst coverage)")
    elif mc < 5_000_000_000:
        components["size"] = 0.6
        rationale.append(f"Mid-cap ${mc/1e9:.1f}B")
    else:
        components["size"] = 0.0

    # Valuation — cheap PE OR positive but reasonable PE
    pe = f.get("pe")
    if pe is not None:
        if 0 < pe < 12:
            components["valuation"] = 1.0
            rationale.append(f"Cheap PE {pe:.1f}")
        elif 12 <= pe < 20:
            components["valuation"] = 0.5
        elif pe >= 40:
            components["valuation"] = -0.6
        else:
            components["valuation"] = 0.2
    else:
        # No PE often means EPS negative — penalize lightly
        components["valuation"] = -0.2

    # Catalyst score (LLM-classified). Try cache only — if missing, neutral.
    try:
        sym = f.get("symbol")
        if sym:
            cat = ni.classify_and_score(sym, days_window=14)
            components["catalyst"] = float(cat.get("score", 0)) * 1.2
            mat = cat.get("max_materiality", 0)
            top_cats = cat.get("top_categories") or []
            top_cat_name = top_cats[0][0] if top_cats else None
            if mat >= 0.5 and top_cat_name and top_cat_name != "noise":
                rationale.append(f"Catalyst: {top_cat_name.replace('_', ' ')} (materiality {mat:.2f})")
        else:
            components["catalyst"] = 0.0
    except Exception:
        components["catalyst"] = 0.0

    # Momentum (the market noticing)
    r1 = f.get("ret_1m") or 0
    components["momentum"] = _clip(r1 * 4, -1, 1)
    if r1 > 0.05:
        rationale.append(f"1M momentum +{r1*100:.1f}%")
    elif r1 < -0.05:
        rationale.append(f"1M momentum {r1*100:.1f}% (no inflection yet)")

    # Above 200d as a long-term-bullish gate
    components["trend"] = 0.4 if f.get("above_sma200") else -0.4

    weights = {
        "size":      0.25,    # small/mid cap is part of the thesis
        "valuation": 0.30,    # central thesis: cheap
        "catalyst":  0.30,    # central thesis: something happening
        "momentum":  0.10,
        "trend":     0.05,
    }
    score = sum(components[k] * weights[k] for k in weights)
    return float(_clip(score, -1, 1)), components, rationale


def _score_catalyst_play(f: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    """Pure catalyst-driven: highest-materiality recent news + setup confirmation.

    For 'something just happened' trades — earnings beat the night before,
    FDA approval today, contract win this week. We don't care about valuation
    or size — we care that the market hasn't fully priced it in yet.

    Filters:
      - At least one material catalyst (materiality >= 0.6) in last 7 days
      - Bullish direction
      - Price not already up 30%+ on the news (some room left)
    """
    from app.services import news_intel as ni
    components: dict[str, float] = {}
    rationale: list[str] = []

    try:
        sym = f.get("symbol")
        if not sym:
            return 0.0, {}, []
        cat = ni.classify_and_score(sym, days_window=7)
    except Exception:
        return 0.0, {}, []

    score = float(cat.get("score") or 0)
    mat = float(cat.get("max_materiality") or 0)
    lead = cat.get("lead_article") or {}
    lead_intel = lead.get("intel") or {}
    direction = lead_intel.get("direction", "neutral")

    if mat < 0.6 or direction != "bullish":
        return 0.0, {}, []   # no catalyst worth surfacing

    components["catalyst_strength"] = mat
    components["catalyst_polarity"] = score

    cat_name = lead_intel.get("category", "noise").replace("_", " ")
    rationale.append(f"Material {cat_name} (materiality {mat:.2f})")
    if lead.get("title"):
        rationale.append(f"Lead: {lead['title'][:80]}")

    # Discount if price already rallied hard (catalyst is being priced in)
    r1m = f.get("ret_1m") or 0
    if r1m > 0.30:
        components["headroom"] = -0.5
        rationale.append(f"⚠ Already +{r1m*100:.0f}% in 1M — limited headroom")
    elif r1m > 0.10:
        components["headroom"] = 0.2
    else:
        components["headroom"] = 0.6   # full room

    # RSI sanity — not catastrophically overbought
    rsi = f.get("rsi14")
    if rsi and rsi > 85:
        components["rsi"] = -0.6
        rationale.append(f"RSI {rsi:.0f} (extreme overbought)")
    else:
        components["rsi"] = 0.3

    weights = {
        "catalyst_strength": 0.50,
        "catalyst_polarity": 0.20,
        "headroom":          0.20,
        "rsi":               0.10,
    }
    final = sum(components[k] * weights[k] for k in weights)
    return float(_clip(final, -1, 1)), components, rationale


_STRATEGIES: dict[str, Any] = {
    "rising_stars":         _score_rising_star,
    "breakouts":            _score_breakout,
    "value_with_catalyst":  _score_value_with_catalyst,
    "smart_money":          _score_smart_money,
    "underpriced_movers":   _score_underpriced_mover,   # NEW — value + catalyst + small/mid
    "catalyst_plays":       _score_catalyst_play,       # NEW — pure LLM-classified catalyst
}


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


# --- main entry ----------------------------------------------------------------


def run_screener(
    strategy: str,
    universe: list[str] | None = None,
    limit: int = 12,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    sectors: list[str] | None = None,
) -> list[dict[str, Any]]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    if universe is None:
        # default universe = the seeded ~150 stocks (skip ETF/index/forex/crypto for stock-style screens)
        all_u = md.all_universe()
        universe = [it["symbol"] for it in all_u if it.get("asset_class") == "stock"]

    feature_packs: list[dict[str, Any]] = []
    # Parallelize per-symbol pulls
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_feature_pack, s): s for s in universe}
        for fut in as_completed(futures):
            fp = fut.result()
            if not fp:
                continue
            if min_market_cap is not None and (fp.get("market_cap") or 0) < min_market_cap:
                continue
            if max_market_cap is not None and (fp.get("market_cap") or 0) > max_market_cap:
                continue
            if sectors and (fp.get("sector") not in sectors):
                continue
            feature_packs.append(fp)

    scorer = _STRATEGIES[strategy]
    scored: list[dict[str, Any]] = []
    for fp in feature_packs:
        score, components, rationale = scorer(fp)
        if not math.isfinite(score):
            continue
        scored.append({
            "symbol": fp["symbol"],
            "name": fp["name"],
            "sector": fp["sector"],
            "last_price": fp["last_price"],
            "market_cap": fp.get("market_cap"),
            "score": round(score, 4),
            "components": {k: round(v, 3) for k, v in components.items()},
            "rationale": rationale,
            "snapshot": {
                "ret_3m_pct": round(fp["ret_3m"] * 100, 2),
                "ret_1m_pct": round(fp["ret_1m"] * 100, 2),
                "rsi14": round(fp["rsi14"], 1),
                "vol_z_20": round(fp["vol_z_20"], 2),
                "pos_in_range_pct": round(fp["pos_in_range"] * 100, 1),
                "n_articles": fp["n_articles"],
                "sentiment_weighted": round(fp["sentiment_weighted"], 3),
            },
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


def list_strategies() -> list[dict[str, str]]:
    return [
        {"key": "rising_stars",
         "label": "Rising stars",
         "category": "growth",
         "description": "Mid/small-cap names with momentum, news heat, and constructive technicals — not yet at peak."},
        {"key": "underpriced_movers",
         "label": "Underpriced movers",
         "category": "value_growth",
         "description": "Small/mid-cap (<$5B), cheap on PE, recent LLM-classified material catalyst, momentum starting — the early-runner setup."},
        {"key": "catalyst_plays",
         "label": "Catalyst plays",
         "category": "event",
         "description": "Pure event-driven: a high-materiality catalyst (FDA approval, earnings beat, M&A, contract win) in the last 7 days, with headroom left."},
        {"key": "smart_money",
         "label": "Smart-money setups",
         "category": "technical",
         "description": "High-confluence price-action setups: BOS/CHOCH, demand/supply zones, fresh FVGs, liquidity sweeps."},
        {"key": "breakouts",
         "label": "Breakouts",
         "category": "technical",
         "description": "Names trading above 50d & 200d SMAs near 52w highs with volume spikes."},
        {"key": "value_with_catalyst",
         "label": "Value + catalyst",
         "category": "value",
         "description": "Cheap on PE with positive news catalyst and improving short-term momentum."},
    ]
