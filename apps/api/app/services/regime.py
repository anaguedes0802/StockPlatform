"""Top-down market-regime overlay — the read a pro takes BEFORE sizing any trade.

A great setup in a risk-off tape is still a bad trade. Professionals don't pick
stocks in a vacuum; they first ask "what kind of market am I in?" and size
accordingly. This module synthesises a single risk-on / neutral / risk-off
read from cross-asset signals, all sourced from liquid ETFs (so it stays
robust even when index/^VIX feeds rate-limit):

  - Trend     : SPY vs its 50- and 200-day moving averages (the tape's posture)
  - Momentum  : SPY 20-day return (is the trend accelerating or stalling)
  - Breadth   : small-caps (IWM) & tech (QQQ) relative strength vs SPY — when
                the broad/risk end of the market leads, appetite is healthy
  - Credit    : high-yield (HYG) vs long Treasuries (TLT) — tightening credit
                = risk appetite; flight to bonds = risk-off
  - Volatility: ^VIX level when available (best-effort), else skipped

Each component scores in [-1, +1]; we weight and aggregate to a [-100, +100]
composite, map it to a regime label, and emit a position-sizing multiplier +
a plain-English playbook. The opportunity engine and opinion synthesis can
read this to down-size in hostile tapes instead of pretending every day is the
same.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import log
from app.services import market_data as md

# Component weights (renormalised over whatever data is actually available).
_WEIGHTS = {
    "trend": 0.35,
    "momentum": 0.20,
    "breadth": 0.20,
    "credit": 0.15,
    "volatility": 0.10,
}

_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_TTL_S = 30 * 60  # regime moves slowly; half-hour cache is plenty


def _closes(symbol: str) -> pd.Series | None:
    try:
        df = md.get_history(symbol, interval="1d", range_="1y")
        if df is None or df.empty or len(df) < 60:
            return None
        return df["close"].astype(float)
    except Exception:
        return None


def _ret_pct(s: pd.Series, lookback: int) -> float | None:
    if s is None or len(s) <= lookback:
        return None
    prev = float(s.iloc[-1 - lookback])
    if prev <= 0:
        return None
    return (float(s.iloc[-1]) - prev) / prev * 100.0


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _vix_level() -> float | None:
    """Best-effort current VIX. yfinance ^VIX often rate-limits — that's fine,
    the component just drops out and its weight redistributes."""
    for sym in ("^VIX", "VIXY"):
        s = _closes(sym)
        if s is not None and len(s):
            return float(s.iloc[-1])
    return None


def assess(force: bool = False) -> dict[str, Any]:
    """Compute the current market regime. Cached for 30 min."""
    now = time.time()
    if not force and _CACHE["data"] is not None and (now - _CACHE["ts"]) < _TTL_S:
        return _CACHE["data"]

    spy = _closes("SPY")
    components: list[dict[str, Any]] = []
    parts: dict[str, float] = {}

    spy_meta: dict[str, Any] = {}
    if spy is not None:
        last = float(spy.iloc[-1])
        sma50 = float(spy.rolling(50).mean().iloc[-1])
        sma200 = float(spy.rolling(200).mean().iloc[-1]) if len(spy) >= 200 else None
        ret20 = _ret_pct(spy, 20)
        spy_meta = {
            "last": round(last, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2) if sma200 is not None else None,
            "ret_20d_pct": round(ret20, 2) if ret20 is not None else None,
        }

        # ---- Trend ----
        t = 0.0
        t += 0.5 if last > sma50 else -0.5
        if sma200 is not None:
            t += 0.5 if last > sma200 else -0.5
            # golden/death cross nuance
            t += 0.0 if sma50 >= sma200 else -0.0
        else:
            t += 0.5 if last > sma50 else -0.5  # double-weight 50 when no 200 yet
        t = _clip(t)
        parts["trend"] = t
        components.append({
            "name": "Trend", "score": round(t, 2),
            "label": _word(t),
            "detail": f"SPY {last:.0f} vs 50d {sma50:.0f}"
                      + (f" / 200d {sma200:.0f}" if sma200 is not None else ""),
        })

        # ---- Momentum ----
        if ret20 is not None:
            m = _clip(ret20 / 5.0)  # +5% over 20d => full risk-on
            parts["momentum"] = m
            components.append({
                "name": "Momentum", "score": round(m, 2), "label": _word(m),
                "detail": f"SPY {ret20:+.1f}% over 20 sessions",
            })

    # ---- Breadth / risk appetite (IWM + QQQ rel-strength vs SPY) ----
    spy20 = _ret_pct(spy, 20) if spy is not None else None
    if spy20 is not None:
        rels: list[float] = []
        detail_bits: list[str] = []
        for sym, lbl in (("IWM", "small-caps"), ("QQQ", "tech")):
            s = _closes(sym)
            r = _ret_pct(s, 20) if s is not None else None
            if r is not None:
                rels.append(r - spy20)
                detail_bits.append(f"{lbl} {r - spy20:+.1f}% vs SPY")
        if rels:
            b = _clip(float(np.mean(rels)) / 3.0)  # +3% rel outperformance => full
            parts["breadth"] = b
            components.append({
                "name": "Breadth", "score": round(b, 2), "label": _word(b),
                "detail": "; ".join(detail_bits),
            })

    # ---- Credit (HYG vs TLT) ----
    hyg = _ret_pct(_closes("HYG"), 20)
    tlt = _ret_pct(_closes("TLT"), 20)
    if hyg is not None and tlt is not None:
        c = _clip((hyg - tlt) / 3.0)
        parts["credit"] = c
        components.append({
            "name": "Credit", "score": round(c, 2), "label": _word(c),
            "detail": f"High-yield {hyg:+.1f}% vs Treasuries {tlt:+.1f}% (20d)",
        })

    # ---- Volatility (VIX, best-effort) ----
    vix = _vix_level()
    if vix is not None:
        # 12 → calm (+1), 20 → neutral (0), 30 → stressed (-1)
        v = _clip((20.0 - vix) / 8.0)
        parts["volatility"] = v
        components.append({
            "name": "Volatility", "score": round(v, 2), "label": _word(v),
            "detail": f"VIX {vix:.1f}",
        })

    # ---- Aggregate (renormalise weights over available components) ----
    avail_w = {k: _WEIGHTS[k] for k in parts}
    wsum = sum(avail_w.values()) or 1.0
    composite = sum(parts[k] * avail_w[k] for k in parts) / wsum  # [-1, +1]
    score = int(round(composite * 100))

    if score >= 25:
        regime, headline, playbook, mult = (
            "risk_on",
            "Risk-on — the tape is supportive.",
            "Trend, breadth and credit are aligned. Trade with the tape, full sizing within your risk limits, let winners run.",
            1.0,
        )
    elif score <= -25:
        regime, headline, playbook, mult = (
            "risk_off",
            "Risk-off — defensive posture warranted.",
            "Preserve capital first. Cut gross exposure, demand only A+ setups, tighten stops, and don't fight a falling tape.",
            0.3,
        )
    else:
        regime, headline, playbook, mult = (
            "neutral",
            "Neutral / mixed — no strong edge in the tape.",
            "Signals are crosscurrents. Trade selectively at reduced size (~half), favour relative-strength names, keep cash dry.",
            0.6,
        )

    data = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "score": score,
        "headline": headline,
        "playbook": playbook,
        "position_sizing_mult": mult,
        "components": components,
        "spy": spy_meta,
        "note": "Composite of trend, momentum, breadth, credit and volatility from liquid "
                "ETF proxies. A great setup in a risk-off tape is still a risky trade — "
                "size to the regime, not just the chart.",
    }
    _CACHE["ts"] = now
    _CACHE["data"] = data
    return data


def _word(score: float) -> str:
    if score >= 0.34:
        return "risk-on"
    if score <= -0.34:
        return "risk-off"
    return "neutral"
