"""Portfolio review — position-aware AI assistant.

For each position the user holds, this service produces a structured "what to
do now" recommendation that combines:
  - The position's history: entry price, days held, P&L, where the user is on
    their original thesis.
  - The current opinion-engine signal pack: verdict, structure, trend,
    forecast, news, insider/institutional flow.
  - "What changed since you bought" — recent structure events, regime
    transitions, sentiment shifts, drawdown thresholds crossed.

Output: a `PositionReview` per holding with verdict (HOLD / ADD / TRIM / SELL /
STOP_OUT), confidence, a recommended stop, a typed list of alerts (info /
warning / critical), and a short narrative.

When `ANTHROPIC_API_KEY` is set we ask Claude to synthesize the recommendation
(better at integrating context). Otherwise a deterministic rule synthesizer
runs — same output shape, less nuanced narrative.

Honest scope: this is on-demand (user clicks "Review with AI"), not a 24/7
monitor. A scheduled background scanner is a Phase 2 build that needs careful
budget management on the Anthropic credit side.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from app.config import settings
from app.core.logging import log
from app.services import market_data as md
from app.services.opinion import collect_signals

ReviewVerdict = Literal["HOLD", "ADD", "TRIM", "SELL", "STOP_OUT"]
AlertSeverity = Literal["info", "warning", "critical"]


@dataclass
class PositionReview:
    symbol: str
    quantity: float
    cost_basis: float            # avg buy price per share
    last_price: float
    market_value: float
    unrealized_pl_pct: float
    days_held: int
    verdict: ReviewVerdict
    confidence: float            # 0..1
    recommended_stop: float | None
    alerts: list[dict[str, Any]] # {severity, kind, message}
    what_changed: list[str]      # bullets — what's different since entry
    narrative: str               # 1-2 sentence summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "cost_basis": self.cost_basis,
            "last_price": self.last_price,
            "market_value": self.market_value,
            "unrealized_pl_pct": round(self.unrealized_pl_pct, 2),
            "days_held": self.days_held,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "recommended_stop": round(self.recommended_stop, 2) if self.recommended_stop else None,
            "alerts": self.alerts,
            "what_changed": self.what_changed,
            "narrative": self.narrative,
        }


def _days_held(transactions: list[Any]) -> int:
    """Earliest BUY of any txn for this symbol → days from then to now."""
    buys = [t for t in transactions if t.side == "buy"]
    if not buys:
        return 0
    earliest = min(t.occurred_at for t in buys)
    # transactions may be stored naive in sqlite — normalize.
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - earliest).days)


def _build_alerts(
    signals: dict[str, Any],
    cost_basis: float,
    last_price: float,
    days_held: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate typed alerts + a 'what changed' list based on signals + position context."""
    alerts: list[dict[str, Any]] = []
    changed: list[str] = []

    pnl_pct = (last_price - cost_basis) / cost_basis * 100 if cost_basis else 0.0
    pa = signals.get("price_action") or {}
    tech = signals.get("technical") or {}
    atr_pct = float(tech.get("atr_pct") or 0.0)
    rec = signals.get("recommendation") or {}
    rec_score = float(rec.get("score") or 0.0)
    earn = signals.get("earnings") or {}
    nxt = earn.get("next") or {}

    # --- critical alerts -----
    if pnl_pct < -15:
        alerts.append({"severity": "critical", "kind": "drawdown",
                       "message": f"Position down {pnl_pct:.1f}% — exceeds 15% drawdown threshold"})
    elif pnl_pct < -8:
        alerts.append({"severity": "warning", "kind": "drawdown",
                       "message": f"Position down {pnl_pct:.1f}%"})

    # Structure breakdown — most recent structure event is bearish CHOCH/BOS
    events = pa.get("recent_structure") or []
    if events:
        last_ev = events[-1]
        if last_ev.get("direction") == "bearish":
            alerts.append({"severity": "warning", "kind": "structure_break",
                           "message": f"Recent {last_ev.get('kind','event')} bearish — trend integrity at risk"})
            changed.append(f"Bearish {last_ev.get('kind','event')} just printed")

    # Stop proximity — if our recommended stop is within 3% of price, warn
    rec_risk = (rec.get("reasoning") or {}).get("risk") or {}
    suggested_stop = rec_risk.get("suggested_stop")
    if suggested_stop and last_price > 0:
        stop_dist = (last_price - suggested_stop) / last_price * 100
        if stop_dist < 3:
            alerts.append({"severity": "warning", "kind": "stop_approaching",
                           "message": f"Price within {stop_dist:.1f}% of suggested stop ${suggested_stop:.2f}"})

    # Earnings in next 14 days
    if nxt.get("days_to_earnings") is not None and 0 < nxt["days_to_earnings"] <= 14:
        alerts.append({"severity": "info", "kind": "earnings",
                       "message": f"Earnings in {nxt['days_to_earnings']:.0f} days — binary event risk"})

    # Recommendation has flipped bearish since a "presumed-bullish" entry
    if rec_score < -0.2 and pnl_pct > 0:
        changed.append("Recommendation score is now bearish despite position being profitable")
        alerts.append({"severity": "warning", "kind": "thesis_change",
                       "message": "Original bullish thesis weakening — consider trimming gains"})

    # Insider/institutional change since entry (we don't track historical, so we use current direction)
    ins = signals.get("insider") or {}
    if ins.get("score", 0) < -0.2:
        changed.append(f"Insider selling pressure ({ins.get('summary','')})")

    pol = signals.get("politicians") or {}
    if pol.get("score", 0) < -0.2:
        changed.append("Politicians selling")

    # Trend
    trend = pa.get("trend")
    if trend == "down":
        alerts.append({"severity": "warning", "kind": "trend_flip",
                       "message": "Trend has flipped down"})

    # High vol regime
    regime = signals.get("regime", "")
    if "high_vol" in regime and atr_pct > 0.04:
        alerts.append({"severity": "info", "kind": "volatility",
                       "message": f"High-vol regime ({regime}) — wider stop appropriate"})

    return alerts, changed


def _deterministic_verdict(
    signals: dict[str, Any],
    cost_basis: float,
    last_price: float,
    days_held: int,
    alerts: list[dict[str, Any]],
) -> tuple[ReviewVerdict, float, str]:
    """Verdict from rules. Returns (verdict, confidence, narrative)."""
    pnl_pct = (last_price - cost_basis) / cost_basis * 100 if cost_basis else 0.0
    rec = signals.get("recommendation") or {}
    rec_score = float(rec.get("score") or 0.0)
    pa = signals.get("price_action") or {}
    trend = pa.get("trend") or "sideways"

    critical = any(a["severity"] == "critical" for a in alerts)
    warnings = sum(1 for a in alerts if a["severity"] == "warning")

    # Stop-out: drawdown > 15% OR price below user-defined stop (we don't have user stops yet)
    if critical and pnl_pct < -15:
        return "STOP_OUT", 0.9, (
            f"Position down {pnl_pct:.1f}%. Drawdown exceeds the 15% pain threshold — "
            f"cut losses, preserve capital."
        )

    # Sell: rec turned strongly bearish AND we have profits to lock in
    if rec_score < -0.30 and pnl_pct > 5:
        return "SELL", 0.75, (
            f"Original thesis is gone (rec score {rec_score:+.2f}). You have a {pnl_pct:+.1f}% "
            f"gain to lock in. Sell into strength rather than wait for the breakdown."
        )

    # Trim: rec turning soft + multiple warnings
    if rec_score < -0.10 and warnings >= 2:
        return "TRIM", 0.65, (
            f"Conditions are deteriorating ({warnings} warnings, rec {rec_score:+.2f}). "
            f"Reduce position by half to protect P&L while keeping core exposure."
        )

    # Add: still strong + dips into demand zone
    if rec_score > 0.30 and pnl_pct < 5 and trend == "up":
        return "ADD", 0.70, (
            f"Setup remains constructive (rec {rec_score:+.2f}, trend up). "
            f"Pullback is an opportunity to add."
        )

    # Hold (default)
    conf = 0.70 if abs(rec_score) < 0.20 else 0.55
    if pnl_pct > 0:
        narrative = (
            f"Position is +{pnl_pct:.1f}% over {days_held} days. "
            f"Thesis intact (rec {rec_score:+.2f}, trend {trend}). Continue to hold."
        )
    else:
        narrative = (
            f"Position is {pnl_pct:+.1f}% over {days_held} days but below average-cost. "
            f"No structural break yet — give it room, but respect the stop."
        )
    return "HOLD", conf, narrative


def _llm_verdict(
    signals: dict[str, Any],
    cost_basis: float,
    last_price: float,
    days_held: int,
    quantity: float,
    alerts: list[dict[str, Any]],
    what_changed: list[str],
) -> tuple[ReviewVerdict, float, str] | None:
    """LLM synthesis via the unified provider (Ollama / Groq / Gemini / Anthropic).
    Returns None on any failure (caller falls back to deterministic verdict)."""
    from app.services import llm
    if not llm.is_available():
        return None

    pnl_pct = (last_price - cost_basis) / cost_basis * 100 if cost_basis else 0.0
    pa = signals.get("price_action") or {}
    rec = signals.get("recommendation") or {}
    fc = signals.get("forecast_30d") or {}
    earnings = signals.get("earnings") or {}
    nxt = earnings.get("next") or {}

    # Compact context — strip noise so we don't burn tokens.
    context = {
        "symbol": signals.get("symbol"),
        "position": {
            "quantity": quantity,
            "cost_basis": round(cost_basis, 2),
            "current_price": round(last_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
        },
        "signals": {
            "verdict_score": rec.get("score"),
            "trend": pa.get("trend"),
            "regime": signals.get("regime"),
            "structure_last": (pa.get("recent_structure") or [])[-1:],
            "forecast_30d_pct": round(((fc.get("point") or last_price) - last_price) / last_price * 100, 1) if fc.get("point") else None,
            "forecast_p_up": fc.get("direction_prob_up"),
            "insider_score": (signals.get("insider") or {}).get("score"),
            "institutional_score": (signals.get("institutional") or {}).get("score"),
            "earnings_days": nxt.get("days_to_earnings"),
        },
        "alerts": alerts,
        "what_changed_since_entry": what_changed,
    }

    system = (
        "You are a disciplined portfolio risk manager reviewing an existing position. "
        "Output ONE JSON object with keys: verdict (HOLD|ADD|TRIM|SELL|STOP_OUT), "
        "confidence (0..1), narrative (2-3 sentences, plain English, no fluff). "
        "Be decisive but conservative — only TRIM/SELL/STOP_OUT when evidence is clear. "
        "ADD only when setup is still constructive and the position is at small gain or breakeven."
    )
    result = llm.generate(system, json.dumps(context, default=str),
                          max_tokens=400, temperature=0.3, expect_json=True)
    if not result or not result.json:
        return None
    data = result.json
    v = str(data.get("verdict", "")).upper()
    if v not in ("HOLD", "ADD", "TRIM", "SELL", "STOP_OUT"):
        return None
    return v, float(data.get("confidence", 0.5)), str(data.get("narrative", "")).strip()


def review_position(
    symbol: str,
    quantity: float,
    cost_basis: float,
    transactions: list[Any],
) -> PositionReview:
    sig = collect_signals(symbol)
    if sig.get("error") or not sig.get("last_price"):
        return PositionReview(
            symbol=symbol, quantity=quantity, cost_basis=cost_basis,
            last_price=0.0, market_value=0.0, unrealized_pl_pct=0.0, days_held=0,
            verdict="HOLD", confidence=0.0, recommended_stop=None,
            alerts=[{"severity": "warning", "kind": "data_unavailable",
                     "message": f"No price data for {symbol}"}],
            what_changed=[], narrative="Unable to load signals.",
        )

    last_price = float(sig["last_price"])
    days = _days_held(transactions)
    market_value = quantity * last_price
    pnl_pct = (last_price - cost_basis) / cost_basis * 100 if cost_basis else 0.0
    alerts, changed = _build_alerts(sig, cost_basis, last_price, days)

    # Pick a recommended stop: use 2× ATR below current OR halfway to cost-basis,
    # whichever is HIGHER (more protective).
    rec = sig.get("recommendation") or {}
    rec_risk = (rec.get("reasoning") or {}).get("risk") or {}
    atr_pct = float((sig.get("technical") or {}).get("atr_pct") or 0.0)
    candidates = []
    if rec_risk.get("suggested_stop"):
        candidates.append(float(rec_risk["suggested_stop"]))
    if atr_pct > 0:
        candidates.append(last_price * (1 - 2 * atr_pct))
    if pnl_pct > 5:
        # Trailing stop at break-even — protect the gain
        candidates.append(cost_basis)
    recommended_stop = max(candidates) if candidates else None

    # Verdict — try LLM, fall back to deterministic
    llm = _llm_verdict(sig, cost_basis, last_price, days, quantity, alerts, changed)
    if llm:
        verdict, confidence, narrative = llm
    else:
        verdict, confidence, narrative = _deterministic_verdict(sig, cost_basis, last_price, days, alerts)

    return PositionReview(
        symbol=symbol, quantity=quantity, cost_basis=cost_basis,
        last_price=last_price, market_value=market_value, unrealized_pl_pct=pnl_pct,
        days_held=days, verdict=verdict, confidence=confidence,
        recommended_stop=recommended_stop, alerts=alerts, what_changed=changed,
        narrative=narrative,
    )


def review_portfolio(
    positions: list[dict[str, Any]],
    transactions_by_symbol: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Review every position in a portfolio. Returns list of dicts (one per holding)."""
    out: list[dict[str, Any]] = []
    for pos in positions:
        sym = pos["symbol"]
        try:
            review = review_position(
                symbol=sym,
                quantity=float(pos["quantity"]),
                cost_basis=float(pos["avg_buy_price"]),
                transactions=transactions_by_symbol.get(sym, []),
            )
            out.append(review.to_dict())
        except Exception as e:
            log.warning("portfolio_review_position_failed", symbol=sym, err=str(e))
            out.append({
                "symbol": sym,
                "verdict": "HOLD",
                "confidence": 0.0,
                "narrative": f"Review failed: {e}",
                "alerts": [],
                "what_changed": [],
            })
    return out
