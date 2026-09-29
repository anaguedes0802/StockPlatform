"""LLM risk-gate for the trading bot.

The quantitative RSI-2 strategy decides *when* to buy (a brief oversold dip in
an uptrend) and *how much* (position-size math). This module adds the layer a
discretionary trader — or Claude — would want on top: pull in the news flow and
the "smart money" footprint (insiders, institutions, politicians, analysts,
options, social) and let an LLM **pressure-test the trade before it fires.**

Design mirrors `opinion.synthesize_with_llm`: the systematic model owns the
numbers; the LLM may only *confirm or reduce risk* — it can VETO a trade or
DOWNSIZE it, never upsize it and never move the strategy's price levels. So the
backtested edge is preserved, but the bot won't blindly buy a dip that's
cratering on a fraud headline or a wave of insider selling.

Fail-open by design: if no LLM provider is available (all throttled / offline),
`llm_review` returns APPROVE with `provider="unavailable"`, so the proven quant
edge keeps trading rather than halting on LLM downtime.
"""
from __future__ import annotations

import json
from typing import Any

from app.core.logging import log
from app.services import analysts as analysts_svc
from app.services import insider as insider_svc
from app.services import institutions as inst_svc
from app.services import llm
from app.services import news_intel
from app.services import options_flow
from app.services import politicians as poli_svc
from app.services import social as social_svc


def _safe(fn, *args, **kwargs) -> Any:
    """Run a signal fetch; never let one failing source sink the bundle."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def gather_context(symbol: str) -> dict[str, Any]:
    """Collect the news + smart-money bundle for one symbol.

    Each block is independently fault-tolerant. This is intentionally lighter
    than `opinion.collect_signals` (no forecast ensemble / price-action engine)
    so it's cheap to run across a whole universe on every bot pass.
    """
    sym = symbol.upper()
    ctx: dict[str, Any] = {"symbol": sym}
    ctx["news"] = _safe(news_intel.classify_and_score, sym)
    ctx["insider"] = _safe(insider_svc.insider_signal, sym)
    ctx["institutional"] = _safe(inst_svc.institutional_signal, sym)
    ctx["analysts"] = _safe(analysts_svc.analyst_signal, sym)
    ctx["social"] = _safe(social_svc.social_signal, sym)
    ctx["options"] = _safe(options_flow.unusual_activity, sym)
    # Politician trades only when a live adapter is wired up (else noisy errors).
    try:
        if poli_svc.has_live_adapter():
            ctx["politicians"] = _safe(poli_svc.political_signal, sym)
    except Exception:  # noqa: BLE001
        pass
    return ctx


_SYSTEM = (
    "You are the risk-gate for an automated long-only mean-reversion trading bot. "
    "A systematic model has ALREADY decided to BUY a brief oversold dip in an "
    "uptrending stock, and has ALREADY sized the position. Your ONLY job is to "
    "pressure-test that entry against the news flow and smart-money footprint and "
    "decide whether to let it through.\n\n"
    "You may ONLY confirm or REDUCE risk. You cannot upsize, change price levels, "
    "or invent a thesis. Choose exactly one decision:\n"
    "  - APPROVE: nothing material argues against buying this dip.\n"
    "  - DOWNSIZE: tradeable, but elevated risk — cut the size.\n"
    "  - VETO: do not enter. Reserve for genuinely dangerous setups: an active "
    "fraud/accounting/regulatory/litigation/bankruptcy catalyst, a guidance "
    "cut or failed trial, heavy cluster insider SELLING, or news that plausibly "
    "breaks the 'dip will bounce' assumption (the dip is falling for a real, bad "
    "reason).\n\n"
    "Be conservative about vetoing on noise: routine volatility, analyst price-"
    "target nudges, or generic market weakness are NOT veto-worthy. A normal "
    "oversold dip with no scary catalyst is an APPROVE.\n\n"
    "Respond with ONLY this JSON, no prose, no code fences:\n"
    "{\n"
    '  "decision": "APPROVE" | "DOWNSIZE" | "VETO",\n'
    '  "conviction": <float 0..1, how confident you are in the decision>,\n'
    '  "size_multiplier": <float 0..1; 1.0 for APPROVE, ~0.3-0.6 for DOWNSIZE, 0 for VETO>,\n'
    '  "rationale": "<one or two sentences citing the specific signals>",\n'
    '  "key_risks": ["<short risk>", "..."]\n'
    "}"
)


def _approve(provider: str, note: str) -> dict[str, Any]:
    return {
        "decision": "APPROVE", "conviction": 0.5, "size_multiplier": 1.0,
        "rationale": note, "key_risks": [], "provider": provider,
    }


def llm_review(symbol: str, quant_signal: dict[str, Any],
               context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask the LLM to APPROVE / DOWNSIZE / VETO a quant BUY trigger.

    Fail-open: returns APPROVE (provider="unavailable") when no LLM is reachable
    or the output can't be parsed — the systematic edge keeps trading.
    """
    if not llm.is_available():
        return _approve("unavailable", "LLM unavailable — deferring to systematic signal")

    ctx = context if context is not None else gather_context(symbol)
    payload = {
        "symbol": symbol.upper(),
        "quant_trigger": {
            "price": quant_signal.get("price"),
            "rsi": quant_signal.get("rsi"),
            "reason": quant_signal.get("reason"),
        },
        "news": ctx.get("news"),
        "insider": ctx.get("insider"),
        "institutional": ctx.get("institutional"),
        "analysts": ctx.get("analysts"),
        "social": ctx.get("social"),
        "options": ctx.get("options"),
        "politicians": ctx.get("politicians"),
    }
    setup = quant_signal.get("setup")
    ask = (f"The systematic model wants to BUY this {setup} swing signal "
           f"({quant_signal.get('reason', '')}). Pressure-test it." if setup else
           "The systematic model wants to BUY this oversold dip. Pressure-test it.")
    if setup:
        payload["quant_trigger"].update({"stop": quant_signal.get("stop"),
                                         "target": quant_signal.get("target")})
    user = (
        ask + "\n\n"
        "SIGNALS:\n" + json.dumps(payload, default=str)[:9000]
        + "\n\nReturn ONLY the JSON schema from the system prompt."
    )

    # Prefer the cloud "quality" tier (best at nuanced structured judgment).
    result = llm.generate(system=_SYSTEM, user=user, max_tokens=400,
                          temperature=0.2, expect_json=True, tier="quality")
    # If no cloud provider is configured, fall back to whatever IS up (e.g. a
    # local Ollama model). The parse + clamp below guard against a weak model's
    # garbled output — bad JSON simply fails open to APPROVE.
    if not result or not result.json:
        provs = llm.available_providers()
        if provs:
            result = llm.generate(system=_SYSTEM, user=user, max_tokens=400,
                                  temperature=0.2, expect_json=True, provider=provs[0])
    if not result or not result.json:
        return _approve("unavailable", "LLM returned no parseable verdict — deferring to signal")

    out = result.json
    decision = str(out.get("decision", "APPROVE")).upper()
    if decision not in ("APPROVE", "DOWNSIZE", "VETO"):
        decision = "APPROVE"

    # Clamp the size multiplier and keep it consistent with the decision.
    try:
        size_mult = float(out.get("size_multiplier", 1.0))
    except (TypeError, ValueError):
        size_mult = 1.0
    size_mult = max(0.0, min(1.0, size_mult))
    if decision == "VETO":
        size_mult = 0.0
    elif decision == "APPROVE":
        size_mult = 1.0
    elif size_mult >= 1.0 or size_mult <= 0.0:  # DOWNSIZE must actually reduce
        size_mult = 0.5

    try:
        conviction = max(0.0, min(1.0, float(out.get("conviction", 0.5))))
    except (TypeError, ValueError):
        conviction = 0.5

    risks = out.get("key_risks")
    if not isinstance(risks, list):
        risks = []

    return {
        "decision": decision,
        "conviction": round(conviction, 2),
        "size_multiplier": round(size_mult, 2),
        "rationale": str(out.get("rationale", ""))[:600],
        "key_risks": [str(r)[:200] for r in risks[:5]],
        "provider": f"{result.provider}:{result.model}",
    }
