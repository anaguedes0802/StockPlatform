"""AI chat with tool-calling.

Streams Server-Sent Events. If ANTHROPIC_API_KEY is unset, returns a stub
response explaining how to enable it.

Tools wrap existing services so the LLM can answer questions like
"should I buy NVDA?" or "compare AMD vs Intel" with grounded data.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtest.engine import run_backtest as _run_backtest
from app.config import settings
from app.db.models import ChatMessage as ChatMessageDB
from app.db.models import ChatSession, User
from app.db.session import get_db
from app.deps import get_current_user, get_optional_user
from app.ml.ensemble import ensemble_forecast
from app.services import market_data as md
from app.services import news as news_svc
from app.services import politicians as pol_svc
from app.services import price_action as pa_svc
from app.services import screener as screener_svc
from app.services.recommendation import recommend as _recommend

router = APIRouter(prefix="/ai", tags=["ai"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: str | None = None
    session_id: str | None = None   # if provided, persist messages to this session


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    n_messages: int = 0


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str


# ---------------- Tool definitions ----------------

TOOLS = [
    {
        "name": "get_quote",
        "description": "Get the latest price, change, and change % for a single symbol.",
        "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
    {
        "name": "get_profile",
        "description": "Get the company profile (sector, industry, market cap, description) for a symbol.",
        "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
    {
        "name": "get_history",
        "description": "Get OHLCV history for a symbol. interval in {1d,1wk,1mo,1h,5m}. range like 1mo, 6mo, 1y, 5y.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {"type": "string", "default": "1d"},
                "range": {"type": "string", "default": "1y"},
                "limit": {"type": "integer", "default": 30, "description": "Cap on returned bars to keep context small."},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_forecast",
        "description": "AI price forecast for a symbol at a horizon (1d, 5d, 30d, 90d).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "horizon": {"type": "string", "default": "30d"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_recommendation",
        "description": "AI buy/hold/sell recommendation with reasoning for a symbol.",
        "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
    {
        "name": "get_news",
        "description": "Get the latest news headlines for a symbol, with sentiment scores.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "default": 8}},
            "required": ["symbol"],
        },
    },
    {
        "name": "compare_symbols",
        "description": "Compare key stats and current sentiment across multiple symbols.",
        "input_schema": {
            "type": "object",
            "properties": {"symbols": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5}},
            "required": ["symbols"],
        },
    },
    {
        "name": "get_price_action",
        "description": "Smart Money Concepts analysis: order blocks, fair value gaps, BOS/CHOCH, liquidity sweeps, demand/supply zones, and a confluence score for the latest bar.",
        "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
    {
        "name": "find_rising_stars",
        "description": "Discover 'upcoming' / early-stage stocks with momentum, news heat, and constructive technicals. "
                       "Strategy in {rising_stars, breakouts, value_with_catalyst}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "default": "rising_stars"},
                "limit": {"type": "integer", "default": 8},
                "max_market_cap_billions": {"type": "number", "description": "Optional cap in $B (filter)"},
            },
        },
    },
    {
        "name": "run_quick_backtest",
        "description": "Run a quick backtest. kind in {sma_cross, rsi, macd_cross, buy_and_hold}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "kind": {"type": "string"},
                "fast": {"type": "integer", "default": 12},
                "slow": {"type": "integer", "default": 26},
                "initial_cash": {"type": "number", "default": 10000},
            },
            "required": ["symbol", "kind"],
        },
    },
]


_HORIZON_DAYS = {"1d": 1, "5d": 5, "30d": 21, "90d": 63}


def _tool_call(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool to its concrete service implementation."""
    try:
        if name == "get_quote":
            return md.get_quote(args["symbol"].upper())
        if name == "get_profile":
            return md.get_profile(args["symbol"].upper())
        if name == "get_history":
            bars = md.get_history_bars(
                args["symbol"].upper(),
                interval=args.get("interval", "1d"),
                range_=args.get("range", "1y"),
            )
            limit = int(args.get("limit", 30))
            return {"bars": bars[-limit:], "n_total": len(bars)}
        if name == "get_forecast":
            sym = args["symbol"].upper()
            h = args.get("horizon", "30d")
            df = md.get_history(sym, interval="1d", range_="3y")
            if df.empty or len(df) < 200:
                return {"error": "not enough history"}
            r, regime, mix, _ = ensemble_forecast(df, _HORIZON_DAYS.get(h, 21), sym)
            return {
                "symbol": sym, "horizon": h,
                "point": r.point, "p10": r.p10, "p90": r.p90,
                "direction_prob_up": r.direction_prob_up,
                "confidence": r.confidence,
                "regime": regime, "model_mix": mix,
                "contributions": r.contributions,
                "drivers": r.drivers[:5],
            }
        if name == "get_recommendation":
            return _recommend(args["symbol"].upper())
        if name == "get_news":
            return news_svc.fetch_news(args["symbol"].upper(), limit=int(args.get("limit", 8)))
        if name == "compare_symbols":
            out = []
            for s in args["symbols"][:5]:
                s = s.upper()
                try:
                    q = md.get_quote(s)
                    ks = md.get_key_stats(s)
                    sent = news_svc.aggregate_sentiment(s, window_n=15)
                    out.append({"symbol": s, "quote": q, "key_stats": ks, "sentiment_weighted": sent.get("weighted")})
                except Exception as e:
                    out.append({"symbol": s, "error": str(e)})
            return out
        if name == "get_price_action":
            sym = args["symbol"].upper()
            df = md.get_history(sym, interval="1d", range_="1y")
            if df.empty:
                return {"error": "no data"}
            r = pa_svc.analyze(df)
            # trim verbose lists for chat context
            if r.get("ok"):
                r["swing_points"] = r["swing_points"][-10:]
                r["fvgs"] = [f for f in r["fvgs"] if not f["filled"]][:6]
                r["order_blocks"] = [o for o in r["order_blocks"] if not o["mitigated"]][:6]
            return r
        if name == "find_rising_stars":
            cap = args.get("max_market_cap_billions")
            return screener_svc.run_screener(
                strategy=args.get("strategy", "rising_stars"),
                limit=int(args.get("limit", 8)),
                max_market_cap=cap * 1e9 if cap else None,
            )
        if name == "run_quick_backtest":
            dsl = {
                "kind": args["kind"],
                "fast": int(args.get("fast", 12)),
                "slow": int(args.get("slow", 26)),
            }
            res = _run_backtest(
                args["symbol"].upper(), dsl, initial_cash=float(args.get("initial_cash", 10_000.0)),
            )
            return {"metrics": res["metrics"], "benchmark_metrics": res["benchmark"]["metrics"], "n_trades": res["metrics"]["n_trades"]}
    except Exception as e:
        return {"error": str(e)}
    return {"error": f"unknown tool: {name}"}


# ---------------- Streaming ----------------

def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _stub_stream(messages: list[ChatMessage]) -> AsyncIterator[str]:
    msg = (
        "(AI chat needs a configured LLM provider. Add one of these to "
        "apps/api/.env: GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, or "
        "install Ollama and pull a model. Then restart the API.)"
    )
    for word in msg.split(" "):
        yield _sse({"type": "delta", "text": word + " "})
    yield _sse({"type": "done"})


def _tools_openai_schema() -> list[dict]:
    """Convert our Anthropic-style TOOLS array to OpenAI function-calling
    schema. Groq is OpenAI-API-compatible so it accepts this directly; same
    schema works for OpenRouter, Together, and most other compatible APIs."""
    out = []
    for t in TOOLS:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


async def _groq_tool_loop(req: ChatRequest) -> AsyncIterator[str]:
    """Multi-turn tool-using chat against Groq.

    Groq exposes OpenAI-compatible /chat/completions which supports the
    same `tools` + `tool_choice` fields as OpenAI's API. We:
      1. Send the conversation + tool schema to Groq.
      2. If the response includes tool calls, execute each via _tool_call(),
         append the result as a `tool` role message, and loop.
      3. When the model returns plain text (no more tool calls), stream that
         word-by-word as SSE deltas — mirrors the Anthropic path's UX.

    Source: https://console.groq.com/docs/tool-use
    """
    import os, json as _json
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        async for c in _stub_stream(req.messages):
            yield c
        return

    # Tool-call models tried in order: the small gpt-oss first (fast, generous
    # free quota), the 120B as fallback (better at multi-tool routing). Both are
    # reasoning models: keep reasoning effort low and leave enough token budget
    # for the hidden reasoning plus the final answer.
    primary_tool_model   = os.environ.get("GROQ_TOOL_MODEL", "openai/gpt-oss-20b")
    secondary_tool_model = "openai/gpt-oss-120b" if primary_tool_model != "openai/gpt-oss-120b" else "openai/gpt-oss-20b"

    system = req.system or (
        "You are the in-platform AI assistant for a stock analysis app. "
        "Your single most important rule: WHENEVER the user mentions a ticker "
        "symbol (e.g. AAPL, FEMY, NVDA, ROKU), you MUST call the relevant "
        "tools BEFORE answering. Do not say 'I don't have data' or 'check the "
        "profile page' — you have tools, use them. "
        "Standard tool sequence for any ticker question: "
        "  1. get_quote — current price + change "
        "  2. get_recommendation — buy/hold/sell signal "
        "  3. get_news — recent headlines + sentiment "
        "  4. get_price_action — Smart-Money technicals "
        "  5. get_forecast — if user asks about future "
        "Call multiple tools in parallel when possible. Cite real numbers "
        "from tool outputs in your reply. Never invent metrics. "
        "You can analyze and present data freely, but do not give personalized "
        "buy/sell advice — phrase actions conditionally ('if X drops to $Y, "
        "the setup gets attractive') rather than imperatively ('buy now')."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    for m in req.messages[-12:]:
        messages.append({"role": m.role, "content": m.content})

    tools_schema = _tools_openai_schema()

    # Multi-turn loop — bounded so a misbehaving model can't run forever.
    # Each round tries the primary model; on 429 we switch to the secondary
    # for this round and remember the switch for the rest of the loop (saves
    # cycling on every round).
    MAX_TOOL_ROUNDS = 5
    final_text: str | None = None
    active_model = primary_tool_model
    for round_idx in range(MAX_TOOL_ROUNDS):
        msg = None
        # Try active model then fallback on 429.
        for attempt_model in (active_model, secondary_tool_model):
            try:
                r = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": attempt_model,
                        "messages": messages,
                        "tools": tools_schema,
                        "tool_choice": "auto",
                        "max_tokens": 4000,
                        "temperature": 0.3,
                        "reasoning_effort": "low",
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=45.0,
                )
                if r.status_code == 429:
                    # Tell the UI we're switching, then retry with the other model.
                    if attempt_model == active_model and attempt_model != secondary_tool_model:
                        active_model = secondary_tool_model
                        continue
                    # Both models 429'd. If tools already ran, present what we
                    # have instead of throwing away the data and asking a blank
                    # LLM to start over.
                    if _has_tool_results(messages):
                        async for c in _stream_tool_results_summary(messages):
                            yield c
                        return
                    yield _sse({"type": "delta", "text": "(All Groq tool-capable models are rate-limited. Falling back to direct chat.)"})
                    async for c in _generic_llm_stream_simple(req, system):
                        yield c
                    return
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                active_model = attempt_model
                break
            except Exception as e:
                if attempt_model == secondary_tool_model:
                    yield _sse({"type": "delta", "text": f"(Tool-calling failed: {e}. Falling back to direct chat.)"})
                    async for c in _generic_llm_stream_simple(req, system):
                        yield c
                    return
                # else try the secondary
                continue
        if msg is None:
            yield _sse({"type": "delta", "text": "(Tool loop produced no message; bailing.)"})
            yield _sse({"type": "done"})
            return

        # If the model returned tool calls, execute them and append the results.
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            # Notify the UI which tools fired (mirrors Anthropic path UX).
            for tc in tool_calls:
                fn = tc.get("function", {})
                yield _sse({"type": "tool_use", "name": fn.get("name"), "input": _safe_args(fn.get("arguments"))})
            # the assistant message that requested tools (without the model's
            # private reasoning, which the API does not accept back as input)
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = _safe_args(fn.get("arguments"))
                try:
                    result = _tool_call(fn.get("name"), args)
                except Exception as e:
                    result = {"error": str(e)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": fn.get("name"),
                    "content": _json.dumps(result, default=str)[:6000],
                })
            continue   # let the model see the tool outputs

        # No more tool calls — model produced its final answer.
        final_text = (msg.get("content") or "").strip()
        break

    if not final_text:
        final_text = "(Tool loop terminated without a final answer. Try a more specific question.)"

    yield _sse({"type": "meta", "provider": "groq", "model": active_model, "tool_loop": True})
    import asyncio as _a
    for chunk in final_text.split(" "):
        yield _sse({"type": "delta", "text": chunk + " "})
        await _a.sleep(0.005)
    yield _sse({"type": "done"})


def _safe_args(raw: Any) -> dict[str, Any]:
    """OpenAI returns function arguments as a JSON string. Parse defensively."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        import json as _json
        return _json.loads(raw)
    except Exception:
        return {}


def _has_tool_results(messages: list[dict]) -> bool:
    """True if any tool role message exists — meaning we collected real data
    that should be presented even if the LLM can't synthesize a summary."""
    return any(m.get("role") == "tool" for m in messages)


async def _stream_tool_results_summary(messages: list[dict]) -> AsyncIterator[str]:
    """Deterministic fallback when both LLM models 429 mid-loop. The tools
    already ran and we have their structured outputs — present them in a
    readable form so the user gets the actual data, not a "please retry"
    message.

    This is the difference between losing 5 expensive tool calls or
    surfacing $0.41 / -3.2% / "no overhead supply" / "8 articles, +0.31
    sentiment" to the user even when the synthesis model is rate-limited.
    """
    import json as _json
    import asyncio as _a

    yield _sse({"type": "delta", "text": "(LLM synthesis quota exhausted — presenting raw tool output instead.)\n\n"})

    # Group tool calls back to (name, args, result) tuples in order.
    pending_calls: dict[str, tuple[str, dict]] = {}   # tool_call_id → (name, args)
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                pending_calls[tc.get("id")] = (fn.get("name"), _safe_args(fn.get("arguments")))

    for m in messages:
        if m.get("role") != "tool":
            continue
        call_id = m.get("tool_call_id")
        name, args = pending_calls.get(call_id, (m.get("name", "?"), {}))
        try:
            result = _json.loads(m.get("content") or "{}")
        except Exception:
            result = m.get("content")
        formatted = _format_tool_output(name, args, result)
        # Stream in a couple of chunks so the UI doesn't flush all at once
        for line in formatted.split("\n"):
            yield _sse({"type": "delta", "text": line + "\n"})
            await _a.sleep(0.01)

    yield _sse({"type": "delta", "text": "\nTry again in ~60 seconds for a synthesized LLM analysis."})
    yield _sse({"type": "done"})


def _format_tool_output(name: str, args: dict, result: Any) -> str:
    """Human-readable rendering of a single tool result. Plain text — keeps
    things readable when streamed directly to chat. Tolerates result being
    either a dict or a JSON-encoded string (tool dispatcher sometimes
    pre-serializes)."""
    sym = args.get("symbol", "") if isinstance(args, dict) else ""
    header = f"━━ {name}({sym}) ━━"
    # Defensively re-parse if result is a JSON string
    if isinstance(result, str):
        try:
            result = _json.loads(result)
        except Exception:
            return f"{header}\n  {result[:400]}\n"
    if isinstance(result, dict) and result.get("error"):
        return f"{header}\n  error: {result['error']}\n"
    try:
        if name == "get_quote":
            q = result.get("quote", result) if isinstance(result, dict) else {}
            return f"{header}\n  price: ${q.get('price')}  change: {q.get('change')} ({q.get('change_pct')}%)\n"
        if name == "get_profile":
            p = result.get("profile", result) if isinstance(result, dict) else {}
            return (f"{header}\n  name: {p.get('name')}\n  sector: {p.get('sector')}\n"
                    f"  market cap: {p.get('market_cap')}\n  description: {(p.get('description') or '')[:200]}\n")
        if name == "get_recommendation":
            return (f"{header}\n  verdict: {result.get('verdict')}  score: {result.get('score')}\n"
                    f"  reasoning: {(result.get('reasoning') or {}).get('summary', '')[:300]}\n")
        if name == "get_news":
            items = result if isinstance(result, list) else (result.get("items") if isinstance(result, dict) else [])
            lines = [header]
            for it in (items or [])[:6]:
                if not isinstance(it, dict):
                    continue
                lines.append(f"  · {(it.get('title') or '')[:120]}  (sent {it.get('sentiment')})")
            return "\n".join(lines) + "\n"
        if name == "get_price_action":
            # Result is the full pa.analyze() dict. Pull the useful summary fields.
            return (f"{header}\n  trend: {result.get('current_trend')}  "
                    f"in_discovery: {result.get('price_in_discovery')}  "
                    f"nearest_demand_below: {result.get('nearest_demand_pct_below')}%  "
                    f"nearest_supply_above: {result.get('nearest_supply_pct_above')}%\n"
                    f"  active_demand: {len(result.get('active_demand') or [])} zones, "
                    f"active_supply: {len(result.get('active_supply') or [])} zones\n")
        if name == "get_forecast":
            fcs = result.get("forecasts", result) if isinstance(result, dict) else result
            if isinstance(fcs, list) and fcs:
                f0 = fcs[0]
                dp = f0.get("direction_prob")
                pup = dp.get("up") if isinstance(dp, dict) else dp
                return (f"{header}\n  horizon: {f0.get('horizon')}  point: ${f0.get('point')}  "
                        f"P(up): {pup}  conf: {f0.get('confidence')}\n")
            if isinstance(fcs, dict):
                return (f"{header}\n  horizon: {fcs.get('horizon')}  point: ${fcs.get('point')}  "
                        f"P(up): {fcs.get('direction_prob_up')}  conf: {fcs.get('confidence')}  "
                        f"regime: {fcs.get('regime')}\n")
        # Generic dict dump (capped)
        return f"{header}\n  {_json.dumps(result, default=str, indent=2)[:600]}\n"
    except Exception as e:
        return f"{header}\n  (could not format: {e})\n"


import json as _json  # noqa: E402 — used by _format_tool_output


async def _generic_llm_stream_simple(req: ChatRequest, system: str | None = None) -> AsyncIterator[str]:
    """Fallback when tool-calling fails — direct chat with no tools.
    Same as _generic_llm_stream but with the optional system arg."""
    from app.services import llm
    convo = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in req.messages[-8:]) + "\n\nASSISTANT:"
    sys = system or (
        "You are the in-platform AI assistant for a stock analysis app. "
        "Keep answers concise (2-4 paragraphs), no personalized advice."
    )
    import asyncio as _a
    result = await _a.to_thread(llm.generate, sys, convo, max_tokens=800, temperature=0.5)
    if not result or not result.text:
        yield _sse({"type": "delta", "text": "(LLM unavailable.)"})
        yield _sse({"type": "done"})
        return
    for chunk in result.text.split(" "):
        yield _sse({"type": "delta", "text": chunk + " "})
        await _a.sleep(0.005)
    yield _sse({"type": "done"})


async def _generic_llm_stream(req: ChatRequest) -> AsyncIterator[str]:
    """Stream a single-turn response via the unified LLM provider (Groq /
    Gemini / Ollama). No tool-calling — just direct chat.

    The system prompt frames the assistant as the platform's analyst, so
    answers stay on-topic. Tool calling is only on the Anthropic path
    (it's the most reliable function-calling implementation).
    """
    from app.services import llm
    if not llm.is_available():
        async for chunk in _stub_stream(req.messages):
            yield chunk
        return

    # Build the chat transcript into one user prompt.
    convo: list[str] = []
    for m in req.messages[-8:]:        # last 8 turns is plenty
        role = m.role.upper()
        convo.append(f"{role}: {m.content}")
    user_msg = "\n\n".join(convo) + "\n\nASSISTANT:"

    system = req.system or (
        "You are the in-platform AI assistant for a stock analysis app. "
        "You can discuss any stock, market dynamics, technicals, fundamentals, "
        "news, and trading strategy. Keep answers concise (2-4 paragraphs), "
        "cite real numbers when possible, never give personalized financial "
        "advice, and acknowledge uncertainty. If a user asks about a specific "
        "ticker and you don't have current data, tell them to open the symbol "
        "page in the platform for real-time data — don't invent numbers."
    )

    # Run the (blocking) llm call in a thread so we don't block the event loop,
    # then "stream" the result word-by-word for a similar UX to Anthropic SSE.
    import asyncio
    result = await asyncio.to_thread(
        llm.generate,
        system, user_msg,
        max_tokens=800, temperature=0.5, expect_json=False,
    )
    if not result or not result.text:
        yield _sse({"type": "delta", "text": "(LLM call failed — all configured providers may be rate-limited. Try again in 30s.)"})
        yield _sse({"type": "done"})
        return

    yield _sse({"type": "meta", "provider": result.provider, "model": result.model})
    text = result.text
    # Word-by-word "streaming" — fake-stream so the UI behavior matches Anthropic.
    for chunk in text.split(" "):
        yield _sse({"type": "delta", "text": chunk + " "})
        await asyncio.sleep(0.005)
    yield _sse({"type": "done"})


async def _anthropic_stream(req: ChatRequest) -> AsyncIterator[str]:
    """Call Anthropic with our tools, surface tool calls + final text as SSE."""
    try:
        from anthropic import Anthropic  # imported lazily so the API runs without it
    except Exception:
        async for chunk in _stub_stream(req.messages):
            yield chunk
        return

    client = Anthropic(api_key=settings.anthropic_api_key)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    system = req.system or (
        "You are a finance analyst assistant inside the StockPlatform app. "
        "Use the provided tools to look up quotes, forecasts, recommendations, news, and run quick backtests "
        "before answering. Always be explicit about uncertainty and avoid giving investment advice; "
        "frame outputs as analysis. Cite the tools' numeric outputs."
    )

    for _ in range(6):  # cap turns
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
        )

        text_chunks: list[str] = []
        tool_uses: list[Any] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_uses.append(block)

        for chunk in text_chunks:
            for word in chunk.split(" "):
                yield _sse({"type": "delta", "text": word + " "})

        if resp.stop_reason != "tool_use" or not tool_uses:
            yield _sse({"type": "done"})
            return

        # Execute each tool, append assistant block + tool_result block, loop.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for tu in tool_uses:
            yield _sse({"type": "tool", "name": tu.name, "input": tu.input})
            result = _tool_call(tu.name, dict(tu.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, default=str)[:8000],  # cap result size
            })
        messages.append({"role": "user", "content": tool_results})

    yield _sse({"type": "done"})


def _persist_user_turn(db: Session, user: User | None, req: ChatRequest) -> str | None:
    """If the user is logged in, persist the LAST user message + (later) the
    assistant response. Returns the session id used (created lazily).

    We can't stream the assistant chunks to the DB easily here — the SSE
    response is generated lazily. We do, however, persist the user prompt
    now and write a simplified assistant placeholder; the UI can call
    /chat/sessions/{id}/finalize with the final assistant text after streaming.
    """
    if user is None or not req.messages:
        return req.session_id
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if last_user is None:
        return req.session_id

    try:
        if req.session_id:
            sess = db.get(ChatSession, uuid.UUID(req.session_id))
            if not sess or sess.user_id != user.id:
                sess = None
        else:
            sess = None
        if sess is None:
            title = (last_user.content[:60] + "…") if len(last_user.content) > 60 else last_user.content
            sess = ChatSession(user_id=user.id, title=title)
            db.add(sess)
            db.commit()
            db.refresh(sess)
        db.add(ChatMessageDB(session_id=sess.id, role="user", content=last_user.content))
        db.commit()
        return str(sess.id)
    except Exception:
        db.rollback()
        return req.session_id


@router.post("/chat")
async def chat(req: ChatRequest,
                db: Session = Depends(get_db),
                user: User | None = Depends(get_optional_user)):
    """Routing (preferred in order):
      1. Anthropic — streaming with tool-calling (best UX, paid)
      2. Groq with OpenAI-compatible tool calling (free, works well)
      3. Any LLM provider (Groq/Gemini/Ollama) — direct chat, no tools
      4. Stub message asking user to configure a provider
    """
    import os
    session_id = _persist_user_turn(db, user, req)
    headers = {"X-Session-Id": session_id} if session_id else {}
    if settings.anthropic_api_key:
        return StreamingResponse(_anthropic_stream(req), media_type="text/event-stream", headers=headers)
    if os.environ.get("GROQ_API_KEY"):
        # Tool-using Groq path. Falls through to direct chat internally on
        # rate-limit or tool-loop failure.
        return StreamingResponse(_groq_tool_loop(req), media_type="text/event-stream", headers=headers)
    # No tool-capable provider — direct chat via whatever's available.
    return StreamingResponse(_generic_llm_stream(req), media_type="text/event-stream", headers=headers)


# ---------------- Session CRUD ----------------


@router.get("/chat/sessions", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[SessionOut]:
    rows = db.scalars(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc())
    ).all()
    out: list[SessionOut] = []
    for s in rows:
        n = db.query(ChatMessageDB).filter(ChatMessageDB.session_id == s.id).count()
        out.append(SessionOut(
            id=str(s.id), title=s.title,
            created_at=s.created_at.isoformat(), updated_at=s.updated_at.isoformat(),
            n_messages=n,
        ))
    return out


@router.get("/chat/sessions/{sid}/messages", response_model=list[MessageOut])
def get_messages(sid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[MessageOut]:
    sess = db.get(ChatSession, sid)
    if not sess or sess.user_id != user.id:
        raise HTTPException(404, "session not found")
    msgs = db.scalars(
        select(ChatMessageDB).where(ChatMessageDB.session_id == sid).order_by(ChatMessageDB.created_at.asc())
    ).all()
    return [MessageOut(role=m.role, content=m.content, created_at=m.created_at.isoformat()) for m in msgs]


class FinalizeIn(BaseModel):
    assistant_content: str
    tool_calls: list = []


@router.post("/chat/sessions/{sid}/finalize", status_code=204)
def finalize_assistant_turn(sid: uuid.UUID, payload: FinalizeIn,
                            db: Session = Depends(get_db),
                            user: User = Depends(get_current_user)) -> None:
    """Called by the client after the SSE stream completes — saves the
    assembled assistant response so the session can be replayed later."""
    sess = db.get(ChatSession, sid)
    if not sess or sess.user_id != user.id:
        raise HTTPException(404, "session not found")
    db.add(ChatMessageDB(session_id=sid, role="assistant",
                          content=payload.assistant_content,
                          tool_calls=payload.tool_calls or []))
    db.commit()


@router.delete("/chat/sessions/{sid}", status_code=204)
def delete_session(sid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    sess = db.get(ChatSession, sid)
    if not sess or sess.user_id != user.id:
        raise HTTPException(404, "session not found")
    db.delete(sess)
    db.commit()
