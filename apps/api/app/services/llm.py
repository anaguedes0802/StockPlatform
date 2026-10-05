"""LLM provider abstraction — Ollama (free, local) > Groq (free, cloud) > Gemini (free, cloud) > Anthropic (paid).

A single `generate(...)` entrypoint picks the best available backend at call
time, so the rest of the codebase doesn't care which LLM is running. All
backends return a `LLMResult` with `.text` (parsed string) and `.json` (best-
effort parsed dict if `expect_json=True`).

Order of preference is configurable via `LLM_PROVIDER` env var:
  - "auto" (default): try in order ollama → groq → gemini → anthropic → None
  - "ollama" | "groq" | "gemini" | "anthropic": force a specific provider

Setup notes (free options):
  - Ollama: install at https://ollama.com, then `ollama pull llama3.1:8b` (or
    `qwen2.5:7b`). Runs locally, no API key, no rate limits.
  - Groq:   https://console.groq.com — sign up free, generate API key, set
    GROQ_API_KEY in .env. ~30 req/min on Llama 3.1 70B.
  - Gemini: https://aistudio.google.com/apikey — free tier 15 rpm on flash.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import time
from typing import Dict

import httpx

from app.config import settings
from app.core.logging import log


# Per-provider cool-off: when a provider returns 429 with a Retry-After hint,
# we mark it as "skip until ts" so subsequent calls hit the next provider
# instead of re-burning the same rate limit.
_cooloff: Dict[str, float] = {}


def _in_cooloff(name: str) -> bool:
    until = _cooloff.get(name, 0.0)
    return time.time() < until


def _set_cooloff(name: str, seconds: float) -> None:
    _cooloff[name] = time.time() + max(1.0, seconds)
    log.info("llm_cooloff", provider=name, seconds=int(seconds))


def _parse_retry_after(resp: httpx.Response, default: float = 30.0) -> float:
    """Parse a Retry-After header (seconds or HTTP date)."""
    h = resp.headers.get("Retry-After")
    if not h:
        return default
    try:
        return float(h)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            dt = parsedate_to_datetime(h)
            return max(1.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return default


@dataclass
class LLMResult:
    text: str
    json: dict[str, Any] | None
    provider: str
    model: str


# ----------------------------------------------------------------------------
# Provider availability checks (cheap — no network)
# ----------------------------------------------------------------------------

def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def _ollama_available() -> bool:
    """Ollama daemon up AND at least one model pulled (otherwise /api/chat 404s)."""
    if _in_cooloff("ollama"):
        return False
    try:
        r = httpx.get(f"{_ollama_url()}/api/version", timeout=1.5)
        if r.status_code != 200:
            return False
        tags = httpx.get(f"{_ollama_url()}/api/tags", timeout=2.0)
        if tags.status_code != 200:
            return False
        models = (tags.json() or {}).get("models") or []
        return len(models) > 0
    except Exception:
        return False


def _ollama_pick_model() -> str:
    """Return the configured OLLAMA_MODEL if pulled, otherwise the first
    installed model. Falls back to the env default if probing fails."""
    configured = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    try:
        r = httpx.get(f"{_ollama_url()}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return configured
        models = [(m.get("name") or "") for m in (r.json() or {}).get("models") or []]
        # Exact match first
        if configured in models:
            return configured
        # Prefix match (e.g. configured "llama3.1" → "llama3.1:8b")
        for m in models:
            if m.startswith(configured.split(":")[0]):
                return m
        return models[0] if models else configured
    except Exception:
        return configured


def _groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY")) and not _in_cooloff("groq")


def _gemini_available() -> bool:
    return (bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            and not _in_cooloff("gemini"))


def _anthropic_available() -> bool:
    return bool(settings.anthropic_api_key) and not _in_cooloff("anthropic")


def available_providers() -> list[str]:
    """List of currently usable providers in preference order."""
    out: list[str] = []
    if _ollama_available():    out.append("ollama")
    if _groq_available():      out.append("groq")
    if _gemini_available():    out.append("gemini")
    if _anthropic_available(): out.append("anthropic")
    return out


# ----------------------------------------------------------------------------
# Per-provider calls
# ----------------------------------------------------------------------------

def _ollama_generate(system: str, user: str, *, max_tokens: int, temperature: float, expect_json: bool, tier: str = "fast") -> LLMResult | None:
    model = _ollama_pick_model()
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if expect_json:
            payload["format"] = "json"   # Ollama's structured-output mode
        # Local models on a laptop are SLOW (a 3B on an 8GB M3 generates ~10-15
        # tok/s, so a 900-token analyst note can take 60-120s). Ollama has no
        # rate limit, so we can afford a long timeout — it's only ever reached
        # as the last-resort fallback when every cloud provider is throttled.
        timeout = httpx.Timeout(connect=5.0, read=240.0, write=10.0, pool=5.0)
        r = httpx.post(f"{_ollama_url()}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "").strip()
    except Exception as e:
        log.warning("ollama_call_failed", err=str(e))
        return None
    return LLMResult(text=text, json=_safe_json(text) if expect_json else None,
                     provider="ollama", model=model)


GROQ_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_QUALITY_MODEL = "openai/gpt-oss-120b"


def _groq_generate(system: str, user: str, *, max_tokens: int, temperature: float, expect_json: bool, tier: str = "fast") -> LLMResult | None:
    """Groq via OpenAI-compatible API. Tries the user-configured model first;
    on 429 (rate or daily quota), automatically falls back to a smaller model
    with a more generous free-tier quota before giving up.

    Two tiers (Groq retired the Llama 3.x models in 2026):
        openai/gpt-oss-120b  ← `quality` tier (analyst narrative, reasoning)
        openai/gpt-oss-20b   ← `fast` tier workhorse (news classification, chat)
    Opinion synthesis asks for the larger model via tier="quality"; high-volume
    calls stay on the small one. If the chosen model 429s we fall back to the
    other so the request still completes.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    if tier == "quality":
        primary = os.environ.get("GROQ_QUALITY_MODEL", GROQ_QUALITY_MODEL)
    else:
        primary = os.environ.get("GROQ_MODEL", GROQ_FAST_MODEL)
    # Fallback chain: both gpt-oss models accept `response_format: json_object`.
    fallbacks = [m for m in (GROQ_QUALITY_MODEL, GROQ_FAST_MODEL) if m != primary]

    for model in [primary, *fallbacks]:
        try:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if expect_json:
                payload["response_format"] = {"type": "json_object"}
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            if r.status_code == 429:
                # Specific model exhausted (or TPM hit) — don't mark whole
                # Groq provider cool, just try the next model. Only cool off
                # the whole provider when EVERY fallback fails.
                log.warning("groq_model_429", model=model,
                            retry_after=_parse_retry_after(r, default=60))
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return LLMResult(text=text, json=_safe_json(text) if expect_json else None,
                             provider="groq", model=model)
        except httpx.HTTPStatusError as e:
            log.warning("groq_call_failed", model=model, status=e.response.status_code, err=str(e)[:200])
            continue
        except Exception as e:
            log.warning("groq_call_failed", model=model, err=str(e)[:200])
            continue
    # All Groq models failed — put the whole provider in a short cool-off so
    # the next call doesn't waste time re-trying them all.
    _set_cooloff("groq", 60)
    return None


def _gemini_generate(system: str, user: str, *, max_tokens: int, temperature: float, expect_json: bool, tier: str = "fast") -> LLMResult | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    # Google retires model names regularly (1.5 and 2.0 Flash are gone); pin a
    # current stable one. See https://ai.google.dev/gemini-api/docs/models
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        # REST API — avoids pulling the google-generativeai SDK (~50 MB) just for this.
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if expect_json:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json=body,
            timeout=30.0,
        )
        if r.status_code == 429:
            # Gemini free tier doesn't always set Retry-After — default 60s.
            _set_cooloff("gemini", _parse_retry_after(r, default=60))
            return None
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        log.warning("gemini_call_failed", err=str(e))
        return None
    return LLMResult(text=text, json=_safe_json(text) if expect_json else None,
                     provider="gemini", model=model)


def _anthropic_generate(system: str, user: str, *, max_tokens: int, temperature: float, expect_json: bool, tier: str = "fast") -> LLMResult | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system + (
                "\n\nIMPORTANT: respond with ONLY a JSON object, no prose or code-fences." if expect_json else ""
            ),
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.warning("anthropic_call_failed", err=str(e))
        return None
    return LLMResult(text=text, json=_safe_json(text) if expect_json else None,
                     provider="anthropic", model=settings.llm_model)


def _safe_json(text: str) -> dict[str, Any] | None:
    """Parse JSON tolerantly — strip code-fences, find first { ... last }."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.split("\n", 1)[1] if "\n" in s else s
    # Try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: extract the largest brace-delimited block
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(s[first:last + 1])
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------------

_DISPATCH = {
    "ollama":    _ollama_generate,
    "groq":      _groq_generate,
    "gemini":    _gemini_generate,
    "anthropic": _anthropic_generate,
}

# Fast tier: cloud-free first for latency (Groq 8B answers in ~1-2s).
#
# Ollama is deliberately NOT in the default chain. On a small machine (e.g. an
# 8GB M3) a local 3B takes 30-90s per call AND its resident weights create
# memory pressure that slows everything else — when cloud is throttled and we
# auto-route high-volume calls (news classification across dozens of names) to
# it, the whole app bogs down. So by default, if cloud is throttled we let
# callers use their own cheap heuristic fallback instead. Opt back in with:
#   LLM_PREFER_LOCAL=1  → Ollama first (good on a big-GPU box, or offline/private)
if os.environ.get("LLM_PREFER_LOCAL") == "1":
    _ORDER_FAST = ("ollama", "groq", "gemini", "anthropic")
else:
    _ORDER_FAST = ("groq", "gemini", "anthropic")
# Quality tier: a small local model (the only kind that fits a typical 8-16GB
# laptop) is genuinely too weak for nuanced structured analysis — tested
# llama3.2:3b echoes the prompt schema instead of reasoning. So quality work
# uses the big cloud models ONLY; when all are throttled we serve the
# deterministic plan, which is better than a 3B's garbled narrative. Set
# LLM_QUALITY_ALLOW_OLLAMA=1 to opt back in if you run a 70B+ local model.
_ORDER_QUALITY = ("groq", "gemini", "anthropic")
if os.environ.get("LLM_QUALITY_ALLOW_OLLAMA") == "1":
    _ORDER_QUALITY = _ORDER_QUALITY + ("ollama",)


def generate(
    system: str,
    user: str,
    *,
    max_tokens: int = 500,
    temperature: float = 0.3,
    expect_json: bool = False,
    provider: str | None = None,
    tier: str = "fast",
) -> LLMResult | None:
    """Try providers in preference order. Returns None when none succeed.

    `tier`:
      - "fast"    : high-volume / low-stakes (news tagging, chat). Prefers the
                    local Ollama model when present (free, private, unlimited).
      - "quality" : analysis where nuance matters (opinion narrative + entry
                    review). Prefers Groq's 70B over a small local model.
    """
    chosen = provider or os.environ.get("LLM_PROVIDER", "auto")
    order: tuple[str, ...]
    if chosen and chosen != "auto":
        order = (chosen,)
    else:
        order = _ORDER_QUALITY if tier == "quality" else _ORDER_FAST

    for name in order:
        fn = _DISPATCH.get(name)
        if not fn:
            continue
        # Skip if obviously not configured
        if name == "ollama" and not _ollama_available():
            continue
        if name == "groq" and not _groq_available():
            continue
        if name == "gemini" and not _gemini_available():
            continue
        if name == "anthropic" and not _anthropic_available():
            continue
        result = fn(system, user, max_tokens=max_tokens, temperature=temperature,
                    expect_json=expect_json, tier=tier)
        if result and result.text:
            return result
    return None


def is_available() -> bool:
    return len(available_providers()) > 0


def cooloff_status() -> dict[str, float]:
    """Return remaining cool-off seconds per provider (0 if not in cool-off)."""
    now = time.time()
    return {k: max(0.0, v - now) for k, v in _cooloff.items()}


def clear_cooloff(name: str | None = None) -> None:
    """Force a provider out of cool-off (or all of them). Useful when the user
    has a paid tier upgrade or knows the quota has reset."""
    if name:
        _cooloff.pop(name, None)
    else:
        _cooloff.clear()
