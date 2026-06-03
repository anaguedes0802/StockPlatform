"""Narrative reconciliation — does a stock's underlying thesis still hold?

The opportunity engine scores names on their OWN price action and per-symbol
catalysts. But many names are really a bet on something else: MSTR / MARA /
COIN are bitcoin proxies, the gold miners are a bet on the gold price, the
E&P names are a bet on crude. A clean-looking technical setup on MSTR while
bitcoin is rolling over — or while the operator whose entire pitch is "we
never sell" is quietly selling the underlying — is a trap the chart can't see.

This runs like a small **agent**: given only a ticker, it
  1. IDENTIFIES the underlying thesis driver itself (LLM, reading the company
     profile + recent headlines) — no hand-maintained map required;
  2. calls deterministic TOOLS to check that driver's live state — the driver
     asset's price trend, and the name's current news; and
  3. RECONCILES the two into a `narrative_alignment` multiplier in roughly
     [0.20, 1.10] that the opportunity engine folds in alongside entry quality:

        final_score = raw_score × entry_quality × narrative_alignment

Caching keeps it cheap: the driver *identity* of a ticker is stable, so it's
cached for a week; only the live tool checks (price trend, conflict scan)
refresh often. When no LLM is configured the agent degrades to a small curated
fallback map so the feature still works offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import log
from app.services import llm
from app.services.opinion import _cache_get, _cache_set


# ---------------------------------------------------------------------------
# Driver-symbol resolution
# ---------------------------------------------------------------------------
# The LLM names a driver in plain language ("Bitcoin", "crude oil"); we resolve
# that to a reliable, fetchable price series. Curated so we never trade off a
# hallucinated ticker. Keyword → symbol; first containment match wins.
_DRIVER_CONCEPTS: list[tuple[tuple[str, ...], str, str]] = [
    # (match keywords, driver_symbol, canonical label)
    (("bitcoin", "btc"),                 "BTC-USD", "Bitcoin"),
    (("ethereum", "ether", "eth"),       "ETH-USD", "Ethereum"),
    (("solana",),                        "SOL-USD", "Solana"),
    (("gold", "bullion"),                "GLD",     "Gold"),
    (("silver",),                        "SLV",     "Silver"),
    (("crude", "oil", "wti", "brent", "petroleum"), "USO", "Crude oil"),
    (("natural gas", "natgas"),          "UNG",     "Natural gas"),
    (("copper",),                        "CPER",    "Copper"),
    (("uranium",),                       "URA",     "Uranium"),
    (("lithium",),                       "LIT",     "Lithium"),
    (("semiconductor", "chips", "chip demand"), "SOXX", "Semiconductors"),
    (("10-year", "treasury yield", "interest rate", "rates"), "^TNX", "Treasury yields"),
    (("s&p", "broad market", "equity market"), "SPY", "Broad market"),
]

# A narrative conflict is the COMPANY undermining its own thesis — selling /
# offloading / abandoning the very asset it's a proxy for, or halting the
# strategy. We detect it deterministically: a headline that names the driver
# AND a reversal action. This generalizes to any driver (a bitcoin treasury
# selling BTC, a gold miner unwinding hedges, a uranium pure-play exiting) and
# stays precise — a false "narrative conflict" is worse than a missed one.
#
# NOTE: driver *price/sentiment* weakness ("BTC slumps", "Citi bearish on
# bitcoin") is deliberately NOT a conflict here — that's already captured by
# driver-health, and folding it in twice would double-count the same signal.
_REVERSAL_ACTION = re.compile(
    r"\b(sells?|sold|selling|sale\s+of|offload\w*|dump\w*|unload\w*|trim\w*|"
    r"reduc\w*|abandon\w*|exit\w*|halt\w*|pause\w*|liquidat\w*|wind\s+down|"
    r"divest\w*|writes?\s+down|write-down)\b",
    re.IGNORECASE,
)

# Offline fallback driver map (only consulted when no LLM is configured).
# symbol → (driver_symbol, label, strength). These are pure proxies (strength 1).
_FALLBACK_DRIVERS: dict[str, tuple[str, str, float]] = {
    "MSTR": ("BTC-USD", "Bitcoin", 1.0), "STRK": ("BTC-USD", "Bitcoin", 1.0),
    "MARA": ("BTC-USD", "Bitcoin", 1.0), "RIOT": ("BTC-USD", "Bitcoin", 1.0),
    "CLSK": ("BTC-USD", "Bitcoin", 1.0), "COIN": ("BTC-USD", "Bitcoin", 0.9),
    "HUT": ("BTC-USD", "Bitcoin", 1.0), "BITF": ("BTC-USD", "Bitcoin", 1.0),
    "CIFR": ("BTC-USD", "Bitcoin", 1.0), "WULF": ("BTC-USD", "Bitcoin", 1.0),
    "BTBT": ("BTC-USD", "Bitcoin", 1.0), "HIVE": ("BTC-USD", "Bitcoin", 1.0),
    "SMLR": ("BTC-USD", "Bitcoin", 0.9), "CEP": ("BTC-USD", "Bitcoin", 1.0),
    "BMNR": ("ETH-USD", "Ethereum", 1.0), "BTCS": ("ETH-USD", "Ethereum", 1.0),
    "NEM": ("GLD", "Gold", 0.9), "GOLD": ("GLD", "Gold", 0.9),
    "AEM": ("GLD", "Gold", 0.9), "KGC": ("GLD", "Gold", 0.9),
    "AU": ("GLD", "Gold", 0.9), "FNV": ("GLD", "Gold", 0.85), "WPM": ("GLD", "Gold", 0.85),
    "OXY": ("USO", "Crude oil", 0.8), "DVN": ("USO", "Crude oil", 0.85),
    "APA": ("USO", "Crude oil", 0.85), "MRO": ("USO", "Crude oil", 0.85),
    "FANG": ("USO", "Crude oil", 0.8), "CTRA": ("USO", "Crude oil", 0.8),
}


@dataclass
class DriverProfile:
    """The thesis driver the agent identified for a ticker (stable, cached)."""
    has_driver: bool = False
    driver_symbol: str | None = None
    driver_label: str | None = None
    relationship: str = "none"        # proxy | correlated | input_cost | none
    strength: float = 0.0             # 0..1 — how dominant the driver is to the thesis
    thesis: str = ""
    conflict_criteria: str = ""       # what would break the thesis (natural language)
    source: str = "none"             # llm | fallback | none

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_driver": self.has_driver,
            "driver_symbol": self.driver_symbol,
            "driver_label": self.driver_label,
            "relationship": self.relationship,
            "strength": round(self.strength, 2),
            "thesis": self.thesis,
            "source": self.source,
        }


@dataclass
class NarrativeAssessment:
    symbol: str
    theme: str | None = None             # driver label, kept name `theme` for the UI/API
    driver_symbol: str | None = None
    driver_label: str | None = None
    relationship: str | None = None
    driver_health: float = 0.0           # [-1, +1]
    multiplier: float = 1.0
    thesis: str = ""
    notes: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    driver_trend: dict[str, Any] | None = None
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "driver_symbol": self.driver_symbol,
            "driver_label": self.driver_label,
            "relationship": self.relationship,
            "driver_health": round(self.driver_health, 3),
            "multiplier": round(self.multiplier, 3),
            "thesis": self.thesis,
            "notes": self.notes,
            "conflicts": self.conflicts,
            "driver_trend": self.driver_trend,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# TOOL 1 — driver price trend
# ---------------------------------------------------------------------------
_TREND_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TREND_TTL_S = 300.0


def driver_trend(driver_symbol: str) -> dict[str, Any] | None:
    """Trend health of a driver asset: 30d return + position vs its 50d average,
    folded into a `health` score in [-1, +1]. Cached, best-effort, never raises."""
    import time as _t
    now = _t.time()
    hit = _TREND_CACHE.get(driver_symbol)
    if hit and (now - hit[0]) <= _TREND_TTL_S:
        return hit[1]
    try:
        from app.services import market_data as md
        df = md.get_history(driver_symbol, interval="1d", range_="6mo")
        if df is None or df.empty or "close" not in df or len(df) < 25:
            return None
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        prior_22 = float(close.iloc[-22])
        ret_30 = (last / prior_22 - 1.0) if prior_22 > 0 else 0.0
        sma50 = float(close.tail(50).mean())
        above_sma50 = last > sma50
        # ±20% over a month saturates to ±1, plus a ±0.2 nudge for trend position.
        health = max(-1.0, min(1.0, ret_30 / 0.20))
        health += 0.20 if above_sma50 else -0.20
        health = max(-1.0, min(1.0, health))
        trend = {
            "symbol": driver_symbol,
            "last": round(last, 2),
            "ret_30d_pct": round(ret_30 * 100, 1),
            "above_sma50": above_sma50,
            "health": round(health, 3),
        }
        _TREND_CACHE[driver_symbol] = (now, trend)
        return trend
    except Exception as e:
        log.debug("driver_trend_failed", driver=driver_symbol, err=str(e))
        return None


_SYMBOL_OK: dict[str, bool] = {}


def _symbol_fetchable(symbol: str) -> bool:
    """Validate that a driver symbol actually returns price data — guards
    against the LLM naming a series we can't pull."""
    if symbol in _SYMBOL_OK:
        return _SYMBOL_OK[symbol]
    ok = False
    try:
        from app.services import market_data as md
        df = md.get_history(symbol, interval="1d", range_="1mo")
        ok = df is not None and not df.empty and len(df) >= 5
    except Exception:
        ok = False
    _SYMBOL_OK[symbol] = ok
    return ok


def _resolve_driver(label: str, llm_symbol: str | None) -> tuple[str | None, str | None]:
    """Resolve a driver to a (fetchable_symbol, canonical_label). Prefer the
    LLM's symbol if it validates; else map the label/symbol text to a curated
    reliable series; else give up (None)."""
    text = f"{label or ''} {llm_symbol or ''}".lower()
    for keys, sym, canon in _DRIVER_CONCEPTS:
        if any(k in text for k in keys):
            if _symbol_fetchable(sym):
                return sym, canon
    if llm_symbol and _symbol_fetchable(llm_symbol):
        return llm_symbol.upper(), (label or llm_symbol).strip()
    return None, None


# ---------------------------------------------------------------------------
# REASONING — identify the driver (LLM, cached long; fallback offline)
# ---------------------------------------------------------------------------

def _fallback_identify(symbol: str) -> DriverProfile:
    info = _FALLBACK_DRIVERS.get(symbol.upper())
    if not info:
        return DriverProfile(source="fallback")
    driver_symbol, label, strength = info
    return DriverProfile(
        has_driver=True, driver_symbol=driver_symbol, driver_label=label,
        relationship="proxy", strength=strength,
        thesis=f"{symbol.upper()} trades as a {label} proxy.",
        conflict_criteria=f"the company selling its {label.lower()}, or {label.lower()} entering a sustained downtrend",
        source="fallback",
    )


def _identify_driver_llm(symbol: str, profile: dict[str, Any],
                         headlines: list[str]) -> DriverProfile | None:
    name = profile.get("name") or symbol
    sector = profile.get("sector") or "?"
    industry = profile.get("industry") or "?"
    desc = (profile.get("description") or "")[:700]
    heads = "\n".join(f"- {h}" for h in headlines[:8])
    system = (
        "You are an equity analyst. Given a company, identify the single "
        "EXTERNAL asset or macro variable whose price most dominates this "
        "stock's thesis — the thing the stock is really a bet on. Examples: a "
        "bitcoin-treasury company or crypto miner → Bitcoin; a gold miner → "
        "Gold; an E&P oil company → crude oil; a uranium miner → uranium. Most "
        "ordinary companies (a bank, a retailer, a diversified software firm) "
        "have NO single dominant external driver — say so honestly.\n\n"
        "relationship: 'proxy' (the stock is a leveraged bet on the asset), "
        "'correlated' (moves with it but diversified), 'input_cost' (the asset "
        "is a major cost, so it moves INVERSELY), or 'none'.\n"
        "strength: 0..1, how dominant the driver is to the thesis (proxy≈0.9-1.0, "
        "loose correlation≈0.3-0.5).\n"
        "driver_name: the asset in plain words (e.g. 'Bitcoin', 'crude oil').\n"
        "driver_symbol: a tradeable ticker for it if you know one (e.g. 'BTC-USD', "
        "'GLD', 'USO'), else null.\n"
        "conflict_criteria: one sentence describing what NEWS would break this "
        "thesis (e.g. 'the company selling its bitcoin holdings').\n\n"
        "Output ONLY JSON: {\"has_driver\": bool, \"driver_name\": str|null, "
        "\"driver_symbol\": str|null, \"relationship\": str, \"strength\": "
        "float, \"thesis\": str (<=20 words), \"conflict_criteria\": str}."
    )
    user = (
        f"Company: {name} ({symbol.upper()})\nSector: {sector}\nIndustry: {industry}\n"
        f"Business: {desc}\nRecent headlines:\n{heads or '(none)'}"
    )
    try:
        res = llm.generate(system, user, max_tokens=400, temperature=0.1,
                           expect_json=True, tier="quality")
    except Exception as e:
        log.warning("narrative_identify_llm_failed", symbol=symbol, err=str(e))
        return None
    if not res or not res.json:
        return None
    j = res.json
    if not bool(j.get("has_driver")):
        return DriverProfile(has_driver=False, relationship="none", source="llm",
                             thesis=str(j.get("thesis", ""))[:160])
    label_raw = str(j.get("driver_name") or "").strip()
    sym_raw = str(j.get("driver_symbol") or "").strip() or None
    driver_symbol, label = _resolve_driver(label_raw, sym_raw)
    rel = str(j.get("relationship", "correlated")).strip().lower()
    if rel not in ("proxy", "correlated", "input_cost", "none"):
        rel = "correlated"
    try:
        strength = max(0.0, min(1.0, float(j.get("strength", 0.5))))
    except (TypeError, ValueError):
        strength = 0.5
    if not driver_symbol or rel == "none":
        return DriverProfile(has_driver=False, relationship="none", source="llm",
                             thesis=str(j.get("thesis", ""))[:160])
    return DriverProfile(
        has_driver=True, driver_symbol=driver_symbol, driver_label=label,
        relationship=rel, strength=strength,
        thesis=str(j.get("thesis", ""))[:160],
        conflict_criteria=str(j.get("conflict_criteria", ""))[:200],
        source="llm",
    )


def identify_driver(symbol: str, articles: list[dict[str, Any]] | None = None) -> DriverProfile:
    """Determine a ticker's dominant external thesis driver. LLM-driven and
    cached for a week (a company's driver is stable); offline fallback to the
    curated map when no LLM is configured."""
    sym = (symbol or "").upper()
    cache_key = f"narrative:driver:v2:{sym}"
    cached = _cache_get(cache_key)
    if cached is not None:
        dp = DriverProfile(**cached)
        return dp

    if not llm.is_available():
        dp = _fallback_identify(sym)
        # Don't cache the offline fallback long — an LLM may come online later.
        _cache_set(cache_key, dp.__dict__, ttl=3600)
        return dp

    try:
        from app.services import market_data as md
        profile = md.get_profile(sym) or {}
    except Exception:
        profile = {}
    heads = [a.get("title") or "" for a in (articles or []) if a.get("title")]
    dp = _identify_driver_llm(sym, profile, heads)
    if dp is None:
        dp = _fallback_identify(sym)
        _cache_set(cache_key, dp.__dict__, ttl=3600)
        return dp
    _cache_set(cache_key, dp.__dict__, ttl=7 * 24 * 3600)
    return dp


# ---------------------------------------------------------------------------
# TOOL 2 — conflict detection over current news
# ---------------------------------------------------------------------------

def _driver_keywords(dp: DriverProfile) -> tuple[str, ...]:
    """The headline tokens that identify the driver (e.g. ('bitcoin','btc'))."""
    for keys, sym, _canon in _DRIVER_CONCEPTS:
        if dp.driver_symbol == sym:
            return keys
    text = f"{dp.driver_label or ''}".lower()
    for keys, _sym, _canon in _DRIVER_CONCEPTS:
        if any(k in text for k in keys):
            return keys
    # fall back to distinctive words from the label itself
    return tuple(w for w in re.findall(r"[a-z]+", text) if len(w) >= 4)


# Editorial boilerplate that contains a reversal verb but is NOT a thesis
# event — "Buy, Sell or Hold" framing, listicles, etc.
_CONFLICT_BOILERPLATE = (
    "buy, sell or hold", "buy or sell", "sell or hold", "to buy or sell",
    "buy/sell", "buy, sell, or hold", "best stock", "stocks to buy",
)
# Max char gap between the driver word and the reversal verb for them to count
# as the same claim (~a few words). "sold 39 bitcoin" → gap ~7; "Buy, Sell or
# Hold ... Stock" has no driver word near the verb → rejected.
_CONFLICT_PROXIMITY = 28


def _detect_conflicts(dp: DriverProfile,
                      articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic, high-precision: a headline is a narrative conflict only if
    its TITLE names the DRIVER and a reversal ACTION *next to each other* — the
    company selling / offloading / abandoning the very thing the stock is a bet
    on. Title-only + proximity + a boilerplate blocklist keep out price/
    sentiment noise and "Buy, Sell or Hold" framing. A false haircut is worse
    than a missed one, so this errs toward precision."""
    keys = _driver_keywords(dp)
    if not keys or not articles:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or title in seen:
            continue
        t = title.lower()
        if any(b in t for b in _CONFLICT_BOILERPLATE):
            continue
        kw_pos = [t.find(k) for k in keys if k in t]
        if not kw_pos:
            continue
        m = _REVERSAL_ACTION.search(t)
        if not m:
            continue
        if min(abs(m.start() - p) for p in kw_pos) > _CONFLICT_PROXIMITY:
            continue
        seen.add(title)
        out.append({"title": title, "url": a.get("url"),
                    "published_at": a.get("published_at"),
                    "matched": m.group(0).strip().lower()})
    return out[:5]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _health_to_multiplier(health: float) -> float:
    """Map driver health [-1,+1] to a multiplier for a BULLISH opportunity.

      health=+1 → 1.10 (confirming, small boost); 0 → 1.00; -1 → 0.45 (broken).
    Asymmetric: a confirming driver is mild corroboration, a contradicting one
    is a strong reason to distrust a bullish chart."""
    if health >= 0:
        return 1.0 + 0.10 * health
    return 1.0 + 0.55 * health


def assess(symbol: str, articles: list[dict[str, Any]] | None = None) -> NarrativeAssessment:
    """Agentic reconciliation: identify the thesis driver, check it against live
    price + news, and return a multiplier the opportunity engine applies on top
    of entry quality. Names with no dominant driver get a neutral 1.0."""
    sym = (symbol or "").upper()
    a = NarrativeAssessment(symbol=sym)

    dp = identify_driver(sym, articles)
    a.source = dp.source
    if not dp.has_driver or not dp.driver_symbol:
        return a  # no opinion — neutral 1.0

    a.theme = dp.driver_label
    a.driver_symbol = dp.driver_symbol
    a.driver_label = dp.driver_label
    a.relationship = dp.relationship
    a.thesis = dp.thesis

    mult = 1.0
    # For an input-cost driver the relationship is inverse: a FALLING input
    # cost is bullish, so flip the health sign before scoring.
    sign = -1.0 if dp.relationship == "input_cost" else 1.0

    # --- TOOL: driver price trend ---
    trend = driver_trend(dp.driver_symbol)
    if trend is not None:
        a.driver_trend = trend
        eff_health = max(-1.0, min(1.0, sign * trend["health"] * dp.strength))
        a.driver_health = eff_health
        mult *= _health_to_multiplier(eff_health)
        arrow = "↑" if trend["ret_30d_pct"] >= 0 else "↓"
        pos = "above" if trend["above_sma50"] else "below"
        if eff_health <= -0.4:
            a.notes.append(
                f"{dp.driver_label} weak: {arrow}{abs(trend['ret_30d_pct'])}% 30d, "
                f"{pos} 50d — driver contradicts thesis"
            )
        elif eff_health >= 0.4:
            a.notes.append(
                f"{dp.driver_label} strong: {arrow}{abs(trend['ret_30d_pct'])}% 30d — driver confirms thesis"
            )
        else:
            a.notes.append(f"{dp.driver_label} flat ({arrow}{abs(trend['ret_30d_pct'])}% 30d) — neutral driver")

    # --- TOOL: conflict scan over current news ---
    if articles is None:
        try:
            from app.services.news import fetch_news
            articles = fetch_news(sym, limit=20)
        except Exception:
            articles = []
    conflicts = _detect_conflicts(dp, articles or [])
    if conflicts:
        a.conflicts = conflicts
        haircut = 1.0 - 0.5 * dp.strength      # strong proxy → ×0.5; loose → milder
        mult *= haircut
        lead = conflicts[0]
        a.notes.append(
            f"Narrative conflict: \"{lead['matched']}\" — {len(conflicts)} headline(s) "
            f"contradict the {(dp.driver_label or 'thesis').lower()} thesis (×{haircut:.2f})"
        )

    a.multiplier = round(max(0.20, min(1.10, mult)), 3)
    return a


def alignment_multiplier(symbol: str, articles: list[dict[str, Any]] | None = None) -> float:
    return assess(symbol, articles).multiplier
