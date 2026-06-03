"""LLM-driven news intelligence — classifies each article into a catalyst
taxonomy with materiality and directional bias.

Per-article taxonomy (drawn from sell-side desk classifications, e.g. JPM /
GS Daily Equity Pre-Market notes):

  CATALYST CATEGORIES
    earnings_beat / earnings_miss   — quarterly results vs consensus
    guidance_raise / guidance_cut   — forward outlook change
    fda_approval / fda_rejection    — biotech regulatory
    contract_win / contract_loss    — material commercial deal
    partnership                     — JV, strategic alliance, OEM deal
    acquisition_target / acquirer  — M&A
    insider_buy_cluster             — meaningful insider activity
    sec_filing                      — 8-K, S-1, material disclosure
    analyst_upgrade / downgrade     — sell-side rating change
    legal_litigation                — material lawsuits, settlements
    product_launch                  — new product / service rollout
    macro                           — Fed, rates, geopolitics affecting name
    technical                       — chart pattern coverage (low signal)
    noise                           — clickbait, recap, no edge

  MATERIALITY (0-1)
    0.0–0.3 = noise / minor                 → ignore in screener boosts
    0.3–0.6 = relevant context              → modest weight
    0.6–1.0 = high-impact, price-moving     → main signal

  DIRECTION
    bullish | bearish | neutral

The LLM gets a BATCH of headlines (10-20 at a time) so each user click ≤ 1
LLM call. Results are cached per article-hash for 24h since news doesn't
change retroactively.

CROSS-ENTITY ROUTING (see bottom of file)
  A headline like "Nvidia invests $2B in Marvell" is a catalyst for the
  TARGET (MRVL), not the well-known actor (NVDA). `extract_cross_entity_
  catalysts` finds (actor, relation, target) triples and routes them to the
  target ticker — for ANY heavyweight actor, not just Nvidia. It runs a
  precise regex pass plus (when available) an LLM pass that catches deals the
  regex misses (paraphrase, buried phrasing), resolving company names to
  tickers and de-duplicating the two.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Any

from app.core.logging import log
from app.services import llm
from app.services.opinion import _cache_get, _cache_set  # reuse the in-mem+Redis cache


CATEGORIES = [
    "earnings_beat", "earnings_miss",
    "guidance_raise", "guidance_cut",
    "fda_approval", "fda_rejection",
    "contract_win", "contract_loss",
    "partnership", "acquisition_target", "acquirer",
    # High-signal "smart-money validation" catalysts. These are the kind of
    # headlines that move a stock double-digits in a single session — an
    # influential operator publicly vouching for the name, a strategic giant
    # taking an equity stake, landing a marquee customer, joining a major
    # index (forced index-fund buying), or a large capacity buildout.
    "executive_endorsement",   # e.g. a peer/customer CEO calls it "the next $1T company"
    "strategic_investment",    # e.g. NVDA takes a $2B stake in the name
    "major_customer_win",      # marquee design win / anchor customer
    "index_inclusion",         # added to S&P 500 / Nasdaq-100 → mechanical buying
    "capacity_expansion",      # new fab / data-center / gigafactory buildout
    "insider_buy_cluster",
    "sec_filing",
    "analyst_upgrade", "analyst_downgrade",
    "legal_litigation",
    "product_launch",
    "macro",
    "technical",
    "noise",
]

# How well-substantiated the claim is. A double-digit move on a *confirmed*
# strategic investment is a real catalyst; the same headline tagged "rumor"
# ("sources say", "reportedly in talks") is mostly noise that often round-trips.
# Multiplier applied to materiality so the screener discounts speculation.
SUBSTANTIATION = ["rumor", "reported", "confirmed", "official"]
_SUBSTANTIATION_MULT = {
    "rumor":     0.55,   # unverified speculation — heavily discount
    "reported":  0.85,   # a credible outlet reports it, not yet official
    "confirmed": 1.00,   # company / counterparty confirmed
    "official":  1.05,   # filing / press release / on-stage statement (capped at 1.0)
}

# Materiality floors per category — used as a fallback when LLM omits the
# field or returns garbage. Captures the "obviously material" categories.
_MATERIALITY_FLOORS = {
    "earnings_beat":         0.75,
    "earnings_miss":         0.75,
    "guidance_raise":        0.80,
    "guidance_cut":          0.85,
    "fda_approval":          0.90,
    "fda_rejection":         0.90,
    "acquisition_target":    0.90,
    "acquirer":              0.70,
    "executive_endorsement": 0.70,
    "strategic_investment":  0.80,
    "major_customer_win":    0.65,
    "index_inclusion":       0.75,
    "capacity_expansion":    0.55,
    "contract_win":          0.60,
    "contract_loss":         0.60,
    "partnership":           0.50,
    "insider_buy_cluster":   0.65,
    "analyst_upgrade":       0.55,
    "analyst_downgrade":     0.55,
    "legal_litigation":      0.50,
    "product_launch":        0.45,
    "sec_filing":            0.40,
    "macro":                 0.35,
    "technical":             0.20,
    "noise":                 0.10,
}

# Default directional bias when LLM omits it — most categories have a natural sign.
_DEFAULT_DIRECTION = {
    "earnings_beat":      "bullish",
    "earnings_miss":      "bearish",
    "guidance_raise":     "bullish",
    "guidance_cut":       "bearish",
    "fda_approval":       "bullish",
    "fda_rejection":      "bearish",
    "contract_win":       "bullish",
    "contract_loss":      "bearish",
    "executive_endorsement": "bullish",
    "strategic_investment":  "bullish",
    "major_customer_win":    "bullish",
    "index_inclusion":       "bullish",
    "capacity_expansion":    "bullish",
    "partnership":        "bullish",
    "acquisition_target": "bullish",   # premium typically paid
    "acquirer":           "neutral",   # ambiguous (premium paid out)
    "insider_buy_cluster":"bullish",
    "analyst_upgrade":    "bullish",
    "analyst_downgrade":  "bearish",
    "legal_litigation":   "bearish",
    "product_launch":     "bullish",
    "sec_filing":         "neutral",
    "macro":              "neutral",
    "technical":          "neutral",
    "noise":              "neutral",
}


def _article_key(symbol: str, article: dict[str, Any]) -> str:
    """Stable key for cache lookup. Title + URL is unique enough."""
    raw = f"{symbol}|{article.get('title','')}|{article.get('url','')}"
    return f"news_intel:v1:{hashlib.sha1(raw.encode()).hexdigest()[:24]}"


def _infer_substantiation(text: str) -> str:
    """Cheap rumor-vs-fact read from the language. Speculative hedging words
    ("reportedly", "sources say", "in talks") mark a rumor; hard confirmation
    verbs and filing/press-release language mark something official."""
    def has(*terms: str) -> bool:
        return any(t in text for t in terms)
    if has("rumor", "rumour", "speculation", "could be", "may be", "reportedly in talks",
           "weighing", "exploring a", "is said to be considering"):
        return "rumor"
    if has("announced", "confirmed", "official", "press release", "8-k", "files with the sec",
           "on stage", "at the keynote", "said at", "told investors", "in a statement"):
        return "official" if has("press release", "8-k", "files with the sec", "official") else "confirmed"
    if has("reportedly", "sources say", "people familiar", "according to people", "rumored to"):
        return "reported"
    return "confirmed"


def _heuristic_fallback(article: dict[str, Any]) -> dict[str, Any]:
    """Keyword-based fallback when the LLM isn't available. Catches the most
    obvious categories so the screener still gets a useful signal."""
    title = (article.get("title") or "").lower()
    summary = (article.get("summary") or "")[:300].lower()
    text = f"{title} {summary}"
    def has(*terms: str) -> bool:
        return any(t in text for t in terms)

    cat = "noise"
    if has("beats earnings", "earnings beat", "tops estimates", "tops eps", "tops revenue"):
        cat = "earnings_beat"
    elif has("misses earnings", "misses estimates", "earnings miss", "below estimates"):
        cat = "earnings_miss"
    elif has("raises guidance", "lifts guidance", "raises outlook", "raises forecast"):
        cat = "guidance_raise"
    elif has("cuts guidance", "lowers guidance", "lowers outlook", "warns on"):
        cat = "guidance_cut"
    elif has("fda approval", "fda approves", "fda clears", "ce mark", "approved by the fda"):
        cat = "fda_approval"
    elif has("fda rejects", "complete response letter", "crl"):
        cat = "fda_rejection"
    elif has("trillion-dollar company", "next trillion", "calls it the next", "praises",
             "endorses", "ceo backs", "biggest fan", "champions"):
        cat = "executive_endorsement"
    elif has("takes a stake", "takes stake", "equity stake", "strategic investment",
             "invests $", "billion investment", "invests in", "leads investment round"):
        cat = "strategic_investment"
    elif has("design win", "anchor customer", "marquee customer", "lands as a customer",
             "signs as customer", "wins as a customer", "secures order from"):
        cat = "major_customer_win"
    elif has("join the s&p 500", "added to the s&p 500", "added to s&p", "index inclusion",
             "joins the nasdaq-100", "added to nasdaq-100", "set to join the s&p"):
        cat = "index_inclusion"
    elif has("new fab", "new plant", "expands capacity", "data center buildout",
             "datacenter buildout", "gigafactory", "capacity expansion", "new factory"):
        cat = "capacity_expansion"
    elif has("acquires", "acquisition", "to be acquired", "buyout offer", "takeover bid"):
        cat = "acquisition_target" if has("to be acquired", "acquired by") else "acquirer"
    elif has("partnership", "joint venture", "strategic alliance"):
        cat = "partnership"
    elif has("contract win", "wins contract", "awarded contract"):
        cat = "contract_win"
    elif has("upgraded", "raises price target", "buy rating", "outperform"):
        cat = "analyst_upgrade"
    elif has("downgraded", "lowers price target", "sell rating", "underperform"):
        cat = "analyst_downgrade"
    elif has("lawsuit", "sec investigation", "doj probe", "settlement"):
        cat = "legal_litigation"
    elif has("launches", "unveils", "introduces", "rolls out"):
        cat = "product_launch"
    elif has("fed rate", "fomc", "tariff", "trade war"):
        cat = "macro"

    return {
        "category": cat,
        "materiality": _MATERIALITY_FLOORS.get(cat, 0.2),
        "direction": _DEFAULT_DIRECTION.get(cat, "neutral"),
        "substantiation": _infer_substantiation(text),
        "rationale": "heuristic_fallback",
        "_source": "heuristic",
    }


def _classify_batch_llm(symbol: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Send a batch of headlines to the LLM, get structured classifications."""
    items = [
        {"i": i, "title": (a.get("title") or "")[:200], "summary": (a.get("summary") or "")[:300]}
        for i, a in enumerate(batch)
    ]
    system = (
        f"You are a sell-side equity research news classifier. For each "
        f"article about {symbol.upper()}, output a structured JSON object. "
        f"Be ruthless about classifying clickbait / recap / 'X stocks to "
        f"watch' as 'noise'. Be generous to material events.\n\n"
        f"Categories (pick one): {', '.join(CATEGORIES)}\n"
        f"  - executive_endorsement: an influential operator (esp. a peer or "
        f"customer CEO) publicly vouches for the company, e.g. calling it 'the "
        f"next trillion-dollar company' on stage.\n"
        f"  - strategic_investment: a major strategic/corporate investor takes "
        f"or raises an equity stake (NOT a passive fund 13F).\n"
        f"  - major_customer_win: lands a marquee/anchor customer or design win.\n"
        f"  - index_inclusion: added to S&P 500 / Nasdaq-100 (forced buying).\n"
        f"Materiality: 0.0 (noise) to 1.0 (highly price-moving)\n"
        f"Direction: bullish | bearish | neutral\n"
        f"Substantiation — how verified the claim is (this matters a lot: an "
        f"unconfirmed rumor that round-trips is not a real catalyst):\n"
        f"  rumor | reported | confirmed | official\n\n"
        f"Output ONLY this JSON: {{\"results\": [{{\"i\": int, "
        f"\"category\": str, \"materiality\": float, \"direction\": str, "
        f"\"substantiation\": str, \"rationale\": str (max 12 words)}}, ...]}}"
    )
    user = json.dumps({"articles": items})
    result = llm.generate(system, user, max_tokens=1200, temperature=0.1, expect_json=True)
    if not result or not result.json:
        return None
    rows = result.json.get("results") or []
    by_i: dict[int, dict[str, Any]] = {int(r.get("i", -1)): r for r in rows if "i" in r}
    out: list[dict[str, Any]] = []
    for i in range(len(batch)):
        r = by_i.get(i) or {}
        cat = str(r.get("category", "noise")).strip().lower().replace(" ", "_")
        if cat not in CATEGORIES:
            cat = "noise"
        subst = str(r.get("substantiation", "confirmed")).strip().lower()
        if subst not in SUBSTANTIATION:
            subst = "confirmed"
        out.append({
            "category": cat,
            "materiality": float(r.get("materiality") or _MATERIALITY_FLOORS.get(cat, 0.2)),
            "direction": str(r.get("direction", _DEFAULT_DIRECTION.get(cat, "neutral"))).lower(),
            "substantiation": subst,
            "rationale": str(r.get("rationale", "")).strip()[:120],
            "_source": result.provider,
        })
    return out


def classify_articles(symbol: str, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each article with category/materiality/direction. Returns
    the input list with each dict augmented (mutates copies, not originals)."""
    if not articles:
        return []
    annotated: list[dict[str, Any]] = []
    to_classify: list[tuple[int, dict[str, Any], str]] = []   # (idx, article, cache_key)

    for i, art in enumerate(articles):
        key = _article_key(symbol, art)
        cached = _cache_get(key)
        new_art = dict(art)
        if cached:
            new_art["intel"] = cached
            annotated.append(new_art)
        else:
            annotated.append(new_art)         # placeholder, will fill below
            to_classify.append((i, art, key))

    if to_classify:
        # Try LLM batch; fall back to heuristics on failure.
        batch_arts = [t[1] for t in to_classify]
        llm_out = None
        if llm.is_available():
            try:
                llm_out = _classify_batch_llm(symbol, batch_arts)
            except Exception as e:
                log.warning("news_intel_llm_failed", err=str(e))
        if llm_out is None:
            llm_out = [_heuristic_fallback(a) for a in batch_arts]

        for (idx, _art, key), intel in zip(to_classify, llm_out):
            annotated[idx]["intel"] = intel
            _cache_set(key, intel, ttl=24 * 3600)   # news doesn't change retroactively

    return annotated


def catalyst_score(annotated_articles: list[dict[str, Any]], days_window: int = 14) -> dict[str, Any]:
    """Aggregate a per-symbol catalyst score from classified articles.

    Returns:
      - score: float in [-1, +1] — directional weighted by materiality
      - max_materiality: highest single-article materiality
      - top_categories: list of (category, count) sorted by frequency
      - lead_article: the highest-materiality bullish or bearish article
        (whatever has bigger |score| contribution), with its intel block

    A symbol with a single fda_approval = ~+0.9.
    A symbol with 10 noise articles = ~0.0.
    A symbol with earnings_beat + analyst_upgrade = ~+0.85.
    """
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_window)
    weighted_sum = 0.0
    weight_total = 0.0
    cats: dict[str, int] = {}
    max_mat = 0.0
    lead_art: dict | None = None
    lead_contribution = 0.0

    for a in annotated_articles:
        intel = a.get("intel") or {}
        if not intel:
            continue
        # Skip stale articles
        ts = a.get("published_at")
        if ts:
            try:
                if isinstance(ts, (int, float)):
                    art_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                else:
                    from email.utils import parsedate_to_datetime
                    try:
                        art_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    except ValueError:
                        art_dt = parsedate_to_datetime(str(ts))
                    if art_dt.tzinfo is None:
                        art_dt = art_dt.replace(tzinfo=timezone.utc)
                if art_dt < cutoff:
                    continue
            except Exception:
                pass

        cat = intel.get("category", "noise")
        mat = float(intel.get("materiality") or 0)
        # Discount the materiality by how substantiated the claim is: an
        # unconfirmed rumor is worth ~half a confirmed one, an on-record /
        # filed statement slightly more. Capped at 1.0.
        subst = intel.get("substantiation", "confirmed")
        mat = min(1.0, mat * _SUBSTANTIATION_MULT.get(subst, 1.0))
        direction = intel.get("direction", "neutral")
        sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)
        contribution = mat * sign

        weighted_sum += contribution
        weight_total += mat
        cats[cat] = cats.get(cat, 0) + 1
        if mat > max_mat:
            max_mat = mat
        if abs(contribution) > abs(lead_contribution):
            lead_contribution = contribution
            lead_art = a

    # Normalize: divide by max possible weight to stay in [-1, +1].
    score = weighted_sum / weight_total if weight_total > 0 else 0.0
    top_categories = sorted(cats.items(), key=lambda t: -t[1])
    return {
        "score": round(max(-1.0, min(1.0, score)), 3),
        "max_materiality": round(max_mat, 3),
        "n_articles_used": int(weight_total > 0) and sum(cats.values()),
        "top_categories": top_categories[:5],
        "lead_article": {
            "title": lead_art.get("title") if lead_art else None,
            "url": lead_art.get("url") if lead_art else None,
            "intel": lead_art.get("intel") if lead_art else None,
        } if lead_art else None,
    }


def _price_reaction(symbol: str) -> dict[str, Any] | None:
    """Today's realized move for the name. A large abnormal move is the market
    *confirming* a catalyst in real time — the single strongest evidence a
    headline is the real thing and not noise. Best-effort; never raises."""
    try:
        from app.services import market_data as md
        q = md.get_quote(symbol) or {}
        chg = q.get("change_pct")
        if chg is None:
            return None
        chg = float(chg)
        # |move| >= 8% in a session is a genuine catalyst-grade reaction for a
        # large/mid cap; scale a 0..1 confirmation strength off that threshold.
        confirm = max(0.0, min(1.0, (abs(chg) - 3.0) / 12.0))  # 3%→0, 15%→1
        return {
            "change_pct": round(chg, 2),
            "direction": "bullish" if chg > 0 else ("bearish" if chg < 0 else "neutral"),
            "confirmation_strength": round(confirm, 3),
        }
    except Exception:
        return None


def classify_and_score(symbol: str, articles: list[dict[str, Any]] | None = None,
                       days_window: int = 14, confirm_with_price: bool = True) -> dict[str, Any]:
    """High-level entry: fetch news (if not provided), classify, return score.

    When `confirm_with_price` is set, the symbol's realized session move is
    folded in: a catalyst whose direction matches a large same-day move gets
    its reported max_materiality boosted toward 1.0 (the tape is confirming
    it), while a 'catalyst' the tape ignored is flagged as unconfirmed.
    """
    if articles is None:
        from app.services.news import fetch_news
        try:
            articles = fetch_news(symbol, limit=20)
        except Exception:
            articles = []
    annotated = classify_articles(symbol, articles or [])
    scored = catalyst_score(annotated, days_window=days_window)

    reaction = _price_reaction(symbol) if confirm_with_price else None
    if reaction and reaction["confirmation_strength"] > 0:
        score_sign = 1.0 if scored["score"] > 0 else (-1.0 if scored["score"] < 0 else 0.0)
        react_sign = 1.0 if reaction["direction"] == "bullish" else (-1.0 if reaction["direction"] == "bearish" else 0.0)
        if score_sign != 0 and score_sign == react_sign:
            # Tape agrees with the catalyst — boost confidence toward 1.0.
            cs = reaction["confirmation_strength"]
            scored["max_materiality"] = round(min(1.0, scored["max_materiality"] + 0.3 * cs), 3)
            scored["price_confirmed"] = True
        else:
            scored["price_confirmed"] = False
    scored["price_reaction"] = reaction

    return {
        "symbol": symbol.upper(),
        "n_articles": len(annotated),
        **scored,
        "articles": [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "published_at": a.get("published_at"),
                "sentiment": a.get("sentiment"),
                "intel": a.get("intel"),
            }
            for a in annotated[:10]
        ],
    }


# -----------------------------------------------------------------------------
# Cross-entity catalyst routing
# -----------------------------------------------------------------------------
# A headline like "Nvidia invests $2B in Marvell" or "Microsoft signs cloud
# deal with CoreWeave" is a catalyst for the *target* (the one receiving the
# investment / deal / endorsement), not for the well-known actor that authored
# it. The naive per-symbol feed files such a story under the actor (NVDA) and
# misses the tradeable beneficiary. This module extracts (actor, relation,
# target) triples from any headline and routes the catalyst to the target
# symbol — for ANY heavyweight actor, not just Nvidia.

# Hand-curated aliases for the names people actually write in headlines, mapped
# to the tickers we track. Keeps the matcher precise (no fuzzy false-positives)
# and covers the mega-cap "actors" whose moves move other stocks. Extended
# automatically below with distinctive tokens from the live universe.
_CURATED_ALIASES: dict[str, str] = {
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "amazon": "AMZN", "amazon.com": "AMZN", "aws": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B", "buffett": "BRK-B",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "palantir": "PLTR",
    "amd": "AMD", "advanced micro devices": "AMD",
    # Common deal/investment targets in the AI/semi supply chain. Tickers are
    # real even if outside the quote universe — fine for the news feed.
    "marvell": "MRVL",
    "coreweave": "CRWV",
    "supermicro": "SMCI", "super micro": "SMCI",
    "micron": "MU",
    "qualcomm": "QCOM",
    "intel": "INTC",
    "arm": "ARM", "arm holdings": "ARM",
    "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "snowflake": "SNOW",
    "softbank": "SFTBY",
    "openai": "OPENAI",  # private — surfaces in feed, not tradeable
}

# Mega-cap actors whose involvement materially raises the signal — an
# investment/deal led by one of these is a bigger catalyst than the same from
# an unknown counterparty.
_MEGA_ACTORS = {"NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "ORCL", "TSLA"}

# Generic leading words we must NOT auto-index as a company alias (too many
# false positives). e.g. "Advanced Micro Devices" → we use the curated "amd".
_GENERIC_TOKENS = {
    "advanced", "american", "global", "general", "first", "united", "national",
    "international", "the", "new", "north", "south", "capital", "holdings",
    "group", "technologies", "technology", "industries", "international",
    "enterprise", "systems", "digital", "data", "energy", "financial",
}

_CORP_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|sa|ag|nv|"
    r"holdings|technologies|technology|platforms|group|class\s+[abc])\b\.?",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """alias (lowercased) -> ticker. Curated names + a distinctive leading
    token per tracked stock (skipping generic words). Built once."""
    idx = dict(_CURATED_ALIASES)
    try:
        from app.services import market_data as md
        for it in md.all_universe():
            sym = str(it.get("symbol") or "").upper()
            name = (it.get("name") or "")
            if not sym or (it.get("asset_class") or "stock") != "stock":
                continue
            cleaned = _CORP_SUFFIX_RE.sub("", name)
            cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", cleaned).strip().lower()
            if not cleaned:
                continue
            tokens = cleaned.split()
            lead = tokens[0]
            # index the distinctive leading token (e.g. "marvell", "coreweave")
            if len(lead) >= 4 and lead not in _GENERIC_TOKENS and lead not in idx:
                idx[lead] = sym
            # also index the full cleaned name if it's multi-word & distinctive
            if len(tokens) >= 2 and cleaned not in idx:
                idx[cleaned] = sym
    except Exception:
        pass
    return idx


@lru_cache(maxsize=1)
def _mention_re() -> "re.Pattern[str]":
    aliases = sorted(_alias_index().keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# relation phrase -> (category, is_acquisition). The TARGET (beneficiary) is
# routed as bullish in every case; the actor side is handled separately.
_RELATIONS: list[tuple[str, str]] = [
    (r"invests?(?:\s+\$?[\d.,]+\s*(?:billion|million|b|m)?)?\s+in", "strategic_investment"),
    (r"takes?\s+(?:a\s+)?(?:\$?[\d.,]+\s*(?:billion|million|b|m)?\s+)?stake\s+in", "strategic_investment"),
    (r"buys?\s+(?:a\s+)?stake\s+in", "strategic_investment"),
    (r"leads?\s+(?:a\s+)?(?:\$?[\d.,]+\s*(?:billion|million|b|m)?\s+)?(?:funding|investment)\s+(?:round\s+)?(?:in|for)", "strategic_investment"),
    (r"pours?\s+\$?[\d.,]+\s*(?:billion|million|b|m)?\s+into", "strategic_investment"),
    (r"backs", "executive_endorsement"),
    (r"to\s+acquire", "acquisition_target"),
    (r"acquires?", "acquisition_target"),
    (r"to\s+buy", "acquisition_target"),
    (r"agrees?\s+to\s+buy", "acquisition_target"),
    (r"partners?\s+with", "partnership"),
    (r"teams?\s+up\s+with", "partnership"),
    (r"(?:signs?|inks?)\s+(?:a\s+)?(?:[\w-]+\s+){0,4}?deal\s+with", "partnership"),
    (r"strikes?\s+(?:a\s+)?(?:[\w-]+\s+){0,4}?deal\s+with", "partnership"),
    (r"joint\s+venture\s+with", "partnership"),
    (r"to\s+supply", "major_customer_win"),
    (r"selects?", "major_customer_win"),
    (r"picks?", "major_customer_win"),
    (r"chooses?", "major_customer_win"),
    (r"names?\s+(?:the\s+)?next\s+trillion", "executive_endorsement"),
    (r"calls?\s+(?:\w+\s+){0,4}?the\s+next", "executive_endorsement"),
]


@lru_cache(maxsize=1)
def _relation_re() -> "re.Pattern[str]":
    pattern = "(" + "|".join(p for p, _ in _RELATIONS) + ")"
    return re.compile(pattern, re.IGNORECASE)


def _relation_category(matched: str) -> str:
    low = matched.lower()
    for pat, cat in _RELATIONS:
        if re.fullmatch(pat, low, re.IGNORECASE):
            return cat
    # fallback: first relation whose head verb appears
    for pat, cat in _RELATIONS:
        if re.match(pat, low, re.IGNORECASE):
            return cat
    return "partnership"


def _build_catalyst(actor: str, target: str, category: str, relation: str,
                    subst: str, art: dict[str, Any], source: str) -> dict[str, Any]:
    if category not in CATEGORIES:
        category = "partnership"
    if subst not in SUBSTANTIATION:
        subst = "confirmed"
    mat = _MATERIALITY_FLOORS.get(category, 0.5)
    if actor in _MEGA_ACTORS:
        mat = min(1.0, mat + 0.10)   # a heavyweight actor raises the stakes
    return {
        "target_symbol": target,      # the tradeable beneficiary
        "actor_symbol": actor,        # who made the move
        "relation": relation.strip().lower()[:80],
        "category": category,
        "direction": "bullish",       # the target is the beneficiary
        "materiality": round(min(1.0, mat * _SUBSTANTIATION_MULT.get(subst, 1.0)), 3),
        "substantiation": subst,
        "title": (art.get("title") or "").strip(),
        "url": art.get("url"),
        "published_at": art.get("published_at"),
        "_source": source,
    }


def _resolve_company(name: str) -> str | None:
    """Resolve a free-text company name (as an LLM or headline emits it) to a
    ticker. Tries the curated/universe alias index first (exact, fast), then a
    universe search as a fuzzy fallback. Returns None if not confidently
    resolvable — we'd rather drop a route than mis-route it."""
    if not name:
        return None
    raw = name.strip()
    if raw.isupper() and 1 <= len(raw) <= 5 and raw.isalpha():
        return raw  # already looks like a ticker
    idx = _alias_index()
    cleaned = _CORP_SUFFIX_RE.sub("", raw)
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", cleaned).strip().lower()
    if not cleaned:
        return None
    if cleaned in idx:
        return idx[cleaned]
    lead = cleaned.split()[0]
    if lead in idx:
        return idx[lead]
    # fuzzy fallback against the tracked universe
    try:
        from app.services import market_data as md
        hits = md.search_universe(cleaned, limit=1, asset_class="stock")
        if hits:
            hit = hits[0]
            hit_name = (hit.get("name") or "").lower()
            # accept only a reasonably tight match (leading token overlap)
            if lead and (lead in hit_name or hit_name.split()[:1] == [lead]):
                return str(hit["symbol"]).upper()
    except Exception:
        pass
    return None


def _extract_cross_entity_regex(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic pass: 'ACTOR <relation> TARGET' triples from headlines.
    Precise but limited to clean headline phrasings."""
    idx = _alias_index()
    mre = _mention_re()
    rre = _relation_re()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for art in articles or []:
        title = (art.get("title") or "").strip()
        if not title:
            continue
        mentions = [(m.start(), idx[m.group(1).lower()]) for m in mre.finditer(title)]
        if len(mentions) < 2:
            continue
        for rm in rre.finditer(title):
            vpos = rm.start()
            category = _relation_category(rm.group(0))
            actor = next((sym for pos, sym in reversed(mentions) if pos < vpos), None)
            target = next((sym for pos, sym in mentions if pos > vpos), None)
            if not actor or not target or actor == target:
                continue
            key = (actor, target, category)
            if key in seen:
                continue
            seen.add(key)
            subst = _infer_substantiation(title.lower())
            out.append(_build_catalyst(actor, target, category, rm.group(0), subst, art, "regex"))
    return out


def _extract_cross_entity_llm(articles: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """LLM pass: reads a batch of headlines (+summaries) and extracts deal
    relationships the regex can't — buried phrasing, unusual verbs, paraphrase.
    Returns None if the LLM is unavailable so the caller can fall back. Cached
    per batch-hash for 24h (news doesn't change retroactively)."""
    if not articles or not llm.is_available():
        return None
    items = [
        {"i": i, "title": (a.get("title") or "")[:200], "summary": (a.get("summary") or "")[:200]}
        for i, a in enumerate(articles) if (a.get("title") or "").strip()
    ]
    if not items:
        return []
    raw_key = "|".join(f"{it['i']}:{it['title']}" for it in items)
    cache_key = f"xentity:v1:{hashlib.sha1(raw_key.encode()).hexdigest()[:24]}"
    cached = _cache_get(cache_key)
    if cached is not None:
        rows = cached
    else:
        system = (
            "You extract corporate-action relationships from market headlines. "
            "For each headline where ONE company takes an action that benefits "
            "ANOTHER company, output the actor (who acts), the target (the "
            "beneficiary), and the relationship. Skip headlines about a single "
            "company, index/market recaps, or macro commentary.\n\n"
            "category (pick one): strategic_investment (stake/funding), "
            "acquisition_target (target being acquired), partnership (JV / deal "
            "/ alliance), major_customer_win (supply / design win / customer), "
            "executive_endorsement (a leader publicly vouches for the company).\n"
            "substantiation: rumor | reported | confirmed | official.\n"
            "Use the company's common name (e.g. 'Nvidia', 'Marvell').\n\n"
            "Output ONLY JSON: {\"results\": [{\"i\": int, \"actor\": str, "
            "\"target\": str, \"category\": str, \"substantiation\": str}, ...]}. "
            "Return an empty list if no headline qualifies."
        )
        user = json.dumps({"headlines": items})
        try:
            result = llm.generate(system, user, max_tokens=1200, temperature=0.1, expect_json=True)
        except Exception as e:
            log.warning("xentity_llm_failed", err=str(e))
            return None
        if not result or not result.json:
            return None
        rows = result.json.get("results") or []
        _cache_set(cache_key, rows, ttl=24 * 3600)

    out: list[dict[str, Any]] = []
    by_i = {int(a["i"]): articles[a["i"]] for a in items}
    for r in rows:
        try:
            i = int(r.get("i", -1))
        except (TypeError, ValueError):
            continue
        art = by_i.get(i)
        if art is None:
            continue
        actor = _resolve_company(str(r.get("actor", "")))
        target = _resolve_company(str(r.get("target", "")))
        if not actor or not target or actor == target:
            continue
        category = str(r.get("category", "partnership")).strip().lower().replace(" ", "_")
        subst = str(r.get("substantiation", "confirmed")).strip().lower()
        relation = f"{category.replace('_', ' ')}"
        out.append(_build_catalyst(actor, target, category, relation, subst, art, "llm"))
    return out


def extract_cross_entity_catalysts(articles: list[dict[str, Any]],
                                   use_llm: bool = True) -> list[dict[str, Any]]:
    """Route 'ACTOR <relation> TARGET' headlines to the TARGET ticker.

    Runs the precise regex pass always, and (when available) an LLM pass that
    catches deals the regex misses. Results are merged and de-duplicated by
    (target, actor, category), preferring the higher-materiality / LLM hit.
    Only emits when both actor and target resolve to tickers and differ —
    precise by design: a missed route beats a wrong one.
    """
    catalysts = _extract_cross_entity_regex(articles)
    if use_llm:
        llm_cats = _extract_cross_entity_llm(articles)
        if llm_cats:
            catalysts = catalysts + llm_cats

    # Dedupe by (target, actor, category) — keep the strongest signal.
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in catalysts:
        key = (c["target_symbol"], c["actor_symbol"], c["category"])
        cur = best.get(key)
        if cur is None or c["materiality"] > cur["materiality"]:
            best[key] = c
    out = list(best.values())
    out.sort(key=lambda c: -c["materiality"])
    return out


def market_cross_entity_catalysts(limit: int = 40) -> list[dict[str, Any]]:
    """Scan market-wide news for cross-entity catalysts and route them to the
    affected tickers. This is the 'Nvidia invested in X' feed — for every
    heavyweight, not just Nvidia."""
    from app.services.news import fetch_market_news
    try:
        articles = fetch_market_news(limit=limit)
    except Exception:
        articles = []
    return extract_cross_entity_catalysts(articles)


def cross_entity_for_symbol(symbol: str, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter routed catalysts down to those where `symbol` is the target —
    i.e. 'someone big just did something for THIS stock'."""
    sym = symbol.upper()
    return [c for c in extract_cross_entity_catalysts(articles) if c["target_symbol"] == sym]
