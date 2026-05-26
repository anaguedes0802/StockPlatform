"""Zero-shot news topic classification.

Goes beyond sentiment polarity: classify each article into one of a fixed
finance taxonomy. Different topics have different price-impact functions:
  - **earnings**         — most impactful (binary event)
  - **m_and_a**          — strong, can be persistent
  - **regulation**       — bearish for the affected name
  - **lawsuit**          — bearish, magnitude depends on materiality
  - **product**          — usually bullish (launches, partnerships)
  - **leadership**       — CEO/CFO change, often volatile
  - **macro**            — affects sector, not just one name
  - **analyst_action**   — upgrade/downgrade reports
  - **insider_activity** — Form 4 coverage
  - **other**            — catchall

Backend: HuggingFace `facebook/bart-large-mnli` via the zero-shot pipeline.
Lazy-loaded — only downloads (~1.6GB) and runs when topic enrichment is requested.

Performance:
  - cold load ~10-30s (CPU)
  - ~150-400ms per article (CPU)
  - per-symbol news enrichment: cached for 1h
"""
from __future__ import annotations

import json
import os
from typing import Any

import redis

from app.config import settings
from app.core.logging import log


TOPICS = [
    "earnings",
    "m_and_a",
    "regulation",
    "lawsuit",
    "product",
    "leadership_change",
    "macro",
    "analyst_action",
    "insider_activity",
    "other",
]

# Per-topic price-impact priors (used by opinion when aggregating).
#   sign = directional bias when the article tone is neutral/positive
#   magnitude = how much we expect price to move on news of this type
TOPIC_PRIORS: dict[str, dict[str, float]] = {
    "earnings":          {"sign": 1.0,  "magnitude": 0.05},
    "m_and_a":           {"sign": 1.0,  "magnitude": 0.08},
    "regulation":        {"sign": -1.0, "magnitude": 0.04},
    "lawsuit":           {"sign": -1.0, "magnitude": 0.03},
    "product":           {"sign": 1.0,  "magnitude": 0.02},
    "leadership_change": {"sign": 0.0,  "magnitude": 0.025},  # direction-uncertain
    "macro":             {"sign": 0.0,  "magnitude": 0.015},
    "analyst_action":    {"sign": 1.0,  "magnitude": 0.02},
    "insider_activity":  {"sign": 0.0,  "magnitude": 0.015},
    "other":             {"sign": 0.0,  "magnitude": 0.01},
}


_redis: redis.Redis | None = None


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _cache_get(key: str) -> Any | None:
    try:
        v = _cache().get(key)
        return json.loads(v) if v else None
    except Exception:
        return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


# ---------- lazy classifier load ----------

_CLASSIFIER: Any | None = None
_CLASSIFIER_FAILED: bool = False


def _try_load_classifier() -> Any | None:
    """Returns the zero-shot pipeline if available; else None. Lazy + once-only."""
    global _CLASSIFIER, _CLASSIFIER_FAILED
    if _CLASSIFIER is not None:
        return _CLASSIFIER
    if _CLASSIFIER_FAILED:
        return None
    if os.environ.get("USE_TOPIC_CLASSIFIER", "1") == "0":
        _CLASSIFIER_FAILED = True
        return None
    try:
        from transformers import pipeline  # type: ignore
        _CLASSIFIER = pipeline(
            task="zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU; switch to 0 for CUDA, "mps" for Apple Silicon
        )
        return _CLASSIFIER
    except Exception as e:
        log.warning("topic_classifier_load_failed", err=str(e))
        _CLASSIFIER_FAILED = True
        return None


def classify_article(text: str) -> dict[str, Any]:
    """Return {topic: str, confidence: float, distribution: {topic: prob}}.
    Returns 'other' / 0.0 if classifier is unavailable or text empty.
    """
    if not text or not text.strip():
        return {"topic": "other", "confidence": 0.0, "distribution": {}}
    pipe = _try_load_classifier()
    if pipe is None:
        return {"topic": "other", "confidence": 0.0, "distribution": {}}
    # Strip whitespace, cap length for performance
    text = text[:1024]
    try:
        result = pipe(text, candidate_labels=TOPICS, multi_label=False)
        labels: list[str] = result["labels"]
        scores: list[float] = result["scores"]
        distribution = {l: float(s) for l, s in zip(labels, scores, strict=True)}
        top = labels[0]
        return {"topic": top, "confidence": float(scores[0]), "distribution": distribution}
    except Exception as e:
        log.warning("topic_classify_failed", err=str(e))
        return {"topic": "other", "confidence": 0.0, "distribution": {}}


def enrich_articles(symbol: str, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add topic + confidence to each article. Cached per (symbol, title-hash) for 1h."""
    cache_key = f"news_topics:{symbol}:{hash(tuple((a.get('title') or '')[:100] for a in articles))}"
    if (cached := _cache_get(cache_key)) is not None:
        return cached
    pipe = _try_load_classifier()
    if pipe is None:
        for a in articles:
            a.setdefault("topic", "other")
            a.setdefault("topic_confidence", 0.0)
        return articles
    out = []
    for a in articles:
        text = " ".join(filter(None, [a.get("title"), a.get("summary")]))
        c = classify_article(text)
        out.append({**a, "topic": c["topic"], "topic_confidence": c["confidence"]})
    _cache_set(cache_key, out, ttl=3600)
    return out


def topic_signal(symbol: str, articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the recent news basket by topic distribution × tone.

    For each enriched article: contribution = sign(sentiment) × topic_magnitude × topic_confidence
    Sum across articles and normalize.
    """
    if not articles:
        return {"score": 0.0, "summary": "No articles to classify.", "topic_breakdown": {}}
    enriched = enrich_articles(symbol, articles)
    breakdown: dict[str, dict[str, Any]] = {}
    total_signed = 0.0
    n_used = 0
    for a in enriched:
        topic = a.get("topic", "other")
        prior = TOPIC_PRIORS.get(topic, TOPIC_PRIORS["other"])
        topic_conf = float(a.get("topic_confidence") or 0)
        sentiment = a.get("sentiment")
        if sentiment is None or topic_conf < 0.30:
            continue
        # Effective direction = topic's directional prior OR (if sign=0) the sentiment polarity
        direction = prior["sign"] if abs(prior["sign"]) > 0.01 else (1.0 if sentiment > 0 else -1.0)
        contribution = direction * prior["magnitude"] * topic_conf
        total_signed += contribution
        n_used += 1
        b = breakdown.setdefault(topic, {"n": 0, "signed_total": 0.0})
        b["n"] += 1
        b["signed_total"] += contribution
    import math
    score = float(max(-1.0, min(1.0, math.tanh(total_signed * 8.0))))
    summary = "; ".join(f"{t}: {b['n']}" for t, b in sorted(breakdown.items(), key=lambda kv: -kv[1]["n"])[:4])
    return {
        "score": round(score, 3),
        "n_articles_used": n_used,
        "topic_breakdown": {t: {"n": b["n"], "signed_total": round(b["signed_total"], 4)} for t, b in breakdown.items()},
        "summary": f"News topics: {summary}" if summary else "Unclassified.",
    }


def is_available() -> bool:
    return _try_load_classifier() is not None
