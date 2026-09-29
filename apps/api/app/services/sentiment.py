"""Sentiment scoring with two backends:

  - **finBERT** (ProsusAI/finbert via HuggingFace transformers) — finance-tuned
    BERT, ~440MB download, runs on CPU. Real per-text 3-class probabilities
    (positive/negative/neutral) converted to a [-1, +1] polarity score.
  - **Lexicon** — tiny built-in scorer used when transformers isn't installed.
    Fast, offline, but coarse.

Selection: finBERT is used when (a) transformers is installed, AND (b)
USE_FINBERT env var is unset or "1" (allows opting out). The first call loads
the model lazily (cold ~5s); subsequent calls are ~50-100ms per batch of 8.
"""
from __future__ import annotations

import os
import re
from typing import Any

_POS = {
    "beat", "beats", "growth", "growing", "rally", "rallies", "surge", "soars",
    "upgrade", "outperform", "strong", "record", "breakthrough", "wins", "expansion",
    "bullish", "buyback", "dividend", "innovation",
}
_NEG = {
    "miss", "missed", "downgrade", "weak", "decline", "drop", "drops", "plunge",
    "lawsuit", "investigation", "fraud", "bearish", "loss", "recession", "warning",
    "cuts", "layoffs", "fine", "fined", "scandal",
}

_WORD = re.compile(r"[A-Za-z']+")


# ---------- lexicon (always available) ----------

def _score_lexicon(text: str) -> tuple[float, float]:
    if not text:
        return 0.0, 0.0
    tokens = [w.lower() for w in _WORD.findall(text)]
    pos = sum(1 for t in tokens if t in _POS)
    neg = sum(1 for t in tokens if t in _NEG)
    hits = pos + neg
    if hits == 0:
        return 0.0, 0.05
    polarity = (pos - neg) / hits
    confidence = min(1.0, hits / 10.0)
    return float(polarity), float(confidence)


# ---------- finBERT (lazy-loaded) ----------

_FINBERT: Any | None = None
_FINBERT_FAILED: bool = False
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _try_load_finbert() -> Any | None:
    """Lazy-load ProsusAI/finbert once. Sets _FINBERT_FAILED on failure so we
    don't retry every call.
    """
    global _FINBERT, _FINBERT_FAILED
    if _FINBERT is not None:
        return _FINBERT
    if _FINBERT_FAILED:
        return None
    if os.environ.get("USE_FINBERT", "1") == "0":
        _FINBERT_FAILED = True
        return None
    try:
        from transformers import pipeline  # type: ignore
        # device=-1 → CPU (safe default); transformers will use MPS/CUDA if available via accelerate.
        _FINBERT = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            top_k=None,  # return all class probabilities
            truncation=True,
            max_length=512,
        )
        return _FINBERT
    except Exception:
        _FINBERT_FAILED = True
        return None


def _score_finbert(text: str) -> tuple[float, float] | None:
    pipe = _try_load_finbert()
    if pipe is None:
        return None
    if not text or not text.strip():
        return 0.0, 0.0
    try:
        out = pipe(text[:2000])
        # pipe with top_k=None returns either a single list or list-of-lists
        records = out[0] if (out and isinstance(out[0], list)) else out
        probs = {r["label"].lower(): float(r["score"]) for r in records}
        polarity = sum(_LABEL_MAP.get(k, 0.0) * v for k, v in probs.items())
        # confidence = max class probability minus 1/3 (random baseline), rescaled
        confidence = max(0.0, min(1.0, (max(probs.values()) - 0.33) / 0.67))
        return float(polarity), float(confidence)
    except Exception:
        return None


def score_text(text: str) -> tuple[float, float]:
    """Return (polarity in [-1, +1], confidence in [0, 1])."""
    f = _score_finbert(text)
    if f is not None:
        return f
    return _score_lexicon(text)


def score_batch(texts: list[str], batch_size: int = 64) -> list[tuple[float, float]]:
    """Score many short texts (headlines) at once — one pipeline call per
    batch instead of per text, which is what makes scoring a multi-year news
    archive feasible on CPU."""
    pipe = _try_load_finbert()
    if pipe is None:
        return [_score_lexicon(t) for t in texts]
    try:
        outs = pipe([t[:512] or " " for t in texts], batch_size=batch_size)
    except Exception:
        return [_score_lexicon(t) for t in texts]
    scored: list[tuple[float, float]] = []
    for text, records in zip(texts, outs, strict=True):
        if not text.strip():
            scored.append((0.0, 0.0))
            continue
        probs = {r["label"].lower(): float(r["score"]) for r in records}
        polarity = sum(_LABEL_MAP.get(k, 0.0) * v for k, v in probs.items())
        confidence = max(0.0, min(1.0, (max(probs.values()) - 0.33) / 0.67))
        scored.append((float(polarity), float(confidence)))
    return scored


def is_finbert_active() -> bool:
    return _try_load_finbert() is not None


def aggregate(scores: list[tuple[float, float]]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "weighted": 0.0, "n": 0}
    weights = [c for _, c in scores] or [1.0] * len(scores)
    wsum = sum(weights) or 1.0
    weighted = sum(p * c for (p, c) in scores) / wsum
    mean = sum(p for p, _ in scores) / len(scores)
    return {"mean": float(mean), "weighted": float(weighted), "n": float(len(scores))}
