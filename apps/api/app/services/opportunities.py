"""Opportunity Engine — continuous multi-strategy hunt.

Runs all 6 screener strategies in parallel, deduplicates per symbol, and
returns a unified ranked feed. This is what powers the "ongoing search" UI
that surfaces hits across Growth / Value+Catalyst / Event / Technical
buckets in one place.

How the score works:
  - Each strategy produces a per-symbol score in [-1, +1]
  - A symbol can appear in multiple strategies (e.g. a small-cap with a fresh
    FDA approval ranks high on BOTH underpriced_movers AND catalyst_plays)
  - Each strategy's score is scaled by its weight: a hand-set prior times a
    multiplier learned from the list's own track record vs SPY
    (track_record.strategy_weights) — strategies whose surfaced names lagged
    the market lose weight as evidence accumulates
  - The thesis (raw) score is `max(strategy_scores)` (best lens wins)
  - A multi-strategy bonus is added: +0.05 per additional strategy the
    symbol shows up in — confluence matters
  - The displayed score is `raw_score × entry_quality × narrative_alignment`:
    entry_quality discounts a strong thesis with a bad entry, and
    narrative_alignment discounts a name whose underlying thesis driver is
    broken (e.g. a bitcoin proxy while BTC is rolling over) — see narrative.py

Background warmup runs this every 5 min during market hours, persists
snapshots via the existing `screener_snapshots` table, and a notification
fires when a symbol crosses score ≥ 0.6 AND wasn't in the prior snapshot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.logging import log
from app.services import screener as screener_svc


# Strategies we hunt across, with category tags + prior weight.
# Priors are deliberately neutral (1.0). The old hand-set values ("catalyst
# plays = biggest alpha" 1.10, breakouts 0.85, ...) had no data behind them and
# the list's own record contradicted them (smart_money was the best performer
# vs SPY, breakouts the worst). The effective weight is prior × a multiplier
# learned from that record — see track_record.strategy_weights.
STRATEGIES_TO_HUNT = [
    ("rising_stars",        "Growth",            1.0),
    ("underpriced_movers",  "Value + Catalyst",  1.0),
    ("catalyst_plays",      "Event",             1.0),
    ("smart_money",         "Technical",         1.0),
    ("breakouts",           "Technical",         1.0),
    ("value_with_catalyst", "Value",             1.0),
]


@dataclass
class Opportunity:
    symbol: str
    name: str | None = None
    sector: str | None = None
    raw_score: float = 0.0          # the thesis score from the best strategy
    entry_quality: float = 1.0      # multiplier in [0, 1] — how good is RIGHT NOW as an entry
    narrative_alignment: float = 1.0  # multiplier — is the underlying thesis driver intact?
    narrative: dict[str, Any] | None = None  # driver/conflict detail (None when no mapped driver)
    score: float = 0.0              # raw_score × entry_quality × narrative_alignment (displayed)
    best_strategy: str = ""
    best_category: str = ""
    in_strategies: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    entry_quality_notes: list[str] = field(default_factory=list)
    catalyst_category: str | None = None
    catalyst_materiality: float | None = None
    catalyst_headline: str | None = None
    last_price: float | None = None
    market_cap: float | None = None
    pe: float | None = None
    rsi14: float | None = None
    momentum_1m: float | None = None
    momentum_3m: float | None = None
    pct_off_52w_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "score": round(self.score, 3),
            "raw_score": round(self.raw_score, 3),
            "entry_quality": round(self.entry_quality, 3),
            "entry_notes": self.entry_quality_notes,
            "narrative_alignment": round(self.narrative_alignment, 3),
            "narrative": self.narrative,
            "best_strategy": self.best_strategy,
            "best_category": self.best_category,
            "in_strategies": self.in_strategies,
            "rationale": self.rationale[:6],
            "catalyst": {
                "category": self.catalyst_category,
                "materiality": self.catalyst_materiality,
                "headline": self.catalyst_headline,
            } if self.catalyst_category else None,
            "last_price": self.last_price,
            "market_cap": self.market_cap,
            "pe": self.pe,
            "rsi14": self.rsi14,
            "momentum_1m_pct": round(self.momentum_1m * 100, 1) if self.momentum_1m is not None else None,
            "momentum_3m_pct": round(self.momentum_3m * 100, 1) if self.momentum_3m is not None else None,
            "pct_off_52w_high": round(self.pct_off_52w_high * 100, 1) if self.pct_off_52w_high is not None else None,
        }


def _compute_entry_quality(features: dict[str, Any]) -> tuple[float, list[str]]:
    """How good is RIGHT NOW as an entry, separately from how strong the thesis is.

    A strong company can be a terrible entry (UNH up 34% in 3M, RSI 80, 1%
    off all-time high → wait for pullback). The opportunity engine should
    surface the names where the thesis is strong AND the entry is clean,
    not just where the thesis is strong.

    Penalties (multiplicative, [0..1] each, multiply together):
      - RSI overbought (>75)        — 0.5–0.7×
      - Within 2% of 52w high       — 0.6×
      - 3-month return > 40%        — 0.4–0.7× (later you are, harder it gets)
      - Vertical move last 20d      — 0.7× (parabolic ≠ entry)

    Bonuses:
      - Pullback in uptrend (RSI 40-55, above 200d) — 1.15×
      - Above 200d AND modest 1M move (5-15%)        — 1.05×
    """
    notes: list[str] = []
    quality = 1.0

    rsi = features.get("rsi14")
    r1m = features.get("ret_1m") or 0
    r3m = features.get("ret_3m") or 0
    pct_off_high = features.get("pct_off_high") or 0   # 0 = at 52w high; bigger = farther
    above200 = features.get("above_sma200")

    # --- Penalties --------------------------------------------------------
    if rsi is not None:
        if rsi > 80:
            quality *= 0.45
            notes.append(f"RSI {rsi:.0f} extreme overbought (×0.45)")
        elif rsi > 75:
            quality *= 0.65
            notes.append(f"RSI {rsi:.0f} overbought (×0.65)")
        elif rsi > 70:
            quality *= 0.82
            notes.append(f"RSI {rsi:.0f} elevated (×0.82)")

    if pct_off_high < 0.02:
        quality *= 0.60
        notes.append(f"<2% off 52w high — no headroom (×0.60)")
    elif pct_off_high < 0.05:
        quality *= 0.82
        notes.append(f"<5% off 52w high (×0.82)")

    if r3m > 0.60:
        quality *= 0.40
        notes.append(f"3M +{r3m*100:.0f}% — parabolic, easy money gone (×0.40)")
    elif r3m > 0.40:
        quality *= 0.60
        notes.append(f"3M +{r3m*100:.0f}% — extended (×0.60)")
    elif r3m > 0.25:
        quality *= 0.80
        notes.append(f"3M +{r3m*100:.0f}% — elevated entry (×0.80)")

    if r1m > 0.25:
        quality *= 0.70
        notes.append(f"1M +{r1m*100:.0f}% — recent spike (×0.70)")

    # --- Bonuses ----------------------------------------------------------
    # Pullback in uptrend: above the long-term trend, but RSI in the middle
    # band (40-55) — that's the "buy the dip" sweet spot.
    if above200 and rsi is not None and 40 <= rsi <= 55:
        quality *= 1.15
        notes.append(f"Pullback in uptrend (RSI {rsi:.0f} + above 200d) (×1.15)")
    elif above200 and 0.05 <= r1m <= 0.15 and (rsi is None or 50 <= rsi <= 65):
        quality *= 1.05
        notes.append("Healthy uptrend, modest momentum (×1.05)")

    quality = max(0.05, min(1.30, quality))   # cap so a great setup can boost by 30%
    return quality, notes


def _run_one_strategy(strategy: str) -> list[dict[str, Any]]:
    """Run a single strategy; defensive — never raise."""
    try:
        return screener_svc.run_screener(strategy, limit=20) or []
    except Exception as e:
        log.warning("opportunity_strategy_failed", strategy=strategy, err=str(e))
        return []


# Per-symbol feature cache built up DURING the screener pass — when each
# strategy runs, it calls _feature_pack(sym) which we capture here so the
# entry-quality pass doesn't have to re-fetch (and trigger Alpaca/yfinance
# rate limits). Refreshed at the start of every scan.
_features_cache: dict[str, dict[str, Any]] = {}


def _capture_features(sym: str) -> dict[str, Any] | None:
    """Memoized one-shot feature fetch — called only when the entry-quality
    pass needs data for a symbol that wasn't seen during strategy execution.
    Returns None on failure so the caller can skip gracefully."""
    if sym in _features_cache:
        return _features_cache[sym]
    try:
        from app.services.screener import _feature_pack
        f = _feature_pack(sym)
        if f:
            _features_cache[sym] = f
            return f
    except Exception as e:
        log.debug("opp_feature_capture_failed", symbol=sym, err=str(e))
    return None


def scan(limit: int = 30) -> dict[str, Any]:
    """Run all 6 strategies in parallel, aggregate, dedupe, rank.

    Returns:
      {
        "as_of": iso timestamp,
        "n_strategies_run": int,
        "n_opportunities": int,
        "opportunities": [Opportunity dicts, sorted by score desc],
      }
    """
    # Parallel-run all strategies (each one is ~10-25s cold)
    results: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=len(STRATEGIES_TO_HUNT)) as ex:
        futures = {ex.submit(_run_one_strategy, s[0]): s[0] for s in STRATEGIES_TO_HUNT}
        for fut in as_completed(futures):
            strat = futures[fut]
            try:
                results[strat] = fut.result() or []
            except Exception:
                results[strat] = []

    # Aggregate per symbol — first pass builds raw scores.
    # weight = hand-set prior × multiplier learned from the list's own track
    # record vs SPY (track_record.strategy_weights); 1.0 when no data yet.
    learned = _learned_weights()
    learned_by = learned.get("strategies", {})
    weights = {s[0]: s[2] * learned_by.get(s[0], {}).get("multiplier", 1.0) for s in STRATEGIES_TO_HUNT}
    categories = {s[0]: s[1] for s in STRATEGIES_TO_HUNT}
    by_symbol: dict[str, Opportunity] = {}

    for strategy, hits in results.items():
        w = weights.get(strategy, 1.0)
        for h in hits:
            sym = h.get("symbol")
            if not sym:
                continue
            score = float(h.get("score") or 0) * w
            opp = by_symbol.setdefault(sym, Opportunity(symbol=sym))
            opp.in_strategies.append(strategy)
            if score > opp.raw_score:
                opp.raw_score = score
                opp.best_strategy = strategy
                opp.best_category = categories.get(strategy, "")
                opp.rationale = h.get("rationale") or []
            opp.name = opp.name or h.get("name")
            opp.last_price = opp.last_price or h.get("last_price")
            if h.get("rationale"):
                for r in h["rationale"]:
                    if "Catalyst:" in r or "Material " in r or "Lead:" in r:
                        if not opp.catalyst_headline and "Lead:" in r:
                            opp.catalyst_headline = r.replace("Lead:", "").strip()
                        if not opp.catalyst_category and "Catalyst:" in r:
                            tail = r.split("Catalyst:", 1)[1].strip()
                            opp.catalyst_category = tail.split("(")[0].strip()
                            try:
                                mat_str = tail.split("materiality")[1].split(")")[0].strip()
                                opp.catalyst_materiality = float(mat_str)
                            except Exception:
                                pass

    # Multi-strategy confluence bonus on the RAW score
    for opp in by_symbol.values():
        if len(opp.in_strategies) >= 2:
            opp.raw_score = min(1.0, opp.raw_score + 0.05 * (len(opp.in_strategies) - 1))

    # Second pass: compute entry quality per opportunity.
    # CRITICAL: only use already-cached feature data — don't make fresh API
    # calls here, or 70+ symbols × 3 backends triggers a 429 cascade.
    # _capture_features wraps _feature_pack which has its own 60s cache; if
    # the data isn't already there, we silently skip entry-quality scoring
    # for that symbol (entry_quality stays at default 1.0).
    for opp in by_symbol.values():
        try:
            feats = _capture_features(opp.symbol)
            if feats:
                opp.rsi14 = feats.get("rsi14")
                opp.momentum_1m = feats.get("ret_1m")
                opp.momentum_3m = feats.get("ret_3m")
                opp.pct_off_52w_high = feats.get("pct_off_high")
                opp.market_cap = opp.market_cap or feats.get("market_cap")
                opp.pe = opp.pe or feats.get("pe")
                opp.sector = opp.sector or feats.get("sector")
                eq, notes = _compute_entry_quality(feats)
                opp.entry_quality = eq
                opp.entry_quality_notes = notes
        except Exception as e:
            log.warning("opp_entry_quality_failed", symbol=opp.symbol, err=str(e))
            opp.entry_quality = 1.0

        # Narrative reconciliation: is the underlying thesis driver intact?
        # The agent runs an LLM driver-ID + news-conflict pass, so gate it to
        # genuine candidates — a name whose thesis×entry is already weak can't
        # become a top pick, and narrative can only lower the score. This bounds
        # LLM/news work to the symbols that could actually surface.
        partial = opp.raw_score * opp.entry_quality
        if partial < 0.45:
            opp.narrative_alignment = 1.0
            opp.narrative = None
        else:
            try:
                from app.services import narrative
                na = narrative.assess(opp.symbol)
                opp.narrative_alignment = na.multiplier
                opp.narrative = na.to_dict() if na.theme else None
                if na.theme and na.notes:
                    opp.rationale = (opp.rationale or []) + na.notes[:2]
            except Exception as e:
                log.warning("opp_narrative_failed", symbol=opp.symbol, err=str(e))
                opp.narrative_alignment = 1.0

        # Final displayed score: thesis × entry-quality × narrative-alignment.
        opp.score = round(opp.raw_score * opp.entry_quality * opp.narrative_alignment, 3)

    # Sort by FINAL score (entry-quality-adjusted), then by confluence
    ranked = sorted(
        by_symbol.values(),
        key=lambda o: (-o.score, -len(o.in_strategies)),
    )[:limit]

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_strategies_run": len(results),
        "n_opportunities": len(by_symbol),
        "opportunities": [
            {**o.to_dict(), "strategy_record": learned_by.get(o.best_strategy)} for o in ranked
        ],
        "strategy_weights": {s[0]: {"prior": s[2], **learned_by.get(s[0], {"multiplier": 1.0}),
                                    "effective": round(weights[s[0]], 3)} for s in STRATEGIES_TO_HUNT},
        "validation": {k: learned.get(k) for k in ("validated", "weeks_of_evidence",
                                                   "overall_avg_vs_spy_12w", "message")},
    }


def _learned_weights() -> dict[str, Any]:
    """Track-record-based strategy multipliers; empty (all 1.0) if unavailable."""
    try:
        from app.db.session import SessionLocal
        from app.services import track_record
        with SessionLocal() as db:
            return track_record.strategy_weights(db)
    except Exception as e:
        log.warning("opp_learned_weights_failed", err=str(e)[:160])
        return {}


# ----------------------------------------------------------------------------
# In-memory result cache so /opportunities serves instantly between scans.
# ----------------------------------------------------------------------------

import time as _time

_last_scan: dict[str, Any] | None = None
_last_scan_ts: float = 0.0


def get_cached(max_age_s: int = 600) -> dict[str, Any] | None:
    """Return the last scan if it's still fresh (default 10min). None if stale
    or never run."""
    if _last_scan and (_time.time() - _last_scan_ts) <= max_age_s:
        return _last_scan
    return None


def scan_and_cache(limit: int = 30) -> dict[str, Any]:
    """Run a fresh scan and update the cache."""
    global _last_scan, _last_scan_ts
    result = scan(limit=limit)
    _last_scan = result
    _last_scan_ts = _time.time()
    return result


# ----------------------------------------------------------------------------
# Notification trigger — fires when a NEW high-conviction opportunity surfaces.
# ----------------------------------------------------------------------------

def diff_and_notify_new(db, user, prior_symbols: set[str], current: dict[str, Any]) -> int:
    """For each opportunity with score >= 0.6 that wasn't in the prior snapshot,
    create a notification. Returns the number created."""
    from app.services.notifications import _create_if_new
    created = 0
    for opp in current.get("opportunities", []):
        if opp["score"] < 0.6:
            continue
        sym = opp["symbol"]
        if sym in prior_symbols:
            continue
        rationale = " · ".join(opp.get("rationale", [])[:2])
        body = f"{opp['best_category']} (score {opp['score']}) — {rationale}"
        n = _create_if_new(
            db, user_id=user.id, kind="opportunity_surfaced",
            severity="warning" if opp["score"] >= 0.75 else "info",
            symbol=sym,
            title=f"🎯 {sym} surfaced as {opp['best_strategy'].replace('_', ' ')}",
            body=body,
            payload={"score": opp["score"], "strategies": opp["in_strategies"],
                     "catalyst": opp.get("catalyst")},
        )
        if n:
            created += 1
    return created
