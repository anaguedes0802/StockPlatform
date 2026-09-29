"""US Congress (STOCK Act) trades.

Sources, in order:
  - **House Clerk PTR archive** (official, free) via app/services/congress_ptr.py
  - **Finnhub** (Senate + House) when FINNHUB_API_KEY is set
  - **Bundled demo fixture** only when neither is available — always flagged `demo: true`

Returns per-symbol and recent feeds, plus a `political_signal()` for the
recommendation engine.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import redis

from app.config import settings
from app.services import congress_ptr
from app.services.politicians_data import BUNDLED_TRADES, RANGE_MIDPOINTS


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


def has_live_adapter() -> bool:
    return bool(settings.finnhub_api_key)


# -----------------------------------------------------------------------------
# Finnhub adapter
# -----------------------------------------------------------------------------

def _finnhub_trades_for_symbol(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """Finnhub /stock/congressional-trading returns list of trades for a symbol."""
    key = f"pol_fin:{symbol}:{limit}"
    if (cached := _cache_get(key)) is not None:
        return cached
    url = "https://finnhub.io/api/v1/stock/congressional-trading"
    try:
        r = httpx.get(url, params={"symbol": symbol.upper(), "token": settings.finnhub_api_key}, timeout=8.0)
        r.raise_for_status()
        data = r.json()
        raw = data.get("data") or []
    except Exception:
        raw = []
    out: list[dict[str, Any]] = []
    for tr in raw[:limit]:
        out.append({
            "politician": tr.get("name"),
            "chamber": "House" if (tr.get("position") or "").lower().startswith("rep") else
                      ("Senate" if (tr.get("position") or "").lower().startswith("sen") else None),
            "party": tr.get("party"),
            "traded_at": tr.get("transactionDate"),
            "disclosed_at": tr.get("reportDate"),
            "symbol": tr.get("symbol"),
            "side": (tr.get("transactionType") or "").lower(),
            "amount_range": _format_range(tr.get("amountFrom"), tr.get("amountTo")),
            "note": None,
        })
    _cache_set(key, out, ttl=2 * 3600)
    return out


def _format_range(a: float | None, b: float | None) -> str:
    if a is None and b is None:
        return ""
    if a and b:
        return f"${_short(a)} – ${_short(b)}"
    return f"${_short(a or b)}"


def _short(v: float) -> str:
    if v >= 1_000_000: return f"{v/1_000_000:.0f}M"
    if v >= 1_000: return f"{v/1_000:.0f}K"
    return f"{v:.0f}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _demo(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The bundled fixture is hand-written sample data: never show it unlabelled."""
    return [{**t, "demo": True, "source": "demo"} for t in trades]


def data_source() -> str:
    if congress_ptr.has_data():
        return "house_clerk" + ("+finnhub_senate" if has_live_adapter() else "")
    return "finnhub" if has_live_adapter() else "demo"


def trades_for_symbol(symbol: str, limit: int = 25) -> list[dict[str, Any]]:
    """Politician trades for one ticker, newest disclosure first.

    House: official PTR archive (congress_ptr). Senate: Finnhub when a key is
    configured. The bundled demo sample is used only when neither exists."""
    sym = symbol.upper()
    out: list[dict[str, Any]] = []
    if congress_ptr.has_data():
        out.extend(congress_ptr.trades(symbol=sym, limit=limit))
    if has_live_adapter():
        live = _finnhub_trades_for_symbol(sym, limit=limit)
        out.extend(t for t in live if not congress_ptr.has_data() or t.get("chamber") == "Senate")
    if not out and not congress_ptr.has_data() and not has_live_adapter():
        out = _demo([t for t in BUNDLED_TRADES if t["symbol"].upper() == sym])
    out.sort(key=lambda t: (t.get("disclosed_at") or t.get("traded_at") or ""), reverse=True)
    return out[:limit]


def recent_trades(limit: int = 30, days: int | None = None) -> list[dict[str, Any]]:
    """Latest disclosures across Congress (House, official source)."""
    if congress_ptr.has_data():
        return congress_ptr.trades(days=days, limit=limit)
    pool = _demo(list(BUNDLED_TRADES))
    pool.sort(key=lambda t: (t.get("disclosed_at") or t.get("traded_at") or ""), reverse=True)
    return pool[:limit]


def by_politician(name: str, limit: int = 30) -> list[dict[str, Any]]:
    if congress_ptr.has_data():
        return congress_ptr.trades(politician=name, limit=limit)
    out = _demo([t for t in BUNDLED_TRADES if t["politician"].lower() == name.lower()])
    out.sort(key=lambda t: t.get("traded_at", ""), reverse=True)
    return out[:limit]


def tracker() -> dict[str, Any]:
    """Congress tracker payload: archive status + per-politician copy scorecards."""
    if not congress_ptr.has_data():
        return {"available": False, "status": None, "scorecards": [],
                "message": "No disclosure archive yet — run scripts/refresh_congress.py --since 2020."}
    return {
        "available": True,
        "status": congress_ptr.status(),
        "scorecards": congress_ptr.scorecards(),
        "method": ("For each politician with >= 5 disclosed stock/option purchases in the window: buy the "
                   "underlying stock at the first open after the filing was published, hold 3 or 12 months, "
                   "and compare with SPY over the identical window. Options are copied as the stock."),
        "caveats": [
            "Disclosures arrive up to 45 days after the trade (median ~3-4 weeks); these numbers use the "
            "publication date, i.e. what you could actually copy.",
            "Most 'star' records come from concentration in mega-cap tech during 2015-2026: a rule holding "
            "the 5 most-traded US stocks did as well as copying Nancy Pelosi (BACKTEST_RESULTS.md, v8).",
            "t-stats treat each purchase as independent, but members' trades cluster in time (and in the "
            "same stocks), so real uncertainty is larger: read |t| < 3 as weak evidence either way.",
            "Across all members with a scorecard, the median copied purchase trailed SPY at 12 months — "
            "most members' disclosed buys are not a market-beating signal.",
            "House only: Senate filings need FINNHUB_API_KEY. Paper (scanned) filings are not parsed.",
        ],
    }


def political_signal(symbol: str) -> dict[str, Any]:
    """Compact signal for the recommendation engine.

    Looks at the last 180 days of disclosures for the symbol:
      - Net dollar bias (buys - sells), weighted by range midpoint
      - High-impact politicians (Pelosi, Tuberville, McCaul) given extra weight
    """
    HIGH_IMPACT = {"Nancy Pelosi", "Tommy Tuberville", "Michael McCaul",
                   "Dan Crenshaw", "Mark Green", "Sheldon Whitehouse"}
    trades = trades_for_symbol(symbol, limit=50)
    if not trades:
        return {"score": 0.0, "n_trades_180d": 0, "summary": "No political trades on record."}

    now = datetime.now(timezone.utc)
    net_dollars = 0.0
    n_buys = n_sells = 0
    contributors: dict[str, float] = {}
    for tr in trades:
        try:
            ts = datetime.fromisoformat(str(tr.get("traded_at") or "")[:10]).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if (now - ts).days > 180:
            continue
        side = (tr.get("side") or "").lower()
        amount = tr.get("amount_mid") or RANGE_MIDPOINTS.get(tr.get("amount_range", ""), 0)
        weight = 2.0 if tr.get("politician") in HIGH_IMPACT else 1.0
        signed = amount * weight
        if "buy" in side or "purchase" in side:
            net_dollars += signed; n_buys += 1
        elif "sell" in side or "sale" in side or "sold" in side:
            net_dollars -= signed; n_sells += 1
        contributors[tr.get("politician")] = contributors.get(tr.get("politician"), 0) + signed * (1 if "buy" in side else -1)

    # Squash to [-1, +1]
    import math
    score = float(math.tanh(net_dollars / 1_000_000))

    parts: list[str] = []
    if n_buys: parts.append(f"{n_buys} buys")
    if n_sells: parts.append(f"{n_sells} sells")
    if abs(net_dollars) > 50_000:
        parts.append(f"weighted net ${net_dollars/1e3:+.0f}K (180d)")
    top = sorted(contributors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    if top:
        parts.append("notable: " + ", ".join(n for n, _ in top))
    summary = "; ".join(parts) if parts else "Limited political activity (180d)."

    return {
        "score": round(score, 3),
        "n_trades_180d": n_buys + n_sells,
        "net_dollars_180d": round(net_dollars, 2),
        "n_buys_180d": n_buys,
        "n_sells_180d": n_sells,
        "top_contributors": [{"politician": n, "signed_dollars": round(v, 2)} for n, v in top],
        "summary": summary,
        "source": data_source(),
    }
