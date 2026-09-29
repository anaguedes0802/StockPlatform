"""SEC EDGAR Form 4 (insider transactions) adapter.

Source: https://data.sec.gov/submissions/CIK{cik}.json
  — lists every filing the company has made, including Form 4s.

For each recent Form 4 filing we pull the primary XML document and parse out:
  - reporting person (insider name)
  - relationship (director / officer / 10% holder)
  - transactionDate
  - transactionCode (P=open-market purchase, S=sale, A=grant, ...)
  - shares
  - transactionPricePerShare
  - sharesOwnedAfter

Free, authoritative, complete — replaces yfinance.insider_transactions which
returns sparse / empty data for most symbols.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

import httpx
import redis

from app.config import settings
from app.services.edgar import cik_for


_UA = "StockPlatform/0.1 contact@example.com"
# EFTS (EDGAR Full-Text Search) — finds Form 4s where the COMPANY is the issuer,
# regardless of whose CIK actually filed them. The plain submissions endpoint
# only lists filings *made by* the company itself, which excludes most Form 4s.
_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_clean}/{primary}"

_redis: redis.Redis | None = None


def _cache() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


# Process-local fallback used when Redis is unreachable. Without it, every
# feature build (i.e. every model fit and every walk-forward fold) re-downloads
# the full history from SEC — tens of seconds per call.
_local_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Any | None:
    try:
        v = _cache().get(key)
        if v:
            return json.loads(v)
    except Exception:
        pass
    hit = _local_cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    _local_cache[key] = (time.time() + ttl, json.loads(json.dumps(value, default=str)))
    try:
        _cache().setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass


# Map transaction codes to human-readable + bullish/bearish polarity
_CODE = {
    "P": ("Open-market purchase", "buy"),
    "S": ("Open-market sale", "sell"),
    "A": ("Grant/award", "grant"),
    "M": ("Option exercise (M)", "grant"),
    "F": ("Tax withholding", "neutral"),
    "G": ("Gift", "neutral"),
    "D": ("Sale to issuer", "sell"),
    "C": ("Conversion", "neutral"),
    "X": ("Option exercise (X)", "grant"),
    "I": ("Discretionary transaction", "neutral"),
    "V": ("Voluntary reported", "neutral"),
}


def _recent_form4_filings(cik: int, limit: int = 30) -> list[dict[str, Any]]:
    """Find Form 4 filings where the company (cik) is the *issuer*, regardless
    of which insider's CIK actually filed. Uses EDGAR Full-Text Search.

    Returns list of {accession, primary, filed, filer_cik}.
    """
    params = {
        "q": "",
        "forms": "4",
        "ciks": f"{cik:010d}",
    }
    try:
        r = httpx.get(_EFTS_URL, params=params,
                      headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    hits = ((data.get("hits") or {}).get("hits") or [])
    out: list[dict[str, Any]] = []
    for h in hits[:limit]:
        src = h.get("_source") or {}
        # _id format: "<accession-with-dashes>:<primary-doc>"
        full_id = h.get("_id") or ""
        if ":" not in full_id:
            continue
        accn, primary = full_id.split(":", 1)
        # The filer (insider) CIK is in the "ciks" list; the issuer (company)
        # is in the "tickers"/"display_names" — for path construction we need
        # the filer's CIK (the entity that submitted), found in the accn prefix.
        # Accession format: 0001234567-25-001234 — first 10 digits = filer CIK.
        try:
            filer_cik = int(accn.split("-")[0])
        except (ValueError, IndexError):
            continue
        out.append({
            "accession": accn,
            "primary": primary,
            "filed": src.get("file_date") or "",
            "filer_cik": filer_cik,
        })
    return out


_NS = {"ns": "http://www.sec.gov/edgar/common"}  # rarely used; Form 4 XMLs are usually unnamespaced


def _parse_form4(xml: str) -> dict[str, Any] | None:
    """Extract the salient fields from a Form 4 XML document.

    Form 4 XMLs are unnamespaced. Structure (abridged):
      <ownershipDocument>
        <reportingOwner>
          <reportingOwnerId><rptOwnerName>...</rptOwnerName></reportingOwnerId>
          <reportingOwnerRelationship>
            <isDirector>1</isDirector><isOfficer>0</isOfficer>...
            <officerTitle>...</officerTitle>
          </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTable>
          <nonDerivativeTransaction>
            <transactionDate><value>2025-10-01</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
              <transactionShares><value>1000</value></transactionShares>
              <transactionPricePerShare><value>250.0</value></transactionPricePerShare>
              <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
              <sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
          </nonDerivativeTransaction>
        </nonDerivativeTable>
      </ownershipDocument>
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None

    name = ((root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName")) or "").strip()
    rel = root.find(".//reportingOwnerRelationship")
    titles: list[str] = []
    if rel is not None:
        if (rel.findtext("isDirector") or "").strip() in ("1", "true"):
            titles.append("Director")
        if (rel.findtext("isOfficer") or "").strip() in ("1", "true"):
            t = (rel.findtext("officerTitle") or "Officer").strip()
            titles.append(t)
        if (rel.findtext("isTenPercentOwner") or "").strip() in ("1", "true"):
            titles.append("10% holder")

    tx: list[dict[str, Any]] = []
    for node in root.findall(".//nonDerivativeTransaction"):
        date_v = (node.findtext("./transactionDate/value") or "").strip()
        code = (node.findtext("./transactionCoding/transactionCode") or "").strip()
        shares = (node.findtext("./transactionAmounts/transactionShares/value") or "").strip()
        price = (node.findtext("./transactionAmounts/transactionPricePerShare/value") or "").strip()
        adc = (node.findtext("./transactionAmounts/transactionAcquiredDisposedCode/value") or "").strip()
        owned_after = (node.findtext("./postTransactionAmounts/sharesOwnedFollowingTransaction/value") or "").strip()
        try:
            shares_f = float(shares) if shares else 0.0
            price_f = float(price) if price else 0.0
            owned_after_f = float(owned_after) if owned_after else 0.0
        except ValueError:
            continue
        signed_shares = shares_f if adc == "A" else -shares_f
        label, polarity = _CODE.get(code, (code, "neutral"))
        tx.append({
            "date": date_v,
            "code": code,
            "code_label": label,
            "polarity": polarity,
            "shares": signed_shares,
            "price": price_f,
            "value": signed_shares * price_f,
            "shares_owned_after": owned_after_f,
        })

    if not name and not tx:
        return None
    return {"insider": name, "titles": titles, "transactions": tx}


def fetch_insider_transactions(symbol: str, limit_filings: int = 25) -> list[dict[str, Any]]:
    """Return individual Form 4 transactions, newest first, for a symbol."""
    cik = cik_for(symbol)
    if cik is None:
        return []
    cache_key = f"form4:{cik}:{limit_filings}"
    if (cached := _cache_get(cache_key)) is not None:
        return cached

    filings = _recent_form4_filings(cik, limit=limit_filings)
    out: list[dict[str, Any]] = []
    for f in filings:
        accn = f["accession"]
        primary = f["primary"]
        if not primary or not primary.lower().endswith((".xml", ".html", ".htm")):
            continue
        accn_clean = accn.replace("-", "")
        # Archive path is keyed by the **issuer** (company) CIK, not the filer.
        # Verified empirically: /Archives/edgar/data/320193/... works for AAPL Form 4s,
        # /Archives/edgar/data/{filer_cik}/... returns 404.
        url = _ARCHIVE_URL.format(cik_int=cik, accn_clean=accn_clean, primary=primary)
        try:
            r = httpx.get(url, headers={"User-Agent": _UA}, timeout=12.0)
            if r.status_code != 200:
                continue
            text = r.text
        except Exception:
            continue
        # Some primary docs are HTML wrappers; we want the XML. Try .xml sibling.
        if "<ownershipDocument" not in text and primary.endswith((".html", ".htm")):
            xml_primary = primary.rsplit(".", 1)[0] + ".xml"
            try:
                r2 = httpx.get(_ARCHIVE_URL.format(cik_int=cik, accn_clean=accn_clean, primary=xml_primary),
                               headers={"User-Agent": _UA}, timeout=12.0)
                if r2.status_code == 200:
                    text = r2.text
            except Exception:
                pass
        parsed = _parse_form4(text)
        if not parsed:
            continue
        for tx in parsed["transactions"]:
            out.append({
                "filed": f["filed"],
                "accession": accn,
                "insider": parsed["insider"],
                "titles": parsed["titles"],
                **tx,
            })

    out.sort(key=lambda r: (r.get("date") or "", r.get("filed") or ""), reverse=True)
    _cache_set(cache_key, out, ttl=4 * 3600)
    return out


def insider_density_features(symbol: str, as_of_dates: list[datetime]) -> dict[str, list[float]]:
    """Per-bar insider density features for ML training.

    For each timestamp in `as_of_dates`, computes:
      - insider_net_buys_90d: count of buys minus count of sells in the
                              preceding 90 days (only counting filings whose
                              `filed` date <= as_of)
      - insider_net_value_90d: signed dollar flow over the same window
                               (tanh-squashed to [-1, +1])
    Backtest-safe (uses filed dates, not transaction dates) so we never see
    the future.
    """
    import math
    txs = fetch_insider_transactions(symbol, limit_filings=200)
    # Only count P/S codes for the density signal — ignore grants & gifts.
    txs = [t for t in txs if t.get("code") in ("P", "S")]
    parsed: list[tuple[datetime, str, float]] = []
    for t in txs:
        try:
            filed = datetime.fromisoformat(str(t.get("filed") or "")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        parsed.append((filed, t["code"], t.get("value", 0.0)))

    nets: list[float] = []
    values: list[float] = []
    for ts in as_of_dates:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        window_start = ts.timestamp() - 90 * 86400
        net = 0
        dollar = 0.0
        for filed, code, v in parsed:
            ft = filed.timestamp()
            if ft <= ts.timestamp() and ft >= window_start:
                if code == "P":
                    net += 1; dollar += v
                elif code == "S":
                    net -= 1; dollar += v   # v is already negative for sells
        nets.append(float(net))
        # Squash dollar flow to [-1, +1] with $5M as a saturating scale
        values.append(float(math.tanh(dollar / 5_000_000)))
    return {
        "insider_net_buys_90d": nets,
        "insider_net_value_90d": values,
    }


def insider_signal_v2(symbol: str) -> dict[str, Any]:
    """Compact summary signal from real EDGAR Form 4 data with cluster detection.

    Cluster effect (the real edge): when ≥3 distinct insiders buy within 30 days,
    that's empirically a much stronger signal than a single large purchase.
    We apply a multiplicative cluster bonus to the raw buy signal.
    """
    txs = fetch_insider_transactions(symbol, limit_filings=100)
    if not txs:
        return {
            "score": 0.0, "n_buys_90d": 0, "n_sells_90d": 0,
            "net_value_90d": 0.0, "summary": "No Form 4 filings in EDGAR.",
            "cluster": {"n_buyers_30d": 0, "is_cluster": False, "bonus": 0.0},
            "source": "sec_edgar",
        }
    now = datetime.now(timezone.utc).timestamp()
    cutoff_90 = now - 90 * 86400
    cutoff_30 = now - 30 * 86400
    n_buys = n_sells = 0
    net_value = 0.0
    drivers: list[str] = []
    buyers_30d: set[str] = set()
    sellers_30d: set[str] = set()
    for t in txs:
        try:
            filed = datetime.fromisoformat(str(t.get("filed") or "")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if filed.timestamp() < cutoff_90:
            continue
        code = t.get("code")
        insider = (t.get("insider") or "").strip()
        if code == "P":
            n_buys += 1; net_value += t.get("value", 0.0)
            if insider:
                drivers.append(f"{insider} bought {abs(t['shares']):.0f} sh @ ${t['price']:.2f} on {t['date']}")
            if filed.timestamp() >= cutoff_30 and insider:
                buyers_30d.add(insider)
        elif code == "S":
            n_sells += 1; net_value += t.get("value", 0.0)
            if filed.timestamp() >= cutoff_30 and insider:
                sellers_30d.add(insider)

    total = n_buys + n_sells
    import math
    if total == 0:
        score = 0.0
    else:
        ratio = (n_buys - n_sells) / total
        dollar_tilt = math.tanh(net_value / 5_000_000)
        score = max(-1.0, min(1.0, 0.6 * ratio + 0.4 * dollar_tilt))

    # Cluster bonus: 3+ distinct insiders buying in 30d boosts the BUY signal.
    # Conversely, 4+ distinct sellers is a stronger SELL signal.
    n_b30 = len(buyers_30d)
    n_s30 = len(sellers_30d)
    cluster_bonus = 0.0
    is_cluster = False
    if n_b30 >= 3 and score > 0:
        # +0.10 per additional buyer beyond the 2-insider baseline, capped at +0.30
        cluster_bonus = min(0.30, 0.10 * (n_b30 - 2))
        is_cluster = True
    elif n_s30 >= 4 and score < 0:
        cluster_bonus = -min(0.20, 0.05 * (n_s30 - 3))
        is_cluster = True
    score = max(-1.0, min(1.0, score + cluster_bonus))

    parts: list[str] = []
    if n_buys: parts.append(f"{n_buys} insider buys (90d)")
    if n_sells: parts.append(f"{n_sells} insider sells (90d)")
    if abs(net_value) > 1_000_000: parts.append(f"net ${net_value/1e6:+.1f}M")
    if n_b30 >= 3:
        parts.append(f"⚡ cluster: {n_b30} distinct buyers in 30d")
    summary = "; ".join(parts) or "No P/S code transactions in last 90d."

    return {
        "score": round(float(score), 3),
        "n_buys_90d": n_buys,
        "n_sells_90d": n_sells,
        "net_value_90d": round(net_value, 2),
        "cluster": {
            "n_buyers_30d": n_b30,
            "n_sellers_30d": n_s30,
            "is_cluster": is_cluster,
            "bonus": round(cluster_bonus, 3),
        },
        "summary": summary,
        "top_buys": drivers[:5],
        "source": "sec_edgar",
    }
