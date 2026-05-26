"""SEC EDGAR 13F-HR adapter.

Pulls each fund's most recent 13F holdings list directly from EDGAR. Free,
authoritative; replaces the hand-curated `famous_investors.py` snapshot.

Approach:
  1. For each fund CIK, hit the submissions endpoint to find the most recent
     13F-HR filing.
  2. Download the primary "informationTable.xml" document for that filing.
  3. Parse holdings (cusip, name of issuer, value, sshPrnamt = share count).
  4. Map CUSIPs → tickers via a local cache populated from yfinance (best-effort).
  5. Cache 6h per fund.

Known limitations:
  - 13F filings have a ~45-day reporting lag. We surface the period_end so users
    can judge staleness.
  - CUSIP→ticker mapping is imperfect; some holdings (private placements,
    options, foreign issues) won't map.
"""
from __future__ import annotations

import json
from typing import Any
from xml.etree import ElementTree as ET

import httpx
import redis

from app.config import settings


_UA = "StockPlatform/0.1 contact@example.com"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_clean}/"
_INDEX_URL = _ARCHIVE_URL + "index.json"

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


# Known fund CIKs + display metadata. Edit to add more.
FUND_REGISTRY: list[dict[str, Any]] = [
    {"cik": 1067983, "name": "Warren Buffett", "fund": "Berkshire Hathaway",
     "philosophy": "Long-duration concentrated quality — wide moats, durable earnings."},
    {"cik": 1336528, "name": "Bill Ackman", "fund": "Pershing Square Capital",
     "philosophy": "Highly concentrated, activist long-only — 8-12 positions."},
    {"cik": 1649339, "name": "Michael Burry", "fund": "Scion Asset Management",
     "philosophy": "Deep-value contrarian; small concentrated book."},
    {"cik": 1536411, "name": "Stanley Druckenmiller", "fund": "Duquesne Family Office",
     "philosophy": "Top-down macro + concentrated equities."},
    {"cik": 1697748, "name": "Cathie Wood", "fund": "ARK Investment Management",
     "philosophy": "Disruptive innovation — high-beta growth."},
    {"cik": 1079114, "name": "David Einhorn", "fund": "Greenlight Capital",
     "philosophy": "Value-oriented long/short."},
    {"cik": 1061165, "name": "Seth Klarman", "fund": "Baupost Group",
     "philosophy": "Absolute-return value with margin of safety."},
    {"cik": 1284812, "name": "Howard Marks", "fund": "Oaktree Capital",
     "philosophy": "Distressed credit + opportunistic equity."},
    {"cik": 1167483, "name": "Chase Coleman", "fund": "Tiger Global Management",
     "philosophy": "Internet/tech growth crossover."},
    {"cik": 1350694, "name": "Ray Dalio", "fund": "Bridgewater Associates",
     "philosophy": "Risk-parity macro."},
]


def _most_recent_13f(cik: int) -> dict[str, Any] | None:
    """Find the latest 13F-HR for this fund. Returns {accession, primary, period_end, filed}."""
    try:
        r = httpx.get(_SUBMISSIONS_URL.format(cik=cik),
                      headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accns = recent.get("accessionNumber") or []
    primaries = recent.get("primaryDocument") or []
    period_ends = recent.get("reportDate") or recent.get("periodOfReport") or []
    filed_dates = recent.get("filingDate") or []
    for f, a, p, pe, fd in zip(forms, accns, primaries, period_ends, filed_dates, strict=True):
        if f in ("13F-HR", "13F-HR/A"):
            return {"accession": a, "primary": p, "period_end": pe, "filed": fd}
    return None


def _fetch_information_table(cik: int, accession: str) -> str | None:
    """Download the information-table XML for a 13F filing.

    Filers name this file inconsistently (sometimes "informationtable.xml",
    sometimes a random numeric ID like "53405.xml"). Strategy:
      1. List the filing directory.
      2. For each .xml file that isn't `primary_doc.xml`, fetch and check
         whether it contains <infoTable> — that's the holdings table.
    """
    accn_clean = accession.replace("-", "")
    try:
        ix = httpx.get(_INDEX_URL.format(cik=cik, accn_clean=accn_clean),
                       headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=15.0)
        ix.raise_for_status()
        items = (ix.json().get("directory") or {}).get("item") or []
    except Exception:
        return None

    # 1) explicit name match (rare but cheap to check)
    candidates: list[str] = []
    for it in items:
        name = it.get("name", "")
        lname = name.lower()
        if not lname.endswith(".xml"):
            continue
        if lname == "primary_doc.xml":
            continue
        # Prefer obvious matches first
        if "informationtable" in lname or "infotable" in lname:
            candidates.insert(0, name)
        else:
            candidates.append(name)

    base = _ARCHIVE_URL.format(cik=cik, accn_clean=accn_clean)
    for name in candidates:
        try:
            r = httpx.get(base + name, headers={"User-Agent": _UA}, timeout=20.0)
            if r.status_code != 200:
                continue
            text = r.text
            if "<infoTable" in text:
                return text
        except Exception:
            continue
    return None


# Strip XML namespaces helper — 13F XMLs use {http://www.sec.gov/edgar/document/thirteenf/informationtable}
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_information_table(xml: str) -> list[dict[str, Any]]:
    """Each <infoTable> entry has: nameOfIssuer, cusip, value (in $thousands),
    shrsOrPrnAmt/sshPrnamt, putCall (optional, options excluded for sanity)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out: list[dict[str, Any]] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "infoTable":
            continue
        rec: dict[str, Any] = {}
        for child in el:
            tag = _strip_ns(child.tag)
            if tag == "shrsOrPrnAmt":
                for sub in child:
                    if _strip_ns(sub.tag) == "sshPrnamt":
                        try:
                            rec["shares"] = int(float(sub.text or "0"))
                        except ValueError:
                            rec["shares"] = 0
            else:
                rec[tag] = (child.text or "").strip()
        if rec.get("putCall"):
            # skip option positions (calls/puts) — they distort weights
            continue
        try:
            # Modern 13F XML (FormatVersion >= 1.0) reports value in dollars
            # directly; older filings were in thousands. Detect by magnitude:
            # any single position > $1T is implausible → treat as thousands.
            v = float(rec.get("value", 0) or 0)
        except ValueError:
            v = 0.0
        out.append({
            "name_of_issuer": rec.get("nameOfIssuer") or "",
            "cusip": rec.get("cusip") or "",
            "value_usd_raw": v,
            "shares": rec.get("shares", 0),
        })

    if not out:
        return out

    # Detect units. SEC requires reporting positions ≥ $200K, so any filing
    # with a max position > $100M is definitely already in dollars; otherwise
    # the filing uses the legacy "thousands of dollars" convention.
    max_v = max(r["value_usd_raw"] for r in out)
    multiplier = 1.0 if max_v > 1e8 else 1000.0
    for r in out:
        r["value_usd"] = r["value_usd_raw"] * multiplier
        del r["value_usd_raw"]

    # Aggregate by CUSIP (filers split positions across subsidiaries).
    agg: dict[str, dict[str, Any]] = {}
    for r in out:
        key = r["cusip"] or r["name_of_issuer"]
        if key in agg:
            agg[key]["value_usd"] += r["value_usd"]
            try:
                agg[key]["shares"] = int(agg[key]["shares"]) + int(r["shares"] or 0)
            except (TypeError, ValueError):
                pass
        else:
            agg[key] = dict(r)
    return list(agg.values())


def _ticker_for_cusip(cusip: str) -> str | None:
    """Best-effort CUSIP→ticker resolution. Uses a local cache. Returns None
    when the CUSIP doesn't map cleanly (private placements, ADRs, etc.).

    For a production system we'd integrate the SEC's company_tickers.json
    inverse (it maps ticker→CIK but not CUSIP), or pay for a CUSIP feed.
    For now we use a small built-in map that covers most common holdings.
    """
    # Built-in CUSIP→ticker map. Covers most large-cap US holdings that appear
    # in famous investor portfolios. Augment by editing this dict.
    BUILTIN = {
        # Tech
        "037833100": "AAPL", "594918104": "MSFT", "67066G104": "NVDA",
        "02079K305": "GOOGL", "02079K107": "GOOG", "023135106": "AMZN",
        "30303M102": "META", "88160R101": "TSLA", "11135F101": "AVGO",
        "79466L302": "CRM", "00724F101": "ADBE", "459200101": "IBM",
        "747525103": "QCOM", "458140100": "INTC", "037833AT7": "AAPL",
        "172967424": "CSCO", "20825C104": "CRWD", "98980L101": "ZS",
        # Financials
        "025816109": "AXP", "060505104": "BAC", "46625H100": "JPM",
        "92826C839": "V", "57636Q104": "MA", "84265V105": "SPGI",
        "172967424": "CSCO", "G0822V134": "CB", "166764100": "CVX",
        "615369105": "MCO", "172967424": "CSCO",
        # Consumer
        "191216100": "KO", "713448108": "PEP", "931142103": "WMT",
        "742718109": "PG", "478160104": "JNJ", "002824100": "ABT",
        "30231G102": "XOM", "67103H107": "OXY", "500754106": "KHC",
        "92826C839": "V", "1101221001": "DIS", "654106103": "NKE",
        # Healthcare
        "532457108": "LLY", "00287Y109": "ABBV", "92343V104": "VZ",
        "913017109": "UNH", "58933Y105": "MRK", "717081103": "PFE",
        # Industrials / Materials
        "532716107": "LIN", "256677105": "DE", "G3856J109": "ACN",
        # International (Berkshire holds some)
        "G42045107": "MC.PA",
        # Banks etc.
        "46625H100": "JPM", "060505104": "BAC", "172967424": "CSCO",
    }
    if cusip in BUILTIN:
        return BUILTIN[cusip]
    return None


def fetch_fund_holdings(cik: int, limit: int = 20) -> dict[str, Any]:
    """Most recent 13F holdings for a fund, normalized to {symbol, value_usd, weight_pct}."""
    cache_key = f"fund13f:{cik}"
    if (cached := _cache_get(cache_key)) is not None:
        return cached
    meta = _most_recent_13f(cik)
    if not meta:
        return {"filing_period": None, "filed": None, "holdings": [], "n_total": 0}
    xml = _fetch_information_table(cik, meta["accession"])
    if not xml:
        return {"filing_period": meta.get("period_end"), "filed": meta.get("filed"), "holdings": [], "n_total": 0}
    raw = _parse_information_table(xml)
    total_value = sum(r["value_usd"] for r in raw) or 1.0
    holdings: list[dict[str, Any]] = []
    for r in raw:
        sym = _ticker_for_cusip(r["cusip"])
        holdings.append({
            "name_of_issuer": r["name_of_issuer"],
            "cusip": r["cusip"],
            "symbol": sym,
            "value_usd": r["value_usd"],
            "shares": r["shares"],
            "weight_pct": round(r["value_usd"] / total_value * 100, 2),
        })
    # sort by weight, take top N
    holdings.sort(key=lambda h: -h["value_usd"])
    out = {
        "filing_period": meta.get("period_end"),
        "filed": meta.get("filed"),
        "holdings": holdings[:limit],
        "n_total": len(raw),
        "portfolio_value_usd": total_value,
    }
    _cache_set(cache_key, out, ttl=6 * 3600)
    return out


def famous_investors_live() -> list[dict[str, Any]]:
    """Return famous investors with their LIVE 13F holdings."""
    out: list[dict[str, Any]] = []
    for inv in FUND_REGISTRY:
        try:
            data = fetch_fund_holdings(inv["cik"], limit=15)
        except Exception:
            data = {"holdings": [], "filing_period": None}
        out.append({
            "name": inv["name"],
            "fund": inv["fund"],
            "cik": inv["cik"],
            "philosophy": inv.get("philosophy"),
            "filing_period": data.get("filing_period"),
            "filed": data.get("filed"),
            "portfolio_value_usd": data.get("portfolio_value_usd"),
            "n_total_holdings": data.get("n_total"),
            "top_holdings": data.get("holdings", []),
        })
    return out


def famous_investors_for_symbol_live(symbol: str) -> list[dict[str, Any]]:
    """Which famous investors currently disclose holding this symbol?"""
    sym = symbol.upper()
    out: list[dict[str, Any]] = []
    for inv in FUND_REGISTRY:
        try:
            data = fetch_fund_holdings(inv["cik"], limit=50)
            for h in data.get("holdings", []):
                if h.get("symbol") == sym:
                    out.append({
                        "investor": inv["name"],
                        "fund": inv["fund"],
                        "filing_period": data.get("filing_period"),
                        "symbol": sym,
                        "weight_pct": h.get("weight_pct"),
                        "shares": h.get("shares"),
                        "value_usd": h.get("value_usd"),
                    })
                    break
        except Exception:
            continue
    out.sort(key=lambda r: -(r.get("weight_pct") or 0))
    return out
