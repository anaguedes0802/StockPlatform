"""SEC EDGAR XBRL adapter — real historical, real filing dates.

Uses two free public endpoints (no key, just identifying User-Agent required):

  1. https://www.sec.gov/files/company_tickers.json
     — full ticker → CIK map, refreshed weekly by SEC.

  2. https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
     — every XBRL fact ever filed by the company, with the real filing date.

Returns a normalized quarterly fundamentals dataframe with proper filing dates
(no 45-day-lag heuristic — we know exactly when each filing was public).

Reference: https://www.sec.gov/edgar/sec-api-documentation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd
import redis

from app.config import settings
from app.core.logging import log


# SEC requires a contactable User-Agent. Generic-but-identifying string is fine.
_UA = "StockPlatform/0.1 contact@example.com"

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


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


# ----------------------------------------------------------------------------
# Ticker → CIK
# ----------------------------------------------------------------------------

_TICKER_TO_CIK: dict[str, int] | None = None


def _load_ticker_map() -> dict[str, int]:
    """Cache the SEC ticker map for 7 days."""
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is not None:
        return _TICKER_TO_CIK
    cached = _cache_get("edgar_ticker_map_v1")
    if cached:
        _TICKER_TO_CIK = {k: int(v) for k, v in cached.items()}
        return _TICKER_TO_CIK
    try:
        r = httpx.get(_TICKER_MAP_URL, headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=20.0)
        r.raise_for_status()
        data = r.json()
        # The JSON is keyed by integer strings — the values are {cik_str, ticker, title}.
        m = {}
        for v in data.values():
            t = (v.get("ticker") or "").upper()
            try:
                m[t] = int(v["cik_str"])
            except (KeyError, ValueError, TypeError):
                continue
        _TICKER_TO_CIK = m
        _cache_set("edgar_ticker_map_v1", m, ttl=7 * 24 * 3600)
        return m
    except Exception:
        _TICKER_TO_CIK = {}
        return _TICKER_TO_CIK


def cik_for(symbol: str) -> int | None:
    return _load_ticker_map().get(symbol.upper())


# ----------------------------------------------------------------------------
# Companyfacts → quarterly fundamentals
# ----------------------------------------------------------------------------

# Concept tag candidates per metric — XBRL tags vary by filer/era.
_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_debt": [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    ],
    # Short-term / current debt — added to long-term debt so total_debt (and
    # debt_to_assets / leverage) reflect the FULL debt load, not just the
    # long-term portion. Several alternate tags cover the same economic concept.
    "short_term_debt": [
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",  # current portion of long-term debt
    ],
}


def _fetch_companyfacts(cik: int) -> dict[str, Any] | None:
    key = f"edgar_facts:{cik}"
    if (cached := _cache_get(key)) is not None:
        return cached
    url = _COMPANYFACTS_URL.format(cik=cik)
    try:
        r = httpx.get(url, headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=30.0)
        if r.status_code == 404:
            log.info("edgar_companyfacts_404", cik=cik)
            _cache_set(key, {}, ttl=24 * 3600)
            return {}
        r.raise_for_status()
        data = r.json()
        _cache_set(key, data, ttl=24 * 3600)
        return data
    except Exception as e:
        log.warning("edgar_companyfacts_failed", cik=cik, err=str(e))
        return None


def _pick_concept(facts: dict[str, Any], candidates: list[str]) -> list[dict] | None:
    """Merge ALL candidate tags' fact lists into one chronological list.

    Reason: companies migrate XBRL tags over time (e.g., AAPL moved from
    `Revenues` to `RevenueFromContractWithCustomerExcludingAssessedTax` after
    ASC 606 in 2018). Picking the first tag with data leaves a hole — we want
    the union sorted by period_end, with later tags overriding earlier ones on
    overlapping periods.
    """
    us_gaap = ((facts.get("facts") or {}).get("us-gaap") or {})
    merged: list[dict] = []
    seen_periods: set[tuple[str, str]] = set()
    # iterate in REVERSE so later (more-modern) tags take precedence on duplicates
    for tag in reversed(candidates):
        node = us_gaap.get(tag)
        if not node:
            continue
        units = node.get("units") or {}
        arr: list[dict] | None = None
        for u in ("USD", "USD/shares", "shares", "USDperShares", "pure"):
            if u in units:
                arr = units[u]; break
        if arr is None:
            for _, a in units.items():
                if a: arr = a; break
        for entry in arr or []:
            k = (str(entry.get("start") or ""), str(entry.get("end") or ""))
            if k in seen_periods:
                continue
            seen_periods.add(k)
            merged.append(entry)
    return merged or None


def _to_dataframe(facts_node: list[dict]) -> pd.DataFrame:
    """Each entry: {start, end, val, accn, fy, fp, form, filed, frame, ...}"""
    rows = []
    for e in facts_node or []:
        rows.append({
            "period_end": e.get("end"),
            "period_start": e.get("start"),
            "value": e.get("val"),
            "fp": e.get("fp"),            # FY, Q1, Q2, Q3
            "form": e.get("form"),         # 10-Q, 10-K, ...
            "filed": e.get("filed"),       # actual filing date (yyyy-mm-dd)
            "frame": e.get("frame"),       # CY2024Q3 etc.
        })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], utc=True, errors="coerce")
    df = df.dropna(subset=["period_end", "filed", "value"])
    df = df.sort_values(["period_end", "filed"])
    return df


def _derive_missing_q4(quarters: pd.DataFrame, annuals: pd.DataFrame) -> pd.DataFrame:
    """Derive the missing discrete quarter (usually Q4) as FY minus the sum of
    the three reported interim quarters of the same fiscal year.

    Rationale: Q4 flow facts (revenue / net income / EPS) are normally reported
    ONLY inside the full-year 10-K (a ~365-day period), never as a standalone
    ~90-day fact. Dropping them (because they fail the 80-100 day span filter)
    distorts TTM (rolling-4) sums and YoY growth. When we have the annual total
    plus exactly the three interim quarters that fall within that fiscal year,
    Q4 = FY - (Q1 + Q2 + Q3). The synthesized Q4's filing date is the annual
    (10-K) filing date — the moment Q4 actually became publicly knowable, which
    preserves the PIT invariant.

    Inputs/outputs carry columns: period_end, filed, value, period_start.
    """
    if annuals.empty:
        return quarters
    derived: list[dict] = []
    q = quarters.sort_values("period_end")
    for _, a in annuals.iterrows():
        a_end = a["period_end"]
        a_start = a["period_start"]
        if pd.isna(a_start) or pd.isna(a_end):
            continue
        # Interim quarters whose period falls within this fiscal year window.
        in_fy = q[(q["period_start"] >= a_start - pd.Timedelta(days=10)) &
                  (q["period_end"] <= a_end + pd.Timedelta(days=10))]
        # Need exactly three interim quarters to infer the fourth.
        if len(in_fy) != 3:
            continue
        # The missing quarter ends at the fiscal year end; skip if we already
        # have a discrete quarter ending there.
        if (q["period_end"] == a_end).any():
            continue
        q4_value = a["value"] - in_fy["value"].sum()
        last_q_end = in_fy["period_end"].max()
        derived.append({
            "period_end": a_end,
            "period_start": last_q_end + pd.Timedelta(days=1),
            "value": q4_value,
            # Q4 only becomes known when the 10-K is filed.
            "filed": a["filed"],
        })
    if not derived:
        return quarters
    add = pd.DataFrame(derived)
    return pd.concat([quarters, add], ignore_index=True)


def _quarterize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reduce to one row per quarter. SEC reports income statement items as
    period totals (Q3 in a 10-Q can be 9-month YTD); for balance-sheet items
    they're as-of values. We use a simple rule:

    - 'flow' (revenue, NI, EPS): keep ~90-day discrete quarters, then synthesize
      the missing Q4 (reported only in the full-year 10-K) as FY minus the three
      interim quarters — see _derive_missing_q4.
    - 'stock' (assets, debt): take the latest filing per period_end as-of value
    """
    if df.empty: return df

    if kind == "flow":
        if "period_start" in df.columns:
            df = df.copy()
            df["period_start"] = pd.to_datetime(df["period_start"], utc=True, errors="coerce")
            df["span_days"] = (df["period_end"] - df["period_start"]).dt.days
            # ~90-day discrete quarters (allow 80-100 days for fiscal variation).
            quarters = df[(df["span_days"] >= 80) & (df["span_days"] <= 100)].copy()
            # ~365-day annual (full-year) figures — these contain the Q4 flow.
            annuals = df[(df["span_days"] >= 350) & (df["span_days"] <= 380)].copy()
            # For each (annual/quarter) period_end, keep the earliest filing
            # before deriving Q4 so sums use first-reported values.
            quarters = quarters.sort_values("filed").drop_duplicates(subset=["period_end"], keep="first")
            annuals = annuals.sort_values("filed").drop_duplicates(subset=["period_end"], keep="first")
            df = _derive_missing_q4(quarters, annuals)
    # Within a period_end, the latest `filed` is the most recent restatement;
    # but for PIT correctness we want the FIRST filing — the value that was
    # public when first reported. Keep the earliest `filed` per period_end.
    df = df.sort_values("filed").drop_duplicates(subset=["period_end"], keep="first")
    return df[["period_end", "filed", "value"]].sort_values("period_end").reset_index(drop=True)


def quarterly_fundamentals(symbol: str) -> pd.DataFrame:
    """Return a quarterly fundamentals dataframe with real filing dates.

    Columns:
      period_end (UTC), available_at (UTC, = filed date), revenue, net_income,
      eps (diluted preferred, else basic), total_assets, total_debt,
      revenue_growth_yoy, eps_growth_yoy, net_margin, debt_to_assets, eps_ttm.
    """
    cik = cik_for(symbol)
    if cik is None:
        return pd.DataFrame()
    facts = _fetch_companyfacts(cik)
    if not facts:
        return pd.DataFrame()

    rev = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["revenue"]) or []), kind="flow") \
            .rename(columns={"value": "revenue"})
    ni = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["net_income"]) or []), kind="flow") \
            .rename(columns={"value": "net_income"})
    eps_d = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["eps_diluted"]) or []), kind="flow") \
            .rename(columns={"value": "eps"})
    eps_b = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["eps_basic"]) or []), kind="flow") \
            .rename(columns={"value": "eps_basic"})
    assets = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["total_assets"]) or []), kind="stock") \
            .rename(columns={"value": "total_assets"})
    debt = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["total_debt"]) or []), kind="stock") \
            .rename(columns={"value": "total_debt"})
    st_debt = _quarterize(_to_dataframe(_pick_concept(facts, _TAGS["short_term_debt"]) or []), kind="stock") \
            .rename(columns={"value": "short_term_debt"})

    if rev.empty and ni.empty:
        return pd.DataFrame()

    # Build a unified frame keyed by period_end. The earliest available_at
    # across the metrics for that period is the date everything becomes "known."
    frames = [(rev, ["revenue"]),
              (ni,  ["net_income"]),
              (eps_d, ["eps"]),
              (eps_b, ["eps_basic"]),
              (assets, ["total_assets"]),
              (debt, ["total_debt"]),
              (st_debt, ["short_term_debt"])]
    base = None
    for df, cols in frames:
        if df.empty: continue
        keep = df.rename(columns={"filed": f"filed_{cols[0]}"})[["period_end", f"filed_{cols[0]}", *cols]]
        base = keep if base is None else base.merge(keep, on="period_end", how="outer")
    if base is None or base.empty:
        return pd.DataFrame()

    # available_at = the *latest* filing date among the metrics that are ACTUALLY
    # known (non-null) for that period. You only "know" a row's full set of
    # fundamentals once the last contributing filing is public.
    #
    # PIT invariant: available_at must be >= the filing date of every non-null
    # fundamental on the row. We pair each metric's value column with its
    # filed_<metric> column and mask out filing dates whose value is NaN, so a
    # metric that happens to be absent for this period cannot shrink available_at
    # below a metric that IS present. pandas' max(skipna=True) ignores the
    # remaining NaT entries and yields NaT only when the whole row is empty
    # (handled below) — it does not crash on an all-NaN row.
    filed_cols = [c for c in base.columns if c.startswith("filed_")]
    masked_filed = pd.DataFrame(index=base.index)
    for fcol in filed_cols:
        metric = fcol[len("filed_"):]
        fseries = pd.to_datetime(base[fcol], utc=True, errors="coerce")
        if metric in base.columns:
            # Only count this filing date when its corresponding value is present.
            value_present = base[metric].notna()
            fseries = fseries.where(value_present)
        masked_filed[fcol] = fseries
    base["available_at"] = masked_filed.max(axis=1, skipna=True)
    # Guard: if a row had no known metric at all, fall back to the latest raw
    # filing date so available_at is never NaT (avoids lookahead via dropped row).
    if filed_cols:
        raw_max = base[filed_cols].apply(lambda s: pd.to_datetime(s, utc=True, errors="coerce")).max(axis=1, skipna=True)
        base["available_at"] = base["available_at"].fillna(raw_max)
    base = base.drop(columns=filed_cols).sort_values("period_end").reset_index(drop=True)

    # Prefer diluted EPS; fill missing with basic
    if "eps" in base.columns and "eps_basic" in base.columns:
        base["eps"] = base["eps"].fillna(base["eps_basic"])
    elif "eps_basic" in base.columns and "eps" not in base.columns:
        base["eps"] = base["eps_basic"]
    base = base.drop(columns=[c for c in ["eps_basic"] if c in base.columns], errors="ignore")

    # Derived columns
    base["revenue"] = pd.to_numeric(base.get("revenue"), errors="coerce")
    base["net_income"] = pd.to_numeric(base.get("net_income"), errors="coerce")
    base["eps"] = pd.to_numeric(base.get("eps"), errors="coerce")
    base["total_assets"] = pd.to_numeric(base.get("total_assets"), errors="coerce")
    base["total_debt"] = pd.to_numeric(base.get("total_debt"), errors="coerce")
    base["short_term_debt"] = pd.to_numeric(base.get("short_term_debt"), errors="coerce")

    # Total debt = long-term debt + short-term/current debt. Use fillna(0) on the
    # addends so a missing component doesn't null the sum, but keep total_debt
    # NaN when BOTH are absent (so debt_to_assets stays NaN rather than a bogus 0).
    _lt = base["total_debt"]
    _st = base["short_term_debt"]
    _combined = _lt.fillna(0) + _st.fillna(0)
    base["total_debt"] = _combined.where(_lt.notna() | _st.notna())
    base = base.drop(columns=["short_term_debt"], errors="ignore")

    base["revenue_growth_yoy"] = base["revenue"].pct_change(4, fill_method=None)
    base["eps_growth_yoy"] = base["eps"].pct_change(4, fill_method=None)
    base["net_margin"] = base["net_income"] / base["revenue"].replace(0, np.nan)
    base["debt_to_assets"] = base["total_debt"] / base["total_assets"].replace(0, np.nan)
    base["eps_ttm"] = base["eps"].rolling(4, min_periods=4).sum()

    return base[[
        "period_end", "available_at",
        "revenue", "net_income", "eps",
        "total_assets", "total_debt",
        "revenue_growth_yoy", "eps_growth_yoy",
        "net_margin", "debt_to_assets", "eps_ttm",
    ]]


def is_available_for(symbol: str) -> bool:
    return cik_for(symbol) is not None
