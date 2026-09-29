"""Point-in-time fundamentals for backtest-safe training.

Primary source: **SEC EDGAR XBRL companyfacts** (app/services/edgar.py).
  Real filing dates, 17+ years of history, free, no key.

Fallback: **yfinance quarterly_financials**, with a conservative *form-aware*
  filing-lag heuristic (~90d for annual/10-K rows, ~45d for quarterly/10-Q rows).
  Used when EDGAR doesn't know the CIK (non-US symbols, ETFs).

Features per quarter (keyed by the actual SEC filing date for EDGAR rows,
or the form-aware lag heuristic for yfinance rows):
  - fund_eps_growth_yoy
  - fund_revenue_growth_yoy
  - fund_net_margin (NI / revenue)
  - fund_debt_to_assets
  - fund_eps_ttm (sum of trailing 4 quarter EPS)

These forward-fill across trading days so every bar sees the most recent
publicly-known fundamentals.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import redis

from app.config import settings
from app.services.market_data import _yf_ticker  # type: ignore


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


# Common alternate names yfinance uses across versions
_REVENUE_FIELDS = ["Total Revenue", "TotalRevenue", "Revenue"]
_NET_INCOME_FIELDS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]
_EPS_FIELDS = ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"]
_ASSETS_FIELDS = ["Total Assets", "TotalAssets"]
_DEBT_FIELDS = ["Total Debt", "TotalDebt", "Long Term Debt"]


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return None


def _quarterly_features(symbol: str) -> pd.DataFrame:
    """Pull quarterly fundamentals and compute features keyed by period_end."""
    t = _yf_ticker(symbol)
    try:
        qf = t.quarterly_financials
        qb = t.quarterly_balance_sheet
    except Exception:
        return pd.DataFrame()
    if qf is None or qf.empty:
        return pd.DataFrame()

    rev = _pick(qf, _REVENUE_FIELDS)
    ni = _pick(qf, _NET_INCOME_FIELDS)
    eps = _pick(qf, _EPS_FIELDS)
    assets = _pick(qb, _ASSETS_FIELDS) if qb is not None else None
    debt = _pick(qb, _DEBT_FIELDS) if qb is not None else None

    if rev is None and ni is None and eps is None:
        return pd.DataFrame()

    # yfinance returns columns indexed by period_end timestamps; transpose to rows.
    rec: list[dict[str, Any]] = []
    period_ends = sorted(set(qf.columns))
    for pe in period_ends:
        try:
            ts = pd.Timestamp(pe).tz_localize("UTC") if pd.Timestamp(pe).tz is None else pd.Timestamp(pe).tz_convert("UTC")
        except Exception:
            continue
        row: dict[str, Any] = {"period_end": ts}
        try:
            row["revenue"] = float(rev.get(pe)) if rev is not None else np.nan
        except Exception:
            row["revenue"] = np.nan
        try:
            row["net_income"] = float(ni.get(pe)) if ni is not None else np.nan
        except Exception:
            row["net_income"] = np.nan
        try:
            row["eps"] = float(eps.get(pe)) if eps is not None else np.nan
        except Exception:
            row["eps"] = np.nan
        try:
            row["total_assets"] = float(assets.get(pe)) if assets is not None else np.nan
        except Exception:
            row["total_assets"] = np.nan
        try:
            row["total_debt"] = float(debt.get(pe)) if debt is not None else np.nan
        except Exception:
            row["total_debt"] = np.nan
        rec.append(row)

    df = pd.DataFrame(rec).sort_values("period_end").reset_index(drop=True)
    if df.empty:
        return df

    # Derived: TTM revenue/net income (sum of trailing 4 quarters)
    df["revenue_ttm"] = df["revenue"].rolling(4, min_periods=4).sum()
    df["net_income_ttm"] = df["net_income"].rolling(4, min_periods=4).sum()
    df["eps_ttm"] = df["eps"].rolling(4, min_periods=4).sum()

    # YoY growth (quarter to same quarter 4 ago)
    df["revenue_growth_yoy"] = df["revenue"].pct_change(4)
    df["eps_growth_yoy"] = df["eps"].pct_change(4)

    df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)
    df["debt_to_assets"] = df["total_debt"] / df["total_assets"].replace(0, np.nan)

    # Form-aware filing lag (yfinance gives no real filing date, so we estimate).
    #
    # Real SEC filing deadlines differ by form:
    #   - 10-K (annual report)    ~60-90 days after fiscal year end.
    #   - 10-Q (quarterly report) ~40-45 days after quarter end.
    # A flat +45d under-lags annual reports and creates lookahead bias on the
    # Q4/full-year row. We detect annual vs quarterly by the gap between this
    # period_end and the previous one: a ~year gap (or the first/only row) means
    # an annual cadence, a ~quarter gap means quarterly cadence.
    _ANNUAL_LAG = pd.Timedelta(days=90)    # conservative 10-K deadline
    _QUARTERLY_LAG = pd.Timedelta(days=45)  # 10-Q deadline
    gap_days = df["period_end"].diff().dt.days
    # Annual when the spacing looks like ~a year (>200d). Unknown spacing
    # (first row / NaN) defaults to the conservative annual lag to avoid lookahead.
    is_annual = gap_days.isna() | (gap_days > 200)
    lag = is_annual.map(lambda a: _ANNUAL_LAG if a else _QUARTERLY_LAG)
    df["available_at"] = df["period_end"] + lag

    return df[[
        "period_end", "available_at",
        "revenue_ttm", "net_income_ttm", "eps_ttm",
        "revenue_growth_yoy", "eps_growth_yoy",
        "net_margin", "debt_to_assets",
    ]]


def _fetch_quarterly(symbol: str) -> pd.DataFrame:
    """Try SEC EDGAR first (real filing dates, 17+ year history); fall back to
    yfinance + 45-day lag heuristic.
    """
    from app.services import edgar as edgar_svc
    try:
        df = edgar_svc.quarterly_fundamentals(symbol)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return _quarterly_features(symbol)


def features_for_index(symbol: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a dataframe aligned to `target_index` with PIT fundamental features.

    Each row at time `t` reflects the most recent quarterly filing where
    `available_at <= t`. We prefer EDGAR's actual filing dates over yfinance's
    +45-day heuristic for accuracy on the "knowable on" boundary.
    """
    if len(target_index) == 0:
        return pd.DataFrame()

    sym = symbol.upper()
    cache_key = f"fund_pit_v2:{sym}"
    cached = _cache_get(cache_key)
    if cached:
        q = pd.DataFrame(cached)
        q["period_end"] = pd.to_datetime(q["period_end"], utc=True)
        q["available_at"] = pd.to_datetime(q["available_at"], utc=True)
    else:
        q = _fetch_quarterly(sym)
        if not q.empty:
            serial = q.copy()
            for col in ["period_end", "available_at"]:
                serial[col] = serial[col].astype(str)
            _cache_set(cache_key, serial.to_dict(orient="records"), ttl=12 * 3600)

    if q.empty:
        # Return zeros — never crash training
        return pd.DataFrame(
            {
                "fund_eps_growth_yoy":    np.zeros(len(target_index)),
                "fund_revenue_growth_yoy": np.zeros(len(target_index)),
                "fund_net_margin":        np.zeros(len(target_index)),
                "fund_debt_to_assets":    np.zeros(len(target_index)),
                "fund_eps_ttm":           np.zeros(len(target_index)),
            },
            index=target_index,
        )

    # asof-merge each target timestamp to the latest known quarter
    target_df = pd.DataFrame(index=target_index)
    target_df["__ts"] = target_index
    q_sorted = q.sort_values("available_at").reset_index(drop=True)
    merged = pd.merge_asof(
        target_df.sort_values("__ts"),
        q_sorted.rename(columns={"available_at": "__ts"}),
        on="__ts",
        direction="backward",
    ).set_index(target_df.index)

    out = pd.DataFrame(index=target_index)
    out["fund_eps_growth_yoy"] = merged["eps_growth_yoy"].fillna(0).clip(-2.0, 2.0)
    out["fund_revenue_growth_yoy"] = merged["revenue_growth_yoy"].fillna(0).clip(-2.0, 2.0)
    out["fund_net_margin"] = merged["net_margin"].fillna(0).clip(-1.0, 1.0)
    out["fund_debt_to_assets"] = merged["debt_to_assets"].fillna(0).clip(0.0, 2.0)
    out["fund_eps_ttm"] = merged["eps_ttm"].fillna(0)
    return out
