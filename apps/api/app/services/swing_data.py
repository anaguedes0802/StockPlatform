"""Daily bars + historical earnings dates for swing-trading backtests.

Why a separate loader instead of `market_data.get_history`:

* `get_history` prefers Alpaca's free IEX feed for US equities. IEX history only
  reaches back to ~2018, and its daily high/low come from IEX prints alone
  (a few % of consolidated volume), so ranges are *narrower* than the real tape.
  A stop-loss backtest on IEX bars therefore under-counts stop hits — an
  optimistic bias. Swing backtests need the consolidated tape and a long
  history that includes 2008, 2011, 2015, 2018, 2020 and 2022.
* Yahoo's public chart endpoint returns that (split + dividend adjusted via
  `adjclose`) back to each instrument's listing, for equities, ETFs and FX.

Earnings dates come from SEC EDGAR 8-K filings carrying **Item 2.02 (Results of
Operations)** — the exact, point-in-time dates companies released results.
That lets the backtest enforce "never hold through earnings" historically,
not just live.

Everything is cached on disk under `artifacts/swing_cache/` (gitignored) so a
backtest over 50 symbols only hits the network once per TTL.
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.core.logging import log

_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "swing_cache"
_BARS_TTL_S = 12 * 3600
_EARN_TTL_S = 7 * 24 * 3600

_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_FILE = "https://data.sec.gov/submissions/{name}"
_SEC_UA = "StockPlatform/0.1 contact@example.com"


# ---------------------------------------------------------------------------
# Asset classes
# ---------------------------------------------------------------------------

def asset_class(symbol: str) -> str:
    """'fx' for Yahoo FX pairs (EURUSD=X), 'crypto' for XXX-USD, else 'equity'."""
    s = symbol.upper()
    if s.endswith("=X"):
        return "fx"
    if s.endswith("-USD") or s.endswith("-USDT"):
        return "crypto"
    return "equity"


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_path(kind: str, key: str) -> Path:
    safe = key.replace("/", "_").replace("=", "_").replace("^", "_")
    return _CACHE_DIR / kind / f"{safe}.pkl"


def _cache_read(kind: str, key: str, ttl_s: int) -> Any | None:
    p = _cache_path(kind, key)
    try:
        if p.exists() and (time.time() - p.stat().st_mtime) < ttl_s:
            with p.open("rb") as f:
                return pickle.load(f)
    except Exception:  # noqa: BLE001 — a corrupt cache entry is just a miss
        return None
    return None


def _cache_write(kind: str, key: str, value: Any) -> None:
    p = _cache_path(kind, key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(value, f)
        tmp.replace(p)
    except Exception as e:  # noqa: BLE001 — caching is best-effort
        log.debug("swing_cache_write_failed", key=key, err=str(e))


# ---------------------------------------------------------------------------
# Yahoo chart → adjusted daily OHLCV
# ---------------------------------------------------------------------------

def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET JSON from Yahoo.

    Yahoo 429s plain HTTP clients; `curl_cffi` impersonates a browser TLS
    fingerprint and gets through. On macOS its wheel needs CoreFoundation
    already loaded, which importing `_scproxy` does. Falls back to httpx.
    """
    try:
        try:
            import _scproxy  # noqa: F401 — macOS: preload CoreFoundation for curl_cffi
        except ImportError:
            pass
        from curl_cffi import requests as cr  # type: ignore

        r = cr.get(url, params=params, impersonate="chrome", timeout=25)
        if r.status_code == 200:
            return r.json()
        raise RuntimeError(f"yahoo HTTP {r.status_code}")
    except ImportError:
        r = httpx.get(url, params=params, timeout=25,
                      headers={"User-Agent": "Mozilla/5.0 StockPlatform/0.1"})
        r.raise_for_status()
        return r.json()


def _parse_chart(payload: dict[str, Any]) -> pd.DataFrame:
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        return pd.DataFrame()
    ts = pd.to_datetime(np.asarray(result["timestamp"], dtype="int64"), unit="s", utc=True)
    q = (result.get("indicators") or {}).get("quote", [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
    }, index=ts, dtype="float64")
    if adj is not None:
        # Yahoo's OHLC are split-adjusted; adjclose also folds in dividends.
        # Scale O/H/L by the same factor (what yfinance's auto_adjust does) so
        # total-return prices are internally consistent bar by bar.
        factor = pd.Series(adj, index=ts, dtype="float64") / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * factor
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    df["volume"] = df["volume"].fillna(0.0)
    # One row per session date. Normalise to midnight UTC of the session date
    # (exchange-local), so equities and FX share a calendar key.
    tz = (result.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    local_dates = df.index.tz_convert(tz).normalize().tz_localize(None)
    df.index = pd.DatetimeIndex(local_dates).tz_localize("UTC")
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def daily_bars(symbol: str, *, start: str | None = None, refresh: bool = False,
               max_age_s: int = _BARS_TTL_S) -> pd.DataFrame:
    """Adjusted daily OHLCV for `symbol` from listing (or `start`) to today.

    Index: tz-aware UTC midnight of each session date. Columns: open, high,
    low, close, volume. Empty DataFrame when the symbol can't be fetched.
    `max_age_s` bounds how stale the disk cache may be (backtests tolerate
    12h; live decisions ask for ~30 min).
    """
    sym = symbol.upper()
    df = None if refresh else _cache_read("bars", sym, max_age_s)
    if df is None:
        try:
            payload = _http_get_json(
                _YAHOO_CHART.format(symbol=sym),
                {"period1": 0, "period2": int(time.time()), "interval": "1d",
                 "events": "div,split", "includeAdjustedClose": "true"},
            )
            df = _parse_chart(payload)
        except Exception as e:  # noqa: BLE001
            log.warning("swing_bars_fetch_failed", symbol=sym, err=str(e))
            df = pd.DataFrame()
        if not df.empty:
            _cache_write("bars", sym, df)
    if not df.empty and asset_class(sym) == "fx":
        df, n_fixed = clean_fx_bars(df)
        df.attrs["repaired_bars"] = n_fixed
    if start and not df.empty:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    return df


def clean_fx_bars(df: pd.DataFrame, *, k_wick: float = 6.0, k_real: float = 4.0,
                  k_spike: float = 8.0) -> tuple[pd.DataFrame, int]:
    """Repair obvious bad prints in Yahoo's indicative FX bars.

    Yahoo FX dailies contain isolated garbage (EURUSD at 1.49 for one day in
    Dec-2008, a 0.0066 EURGBP low, a 0.979 EURGBP close in Oct-2022) that would
    trigger stops at prices nobody could trade. Scale = rolling median
    |close-to-close| move over the prior 63 bars.

    1. **Spike bars**: the close jumps > `k_spike`×scale and the *next* close
       gives back ≥ 75% of it → the whole bar is replaced by an interpolated
       one. This is a centred data-repair filter (it looks one bar ahead), which
       is acceptable only because genuine shocks do not fully revert overnight:
       SNB Jan-2015, Brexit and Oct-2008 all persist and are kept.
    2. **Bad opens**: open > `k_wick`×scale from the prior close while the close
       did not move > `k_real`×scale → open = prior close.
    3. **Bad wicks**: a wick > `k_wick`×scale beyond the open/close body, unless
       the close moved > `k_real`×scale *and* the wick is at most 3× that move
       (a real volatile day) → clipped to one scale beyond the body.

    Returns (clean_df, n_bars_repaired).
    """
    if df.empty or len(df) < 20:
        return df, 0
    d = df.copy()
    c = d["close"]
    scale = c.diff().abs().rolling(63, min_periods=10).median().shift(1)

    prev_c, next_c = c.shift(1), c.shift(-1)
    jump = c - prev_c
    spike = (jump.abs() > k_spike * scale) & ((next_c - prev_c).abs() < 0.25 * jump.abs())
    spike &= prev_c.notna() & next_c.notna()
    if spike.any():
        mid = (prev_c + next_c) / 2
        d.loc[spike, "open"] = prev_c[spike]
        d.loc[spike, "close"] = mid[spike]
        d.loc[spike, "high"] = np.maximum(prev_c, mid)[spike]
        d.loc[spike, "low"] = np.minimum(prev_c, mid)[spike]

    prev_c = d["close"].shift(1)
    move = (d["close"] - prev_c).abs()
    real = move > k_real * scale
    bad_open = ((d["open"] - prev_c).abs() > k_wick * scale) & ~real & prev_c.notna()
    d.loc[bad_open, "open"] = prev_c[bad_open]
    body_hi = d[["open", "close"]].max(axis=1)
    body_lo = d[["open", "close"]].min(axis=1)
    up_w, dn_w = d["high"] - body_hi, body_lo - d["low"]
    bad_hi = (up_w > k_wick * scale) & ~(real & (up_w <= 3 * move))
    bad_lo = (dn_w > k_wick * scale) & ~(real & (dn_w <= 3 * move))
    d.loc[bad_hi, "high"] = (body_hi + scale)[bad_hi]
    d.loc[bad_lo, "low"] = (body_lo - scale)[bad_lo]
    d["high"] = d[["high", "open", "close"]].max(axis=1)
    d["low"] = d[["low", "open", "close"]].min(axis=1)
    n = int((spike | bad_open | bad_hi | bad_lo).sum())
    return d, n


def intraday_bars(symbol: str, start: datetime, end: datetime | None = None,
                  interval: str = "5m") -> pd.DataFrame:
    """Regular-session intraday bars (bar-start timestamps, UTC) from Yahoo.

    Used by the simulated paper broker to decide whether a resting stop or
    target was touched *after* a fill. Yahoo serves 5-minute bars for the
    last ~60 days. Not cached: callers ask for small, fresh windows.
    """
    end = end or datetime.now(timezone.utc)
    try:
        payload = _http_get_json(
            _YAHOO_CHART.format(symbol=symbol.upper()),
            {"period1": int(start.timestamp()), "period2": int(end.timestamp()),
             "interval": interval, "includePrePost": "false"},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("intraday_bars_failed", symbol=symbol, err=str(e)[:200])
        return pd.DataFrame()
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        return pd.DataFrame()
    q = (result.get("indicators") or {}).get("quote", [{}])[0]
    idx = pd.to_datetime(np.asarray(result["timestamp"], dtype="int64"), unit="s", utc=True)
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close")}, index=idx, dtype="float64")
    df = df.dropna()
    return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


def load_universe(symbols: list[str], *, start: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch bars for many symbols. Returns (data, errors)."""
    data: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for s in symbols:
        df = daily_bars(s, start=start)
        if df.empty:
            errors[s.upper()] = "no data"
        else:
            data[s.upper()] = df
    return data, errors


# ---------------------------------------------------------------------------
# SEC EDGAR 8-K Item 2.02 → historical earnings-release dates
# ---------------------------------------------------------------------------

def _edgar_json(url: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(url, headers={"User-Agent": _SEC_UA, "Accept": "application/json"}, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.debug("edgar_submissions_failed", url=url, err=str(e))
        return None


def _item_202_dates(block: dict[str, Any]) -> list[str]:
    forms = block.get("form") or []
    dates = block.get("filingDate") or []
    items = block.get("items") or []
    out = []
    for f, d, it in zip(forms, dates, items):
        if str(f).startswith("8-K") and "2.02" in str(it or ""):
            out.append(str(d))
    return out


def earnings_dates(symbol: str) -> list[pd.Timestamp] | None:
    """Historical earnings-release dates (8-K Item 2.02 filing dates), ascending.

    Returns None when the symbol has no SEC CIK (ETFs, FX, foreign filers) —
    callers must treat None as "no earnings data", which is different from
    "no earnings". Cached on disk for a week.
    """
    if asset_class(symbol) != "equity":
        return None
    sym = symbol.upper()
    cached = _cache_read("earnings", sym, _EARN_TTL_S)
    if cached is not None:
        return cached or None
    from app.services import edgar  # lazy: pulls the SEC ticker map

    cik = edgar.cik_for(sym)
    if not cik:
        _cache_write("earnings", sym, [])
        return None
    sub = _edgar_json(_EDGAR_SUBMISSIONS.format(cik=cik))
    if not sub:
        return None  # transient: don't cache a failure
    filings = sub.get("filings") or {}
    dates = _item_202_dates(filings.get("recent") or {})
    for f in filings.get("files") or []:
        # Older pages. Item codes on 8-Ks exist from Aug-2004 onward.
        if str(f.get("filingTo", "")) < "2004-08-01":
            continue
        page = _edgar_json(_EDGAR_FILE.format(name=f.get("name")))
        if page:
            dates.extend(_item_202_dates(page))
    out = sorted({pd.Timestamp(d, tz="UTC") for d in dates})
    _cache_write("earnings", sym, out)
    return out or None


def earnings_for(symbols: list[str]) -> dict[str, list[pd.Timestamp] | None]:
    return {s.upper(): earnings_dates(s) for s in symbols}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
