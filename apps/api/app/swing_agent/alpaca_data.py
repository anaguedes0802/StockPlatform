"""Alpaca market-data client for the swing agent: bars, calendar, corporate actions.

Why Alpaca (details in SWING_AGENT.md § Phase 1):

* The free plan serves the **consolidated SIP tape** historically, as long as
  the request ends more than 15 minutes ago. Intraday history starts on
  2016-01-04.
* Delisted symbols keep their history (SIVB, FRC, TWTR, CELG, ATVI were
  checked), which is what makes a point-in-time universe possible.
* `asof` maps a symbol to the entity that carried it on that date, including
  renames (FB asof 2019 → the same series as META today).
* The paper-trading API is the same vendor, so backtest and live read
  identical bars.

The existing `app.services.alpaca_bars` adapter targets the app's charts (IEX
feed, range strings). This one is for research: explicit start/end, feed,
adjustment and asof, cross-symbol pagination, rate limiting and retries.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import httpx
import pandas as pd

from app.config import settings
from app.core.logging import log

DATA_URL = "https://data.alpaca.markets"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_COLS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
         "n": "trades", "vw": "vwap"}


class AlpacaDataError(RuntimeError):
    pass


class AlpacaData:
    def __init__(self, *, feed: str = "sip", max_rpm: int = 180, timeout: float = 40.0,
                 max_retries: int = 6, client: httpx.Client | None = None) -> None:
        if not (settings.alpaca_api_key and settings.alpaca_api_secret):
            raise AlpacaDataError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")
        self.feed = feed
        self._min_gap = 60.0 / max(1, max_rpm)
        self._last = 0.0
        self._retries = max_retries
        self.requests = 0
        self._http = client or httpx.Client(timeout=timeout, headers={
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        })
        tk = settings.alpaca_trading_api_key or settings.alpaca_api_key
        ts = settings.alpaca_trading_api_secret or settings.alpaca_api_secret
        self._trading_headers = {"APCA-API-KEY-ID": tk, "APCA-API-SECRET-KEY": ts}

    # -- transport ---------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        """GET with a client-side rate limit and exponential backoff.

        Retries 429/5xx and network errors; anything else (bad symbol, 403 for
        a feed the plan doesn't cover) raises immediately.
        """
        delay = 2.0
        for attempt in range(self._retries + 1):
            wait = self._min_gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                r = self._http.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                err: str = f"network: {e}"
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in _RETRY_STATUS:
                    raise AlpacaDataError(f"HTTP {r.status_code} {url}: {r.text[:200]}")
                err = f"HTTP {r.status_code}"
                ra = r.headers.get("retry-after")
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
            if attempt == self._retries:
                raise AlpacaDataError(f"giving up after {attempt + 1} tries: {err} {url}")
            log.warning("alpaca_retry", url=url, err=err, attempt=attempt + 1, sleep_s=delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise AssertionError("unreachable")

    # -- bars --------------------------------------------------------------

    def bars(self, symbols: Iterable[str], timeframe: str, start: str | pd.Timestamp,
             end: str | pd.Timestamp, *, adjustment: str = "split", asof: str | None = None,
             page_limit: int = 10000) -> dict[str, pd.DataFrame]:
        """Bars for several symbols, all pages. Index = bar START time (UTC).

        `adjustment`: raw | split | dividend | all. `asof`: YYYY-MM-DD symbol
        mapping date (None = today's mapping).
        """
        syms = sorted({s.upper() for s in symbols})
        params: dict[str, Any] = {
            "symbols": ",".join(syms), "timeframe": timeframe,
            "start": _iso(start), "end": _iso(end), "feed": self.feed,
            "adjustment": adjustment, "limit": page_limit, "sort": "asc",
        }
        if asof:
            params["asof"] = asof
        chunks: dict[str, list[dict[str, Any]]] = {}
        token: str | None = None
        while True:
            if token:
                params["page_token"] = token
            js = self._get(f"{DATA_URL}/v2/stocks/bars", params)
            for sym, rows in (js.get("bars") or {}).items():
                chunks.setdefault(sym, []).extend(rows)
            token = js.get("next_page_token")
            if not token:
                break
        return {s: bars_frame(rows) for s, rows in chunks.items() if rows}

    # -- reference data ----------------------------------------------------

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        base = settings.alpaca_trading_base_url.rstrip("/")
        return self._get(f"{base}/v2/calendar", {"start": start, "end": end},
                         headers=self._trading_headers)

    def corporate_actions(self, symbols: Iterable[str], start: str, end: str,
                          types: str = "cash_dividend,forward_split,reverse_split") -> dict[str, list]:
        params: dict[str, Any] = {"symbols": ",".join(sorted(set(symbols))), "types": types,
                                  "start": start, "end": end, "limit": 1000}
        out: dict[str, list] = {}
        token = None
        while True:
            if token:
                params["page_token"] = token
            js = self._get(f"{DATA_URL}/v1/corporate-actions", params)
            for kind, rows in (js.get("corporate_actions") or {}).items():
                out.setdefault(kind, []).extend(rows)
            token = js.get("next_page_token")
            if not token:
                return out


def bars_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=_COLS)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), utc=True), name="ts")
    for c in _COLS.values():
        if c not in df:
            df[c] = float("nan")
    df = df[list(_COLS.values())].astype("float64")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _iso(x: str | pd.Timestamp) -> str:
    if isinstance(x, str):
        return x
    ts = pd.Timestamp(x)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
