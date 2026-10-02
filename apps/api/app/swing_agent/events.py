"""Event calendars the agent must avoid: earnings and macro releases.

* **Earnings, primary: SEC 8-K Item 2.02** ("Results of Operations"). Official,
  kept for delisted companies, and the filing's acceptance time tells
  before-open (BMO) from after-close (AMC). Historical tickers resolve to a CIK
  through EDGAR's legacy company lookup (it still knows TWTR, SIVB…).
* **Earnings, secondary: Nasdaq's calendar** (`api.nasdaq.com`, one request
  per date). It re-labels history with today's tickers (Facebook's 2019
  reports appear as META) and drops delisted companies entirely (no TWTR,
  CELG, SIVB), so it is survivor-biased and only used to fill gaps and timing.
  Both are dates reports actually happened. Companies confirm dates 2–4 weeks
  ahead, so "skip entries N days before earnings" is realistic; a surprise
  reschedule is not modelled.
* **FOMC** — statement dates from federalreserve.gov (calendar page for the
  last ~5 years, `fomchistorical{year}.htm` before that), 14:00 ET. Includes
  unscheduled statements (e.g. 2020-03-03, 2020-03-15).
* **CPI and employment report (NFP)** — release dates from the BLS archive
  index pages; both at 08:30 ET.

Raw responses are cached under `<root>/events/`.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging import log
from app.swing_agent.calendar import TradingCalendar

_NASDAQ = "https://api.nasdaq.com/api/calendar/earnings"
_FED_CAL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_FED_HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
_BLS = {"cpi": "https://www.bls.gov/bls/news-release/cpi.htm",
        "nfp": "https://www.bls.gov/bls/news-release/empsit.htm"}
_BLS_LINK = {"cpi": r"cpi_(\d{8})\.htm", "nfp": r"empsit_(\d{8})\.htm"}
_TIMING = {"time-pre-market": "bmo", "time-after-hours": "amc"}


def _browser_get(url: str, params: dict[str, Any] | None = None, tries: int = 4) -> Any:
    """GET through curl_cffi's Chrome fingerprint (Nasdaq and BLS reject
    plain clients). macOS needs `_scproxy` imported first."""
    try:
        import _scproxy  # noqa: F401
    except ImportError:
        pass
    from curl_cffi import requests as cr  # type: ignore

    delay = 2.0
    for i in range(tries):
        try:
            r = cr.get(url, params=params, impersonate="chrome", timeout=30)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            err = str(e)
        log.warning("events_retry", url=url, err=err, attempt=i + 1)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"failed {url} {params}: {err}")


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def earnings_day(d: date, cache_dir: Path) -> list[dict[str, str]]:
    p = cache_dir / f"{d.isoformat()}.json"
    if p.exists():
        return json.loads(p.read_text())
    js = _browser_get(_NASDAQ, {"date": d.isoformat()}).json()
    rows = ((js.get("data") or {}).get("rows")) or []
    slim = [{"symbol": (r.get("symbol") or "").strip().upper(), "time": r.get("time") or "",
             "fq": r.get("fiscalQuarterEnding") or ""} for r in rows]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(slim))
    return slim


def sync_earnings(cal: TradingCalendar, start: date, end: date, cache_dir: Path,
                  workers: int = 6, progress: Any = None) -> int:
    """Fetch every calendar date in [start, end] (weekends too: a few reports
    land on Saturdays). Days already cached are skipped. Returns the number
    of network fetches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = [d for d in pd.date_range(start, end, freq="D").date
            if not (cache_dir / f"{d.isoformat()}.json").exists()]
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(earnings_day, d, cache_dir): d for d in todo}
        for f in as_completed(futs):
            f.result()
            n += 1
            if progress and n % 200 == 0:
                progress(n, len(todo))
    return n


def earnings_table(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(cache_dir.glob("*.json")):
        d = pd.Timestamp(p.stem)
        for r in json.loads(p.read_text()):
            rows.append((r["symbol"], d, _TIMING.get(r["time"], "unknown"), r["fq"]))
    return pd.DataFrame(rows, columns=["symbol", "date", "timing", "fq"])


def reaction_sessions(earn: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
    """The session(s) whose open gaps on each report.

    BMO on D → D (or the next session if D isn't one). AMC on D → the next
    session after D. Unknown timing → both, which is the conservative choice.
    """
    out = []
    for sym, d, timing in earn[["symbol", "date", "timing"]].itertuples(index=False):
        dd = d.date()
        same = cal.session(dd) or cal.next(dd)
        nxt = cal.next(dd)
        days = {"bmo": [same.day], "intraday": [same.day], "amc": [nxt.day]}.get(
            timing, sorted({same.day, nxt.day}))
        out.extend((sym, pd.Timestamp(x), timing) for x in days)
    return pd.DataFrame(out, columns=["symbol", "session", "timing"]).drop_duplicates()


_SEC_UA = {"User-Agent": "StockPlatform-research/0.1 research@stockplatform.local"}
_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SUB_PAGE = "https://data.sec.gov/submissions/{name}"


def _sec_get(url: str, params: dict[str, Any] | None = None) -> Any:
    import httpx

    delay = 1.0
    for i in range(5):
        try:
            r = httpx.get(url, params=params, headers=_SEC_UA, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            err = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            err = str(e)
        log.warning("sec_retry", url=url, err=err, attempt=i + 1)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"SEC failed {url}: {err}")


def cik_for_ticker(ticker: str) -> int | None:
    """CIK via EDGAR's legacy lookup, which also resolves delisted tickers."""
    time.sleep(0.12)  # SEC fair-access: ≤ 10 requests/s
    r = _sec_get(_BROWSE, {"action": "getcompany", "CIK": ticker.replace(".", "-"),
                           "type": "8-K", "count": "1", "output": "atom"})
    m = re.search(r"<cik>(\d+)</cik>", r.text) if r is not None else None
    return int(m.group(1)) if m else None


def cik_by_name(name: str) -> tuple[int, str] | None:
    """Top EDGAR entity match for a company name (e.g. Alpaca's asset name
    for a delisted symbol). Callers must validate the hit."""
    import re as _re

    q = _re.sub(r"\b(common stock|class [a-c]|ordinary shares|inc\.?|corp\.?|corporation|plc|"
                r"co\.?|ltd\.?|holdings?|company)\b", " ", name, flags=_re.I)
    q = " ".join(q.replace(",", " ").split()[:3])
    if not q:
        return None
    time.sleep(0.12)
    r = _sec_get("https://efts.sec.gov/LATEST/search-index", {"keysTyped": q})
    hits = (r.json().get("hits") or {}).get("hits") or [] if r is not None else []
    if not hits:
        return None
    h = hits[0]
    return int(h["_id"]), str(h["_source"].get("entity"))


def _items_202(block: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for f, d, acc, it in zip(block.get("form") or [], block.get("filingDate") or [],
                             block.get("acceptanceDateTime") or [], block.get("items") or []):
        if str(f).startswith("8-K") and "2.02" in str(it or ""):
            out.append((str(d), str(acc)))
    return out


def edgar_earnings(cik: int, since: str = "2015-06-01") -> list[tuple[pd.Timestamp, str]]:
    """(filing date, timing) of every 8-K Item 2.02 since `since`.

    acceptanceDateTime is UTC ('Z'); converted to New York time it gives
    bmo (< 09:30), amc (≥ 16:00) or intraday (released during the session,
    the stock reacts the same day — treated like bmo for the gap, flagged).
    """
    time.sleep(0.12)
    r = _sec_get(_SUBMISSIONS.format(cik=cik))
    if r is None:
        return []
    js = r.json()
    rows = _items_202(js.get("filings", {}).get("recent", {}))
    for f in js.get("filings", {}).get("files", []):
        if str(f.get("filingTo", "")) >= since:
            time.sleep(0.12)
            page = _sec_get(_SUB_PAGE.format(name=f["name"]))
            if page is not None:
                rows += _items_202(page.json())
    out = []
    for d, acc in rows:
        if d < since:
            continue
        ts = pd.Timestamp(acc.replace("Z", "")).tz_localize("UTC").tz_convert("America/New_York") \
            if acc else None
        if ts is None:
            timing = "unknown"
        elif ts.date().isoformat() != d:
            timing = "unknown"   # accepted on another day than filed (after 17:30 cut-off)
        elif ts.hour * 60 + ts.minute < 9 * 60 + 30:
            timing = "bmo"
        elif ts.hour >= 16:
            timing = "amc"
        else:
            timing = "intraday"
        out.append((pd.Timestamp(d), timing))
    return sorted(set(out))


def merge_earnings(edgar: pd.DataFrame, nasdaq: pd.DataFrame) -> pd.DataFrame:
    """Union of both sources per symbol, conservative.

    Same symbol and date in both → one row; a known timing beats 'unknown',
    and two known timings that disagree become 'unknown' (avoid both
    sessions). Different dates are both kept: some companies file the 8-K a
    day after the press release, and blacking out one extra session is
    cheaper than holding through a report.
    """
    e = edgar.assign(source="edgar")
    n = nasdaq.assign(source="nasdaq")
    rows = []
    for (sym, d), g in pd.concat([e, n], ignore_index=True).groupby(["symbol", "date"]):
        known = sorted(set(g["timing"]) - {"unknown"})
        timing = known[0] if len(known) == 1 else "unknown"
        src = "both" if g["source"].nunique() == 2 else g["source"].iloc[0]
        rows.append((sym, d, timing, src))
    return pd.DataFrame(rows, columns=["symbol", "date", "timing", "source"])


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------

def fomc_dates(first_year: int, last_year: int) -> list[date]:
    found: set[str] = set()
    pages = [_FED_CAL] + [_FED_HIST.format(year=y) for y in range(first_year, last_year + 1)]
    for url in pages:
        try:
            html = _browser_get(url, tries=1 if "historical" in url else 4).text
        except RuntimeError:
            continue  # recent years have no "historical" page yet
        found.update(re.findall(r"monetary(\d{8})a\.htm", html))
    return sorted(d for x in found
                  if first_year <= (d := date(int(x[:4]), int(x[4:6]), int(x[6:]))).year <= last_year)


def bls_dates(kind: str) -> list[date]:
    html = _browser_get(_BLS[kind]).text
    out = set()
    for x in re.findall(_BLS_LINK[kind], html):
        out.add(date(int(x[4:]), int(x[:2]), int(x[2:4])))  # MMDDYYYY
    return sorted(out)


def macro_table(first_year: int, last_year: int) -> pd.DataFrame:
    rows = [(d, "fomc", "14:00") for d in fomc_dates(first_year, last_year)]
    for k in ("cpi", "nfp"):
        rows += [(d, k, "08:30") for d in bls_dates(k) if first_year <= d.year <= last_year]
    df = pd.DataFrame(rows, columns=["date", "kind", "time_et"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "kind"], ignore_index=True)
