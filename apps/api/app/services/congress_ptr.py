"""US House STOCK Act disclosures from the official source.

Replaces the hand-written demo fixture and the dead housestockwatcher.com feed.

Source (free, no key): the House Clerk publishes
  - a yearly index   https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip
    (XML listing every filing; FilingType "P" = Periodic Transaction Report)
  - each PTR as PDF  https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf

Electronically filed PTRs (doc ids starting with "2") contain real text and are
parsed here; paper filings (ids starting with "8"/"9") are scanned images and are
recorded as unparseable rather than guessed at. Party/state come from the public
unitedstates/congress-legislators dataset.

Everything is archived in SQLite (filings never change once published), so a
sync only downloads what is new. Run `scripts/refresh_congress.py` daily.

Senate filings (efdsearch.senate.gov) sit behind a session/terms gate and are
not covered yet.
"""
from __future__ import annotations

import io
import json
import math
import re
import sqlite3
import threading
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from app.core.logging import log

DB_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "congress" / "house_ptr.sqlite"
_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
_LEGISLATORS_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.json"
_UA = {"User-Agent": "StockPlatform/0.1 (personal research; contact@example.com)"}
_MIN_INTERVAL_S = 0.35
_local = threading.local()
_http_lock = threading.Lock()
_last_call = 0.0

# Rows omitting the ticker for well-known listed companies (seen in real filings).
_NAME_TICKERS = {
    "alliancebernstein": "AB", "alliance bernstein": "AB", "american express": "AXP",
    "crowdstrike": "CRWD", "interactive brokers": "IBKR", "alphabet inc. - class c": "GOOG",
    "tempus ai": "TEM", "roblox corporation class a": "RBLX",
}
_TYPE = {"P": "buy", "S": "sell", "S (partial)": "sell_partial", "E": "exchange"}
_TX_RE = re.compile(
    r"(?:^|\s)(P|S \(partial\)|S|E)\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+\$([\d,]+)(?:\s*-\s*\$?([\d,]+))?",
    re.IGNORECASE,  # 2019-early-2020 filings extract with scrambled letter case ("s" = sale)
)
# Early-2020 layout: "...FILING STATUS : New DESCRIPTION : <text>.<owner>" all run together.
_OLD_DESC = re.compile(r"(?i)d?\s*es\s*cription\s*:\s*(.*)$")
_TICKER_RE = re.compile(r"\(([A-Za-z][A-Za-z.\-]{0,5})\)")
_ASSET_RE = re.compile(r"\[(\w{2})\]")
_OWNER_RE = re.compile(r"^(SP|JT|DC)\s+")


# --------------------------------------------------------------------------- db

def _db() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS filings (
                doc_id TEXT PRIMARY KEY, year INTEGER, last TEXT, first TEXT,
                state_dst TEXT, filed TEXT, status INTEGER DEFAULT 0);  -- 0 pending, 1 parsed, -1 unparseable
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, politician TEXT, party TEXT,
                state TEXT, owner TEXT, symbol TEXT, asset_name TEXT, asset_type TEXT, side TEXT,
                traded_at TEXT, notified_at TEXT, disclosed_at TEXT,
                amount_min INTEGER, amount_max INTEGER, description TEXT,
                UNIQUE (doc_id, asset_name, side, traded_at, amount_min, description));
            CREATE INDEX IF NOT EXISTS ix_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS ix_trades_pol ON trades(politician);
            CREATE INDEX IF NOT EXISTS ix_trades_disclosed ON trades(disclosed_at);
            CREATE TABLE IF NOT EXISTS scorecards (
                politician TEXT PRIMARY KEY, data TEXT, computed_at TEXT);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(filings)")}
        if "text" not in cols:   # extracted PDF text, kept so parser fixes never re-download
            conn.execute("ALTER TABLE filings ADD COLUMN text TEXT")
        if "subholding" not in {r[1] for r in conn.execute("PRAGMA table_info(trades)")}:
            conn.execute("ALTER TABLE trades ADD COLUMN subholding TEXT")
        _local.conn = conn
    return conn


def has_data() -> bool:
    try:
        return _db().execute("SELECT 1 FROM trades LIMIT 1").fetchone() is not None
    except Exception:
        return False


def _get(url: str, timeout: float = 60) -> httpx.Response | None:
    global _last_call
    for attempt in range(4):
        with _http_lock:
            wait = _MIN_INTERVAL_S - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        try:
            r = httpx.get(url, headers=_UA, timeout=timeout, follow_redirects=True)
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        return None
    return None


# ------------------------------------------------------------------- parsing

def _mmddyyyy(s: str) -> str:
    return datetime.strptime(s, "%m/%d/%Y").date().isoformat()


_LABEL = re.compile(r"^(F\s+S|S\s+O|L|D|C)\s*:\s*(.*)$")
_HEADER = re.compile(r"^(Filing ID|ID Owner|Type$|Date Notification|Date$|Amount Cap|Gains >|\$200\?|\* For the complete|"
                     r"P T R|Clerk of the House|F I$|Name:|Status:|State/District:|T$|I P O|I CERTIFY|C S$|Digitally Signed|"
                     r"Yes No|my knowledge)")
_OWNER_LINE = re.compile(r"^(SP|JT|DC)\s+")


_DATES_START = re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{2}/\d{2}/\d{4}\s+\$")


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("\x00", "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or _HEADER.match(line):
            continue
        # older layout breaks "P" / "[ST] P" and the dates onto separate lines
        if out and _DATES_START.match(line):
            out[-1] = f"{out[-1]} {line}"
            continue
        out.append(line)
    return out


def parse_ptr_text(text: str) -> list[dict[str, Any]]:
    """Parse the text of an electronically filed PTR (2019+ layout) into rows.

    pypdf "plain" text, one element per line:
        JT Boston Scientific Corporation            <- owner code + asset name
        Common Stock (BSX) [ST]                     <- name continues, ticker, asset type
        S (partial) 09/04/2026 09/15/2026 $50,001 - <- type, tx date, notification date, amount
        $100,000                                    <- amount upper bound (may wrap)
        F S: New                                    <- labelled lines belong to the row above:
        S O: Hern Family Revocable Trust               filing status, sub-account,
        D: Purchased 10,000 shares.                    description (may wrap), location
    The asset text may also share the line with the type/dates anchor.
    Page headers repeat mid-document and are dropped first.
    """
    out: list[dict[str, Any]] = []
    asset_buf: list[str] = []
    in_desc = False
    for line in _clean_lines(text):
        m = _TX_RE.search(line)
        if m:
            asset_text = " ".join(asset_buf + [line[:m.start()].strip()]).strip()
            asset_buf, in_desc = [], False
            od = _OLD_DESC.search(asset_text)
            if od:
                body = od.group(1)
                # the description ends with "." glued to the next row's owner code ("...$1600.sP amazon...")
                ends = list(re.finditer(r"\.\s*(sp|jt|dc)\s+", body, re.I))
                cut = ends[-1].start() if ends else body.rfind(". ")
                if out and cut > 0:
                    out[-1]["description"] = body[:cut + 1].strip()[:300]
                asset_text = body[cut + 1:].strip() if cut >= 0 else body
            # drop anything before the last standalone owner code (mangled page headers)
            owners = list(re.finditer(r"(?:^|\s)(sp|jt|dc)\s+(?=\S)", asset_text, re.I))
            if owners:
                asset_text = owners[-1].group(1).upper() + " " + asset_text[owners[-1].end():]
            owner = ""
            om = _OWNER_LINE.match(asset_text)
            if om:
                owner, asset_text = om.group(1), asset_text[om.end():]
            tk = _TICKER_RE.search(asset_text)
            ticker = tk.group(1).upper() if tk else None
            if not ticker:
                low = asset_text.lower()
                ticker = next((t for k, t in _NAME_TICKERS.items() if k in low), None)
            at = _ASSET_RE.search(asset_text)
            name = re.sub(r"\s+", " ", _ASSET_RE.sub("", _TICKER_RE.sub("", asset_text))).strip(" -")
            out.append({
                "owner": owner, "symbol": ticker, "asset_name": name[:160],
                "asset_type": at.group(1).upper() if at else "",
                "side": _TYPE[m.group(1).upper().replace("(PARTIAL)", "(partial)")],
                "traded_at": _mmddyyyy(m.group(2)), "notified_at": _mmddyyyy(m.group(3)),
                "amount_min": int(m.group(4).replace(",", "")),
                "amount_max": int(m.group(5).replace(",", "")) if m.group(5) else None,
                "description": "", "subholding": "",
            })
            continue
        if out and out[-1]["amount_max"] is None and not asset_buf and re.fullmatch(r"\$[\d,]+", line):
            out[-1]["amount_max"] = int(line[1:].replace(",", ""))
            continue
        lab = _LABEL.match(line)
        if lab and out and not asset_buf:
            key, val = re.sub(r"\s+", " ", lab.group(1)), lab.group(2).strip()
            in_desc = key == "D"
            if key == "D":
                out[-1]["description"] = val
            elif key == "S O":
                out[-1]["subholding"] = val
            continue
        if in_desc and out and not asset_buf and not _OWNER_LINE.match(line) and "[" not in line \
                and not out[-1]["description"].endswith("."):
            out[-1]["description"] = (out[-1]["description"] + " " + line)[:300]
            continue
        in_desc = False
        asset_buf.append(line)
    for t in out:
        if not t["asset_type"]:
            t["asset_type"] = "OP" if "option" in t["description"].lower() else "ST"
    return out


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    text = "\n".join((page.extract_text(extraction_mode="plain") or "") for page in reader.pages)
    # some PDFs decode to lone UTF-16 surrogates, which SQLite (UTF-8) rejects
    return text.encode("utf-8", "replace").decode("utf-8")


# --------------------------------------------------------------- legislators

def _party_map() -> dict[tuple[str, str], tuple[str, str]]:
    """(LAST, STATE) -> (party letter, full name)."""
    row = _db().execute("SELECT value FROM meta WHERE key='legislators'").fetchone()
    data = json.loads(row["value"]) if row else None
    if data is None:
        r = _get(_LEGISLATORS_URL, timeout=30)
        data = r.json() if r is not None else []
        with _db():
            _db().execute("INSERT OR REPLACE INTO meta VALUES ('legislators', ?)", (json.dumps(data),))
    out = {}
    for p in data:
        term = (p.get("terms") or [{}])[-1]
        party = (term.get("party") or "")[:1]
        full = (p.get("name") or {}).get("official_full") or \
            f"{p['name'].get('first', '')} {p['name'].get('last', '')}".strip()
        out[(p["name"]["last"].upper(), term.get("state", ""))] = (party, full)
    return out


# ---------------------------------------------------------------------- sync

def _sync_index(year: int) -> int:
    r = _get(_INDEX_URL.format(year=year))
    if r is None:
        return 0
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(r.content)).read(f"{year}FD.xml"))
    rows = []
    for m in root.findall("Member"):
        if m.findtext("FilingType") != "P":
            continue
        filed = m.findtext("FilingDate") or ""
        try:
            filed = datetime.strptime(filed, "%m/%d/%Y").date().isoformat()
        except ValueError:
            pass
        rows.append((m.findtext("DocID"), year, (m.findtext("Last") or "").strip(),
                     (m.findtext("First") or "").strip(), m.findtext("StateDst") or "", filed))
    with _db():
        _db().executemany("INSERT OR IGNORE INTO filings (doc_id, year, last, first, state_dst, filed) "
                          "VALUES (?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def _parse_filing(f: sqlite3.Row, parties: dict) -> int:
    conn = _db()
    if not str(f["doc_id"]).startswith("2"):   # paper filing -> scanned image
        with conn:
            conn.execute("UPDATE filings SET status=-1 WHERE doc_id=?", (f["doc_id"],))
        return 0
    text = f["text"]
    if text is None:
        r = _get(_PDF_URL.format(year=f["year"], doc_id=f["doc_id"]))
        if r is None:
            return 0   # stays pending, retried next sync
        try:
            text = _pdf_text(r.content)
        except Exception as e:
            log.warning("congress_ptr.pdf_failed", doc=f["doc_id"], err=str(e)[:120])
            text = ""
        with conn:
            conn.execute("UPDATE filings SET text=? WHERE doc_id=?", (text, f["doc_id"]))
    rows = parse_ptr_text(text)
    state = (f["state_dst"] or "")[:2]
    party, full = parties.get((f["last"].upper(), state), ("", ""))
    name = full or f"{f['first']} {f['last']}".strip()
    name = re.sub(r"^(Hon\.?\s+)", "", name)
    with conn:
        for t in rows:
            conn.execute(
                "INSERT OR IGNORE INTO trades (doc_id, politician, party, state, owner, symbol, asset_name, "
                "asset_type, side, traded_at, notified_at, disclosed_at, amount_min, amount_max, description, "
                "subholding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f["doc_id"], name, party, f["state_dst"], t["owner"], t["symbol"], t["asset_name"],
                 t["asset_type"], t["side"], t["traded_at"], t["notified_at"], f["filed"],
                 t["amount_min"], t["amount_max"], t["description"], t.get("subholding") or ""))
        conn.execute("UPDATE filings SET status=? WHERE doc_id=?", (1 if rows else -1, f["doc_id"]))
    return len(rows)


def reparse() -> dict[str, int]:
    """Re-run the parser over every archived filing's stored text (no downloads).
    Use after a parser fix; filings archived before text was stored go back to pending."""
    conn = _db()
    with conn:
        conn.execute("DELETE FROM trades")
        conn.execute("UPDATE filings SET status=0 WHERE doc_id LIKE '2%'")
    parties = _party_map()
    n = 0
    for f in conn.execute("SELECT * FROM filings WHERE status=0 AND text IS NOT NULL").fetchall():
        n += _parse_filing(f, parties)
    return {"trades": n}


def sync(years: list[int] | None = None, max_filings: int | None = None) -> dict[str, int]:
    """Refresh the yearly indexes and parse every pending electronic PTR."""
    this_year = date.today().year
    years = years or [this_year - 1, this_year]
    n_index = sum(_sync_index(y) for y in years)
    parties = _party_map()
    pending = _db().execute(
        "SELECT * FROM filings WHERE status=0 AND year IN (%s) ORDER BY filed DESC" % ",".join("?" * len(years)),
        years).fetchall()
    if max_filings:
        pending = pending[:max_filings]
    n_trades = 0
    for i, f in enumerate(pending):
        try:
            n_trades += _parse_filing(f, parties)
        except Exception as e:   # one bad filing must not abort the whole sync
            log.warning("congress_ptr.filing_failed", doc=f["doc_id"], err=str(e)[:160])
        if i % 50 == 0:
            log.info("congress_ptr.sync", done=i, of=len(pending), trades=n_trades)
    with _db():
        _db().execute("INSERT OR REPLACE INTO meta VALUES ('last_sync', ?)", (datetime.utcnow().isoformat(),))
    return {"index_rows": n_index, "filings_parsed": len(pending), "trades_added": n_trades}


# -------------------------------------------------------------------- queries

def _fmt_amount(lo: int | None, hi: int | None) -> str:
    if lo is None:
        return ""
    return f"${lo:,} – ${hi:,}" if hi else f"${lo:,}+"


def _to_app(row: sqlite3.Row) -> dict[str, Any]:
    lo, hi = row["amount_min"], row["amount_max"]
    party = f"{row['party']}-{(row['state'] or '')[:2]}" if row["party"] else (row["state"] or "")[:2]
    return {
        "politician": row["politician"], "chamber": "House", "party": party,
        "traded_at": row["traded_at"], "disclosed_at": row["disclosed_at"],
        "symbol": row["symbol"], "asset_name": row["asset_name"], "asset_type": row["asset_type"],
        "side": row["side"], "owner": row["owner"],
        "amount_range": _fmt_amount(lo, hi),
        "amount_mid": (lo + hi) / 2 if lo and hi else lo,
        "note": row["description"],
        "subholding": row["subholding"] if "subholding" in row.keys() else "",
        "source_url": _PDF_URL.format(year=row["disclosed_at"][:4], doc_id=row["doc_id"]),
        "source": "house_clerk",
    }


def trades(symbol: str | None = None, politician: str | None = None, days: int | None = None,
           limit: int = 50, side: str | None = None) -> list[dict[str, Any]]:
    q, args = "SELECT * FROM trades WHERE 1=1", []
    if symbol:
        q += " AND symbol = ?"; args.append(symbol.upper())
    if politician:
        q += " AND lower(politician) LIKE ?"; args.append(f"%{politician.lower()}%")
    if days:
        q += " AND disclosed_at >= ?"; args.append((date.today() - timedelta(days=days)).isoformat())
    if side:
        q += " AND side LIKE ?"; args.append(f"{side}%")
    q += " ORDER BY disclosed_at DESC, traded_at DESC LIMIT ?"
    args.append(limit)
    return [_to_app(r) for r in _db().execute(q, args).fetchall()]


def status() -> dict[str, Any]:
    c = _db()
    row = c.execute("SELECT value FROM meta WHERE key='last_sync'").fetchone()
    by = {r["status"]: r["n"] for r in c.execute("SELECT status, count(*) n FROM filings GROUP BY status")}
    span = c.execute("SELECT min(disclosed_at) a, max(disclosed_at) b, count(*) n, "
                     "count(DISTINCT politician) p FROM trades").fetchone()
    return {"last_sync": row["value"] if row else None, "filings_parsed": by.get(1, 0),
            "filings_unparseable_paper": by.get(-1, 0), "filings_pending": by.get(0, 0),
            "trades": span["n"], "politicians": span["p"], "first_disclosure": span["a"],
            "last_disclosure": span["b"]}


# ----------------------------------------------------------------- scorecards

def compute_scorecards(min_buys: int = 5, lookback_years: int = 5) -> int:
    """Per politician: what copying their disclosed stock purchases would have
    earned — buy at the first open after the filing date, hold 3 / 12 months —
    versus SPY over the identical window. Stored for the tracker endpoint."""
    import numpy as np

    from app.services import market_data as md

    since = (date.today() - timedelta(days=365 * lookback_years)).isoformat()
    rows = _db().execute(
        "SELECT politician, party, state, symbol, disclosed_at, traded_at, asset_type FROM trades "
        "WHERE side='buy' AND symbol IS NOT NULL AND disclosed_at >= ? AND asset_type IN ('ST','OP')",
        (since,)).fetchall()
    if not rows:
        return 0
    by_pol: dict[str, list] = {}
    for r in rows:
        by_pol.setdefault(r["politician"], []).append(r)
    eligible = {p: rs for p, rs in by_pol.items() if len(rs) >= min_buys}
    symbols = {r["symbol"] for rs in eligible.values() for r in rs} | {"SPY"}
    prices: dict[str, Any] = {}
    for s in symbols:
        try:
            # must be a range alpaca_bars/_RANGE_TO_DAYS knows ("6y" silently fell back to 1y)
            df = md.get_history(s, interval="1d", range_="7y" if lookback_years <= 5 else "10y")
            if not df.empty:
                o = df["open"].copy()
                o.index = o.index.tz_convert(None).normalize() if o.index.tz is not None else o.index.normalize()
                prices[s] = o
        except Exception:
            continue
    spy = prices.get("SPY")
    if spy is None:
        return 0

    def fwd(series, day: str, h: int) -> float | None:
        idx = series.index
        p = idx.searchsorted(np.datetime64(day) + np.timedelta64(1, "D"))
        if p + h >= len(idx):
            return None
        a, b = float(series.iloc[p]), float(series.iloc[p + h])
        return math.log(b / a) if a > 0 and b > 0 else None

    now = datetime.utcnow().isoformat()
    n = 0
    for pol, rs in eligible.items():
        ex = {63: [], 252: []}
        lags, tickers = [], {}
        for r in rs:
            tickers[r["symbol"]] = tickers.get(r["symbol"], 0) + 1
            try:
                lags.append((date.fromisoformat(r["disclosed_at"]) - date.fromisoformat(r["traded_at"])).days)
            except ValueError:
                pass
            s = prices.get(r["symbol"])
            if s is None:
                continue
            for h in ex:
                a, b = fwd(s, r["disclosed_at"], h), fwd(spy, r["disclosed_at"], h)
                if a is not None and b is not None:
                    ex[h].append(a - b)

        def summ(v: list[float]) -> dict | None:
            if len(v) < 3:
                return None
            arr = np.array(v)
            return {"n": len(v), "avg_excess_pct": round(float(np.mean(arr)) * 100, 2),
                    "median_excess_pct": round(float(np.median(arr)) * 100, 2),
                    "hit_rate_pct": round(float((arr > 0).mean()) * 100, 1),
                    "t_stat": round(float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))), 2)
                    if arr.std(ddof=1) > 0 else None}

        card = {
            "politician": pol, "party": rs[0]["party"], "state": (rs[0]["state"] or "")[:2],
            "n_buys": len(rs), "n_option_buys": sum(1 for r in rs if r["asset_type"] == "OP"),
            "median_disclosure_lag_days": int(np.median(lags)) if lags else None,
            "top_tickers": [t for t, _ in sorted(tickers.items(), key=lambda kv: -kv[1])[:6]],
            "copy_3m_vs_spy": summ(ex[63]), "copy_12m_vs_spy": summ(ex[252]),
            "window_years": lookback_years,
        }
        with _db():
            _db().execute("INSERT OR REPLACE INTO scorecards VALUES (?, ?, ?)", (pol, json.dumps(card), now))
        n += 1
    return n


def scorecards() -> list[dict[str, Any]]:
    out = [json.loads(r["data"]) | {"computed_at": r["computed_at"]}
           for r in _db().execute("SELECT data, computed_at FROM scorecards").fetchall()]
    out.sort(key=lambda c: -c["n_buys"])
    return out
