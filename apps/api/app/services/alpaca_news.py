"""Point-in-time news archive from Alpaca's market-data news API (Benzinga feed).

Why this exists: every news feature in the forecaster so far was either
inference-only (live broadcast onto the last bars) or needed an Alpha Vantage
key with a 25-requests/day quota, so training always saw zeros. Alpaca serves
per-ticker, timestamped news back to 2016 with the account keys the platform
already has, which makes a real historical news signal possible.

  GET https://data.alpaca.markets/v1beta1/news?symbols=...&start=...&end=...

Articles are archived in SQLite (history is immutable, so each week is fetched
once). Sentiment is scored lazily per headline (finBERT when available, the
lexicon otherwise) and stored next to the article.

`daily_features(symbol, index)` turns the archive into per-bar features that
only use articles published *before that bar's close* (16:00 New York), so a
backtest never sees news from after the decision time.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.core.logging import log

_URL = "https://data.alpaca.markets/v1beta1/news"
DB_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "news" / "alpaca_news.sqlite"
# The free data plan allows 200 requests/minute *per account*, shared with the
# running app (bars, quotes). Backfills default to half of it.
_min_interval_s = 0.6
_lock = threading.Lock()
_schema_lock = threading.Lock()
_last_call = 0.0
_local = threading.local()


def _db() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=60)
        with _schema_lock:   # concurrent first connections race on WAL/DDL otherwise
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
                headline TEXT, summary TEXT, source TEXT,
                sentiment REAL, sentiment_conf REAL);
            CREATE TABLE IF NOT EXISTS article_symbols (
                id INTEGER NOT NULL, symbol TEXT NOT NULL, PRIMARY KEY (symbol, id));
            CREATE TABLE IF NOT EXISTS fetched_weeks (
                week_start TEXT NOT NULL, symbols_key TEXT NOT NULL,
                PRIMARY KEY (week_start, symbols_key));
            CREATE INDEX IF NOT EXISTS ix_articles_created ON articles(created_at);
            """)
        _local.conn = conn
    return conn


def _headers() -> dict[str, str]:
    return {"APCA-API-KEY-ID": settings.alpaca_api_key or "",
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret or ""}


def is_available() -> bool:
    return bool(settings.alpaca_api_key and settings.alpaca_api_secret)


def _get(params: dict) -> dict | None:
    global _last_call
    for attempt in range(5):
        with _lock:
            wait = _min_interval_s - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        try:
            r = httpx.get(_URL, params=params, headers=_headers(), timeout=30)
        except httpx.HTTPError:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        log.warning("alpaca_news.http", status=r.status_code, body=r.text[:200])
        return None
    return None


def _symbols_key(symbols: list[str]) -> str:
    return ",".join(sorted(symbols))


def _fetch_week(symbols: list[str], week_start: datetime) -> int:
    """Fetch and store one week of articles for `symbols`. Returns #articles."""
    key = _symbols_key(symbols)
    ws = week_start.strftime("%Y-%m-%d")
    conn = _db()
    if conn.execute("SELECT 1 FROM fetched_weeks WHERE week_start=? AND symbols_key=?",
                    (ws, key)).fetchone():
        return 0
    end = week_start + timedelta(days=7)
    params = {"symbols": ",".join(symbols), "limit": 50, "sort": "asc",
              "start": week_start.strftime("%Y-%m-%dT00:00:00Z"),
              "end": end.strftime("%Y-%m-%dT00:00:00Z")}
    n = 0
    token = None
    while True:
        if token:
            params["page_token"] = token
        data = _get(params)
        if data is None:
            return n   # leave the week unmarked so a later sync retries it
        rows = data.get("news") or []
        with conn:
            for a in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO articles (id, created_at, headline, summary, source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (a["id"], a["created_at"], a.get("headline"),
                     (a.get("summary") or "")[:500], a.get("source")),
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO article_symbols (id, symbol) VALUES (?, ?)",
                    [(a["id"], s) for s in a.get("symbols") or []],
                )
        n += len(rows)
        token = data.get("next_page_token")
        if not token:
            break
    with conn:
        conn.execute("INSERT OR IGNORE INTO fetched_weeks VALUES (?, ?)", (ws, key))
    return n


def sync(symbols: list[str], start: str = "2016-01-04", end: str | None = None,
         workers: int = 3, max_requests_per_min: int = 100) -> int:
    """Backfill the archive week by week. Idempotent — finished weeks are skipped.
    `max_requests_per_min` leaves the rest of the account quota to the live app."""
    from concurrent.futures import ThreadPoolExecutor

    global _min_interval_s
    _min_interval_s = 60.0 / max(1, max_requests_per_min)

    t0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    t0 -= timedelta(days=t0.weekday())
    t1 = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
    weeks = []
    w = t0
    while w < t1:
        weeks.append(w)
        w += timedelta(days=7)
    def fetch(wk: datetime) -> int:
        try:
            return _fetch_week(symbols, wk)
        except Exception as e:   # week stays unmarked; the next sync retries it
            log.warning("alpaca_news.week_failed", week=wk.date().isoformat(), error=str(e))
            return 0

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, n in enumerate(ex.map(fetch, weeks)):
            total += n
            if i % 26 == 0:
                log.info("alpaca_news.sync", week=weeks[i].date().isoformat(), total=total)
    return total


def score_pending(batch: int = 256, limit: int | None = None) -> int:
    """Score headlines that have no sentiment yet (finBERT, else lexicon)."""
    from app.services import sentiment

    conn = _db()
    done = 0
    while limit is None or done < limit:
        rows = conn.execute("SELECT id, headline FROM articles WHERE sentiment IS NULL LIMIT ?",
                            (batch,)).fetchall()
        if not rows:
            break
        scored = sentiment.score_batch([h or "" for _, h in rows])
        with conn:
            conn.executemany("UPDATE articles SET sentiment=?, sentiment_conf=? WHERE id=?",
                             [(s, c, i) for (i, _), (s, c) in zip(rows, scored, strict=True)])
        done += len(rows)
    return done


def _articles_for(symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT a.created_at, a.sentiment FROM articles a "
        "JOIN article_symbols s ON s.id = a.id WHERE s.symbol = ?",
        _db(), params=(symbol,),
    )
    if df.empty:
        return df
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("America/New_York")
    # An article published after the 16:00 close is only actionable next session.
    df["session"] = (ts + pd.Timedelta(hours=8)).dt.normalize().dt.tz_localize(None)
    return df


def daily_features(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-bar news features using only articles available at that bar's close.

    news_n_5d        articles in the last 5 sessions (log1p)
    news_att_z       5d article count vs its trailing 1y mean/std (attention shock)
    news_sent_5d     mean headline sentiment over 5 sessions (0 when no news)
    news_sent_21d    mean headline sentiment over 21 sessions
    news_sent_chg    news_sent_5d − news_sent_21d
    """
    cols = ["news_n_5d", "news_att_z", "news_sent_5d", "news_sent_21d", "news_sent_chg"]
    out = pd.DataFrame(0.0, index=index, columns=cols)
    arts = _articles_for(symbol)
    if arts.empty:
        return out
    bars = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    # map every article to the first bar on/after its session date
    pos = np.searchsorted(bars.values, arts["session"].values.astype("datetime64[ns]"), side="left")
    arts = arts.assign(pos=pos)
    arts = arts[arts["pos"] < len(bars)]
    cnt = np.bincount(arts["pos"], minlength=len(bars)).astype(float)
    sent = arts.dropna(subset=["sentiment"])
    ssum = np.bincount(sent["pos"], weights=sent["sentiment"], minlength=len(bars))
    scnt = np.bincount(sent["pos"], minlength=len(bars)).astype(float)

    c = pd.Series(cnt)
    n5 = c.rolling(5, min_periods=1).sum()
    mu = n5.rolling(252, min_periods=60).mean()
    sd = n5.rolling(252, min_periods=60).std()
    s5 = pd.Series(ssum).rolling(5, min_periods=1).sum() / pd.Series(scnt).rolling(5, min_periods=1).sum()
    s21 = pd.Series(ssum).rolling(21, min_periods=1).sum() / pd.Series(scnt).rolling(21, min_periods=1).sum()
    out["news_n_5d"] = np.log1p(n5.values)
    out["news_att_z"] = ((n5 - mu) / (sd + 1e-9)).fillna(0.0).clip(-5, 10).values
    out["news_sent_5d"] = s5.fillna(0.0).values
    out["news_sent_21d"] = s21.fillna(0.0).values
    out["news_sent_chg"] = (s5.fillna(0.0) - s21.fillna(0.0)).values
    return out
