"""Bar shaping + on-disk store: session filtering, 1h resampling, "known at" views.

Time conventions (every later phase relies on these; tests pin them):

* Intraday bars are indexed by bar **start** (UTC), as Alpaca serves them.
  Each row also carries `t_close`, the moment the bar is complete. A bar may
  only be used by a decision taken at or after its `t_close`.
* 1h bars are built from 15m bars and aligned to the 09:30 open
  (09:30, 10:30, …, 15:30). The last hour is 30 minutes long, and on a 13:00
  half-day the session simply has fewer buckets. Alpaca's own 1Hour bars are
  clock-aligned (09:00–10:00 mixes pre-market with the open), so they are not
  used.
* Daily bars are indexed by session date and carry `t_close` = that
  session's close. During session D, `known_at(daily, ts)` therefore returns
  bars up to D-1 only: the daily regime cannot see today's unfinished bar.
* Prices are split-adjusted. `split_factor` converts back to the raw price
  traded on that day (raw = adjusted × split_factor), which the cost model
  needs for per-share fees and tick-size spreads. `div_ret` is the cash
  dividend on its ex-date as a fraction of the previous close; the backtest
  credits it to held positions instead of using dividend-adjusted prices.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent import config
from app.swing_agent.calendar import NY, TradingCalendar

OHLCV = ["open", "high", "low", "close", "volume", "trades", "vwap"]


# ---------------------------------------------------------------------------
# Pure transforms
# ---------------------------------------------------------------------------

def session_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """New York calendar date of each UTC timestamp, as tz-naive midnight."""
    return pd.DatetimeIndex(index.tz_convert(NY).date).astype("datetime64[ns]")


def split_sessions(df: pd.DataFrame, cal: TradingCalendar,
                   bar_minutes: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(regular-hours bars, per-session pre-market summary).

    A bar is regular-hours when [start, start + bar) lies inside
    [open, close). Bars on non-sessions (e.g. a holiday with stray prints)
    are dropped. Pre-market = 04:00 → open.
    """
    if df.empty:
        return df.assign(session=pd.NaT, t_close=pd.NaT), pd.DataFrame()
    days = session_days(df.index)
    cf = cal.frame()
    opens = pd.DatetimeIndex(cf["open"].reindex(days))    # UTC; NaT off-session
    closes = pd.DatetimeIndex(cf["close"].reindex(days))
    start = df.index
    bar = pd.Timedelta(minutes=bar_minutes)
    valid = ~opens.isna()
    rth = valid & (start >= opens) & (start + bar <= closes)
    pre = valid & (start < opens)

    out = df.loc[rth].copy()
    out["session"] = days[rth]
    out["t_close"] = start[rth] + bar

    pm = df.loc[pre].copy()
    if pm.empty:
        summary = pd.DataFrame(columns=["pm_volume", "pm_high", "pm_low", "pm_last"])
    else:
        pm["session"] = days[pre]
        g = pm.groupby("session")
        summary = pd.DataFrame({"pm_volume": g["volume"].sum(), "pm_high": g["high"].max(),
                                "pm_low": g["low"].min(), "pm_last": g["close"].last()})
    return out, summary


def resample_hourly(m15: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
    """Session-aligned 1h bars from regular-hours 15m bars.

    `t_close` is the END of the hour bucket (capped at the session close),
    not the close of the last 15m bar that happened to trade: the hour is
    complete only when the clock says so.
    """
    if m15.empty:
        return m15.copy()
    cf = cal.frame()
    opens = pd.DatetimeIndex(cf["open"].reindex(m15["session"]))
    closes = pd.DatetimeIndex(cf["close"].reindex(m15["session"]))
    k = (m15.index - opens) // pd.Timedelta(minutes=60)
    start = opens + pd.to_timedelta(np.asarray(k) * 60, unit="m")
    end = start + pd.Timedelta(minutes=60)
    end = end.where(end <= closes, closes)
    tmp = m15.assign(_start=start, _end=end,
                     _pv=m15["vwap"].fillna(m15["close"]) * m15["volume"])
    g = tmp.groupby("_start", sort=True)
    h = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(), "trades": g["trades"].sum(),
        "_pv": g["_pv"].sum(), "session": g["session"].first(), "t_close": g["_end"].first(),
        "n_bars": g["open"].size(),
    })
    h["vwap"] = np.where(h["volume"] > 0, h["_pv"] / h["volume"].replace(0, np.nan), h["close"])
    h.index.name = "ts"
    return h[[*OHLCV, "session", "t_close", "n_bars"]]


def daily_frame(raw: pd.DataFrame, split: pd.DataFrame, total: pd.DataFrame,
                cal: TradingCalendar) -> pd.DataFrame:
    """Merge Alpaca daily bars fetched raw / split-adjusted / all-adjusted."""
    def by_day(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d.index = session_days(d.index)
        return d[~d.index.duplicated(keep="last")]

    s, r, t = by_day(split), by_day(raw), by_day(total)
    out = s[OHLCV].copy()
    out["close_raw"] = r["close"].reindex(out.index)
    out["split_factor"] = (out["close_raw"] / out["close"]).round(6)
    tr = t["close"].reindex(out.index)
    out["tr_close"] = tr
    div = (tr / tr.shift(1)) / (out["close"] / out["close"].shift(1)) - 1.0
    out["div_ret"] = div.where(div.abs() > 1e-5, 0.0).fillna(0.0)
    out["dollar_volume"] = out["close_raw"] * r["volume"].reindex(out.index)
    cf = cal.frame()
    on_session = out.index.isin(cf.index)
    out = out.loc[on_session]
    out["t_close"] = pd.DatetimeIndex(cf["close"].reindex(out.index))
    out.index.name = "session"
    return out


def known_at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Rows complete at `ts` (t_close ≤ ts). The only sanctioned way for a
    decision at `ts` to read bars."""
    if df.empty:
        return df
    return df.loc[df["t_close"] <= ts]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_key(ticker: str, asof: str | None) -> str:
    return f"{ticker}@{asof}" if asof else ticker


def parse_key(key: str) -> tuple[str, str | None]:
    t, _, a = key.partition("@")
    return t, (a or None)


class BarStore:
    """Parquet files per (ticker, asof) key + a manifest of what was fetched."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(self._manifest_path.read_text()) if self._manifest_path.exists() else {})

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.parquet"

    def has(self, kind: str, key: str) -> bool:
        return self._path(kind, key).exists()

    def write(self, kind: str, key: str, df: pd.DataFrame, **meta: Any) -> None:
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, compression="zstd")
        tmp.replace(p)
        entry = self.manifest.setdefault(key, {})
        entry[kind] = {
            "rows": len(df),
            "start": str(df.index.min()) if len(df) else None,
            "end": str(df.index.max()) if len(df) else None,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **meta,
        }

    def read(self, kind: str, key: str, segment: str | None = "research") -> pd.DataFrame:
        """Load a frame, clipped to a split. Default 'research' (train +
        validation) so nothing reads the final test by accident; pass
        segment=None for data-quality work, or 'final_test' via FinalTestLock."""
        p = self._path(kind, key)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        return clip_segment(df, segment)

    def save_manifest(self) -> None:
        """Merge into the on-disk manifest, so several download processes
        writing different keys/kinds don't erase each other's entries."""
        disk = json.loads(self._manifest_path.read_text()) if self._manifest_path.exists() else {}
        for key, kinds in self.manifest.items():
            disk.setdefault(key, {}).update(kinds)
        self.manifest = disk
        tmp = self._manifest_path.with_suffix(f".{id(self)}.tmp")
        tmp.write_text(json.dumps(disk, indent=1, sort_keys=True))
        tmp.replace(self._manifest_path)


def clip_segment(df: pd.DataFrame, segment: str | None) -> pd.DataFrame:
    if segment is None or df.empty:
        return df
    if segment == "final_test":
        if not FinalTestLock().opened:
            raise PermissionError("final test segment requested without FinalTestLock.open()")
        lo, hi = config.segment("warmup")[0], config.segment("test")[1]
    else:
        lo, hi = config.segment("warmup")[0], config.segment(segment)[1]
    days = df["session"] if "session" in df else (
        pd.Series(df.index, index=df.index) if df.index.tz is None else
        pd.Series(session_days(df.index), index=df.index))
    days = pd.DatetimeIndex(days)
    return df.loc[(days >= lo) & (days <= hi)]


class FinalTestLock:
    """One-shot gate for the final test segment.

    `open()` writes a lock file with the config hash and a purpose; a second
    `open()` raises. The final evaluation is therefore provably run once, and
    the lock file is the receipt.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.path = (root or config.root()) / "FINAL_TEST_LOCK.json"

    @property
    def opened(self) -> bool:
        return self.path.exists()

    def open(self, purpose: str) -> dict[str, Any]:
        if self.opened:
            raise PermissionError(f"final test already used: {self.path.read_text()}")
        rec = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "purpose": purpose, "config_hash": config.config_hash()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=1))
        return rec


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
