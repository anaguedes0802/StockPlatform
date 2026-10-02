"""NYSE trading calendar: sessions, holidays, early closes, special closures.

Source: Alpaca's `/v2/calendar` (the broker's own calendar, so backtest and
live agree). It already encodes holidays, 13:00 half-days and one-off
closures such as 2018-12-05 (Bush) and 2025-01-09 (Carter). Times are given
in New York local time; converting through `zoneinfo` makes every session's
UTC open/close correct across DST changes.

Cached as JSON under `<root>/calendar.json`.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    day: date
    open: pd.Timestamp    # UTC
    close: pd.Timestamp   # UTC

    @property
    def early_close(self) -> bool:
        return self.close.tz_convert(NY).time() < time(16, 0)

    @property
    def minutes(self) -> int:
        return int((self.close - self.open).total_seconds() // 60)


def _utc(d: date, hhmm: str) -> pd.Timestamp:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY)).tz_convert("UTC")


class TradingCalendar:
    def __init__(self, sessions: list[Session]) -> None:
        self.sessions = sorted(sessions, key=lambda s: s.day)
        self._days = [s.day for s in self.sessions]
        self._by_day = {s.day: s for s in self.sessions}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> TradingCalendar:
        out = []
        for r in rows:
            d = date.fromisoformat(r["date"])
            out.append(Session(d, _utc(d, r["open"]), _utc(d, r["close"])))
        return cls(out)

    @classmethod
    def load(cls, path: Path) -> TradingCalendar:
        return cls.from_rows(json.loads(path.read_text()))

    @classmethod
    def fetch(cls, client: Any, start: str, end: str, path: Path | None = None) -> TradingCalendar:
        rows = client.calendar(start, end)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows))
        return cls.from_rows(rows)

    # -- queries ---------------------------------------------------------------

    @property
    def first(self) -> date:
        return self._days[0]

    @property
    def last(self) -> date:
        return self._days[-1]

    def session(self, d: date) -> Session | None:
        return self._by_day.get(_as_date(d))

    def is_session(self, d: date) -> bool:
        return _as_date(d) in self._by_day

    def between(self, start: date, end: date) -> list[Session]:
        lo = bisect.bisect_left(self._days, _as_date(start))
        hi = bisect.bisect_right(self._days, _as_date(end))
        return self.sessions[lo:hi]

    def previous(self, d: date, n: int = 1) -> Session:
        """The n-th session strictly before `d`."""
        i = bisect.bisect_left(self._days, _as_date(d)) - n
        if i < 0:
            raise KeyError(f"no session {n} before {d}")
        return self.sessions[i]

    def next(self, d: date, n: int = 1) -> Session:
        """The n-th session strictly after `d`."""
        i = bisect.bisect_right(self._days, _as_date(d)) + n - 1
        if i >= len(self.sessions):
            raise KeyError(f"no session {n} after {d}")
        return self.sessions[i]

    def session_at(self, ts: pd.Timestamp) -> Session | None:
        """The session whose regular hours contain `ts` (open ≤ ts < close)."""
        s = self.session(pd.Timestamp(ts).tz_convert(NY).date())
        return s if s and s.open <= ts < s.close else None

    def last_closed(self, ts: pd.Timestamp) -> Session | None:
        """The latest session whose close is ≤ ts — the newest daily bar one may use."""
        d = pd.Timestamp(ts).tz_convert(NY).date()
        s = self.session(d)
        if s and s.close <= ts:
            return s
        try:
            return self.previous(d)
        except KeyError:
            return None

    def month_starts(self, start: date, end: date) -> list[Session]:
        """First session of each calendar month in [start, end]."""
        out, seen = [], set()
        for s in self.between(start, end):
            k = (s.day.year, s.day.month)
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": [s.open for s in self.sessions], "close": [s.close for s in self.sessions],
             "early_close": [s.early_close for s in self.sessions]},
            index=pd.DatetimeIndex([pd.Timestamp(s.day) for s in self.sessions], name="session"),
        )


def _as_date(d: Any) -> date:
    """A calendar date. Tz-aware timestamps are read in New York time, so
    20:00 ET (00:00 UTC next day) still belongs to that ET date."""
    if isinstance(d, datetime):
        if d.tzinfo is not None:
            return pd.Timestamp(d).tz_convert(NY).date()
        return d.date()
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    return d
