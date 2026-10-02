"""The one accessor later phases use: corrected daily / 15m / 1h bars per key.

Reads Alpaca's stored bars and applies `adjustments.json` (built by
`scripts/swing_agent_data.py corporate`): invalid split steps undone,
split_factor / div_ret / tr_close / tr_factor rebuilt. Every frame is clipped
to a split ('research' by default; see store.clip_segment).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.swing_agent import config, corporate
from app.swing_agent import store as st
from app.swing_agent.calendar import TradingCalendar


class MarketData:
    def __init__(self, root: Path | None = None, segment: str | None = "research") -> None:
        self.root = root or config.root()
        self.segment = segment
        self.store = st.BarStore(self.root)
        p = self.root / "adjustments.json"
        self._adj = json.loads(p.read_text()) if p.exists() else {}
        self.calendar = TradingCalendar.load(self.root / "calendar.json")

    def _splits(self, key: str) -> pd.DataFrame:
        rows = (self._adj.get(key) or {}).get("splits") or []
        df = pd.DataFrame(rows, columns=["ex_date", "ratio", "observed_ratio", "state"])
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        return df

    def _dists(self, key: str) -> pd.DataFrame:
        rows = (self._adj.get(key) or {}).get("distributions") or []
        df = pd.DataFrame(rows, columns=["ex_date", "amount_raw", "kind", "new_symbol"])
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        return df

    def daily(self, key: str) -> pd.DataFrame:
        src = self.store.read("daily", key, segment=None)
        if src.empty or key.startswith("^"):
            return st.clip_segment(src, self.segment)
        if key not in self._adj:
            raise KeyError(f"no adjustments for {key}; run the 'corporate' step")
        out = corporate.adjust_daily(src, self._splits(key), self._dists(key))
        return st.clip_segment(out, self.segment)

    def m15(self, key: str) -> pd.DataFrame:
        """Regular-hours 15m bars, split steps fixed like the daily series,
        plus the session's `tr_factor` and `split_factor` from the daily
        frame (signal prices = price × tr_factor; raw = price × split_factor)."""
        df = self.store.read("m15", key, segment=None)
        if df.empty:
            return df
        df = df.copy()
        fix = corporate.intraday_fix(df["session"], self._splits(key))
        if (fix != 1).any():
            for c in ("open", "high", "low", "close", "vwap"):
                df[c] = df[c] * fix
            df["volume"] = df["volume"] / fix
        if key in self._adj:
            d = corporate.adjust_daily(self.store.read("daily", key, segment=None),
                                       self._splits(key), self._dists(key))
            for c in ("tr_factor", "split_factor"):
                df[c] = d[c].reindex(df["session"]).to_numpy()
        else:
            df["tr_factor"], df["split_factor"] = 1.0, 1.0
        return st.clip_segment(df, self.segment)

    def h1(self, key: str) -> pd.DataFrame:
        return st.resample_hourly(self.m15(key), self.calendar)

    def premarket(self, key: str) -> pd.DataFrame:
        return self.store.read("premarket", key, segment=None)

    def cash_merger(self, key: str) -> float | None:
        return (self._adj.get(key) or {}).get("cash_merger")
