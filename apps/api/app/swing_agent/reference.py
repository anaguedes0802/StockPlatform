"""Daily reference series Alpaca does not carry: the VIX index.

Primary source is Cboe's own history file (daily OHLC since 1990); Yahoo's
^VIX is the fallback, per the "yfinance only as a daily fallback" rule. The
VIX settles at 16:15 ET, a quarter-hour after the stock close, so `t_close`
is the session close + 15 minutes: a decision at 16:00 cannot see that day's
VIX close.
"""
from __future__ import annotations

import io

import httpx
import pandas as pd

from app.core.logging import log
from app.swing_agent.calendar import TradingCalendar

CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def vix_daily(cal: TradingCalendar) -> tuple[pd.DataFrame, str]:
    try:
        r = httpx.get(CBOE_VIX, timeout=30, follow_redirects=True)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df.index = pd.to_datetime(df.pop("date"), format="%m/%d/%Y")
        source = "cboe"
    except Exception as e:  # noqa: BLE001
        log.warning("vix_cboe_failed", err=str(e))
        from app.services import swing_data

        df = swing_data.daily_bars("^VIX")
        df.index = pd.DatetimeIndex(df.index.tz_convert(None).normalize())
        source = "yahoo"
    df = df[["open", "high", "low", "close"]].astype(float)
    cf = cal.frame()
    df = df[df.index.isin(cf.index)]
    df.index.name = "session"
    df["t_close"] = pd.DatetimeIndex(cf["close"].reindex(df.index)) + pd.Timedelta(minutes=15)
    return df, source
