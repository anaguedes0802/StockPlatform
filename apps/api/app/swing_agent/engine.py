"""Portfolio backtest engine on 15m bars, shared by every phase (and by paper).

One account, one cash balance, bar by bar:

1. Session start: dividends are credited to shares held overnight; a held
   stock whose data has ended is closed at its cash take-out price if Alpaca
   recorded one, else its last close.
2. Each 15m bar, in order:
   a. orders sent at the previous bar's close fill at this bar's OPEN
      (market order: half-spread + slippage, against the trader);
   b. open positions check their stop first, then their take-profit. A bar
      that opens through the stop fills at the open (the gap); otherwise a
      low at or under the stop fills at the stop. If the same bar touches
      both, the stop wins. A position filled at this bar's open only checks
      its stop (we can't know the intrabar order);
   c. at the bar's close, trailing stops ratchet (never down); end-of-day
      exits (time limit, strategy exit, earnings tomorrow, data ending) are
      decided on the bar that closes 15 minutes before the close;
   d. new signals on this bar are filtered (universe that month and still an
      index member, earnings and macro blocks, max positions, one position
      per stock), ranked by score, sized, and sent for the next bar.
3. Session end: decided exits fill in the closing auction at the official
   close; unfilled entry orders are cancelled; equity is marked at the
   official closes.

Sizing: shares = 1% of equity ÷ (stop distance), whole raw shares, capped at
25% of equity per position, 100% gross (cash account), the cash actually
free, and 1% of the stock's median 15m volume.

Prices are split-adjusted execution prices; raw price = price × split
factor, used for costs and whole-share sizing.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.swing_agent import costs as cx

SLOTS = 32  # > 26 regular-hours 15m bars per session


@dataclass
class Book:
    """Everything the engine needs about one key, aligned to the session index."""
    key: str
    is_etf: bool
    sess: np.ndarray            # global session index per bar
    slot: np.ndarray
    mod: np.ndarray             # minute of day (ET) at the bar's close
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    sf: np.ndarray              # split factor per bar
    med_vol: np.ndarray         # median 15m volume, previous 20 sessions (adjusted shares)
    micro: np.ndarray           # intraday regime known at the bar
    stock_regime: np.ndarray    # stock's daily regime as of yesterday
    sig: dict[str, dict[str, np.ndarray]]   # strategy → column → array
    daily_close: np.ndarray     # per global session (NaN if no bar)
    daily_sf: np.ndarray
    div_amt: np.ndarray         # dividend per adjusted share on its ex-session
    last_session: int
    react: list[int]            # earnings reaction session indices, sorted
    cash_merger: float | None = None
    row_at: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = int(self.sess.max()) + 2 if len(self.sess) else 1
        self.row_at = np.full(n * SLOTS, -1, dtype=np.int64)
        self.row_at[self.sess * SLOTS + self.slot] = np.arange(len(self.sess))

    def row(self, si: int, k: int) -> int:
        j = si * SLOTS + k
        return int(self.row_at[j]) if 0 <= j < len(self.row_at) else -1


@dataclass
class Pos:
    key: str
    strategy: str
    entry_si: int
    entry_mod: int
    shares: float               # adjusted shares
    entry: float
    stop: float
    stop0: float
    target: float
    trail_after_px: float
    trail_dist: float
    max_sessions: int
    tags: dict[str, Any]
    trail_on: bool = False
    max_close: float = -math.inf
    min_low: float = math.inf
    max_high: float = -math.inf
    costs: float = 0.0
    divs: float = 0.0
    fresh: bool = True
    moc_reason: str | None = None


@dataclass
class Order:
    key: str
    strategy: str
    shares_raw: int
    stop_dist: float
    target_dist: float
    trail_after: float
    trail_dist: float
    max_sessions: int
    tags: dict[str, Any]
    notional: float = 0.0       # estimated at the signal close, for cash / gross checks


@dataclass
class Context:
    sessions: list[pd.Timestamp]                 # global session dates
    n_slots: list[int]                            # bars per session
    universe: dict[str, set[str]]                 # "YYYY-MM" → keys tradable that month
    member: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]   # key → membership interval
    refs: set[str]                                # always tradable (SPY, QQQ)
    market_regime: dict[int, str]                 # session index → regime known that morning
    fomc: set[int]
    late_start: dict[int, int]                    # session index → earliest signal minute


def _fill(book: Book, i: int, px: float, side: int, kind: str, shares_adj: float,
          c: dict[str, Any]) -> tuple[float, float]:
    """Execution price and $ cost (spread+slippage+fees) of one fill on bar i."""
    sf = book.sf[i]
    raw, fees = cx.fill(px * sf, side, kind, shares_adj / sf, c)
    exec_px = raw / sf
    return exec_px, abs(exec_px - px) * shares_adj + fees


def run(books: dict[str, Book], ctx: Context, strategies: dict[str, str], ex: dict[str, Any],
        c: dict[str, Any], capital: float, start_si: int, end_si: int) -> dict[str, Any]:
    """`strategies`: strategy name → signal column to use ('entry' or 'setup_entry')."""
    cash = capital
    pos: dict[str, Pos] = {}
    pending: list[Order] = []
    trades: list[dict[str, Any]] = []
    equity: list[tuple[pd.Timestamp, float, float, int]] = []
    last_px: dict[str, float] = {}
    cand: dict[int, list[tuple[float, str, str, int]]] = {}
    for key, b in books.items():
        for sname, col in strategies.items():
            s = b.sig.get(sname)
            if s is None:
                continue
            idx = np.flatnonzero(s[col])
            for i in idx:
                si = int(b.sess[i])
                if start_si <= si <= end_si:
                    cand.setdefault(si * SLOTS + int(b.slot[i]), []).append(
                        (float(s["score"][i]) if np.isfinite(s["score"][i]) else 0.0, key, sname, int(i)))
    rpt, max_pos = ex["risk_per_trade"], ex["max_positions"]

    def close_pos(p: Pos, px: float, kind: str, reason: str, si: int, mod: int, i: int) -> None:
        nonlocal cash
        b = books[p.key]
        fpx, cost = _fill(b, i, px, -1, kind, p.shares, c) if i >= 0 else _fill_daily(b, si, px, kind, p.shares, c)
        exit_fees = cost - abs(fpx - px) * p.shares        # spread/slippage are already in fpx
        cash += fpx * p.shares - exit_fees
        p.costs += cost
        risk = (p.entry - p.stop0) * p.shares
        pnl = (fpx - p.entry) * p.shares + p.divs - _entry_fees(p) - exit_fees
        trades.append({
            "key": p.key, "strategy": p.strategy, "entry_date": ctx.sessions[p.entry_si],
            "entry_mod": p.entry_mod, "exit_date": ctx.sessions[si], "exit_mod": mod, "reason": reason,
            "entry": p.entry, "exit": fpx, "stop0": p.stop0, "shares": p.shares, "pnl": pnl,
            "costs": p.costs, "divs": p.divs, "risk": risk, "r": pnl / risk if risk > 0 else np.nan,
            "sessions": si - p.entry_si,
            "mae_r": (p.min_low - p.entry) / (p.entry - p.stop0) if p.entry > p.stop0 else np.nan,
            "mfe_r": (p.max_high - p.entry) / (p.entry - p.stop0) if p.entry > p.stop0 else np.nan,
            **p.tags})
        del pos[p.key]

    for si in range(start_si, end_si + 1):
        day = ctx.sessions[si]
        # 1. session start: dividends, ended data
        for p in list(pos.values()):
            b = books[p.key]
            amt = b.div_amt[si] if si < len(b.div_amt) else 0.0
            if amt and np.isfinite(amt):
                cash += amt * p.shares
                p.divs += amt * p.shares
            if b.last_session < si:
                px = b.cash_merger / b.daily_sf[b.last_session] if b.cash_merger else b.daily_close[b.last_session]
                close_pos(p, px, "limit", "delisted", b.last_session, 0, -1)
        nk = ctx.n_slots[si]
        for k in range(nk):
            # 2a. fills of orders sent last bar
            still = []
            for od in pending:
                b = books[od.key]
                i = b.row(si, k)
                if i < 0:
                    still.append(od)
                    continue
                shares_adj = od.shares_raw * b.sf[i]
                fpx, cost = _fill(b, i, b.o[i], +1, "market", shares_adj, c)
                need = fpx * shares_adj + (cost - abs(fpx - b.o[i]) * shares_adj)
                if need > cash:
                    n_raw = int((cash * 0.995) // (fpx * b.sf[i]))
                    if n_raw < 1:
                        continue
                    shares_adj = n_raw * b.sf[i]
                    fpx, cost = _fill(b, i, b.o[i], +1, "market", shares_adj, c)
                fees = cost - abs(fpx - b.o[i]) * shares_adj
                cash -= fpx * shares_adj + fees
                p = Pos(od.key, od.strategy, si, int(b.mod[i]), shares_adj, fpx, fpx - od.stop_dist,
                        fpx - od.stop_dist,
                        fpx + od.target_dist if np.isfinite(od.target_dist) else math.inf,
                        fpx + od.trail_after if np.isfinite(od.trail_after) else math.inf,
                        od.trail_dist if np.isfinite(od.trail_dist) else math.nan,
                        od.max_sessions, od.tags)
                p.costs, p.tags = cost, {**od.tags, "entry_fees": fees}
                pos[od.key] = p
            pending = still
            # 2b/2c. manage open positions on this bar
            for p in list(pos.values()):
                b = books[p.key]
                i = b.row(si, k)
                if i < 0:
                    continue
                o, h, l, cl = b.o[i], b.h[i], b.l[i], b.c[i]
                mod = int(b.mod[i])
                if not p.fresh and o <= p.stop:
                    why = ("trail" if p.trail_on else "stop") + ("_gap" if k == 0 else "")
                    close_pos(p, o, "stop", why, si, mod, i)
                    continue
                if l <= p.stop:
                    close_pos(p, p.stop, "stop", "trail" if p.trail_on else "stop", si, mod, i)
                    continue
                if not p.fresh and math.isfinite(p.target):
                    if o >= p.target:
                        close_pos(p, o, "limit", "target", si, mod, i)
                        continue
                    if h > p.target:
                        close_pos(p, p.target, "limit", "target", si, mod, i)
                        continue
                p.fresh = False
                p.min_low, p.max_high = min(p.min_low, l), max(p.max_high, h)
                p.max_close = max(p.max_close, cl)
                last_px[p.key] = cl
                if math.isfinite(p.trail_dist):
                    if not p.trail_on and cl >= p.trail_after_px:
                        p.trail_on = True
                    if p.trail_on:
                        p.stop = max(p.stop, p.max_close - p.trail_dist)
                if k == nk - 2 and p.moc_reason is None:
                    s = b.sig.get(p.strategy, {})
                    if si - p.entry_si >= p.max_sessions:
                        p.moc_reason = "time"
                    elif "exit_moc" in s and bool(s["exit_moc"][i]) and si > p.entry_si:
                        p.moc_reason = "strategy_exit"
                    elif ex.get("exit_before_earnings") and _reacts(b, si + 1, si + 1):
                        p.moc_reason = "earnings"
                    elif b.last_session == si:
                        p.moc_reason = "data_ends"
            # 2d. new signals
            cands = cand.get(si * SLOTS + k)
            if cands:
                month = f"{day.year:04d}-{day.month:02d}"
                eq = cash + sum(p.shares * last_px.get(p.key, p.entry) for p in pos.values())
                committed = sum(o.notional for o in pending)
                gross = sum(p.shares * last_px.get(p.key, p.entry) for p in pos.values()) + committed
                for score, key, sname, i in sorted(cands, key=lambda x: -x[0]):
                    if len(pos) + len(pending) >= max_pos:
                        break
                    if key in pos or any(o.key == key for o in pending):
                        continue
                    if not _tradable(key, day, month, ctx):
                        continue
                    b = books[key]
                    blk = ex["earnings_block_sessions"]       # < 0: off (intraday, flat at the close)
                    if blk >= 0 and not b.is_etf and _reacts(b, si, si + blk):
                        continue
                    if si in ctx.fomc or b.mod[i] < ctx.late_start.get(si, 0):
                        continue
                    s = b.sig[sname]
                    sd = float(s["stop_dist"][i])
                    if not (np.isfinite(sd) and sd > 0):
                        continue
                    px = b.c[i]
                    sf = b.sf[i]
                    n = math.floor(rpt * eq / (sd * sf))
                    n = min(n, math.floor(ex["max_position_pct"] * eq / (px * sf)))
                    n = min(n, math.floor(max(0.0, ex["max_gross"] * eq - gross) / (px * sf)))
                    n = min(n, math.floor(max(0.0, cash * 0.995 - committed) / (px * sf)))
                    if np.isfinite(b.med_vol[i]):
                        n = min(n, math.floor(ex["max_order_pct_volume"] * b.med_vol[i] / sf))
                    if n < 1:
                        continue
                    tags = {"market_regime": ctx.market_regime.get(si), "micro": b.micro[i],
                            "stock_regime": b.stock_regime[i], "signal_mod": int(b.mod[i])}
                    pending.append(Order(key, sname, n, sd, float(s["target_dist"][i]),
                                         float(s["trail_after"][i]), float(s["trail_dist"][i]),
                                         int(s["max_sessions"][i]), tags, n * sf * px))
                    committed += n * sf * px
                    gross += n * sf * px
        # 3. session end
        pending = []
        for p in list(pos.values()):
            if p.moc_reason:
                b = books[p.key]
                px = b.daily_close[si] if np.isfinite(b.daily_close[si]) else last_px.get(p.key, p.entry)
                close_pos(p, px, "moc", p.moc_reason, si, 960, -1)
        mtm = 0.0
        for p in pos.values():
            b = books[p.key]
            px = b.daily_close[si] if si < len(b.daily_close) and np.isfinite(b.daily_close[si]) \
                else last_px.get(p.key, p.entry)
            last_px[p.key] = px
            mtm += p.shares * px
        equity.append((day, cash + mtm, mtm, len(pos)))
    # close what is left at the last mark
    for p in list(pos.values()):
        b = books[p.key]
        close_pos(p, last_px.get(p.key, p.entry), "moc", "end", end_si, 960, -1)
    eq = pd.DataFrame(equity, columns=["session", "equity", "exposure", "positions"]).set_index("session")
    return {"equity": eq, "trades": pd.DataFrame(trades)}


def _entry_fees(p: Pos) -> float:
    return float(p.tags.get("entry_fees", 0.0))


def _fill_daily(b: Book, si: int, px: float, kind: str, shares_adj: float,
                c: dict[str, Any]) -> tuple[float, float]:
    sf = b.daily_sf[si] if si < len(b.daily_sf) and np.isfinite(b.daily_sf[si]) else 1.0
    raw, fees = cx.fill(px * sf, -1, kind, shares_adj / sf, c)
    exec_px = raw / sf
    return exec_px, abs(exec_px - px) * shares_adj + fees


def _reacts(b: Book, lo: int, hi: int) -> bool:
    j = bisect.bisect_left(b.react, lo)
    return j < len(b.react) and b.react[j] <= hi


def _tradable(key: str, day: pd.Timestamp, month: str, ctx: Context) -> bool:
    if key in ctx.refs:
        return True
    if key not in ctx.universe.get(month, ()):
        return False
    lo, hi = ctx.member.get(key, (None, None))
    return (lo is None or day >= lo) and (hi is None or day <= hi)
