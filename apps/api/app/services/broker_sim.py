"""Simulated paper broker: a private ledger per bot, same interface as Alpaca.

Why: two bots on one Alpaca paper account contaminate each other. They share
buying power, each would close the other's positions, and a filtered bot's
trades are a subset of the unfiltered bot's, so they collide on the same
symbols. An A/B test needs independent accounts. This broker gives every bot
its own ledger (stored in `run_state["sim_ledger"]`) while running the exact
production code path (`trading_bot.run_live`).

Fill model
* Market orders fill at the live quote, plus/minus `slippage_bps`, plus
  `commission_bps` on notional.
* Bracket legs rest between runs. `process_resting_orders` replays 5-minute
  bars *after* the fill: a bar trading through the stop fills the stop (or
  that bar's open if it gapped through), a bar reaching the target fills the
  target (or the open if it gapped past). If one bar touches both, the stop is
  assumed first. Stops pay slippage, targets are limits and don't.
* Fractional quantities are allowed, but `run_live` sends whole shares for
  brackets anyway.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")
DEFAULT_CASH = 100_000.0
MAX_FILLS = 5_000
BAR = pd.Timedelta(minutes=5)


def _default_quote(symbol: str) -> float:
    from app.services import market_data as md
    q = md.get_quote(symbol)
    px = float(q.get("price") or 0.0)
    if px <= 0:
        raise RuntimeError(f"no live price for {symbol}")
    return px


def _default_intraday(symbol: str, start: datetime) -> pd.DataFrame:
    from app.services import swing_data as sd
    return sd.intraday_bars(symbol, start)


def _default_clock() -> bool:
    from app.services import broker_alpaca
    if broker_alpaca.is_configured():
        try:
            return broker_alpaca.is_market_open()
        except Exception:  # noqa: BLE001 — fall back to the calendar-free check
            pass
    et = datetime.now(_ET)
    return et.weekday() < 5 and time(9, 30) <= et.time() < time(16, 0)


def new_ledger(initial_cash: float = DEFAULT_CASH, now: datetime | None = None) -> dict[str, Any]:
    return {"initial_cash": float(initial_cash), "cash": float(initial_cash),
            "created_at": (now or datetime.now(timezone.utc)).isoformat(),
            "positions": {}, "fills": [], "equity_history": [], "seq": 0}


class SimBroker:
    """In-process paper broker. Mutates `self.ledger`; persist it after use."""

    def __init__(self, ledger: dict[str, Any] | None = None, *, initial_cash: float = DEFAULT_CASH,
                 slippage_bps: float = 5.0, commission_bps: float = 1.0,
                 quote_fn: Callable[[str], float] | None = None,
                 intraday_fn: Callable[[str, datetime], pd.DataFrame] | None = None,
                 clock_fn: Callable[[], bool] | None = None,
                 now_fn: Callable[[], datetime] | None = None):
        self.ledger = ledger if ledger else new_ledger(initial_cash)
        self.slip = slippage_bps / 1e4
        self.comm = commission_bps / 1e4
        self._quote = quote_fn or _default_quote
        self._intraday = intraday_fn or _default_intraday
        self._clock = clock_fn or _default_clock
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._px_cache: dict[str, float] = {}

    # ---------------- interface used by trading_bot.run_live ----------------

    def is_configured(self) -> bool:
        return True

    def is_paper(self) -> bool:
        return True

    def is_market_open(self) -> bool:
        return bool(self._clock())

    def _price(self, symbol: str) -> float:
        sym = symbol.upper()
        if sym not in self._px_cache:
            self._px_cache[sym] = float(self._quote(sym))
        return self._px_cache[sym]

    def _mark_price(self, sym: str, pos: dict[str, Any]) -> float:
        try:
            px = self._price(sym)
            pos["last_price"] = px
            return px
        except Exception:  # noqa: BLE001 — fall back to the last known price
            return float(pos.get("last_price") or pos["avg_entry_price"])

    def get_account(self) -> dict[str, Any]:
        mv = sum(p["qty"] * self._mark_price(s, p) for s, p in self.ledger["positions"].items())
        eq = self.ledger["cash"] + mv
        return {"account_number": "SIM", "status": "ACTIVE", "currency": "USD",
                "cash": round(self.ledger["cash"], 2), "equity": round(eq, 2),
                "buying_power": round(max(self.ledger["cash"], 0.0), 2),
                "portfolio_value": round(eq, 2), "pattern_day_trader": False,
                "trading_blocked": False, "is_paper": True}

    def list_positions(self) -> list[dict[str, Any]]:
        out = []
        for sym, p in self.ledger["positions"].items():
            px = self._mark_price(sym, p)
            upl = p["qty"] * (px - p["avg_entry_price"])
            out.append({"symbol": sym, "qty": p["qty"], "avg_entry_price": p["avg_entry_price"],
                        "current_price": px, "market_value": p["qty"] * px, "unrealized_pl": upl,
                        "unrealized_plpc": (px / p["avg_entry_price"] - 1) * 100, "side": "long"})
        return out

    def _next_id(self, prefix: str) -> str:
        self.ledger["seq"] = int(self.ledger.get("seq", 0)) + 1
        return f"sim-{prefix}-{self.ledger['seq']}"

    def _record(self, **fill: Any) -> dict[str, Any]:
        fill.setdefault("ts", self._now().isoformat())
        self.ledger["fills"].append(fill)
        self.ledger["fills"] = self.ledger["fills"][-MAX_FILLS:]
        return fill

    def _buy(self, symbol: str, *, qty: float | None, notional: float | None) -> tuple[str, float, float]:
        sym = symbol.upper()
        px = self._price(sym) * (1 + self.slip)
        if qty is None:
            if not notional or notional <= 0:
                raise ValueError("market order needs qty or notional")
            qty = notional / (px * (1 + self.comm))
        qty = float(qty)
        cost = qty * px * (1 + self.comm)
        if qty <= 0 or cost > self.ledger["cash"] + 1e-6:
            raise ValueError(f"insufficient simulated cash for {sym}: need {cost:.2f}, have {self.ledger['cash']:.2f}")
        self.ledger["cash"] -= cost
        pos = self.ledger["positions"].get(sym)
        now = self._now().isoformat()
        if pos:
            tot = pos["qty"] + qty
            pos["avg_entry_price"] = (pos["avg_entry_price"] * pos["qty"] + px * qty) / tot
            pos["qty"] = tot
        else:
            self.ledger["positions"][sym] = {"qty": qty, "avg_entry_price": px, "opened_at": now,
                                             "stop": None, "target": None, "stop_order_id": None,
                                             "checked_through": now, "last_price": px}
        oid = self._next_id("o")
        self._record(order_id=oid, symbol=sym, side="buy", qty=round(qty, 6), price=round(px, 4),
                     reason="market")
        return oid, qty, px

    def submit_market_order(self, symbol: str, *, notional: float | None = None, qty: float | None = None,
                            side: str = "buy") -> dict[str, Any]:
        if side != "buy":
            return self.close_position(symbol)
        oid, q, px = self._buy(symbol, qty=qty, notional=notional)
        return {"id": oid, "symbol": symbol.upper(), "side": "buy", "qty": str(q), "notional": notional,
                "type": "market", "status": "filled", "filled_avg_price": str(round(px, 4))}

    def submit_bracket_order(self, symbol: str, *, notional: float | None = None, qty: float | None = None,
                             side: str = "buy", stop_loss_price: float | None = None,
                             take_profit_price: float | None = None,
                             time_in_force: str = "gtc") -> dict[str, Any]:
        oid, q, px = self._buy(symbol, qty=qty, notional=notional)
        pos = self.ledger["positions"][symbol.upper()]
        pos["stop"] = float(stop_loss_price) if stop_loss_price else None
        pos["target"] = float(take_profit_price) if take_profit_price else None
        pos["stop_order_id"] = self._next_id("stop") if stop_loss_price else None
        legs = []
        if pos["stop_order_id"]:
            legs.append({"id": pos["stop_order_id"], "type": "stop", "side": "sell",
                         "stop_price": pos["stop"], "status": "new"})
        if pos["target"]:
            legs.append({"id": self._next_id("tp"), "type": "limit", "side": "sell",
                         "limit_price": pos["target"], "status": "new"})
        return {"id": oid, "symbol": symbol.upper(), "side": "buy", "qty": str(q), "type": "market",
                "status": "filled", "filled_avg_price": str(round(px, 4)),
                "order_class": "bracket" if pos["target"] and pos["stop"] else "oto",
                "time_in_force": time_in_force, "legs": legs}

    def cancel_open_orders(self, symbol: str) -> int:
        pos = self.ledger["positions"].get(symbol.upper())
        if not pos:
            return 0
        n = int(pos.get("stop") is not None) + int(pos.get("target") is not None)
        pos["stop"] = pos["target"] = pos["stop_order_id"] = None
        return n

    def replace_stop(self, order_id: str, stop_price: float) -> dict[str, Any]:
        for sym, pos in self.ledger["positions"].items():
            if pos.get("stop_order_id") == order_id:
                pos["stop"] = float(stop_price)
                return {"id": order_id, "symbol": sym, "stop_price": stop_price, "status": "replaced"}
        raise ValueError(f"no resting stop {order_id}")

    def _sell(self, sym: str, px: float, *, reason: str, ts: str | None = None) -> dict[str, Any]:
        pos = self.ledger["positions"].pop(sym)
        proceeds = pos["qty"] * px * (1 - self.comm)
        self.ledger["cash"] += proceeds
        realized = proceeds - pos["qty"] * pos["avg_entry_price"]
        return self._record(order_id=self._next_id("o"), symbol=sym, side="sell", qty=round(pos["qty"], 6),
                            price=round(px, 4), reason=reason, realized_pl=round(realized, 2),
                            entry_price=round(pos["avg_entry_price"], 4), opened_at=pos.get("opened_at"),
                            **({"ts": ts} if ts else {}))

    def close_position(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        if sym not in self.ledger["positions"]:
            return {"symbol": sym, "status": "no_position"}
        px = self._price(sym) * (1 - self.slip)
        f = self._sell(sym, px, reason="market")
        return {"id": f["order_id"], "symbol": sym, "side": "sell", "qty": str(f["qty"]),
                "type": "market", "status": "filled", "filled_avg_price": str(f["price"]),
                "realized_pl": f["realized_pl"]}

    # ---------------- simulation-only ----------------

    def process_resting_orders(self) -> list[dict[str, Any]]:
        """Fill stop/target legs touched since each position was last checked."""
        fills = []
        for sym in list(self.ledger["positions"]):
            pos = self.ledger["positions"][sym]
            if pos.get("stop") is None and pos.get("target") is None:
                continue
            since = pd.Timestamp(pos.get("checked_through") or pos["opened_at"]).to_pydatetime()
            try:
                bars = self._intraday(sym, since)
            except Exception:  # noqa: BLE001 — try again next run
                continue
            if bars is None or bars.empty:
                continue
            # Only bars that *start* after the last check (the bar containing
            # the fill may have traded through the stop before we owned it) and
            # have fully closed (a forming bar is re-read next run).
            now = pd.Timestamp(self._now())
            bars = bars[(bars.index > pd.Timestamp(since)) & (bars.index + BAR <= now)]
            done = False
            for ts, b in bars.iterrows():
                stop, tgt = pos.get("stop"), pos.get("target")
                if stop is not None and b["low"] <= stop:
                    px = min(float(b["open"]), stop) * (1 - self.slip)
                    fills.append(self._sell(sym, px, reason="stop", ts=ts.isoformat()))
                    done = True
                    break
                if tgt is not None and b["high"] >= tgt:
                    px = max(float(b["open"]), tgt)
                    fills.append(self._sell(sym, px, reason="target", ts=ts.isoformat()))
                    done = True
                    break
            if not done and len(bars):
                pos["checked_through"] = bars.index[-1].isoformat()
                pos["last_price"] = float(bars["close"].iloc[-1])
        return fills

    def mark(self) -> dict[str, Any]:
        """Record today's equity (one point per New York date)."""
        acct = self.get_account()
        day = self._now().astimezone(_ET).date().isoformat()
        hist = [h for h in self.ledger["equity_history"] if h["date"] != day]
        hist.append({"date": day, "equity": acct["equity"]})
        self.ledger["equity_history"] = hist[-3000:]
        return acct


def summary(ledger: dict[str, Any] | None) -> dict[str, Any] | None:
    """Ledger overview from stored prices only (no network), for the UI."""
    if not ledger:
        return None
    pos = []
    mv = 0.0
    for sym, p in (ledger.get("positions") or {}).items():
        px = float(p.get("last_price") or p["avg_entry_price"])
        mv += p["qty"] * px
        pos.append({"symbol": sym, "qty": round(p["qty"], 4), "avg_entry_price": round(p["avg_entry_price"], 4),
                    "last_price": round(px, 4), "stop": p.get("stop"), "target": p.get("target"),
                    "opened_at": p.get("opened_at"),
                    "unrealized_pct": round((px / p["avg_entry_price"] - 1) * 100, 2)})
    equity = ledger["cash"] + mv
    sells = [f for f in ledger.get("fills", []) if f.get("side") == "sell"]
    wins = [f for f in sells if (f.get("realized_pl") or 0) > 0]
    return {
        "initial_cash": ledger["initial_cash"], "cash": round(ledger["cash"], 2), "equity": round(equity, 2),
        "return_pct": round((equity / ledger["initial_cash"] - 1) * 100, 3),
        "created_at": ledger.get("created_at"), "positions": pos,
        "closed_trades": len(sells),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 1) if sells else None,
        "realized_pl": round(sum(f.get("realized_pl") or 0 for f in sells), 2),
        "recent_fills": list(reversed(ledger.get("fills", [])[-30:])),
        "equity_history": ledger.get("equity_history", []),
    }
