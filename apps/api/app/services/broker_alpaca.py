"""Alpaca trading (order execution) adapter for the live bot.

This is the thin layer that actually talks to a broker. By default it points at
Alpaca's **paper** trading endpoint (`paper-api.alpaca.markets`): orders execute
in real time against live market prices but settle in a simulated account with
fake money — full live behavior, zero capital at risk. Repointing
`alpaca_trading_base_url` at `api.alpaca.markets` makes it trade real money.

Everything here is intentionally small and synchronous (matching
`alpaca_bars.py`): get the account, list positions, submit a market order,
close a position, and check whether the market is open. The bot's risk logic
(arm/disarm, daily kill-switch, position sizing) lives one layer up in
`trading_bot.run_live` — this module just executes.

Docs: https://docs.alpaca.markets/reference/
"""
from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.core.logging import log


class BrokerError(RuntimeError):
    """Raised when the broker rejects a request or is misconfigured."""


def _key() -> str | None:
    return settings.alpaca_trading_api_key or settings.alpaca_api_key


def _secret() -> str | None:
    return settings.alpaca_trading_api_secret or settings.alpaca_api_secret


def is_configured() -> bool:
    return bool(_key() and _secret())


def is_paper() -> bool:
    return "paper-api" in (settings.alpaca_trading_base_url or "")


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": _key() or "",
        "APCA-API-SECRET-KEY": _secret() or "",
        "accept": "application/json",
    }


def _base() -> str:
    return (settings.alpaca_trading_base_url or "https://paper-api.alpaca.markets").rstrip("/")


def _request(method: str, path: str, *, json: dict | None = None) -> Any:
    if not is_configured():
        raise BrokerError("Alpaca trading credentials are not configured")
    url = f"{_base()}/v2{path}"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.request(method, url, headers=_headers(), json=json)
    except httpx.HTTPError as e:  # network-level failure
        log.warning("alpaca_trade_network_error", path=path, err=str(e))
        raise BrokerError(f"broker request failed: {e}") from e
    if r.status_code == 401:
        raise BrokerError("Alpaca rejected the trading credentials (401)")
    if r.status_code >= 400:
        body = r.text[:300]
        log.warning("alpaca_trade_error", path=path, status=r.status_code, body=body)
        raise BrokerError(f"broker error {r.status_code}: {body}")
    if r.status_code == 204 or not r.content:
        return None
    return r.json()


# ---------- account / market ----------

def get_account() -> dict[str, Any]:
    """Account snapshot: equity, cash, buying power, day-trade status, etc."""
    a = _request("GET", "/account")
    return {
        "account_number": a.get("account_number"),
        "status": a.get("status"),
        "currency": a.get("currency"),
        "cash": float(a.get("cash", 0) or 0),
        "equity": float(a.get("equity", 0) or 0),
        "buying_power": float(a.get("buying_power", 0) or 0),
        "portfolio_value": float(a.get("portfolio_value", 0) or 0),
        "pattern_day_trader": bool(a.get("pattern_day_trader", False)),
        "trading_blocked": bool(a.get("trading_blocked", False)),
        "is_paper": is_paper(),
    }


def is_market_open() -> bool:
    clock = _request("GET", "/clock")
    return bool(clock.get("is_open", False))


def market_clock() -> dict[str, Any]:
    return _request("GET", "/clock")


# ---------- positions ----------

def list_positions() -> list[dict[str, Any]]:
    rows = _request("GET", "/positions") or []
    return [_norm_position(p) for p in rows]


def get_position(symbol: str) -> dict[str, Any] | None:
    try:
        p = _request("GET", f"/positions/{symbol.upper()}")
    except BrokerError as e:
        if "404" in str(e):
            return None
        raise
    return _norm_position(p)


def _norm_position(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": p.get("symbol"),
        "qty": float(p.get("qty", 0) or 0),
        "avg_entry_price": float(p.get("avg_entry_price", 0) or 0),
        "current_price": float(p.get("current_price", 0) or 0),
        "market_value": float(p.get("market_value", 0) or 0),
        "unrealized_pl": float(p.get("unrealized_pl", 0) or 0),
        "unrealized_plpc": float(p.get("unrealized_plpc", 0) or 0) * 100,
        "side": p.get("side", "long"),
    }


# ---------- orders ----------

def submit_market_order(symbol: str, *, notional: float | None = None, qty: float | None = None,
                        side: str = "buy", client_order_id: str | None = None) -> dict[str, Any]:
    """Submit a market order. Provide `notional` (dollar amount) or `qty`.

    Notional orders let Alpaca fractionalize the share count to hit a dollar
    target — convenient for sizing to a fixed per-position budget.
    """
    body: dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if notional is not None:
        body["notional"] = round(float(notional), 2)
    elif qty is not None:
        body["qty"] = str(qty)
    else:
        raise BrokerError("submit_market_order requires notional or qty")
    if client_order_id:
        body["client_order_id"] = client_order_id[:48]
    o = _request("POST", "/orders", json=body)
    return _norm_order(o)


def get_order(order_id: str) -> dict[str, Any]:
    """One order by id, including fill progress (filled_qty / filled_avg_price)."""
    o = _request("GET", f"/orders/{order_id}")
    out = _norm_order(o)
    out["filled_qty"] = o.get("filled_qty")
    out["client_order_id"] = o.get("client_order_id")
    return out


def get_asset(symbol: str) -> dict[str, Any]:
    """Tradability flags for a symbol (tradable, fractionable, ...)."""
    return _request("GET", f"/assets/{symbol.upper()}") or {}


def submit_bracket_order(symbol: str, *, notional: float | None = None, qty: float | None = None,
                         side: str = "buy", stop_loss_price: float | None = None,
                         take_profit_price: float | None = None,
                         time_in_force: str = "gtc") -> dict[str, Any]:
    """Submit a market entry with broker-side protective legs (bracket / OTO).

    A bracket attaches a stop-loss (and optionally take-profit) that lives on
    the broker, so the stop is enforced even if our bot never runs again or the
    price gaps down while we're idle. Provide `notional` or `qty` for the entry,
    plus at least a `stop_loss_price`.

    Alpaca's bracket order class requires whole-share `qty` (notional/fractional
    cannot carry child legs); when only `notional` is given we leave it to the
    caller to pass `qty`, and fall back to a plain OTO with whatever legs exist.

    `time_in_force` defaults to **gtc**: child legs inherit the parent's TIF,
    and a `day` bracket's protective stop is cancelled at the close — exactly
    when a multi-day (swing) position needs it most, overnight.
    """
    if stop_loss_price is None and take_profit_price is None:
        raise BrokerError("submit_bracket_order requires a stop_loss_price or take_profit_price")
    body: dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": time_in_force,
    }
    # Bracket (both legs) vs OTO (one leg). Alpaca rejects fractional/notional
    # entries that carry child legs, so a bracket needs an integer qty.
    if take_profit_price is not None and stop_loss_price is not None:
        body["order_class"] = "bracket"
    else:
        body["order_class"] = "oto"
    if take_profit_price is not None:
        body["take_profit"] = {"limit_price": round(float(take_profit_price), 2)}
    if stop_loss_price is not None:
        body["stop_loss"] = {"stop_price": round(float(stop_loss_price), 2)}
    if qty is not None:
        body["qty"] = str(qty)
    elif notional is not None:
        body["notional"] = round(float(notional), 2)
    else:
        raise BrokerError("submit_bracket_order requires notional or qty")
    o = _request("POST", "/orders", json=body)
    return _norm_order(o)


def list_open_orders(symbol: str | None = None) -> list[dict[str, Any]]:
    """Open orders (optionally for one symbol), child legs flattened in."""
    path = "/orders?status=open&nested=true&limit=500"
    if symbol:
        path += f"&symbols={symbol.upper()}"
    rows = _request("GET", path) or []
    out: list[dict[str, Any]] = []
    for o in rows:
        out.append(_norm_order(o))
        for leg in o.get("legs") or []:
            if leg.get("status") in ("new", "accepted", "held", "pending_new", "partially_filled"):
                out.append(_norm_order(leg))
    return out


def cancel_order(order_id: str) -> None:
    _request("DELETE", f"/orders/{order_id}")


def cancel_open_orders(symbol: str) -> int:
    """Cancel every open order for `symbol` (incl. bracket legs). Returns count.

    Must run before closing a position that carries broker-side stop/target
    legs: otherwise the close can be rejected (shares are reserved by the
    legs), or the orphaned GTC stop can fire later and open an unintended
    short.
    """
    n = 0
    for o in list_open_orders(symbol):
        if o.get("id"):
            try:
                cancel_order(o["id"])
                n += 1
            except BrokerError as e:
                # A leg is cancelled together with its parent; 404/422 on the
                # second cancel is expected.
                log.info("alpaca_cancel_skip", order=o["id"], err=str(e)[:120])
    return n


def replace_stop(order_id: str, stop_price: float) -> dict[str, Any]:
    """Move a resting stop order (e.g. a bracket's stop leg) to `stop_price`."""
    o = _request("PATCH", f"/orders/{order_id}", json={"stop_price": round(float(stop_price), 2)})
    return _norm_order(o)


def close_position(symbol: str) -> dict[str, Any]:
    """Liquidate the entire position in `symbol` at market.

    Returns the normalized order; when the broker reports realized P/L on the
    closing order we surface it as `realized_pl` so the caller can book the
    actual gain/loss rather than an unrealized-P/L estimate.
    """
    o = _request("DELETE", f"/positions/{symbol.upper()}")
    if not o:
        return {"symbol": symbol.upper(), "status": "closed"}
    out = _norm_order(o)
    rp = o.get("realized_pl", o.get("realized_pnl"))
    if rp is not None:
        try:
            out["realized_pl"] = float(rp)
        except (TypeError, ValueError):
            pass
    return out


def _norm_order(o: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": o.get("id"),
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "qty": o.get("qty"),
        "notional": o.get("notional"),
        "type": o.get("type"),
        "status": o.get("status"),
        "submitted_at": o.get("submitted_at"),
        "filled_avg_price": o.get("filled_avg_price"),
        "order_class": o.get("order_class"),
        "time_in_force": o.get("time_in_force"),
        "stop_price": o.get("stop_price"),
        "limit_price": o.get("limit_price"),
    }
    legs = o.get("legs")
    if legs:
        out["legs"] = [{"id": g.get("id"), "type": g.get("type"), "side": g.get("side"),
                        "stop_price": g.get("stop_price"), "limit_price": g.get("limit_price"),
                        "status": g.get("status")} for g in legs]
    return out
