"""Forward test of the stock-selection strategy with fake money.

Why: backtests can mislead (survivorship bias, calendar luck, subtle look-ahead).
The only unfakeable test is the future. This runs the exact live ranking from
app/ml/stock_selection.py (via cross_sectional._rank_momentum_pead) once a month
and holds the result in Alpaca's paper account — real market prices, fake money:

  - universe: current S&P 500 + S&P 400 (the backtest universe, not the app's
    hand-picked list, which was shown to be hindsight-biased)
  - if SPY is above its 10-month SMA: equal-weight the top 10% of the ranking
  - otherwise: hold SHY (short-term Treasuries) until the next rebalance
  - rebalance when >= 28 calendar days have passed since the last one

Two execution modes (settings.paper_strategy_broker):
  - "sim" (default): our own paper broker — market orders fill immediately at
    the live quote plus SLIPPAGE_BPS, minus COMMISSION_BPS, booked straight
    into the ledger. No external account needed.
  - "alpaca": orders go to the Alpaca PAPER account, tagged with a
    client_order_id prefix; fills are synced from the broker. Refuses to run
    against a non-paper (real-money) endpoint.
Either way the strategy keeps its OWN ledger (SQLite) and only ever sells what
it bought, so it can share an account with the Trading Bot.

The order logic lives in `plan_orders` so the historical replay
(paper_replay.py) runs exactly the same rules.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.logging import log
from app.services import broker_alpaca as broker

DB_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "paper_strategy" / "ledger.sqlite"
TAG = "ssel"                    # client_order_id prefix
REBALANCE_EVERY_DAYS = 28       # ~21 trading sessions, the backtest's hold period
CASH_BUFFER = 0.01              # keep 1% of strategy value in cash (fees / rounding)
DRIFT_BAND = 0.25               # only trim/top-up names more than 25% off target
MIN_ORDER = 5.0                 # dollars
DEFENSIVE = "SHY"
SLIPPAGE_BPS = 5.0              # sim + replay fill model: half-spread / impact per side
COMMISSION_BPS = 1.0
DEFAULT_SIM_CAPITAL = 10_000.0
_local = threading.local()


class NotPaperError(RuntimeError):
    pass


# --------------------------------------------------------------------- ledger

def _db() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY, client_order_id TEXT, rebalance_id TEXT, symbol TEXT,
                side TEXT, notional REAL, qty REAL, status TEXT, filled_qty REAL DEFAULT 0,
                filled_avg_price REAL, booked INTEGER DEFAULT 0, submitted_at TEXT);
            CREATE TABLE IF NOT EXISTS holdings (symbol TEXT PRIMARY KEY, qty REAL, cost REAL);
            CREATE TABLE IF NOT EXISTS rebalances (
                id TEXT PRIMARY KEY, at TEXT, risk_on INTEGER, as_of TEXT, n_ranked INTEGER,
                targets TEXT, value_before REAL, note TEXT);
            CREATE TABLE IF NOT EXISTS snapshots (day TEXT PRIMARY KEY, value REAL, spy REAL);
        """)
        _local.conn = conn
    return conn


def _meta(key: str, default: Any = None) -> Any:
    r = _db().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def _set_meta(key: str, value: Any) -> None:
    with _db():
        _db().execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))


def _mode() -> str:
    from app.config import settings
    return "alpaca" if (settings.paper_strategy_broker or "sim").lower() == "alpaca" else "sim"


def _require_paper() -> None:
    started = _meta("broker")
    if started and started != _mode():
        raise NotPaperError(f"This forward test was started with broker '{started}'; "
                            f"paper_strategy_broker is now '{_mode()}'. Switch it back to keep one ledger.")
    if _mode() == "sim":
        return
    if not broker.is_configured():
        raise NotPaperError("Alpaca trading credentials are not configured")
    if not broker.is_paper():
        raise NotPaperError("Refusing to run: alpaca_trading_base_url is not the PAPER endpoint")


def _market_open() -> bool:
    if broker.is_configured():
        try:
            return broker.is_market_open()   # read-only clock call
        except broker.BrokerError:
            pass
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


# ------------------------------------------------------------------ fills

def sync_fills() -> int:
    """Pull fill status for our non-final orders and book filled quantities
    into holdings / cash. Idempotent: each order is booked once, when final.
    (Sim orders are booked at submission, so there is nothing to sync.)"""
    conn = _db()
    pending = conn.execute("SELECT * FROM orders WHERE booked=0").fetchall()
    booked = 0
    for o in pending:
        try:
            b = broker.get_order(o["id"])
        except broker.BrokerError as e:
            log.warning("paper_strategy.order_lookup_failed", id=o["id"], err=str(e)[:120])
            continue
        status = b.get("status") or ""
        fq = float(b.get("filled_qty") or 0)
        px = float(b.get("filled_avg_price") or 0)
        final = status in ("filled", "canceled", "expired", "rejected")   # partial fills book what filled
        with conn:
            conn.execute("UPDATE orders SET status=?, filled_qty=?, filled_avg_price=? WHERE id=?",
                         (status, fq, px or None, o["id"]))
            if final:
                if fq > 0 and px > 0:
                    cash = float(_meta("cash", 0.0))
                    h = conn.execute("SELECT * FROM holdings WHERE symbol=?", (o["symbol"],)).fetchone()
                    qty, cost = (h["qty"], h["cost"]) if h else (0.0, 0.0)
                    if o["side"] == "buy":
                        qty, cost, cash = qty + fq, cost + fq * px, cash - fq * px
                    else:
                        avg = cost / qty if qty else 0.0
                        qty, cost, cash = max(0.0, qty - fq), max(0.0, cost - avg * fq), cash + fq * px
                    if qty > 1e-9:
                        conn.execute("INSERT OR REPLACE INTO holdings VALUES (?, ?, ?)", (o["symbol"], qty, cost))
                    else:
                        conn.execute("DELETE FROM holdings WHERE symbol=?", (o["symbol"],))
                    conn.execute("INSERT OR REPLACE INTO meta VALUES ('cash', ?)", (json.dumps(cash),))
                conn.execute("UPDATE orders SET booked=1 WHERE id=?", (o["id"],))
                booked += 1
    return booked


# ----------------------------------------------------------------- valuation

def _prices(symbols: list[str]) -> dict[str, float]:
    from app.services import market_data as md
    out = {}
    for s in symbols:
        try:
            q = md.get_quote(s)
            p = float(q.get("price") or 0)
            if p > 0:
                out[s] = p
        except Exception:
            continue
    return out


def holdings() -> dict[str, dict[str, float]]:
    return {r["symbol"]: {"qty": r["qty"], "cost": r["cost"]}
            for r in _db().execute("SELECT * FROM holdings").fetchall()}


def valuation() -> dict[str, Any]:
    h = holdings()
    px = _prices(list(h))
    positions = []
    mv = 0.0
    for s, x in sorted(h.items()):
        p = px.get(s)
        v = x["qty"] * p if p else x["cost"]
        mv += v
        positions.append({"symbol": s, "qty": round(x["qty"], 6), "price": p, "market_value": round(v, 2),
                          "cost": round(x["cost"], 2),
                          "pnl_pct": round((v / x["cost"] - 1) * 100, 2) if x["cost"] else None})
    cash = float(_meta("cash", 0.0))
    return {"cash": round(cash, 2), "positions_value": round(mv, 2), "value": round(cash + mv, 2),
            "positions": positions}


# ----------------------------------------------------------------- universe

def universe_sectors() -> dict[str, str | None]:
    """symbol -> GICS sector for the current S&P 500 + 400 (cached 7 days)."""
    _universe()
    return (_meta("universe") or {}).get("sectors", {})


def _universe() -> list[str]:
    cached = _meta("universe")
    if cached and (time.time() - cached["at"]) < 7 * 86400 and cached.get("sectors"):
        return cached["symbols"]
    symbols: list[str] = []
    sectors: dict[str, str | None] = {}
    try:
        import io

        import httpx
        import pandas as pd
        for url in ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"):
            html = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 (StockPlatform research)"}, timeout=30).text
            table = next(t for t in pd.read_html(io.StringIO(html)) if "Symbol" in t.columns)
            for sym, sec in zip(table["Symbol"].astype(str), table.get("GICS Sector", [None] * len(table)),
                                strict=False):
                symbols.append(sym.replace(".", "-"))
                sectors.setdefault(sym.replace(".", "-"), sec)
    except Exception as e:
        log.warning("paper_strategy.universe_fetch_failed", err=str(e)[:160])
    if len(symbols) < 400:
        if cached:
            return cached["symbols"]
        from app.services import market_data as md
        return [it["symbol"] for it in md.all_universe() if it.get("asset_class") == "stock"]
    symbols = sorted(set(symbols))
    _set_meta("universe", {"at": time.time(), "symbols": symbols, "sectors": sectors})
    return symbols


def _prefetch_earnings(symbols: list[str], pause_s: float = 0.6) -> None:
    """Fill the earnings disk cache slowly and sequentially: half the ranking is
    the earnings-surprise factor, and Yahoo blocks bursts. Cached entries (3 days)
    are skipped, so after the first run this costs almost nothing."""
    from app.services import earnings as earnings_svc
    for s in symbols:
        key = f"earn:{s}:8"
        if earnings_svc._disk_get(key, max_age_s=3 * 86400) is not None:
            continue
        earnings_svc.earnings_dates(s, limit=8)
        time.sleep(pause_s)


# ---------------------------------------------------------------- planning

def plan_orders(targets: list[str], positions: dict[str, dict[str, float]], qty: dict[str, float],
                cash: float, value: float) -> tuple[list[dict], list[dict], float]:
    """The rebalance rule, shared by the live forward test and the historical
    replay. `positions` maps symbol -> {"market_value", "price"}; `qty` is
    shares held. Returns (sells, buys, per_name_target).

    Exits sell everything; holdings more than DRIFT_BAND above target are
    trimmed; new names are bought at target; holdings more than DRIFT_BAND
    below target are topped up. Buys never exceed cash + sale proceeds.
    """
    per = value * (1 - CASH_BUFFER) / max(1, len(targets))
    tset = set(targets)
    sells, buys = [], []
    for s, p in positions.items():
        if s not in tset:
            sells.append({"symbol": s, "qty": qty.get(s, 0.0), "reason": "left the portfolio",
                          "est_value": p["market_value"]})
        elif p["market_value"] > per * (1 + DRIFT_BAND) and p.get("price"):
            excess = p["market_value"] - per
            sells.append({"symbol": s, "qty": round(excess / p["price"], 6), "reason": "trim to target",
                          "est_value": round(excess, 2)})
    for s in targets:
        have = positions.get(s, {}).get("market_value", 0.0)
        if s not in positions:
            buys.append({"symbol": s, "notional": round(per, 2), "reason": "new entrant"})
        elif have < per * (1 - DRIFT_BAND):
            buys.append({"symbol": s, "notional": round(per - have, 2), "reason": "top up to target"})
    buys = [b for b in buys if b["notional"] >= MIN_ORDER]
    sells = [x for x in sells if x["est_value"] >= MIN_ORDER or x["reason"] == "left the portfolio"]
    budget = cash + sum(x["est_value"] for x in sells) - value * CASH_BUFFER
    total = sum(b["notional"] for b in buys)
    if total > budget > 0:
        scale = budget / total
        for b in buys:
            b["notional"] = round(b["notional"] * scale, 2)
    return sells, buys, per


def due() -> bool:
    last = _db().execute("SELECT at FROM rebalances ORDER BY at DESC LIMIT 1").fetchone()
    if not last:
        return True
    return datetime.now(timezone.utc) - datetime.fromisoformat(last["at"]) >= timedelta(days=REBALANCE_EVERY_DAYS)


def plan(initial_capital: float | None = None) -> dict[str, Any]:
    """Compute the target book and the orders needed to reach it. No side effects."""
    from app.services.cross_sectional import _rank_momentum_pead

    _require_paper()
    universe = _universe()
    _prefetch_earnings(universe)
    ranking = _rank_momentum_pead(universe, top_n=len(universe), bottom_n=0)
    regime = ranking.get("market_regime") or {}
    risk_on = bool(regime.get("risk_on", True))
    picks = [r["symbol"] for r in ranking.get("longs", []) if r.get("in_portfolio")]
    targets_syms = picks if risk_on else [DEFENSIVE]

    val = valuation()
    started = _meta("started_at") is not None
    if not started:
        if initial_capital:
            cap = initial_capital
        elif _mode() == "alpaca":
            cap = float(broker.get_account().get("equity") or 0) * 0.98
        else:
            cap = DEFAULT_SIM_CAPITAL
        value = cash = cap
    else:
        value, cash = val["value"], val["cash"]
    current = {p["symbol"]: p for p in val["positions"]}
    held = {s: h["qty"] for s, h in holdings().items()}
    sells, buys, per = plan_orders(targets_syms, current, held, cash, value)

    return {
        "as_of": ranking.get("as_of"), "risk_on": risk_on, "regime": regime,
        "n_ranked": ranking.get("n_scored"), "universe_size": len(universe),
        "earnings_coverage": ranking.get("earnings_coverage"),
        "targets": targets_syms, "target_notional_each": round(per, 2),
        "strategy_value": round(value, 2), "cash": round(cash, 2), "broker": _mode(),
        "sells": sells, "buys": buys, "first_run": not started,
        "top_ranked": [{"symbol": r["symbol"], "percentile": r.get("percentile"),
                        "components": r.get("components")} for r in ranking.get("longs", [])[:len(targets_syms)]],
    }


# --------------------------------------------------------------- execution

def rebalance(execute: bool = False, force: bool = False, allow_closed: bool = False,
              initial_capital: float | None = None) -> dict[str, Any]:
    """Plan and (optionally) submit this month's orders. Sells go first."""
    _require_paper()
    sync_fills()
    if not force and not due():
        return {"skipped": True, "reason": f"last rebalance < {REBALANCE_EVERY_DAYS} days ago"}
    p = plan(initial_capital)
    if not execute:
        return {"executed": False, **p}
    if not allow_closed and not _market_open():
        return {"executed": False, "reason": "market closed — rerun during market hours", **p}

    if p["first_run"]:
        _set_meta("broker", _mode())
        _set_meta("started_at", datetime.now(timezone.utc).isoformat())
        _set_meta("initial_capital", p["strategy_value"])
        _set_meta("cash", p["cash"])
    rid = uuid.uuid4().hex[:10]
    submitted, errors = [], []

    def send(symbol: str, side: str, **kw) -> None:
        coid = f"{TAG}-{rid}-{side[0]}-{symbol}-{uuid.uuid4().hex[:6]}"
        if _mode() == "sim":
            try:
                submitted.append(_sim_fill(rid, coid, symbol, side, **kw))
            except Exception as e:
                errors.append({"symbol": symbol, "side": side, "error": str(e)[:200]})
            return
        try:
            o = broker.submit_market_order(symbol, side=side, client_order_id=coid, **kw)
        except broker.BrokerError as e:
            # non-fractionable names reject notional orders: retry with whole shares
            if side == "buy" and "notional" in kw and ("fraction" in str(e).lower() or "not fractionable" in str(e).lower()):
                px = _prices([symbol]).get(symbol)
                shares = math.floor(kw["notional"] / px) if px else 0
                if shares >= 1:
                    return send(symbol, side, qty=shares)
            errors.append({"symbol": symbol, "side": side, "error": str(e)[:200]})
            return
        with _db():
            _db().execute("INSERT INTO orders (id, client_order_id, rebalance_id, symbol, side, notional, qty, "
                          "status, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (o["id"], coid, rid, symbol, side, kw.get("notional"), kw.get("qty"),
                           o.get("status"), o.get("submitted_at")))
        submitted.append({"symbol": symbol, "side": side, **kw, "status": o.get("status")})

    for x in p["sells"]:
        send(x["symbol"], "sell", qty=round(x["qty"], 6))
    for b in p["buys"]:
        send(b["symbol"], "buy", notional=b["notional"])

    with _db():
        _db().execute("INSERT INTO rebalances VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (rid, datetime.now(timezone.utc).isoformat(), int(p["risk_on"]), p["as_of"], p["n_ranked"],
                       json.dumps(p["targets"]), p["strategy_value"],
                       json.dumps({"errors": errors, "n_orders": len(submitted)})))
    time.sleep(3)
    sync_fills()
    return {"executed": True, "rebalance_id": rid, "orders": submitted, "errors": errors, **p}


def _sim_fill(rid: str, coid: str, symbol: str, side: str, notional: float | None = None,
              qty: float | None = None) -> dict[str, Any]:
    """Our own paper broker: fill a market order now at the live quote with
    modeled slippage and commission, and book it straight into the ledger."""
    px = _prices([symbol]).get(symbol)
    if not px:
        raise RuntimeError(f"no live price for {symbol}")
    slip, comm = SLIPPAGE_BPS / 1e4, COMMISSION_BPS / 1e4
    conn = _db()
    h = conn.execute("SELECT * FROM holdings WHERE symbol=?", (symbol,)).fetchone()
    have_q, have_c = (h["qty"], h["cost"]) if h else (0.0, 0.0)
    cash = float(_meta("cash", 0.0))
    if side == "buy":
        fill = px * (1 + slip)
        q = (notional / (1 + comm)) / fill if notional else float(qty or 0)
        spend = q * fill * (1 + comm)
        have_q, have_c, cash = have_q + q, have_c + spend, cash - spend
    else:
        fill = px * (1 - slip)
        q = min(float(qty or 0), have_q)
        proceeds = q * fill * (1 - comm)
        avg = have_c / have_q if have_q else 0.0
        have_q, have_c, cash = have_q - q, max(0.0, have_c - avg * q), cash + proceeds
    oid = f"sim-{uuid.uuid4().hex[:12]}"
    with conn:
        if have_q > 1e-9:
            conn.execute("INSERT OR REPLACE INTO holdings VALUES (?, ?, ?)", (symbol, have_q, have_c))
        else:
            conn.execute("DELETE FROM holdings WHERE symbol=?", (symbol,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('cash', ?)", (json.dumps(cash),))
        conn.execute("INSERT INTO orders (id, client_order_id, rebalance_id, symbol, side, notional, qty, status, "
                     "filled_qty, filled_avg_price, booked, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, 1, ?)",
                     (oid, coid, rid, symbol, side, notional, qty, q, fill, datetime.now(timezone.utc).isoformat()))
    return {"symbol": symbol, "side": side, "qty": round(q, 6), "fill_price": round(fill, 4), "status": "filled"}


# ------------------------------------------------------------- performance

def snapshot() -> dict[str, Any] | None:
    """Record today's strategy value and SPY close (once per day)."""
    if _meta("started_at") is None:
        return None
    sync_fills()
    v = valuation()["value"]
    spy = _prices(["SPY"]).get("SPY")
    with _db():
        _db().execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)", (date.today().isoformat(), v, spy))
    return {"value": v, "spy": spy}


def status() -> dict[str, Any]:
    started = _meta("started_at")
    mode = _meta("broker") or _mode()
    out: dict[str, Any] = {
        "broker": mode,
        "is_paper": True if mode == "sim" else broker.is_paper(),
        "configured": True if mode == "sim" else broker.is_configured(),
        "started_at": started, "initial_capital": _meta("initial_capital"),
        "rebalance_every_days": REBALANCE_EVERY_DAYS, "due": due(),
    }
    if not started:
        return {**out, "running": False}
    try:
        sync_fills()
    except Exception:
        pass
    val = valuation()
    snaps = [dict(r) for r in _db().execute("SELECT * FROM snapshots ORDER BY day").fetchall()]
    init = float(out["initial_capital"] or 0) or 1.0
    spy0 = next((s["spy"] for s in snaps if s["spy"]), None)
    curve = [{"day": s["day"], "strategy_pct": round((s["value"] / init - 1) * 100, 2),
              "spy_pct": round((s["spy"] / spy0 - 1) * 100, 2) if spy0 and s["spy"] else None} for s in snaps]
    rebs = [dict(r) | {"targets": json.loads(r["targets"]), "note": json.loads(r["note"] or "{}")}
            for r in _db().execute("SELECT * FROM rebalances ORDER BY at DESC").fetchall()]
    last = rebs[0]["at"] if rebs else None
    return {
        **out, "running": True, **val,
        "return_pct": round((val["value"] / init - 1) * 100, 2),
        "spy_return_pct": curve[-1]["spy_pct"] if curve else None,
        "curve": curve, "rebalances": rebs[:24],
        "next_rebalance_after": (datetime.fromisoformat(last) + timedelta(days=REBALANCE_EVERY_DAYS)).isoformat()
        if last else None,
        "open_orders": [dict(o) for o in _db().execute("SELECT * FROM orders WHERE booked=0").fetchall()],
    }
