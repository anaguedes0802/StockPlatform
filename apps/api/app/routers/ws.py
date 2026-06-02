"""WebSocket gateway.

Two backends:
  - Alpaca real-time stream (when ALPACA_API_KEY/SECRET are set) — pushes ticks
    on the actual trade, sub-second latency on the free IEX feed.
  - yfinance polling (default fallback) — 5s polling, only emits on price change.

The wire protocol exposed to the web client is identical in both modes:
    client → {"action":"subscribe","symbols":["AAPL","MSFT"]}
    client → {"action":"unsubscribe","symbols":["AAPL"]}
    server → {"type":"quote","symbol":"AAPL","price":...,"ts":"..."}
    server → {"type":"bar","symbol":"AAPL","interval":"1m","o":...,"h":...,"l":...,"c":...,"v":...,"ts":"..."}   (Alpaca only)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services import market_data as md
from app.services.realtime_alpaca import (
    AlpacaStream,
    _is_crypto_symbol,
    get_stream,
    stream_for_symbol,
)

router = APIRouter(tags=["ws"])

SYMBOL_RE = re.compile(r"^[A-Z0-9.\-=^]{1,20}$")


async def _yfinance_fan_out(ws: WebSocket, symbols: set[str], interval: float = 5.0) -> None:
    """Fallback: poll quotes for the given symbols and stream changes."""
    last: dict[str, float] = {}
    try:
        while True:
            for sym in list(symbols):
                try:
                    q = await asyncio.to_thread(md.get_quote, sym)
                except Exception:
                    continue
                price = float(q.get("price") or 0.0)
                if price <= 0:
                    continue
                if last.get(sym) == price:
                    continue
                last[sym] = price
                payload: dict[str, Any] = {
                    "type": "quote",
                    "symbol": sym,
                    "price": price,
                    "change": q.get("change"),
                    "change_pct": q.get("change_pct"),
                    "ts": q.get("ts"),
                    "source": "yfinance",
                }
                await ws.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(interval)
    except (asyncio.CancelledError, WebSocketDisconnect):
        return


async def _alpaca_fan_out(
    ws: WebSocket, q: asyncio.Queue, last_tick: dict[str, float]
) -> None:
    """Alpaca path: drain the per-consumer queue onto the WS.

    Records the arrival time of each symbol's tick so the extended-hours
    supplement can tell when the IEX stream has gone quiet (pre/post-market).
    """
    try:
        while True:
            payload = await q.get()
            sym = payload.get("symbol")
            if sym:
                last_tick[sym] = time.monotonic()
            await ws.send_text(json.dumps(payload, default=str))
    except (asyncio.CancelledError, WebSocketDisconnect):
        return


async def _extended_hours_supplement(
    ws: WebSocket,
    symbols: set[str],
    last_tick: dict[str, float],
    interval: float = 8.0,
    quiet_after: float = 30.0,
) -> None:
    """Fill the gap when the Alpaca IEX stream goes quiet (pre/post-market).

    Alpaca's free IEX feed sees almost no extended-hours volume, so the live
    chart freezes outside 09:30–16:00 ET even though Yahoo (SIP tape) keeps
    moving. For each equity symbol with no recent Alpaca tick, poll the
    Yahoo-backed quote and emit on price change. Symbols actively printing on
    Alpaca (regular hours) and crypto (streams 24/7) are skipped, so this never
    double-emits during the regular session.
    """
    last_price: dict[str, float] = {}
    try:
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for sym in list(symbols):
                if _is_crypto_symbol(sym):
                    continue
                if now - last_tick.get(sym, 0.0) < quiet_after:
                    continue
                try:
                    qd = await asyncio.to_thread(md.get_quote, sym)
                except Exception:
                    continue
                price = float(qd.get("price") or 0.0)
                if price <= 0 or last_price.get(sym) == price:
                    continue
                last_price[sym] = price
                await ws.send_text(json.dumps({
                    "type": "quote",
                    "symbol": sym,
                    "price": price,
                    "change": qd.get("change"),
                    "change_pct": qd.get("change_pct"),
                    "ts": qd.get("ts"),
                    "source": "yfinance-ext",
                }, default=str))
    except (asyncio.CancelledError, WebSocketDisconnect):
        return


@router.websocket("/ws/quotes")
async def quotes_ws(ws: WebSocket) -> None:
    await ws.accept()
    symbols: set[str] = set()
    task: asyncio.Task | None = None

    use_alpaca = AlpacaStream.is_configured()
    consumer_q: asyncio.Queue | None = None
    used_streams: set[AlpacaStream] = set()   # track every stream we touched (equity + crypto)
    supplement_task: asyncio.Task | None = None
    last_tick: dict[str, float] = {}          # symbol → monotonic time of last Alpaca tick

    if use_alpaca:
        consumer_q = asyncio.Queue(maxsize=2048)

    async def restart_yf() -> None:
        nonlocal task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        if symbols:
            task = asyncio.create_task(_yfinance_fan_out(ws, symbols))

    # Alpaca path uses a single long-lived drain task; subscriptions are managed
    # by the per-symbol stream (equity or crypto), all pushing to one queue.
    if use_alpaca and consumer_q is not None:
        task = asyncio.create_task(_alpaca_fan_out(ws, consumer_q, last_tick))
        supplement_task = asyncio.create_task(
            _extended_hours_supplement(ws, symbols, last_tick)
        )

    async def _alpaca_subscribe(sym: str) -> None:
        """Route a symbol to the right Alpaca stream (crypto trades 24/7;
        equity only during US market hours)."""
        st = stream_for_symbol(sym)
        await st.start()        # idempotent — starts the equity/crypto connection
        used_streams.add(st)
        await st.subscribe(sym, consumer_q)

    try:
        await ws.send_text(json.dumps({"type": "hello", "source": "alpaca" if use_alpaca else "yfinance"}))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "error": "invalid json"}))
                continue
            action = msg.get("action")
            syms = [s.upper() for s in (msg.get("symbols") or []) if isinstance(s, str) and SYMBOL_RE.match(s.upper())]

            if action == "subscribe":
                new = [s for s in syms[:50] if s not in symbols]
                symbols.update(syms[:50])
                if use_alpaca and consumer_q is not None:
                    for s in new:
                        try:
                            await _alpaca_subscribe(s)
                        except Exception:
                            # If a stream won't init, fall back to yfinance polling
                            await restart_yf()
                else:
                    await restart_yf()
                await ws.send_text(json.dumps({"type": "subscribed", "symbols": sorted(symbols)}))

            elif action == "unsubscribe":
                for s in syms:
                    symbols.discard(s)
                if use_alpaca and consumer_q is not None:
                    for s in syms:
                        with contextlib.suppress(Exception):
                            await stream_for_symbol(s).unsubscribe(s, consumer_q)
                else:
                    await restart_yf()
                await ws.send_text(json.dumps({"type": "unsubscribed", "symbols": sorted(symbols)}))

            elif action == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()
        if supplement_task and not supplement_task.done():
            supplement_task.cancel()
        if consumer_q is not None:
            for st in used_streams:
                with contextlib.suppress(Exception):
                    await st.unsubscribe_all(consumer_q)
