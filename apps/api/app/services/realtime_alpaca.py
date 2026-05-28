"""Alpaca Markets real-time WebSocket adapter.

Alpaca exposes two stock feeds:
  - wss://stream.data.alpaca.markets/v2/iex   — IEX exchange only, free tier
  - wss://stream.data.alpaca.markets/v2/sip   — full consolidated SIP, paid

The free IEX feed is real-time (not delayed) but reflects only the IEX exchange's
prints — a smaller slice of total US equity volume. Fine for retail UX; not
suitable for execution algos that need true NBBO.

This module owns ONE shared WebSocket connection. FastAPI WS handlers register
consumer queues per-symbol; we forward incoming trade ticks to every queue
subscribed to that symbol. When the last consumer unsubscribes from a symbol we
also tell Alpaca to drop it.

Protocol (Alpaca v2 stream):
    server → [{"T":"success","msg":"connected"}]
    client → {"action":"auth","key":"...","secret":"..."}
    server → [{"T":"success","msg":"authenticated"}]
    client → {"action":"subscribe","trades":["AAPL"],"quotes":["AAPL"]}
    server → [{"T":"t","S":"AAPL","p":150.0,"s":100,"t":"2025-..."}]
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from app.config import settings
from app.core.logging import log


def _is_crypto_symbol(sym: str) -> bool:
    s = sym.upper()
    return s.endswith("-USD") and "/" not in s


def _to_alpaca_crypto(sym: str) -> str:
    # BTC-USD → BTC/USD (Alpaca crypto pair format)
    return sym.upper().replace("-", "/")


def _from_alpaca_crypto(pair: str) -> str:
    # BTC/USD → BTC-USD (yfinance-style, what the rest of the app uses)
    return pair.upper().replace("/", "-")


class AlpacaStream:
    """Singleton manager for a shared Alpaca WS connection.

    Two flavors selected by `kind`:
      - "equity" → wss://stream.data.alpaca.markets/v2/{feed}  (US stocks, IEX)
      - "crypto" → wss://stream.data.alpaca.markets/v1beta3/crypto/us (24/7)

    Multiple concurrent FastAPI /ws/quotes clients share one upstream connection
    per flavor. Each consumer gets an `asyncio.Queue` that receives ticks for
    the symbols it cares about. Crypto symbols are stored/keyed in the consumer
    map using the yfinance style (BTC-USD) but sent to Alpaca as pairs (BTC/USD).
    """

    _instances: "dict[str, AlpacaStream]" = {}

    def __init__(self, kind: str = "equity") -> None:
        self.kind = kind
        self._ws: Any = None  # websockets client connection
        self._connect_task: asyncio.Task | None = None
        self._consumers: dict[str, set[asyncio.Queue]] = {}  # symbol (app-style) → queues
        self._connected = asyncio.Event()
        self._lock = asyncio.Lock()
        self._stop = False
        if kind == "crypto":
            self._url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        else:
            self._url = f"wss://stream.data.alpaca.markets/v2/{settings.alpaca_feed}"

    @classmethod
    def instance(cls, kind: str = "equity") -> "AlpacaStream":
        if kind not in cls._instances:
            cls._instances[kind] = cls(kind)
        return cls._instances[kind]

    @staticmethod
    def is_configured() -> bool:
        return bool(settings.alpaca_api_key and settings.alpaca_api_secret)

    def _app_to_wire(self, sym: str) -> str:
        return _to_alpaca_crypto(sym) if self.kind == "crypto" else sym

    def _wire_to_app(self, sym: str) -> str:
        return _from_alpaca_crypto(sym) if self.kind == "crypto" else sym

    # ------------------------------------------------------------------ public

    async def start(self) -> None:
        """Idempotently ensure the upstream connection task is running."""
        if not self.is_configured():
            raise RuntimeError("alpaca not configured")
        async with self._lock:
            if self._connect_task is None or self._connect_task.done():
                self._stop = False
                self._connect_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop = True
        if self._connect_task:
            self._connect_task.cancel()
            with contextlib.suppress(Exception):
                await self._connect_task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None
        self._connected.clear()

    async def subscribe(self, symbol: str, q: asyncio.Queue) -> None:
        """Register a consumer queue to receive ticks for `symbol`."""
        symbol = symbol.upper()
        first = symbol not in self._consumers or not self._consumers[symbol]
        self._consumers.setdefault(symbol, set()).add(q)
        if first:
            await self._send_subscribe([symbol])

    async def unsubscribe(self, symbol: str, q: asyncio.Queue) -> None:
        symbol = symbol.upper()
        qs = self._consumers.get(symbol)
        if not qs:
            return
        qs.discard(q)
        if not qs:
            self._consumers.pop(symbol, None)
            await self._send_unsubscribe([symbol])

    async def unsubscribe_all(self, q: asyncio.Queue) -> None:
        for sym in list(self._consumers.keys()):
            await self.unsubscribe(sym, q)

    # ------------------------------------------------------------- internals

    async def _send_subscribe(self, symbols: list[str]) -> None:
        await self._connected.wait()
        if self._ws is None:
            return
        wire = [self._app_to_wire(s) for s in symbols]
        msg = {"action": "subscribe", "trades": wire, "bars": wire}
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps(msg))
            log.info("alpaca_subscribe", kind=self.kind, symbols=wire)

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        if self._ws is None or not self._connected.is_set():
            return
        wire = [self._app_to_wire(s) for s in symbols]
        msg = {"action": "unsubscribe", "trades": wire, "bars": wire}
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps(msg))

    async def _run(self) -> None:
        """Reconnect loop with exponential backoff."""
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(self._url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    # 1. wait for {"T":"success","msg":"connected"}
                    hello = await asyncio.wait_for(ws.recv(), timeout=10)
                    log.info("alpaca_ws_open", hello=hello[:200])
                    # 2. auth
                    await ws.send(json.dumps({
                        "action": "auth",
                        "key": settings.alpaca_api_key,
                        "secret": settings.alpaca_api_secret,
                    }))
                    auth_resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    if "authenticated" not in auth_resp:
                        log.error("alpaca_auth_failed", resp=auth_resp[:200])
                        # Don't tight-loop on bad creds.
                        await asyncio.sleep(30)
                        continue
                    log.info("alpaca_authenticated")
                    self._connected.set()
                    backoff = 1.0
                    # Re-subscribe to anything consumers want.
                    if self._consumers:
                        await self._send_subscribe(list(self._consumers.keys()))
                    # 3. stream loop
                    async for raw in ws:
                        await self._dispatch(raw)
            except (ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                log.warning("alpaca_ws_disconnect", err=str(e))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("alpaca_ws_error", err=str(e))
            finally:
                self._connected.clear()
                self._ws = None
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(messages, list):
            messages = [messages]
        for m in messages:
            t = m.get("T")
            wire_sym = m.get("S")
            if not wire_sym:
                continue
            sym = self._wire_to_app(wire_sym)   # BTC/USD → BTC-USD for crypto
            qs = self._consumers.get(sym)
            if not qs:
                continue
            if t == "t":  # trade
                payload = {
                    "type": "quote",
                    "symbol": sym,
                    "price": float(m.get("p") or 0),
                    "size": int(m.get("s") or 0),
                    "ts": m.get("t"),
                    "source": "alpaca",
                }
            elif t == "b":  # 1-min bar
                payload = {
                    "type": "bar",
                    "symbol": sym,
                    "interval": "1m",
                    "o": float(m.get("o") or 0),
                    "h": float(m.get("h") or 0),
                    "l": float(m.get("l") or 0),
                    "c": float(m.get("c") or 0),
                    "v": int(m.get("v") or 0),
                    "ts": m.get("t"),
                    "source": "alpaca",
                }
            else:
                continue  # ignore quotes / errors / subscription confirmations
            for q in list(qs):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(payload)


def get_stream(kind: str = "equity") -> AlpacaStream:
    return AlpacaStream.instance(kind)


def stream_for_symbol(symbol: str) -> AlpacaStream:
    """Pick the right stream (crypto vs equity) for a given symbol."""
    return AlpacaStream.instance("crypto" if _is_crypto_symbol(symbol) else "equity")
