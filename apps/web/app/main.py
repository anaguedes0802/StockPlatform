"""StockPlatform web UI — server-rendered pages over the FastAPI backend.

No Node toolchain: each page is a Jinja2 template styled with Tailwind and
made interactive with Alpine.js, all loaded from a CDN. The browser calls the
API through the `/api/*` proxy below (same origin, so no CORS), exactly like
the BFF rewrite a Next.js app would use. Only the live-quotes WebSocket goes
to the API directly, at API_PUBLIC_URL.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from app.nav import FOOTER_ITEMS, NAV_GROUPS, PAGES

BASE_DIR = Path(__file__).resolve().parent.parent
API_INTERNAL_URL = os.environ.get("API_INTERNAL_URL", "http://localhost:8000")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "http://localhost:8000")

# Headers that describe one hop of the connection, not the payload.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Long read timeout: forecasts train on first call and backtests can take minutes.
    app.state.api = httpx.AsyncClient(
        base_url=API_INTERNAL_URL, timeout=httpx.Timeout(600.0, connect=10.0)
    )
    try:
        yield
    finally:
        await app.state.api.aclose()


app = FastAPI(title="StockPlatform Web", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def render(request: Request, template: str, title: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {
        "title": title,
        "nav_groups": NAV_GROUPS,
        "footer_items": FOOTER_ITEMS,
        "api_public_url": API_PUBLIC_URL,
        **ctx,
    })


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def api_proxy(path: str, request: Request) -> Response:
    """Forward the call to the API and stream the answer back (keeps chat SSE live)."""
    client: httpx.AsyncClient = request.app.state.api
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    if request.client:
        headers["x-forwarded-for"] = request.client.host
    upstream = client.build_request(
        request.method, f"/{path}", params=request.query_params,
        headers=headers, content=await request.body(),
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"detail": f"API unreachable at {API_INTERNAL_URL}: {e}"}, status_code=502)
    # aiter_bytes() hands back decoded bytes, so the upstream encoding header goes too.
    out_headers = {k: v for k, v in resp.headers.items()
                   if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"}
    return StreamingResponse(
        resp.aiter_bytes(), status_code=resp.status_code, headers=out_headers,
        background=BackgroundTask(resp.aclose),
    )


@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse("/dashboard")


@app.get("/stocks/{symbol}", response_class=HTMLResponse)
def stock_page(request: Request, symbol: str) -> HTMLResponse:
    symbol = symbol.upper()
    return render(request, "pages/stock.html", symbol, symbol=symbol)


@app.get("/{page}", response_class=HTMLResponse)
def page(request: Request, page: str) -> HTMLResponse:
    entry = PAGES.get(f"/{page}")
    if entry is None:
        raise HTTPException(404, "page not found")
    template, title = entry
    return render(request, template, title)
