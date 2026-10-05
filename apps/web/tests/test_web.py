"""Web UI: every page renders, the sidebar marks the right page, and /api is proxied."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nav import FOOTER_ITEMS, NAV_GROUPS, PAGES

client = TestClient(app)


@pytest.mark.parametrize("path", [*PAGES, "/stocks/AAPL"])
def test_every_page_renders(path):
    r = client.get(path)
    assert r.status_code == 200
    assert "<title>" in r.text
    assert "x-data=" in r.text  # every page is an Alpine component


def test_sidebar_links_point_to_real_pages():
    hrefs = [item.href for g in NAV_GROUPS for item in g.items] + [i.href for i in FOOTER_ITEMS]
    for href in hrefs:
        assert client.get(href).status_code == 200, href


def test_active_page_is_marked():
    html = client.get("/news").text
    news_link = html[html.index('href="/news"'):]
    assert 'aria-current="page"' in news_link[: news_link.index(">")]


def test_stock_route_uppercases_symbol():
    assert "MSFT · StockPlatform" in client.get("/stocks/msft").text


def test_root_redirects_to_dashboard():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/dashboard"


def test_unknown_page_is_404():
    assert client.get("/no-such-page").status_code == 404


def test_api_proxy_forwards_method_path_query_body_and_auth():
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.update(method=request.method, path=request.url.path, query=request.url.query.decode(),
                    body=request.content, auth=request.headers.get("authorization"))
        return httpx.Response(201, json={"ok": True}, headers={"x-session-id": "abc"})

    app.state.api = httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url="http://api")
    r = client.post("/api/ai/recommend?x=1", json={"symbol": "AAPL"},
                    headers={"Authorization": "Bearer t0k"})
    assert r.status_code == 201
    assert r.json() == {"ok": True}
    assert r.headers["x-session-id"] == "abc"
    assert seen == {"method": "POST", "path": "/ai/recommend", "query": "x=1",
                    "body": b'{"symbol":"AAPL"}', "auth": "Bearer t0k"}


def test_api_proxy_reports_unreachable_api():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app.state.api = httpx.AsyncClient(transport=httpx.MockTransport(down), base_url="http://api")
    r = client.get("/api/healthz")
    assert r.status_code == 502
    assert "API unreachable" in r.json()["detail"]
