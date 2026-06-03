"""End-to-end live smoke test of the API. Run with the venv while uvicorn is up.

Hits every major surface (stock data, AI/analysis, charts data, screener,
discovery, bots, investing) and prints a PASS/FAIL matrix with a data sanity
note for each. Read-only except for creating + deleting one throwaway bot.
"""
from __future__ import annotations

import sys
import time

import httpx

BASE = "http://127.0.0.1:8765"
SYM = "AAPL"


def note(d):
    if isinstance(d, list):
        return f"{len(d)} items"
    if isinstance(d, dict):
        return "keys: " + ", ".join(list(d)[:7])
    return str(d)[:60]


def main() -> None:
    c = httpx.Client(timeout=160.0)
    email = f"e2e_{int(time.time())}@example.com"
    tok = ""
    try:
        r = c.post(f"{BASE}/auth/register", json={"email": email, "password": "Test1234!pw"})
        tok = r.json().get("access", "")
    except Exception as e:
        print("auth register failed:", e)
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    print(f"auth: {'ok' if tok else 'FAILED'}  ({email})\n")

    passed = failed = 0

    def check(method, path, label, *, json=None, auth=False, ok_codes=(200, 201)):
        nonlocal passed, failed
        try:
            r = c.request(method, f"{BASE}{path}", json=json, headers=hdr if auth else {})
            good = r.status_code in ok_codes
            try:
                n = note(r.json())
            except Exception:
                n = r.text[:50]
            print(f"  {'PASS' if good else 'FAIL'}  {label:<34} {r.status_code}  {n}")
            passed += good
            failed += (not good)
            return r
        except Exception as e:
            print(f"  ERR   {label:<34} -    {str(e)[:60]}")
            failed += 1
            return None

    print("===== STOCK DATA / ANALYSIS =====")
    check("GET", "/stocks/search?q=app", "search")
    check("GET", f"/stocks/{SYM}", "detail (quote/profile/stats)")
    check("GET", f"/stocks/{SYM}/history?interval=1d&range=1y", "history daily")
    check("GET", f"/stocks/{SYM}/history?interval=5m&range=1d", "history 5m (premarket)")
    check("GET", f"/stocks/{SYM}/indicators?names=sma_20,rsi_14&interval=1d&range=1y", "indicators")
    check("GET", f"/stocks/{SYM}/price-action?interval=1d&range=1y", "price-action (SMC)")
    check("GET", f"/stocks/{SYM}/news?limit=5", "news")
    check("GET", f"/stocks/{SYM}/analysts", "analysts")
    check("GET", f"/stocks/{SYM}/insider?limit=5", "insider")
    check("GET", f"/stocks/{SYM}/institutional", "institutional")
    check("GET", f"/stocks/{SYM}/politicians?limit=5", "politicians")
    check("GET", f"/stocks/{SYM}/options", "options flow")
    check("GET", f"/stocks/{SYM}/earnings", "earnings")

    print("===== AI / FORECAST / REGIME =====")
    check("GET", f"/ai/opinion/{SYM}?use_llm=false", "opinion (deterministic)")
    check("POST", "/ai/recommend", "recommend", json={"symbol": SYM})
    check("POST", "/ai/forecast", "forecast (ensemble)", json={"symbol": SYM, "horizons": ["5d"]})
    check("GET", "/regime", "market regime")

    print("===== SCREENER / DISCOVERY =====")
    check("GET", "/screener/strategies", "screener strategies")
    check("GET", "/screener/run/rising_star?limit=8", "screener: rising_star")
    check("GET", "/screener/cross-sectional/composite?top_n=8", "cross-sectional composite")
    check("GET", "/opportunities?limit=6", "opportunities (engine)")
    check("GET", "/discovery/opportunities?limit=6", "discovery: opportunities")
    check("GET", "/discovery/catalyst-news?limit=6", "discovery: catalyst news")

    print("===== BOTS / INVESTING =====")
    check("GET", "/bot/config", "bot config")
    check("POST", "/bot/backtest", "backtest (portfolio)", json={"range": "10y"})
    check("POST", "/bot/backtest", "backtest (single)", json={"symbol": SYM})
    check("GET", f"/bot/signal/{SYM}?llm=false", "today signal")
    check("GET", "/bot/account", "broker account")
    check("GET", "/bot/invest/timing", "invest timing", auth=True)
    check("GET", "/bot/invest/quality-picks?top_n=6", "quality picks")
    check("POST", "/bot/invest/backtest", "DCA backtest", json={"monthly_contribution": 500})
    check("POST", "/bot/invest/picks-backtest", "picks backtest", json={"start_date": "2022-10-12", "amount": 10000})

    print("===== AUTHED BOT CRUD =====")
    r = check("POST", "/bot/bots", "create bot", json={"name": "E2E", "bot_type": "trader"}, auth=True)
    bid = r.json().get("id") if r and r.status_code == 201 else None
    check("GET", "/bot/bots", "list bots", auth=True)
    if bid:
        check("POST", f"/bot/bots/{bid}/scan", "scan bot", auth=True)
        check("DELETE", f"/bot/bots/{bid}", "delete bot", auth=True, ok_codes=(204,))

    print("===== MISC =====")
    check("GET", "/track-record", "track record")
    check("GET", "/famous-investors", "famous investors")
    check("GET", "/politicians/recent?limit=10", "politicians recent")

    print(f"\n==== {passed} passed, {failed} failed ====")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
