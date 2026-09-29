"""Walk-forward backtest of the cross-sectional stock-selection composite.

Usage (from the repo root):
    python scripts/backtest_stock_selection.py                      # S&P 500 + 400
    python scripts/backtest_stock_selection.py --universe sp500 --top 0.1
    python scripts/backtest_stock_selection.py --gate --all-offsets     # the honest headline numbers
    python scripts/backtest_stock_selection.py --no-earnings        # momentum only
    python scripts/backtest_stock_selection.py --news               # + Alpaca news factors

Protocol (see app/ml/stock_selection.py and BACKTEST_RESULTS.md, "v6"):
  - Universe (--universe): "sp900" = current S&P 500 + S&P 400 (default),
    "sp500", or "bundled" = the app's own list in universe_data.py. Each date
    also requires >= 1y history, price > $5 and > $5M average daily value.
    Do NOT judge the strategy on "bundled": that list was curated with
    hindsight (MSTR, COIN, PLTR, SMCI, ... were added *because* they became
    famous), so momentum on it mostly rediscovers the curation.
  - Every `hold` sessions: rank at the close, buy the top `top` fraction at the
    next open (equal weight), exit at the open `hold` sessions later.
  - 10 bps per side on turnover. Benchmarks: equal-weight universe, SPY.
  - Reported separately for the DESIGN period (2017-2022, where choices were
    made) and the HOLDOUT (2023+, never used for any decision).

Known bias, not fixable with free data: the universe is *today's* list, so
stocks that were delisted or fell out of the index are missing (survivorship
bias). That inflates absolute CAGRs for every strategy including the
equal-weight benchmark; the excess over equal weight is the fairer number.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "apps/api")

from app.ml import stock_selection as ss  # noqa: E402
from app.services import earnings as earnings_svc  # noqa: E402
from app.services.universe_data import all_instruments  # noqa: E402

DESIGN_END = "2022-12-31"


_WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}


def constituents(universe: str) -> dict[str, str | None]:
    """symbol -> sector for the requested universe."""
    if universe == "bundled":
        return {i[0]: i[4] for i in all_instruments() if i[3] == "stock" and "." not in i[0]}
    import io

    import httpx

    out: dict[str, str | None] = {}
    for key in (["sp500"] if universe == "sp500" else ["sp500", "sp400"]):
        html = httpx.get(_WIKI[key], headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=30).text
        table = next(t for t in pd.read_html(io.StringIO(html)) if "Symbol" in t.columns)
        for sym, sec in zip(table["Symbol"].astype(str), table["GICS Sector"], strict=True):
            out.setdefault(sym.replace(".", "-"), sec)
    return out


def load(start: str, universe: str) -> tuple[dict[str, pd.DataFrame], dict[str, str | None]]:
    sectors = constituents(universe)
    symbols = list(sectors) + ["SPY", "SHY"]
    parts = []
    for i in range(0, len(symbols), 100):   # Yahoo drops very large batch requests
        parts.append(yf.download(symbols[i:i + 100], start=start, auto_adjust=True,
                                 progress=False, threads=False, group_by="column"))
    px = {}
    for f in ("Open", "Close", "Volume"):
        df = pd.concat([p[f] for p in parts if len(p)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        px[f.lower()] = df
    return px, sectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["sp900", "sp500", "bundled"], default="sp900")
    ap.add_argument("--start", default="2005-01-01", help="price history start (factors need ~1y warm-up)")
    ap.add_argument("--test-start", default="2006-06-01")
    ap.add_argument("--gate", action="store_true",
                    help="trend gate: stocks while SPY > 10-month SMA, else SHY")
    ap.add_argument("--all-offsets", action="store_true",
                    help="re-run for every rebalance-day offset (0..hold-1) and report median/range; "
                         "a single calendar alignment can swing multi-year CAGR by several points")
    ap.add_argument("--hold", type=int, default=ss.HOLD_BARS)
    ap.add_argument("--top", type=float, default=ss.TOP_FRACTION)
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--news", action="store_true", help="add Alpaca news factors (needs the archive)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    px, sectors = load(args.start, args.universe)
    close = px["close"].drop(columns=["SPY", "SHY"])
    close = close.loc[:, close.notna().sum() > 300]
    stocks = list(close.columns)
    print(f"{len(stocks)} stocks, {close.index[0].date()} -> {close.index[-1].date()}")

    events = None
    weights = dict(ss.WEIGHTS)
    if args.no_earnings:
        weights = {k: v for k, v in weights.items() if not k.startswith("earn_")}
    else:
        events = {}
        for i, s in enumerate(stocks):
            events[s] = [ss.EarningsEvent(pd.Timestamp(e["ts"]), e["surprise_pct"])
                         for e in earnings_svc.earnings_history(s, limit=60)]
            if i % 25 == 0:
                print(f"  earnings {i}/{len(stocks)}", flush=True)

    factors = ss.compute_factors(close, px["close"]["SPY"], sectors, events)
    if args.news:
        from app.services import alpaca_news
        nf = {s: alpaca_news.daily_features(s, close.index) for s in stocks}
        for k, w in ss.NEWS_WEIGHTS.items():
            factors[k] = pd.DataFrame({s: nf[s][k] for s in stocks}, index=close.index)
            weights[k] = w
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    mask = ss.universe_mask(close, px["volume"][stocks])
    score, _ = ss.composite(factors, mask, weights)
    spy_open = px["open"]["SPY"]
    risk_on = ss.market_risk_on(px["close"]["SPY"]) if args.gate else None
    cash = px["open"]["SHY"] if args.gate else None

    def one(test_start: str) -> pd.DataFrame:
        r = ss.backtest(score, px["open"][stocks], mask, hold=args.hold, top=args.top,
                        start=test_start, risk_on=risk_on, cash_open=cash)
        # SPY over the identical next-open -> open holding windows
        r["benchmark"] = np.log(spy_open.shift(-(1 + args.hold)) / spy_open.shift(-1)).reindex(r.index)
        return r

    segments = {"2006-2016": ("2006", "2016-12-31"), "2017-2022 (design)": ("2017", DESIGN_END),
                "2023+ (holdout)": ("2023", "2100"), "full": ("1900", "2100")}
    if not args.all_offsets:
        r = one(args.test_start)
        res = {"weights": weights, "hold": args.hold, "top": args.top, "gate": args.gate,
               **{k: ss.summarize(r.loc[lo:hi], args.hold) for k, (lo, hi) in segments.items()}}
    else:
        days = close.index[close.index >= args.test_start]
        runs = [one(str(days[k].date())) for k in range(args.hold)]
        res = {"weights": weights, "hold": args.hold, "top": args.top, "gate": args.gate,
               "offsets": args.hold}
        books = ["long", "equal_weight", "benchmark"] + (["gated"] if args.gate else [])
        for seg, (lo, hi) in segments.items():
            res[seg] = {}
            for book in books:
                cagr = [ss.summarize(r.loc[lo:hi].assign(long=r.loc[lo:hi][book]), args.hold)["long"]
                        for r in runs]
                vals = pd.DataFrame(cagr)
                res[seg][book] = {"cagr_pct_median": round(vals["cagr_pct"].median(), 1),
                                  "cagr_pct_range": [vals["cagr_pct"].min(), vals["cagr_pct"].max()],
                                  "sharpe_median": round(vals["sharpe"].median(), 2),
                                  "max_drawdown_pct_median": round(vals["max_drawdown_pct"].median(), 1)}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
