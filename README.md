# StockPlatform

An AI-native stock analysis, forecasting, and portfolio platform.
Hybrid forecasting (technical + fundamental + sentiment + macro), explainable recommendations, backtesting, and a professional-grade UI.

> **Status:** v0 — runnable foundation.
> **For the full design** (15 deliverables, schemas, scaling, deployment, monetization, future roadmap) see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Quick start

```bash
docker compose up -d
```

Then open:
- **Web app**: http://localhost:3000
- **API docs (Swagger)**: http://localhost:8000/docs

The first time you click on a symbol's AI panel, the XGBoost model trains on-the-fly (~5–15s for the first horizon) and is cached to disk for subsequent calls.

### Without Docker

```bash
# Backend
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -e .
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd apps/web
npm install
npm run dev
```

You need a running Postgres + Redis. The easiest way is `docker compose up -d postgres redis`.

---

## What's in this repo

```
StockPlatform/
├── ARCHITECTURE.md      ← Full design doc (read this)
├── docker-compose.yml
├── apps/
│   ├── web/             ← Next.js 15 + Tailwind + lightweight-charts
│   └── api/             ← FastAPI + SQLAlchemy + Alembic + XGBoost
└── infra/, scripts/     ← Reserved for k8s, terraform, seed scripts
```

### What works end-to-end today

- **Stock universe** of 200+ symbols across US large-caps, ETFs, indexes, crypto, forex, and major EU names. Search with asset-class + sector filters.
- **Quotes, profiles, OHLCV history** via yfinance + Redis caching.
- **Interactive candlestick charts** (TradingView's Lightweight Charts v5) with timeframe switching and indicator overlays (SMA 20/50/200, EMA 12/26).
- **Technical indicators API**: SMA, EMA, RSI, MACD, Bollinger, ATR, VWAP, Stochastic, Ichimoku.
- **AI forecast** at 1d/5d/30d horizons: regime-aware ensemble of **XGBoost** (quantile heads + SHAP explainability) and **LSTM** (PyTorch, pinball loss + gradient×input attribution). Returns p10/p25/p50/p75/p90 bands, direction probability, contributions split into technical/fundamental/sentiment/macro/events buckets, and top drivers.
- **Recommendation engine**: regime-aware blend of forecast, technical, fundamental, and news-sentiment signals → STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL with structured reasoning.
- **Regime detection**: rule-based bull/bear × low/high vol + sideways classifier; weights change per regime.
- **News + sentiment**: yfinance news + Yahoo Finance RSS aggregation, lexicon-based sentiment (finBERT-ready), impact scoring, intelligence feed page with trending-sentiment panel.
- **Live quotes via WebSocket** with subscribe/unsubscribe protocol; dashboard watchlist updates in real time.
- **AI chat** with tool-calling against the platform's own APIs (get_quote, get_forecast, get_recommendation, get_news, compare_symbols, run_quick_backtest, …). Anthropic SDK; stub fallback when no API key.
- **Portfolio CRUD** (transactions → positions, avg cost, sector allocation, unrealized P&L). Sector allocation chart.
- **Swing trading** (`/swing` page, `/swing/*` API): breakout, pullback and RSI(2) setups; a daily scanner that sizes each setup by % of equity at risk with whole shares; a **portfolio-level backtester** (shared capital, gap-aware stops, costs, SEC-dated earnings blackout, benchmark, split-half stability, Monte Carlo, explicit verdict); swing bots that trade through Alpaca with GTC broker-side stops and trailing-stop ratchets; optional daily auto-run (`BOT_AUTORUN=1`). Honest results, including why FX was rejected: `BACKTEST_RESULTS.md` § Swing trading.
- **ML signal filter** (meta-labeling, `app/ml/signal_filter.py`): walk-forward tested against random and perfect-foresight filters; available per bot and as an AI score in the swing scanner, **off by default** because it did not help the ETF breakout (results in `BACKTEST_RESULTS.md` § ML signal filter).
- **AI risk-gate forward test**: every LLM gate verdict is logged and later scored against the trade it approved, shrank or vetoed; see the *AI gate track record* panel on the Trading Bot page and `GET /bot/gate/report`.
- **Simulated paper broker** (`broker_sim`): a private $100k ledger per bot, running the same production bot code, so bots can be A/B-tested without sharing an Alpaca account. A forward test of RSI(2) with vs without the ML filter is running (see `BACKTEST_RESULTS.md` § Forward paper test).
- **Insider transactions, full history** (`insider_bulk`): every Form 4 purchase and sale since 2006 from the SEC's quarterly data sets, keyed to filing date. A placebo-controlled test of insider cluster buys found no edge in liquid, still-listed stocks (`BACKTEST_RESULTS.md` § Insider cluster buys).
- **Strategy Lab UI** with parameterized presets, equity curve vs Buy & Hold, trade list, full metrics (CAGR / Sharpe / Sortino / MDD).
- **JWT auth** (register / login / refresh / logout) with bcrypt, **refresh-token rotation** (hashed, single-use, revocable in DB), and **TOTP 2FA** (setup + verify + disable) with QR enrollment in Settings.
- **Rate limiting** via Redis token buckets.
- **Dark, dense UI** with Bloomberg-like density on the stock page and clean dashboard.

### Routes shipped

42 HTTP routes + WebSocket: `/auth/*` (including 2FA), `/stocks/*`, `/stocks/_universe/{sectors,asset-classes}`, `/ai/forecast`, `/ai/recommend`, `/ai/chat`, `/news/*`, `/portfolios/*`, `/watchlists/*`, `/strategies/*`, `/backtest/run`, `/ws/quotes`.

13 web pages: dashboard, stock detail, screener, news, portfolio, strategy lab, AI chat, login, register, settings.

### What's still stubbed (and where to extend)

| Stub | Where | What to add |
|---|---|---|
| Transformer / TFT / Prophet / ARIMA forecasters | `apps/api/app/ml/ensemble.py` | Implement `fit`/`predict` (Nixtla `neuralforecast` does most of the heavy lifting) |
| finBERT sentiment | `apps/api/app/services/sentiment.py` | Replace `score_text` with HF `ProsusAI/finbert` pipeline |
| Social pipelines | (new service) | Add Reddit/StockTwits/X ingestion → `social_posts` |
| OAuth (Google/GitHub) | `apps/api/app/routers/auth.py` | Add Auth.js on web; exchange ID tokens for app JWTs |
| Monte Carlo backtest overlay | `apps/api/app/backtest/engine.py` | Bootstrap resample trade returns; surface percentile bands |
| TimescaleDB hypertable conversion | `apps/api/alembic/versions/` | `SELECT create_hypertable('price_bars', 'ts')` migration |
| Audit logging hooks | `apps/api/app/core/` | Middleware to write `audit_log` on auth + portfolio mutations |
| Live ticks from provider WS | `apps/api/app/routers/ws.py` | Replace yfinance polling with a Polygon/Finnhub WS fan-in |

---

## Quick API tour

```bash
# Search
curl 'http://localhost:8000/stocks/search?q=appl'

# Quote + profile + key stats
curl http://localhost:8000/stocks/AAPL

# Daily history (default 1y)
curl 'http://localhost:8000/stocks/AAPL/history?interval=1d&range=6mo'

# Indicators
curl 'http://localhost:8000/stocks/AAPL/indicators?names=sma_20,rsi_14,macd&range=1y'

# Forecast (trains XGBoost on first call, then caches)
curl -X POST http://localhost:8000/ai/forecast \
     -H 'Content-Type: application/json' \
     -d '{"symbol":"AAPL","horizons":["1d","5d","30d"]}'

# Recommendation
curl -X POST http://localhost:8000/ai/recommend \
     -H 'Content-Type: application/json' \
     -d '{"symbol":"AAPL"}'
```

---

## Tech stack (summary)

- **Frontend**: Next.js 15 (App Router, RSC), TypeScript, Tailwind, TradingView Lightweight Charts, Recharts, TanStack Query, Zustand
- **Backend**: Python 3.11, FastAPI, SQLAlchemy 2, Alembic, Pydantic v2
- **Data**: PostgreSQL (TimescaleDB image), Redis
- **ML**: scikit-learn, XGBoost, SHAP. Slot-in support planned for Nixtla `neuralforecast` (TFT/NHiTS/LSTM)
- **Backtest**: pandas-native simulator (slot-in support planned for vectorbt)
- **Charts**: `lightweight-charts` (MIT) — by TradingView

Full list in `ARCHITECTURE.md §15`.

---

## License & disclaimer

Code: choose your own license before publishing. We recommend MIT for the frontend and Apache-2.0 for the backend (compatible with the ML libs).

**This platform is informational, not investment advice.** It does not execute trades, custody assets, or operate as a registered advisor. Forecasts and recommendations are model outputs — frequently wrong, always probabilistic. Past performance does not guarantee future results.
