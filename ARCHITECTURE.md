# StockPlatform — Architecture & Design

> An AI-native stock analysis, forecasting, and portfolio platform.
> Inspired by TradingView × Bloomberg × QuantConnect × Robinhood, with a hybrid AI forecasting and recommendation core.

This document is the source of truth for system design. It covers all 15 deliverables. Code in this repo is a runnable foundation that implements the **bold-italicized** pieces; everything else is specified well enough to implement incrementally.

> **Status note (October 2026).** This is the design written at the start of the project and it describes the *target* architecture. Several pieces are still design only (separate microservices, MLflow, NATS, Celery, Kubernetes, Timescale hypertables). The [README](./README.md) lists what is implemented and what the experiments found. One change from the original plan: the web UI is now a Python app (FastAPI + Jinja2 templates + Tailwind + Alpine.js, no Node toolchain) instead of Next.js; see [§14](#14-frontend-architecture--ux).

---

## Table of contents

1. [Product principles](#1-product-principles)
2. [High-level architecture](#2-high-level-architecture)
3. [Monorepo layout](#3-monorepo-layout)
4. [Database schema](#4-database-schema)
5. [API surface](#5-api-surface)
6. [AI / ML pipeline](#6-ai--ml-pipeline)
7. [Forecasting methodology](#7-forecasting-methodology)
8. [External factors & market intelligence](#8-external-factors--market-intelligence)
9. [Recommendation engine](#9-recommendation-engine)
10. [Portfolio & risk engine](#10-portfolio--risk-engine)
11. [Backtesting engine](#11-backtesting-engine)
12. [News + sentiment intelligence](#12-news--sentiment-intelligence)
13. [AI chat assistant](#13-ai-chat-assistant)
14. [Frontend architecture & UX](#14-frontend-architecture--ux)
15. [Tech stack & OSS leverage](#15-tech-stack--oss-leverage)
16. [Security & reliability](#16-security--reliability)
17. [Performance & scaling](#17-performance--scaling)
18. [Deployment plan](#18-deployment-plan)
19. [Monetization](#19-monetization)
20. [Roadmap & future improvements](#20-roadmap--future-improvements)

---

## 1. Product principles

- **Explainable over magical.** Every prediction shows its contributing factors with quantitative weights.
- **Hybrid, not just deep-learning.** Technical + fundamental + sentiment + macro signals are first-class inputs.
- **Regime-aware.** The system explicitly detects bull/bear/sideways/high-vol regimes and routes to different model weights.
- **Real numbers, real risk.** Sharpe, max drawdown, VaR, beta, and confidence intervals are surfaced everywhere a number is shown.
- **Modular by service boundary.** Each capability (market-data, ml, sentiment, portfolio, backtest) is independently deployable.
- **No black boxes for the user.** Recommendation, forecast, and risk all carry "why this answer" payloads.

---

## 2. High-level architecture

```
                          ┌──────────────────────┐
                          │   Web UI (Python)    │  (FastAPI + Jinja2, Tailwind, Alpine.js)
                          │  TradingView Charts  │
                          └──────────┬───────────┘
                                     │  HTTPS / WSS
                          ┌──────────▼───────────┐
                          │   API Gateway (BFF)  │  FastAPI + JWT + rate-limit
                          └──┬───┬───┬───┬───┬───┘
                             │   │   │   │   │
        ┌────────────────────┘   │   │   │   └──────────────────┐
        │                        │   │   │                      │
┌───────▼────────┐  ┌────────────▼─┐ │ ┌─▼──────────┐  ┌────────▼──────┐
│ market-data    │  │  ml-service  │ │ │ portfolio  │  │ news/sentiment│
│ (yfinance,     │  │ (forecast,   │ │ │ + backtest │  │  (RSS, NewsAPI│
│  polygon, fh)  │  │  explain,    │ │ │  vectorbt) │  │  finBERT)     │
└───────┬────────┘  │  SHAP)       │ │ └──────┬─────┘  └──────┬────────┘
        │           └──────┬───────┘ │        │                │
        │                  │         │        │                │
        │           ┌──────▼────┐    │        │                │
        │           │ MLflow +  │    │        │                │
        │           │ Feast FS  │    │        │                │
        │           └──────┬────┘    │        │                │
        │                  │         │        │                │
┌───────▼──────────────────▼─────────▼────────▼────────────────▼───────┐
│                       Postgres (TimescaleDB) + Redis                  │
│                       S3/MinIO (model artifacts, news cache)          │
│                       Kafka/NATS (events: ticks, news, predictions)   │
└───────────────────────────────────────────────────────────────────────┘
```

### Service boundaries (deployable units)

| Service | Language | Purpose |
|---|---|---|
| `web` | Python / FastAPI + Jinja2 | Server-rendered pages, `/api` BFF proxy, websocket consumer |
| `api` | Python / FastAPI | Auth, REST/WS gateway, business logic, portfolio, auth, user data |
| `market-data` | Python | Provider adapters (yfinance/polygon/finnhub/alpha vantage), caching, normalization, websocket fan-out |
| `ml-service` | Python | Forecasting, explainability, ensemble, drift detection |
| `news-service` | Python | RSS/NewsAPI ingestion, finBERT sentiment, entity recognition, event detection |
| `social-service` | Python | Reddit/StockTwits/X ingestion, hype/panic detection |
| `backtest-service` | Python | Strategy DSL, vectorbt execution, monte carlo |
| `chat-service` | Python | LLM (Claude/GPT) with tool-calling against the other services |
| `worker` | Python | Celery/RQ for scheduled retraining, news ingestion, daily portfolio snapshots |

For v0 (this repo), `market-data`, `ml-service`, `news-service`, `backtest-service`, `chat-service` are **modules inside the `api` process** to keep local dev simple. They are designed to split into separate services without code rewrites — they communicate only via typed function interfaces, no shared state besides Postgres/Redis.

### Async event bus

Use **NATS JetStream** (lighter than Kafka, good enough for this domain) for:
- `ticks.{symbol}` — live price ticks
- `news.created` — newly ingested article
- `prediction.created` — model output for symbol/horizon
- `event.detected` — earnings, FOMC, lawsuit, etc.
- `portfolio.changed` — user edit
- `signal.generated` — AI buy/sell signal

Consumers fan out to frontend via WebSockets.

---

## 3. Monorepo layout

```
StockPlatform/
├── ARCHITECTURE.md                  # this doc
├── README.md                        # quickstart
├── docker-compose.yml               # postgres, redis, api, web (+ optional mlflow, minio, nats)
├── .env.example                     # required env vars
├── apps/
│   ├── web/                         # Python web UI: FastAPI + Jinja2 templates, Tailwind, Alpine.js
│   │   ├── app/                     # page routes (nav.py) + /api proxy (main.py)
│   │   ├── templates/pages/         # one template per page
│   │   │   ├── dashboard.html
│   │   │   ├── stock.html
│   │   │   ├── portfolio.html
│   │   │   ├── strategy_lab.html
│   │   │   ├── ai_chat.html
│   │   │   └── api/[...path]/route.ts   # BFF proxy
│   │   ├── components/
│   │   │   ├── chart/CandlestickChart.tsx
│   │   │   ├── chart/IndicatorOverlay.tsx
│   │   │   ├── search/StockSearch.tsx
│   │   │   ├── forecast/ForecastCard.tsx
│   │   │   ├── forecast/ExplainabilityBreakdown.tsx
│   │   │   ├── recommendation/RecCard.tsx
│   │   │   ├── portfolio/PositionTable.tsx
│   │   │   └── ui/* (button, card, tabs, ...)
│   │   ├── lib/api.ts               # typed API client
│   │   ├── lib/ws.ts                # websocket client
│   │   └── lib/theme.ts
│   ├── api/                         # FastAPI service
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── deps.py              # DI: db session, current user, redis
│   │   │   ├── db/
│   │   │   │   ├── base.py
│   │   │   │   ├── session.py
│   │   │   │   └── models.py        # SQLAlchemy models
│   │   │   ├── schemas/             # Pydantic
│   │   │   ├── routers/
│   │   │   │   ├── auth.py
│   │   │   │   ├── stocks.py
│   │   │   │   ├── portfolio.py
│   │   │   │   ├── ai.py            # forecast, recommend, explain
│   │   │   │   ├── news.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── strategy.py      # backtest
│   │   │   │   └── ws.py            # websockets
│   │   │   ├── services/
│   │   │   │   ├── market_data.py   # yfinance wrapper + cache
│   │   │   │   ├── indicators.py    # SMA, EMA, RSI, MACD, BB, VWAP, ATR, ...
│   │   │   │   ├── fundamentals.py
│   │   │   │   ├── recommendation.py
│   │   │   │   ├── risk.py          # beta, sharpe, drawdown, VaR
│   │   │   │   ├── sentiment.py     # finBERT wrapper
│   │   │   │   └── news.py          # ingestion + summarization
│   │   │   ├── ml/
│   │   │   │   ├── base.py          # Forecaster ABC
│   │   │   │   ├── features.py      # feature pipeline
│   │   │   │   ├── xgboost_model.py # baseline (implemented)
│   │   │   │   ├── lstm_model.py    # stub
│   │   │   │   ├── transformer.py   # stub
│   │   │   │   ├── tft.py           # stub (use neuralforecast)
│   │   │   │   ├── prophet_model.py # stub
│   │   │   │   ├── arima_model.py   # stub
│   │   │   │   ├── ensemble.py      # weighted vote + regime routing
│   │   │   │   ├── regime.py        # HMM / volatility regime detector
│   │   │   │   ├── explain.py       # SHAP wrapper
│   │   │   │   └── registry.py      # MLflow client
│   │   │   ├── backtest/
│   │   │   │   ├── engine.py        # vectorbt wrapper
│   │   │   │   └── strategies.py    # SMA cross, RSI, MACD, breakout, ML-driven
│   │   │   └── core/
│   │   │       ├── security.py      # JWT, hashing, oauth helpers
│   │   │       ├── rate_limit.py
│   │   │       └── logging.py
│   │   ├── alembic/                 # migrations
│   │   ├── tests/
│   │   ├── pyproject.toml
│   │   └── Dockerfile
│   └── ml/                          # (future) split-out ml-service
├── packages/
│   ├── shared-types/                # TS + Python (datamodel.codegen) shared schemas
│   └── ui/                          # reusable React components if multi-app
├── infra/
│   ├── k8s/                         # production manifests (helm chart)
│   ├── terraform/                   # cloud infra
│   └── grafana/                     # dashboards
└── scripts/
    ├── seed_universe.py             # populate stock universe table
    ├── train_baseline.py            # train XGBoost baseline
    └── ingest_news.py
```

---

## 4. Database schema

PostgreSQL with the **TimescaleDB** extension for time-series tables. Migrations via Alembic.

### Core tables

```sql
-- users & auth
users (
  id UUID PK,
  email CITEXT UNIQUE NOT NULL,
  password_hash TEXT,            -- nullable if oauth-only
  display_name TEXT,
  totp_secret TEXT,              -- 2FA
  role TEXT DEFAULT 'user',      -- user | admin
  created_at TIMESTAMPTZ,
  last_login_at TIMESTAMPTZ
);

oauth_accounts (
  id UUID PK, user_id UUID FK, provider TEXT, provider_user_id TEXT,
  UNIQUE(provider, provider_user_id)
);

refresh_tokens (
  id UUID PK, user_id UUID FK, token_hash TEXT, expires_at TIMESTAMPTZ, revoked_at TIMESTAMPTZ
);

-- universe of tradable instruments
instruments (
  symbol TEXT PK,                 -- "AAPL", "BTC-USD", "EURUSD=X"
  name TEXT,
  exchange TEXT,
  asset_class TEXT,              -- stock | etf | index | crypto | forex
  sector TEXT, industry TEXT,
  country TEXT, currency TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  metadata JSONB                  -- shares outstanding, ipo date, etc.
);
CREATE INDEX ON instruments USING gin (to_tsvector('simple', symbol || ' ' || coalesce(name,'')));

-- OHLCV (TimescaleDB hypertable)
price_bars (
  symbol TEXT,
  ts TIMESTAMPTZ,
  open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
  volume BIGINT,
  resolution TEXT,                -- 1m, 5m, 1h, 1d
  PRIMARY KEY (symbol, resolution, ts)
);
SELECT create_hypertable('price_bars', 'ts');

-- fundamentals (snapshot per quarter/year)
fundamentals (
  id BIGSERIAL PK, symbol TEXT, period_end DATE, period_type TEXT,
  pe NUMERIC, pb NUMERIC, eps NUMERIC, revenue NUMERIC, net_income NUMERIC,
  debt_to_equity NUMERIC, free_cash_flow NUMERIC, gross_margin NUMERIC,
  raw JSONB,
  UNIQUE (symbol, period_end, period_type)
);

-- watchlists & portfolios
watchlists (id UUID PK, user_id UUID FK, name TEXT, created_at TIMESTAMPTZ);
watchlist_items (watchlist_id UUID FK, symbol TEXT, PRIMARY KEY (watchlist_id, symbol));

portfolios (id UUID PK, user_id UUID FK, name TEXT, base_currency TEXT, created_at TIMESTAMPTZ);

transactions (
  id UUID PK, portfolio_id UUID FK, symbol TEXT,
  side TEXT,                      -- buy | sell
  quantity NUMERIC, price NUMERIC, fee NUMERIC DEFAULT 0,
  occurred_at TIMESTAMPTZ, note TEXT
);
-- positions are derived from transactions; cached via materialized view

CREATE MATERIALIZED VIEW positions AS
SELECT
  portfolio_id, symbol,
  SUM(CASE WHEN side='buy' THEN quantity ELSE -quantity END) AS quantity,
  SUM(CASE WHEN side='buy' THEN quantity*price ELSE 0 END)
    / NULLIF(SUM(CASE WHEN side='buy' THEN quantity ELSE 0 END),0) AS avg_buy_price
FROM transactions GROUP BY portfolio_id, symbol;

-- predictions
predictions (
  id BIGSERIAL PK, symbol TEXT, model TEXT, horizon TEXT,  -- 1d, 5d, 30d, 90d
  created_at TIMESTAMPTZ,
  target_ts TIMESTAMPTZ,
  point_estimate NUMERIC,
  lower_80 NUMERIC, upper_80 NUMERIC,
  lower_95 NUMERIC, upper_95 NUMERIC,
  confidence NUMERIC,              -- 0..1
  contributions JSONB              -- {technical: 0.35, sentiment: 0.20, ...}
);
CREATE INDEX ON predictions (symbol, horizon, created_at DESC);

-- recommendations
recommendations (
  id BIGSERIAL PK, symbol TEXT, created_at TIMESTAMPTZ,
  label TEXT,                      -- strong_buy|buy|hold|sell|strong_sell
  score NUMERIC,                   -- -1..1
  confidence NUMERIC,
  reasoning JSONB                  -- structured "why" payload
);

-- news + sentiment
news_articles (
  id BIGSERIAL PK,
  source TEXT, url TEXT UNIQUE,
  title TEXT, summary TEXT, body TEXT,
  published_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ,
  entities JSONB,                  -- tickers, people, orgs
  sentiment NUMERIC,               -- -1..1
  sentiment_confidence NUMERIC,
  topics JSONB,
  impact_score NUMERIC             -- estimated short-term price impact
);
CREATE INDEX ON news_articles USING gin (entities);

social_posts (
  id BIGSERIAL PK, platform TEXT, post_id TEXT UNIQUE,
  symbol TEXT, author TEXT, content TEXT,
  posted_at TIMESTAMPTZ, sentiment NUMERIC, hype_score NUMERIC, ingested_at TIMESTAMPTZ
);

-- strategies & backtests
strategies (
  id UUID PK, user_id UUID FK, name TEXT,
  dsl JSONB,                       -- typed spec (see §11)
  created_at TIMESTAMPTZ
);

backtests (
  id UUID PK, strategy_id UUID FK,
  symbol TEXT, start_date DATE, end_date DATE,
  initial_cash NUMERIC,
  metrics JSONB,                   -- cagr, sharpe, mdd, win_rate, ...
  equity_curve JSONB,              -- compact, downsampled
  trades JSONB,
  created_at TIMESTAMPTZ
);

-- ml model registry mirror (the source of truth is MLflow)
models (
  id UUID PK, name TEXT, version TEXT, symbol_scope TEXT,  -- '*' or 'AAPL'
  task TEXT,                       -- forecast | classify | regime
  metrics JSONB,
  artifact_uri TEXT,               -- s3://bucket/model.pkl
  trained_at TIMESTAMPTZ,
  is_active BOOLEAN
);

-- audit
audit_log (
  id BIGSERIAL PK, user_id UUID, action TEXT, resource TEXT,
  ip INET, ua TEXT, ts TIMESTAMPTZ, payload JSONB
);
```

### Retention

- `price_bars` 1m/5m: keep 90 days raw, then downsample to 1h (Timescale continuous aggregates).
- `price_bars` 1d: keep forever.
- `predictions`: keep 2 years (use for model drift / calibration tracking).
- `news_articles`: keep 1 year body, embeddings forever.

---

## 5. API surface

REST is the primary interface; WebSockets stream real-time updates. All responses are JSON. All authenticated routes require `Authorization: Bearer <jwt>`.

### Auth
```
POST   /auth/register                {email, password}
POST   /auth/login                   {email, password} → {access, refresh}
POST   /auth/refresh                 {refresh} → {access}
POST   /auth/2fa/enable              → {secret, qr_url}
POST   /auth/2fa/verify              {code}
POST   /auth/logout
GET    /auth/me
```

### Stocks / market data
```
GET  /stocks/search?q=appl&limit=20             → [{symbol, name, exchange, asset_class}]
GET  /stocks/{symbol}                            → {profile, latest_quote, key_stats}
GET  /stocks/{symbol}/history
       ?interval=1d&range=1y                    → [{ts, o, h, l, c, v}]
GET  /stocks/{symbol}/indicators
       ?names=sma_20,rsi_14,macd                → {sma_20: [...], rsi_14: [...], ...}
GET  /stocks/{symbol}/fundamentals               → {pe, eps, revenue_growth, ...}
GET  /stocks/{symbol}/news?limit=20              → [{title, sentiment, impact, ...}]
GET  /stocks/{symbol}/peers                      → [...]
WS   /ws/quotes?symbols=AAPL,TSLA                → live ticks
```

### AI
```
POST /ai/forecast
  body: {symbol, horizons:[1d,5d,30d,90d], models:["xgboost","ensemble"]}
  → {forecasts: [{horizon, point, lower_80, upper_80, lower_95, upper_95,
                  confidence, contributions}]}

POST /ai/recommend                {symbol}
  → {label, score, confidence, reasoning: {
       technical: {...}, fundamental: {...}, sentiment: {...},
       macro: {...}, events: [...], risk_flags: [...]
     }}

POST /ai/explain                  {symbol, prediction_id}
  → SHAP-style feature attributions

POST /ai/portfolio-analysis       {portfolio_id}
  → {risk_score, diversification, beta, sharpe, suggestions: [...]}

POST /ai/chat                     {session_id, message, context?}  (streams SSE)
```

### Portfolio
```
GET    /portfolios
POST   /portfolios                          {name, base_currency}
GET    /portfolios/{id}                     → summary + positions + metrics
POST   /portfolios/{id}/transactions        {symbol, side, qty, price, occurred_at}
DELETE /portfolios/{id}/transactions/{tid}
GET    /portfolios/{id}/performance?range=1y
GET    /portfolios/{id}/risk
```

### Strategy & backtest
```
POST /strategies                            {name, dsl}
GET  /strategies
POST /strategies/{id}/backtest              {symbol, start, end, initial_cash}
GET  /backtests/{id}                        → metrics + equity curve + trades
POST /backtests/{id}/montecarlo             {n_simulations}
```

### Watchlists
```
GET    /watchlists
POST   /watchlists                          {name}
POST   /watchlists/{id}/items               {symbol}
DELETE /watchlists/{id}/items/{symbol}
```

### Rate limits (defaults)

- Anonymous: 30 req/min/IP
- Authenticated (free): 120 req/min/user
- Authenticated (pro): 1200 req/min/user
- AI endpoints: separate budget (10/min free, 120/min pro)

Implemented via Redis token buckets (see `core/rate_limit.py`).

---

## 6. AI / ML pipeline

### Data flow

```
   raw provider data  ──►  market-data service  ──►  Postgres/TimescaleDB
       (yfinance,                 (normalize +              │
        polygon, ...)              cache)                   │
                                                            ▼
   news/social/macro  ──►  ingestion workers   ──►  news, social_posts, macro_series
                                                            │
                                                            ▼
                                                  ┌─────────────────┐
                                                  │ Feature pipeline │  (apps/api/app/ml/features.py)
                                                  │  - technical     │
                                                  │  - fundamental   │
                                                  │  - sentiment agg │
                                                  │  - macro context │
                                                  │  - regime tag    │
                                                  └────────┬─────────┘
                                                           │
                                            ┌──────────────┼──────────────┐
                                            ▼              ▼              ▼
                                       ┌────────┐   ┌───────────┐   ┌──────────┐
                                       │XGBoost │   │  LSTM/    │   │Prophet/  │
                                       │(impl)  │   │  TFT (NF) │   │ ARIMA    │
                                       └───┬────┘   └─────┬─────┘   └────┬─────┘
                                           │              │               │
                                           └──────┬───────┴───────┬───────┘
                                                  ▼               ▼
                                          ┌───────────────────────────────┐
                                          │  Ensemble (regime-weighted)   │
                                          │  + SHAP explanation merger    │
                                          └──────────────┬────────────────┘
                                                         ▼
                                                  predictions table
                                                         │
                                                         ▼
                                          ┌───────────────────────────────┐
                                          │  Recommendation engine        │
                                          │   = f(forecast, indicators,   │
                                          │       sentiment, macro,       │
                                          │       fundamentals, risk)     │
                                          └──────────────┬────────────────┘
                                                         ▼
                                                  recommendations table
```

### Feature catalog (initial)

**Price-derived (per bar, multi-horizon):** returns 1/5/20/60d, log-vol 5/20/60d, ATR, drawdown, RSI(14), MACD hist, BB %B, OBV slope, VWAP gap.

**Fundamental (current snapshot):** PE, PB, EPS growth YoY, revenue growth YoY, debt/equity, FCF yield, gross margin, insider net buys (90d).

**Sentiment (rolling windows):** finBERT news sentiment mean & dispersion 1/7/30d, article count z-score, social hype score, social polarity 1/7d.

**Macro (joined by date):** 10Y yield, 2Y-10Y spread, DXY, WTI, gold, VIX, CPI YoY, unemployment.

**Event flags (binary, decaying):** earnings T-7..T+3, FOMC ±5d, ex-div ±2d.

**Regime tags (one-hot):** {bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol, sideways}.

Total v0 feature count: ~80. Stored to a feature table (or Feast feature store later).

### Training cadence

- **Daily retrain** of fast models (XGBoost, Prophet) — overnight, per universe slice.
- **Weekly retrain** of deep models (LSTM/TFT) on GPU worker.
- **Continuous validation**: each new prediction is logged with its actual outcome on close → calibration & drift metrics in MLflow.
- **Drift alert** if rolling 30d MAPE > 1.5× baseline → auto-trigger retrain & open a model_alert.

### Hyperparameter tuning

`optuna` with multi-objective: minimize CRPS (calibration-aware), maximize directional accuracy. Budget: 50 trials/model/week.

### Model registry

**MLflow** (single source of truth) mirrored to the `models` table for fast queries. Active model per (symbol_scope, horizon) is flagged.

---

## 7. Forecasting methodology

### Horizons

| Horizon | Use | Primary models |
|---|---|---|
| 1d | Day-trader signal, intraday risk | XGBoost on intraday + sentiment |
| 5d | Swing trading | XGBoost ensemble, LSTM |
| 30d | Tactical positioning | TFT, LSTM, Prophet |
| 90d | Strategic / fundamental | TFT, fundamental-weighted regressor |

### Output contract (every forecast)

```json
{
  "symbol": "NVDA",
  "horizon": "30d",
  "as_of": "2026-05-26T20:00:00Z",
  "target_date": "2026-06-25",
  "point": 178.42,
  "intervals": {
    "p10": 152.10, "p25": 165.30, "p50": 178.42,
    "p75": 192.80, "p90": 211.05
  },
  "direction_prob": { "up": 0.61, "down": 0.39 },
  "expected_volatility_pct": 28.5,
  "confidence": 0.62,
  "contributions": {
    "technical": 0.32,
    "fundamental": 0.18,
    "sentiment_news": 0.21,
    "sentiment_social": 0.07,
    "macro": 0.12,
    "events": 0.10
  },
  "drivers": [
    {"name": "RSI_14 oversold reversal", "direction": "up", "weight": 0.11},
    {"name": "Q1 earnings beat (+22%)", "direction": "up", "weight": 0.17},
    {"name": "10Y yield rising trend", "direction": "down", "weight": 0.08}
  ],
  "model_mix": { "xgboost": 0.5, "lstm": 0.3, "prophet": 0.2 }
}
```

This payload is what the **ForecastCard** in the UI renders. The `drivers` array gives the user a plain-English "why."

### Confidence intervals

- Tree/linear models → conformal prediction (`mapie`) for distribution-free intervals.
- Deep models → quantile loss (predict p10/p50/p90 directly).
- Ensemble → weighted quantile aggregation.

### Direction probability

A separate binary classifier (XGBoost) predicts P(close at T+h > close at T). Surfaced separately because users often care about direction more than point estimate.

### Calibration

Tracked via reliability diagrams; auto-isotonic recalibration if Brier score drifts > 10% from baseline.

---

## 8. External factors & market intelligence

This is the most differentiated part of the platform.

### News pipeline

1. **Sources:** NewsAPI, AlphaVantage news endpoint, Finnhub news, RSS from major outlets (Reuters, Bloomberg-via-Yahoo, MarketWatch, FT-RSS), SEC EDGAR (8-K, 10-K, 13F).
2. **Ingestion:** `news-service` polls every 60s for high-priority symbols (in any user watchlist or top-500 by volume), every 5m for the rest.
3. **NLP:**
   - Entity extraction: spaCy + custom ticker linker (Aho-Corasick over instrument names + ticker patterns).
   - Sentiment: **finBERT** (ProsusAI/finbert) for finance-tuned polarity.
   - Topic: zero-shot classification (BART-MNLI) over a fixed taxonomy: earnings, M&A, regulation, lawsuit, product, leadership, macro, analyst-rating.
   - Event detection: rules + LLM (Claude) for novel events.
4. **Impact estimation:** a small regression model predicts |Δprice over 1d| given (topic, sentiment, source authority, related volume). Output is `impact_score ∈ [0,1]`.
5. **Storage:** `news_articles` table; embeddings (text-embedding-3-small) in pgvector for semantic search.

### Social pipeline

- **Reddit:** PRAW with subreddit allow-list (`r/wallstreetbets`, `r/stocks`, `r/investing`, ticker-specific).
- **X/Twitter:** filtered firehose by cashtag (requires v2 API access).
- **StockTwits:** public stream API.
- **Signals computed per (symbol, window):**
  - `mention_count_zscore` (rolling 30d baseline)
  - `polarity` (finBERT applied to posts)
  - `hype_score` = ema(mention_count) × |polarity|
  - `panic_index` = share of negative posts with high engagement

### Macroeconomic series

Ingested from FRED (free) and ECB SDW. Cached daily. Used as features and surfaced in the UI's "Macro context" card.

### Event-aware forecasting

A dedicated **event detector** publishes typed events to `event.detected`:

```python
class DetectedEvent(BaseModel):
    symbol: str | None        # None for macro events
    type: Literal["earnings", "product_launch", "lawsuit", "regulation",
                  "geopolitical", "supply_chain", "ai_announcement",
                  "cyber_incident", "fomc", "cpi_release", "merger"]
    direction_bias: Literal["up", "down", "uncertain"]
    expected_volatility_increase: float    # multiplier
    confidence: float
    source_article_ids: list[int]
    valid_until: datetime
```

The ensemble layer reads active events for the target symbol and applies an **event-conditioned residual model** that adjusts the point forecast and widens intervals.

### Adaptive regime weighting

A regime detector (gaussian HMM on returns + VIX + breadth) classifies the current market into one of 5 regimes. The ensemble has per-regime weights learned offline:

```python
ENSEMBLE_WEIGHTS = {
  "bull_low_vol":   {"xgboost": 0.5, "lstm": 0.3, "prophet": 0.2, "sentiment_blend": 0.10},
  "bear_high_vol":  {"xgboost": 0.3, "lstm": 0.2, "prophet": 0.1, "sentiment_blend": 0.40},
  ...
}
```

So during high-vol regimes, sentiment and news automatically dominate; during stable regimes, technical patterns lead.

### Explainability (the "why")

Every prediction's `contributions` field is computed as:

```
contributions[category] = sum(|SHAP value|) of features in that category
                          / sum(|SHAP value|) of all features
```

For deep models that don't natively expose SHAP, use **integrated gradients** (Captum). Categories map to user-facing buckets: technical / fundamental / sentiment_news / sentiment_social / macro / events.

---

## 9. Recommendation engine

The recommendation is **not the forecast** — it's a decision derived from forecast + risk + context.

### Inputs

```python
RecommendationInput = {
    "forecast_1d": Forecast, "forecast_5d": Forecast,
    "forecast_30d": Forecast, "forecast_90d": Forecast,
    "indicators": {...},            # current values
    "fundamentals": {...},
    "news_sentiment_7d": float,
    "social_hype": float,
    "macro_regime": str,
    "active_events": list[DetectedEvent],
    "current_price": float,
    "atr": float,                   # for stop placement
}
```

### Scoring

A bounded score in `[-1, +1]` from a weighted blend:

```
score = w_d * direction_signal       # from forecasts
      + w_t * technical_signal       # rsi/macd/bb composite
      + w_f * fundamental_signal     # value + growth z-scores
      + w_s * sentiment_signal       # news + social
      + w_m * macro_signal
      - w_r * risk_penalty           # if drawdown/vol very high
```

Weights are regime-conditioned (same table as forecast ensemble).

### Label mapping

| Score range | Label |
|---|---|
| `> 0.6` | STRONG_BUY |
| `0.2 .. 0.6` | BUY |
| `-0.2 .. 0.2` | HOLD |
| `-0.6 .. -0.2` | SELL |
| `< -0.6` | STRONG_SELL |

Plus a `confidence ∈ [0,1]` derived from forecast confidence × signal agreement.

### Reasoning payload (structured)

```json
{
  "label": "BUY",
  "score": 0.42,
  "confidence": 0.71,
  "reasoning": {
    "technical": {
      "summary": "RSI exiting oversold; MACD bullish cross 2 days ago.",
      "bullish": ["RSI 14 = 36 (rising)", "MACD histogram positive"],
      "bearish": ["Below 200-day SMA"]
    },
    "fundamental": {
      "summary": "Strong earnings momentum; valuation slightly stretched.",
      "highlights": [
        "EPS growth YoY +22%",
        "Revenue growth +17%",
        "P/E 38 vs sector median 24"
      ]
    },
    "sentiment": { "summary": "News sentiment +0.31 (7d avg). Social hype elevated but not extreme." },
    "macro": { "summary": "Sector tailwind from declining real yields." },
    "events": [
      { "type": "earnings", "in_days": 12, "direction_bias": "up", "vol_multiplier": 1.6 }
    ],
    "risk": {
      "atr_pct_of_price": 3.4,
      "max_drawdown_90d_pct": 11.2,
      "suggested_stop": 162.40,
      "suggested_target_1": 192.00,
      "position_size_pct_of_portfolio_max": 5.0
    }
  }
}
```

Users see a **RecCard** with these sections rendered as collapsible accordions and a "what would change my mind" tooltip on each driver.

---

## 10. Portfolio & risk engine

### Computed per portfolio

- **Market value** (sum of `qty × last_price`)
- **Cost basis** (sum of buys, FIFO/avg-cost — user-selectable)
- **Unrealized P&L**, **realized P&L** (from closed lots)
- **Time-weighted return** (Modified Dietz daily snapshots)
- **Beta** vs benchmark (default SPY) — 1y rolling
- **Sharpe**, **Sortino** — 1y trailing
- **Max drawdown** — full history
- **VaR(95)** — historical and parametric
- **Sector / asset-class allocation** with concentration warnings (HHI)
- **Correlation matrix** between holdings

### AI portfolio analysis

- **Overexposure detection:** any single name > 25%, any sector > 40%, or correlation cluster > 60% triggers a flag.
- **Rebalancing suggestions:** mean-variance optimization (`scipy` or `riskfolio-lib`) with current weights as warm start and turnover penalty.
- **Hedging suggestions:** for high-beta portfolios, suggest inverse ETFs or index puts sized to neutralize 30-50% of beta.
- **Tax-loss harvesting candidate detector** (lots with unrealized losses, no wash-sale conflict in last 30d).

---

## 11. Backtesting engine

Built on **vectorbt** (vectorized, fast) with a typed strategy DSL on top so strategies are storable as JSON.

### Strategy DSL example

```json
{
  "name": "EMA cross + RSI filter",
  "rules": [
    { "if": "ema(close, 12) > ema(close, 26)",
      "and": "rsi(close, 14) > 40",
      "action": "enter_long",
      "size": "10% equity" },
    { "if": "ema(close, 12) < ema(close, 26)",
      "action": "exit_long" }
  ],
  "stops": { "stop_loss_pct": 5, "trailing_stop_pct": 8 },
  "execution": { "slippage_bps": 5, "commission_bps": 1 }
}
```

The DSL compiles to vectorbt signals. AI-driven strategies expose `predict()` from a registered model as a signal source.

### Outputs

- Equity curve (downsampled to ≤2000 points for web)
- CAGR, total return, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, average win/loss, expectancy, trade list
- Benchmark overlay (default SPY)
- **Monte Carlo:** bootstrap resampling of trade returns × N simulations → percentile bands on terminal equity and max drawdown distribution
- **Walk-forward optimization** for parameter sweeps

---

## 12. News + sentiment intelligence

(See §8 for the pipeline; this section is the **product surface**.)

### "Intelligence feed" page

- Reverse-chronological stream of news + social, filtered by user's watchlist + portfolio.
- Each item: title, source, time, sentiment chip, impact bar, affected tickers, "summary" (LLM-generated 2-line), "what this means" (LLM, optional, costs an AI credit).
- Aggregations per symbol: 24h sentiment bar, 7d sentiment sparkline, share-of-voice vs sector.
- "Trending" panel: symbols with unusual mention spikes (z-score > 2).

### Earnings call summarization

When an earnings transcript drops, the chat-service summarizes into:
- Headline numbers (beat/miss vs consensus)
- Guidance change
- Key themes (extracted spans with source quotes)
- Tone shift vs previous call (delta in sentiment, hedging language frequency)

### SEC filings

8-K → LLM extracts material event type, classifies impact, links to article surfaces.

---

## 13. AI chat assistant

LLM with **tool-calling** against the platform's own APIs. Default model: Claude Sonnet (or Opus on Pro plan).

### Tools exposed to the assistant

```
get_quote(symbol)
get_history(symbol, interval, range)
get_indicators(symbol, names)
get_fundamentals(symbol)
get_news(symbol, limit)
get_forecast(symbol, horizon)
get_recommendation(symbol)
get_portfolio_summary(portfolio_id)
get_portfolio_risk(portfolio_id)
compare_symbols(symbols, metrics)
run_backtest(strategy_dsl, symbol, range)
search_universe(query)
```

The assistant has read-only access to the **current user's portfolio** by default; trade execution is **never** exposed (this platform doesn't execute trades — see §16).

### Context

Each chat session carries:
- User id, active portfolio id
- Time anchor (for "as of today" reasoning)
- Recently viewed symbols (last 10)
- Active recommendations (for "should I buy NVDA?" the system pre-attaches the latest rec)

### Streaming

SSE over HTTP. Frontend renders intermediate tool calls as collapsible "thinking" cards.

---

## 14. Frontend architecture & UX

### Stack

- **FastAPI + Jinja2** serve one server-rendered template per page and proxy `/api/*` to the backend (same origin, no CORS)
- **Tailwind CSS** (Play CDN, same theme tokens as the original design)
- **Alpine.js** for interactivity (state, fetches, live updates) directly in the HTML, with no build step
- **TradingView Lightweight Charts** (MIT) for candlestick + indicator overlays
- **Chart.js** for everything non-financial (portfolio donuts, equity and comparison lines)
- **JWT** access/refresh tokens from the API, kept in `localStorage`

*Originally planned and first built as Next.js 15 + React; rebuilt as a Python app so the project needs no Node toolchain (the browser still runs small Alpine.js snippets).*

### Pages

| Route | Purpose |
|---|---|
| `/` | Marketing / login |
| `/dashboard` | Watchlist + portfolio glance + top movers + intelligence feed |
| `/stocks/[symbol]` | Big chart, indicator drawer, AI forecast panel, recommendation, news, fundamentals |
| `/portfolio` | Positions, allocation, performance, AI risk panel |
| `/strategy-lab` | Build strategy (form-based or DSL editor), run backtest, view results |
| `/screener` | Filter universe by metrics (PE, growth, sentiment, AI score) |
| `/ai-chat` | Persistent chat with portfolio context |
| `/settings` | Profile, API keys, billing, 2FA |

### Design system

- **Dark by default**, light theme available.
- Palette: near-black `#0B0E14` background; surfaces `#11151D` / `#161B26`; accent neon-cyan `#5EEAD4`; success `#22C55E`; danger `#EF4444`; gridlines `#1E2533`.
- **Typography:** Inter for UI, JetBrains Mono for numbers (tabular figures!). Always use `font-variant-numeric: tabular-nums` on price columns.
- **Density:** Bloomberg-like density on `/stocks/[symbol]` (lots of data per pixel) vs Robinhood-clean on `/dashboard`.
- **Motion:** Framer Motion for non-chart transitions. No animation on numeric updates beyond a 200ms color flash (green/red) — distracting otherwise.
- **Glassmorphism** reserved for the floating AI panel and modals; surfaces themselves are solid.
- **Responsiveness:** desktop-first; tablet supported; mobile is a separate condensed layout (no multi-pane).

### Key components

- `<CandlestickChart>` — wraps lightweight-charts, accepts `data`, `indicators[]`, `overlays[]`. Exposes imperative API for drawing tools.
- `<ForecastCard>` — point + intervals + driver chips + "explain" expander.
- `<RecCard>` — label badge with score gauge + reasoning accordion.
- `<ExplainabilityBreakdown>` — stacked horizontal bar of contribution categories.
- `<RegimeBadge>` — current market regime pill.
- `<PositionTable>` — virtualized, with inline sparkline per row.
- `<StrategyEditor>` — JSON Schema-driven form + raw DSL toggle.

---

## 15. Tech stack & OSS leverage

We lean heavily on best-in-class OSS so we're not reinventing.

| Concern | Library | License | Why |
|---|---|---|---|
| Charts | [`lightweight-charts`](https://github.com/tradingview/lightweight-charts) | Apache-2.0 | Used by TradingView itself |
| Time-series forecasting | [`statsforecast`](https://github.com/Nixtla/statsforecast), [`neuralforecast`](https://github.com/Nixtla/neuralforecast), [`mlforecast`](https://github.com/Nixtla/mlforecast) | Apache-2.0 | Nixtla family — ARIMA, AutoARIMA, Prophet, TFT, NHiTS, NBEATS |
| Tree boosting | [`xgboost`](https://github.com/dmlc/xgboost), [`lightgbm`](https://github.com/microsoft/LightGBM) | Apache-2.0, MIT | Fast, well-calibrated |
| Conformal intervals | [`mapie`](https://github.com/scikit-learn-contrib/MAPIE) | BSD-3 | Distribution-free intervals |
| Explainability | [`shap`](https://github.com/shap/shap), [`captum`](https://github.com/pytorch/captum) | MIT, BSD-3 | SHAP for trees, IG for deep |
| Backtesting | [`vectorbt`](https://github.com/polakowo/vectorbt) | Apache-2.0 | Fast vectorized |
| Portfolio optimization | [`riskfolio-lib`](https://github.com/dcajasn/Riskfolio-Lib) | BSD-3 | Many objective functions |
| Market data | [`yfinance`](https://github.com/ranaroussi/yfinance) (free), Polygon SDK, Finnhub SDK | Apache-2.0 / proprietary | Provider-pluggable |
| Open finance terminal patterns | [`OpenBB`](https://github.com/OpenBB-finance/OpenBBTerminal) | AGPL — *do not vendor*, just reference for ideas | |
| NLP | [`transformers`](https://huggingface.co/docs/transformers) + [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) | Apache-2.0 | Finance sentiment |
| Embeddings + vector | [`pgvector`](https://github.com/pgvector/pgvector) | PostgreSQL | News semantic search |
| Workflow / scheduling | Celery + Redis, or [`Prefect`](https://github.com/PrefectHQ/prefect) | Apache-2.0 | Data jobs |
| Experiment tracking | [`MLflow`](https://github.com/mlflow/mlflow) | Apache-2.0 | Model registry, runs |
| Feature store (later) | [`Feast`](https://github.com/feast-dev/feast) | Apache-2.0 | Online/offline parity |
| Time-series DB | TimescaleDB (Postgres ext) | Apache-2.0 / TSL | Cheap, familiar SQL |
| Auth | `python-jose` + `passlib[bcrypt]` + `pyotp` (FastAPI) | MIT | |
| Observability | OpenTelemetry, Prometheus, Grafana, Loki | Apache-2.0 | |
| Container | Docker, docker-compose v2; Helm for prod | | |

**License note:** OpenBB is AGPL — we reference its ideas, but do not vendor or fork code into this repo. Everything else above is permissively licensed.

---

## 16. Security & reliability

### Authentication
- Email/password with bcrypt (cost 12).
- OAuth (Google, GitHub) via Auth.js on the web side, federated to API via signed JWT exchange.
- **2FA (TOTP)** required for accounts with portfolio value > $0 (configurable threshold).
- Access JWT (15min) + refresh JWT (30d, rotating, hashed in DB; one-time use).

### Authorization
- Role-based: `user`, `admin`.
- Object-level: every portfolio/strategy/watchlist row is scoped to `user_id` — enforced at the repository layer; defense-in-depth row-level security policy in Postgres.

### Secrets
- `.env` for local dev. `Doppler` / `AWS Secrets Manager` / `Vault` in prod.
- No secret ever logged. Pydantic `SecretStr` for sensitive fields.

### Transport
- HTTPS only (HSTS, TLS 1.2+).
- HttpOnly + Secure + SameSite=Lax cookies for refresh tokens; access token in memory on web.
- CSRF protection on state-changing routes via double-submit cookie.

### Input
- All bodies validated by Pydantic.
- SQL via SQLAlchemy ORM — no string interpolation.
- Symbol allow-list (regex `^[A-Z0-9.\-=^]{1,20}$`) before hitting providers.

### Rate limiting & abuse
- Redis token bucket per (user, endpoint-class) and per IP.
- WAF (Cloudflare) in prod with bot challenges on `/auth/*`.
- Captcha on register + 3 failed logins.

### Audit
- Every auth event, portfolio mutation, strategy run logged to `audit_log` (append-only, daily partition).
- Admin-only `/admin/audit` UI.

### Privacy & compliance
- GDPR: data export and deletion endpoints (`/me/export`, `DELETE /me`).
- PII minimization: store only email + display name.
- **This platform is informational, not advisory or executional.** No order routing, no custody, no fiduciary relationship. Add disclaimer footer and an interstitial on first AI recommendation:
  > "Forecasts and recommendations are model outputs for informational purposes. Not investment advice. Past performance does not guarantee future results."

### Reliability
- All external provider calls wrapped in retry-with-backoff + circuit breaker (`tenacity`, `pybreaker`).
- Health endpoints: `/healthz` (liveness), `/readyz` (readiness — checks DB, Redis, MLflow).
- Postgres: streaming replication + daily logical backup to S3 (encrypted).
- RPO 1h, RTO 1h target.

---

## 17. Performance & scaling

### Caching strategy

| What | Where | TTL |
|---|---|---|
| Quote (last price) | Redis | 5s |
| Intraday bars | Redis | 60s |
| Daily bars | Redis | 1h, invalidated on EOD job |
| Indicators | Computed on-demand from cached bars | derived |
| Forecast (per symbol+horizon) | Postgres `predictions`; Redis short-cache | 5 min for serving; new row written hourly |
| Search results | Redis | 5 min |
| Fundamentals | Redis | 12h |
| News feed | Redis sorted set per symbol | 5 min |

### Async

- Web → API: HTTP. WS for streams.
- API → providers: `httpx.AsyncClient` everywhere.
- Long-running jobs (backtest, large MC sim, model training) → Celery task; client polls or subscribes to `task.<id>.complete` via WS.

### Streaming

- One WS connection per browser tab; multiplexes subscriptions.
- Server fans out from NATS subjects to per-connection subscriber sets.
- Backpressure: drop oldest tick if buffer > N (UX is fine with sub-second latency).

### Horizontal scale

- All services stateless behind a load balancer.
- Sticky-session not required (WS uses Redis pub/sub or NATS).
- Postgres: read replicas for analytical queries (portfolio metrics, backtest reads).
- ML inference: model-server per (model_family, horizon) replica set; route by Envoy/Istio header.

### GPU

- Optional `ml-gpu` deployment for LSTM/TFT inference and training. CPU fallback always works (slower).
- Triton or `vllm` for serving Transformer models if scale demands.

### Observability

- OpenTelemetry traces (FastAPI → service modules → DB → providers).
- Prometheus metrics: request counts, p50/p95/p99, provider error rates, model inference latency, prediction outcome calibration.
- Loki for logs. Grafana dashboards committed to `infra/grafana/`.
- Sentry for exceptions (web + api).

---

## 18. Deployment plan

### Local (this repo)
```
docker compose up -d
# brings up: postgres, redis, api, web
# http://localhost:3000 (web), http://localhost:8000/docs (api)
```

### Staging / single-VM ("phase 1")
- Fly.io or Railway: one Postgres, one Redis, api as a Machine/service, web as a Machine/service.
- Cost: ~$25-50/mo.
- Suitable for first 1,000 users.

### Production ("phase 2")
- AWS or GCP, managed Kubernetes.
- Terraform-managed: VPC, RDS Postgres (Multi-AZ, with Timescale on self-hosted EC2 or use AlloyDB/Aurora), ElastiCache Redis, S3, ECR.
- Helm chart in `infra/k8s/` with separate deployments per service.
- HPA: scale api on CPU + request-rate, scale ml-service on inference queue depth.
- CDN: Cloudflare in front of web.
- CI/CD: GitHub Actions → build, test, push image, ArgoCD deploys to k8s.
- Blue/green for api; canary for ml-service (5% → 25% → 100% based on calibration metrics).

### "Phase 3" (large scale)
- Split market-data, ml-service, news-service into independent deployments.
- Add NATS JetStream for the event bus.
- Add Feast feature store.
- Move heavy training to a Ray cluster.

---

## 19. Monetization

| Tier | Price | Includes |
|---|---|---|
| **Free** | $0 | 1 portfolio, 1 watchlist (20 symbols), daily forecasts, 7d news/sentiment, basic backtests, 10 AI chat msgs/day, ads |
| **Pro** | $19/mo | Unlimited watchlists, intraday forecasts, full news history, advanced backtests + MC, 500 AI chat msgs/day, no ads, exports |
| **Trader** | $49/mo | All Pro + real-time data, options flow, smart-money tracking, API access (10k req/day), 5 portfolios |
| **Quant** | $149/mo | All Trader + custom model training jobs, walk-forward optimization, premium AI models (Opus), 100k API req/day, priority support |
| **Enterprise** | custom | SSO, multi-seat, dedicated infra, custom data feeds, SLAs |

Other revenue:
- **Affiliate** broker links (Alpaca, IBKR) — disclosed.
- **Data marketplace** later: users publish backtested strategies; we take 20% rev share on subscribers.
- **API plans** (Stripe metered billing).

Stripe handles checkout + subscriptions + invoices. Webhook → updates `users.subscription_tier`.

---

## 20. Roadmap & future improvements

### Near-term (next 8 weeks)
1. Wire Polygon/Finnhub providers behind the same adapter interface.
2. Train and ship LSTM model for top-50 symbols.
3. News pipeline live with finBERT.
4. Auth.js + 2FA fully wired.
5. Strategy lab GA with vectorbt backtest + MC.
6. Stripe billing.

### Mid-term
7. Options flow analytics (UnusualWhales-style — requires options data feed).
8. Smart-money / 13F filings tracker.
9. Reinforcement learning agent (PPO over a custom Gym env) for tactical allocation.
10. Multi-asset portfolios (FX, crypto, bonds proxies).
11. Mobile app (Expo / React Native).

### Long-term
12. Live brokerage integration (read-only first via Plaid Investments; later trade execution via Alpaca with explicit user consent and a hard-gated approval flow).
13. Social: shareable forecasts, leaderboards, copy-trading (paper only).
14. LLM fine-tuned on the platform's own (anonymized) explanations to lower cost per chat.
15. On-device inference for premium offline mode.
16. Multi-language UI (i18n) — EN, PT-PT, ES, DE, FR.

---

## Appendix A — Disclaimers (must appear in product)

> This platform provides informational analyses and AI-generated estimates. It is **not** investment advice, a solicitation, or a recommendation to buy or sell any security. Forecasts are probabilistic and frequently wrong. Past performance does not guarantee future results. You are solely responsible for your investment decisions. The operators of this platform are not registered investment advisors.

## Appendix B — Glossary
- **CAGR** — Compound Annual Growth Rate
- **MDD** — Maximum Drawdown
- **Sharpe** — (Return − RiskFreeRate) / StdDev
- **Sortino** — Sharpe but using downside deviation
- **VaR** — Value at Risk
- **HMM** — Hidden Markov Model
- **TFT** — Temporal Fusion Transformer
- **NHiTS** — Neural Hierarchical Interpolation for Time Series
- **CRPS** — Continuous Ranked Probability Score
- **HHI** — Herfindahl–Hirschman Index (concentration)
