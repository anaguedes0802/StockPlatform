# Forecaster backtest — honest results

> **TL;DR of model evolution measured here**
> - **v0** — XGBoost on price-only features
> - **v1** — XGBoost trained with **VIX / 10Y yield / DXY** macro features joined into the feature matrix
> - **v2** — Ensemble: v1 XGBoost + Nixtla `statsforecast` **AutoARIMA**, regime-weighted blend
> - **v5** — Dense walk-forward (~950 OOS predictions/run instead of 16) exposed that v2–v4 lost to a
>   "past returns × today's volatility" baseline on every series. Fixed an ARIMA interval bug, in-sample
>   conformal, non-stationary features, overfit trees and a train/serve skew; XGBoost is now a gated
>   overlay on a volatility-scaled drift core. **See [§ v5](#v5--measure-properly-then-stop-fitting-noise) — it supersedes the tables below.**
>
> - **v6** — Cross-sectional stock selection on S&P 500+400 with new free data (Alpaca news archive, 25y of
>   earnings surprises). Earnings surprise + momentum, top 10%, monthly: 23.2% CAGR vs 16.9% equal-weight /
>   15.2% SPY over 2017–26 after costs (t=1.8; survivorship-biased). News, ML rankers, sector rotation and
>   trend filters did not help. **See [§ v6](#v6--stock-selection-rank-stocks-against-each-other).**
>
> - **v7** — 20-year test (2006–2026), medians over all 21 rebalance days. Stock selection ≈17%/yr (DD −52%);
>   with a 10-month SPY trend gate ≈14%/yr, Sharpe 0.86, DD −27% (SPY: 11.4%, DD −50%). Value/quality and
>   multi-asset blends did not improve it. **See [§ v7](#v7--20-years-risk-management-and-calendar-luck-checks).**
>
> - **v8** — Copying Nancy Pelosi from official filings: +4.4% vs SPY per purchase at 12 months after the disclosure
>   lag, but a plain "5 biggest stocks" rule matched it. Real House disclosures now power the app's
>   Congress tracker. **See [§ v8](#v8--copying-congress-nancy-pelosi--co).**
>
> - **v9** — The strategy now runs forward on the Alpaca paper account (`/live-test`). SUE, revenue surprise and
>   price-scaled surprise were tested and rejected (none beat the current measure in both design periods).
>   **See [§ v9](#v9--forward-test-on-paper-and-better-earnings-surprise-measures-rejected).**
>
> See [§ Lift measurement (v0 → v2)](#lift-measurement-v0--v2) for the head-to-head numbers on the same walk-forward folds.

These are **real numbers** from walk-forward validation of the XGBoost forecaster against historical data via yfinance.
Run yourself with:

```bash
python scripts/backtest_forecaster.py AAPL --horizon 5 --range 5y --step 60
```

## Methodology

- **Walk-forward**: starting at bar #300, every 60 days we re-train from scratch on all data up to that bar and predict the realized h-day-ahead log return.
- **Horizons tested**: 5d (short-swing) and 21d (one-month tactical).
- **Window**: 2022-08-02 → 2026-03-05 (~5 years, 16 folds per run).
- **Targets**: directional accuracy, MAE on log-returns, p10-p90 interval coverage, long-only signal Sharpe vs buy-and-hold.
- **Honest baselines**:
  - *Naive trailing*: predict next h-day return = last realized h-day return.
  - *Majority*: always predict the majority class (up if up-rate ≥ 0.5).
  - *Zero return*: predict 0 (a perfectly calibrated random walk).

## Results

### AAPL — 5-day horizon, 16 folds

| Metric | Model | Naive trailing | Majority | Zero-ret |
|---|---:|---:|---:|---:|
| Directional accuracy | **0.438** | 0.625 | 0.625 | — |
| MAE (log-return) | **0.0211** | 0.0320 | — | 0.0212 |
| p10–p90 coverage (target 0.80) | **0.812** ✅ | — | — | — |
| Long-only signal CAGR | **+25.7%** | — | — | — |
| Buy & hold CAGR | **+73.2%** | — | — | — |
| Long-only Sharpe | 1.90 | — | — | 3.26 |

### AAPL — 21-day horizon, 16 folds

| Metric | Model | Naive trailing | Majority | Zero-ret |
|---|---:|---:|---:|---:|
| Directional accuracy | **0.562** | 0.500 | 0.562 | — |
| MAE (log-return) | **0.0688** | 0.0704 | — | 0.0489 |
| p10–p90 coverage | **0.438** ❌ | — | — | — |
| Long-only signal CAGR | **+2.9%** | — | — | — |
| Buy & hold CAGR | **−1.5%** | — | — | — |

### SPY — 5-day horizon, 16 folds

| Metric | Model | Naive trailing | Majority | Zero-ret |
|---|---:|---:|---:|---:|
| Directional accuracy | **0.500** | 0.500 | 0.688 | — |
| MAE (log-return) | **0.0145** | 0.0209 | — | 0.0101 |
| p10–p90 coverage | **0.875** ✅ | — | — | — |
| Long-only signal CAGR | **+5.3%** | — | — | — |
| Buy & hold CAGR | **+11.5%** | — | — | — |

## What this means (read carefully)

1. **The 5-day directional model is not better than a coin flip.** On AAPL it's *worse* than chance (43.8%); on SPY it's exactly 50%. A constant "always up" baseline beats it on both symbols because both markets trended up over the window.

2. **The point estimate barely beats predicting zero.** AAPL 5d MAE: 0.0211 model vs 0.0212 zero-return. The model has not extracted meaningful signal beyond "the next 5-day return is approximately zero, plus noise."

3. **Interval calibration is the bright spot.** The p10-p90 uncertainty bands cover the realized outcome ~80% of the time on 5d horizons, which is the target. The model *knows what it doesn't know.* At 21d the bands collapse to 44% coverage — the model is overconfident on longer horizons.

4. **The naive trailing baseline is a strong competitor for direction.** Don't believe the marketing — a 1-line baseline gets 62.5% on AAPL 5d in this window.

5. **21-day forecasts modestly outperform** buy-and-hold during the chop in the AAPL window (+2.9% vs −1.5%), but the sample is tiny (16 folds) and the absolute returns are noise.

## Why the model is this weak (today)

The v0 forecaster uses **only price-derived features**. All of these from `apps/api/app/ml/features.py` are wired into the inference path but **always zero in training**:

- `news_sent_7d`, `news_count_z_30d` — news sentiment
- `social_hype` — social media buzz
- `fund_pe`, `fund_growth` — fundamentals
- `macro_yield_10y`, `macro_vix` — macro context
- `evt_earnings_window` — event flags

The news/sentiment service exists at `/news/*` endpoints, but **its outputs are not joined to the training dataframe**. The macro ingest job and the fundamentals join are not implemented. So at training time XGBoost sees a feature matrix that is effectively *only* technical indicators of price and volume. That's not enough to predict a 5-day return.

## What to do about it (in order of expected lift)

| Improvement | Expected lift on 5d directional acc. | Effort |
|---|---|---|
| Join `news_sent_7d` from the news service into training | +3-5 pp | Small |
| Replace lexicon sentiment with **finBERT** | +1-3 pp | Medium |
| Add **fundamental** features at training time (PE, EPS growth, etc.) | +2-4 pp on longer horizons | Medium |
| Add **macro** features (10Y yield, VIX, DXY) from FRED | +1-3 pp | Medium |
| Add **earnings/event window** flag | +1-2 pp around earnings | Small |
| Per-symbol-cluster hyperparameter tuning | +1-2 pp | Medium |
| **Implement walk-forward in training** (purged k-fold) and add CRPS to model selection | better calibration on long horizons | Large |
| Wire LSTM to actually contribute (with sentiment+macro) | uncertain | Large |
| Train on a 10y window with regime tags as input | +2-4 pp | Medium |
| Replace XGBoost with TFT (Nixtla NeuralForecast) | uncertain — needs benchmarking | Large |

The point of shipping a weak-but-honest model with proper backtesting infrastructure is so each of those improvements can be A/B'd against this exact baseline.

## Retraining cadence

**Today**: lazy — models train on first request per (symbol, horizon) and live forever in `apps/api/artifacts/models/`. There is no automatic retraining.

**Recommended (after wiring news/macro)**:

```bash
# Daily, 22:00 UTC (after US close), heavy symbols + short horizons
0 22 * * 1-5  cd /app && .venv/bin/python scripts/retrain_all.py \
    --symbols AAPL MSFT NVDA GOOGL AMZN META TSLA AMD SPY QQQ \
    --horizons 1 5 21

# Weekly, Saturday 06:00 UTC, full universe + all horizons
0 6 * * 6     cd /app && .venv/bin/python scripts/retrain_all.py \
    --horizons 1 5 21 63
```

In production: replace cron with Celery Beat or Prefect, push the artifacts to S3/MinIO, and register each run in MLflow (already specified in `ARCHITECTURE.md` §6). Add a model-drift check that triggers an out-of-band retrain when 30d rolling MAPE drifts > 1.5× the baseline.

## Freshness in the API

Every `/ai/forecast` response now includes a `freshness` field per horizon:

```json
"freshness": {
  "xgboost": {
    "trained_at": "2026-05-26T18:14:22+00:00",
    "n_train_rows": 720,
    "last_train_bar_ts": "2026-05-24T00:00:00+00:00"
  }
}
```

Use it client-side to warn when a model is more than N days stale.

---

## Lift measurement (v0 → v2)

Same walk-forward setup (16 folds, 2022-08 → 2026-03, 60-day step). Direct head-to-head:

### MAE on log-returns (lower = better)

| Symbol/Horizon | v0 (price-only) | v1 (+ macro) | v2 (ensemble) | Zero baseline | Lift vs v0 |
|---|---:|---:|---:|---:|---:|
| **AAPL 5d** | 0.0211 | 0.0201 | **0.0197** | 0.0212 | **−7%** |
| **AAPL 21d** | 0.0688 | — | **0.0581** | 0.0489 | **−16%** |
| **SPY 5d** | 0.0145 | — | **0.0123** | 0.0101 | **−15%** |
| **NVDA 5d** | — | — | 0.0373 | 0.0418 | beats zero-baseline ✅ |

Point-estimate MAE improves consistently. The improvement is small in absolute terms because **5d returns themselves are small** (≈2% absolute on AAPL) — a 7% MAE reduction on top of that is what's economically meaningful.

### Directional accuracy

| Symbol/Horizon | v0 | v2 | Best naive baseline |
|---|---:|---:|---:|
| AAPL 5d | 43.8% | 50.0% | 62.5% (majority) |
| AAPL 21d | 56.2% | 37.5% ❌ | 56.2% (majority) |
| SPY 5d | 50.0% | 50.0% | 68.8% (majority) |
| NVDA 5d | — | 31.2% ❌ | 50.0% (majority) |

**Be honest: directional accuracy did not improve and is still consistently below the majority baseline.** During this 2022→2026 window markets trended strongly up — "always long" beats almost any model. The model is correctly predicting *roughly zero return* most of the time, which is statistically defensible but useless as a "buy/sell" signal source.

This is *not* a bug in our build — it's the well-known difficulty of short-horizon equity direction prediction. The honest take: until news sentiment, fundamentals, and event flags are joined into training (not just inference), the model will not beat majority on direction.

### Interval calibration (p10–p90 coverage — target ≈ 80%)

| Symbol/Horizon | v0 | v2 |
|---|---:|---:|
| AAPL 5d | 81.2% ✅ | 100.0% (slightly wide) |
| **AAPL 21d** | **43.8% ❌** | **100.0% ✅** |
| SPY 5d | 87.5% ✅ | 100.0% (slightly wide) |
| NVDA 5d | — | 100.0% |

**This is the biggest real improvement.** The v0 21d intervals were dangerously narrow (44% coverage — false confidence). The v2 ensemble's intervals are now properly humble. They're slightly *too* wide on 5d, but over-coverage is the correct failure mode for a financial forecast — pretend less certainty than you have, not more.

### What changed and why each piece matters

1. **Macro features (VIX, 10Y yield, DXY) joined into training.**
   - VIX provides a regime signal: high VIX → mean reversion dynamics dominate; low VIX → trend-following works.
   - 10Y yield (^TNX) is a direct discount-rate input; growth-stock returns are sensitive to its level and direction.
   - DXY (USD index) drives the cross-asset story for multinationals and commodities.
   - Implementation: [apps/api/app/services/macro.py](apps/api/app/services/macro.py). Per-series features: level, 5d change, 60d z-score. Joined by date in `build_features()`.

2. **AutoARIMA ensemble member** (Nixtla `statsforecast`, MIT).
   - Classical models are still hard to beat on noisy financial series at short horizons.
   - AutoARIMA gives a different functional form than gradient-boosted trees → ensemble diversity reduces variance.
   - 35% weight in sideways regimes, 25-30% in trending regimes. See `REGIME_MODEL_MIX` in [apps/api/app/ml/ensemble.py](apps/api/app/ml/ensemble.py).

3. **Renormalized weight-blending** ([apps/api/app/ml/ensemble.py](apps/api/app/ml/ensemble.py)).
   - If torch isn't installed → LSTM drops out and remaining weights renormalize.
   - If a single fold's AutoARIMA fit fails → XGBoost-only for that fold, no crash.
   - Production-safe: any model can fail without taking down the forecast.

### What this means for the recommendation engine

The recommendation engine consumes the 30d ensemble forecast and blends it with technicals + fundamentals + sentiment. With v2's improved point estimate *and* properly-sized uncertainty bands, the **confidence** field on recommendations is now meaningful — high confidence means the ensemble agreed on a narrow band, low confidence means the bands are wide and conviction should be low. The label thresholds in `services/recommendation.py` already use confidence, so this lift propagates without code changes.

### Reproduce these numbers

```bash
# v0 baseline (price-only XGBoost — requires reverting features.py to pre-macro)
git checkout <pre-v1-commit>  apps/api/app/ml/features.py
python scripts/backtest_forecaster.py AAPL --horizon 5 --range 5y --step 60 --mode xgboost

# v1 (XGBoost + macro)
python scripts/backtest_forecaster.py AAPL --horizon 5 --range 5y --step 60 --mode xgboost

# v2 (full ensemble)
python scripts/backtest_forecaster.py AAPL --horizon 5 --range 5y --step 60 --mode ensemble
```

---

## Smart Money Concepts (v3)

After v2, we added a **rules-based price-action engine** ([apps/api/app/services/price_action.py](apps/api/app/services/price_action.py)) that runs alongside the statistical models:

| Concept | What it detects | Used by |
|---|---|---|
| **Swing points** | N-bar confirmed local highs/lows | All downstream detectors |
| **BOS** (Break of Structure) | Higher-high or lower-low broken in an established trend | Trend continuation signal |
| **CHOCH** (Change of Character) | First HH after LLs (or vice versa) | Trend reversal signal |
| **FVG** (Fair Value Gap) | 3-bar imbalance, strength = body / ATR; tracked until filled | Magnet zone + entry trigger |
| **Order Block** | Last opposite-color candle before a strong impulse | Demand / supply zone |
| **Liquidity Sweep** | Wick takes prior swing then reverses within N bars | Stop-hunt / reversal cue |
| **Demand/Supply zones** | Clustered swings + OBs + unfilled FVGs | Composite zone with strength score |
| **Confluence score** | Weighted blend, ∈ [−1, +1] | Recommendation engine + smart-money screener |

The confluence score is a **separate signal**, not a forecast. It captures *structural setups* (institutions defending a level, post-CHOCH continuation, FVG fill expectations) that pure statistical models miss. It feeds into [services/recommendation.py](apps/api/app/services/recommendation.py) with a regime-dependent weight (15-25%; highest in choppy/high-vol regimes where structure dominates), and powers the **smart_money** strategy in the screener.

Why this matters in practice:
- The forecast says *what* the next-30d return is statistically likely to be.
- The price-action engine says *where institutions parked orders* and what setup the price is currently in.
- They're orthogonal — both contribute information.

### Endpoint

```bash
curl 'http://localhost:8000/stocks/AAPL/price-action?range=1y' | jq '.confluence'
```

Returns the full structure list (swings, FVGs, OBs, BOS/CHOCH events, liquidity sweeps, zones) plus the confluence score for the latest bar. The chart on the stock page overlays unfilled FVGs / unmitigated OBs as dashed-line bands and BOS/CHOCH as arrow markers — toggle with the `SMC` button in the chart header.

---

## Live HTTP testing (May 26)

Booted the server (`uvicorn app.main:app --host 127.0.0.1 --port 8765`) and hit every public endpoint with curl. Real data returned, real timings:

| Endpoint | Result |
|---|---|
| `GET /healthz` | `{"ok":true}` |
| `GET /stocks/search?q=appl` | AAPL, AMAT returned (with rank scoring) |
| `GET /stocks/AAPL` | live yfinance quote: $308.82, PE 37.4, beta 1.065 |
| `GET /stocks/AAPL/history?range=1mo` | 22 daily bars |
| `GET /stocks/AAPL/indicators?names=rsi_14,macd,sma_50` | RSI 79, MACD positive |
| `GET /stocks/AAPL/news?limit=3` | live Motley Fool / Barron's / WSJ articles |
| `GET /stocks/AAPL/price-action` | trend up, 40 swings, 40 FVGs, 30 OBs, 10 BOS/CHOCH events, confluence +0.28 bullish |
| `GET /stocks/AAPL/insider` | yfinance Form 4 transactions returned |
| `GET /stocks/AAPL/institutional` | 10 institutional holders (BlackRock 7.8%, Vanguard 6.5%, State Street 4.1%), famous = Warren Buffett |
| `GET /stocks/NVDA/politicians` | 2 trades: Pelosi $1M-$5M, McCaul $50K-$100K |
| `GET /famous-investors` | 10 funds with curated top holdings |
| `GET /politicians/recent?limit=5` | 5 recent disclosures (Gottheimer, MTG, Crenshaw, Tuberville…) |
| `POST /ai/forecast {"symbol":"AAPL","horizons":["5d","30d"]}` | **15.8s** first call (trains all models cold), point=$320.59 / p10=$291 / p90=$326 at 5d |
| `POST /ai/recommend {"symbol":"AAPL"}` | HOLD, score −0.02, includes price-action drivers |
| `GET /ai/opinion/AAPL` | **2.7s** (models cached) — full structured opinion with entry/stop/target |
| `GET /screener/run/rising_stars` | **22.8s** parallel scoring across 130+ stocks; top picks: CVS, ROKU, EOG, QCOM, OXY |
| `GET /screener/run/smart_money` | **20.1s** — top picks: MU, ASML.AS, C, PFE, TSM (all at high-confluence demand zones) |

**Bugs surfaced and fixed during live testing:**

1. **Rate limiter crashed when Redis is down** — replaced with fail-open + comment about per-process bucket fallback for prod ([apps/api/app/core/rate_limit.py](apps/api/app/core/rate_limit.py)).
2. **Smart-money screener rationale showed 30+ component names** — the zone clusterer was dumping the entire raw component list. Now summarizes as `"4×OB, 3×swing_low, 2×FVG"` via `Counter.most_common(3)` ([apps/api/app/services/price_action.py](apps/api/app/services/price_action.py)).
3. **Opinion thesis said "bullish (hold)"** — the language was decoupled from the verdict thresholds. Rewrote `_thesis_one_liner` to use verdict-aligned phrasing ([apps/api/app/services/opinion.py](apps/api/app/services/opinion.py)).

All three would have hit users in production — this is exactly why live HTTP testing matters in addition to unit tests.

---

## Point-in-time archive (v3 attempt)

Built two new services to address the "no PIT data" caveat from v2:

- **[services/historical_news.py](apps/api/app/services/historical_news.py)** — Alpha Vantage NEWS_SENTIMENT adapter with 30-day chunked pagination, per-article relevance-weighted sentiment scoring, daily aggregation, 30-day rolling z-score on article count. Falls back to zeros (no crash) when `ALPHAVANTAGE_API_KEY` is unset.
- **[services/fundamentals_pit.py](apps/api/app/services/fundamentals_pit.py)** — yfinance quarterly_financials + balance_sheet, with a conservative +45-day filing lag (`available_at = period_end + 45d`) so each bar only sees data that was public on that date. Computes TTM EPS, YoY revenue/EPS growth, net margin, debt/assets.

`build_features(symbol=..., pit_history=True)` joins both via `pandas.merge_asof(direction="backward")` — proper temporal-asof to guarantee no look-ahead. XGBoost trainer sets `_training_symbol` to opt into PIT-history mode.

### v3 result (AAPL 5d, same 16 folds)

| Metric | v0 (price only) | v1 (+macro) | v2 (+ensemble) | **v3 (+PIT)** |
|---|---:|---:|---:|---:|
| Directional accuracy | 43.8% | 50.0% | 50.0% | **37.5%** ❌ |
| MAE (log-return) | 0.0211 | 0.0201 | 0.0197 | **0.0212** ≈ |
| p10-p90 coverage | 81.2% | 75.0% | 100% | **68.8%** |

**Result: PIT features did not help; slightly hurt on this window.**

### Why — and being honest about it

yfinance exposes only **~5 recent quarters** of financials (Q1 2025 through Q1 2026 for AAPL). For our 5-year backtest window (2022-08 → 2026-03), that means:

- 80%+ of training rows see `fund_eps_growth_yoy = 0` and other PIT columns at zero
- Only the most recent ~6 months of bars get real fundamental values
- News sentiment is also zero (no Alpha Vantage key in the test environment)

So we added **mostly-constant features that briefly change for a few recent bars** — a near-textbook overfitting hazard. XGBoost dutifully tried to fit them, with predictable noise as the result.

**This is a data sourcing problem, not a code/architecture problem.** The PIT plumbing — `+45d filing lag`, `merge_asof(direction='backward')` — is mathematically correct. To get real lift, you need:

| Source | Coverage | Cost |
|---|---|---|
| SEC EDGAR 10-Q/10-K archive | full history since 2001 | free (just need user-agent + parser) |
| SimFin Pro | 5+ year clean PIT bundle | $50-100/mo |
| Sharadar SF1 (Quandl) | 25+ year PIT bundle | discontinued for retail; data still circulating |
| Alpha Vantage NEWS_SENTIMENT | ~2y archive, per-article | free with key (25 req/day on free tier) |
| GDELT | global news events since 2015 | free |

Recommended next step: a `scripts/refresh_fundamentals.py` that scrapes SEC EDGAR 10-Q XBRL filings into a local table, indexed by `(symbol, period_end, filing_date)`. That gives 5-10× more training rows with real values, and the v3 features will then earn their keep.

The right takeaway from this iteration: **measure honestly, don't claim a lift when there isn't one.** v2 (regime ensemble + macro) remains the current production baseline; v3 plumbing is in place ready for richer data.

---

## v4 — SEC EDGAR full-history fundamentals + finBERT sentiment

The v3 bottleneck was data sparsity (yfinance only ships ~5 quarters). v4 fixes that with two real, free, no-key changes:

- **SEC EDGAR XBRL companyfacts** ([app/services/edgar.py](apps/api/app/services/edgar.py))
  Real filing dates (no +45-day heuristic), full XBRL history back to ~2008.
  AAPL/NVDA/TSLA each return **65–72 quarters** with revenue/EPS/assets/debt and real `filed` dates.
- **finBERT** ([app/services/sentiment.py](apps/api/app/services/sentiment.py))
  ProsusAI/finbert via HuggingFace, lazy-loaded. Replaces the lexicon scorer everywhere
  `sentiment.score_text()` is called — including the news service. ~14s cold load,
  ~50–100ms per batch after. Falls back to lexicon if transformers isn't installed.

`fundamentals_pit.py` now prefers EDGAR over yfinance automatically.

### Direction accuracy (16 folds, 2022-08 → 2026-03)

| | v0 | v2 | **v4** | Naive | Majority |
|---|---:|---:|---:|---:|---:|
| AAPL 5d  | 43.8% | 50.0% | **43.8%** | 62.5% | 62.5% |
| AAPL 21d | 56.2% | 37.5% | **43.8%** | 50.0% | 56.2% |
| NVDA 5d  | —     | 31.2% | **50.0%** | 43.8% | 50.0% |
| NVDA 21d | —     | —     | **50.0%** | 43.8% | 68.8% |

### MAE (log-returns)

| | v2 | **v4** | Zero baseline |
|---|---:|---:|---:|
| AAPL 5d  | 0.0197 | 0.0233 | 0.0212 |
| AAPL 21d | 0.0581 | 0.0668 | 0.0489 |
| NVDA 5d  | 0.0373 | 0.0381 | 0.0418 |
| NVDA 21d | —      | 0.1332 | 0.1245 |

### Honest conclusion

**EDGAR fundamentals did not move the needle on short-horizon forecasting**, despite providing real, complete, point-in-time data with proper filing dates. This is consistent with academic findings:

- Quarterly fundamentals change ~4× per year; at daily resolution they are stair-step features that look nearly identical to the model for 60-day chunks. There's almost nothing for tree splits to learn.
- The 2022–2026 window was unusually fundamentals-disconnected — markets traded on AI narrative *ahead* of when revenue arrived. NVDA's price ran far ahead of its fundamental story. A model that trusted trailing fundamentals would lag the rotation.
- The **one real lift** was NVDA 5d: 31% → 50% directional (i.e., from worse-than-chance to chance-level). That's the case where v2 was visibly broken and v4 fixed it.

**What would actually lift directional accuracy further** (in expected impact order):

| Signal | Why it works | Status |
|---|---|---|
| **Earnings surprises** (Δ vs consensus, not level) | Markets react to surprises, not absolute numbers. | Needs consensus data (Zacks/Refinitiv) — not free. |
| **Analyst revisions delta** | Consensus *change* is one of the most reliable real signals. | Same source constraint. |
| **News flow at the bar resolution** (Alpha Vantage NEWS_SENTIMENT for the training window) | Real per-day sentiment captures real catalyst flow. | Plumbing ready, needs `ALPHAVANTAGE_API_KEY`. |
| Insider transactions density as a continuous feature | Cluster of Form 4 buys ahead of earnings is well-documented edge. | Service exists; not yet joined into training. |
| Per-symbol hyperparameter tuning (Optuna) | Heterogeneous symbols benefit from different hyperparams. | Code-only, no data — easy next. |
| Finer-grained price action features (e.g., FVG strength, unmitigated OB density) as model features | Adds the SMC signal directly to the model, not just the rec engine. | Code-only. |

The right thing to do with **what we now have** is:

1. Surface the v4 fundamentals in the **opinion synthesizer**, where they make the bull/bear case far more concrete (e.g., "NVDA revenue +85% YoY, EPS +214% YoY through 2026-Q1"). They're now ready for that purpose — the data is real and dated correctly.
2. Use finBERT for **live news scoring on the platform** — that's an immediate quality upgrade for the news feed + recommendation engine, no backtest needed to validate.
3. Hold v2 (regime ensemble + macro) as the **forecast baseline** until earnings-surprise or per-bar news signals are wired.

This is also what an honest quant report looks like: most ideas fail, one or two of the smaller ones move the needle on edge cases. Document it, keep what's measured-good, hold the rest behind feature flags.

---

## v5 — measure properly, then stop fitting noise

### 1. The old harness couldn't tell any two versions apart

Every table above scores **one prediction per fold**, so 16 predictions per run. With n = 16, one prediction flipping moves directional accuracy by 6.25 pp, and the standard error is ±12 pp. None of the v0 → v4 deltas (43.8% vs 50.0% vs 37.5%) are distinguishable from noise. The "majority" baseline was also computed on the *test* period, which is hindsight.

`scripts/backtest_forecaster.py` now defaults to **dense** walk-forward. It uses the same anchors and the same retrain-every-60-bars cadence, but it predicts **every** out-of-sample bar: ~950 predictions per run (n_eff ≈ 950 / h after accounting for overlapping labels). All baselines are computed from training data only. It also reports proper scoring rules:

- **Pinball loss** vs *climatology*: the empirical q10/q50/q90 of past h-day returns. This is the bar a quantile forecast must clear.
- **Brier score** vs the training up-rate.
- **Information coefficient**: the Spearman correlation between the forecast and the realized return.

The old mode is still available with `--sparse`.

### 2. What the dense harness showed about v2–v4 (production ensemble)

Measured over 2022-12 → 2026-09, 4 symbols × 2 horizons:

- **Worse than climatology on every series.** Pinball skill ranged from −9% to −56% and Brier skill from −11% to −47%. MAE was worse than predicting a 0% return everywhere.
- **The XGBoost median carried no information.** Once price-level features are removed, IC ≈ 0 (often negative).
- **Directional accuracy was below "always up" on all 8 series.** The classifier called "up" only ~43% of the time in markets that rose 57–72% of the time.

### 3. Root causes (all fixed)

| Defect | Effect | Fix |
|---|---|---|
| **ARIMA summed per-step 80% bounds** across the horizon (band ∝ h instead of √h) | ARIMA bands were 2.2× too wide at 5d and 4.6× at 21d. The "100% coverage ✅" in v2 was this bug, not humility | Add variances (`arima_model._cumulative`) |
| **Conformal calibration used in-sample residuals** (the heads were trained on the calibration rows), and it rebuilt features *without* PIT history, which is a different feature set from training | Almost no widening, so XGBoost bands covered 48–67% | Replaced by `ml/calibration.py` (below) |
| **Price-level features** (`sma_*`, `atr_14`, `macd`, `drawdown` from first bar, `*_level` macro, `fund_eps_ttm`) | Trees used them as a proxy for calendar time and cannot extrapolate once price leaves the training range | Excluded via `NON_STATIONARY_FEATURES`. Added scale-free `macd_pct`, `macd_hist_pct` and `drawdown_252` |
| **Overfit trees**: depth 5 × 300 on ~1k rows whose effective size is ~rows/h | Confident noise, with a Brier score ~45% worse than the base rate | Depth 3, 200 trees, `min_child_weight=30`, λ=5, row/col subsampling |
| **Train/serve skew**: `predict()` built features without `pit_history` | SMC / insider / PIT-fundamental columns were **all zero at inference** | `predict()` now uses the same PIT join as `fit()` |
| **Cached models lived forever** | Old bundles would keep serving after any fix | `MODEL_VERSION`. `load()` rejects stale bundles, so the next request retrains |
| Insider (Form 4) feature re-downloaded ~200 SEC filings on every feature build when Redis is down | ~65 s per fit | Process-local cache fallback in `edgar_form4` / `fundamentals_pit` |

### 4. New model design (`ml/calibration.py`)

```
core    = volatility-scaled drift   (always on)
          q50 = median historical h-day log-return
          q_τ = q50 + z_τ · σ_t      σ_t = blend(vol_21, vol_63) · √(h/252)
                                     z_τ = empirical quantiles of standardized training residuals
overlay = XGBoost median / direction heads, weights λ, κ ∈ [0, 1]
```

λ and κ are chosen on a **purged** held-out tail: the last 25% of training rows, with an h-bar gap so no training label overlaps the tail. The overlay only gets weight if it beats the core with **t ≥ 2 over non-overlapping h-bar blocks**. An earlier "≥1% better" gate let noise through at 21d, where a 250-row tail holds only ~12 independent observations.

With today's features, the overlay earns **0 weight**. So the forecast *is* the heteroscedastic drift model, and the API says so: when λ = 0 the drivers read "historical median return / current volatility" instead of SHAP values for a model that isn't driving the number. The gate is how future features prove themselves. Once per-bar news (needs `ALPHAVANTAGE_API_KEY`) or earnings-surprise data land, the overlay picks up weight automatically if it earns it, and not before.

The empirical z-quantiles capture real skew. AAPL z₁₀ = −1.39 vs z₉₀ = +1.15, so the downside tail is wider than the upside, which a Gaussian band can't express. p25/p75 are now real quantiles instead of "half the p10–p90 spread".

### 5. Results — production ensemble, dense walk-forward, before → after

~950 out-of-sample predictions per row, 2022-12-09 → 2026-09-22. Bold = after. Parentheses in the last two columns are the baselines.

| Series | Pinball skill vs climatology | p10–p90 coverage (width) | Brier skill vs base rate | MAE (zero-return MAE) | Dir. acc (always-up) |
|---|---:|---:|---:|---:|---:|
| AAPL 5d | -8.7% → **+0.7%** | 92% (0.133) → **80% (0.090)** | -20% → **-0.4%** | 0.0304 → **0.0289** (0.0290) | 50.5% → **55.7%** (57.3%) |
| AAPL 21d | -46.1% → **-2.7%** | 100% (0.449) → **79% (0.190)** | -26% → **-3.5%** | 0.0646 → **0.0591** (0.0580) | 47.7% → **53.2%** (63.7%) |
| MSFT 5d | -9.4% → **+1.1%** | 91% (0.122) → **82% (0.089)** | -22% → **-0.3%** | 0.0295 → **0.0270** (0.0271) | 47.3% → **57.2%** (57.2%) |
| MSFT 21d | -31.7% → **-0.8%** | 97% (0.433) → **76% (0.179)** | -25% → **-1.4%** | 0.0647 → **0.0597** (0.0578) | 46.7% → **49.5%** (58.7%) |
| NVDA 5d | -9.2% → **+2.9%** | 95% (0.249) → **83% (0.172)** | -18% → **+0.6%** | 0.0514 → **0.0491** (0.0503) | 49.5% → **59.7%** (59.9%) |
| NVDA 21d | -42.6% → **+4.2%** | 100% (0.883) → **88% (0.394)** | -11% → **+2.9%** | 0.1101 → **0.0995** (0.1031) | 54.4% → **62.6%** (68.3%) |
| SPY 5d | -15.8% → **+3.4%** | 94% (0.079) → **85% (0.055)** | -15% → **-0.4%** | 0.0170 → **0.0146** (0.0151) | 53.8% → **61.7%** (61.7%) |
| SPY 21d | -56.3% → **+3.7%** | 99% (0.273) → **87% (0.111)** | -47% → **-0.1%** | 0.0379 → **0.0298** (0.0321) | 43.9% → **65.2%** (71.9%) |

XGBoost-only mode (no ARIMA) went from 48–67% coverage to 72–83%, and its pinball skill went from −12…−38% to −6…+5%. With the variance bug fixed, the ARIMA member now mostly *helps*. The ensemble beats XGBoost-only on pinball for AAPL 5d/21d, MSFT 21d and NVDA 5d/21d, and is within 1.3 pp on the rest, so it stays the production path.

### 6. How to read this honestly

- **This is a calibration win, not an alpha win.** The forecast now matches or modestly beats "past returns, scaled to today's volatility" instead of losing to it by up to 56%. Its bands mean what they say (76–88% for a nominal 80%, where before they were either far too narrow or 2–4× too wide). It no longer issues confident directional calls it can't back up.
- **Direction ≈ "always up", by design.** P(up) is the historical up-rate unless the overlay earns weight. It trails "always up" only where the expanding window still remembers 2022: AAPL/MSFT 21d early folds had a < 50% training up-rate, so the model said "down" into the 2023 rally.
- **IC is not meaningful for the core.** The median is a per-fold constant, so IC just measures whether folds with higher trailing drift had higher forward returns (they didn't: drift mean-reverts). It becomes the key metric again once the overlay turns on.
- **Remaining negatives**: AAPL 21d (−2.7%) and MSFT 21d (−0.8%) pinball, both driven by that 2022 drift anchor. A shorter or regime-aware drift window is the obvious next experiment, but it's a knob that is easy to overfit on 4 years of data, so it's left alone.

### Reproduce

```bash
# dense (default) — ~2–4 min per run after the first SEC/EDGAR fetch
python scripts/backtest_forecaster.py AAPL --horizon 5 --mode ensemble
python scripts/backtest_forecaster.py SPY --horizon 21 --mode xgboost --out spy21.json
# legacy one-prediction-per-fold numbers
python scripts/backtest_forecaster.py AAPL --horizon 5 --mode ensemble --sparse
```

---

## v6 — stock selection: rank stocks against each other

v5 showed that forecasting a single stock's 5–21-day return from its own history is noise. v6 asks a different question: *which* stocks will do better than the others next month? Every date then gives ~700 labelled comparisons instead of one, and a few effects are documented in the academic literature.

**Protocol.** Decide at the close, buy at the next open, hold 21 sessions, and pay 10 bps per side on turnover. Every choice (factor, sign, weight, portfolio size) was made on the **2017–2022 design period**. The **2023–2026 holdout** was never used for a decision. Code: [app/ml/stock_selection.py](apps/api/app/ml/stock_selection.py). Reproduce with `python scripts/backtest_stock_selection.py`.

### New data sources (all free)

| Source | What | Coverage |
|---|---|---|
| Alpaca / Benzinga news ([alpaca_news.py](apps/api/app/services/alpaca_news.py)) | Per-ticker timestamped headlines, archived locally and scored with finBERT. Features use only articles published before each bar's close | 2016 → today, ~410k articles for the app's universe |
| yfinance earnings history (`earnings.earnings_history`, needs `lxml`) | Announcement time, consensus EPS, reported EPS, surprise % | ~25 years per stock |
| Wikipedia S&P 500 / 400 lists | A broad, non-curated universe | ~900 stocks |

`lxml` was missing from the venv. yfinance needs it for *every* earnings-date call, so the app's existing `pead_signal` / `earnings_dates` were silently returning nothing. It's now a declared dependency.

### The most important finding: the app's own stock list is hindsight-biased

On the bundled list in `universe_data.py` (172 US stocks), momentum looked excellent: +10%/yr over equal weight in 2017–22. On the same list restricted to its S&P 500 members, it was +2.5%/yr. On the S&P 500+400 it was +1.2%/yr. The difference is ~50 names such as MSTR, COIN, PLTR, SMCI and HOOD, which were added *because* they had already soared. A momentum strategy on that list mostly rediscovers the curation. **Never judge a strategy on the bundled list.** The backtest script defaults to S&P 500+400.

The broad universe still has survivorship bias: it uses *today's* index members, so companies that went bust or were dropped are missing. That inflates the absolute CAGR of every strategy and of the equal-weight benchmark. **The excess over equal weight is the fair number.**

### What was tested (S&P 500+400, 21-day hold, excess over equal weight, after costs)

| Signal | 2017–22 | 2023–26 | Verdict |
|---|---:|---:|---|
| **Earnings surprise** (EPS vs consensus, ≤ 63 sessions old) | **+2.9%/yr (t=2.3)** | **+2.9%/yr** | Only factor positive and stable in both periods |
| 12-1 / residual / sector-relative momentum (composite) | +1.2% | +7.1% | Weak, then strong; best-documented long-run premium |
| Announcement-day return | ≈0 | ≈0 | Noise |
| Short-term reversal, size, liquidity, 200-day gap | + | ≈0 or − | Decayed |
| Volatility / beta / idiosyncratic vol | − or + | flipped sign | Unstable |
| Pooled XGBoost ranker (18 price features ± earnings) | −1.6…−4.5% | +8…+9% | Loses in one period, so it's fitting a regime, not a signal |
| Ridge ranker; adaptive IC-weighted blend | +7…+14% | −10…+0% | Chases whatever worked lately |
| SPY 200-day trend filter on top of the portfolio | −8% | −4% | Whipsaws out of V-shaped recoveries |
| Sector-ETF momentum (top 1 / top 3 of 11) | +16% / +1.5% | 0% / +0.5% | No durable edge |
| **News** (bundled∩S&P 500, 114 stocks): attention, sentiment 5d/21d, sentiment change | mixed | mixed | At a 5-day hold, costs (~8%/yr) swamp everything; article count flipped sign (IC t +4.3 → −2.5); only *contrarian* 5-day sentiment at a 21-day hold was positive in both periods (+1.6% / +7.2%, t≤1.8); adding news to momentum made it worse in nearly every cut |

### Production composite and result

50% earnings surprise plus 50% momentum (1/6 each of 12-1, residual and sector-relative), as cross-sectional ranks. Hold the top 10%, rebalance every 21 sessions. The split and the top-10% size were both chosen on design-period results (top 10%: +3.8%/yr vs +1.4%/yr for top 20%) plus the literature prior on momentum.

| S&P 500+400, after costs | CAGR | Sharpe | Max DD | Excess vs EW |
|---|---:|---:|---:|---:|
| **Top 10%**, 2017–22 | 19.5% | 1.04 | −20.8% | +3.8%/yr |
| **Top 10%**, 2023–26 (holdout) | 29.7% | 1.44 | −18.1% | +7.9%/yr |
| **Top 10%**, full 2017–26 | **23.2%** | **1.20** | −20.8% | **+5.3%/yr (t=1.8)** |
| Equal-weight universe, full | 16.9% | 0.89 | −25.4% | — |
| SPY, full | 15.2% | 0.95 | −20.4% | — |

**Read honestly:** positive in both periods and consistent with published effects, but t = 1.8 over 9 years is *suggestive, not proven*. Part of the gap to SPY is survivorship bias in the stock universe (SPY has none). Rank IC is ~0.015–0.023. This is a modest tilt, not a money machine. It is live in the screener as the `momentum_pead` cross-sectional strategy (`GET /screener/cross-sectional/momentum_pead`), which runs exactly the backtested code on the latest bar.

### Not done / next

- News on the full S&P 500+400: the archive backfill for the 783 extra symbols was still running when this was written. Rerun with `scripts/backtest_stock_selection.py --news` once `alpaca_news.sync` finishes. Bundled-universe results give no reason to expect a lift.
- Point-in-time index membership (e.g. a paid CRSP / Norgate feed) would remove the remaining survivorship bias. It is the main thing standing between these numbers and real confidence.
- The screener ranks the app's bundled list live. Ranking all ~900 names needs a nightly precompute job: ~900 price and earnings fetches don't fit in a request under the shared Alpaca/Yahoo rate limits.

---

## v7 — 20 years, risk management, and calendar-luck checks

v6 only covered 2017–2026. v7 extends the stock prices back to 2004 for the same S&P 500+400 list. The 2006–2016 stretch is a pre-sample test: no v6 decision looked at it, and it includes 2008 and the 2009 momentum crash. v7 also adds point-in-time value/quality factors from SEC XBRL filings, and a multi-asset ETF layer. Reproduce with `python scripts/backtest_stock_selection.py --gate --all-offsets`.

### Calendar luck is real — report medians over all rebalance days

A monthly strategy rebalanced every 21 sessions can start on any of 21 days. The same strategy's 2017–22 CAGR ranged from **6.8% to 18.1%** depending only on that alignment. Every number below is the **median over all 21 offsets** (range in brackets), 2006-06 → 2026-09, after costs:

| Strategy | CAGR | Sharpe | Max DD (median) |
|---|---:|---:|---:|
| SPY | 11.4% [11.2–11.5] | 0.65 | −50% |
| Equal-weight universe | 14.4% [14.1–14.7] | 0.69 | −48% |
| **Stock selection, always invested** (v6 composite, top 10%) | **16.9%** [15.6–18.3] | 0.79 | −52% |
| SPY with trend gate (SPY > 10-month SMA, else SHY) | 9.1% [7.7–10.6] | 0.73 | −24% |
| **Stock selection + trend gate** | **14.4%** [11.4–16.2] | **0.86** | **−27%** |

By period (medians), always-invested vs gated:

| | 2006–16 | 2017–22 | 2023–26 |
|---|---:|---:|---:|
| Equal-weight | 12.5% | 15.8% | 18.5% |
| Stock selection | 11.4% | 19.2% | 30.5% |
| Stock selection + gate | 12.9% | 11.6% | 23.1% |

**Read honestly:**

- **Stock selection did not beat equal weight in 2006–16.** Momentum's 2009 crash (+12% vs +47% for equal weight) wiped out the edge. Over 20 years it adds ~+2.5%/yr over equal weight, all of it after 2017.
- **The trend gate is a risk tool, not a return tool.** It roughly halves the worst drawdown and sidesteps the momentum crash. The cost is ~2.5%/yr, lost to whipsaws in V-shaped recoveries (2020) and 2022. It's exposed as optional: `ss.market_risk_on` and `market_regime` in the screener output.
- **Survivorship bias grows further back.** The 2006–16 universe only contains companies that are still in the index today, which flatters both the stock books and equal weight, but not SPY. Treat the gap to SPY as an upper bound.

### Tested and rejected in v7

| Idea | Result |
|---|---|
| Value (E/P, B/P from SEC filings, point-in-time) | +2.3% / +0.5% / +5.2%/yr excess in 2010–16 / 17–22 / 23–26 alone; adding it to the composite lowered every period |
| Quality (gross profitability, ROA, low leverage) | ≈0 / 0 / +6.7%/yr; adding it lowered 2017–22 |
| GTAA (11 ETFs, each held only above its 10-month SMA) | Best risk profile of all: Sharpe ~1.0 in both halves, max DD −10%, +1% in 2008. But only 5.6%/yr |
| Dual momentum (GEM), top-3/5 ETF momentum | 7–8%/yr, no better than 60/40 on risk |
| 50/50 or 70/30 stock selection + GTAA blends | In between; no improvement over the gated book |
| Using GTAA instead of SHY as the risk-off asset | Same result (14.3% vs 14.3% in 2006–16), so the simpler SHY was kept |

### Bottom line

In this backtest, with free data, the best configurations were:

- **Max return:** the stock-selection book, ~17%/yr, with market-like crashes.
- **Balanced:** the same book behind a 10-month trend gate, ~14%/yr, Sharpe 0.86, about half the drawdown.

Both beat SPY in the backtest. The edge over a naive equal-weight portfolio is modest (+0 to +2.5%/yr), inconsistent across decades, and partly survivorship bias. Paper-trade it before trusting it, and get point-in-time index membership data before quoting any of these numbers as expected returns.

---

## v8 — copying Congress (Nancy Pelosi & co.)

**Data.** Pelosi's trades were rebuilt from the official House Clerk Periodic Transaction Reports: 65 filings covering 2014–2026 and 106 purchases, about half of them call options. The same parser now feeds the app ([congress_ptr.py](apps/api/app/services/congress_ptr.py), refreshed by `scripts/refresh_congress.py`).

The previous politician data was a hand-written demo fixture. Its "live" source, housestockwatcher.com, no longer exists. The fixture is still shipped as a last-resort fallback, but it is now always labelled `demo: true`.

**Method.** Copy each purchase at the first open *after the filing date* (the earliest a follower could act) and compare with SPY and QQQ over the identical window. The disclosure lag has a median of 24 days and a maximum of 45. Options are copied as the underlying stock.

| Pelosi purchases | n | vs SPY | t | Beat SPY |
|---|---:|---:|---:|---:|
| From *her* trade date, 12 months | 82 | +5.8% | 2.5 | 62% |
| **Copied from the filing date, 12 months** | 82 | **+4.4%** | 2.1 | 63% |
| Copied from the filing date, 3 months | 94 | −0.8% | 0.8 | 50% |

**Copy portfolio**, holding everything she disclosed buying in the last 12 months, equal weight, monthly, 2015–2026:

| Portfolio | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| Copy Pelosi | 29.2% | 0.98 | −41% |
| **Control: the 5 most-traded US stocks, equal weight, monthly** | **32.0%** | 1.03 | −42% |
| QQQ | 18.9% | 0.91 | −33% |
| SPY | 13.8% | 0.91 | −20% |

**Verdict.** Copying her *would* have made money after the disclosure lag. But a mechanical "hold the 5 biggest stocks" rule did slightly better with the same risk. Her record comes from concentration in mega-cap tech (≈5 names: AAPL, NVDA, MSFT, AMZN, GOOGL), plus option leverage, during a decade that rewarded exactly that. It is not evidence of copyable information.

Her record here is also flattered: SunEdison, which went bankrupt, and the pre-2020 Hertz trades have no price data and were excluded.

Congress more broadly: the NANC ETF (copies Democratic members' disclosed buys, real money since Feb 2023) returned 22.5%/yr vs SPY 20.5% and QQQ 28.7%. KRUZ (Republicans) trailed SPY and closed in 2025.

The in-app **Congress tracker** (`/congress`) runs the same copy-after-filing test for every House member with ≥ 5 purchases. The data is 32,833 disclosed trades from 242 members, 2020 → 2026, parsed from 3,244 electronic filings; 622 scanned paper filings are skipped.

Over the last 5 years, copying the **median member's** purchases trailed SPY by **−6.5% at 12 months**. 26 members trailed with t ≤ −2, and only 6 led with t ≥ 2.

Pelosi's last-5-year window, which includes her 2021–22 buys before the 2022 drawdown, is −6.2% vs SPY at 12 months (n = 25). That contrasts with +4.4% over 2014–2026: her record is very period-dependent.

These t-stats treat trades as independent. Trades cluster in time and in the same stocks, so real uncertainty is larger. Disclosed congressional buying is not a market-beating signal on average.

---

## v9 — forward test on paper, and better earnings-surprise measures (rejected)

### Paper forward test

[app/services/paper_strategy.py](apps/api/app/services/paper_strategy.py) runs the production composite live on the Alpaca **paper** account. It is surfaced on the `/live-test` page and driven by `scripts/paper_strategy.py`.

- Universe: current S&P 500 + 400.
- Once a rebalance is due (≥ 28 days since the last one): if SPY is above its 10-month SMA, hold the top 10% equal weight; otherwise hold SHY.
- The strategy keeps its own ledger (tagged `client_order_id`, fills synced from the broker), so it only ever sells what it bought and can share the paper account with the Trading Bot.
- It refuses to run against a real-money endpoint.

Fixes needed to make the live ranking honest at this scale:

- **Alpaca multi-symbol bars** (`alpaca_bars.get_bars_multi`, ~100 symbols per request). Per-symbol fetching of 900 names hit HTTP 429, and the yfinance fallback was rate-limited too.
- **IEX liquidity floor.** The free IEX feed reports only ~2–5% of consolidated volume, so the backtest's $5M/day filter was scaled for live data. Without it, most mid-caps were silently excluded.
- **Earnings cache.** `earnings.earnings_dates` cached Yahoo failures as "no earnings" for 6 h, which silently zeroed half of the composite. Failures are no longer cached, and there is a 3-day disk cache.

### Built-in simulator + historical replay of the live rules

The forward test now defaults to the platform's own paper broker (`paper_strategy_broker = "sim"`): fills at the live quote + 5 bps slippage + 1 bp commission, booked into the strategy ledger. Alpaca paper remains an option.

[paper_replay.py](apps/api/app/services/paper_replay.py) replays the *exact* live rules day by day, sharing `plan_orders`: same drift band, cash buffer, 28-day cadence and SPY gate, next-open fills with costs, and share holdings that drift between rebalances. `/live-test` shows it. $10k, 2016-01-04 → 2026-09-28, S&P 500+400:

| | Strategy | SPY |
|---|---:|---:|
| CAGR | **18.5%** | 15.2% |
| Sharpe | **1.01** | 0.79 |
| Max drawdown | **−27.1%** | −33.7% |
| 2016–19 / 2020–22 / 2023+ CAGR | 15.5% / 8.7% / 30.5% | 14.8% / 7.3% / 22.2% |

It beat SPY in 7 of 11 calendar years. It lagged badly in 2019 (+17.0% vs +31.1%) and was level in 2022 (−18.7% vs −18.6%). There were 140 rebalances, with ~64% turnover each (costs included).

Same caveat as v6/v7: survivorship bias (today's index members) flatters the stock side and not SPY, so read the gap as an upper bound. The forward test is the real check.

### Better surprise measures: tested, none adopted

Rule: a variant replaces the current %-surprise only if it beats it in **both** 2010–16 and 2017–22. S&P 500+400, top 10%, excess vs equal weight per year:

| Earnings block in the composite | 2010–16 | 2017–22 | 2023–26 |
|---|---:|---:|---:|
| **% surprise vs consensus (current)** | **+0.4%** | **+3.0%** | +8.1% |
| Surprise / price | −0.9% | +2.2% | +9.1% |
| SUE (SEC EPS, seasonal random walk, standardized) | −0.4% | −1.9% | +9.0% |
| % surprise + SUE + SUR (revenue) | +0.3% | +2.4% | +7.7% |
| Surprise/price + SUE + SUR | +0.1% | +2.8% | +8.6% |

None passes. Surprise/price looks best in the 2023–26 holdout (t = 2.8 standalone) but lost money in 2010–16. Picking it would be choosing on the holdout. The current measure stays.

---

# Trading Bot backtest — RSI(2) mean-reversion

The automated trading bot (`app/services/trading_bot.py` + `app/backtest/bot_engine.py`,
surfaced at `/bot/*` and the **Trading Bot** web page) runs a trend-filtered
short-term mean-reversion strategy. Unlike the forecaster numbers above, this is
a **trade-level** backtest: every entry is paired with an exit, so the headline
metric is a true **win rate** (winning trades ÷ total trades), not directional
accuracy.

## Strategy (tuned default)

- **Trend filter**: only go long while `close > SMA(200)`.
- **Entry**: `RSI(2) < 10` (a brief, sharp oversold dip inside the uptrend).
- **Exit**: whichever comes first —
  - take-profit at **+1.0%** (captures the mean-reversion pop), or
  - `close > SMA(15)` (price has reverted), or
  - protective stop at **−25%**, or
  - time stop after **40 bars**.
- **Fills**: next bar's open (no look-ahead). 1 bp commission per side.

## Results — pooled over the default universe, ~10y daily

Universe: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META.

| Metric | Value |
|---|---|
| **Pooled win rate** | **86.7%** |
| Trades | 376 |
| Profit factor | 2.57 |
| Avg win | +1.53% |
| Avg loss | −3.88% |
| Expectancy / trade | +0.81% |

### Per-symbol

| Symbol | Trades | Win % | Profit factor | Total return | Max DD | Sharpe |
|---|---|---|---|---|---|---|
| SPY   | 49 | 83.7% | 2.86 | +31.5%  | −8.4%  | 0.74 |
| QQQ   | 50 | 86.0% | 2.56 | +31.9%  | −11.2% | 0.64 |
| AAPL  | 50 | 82.0% | 1.81 | +22.5%  | −11.5% | 0.39 |
| MSFT  | 35 | 88.6% | 1.93 | +18.5%  | −13.8% | 0.41 |
| AMZN  | 43 | 79.1% | 1.84 | +19.1%  | −13.6% | 0.38 |
| GOOGL | 48 | 87.5% | 2.18 | +33.5%  | −15.4% | 0.56 |
| NVDA  | 59 | 94.9% | 4.12 | +184.7% | −25.0% | 1.04 |
| META  | 42 | 90.5% | 2.62 | +46.3%  | −17.2% | 0.57 |

Reproduce:

```bash
cd apps/api
python -c "import json; from app.services import trading_bot as b; print(json.dumps(b.backtest_portfolio(range_='10y')['pooled'], indent=2))"
```

## How to read this honestly

A high win rate is **not** a free lunch — it is the signature of an asymmetric
payoff. The strategy wins often (small +1% pops) and loses rarely but larger
(avg loss −3.9%, with a −25% tail stop). Profit factor (2.57) and positive
expectancy (+0.81%/trade) are what confirm the edge is real rather than an
artifact of the win-rate framing. The bot is **paper-only**: it computes signals
and would-be trades but never sends broker orders.

---

# Swing trading — portfolio backtests (v6)

Module: `app/services/swing.py` (setups, scanner, live helpers),
`app/backtest/swing_engine.py` (engine), `app/services/swing_data.py` (data),
surfaced at `/swing/*` and the **Swing Trading** page.

## Why a new engine

The RSI(2) numbers above pool per-symbol trades, each using 100% of equity. That
is not how an account behaves. The swing engine runs **one shared account**
across the whole universe:

- Signals are read at the daily close. Entries fill at the next open.
- Each trade risks 1% of current equity between the fill and its initial stop.
  Size is capped at 20% of equity per position, 8 positions, 100% gross
  (no margin), and 8% total open risk.
- Stops and targets rest against each bar's range. A bar that **opens through a
  stop fills at the open**, which models the overnight gap. If a bar touches
  both stop and target, the stop is assumed to hit first.
- Costs: 1 bp commission + 5 bps slippage per fill for equities. FX uses 2 bps
  per fill plus 1%/yr financing on notional. Idle cash earns nothing.
- Earnings: positions close at the last close before an SEC 8-K Item 2.02
  filing, and no entry is taken with a report inside 7 days.
- Each run reports a buy-and-hold SPY benchmark, split-half stability, a
  2,000-path bootstrap of trade R-multiples, and a verdict from fixed checks.

**Parameters were fixed before testing** (textbook defaults, listed in
`swing.SETUPS`) and were not tuned on these results.

## Data

- **Equities/ETFs:** Yahoo's chart endpoint, split- and dividend-adjusted,
  consolidated tape, back to 2004. The platform's default Alpaca IEX feed was
  *not* used. It starts in late 2018, and its IEX-only highs and lows are
  narrower than the real range, which under-counts stop hits.
- **FX:** Yahoo indicative daily bars. These contain garbage prints, such as
  EURUSD at 1.49 for one day in Dec-2008, a 0.0066 EURGBP low, and a 0.979
  EURGBP close in Oct-2022. `swing_data.clean_fx_bars` repairs them and keeps
  genuine shocks (SNB 2015, Brexit, Oct-2008). About 0.3–1.6% of bars per
  pair were repaired. Before cleaning, FX results were even worse.
- **Earnings dates:** exact historical release dates from SEC EDGAR 8-K Item
  2.02 filings. Coverage was 50/50 for the large-cap universe. ETFs and FX have
  none.

## Results — 2005-01-03 → 2026-09-28 (≈21.7 years)

SPY buy-and-hold over the same window: **CAGR 10.89%, Sharpe 0.64, max DD −55.2%**.

**Precision warning.** A portfolio with limited slots is path-dependent. One
signal that flips by a hair changes which trades get capital for months
afterwards. Re-downloading identical Yahoo history, which differs only by
rounding, moved single-run CAGRs by up to ~0.9 points. So each row below is the
**median of 20 runs on prices nudged by ±0.001%**, far below a price tick,
with the min–max range. The verdict column counts how many of the 20 runs got
each verdict.

| Setup | Universe | CAGR median [range] | Sharpe median [range] | Max DD | Trades | Win % | E[R]/trade | Verdict (of 20) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Breakout 55/20 | ETFs (28) | 3.98% [3.71, 4.36] | 0.51 [0.48, 0.55] | −23.0% | 1,098 | 42.2 | +0.22R | real but weak edge (20) |
| Pullback | ETFs | 0.78% [0.28, 1.17] | 0.13 [0.08, 0.17] | −26.4% | 4,238 | 41.6 | +0.005R | no edge (20) |
| RSI(2) | ETFs | 1.09% [1.07, 1.19] | 0.35 [0.34, 0.38] | −10.4% | 2,125 | 74.5 | +0.006R | inconclusive (20) |
| Breakout 55/20 | US large caps (50)† | 3.46% [2.82, 3.84] | 0.34 [0.29, 0.37] | −41.1% | 1,790 | 41.2 | +0.10R | inconclusive (20) |
| Pullback | US large caps† | 4.50% [3.62, 5.21] | 0.39 [0.33, 0.44] | −34.2% | 5,095 | 43.2 | +0.039R | inconclusive (19) |
| RSI(2) | US large caps† | 1.93% [1.78, 2.05] | 0.49 [0.44, 0.51] | −9.2% | 3,413 | 80.4 | +0.008R | real but weak edge (20) |
| Breakout 55/20 | FX majors (12) | −5.87% [−6.04, −5.68] | −0.34 [−0.35, −0.32] | −79.9% | 977 | 31.2 | −0.12R | no edge (20) |
| Pullback | FX majors | −4.84% [−5.70, −3.66] | −0.32 [−0.41, −0.22] | −67.9% | 2,511 | 40.7 | +0.010R | no edge (20) |

Max DD, trades, win % and E[R] are from the un-nudged run.
† Survivorship-biased: these are *today's* large caps, so the backtest never
holds the ones that shrank or were delisted. Treat these rows as an upper bound.

### Robustness

- **Doubling all costs** leaves only the ETF breakout positive in both halves
  (CAGR 2.57%, Sharpe 0.34, t 1.60). Every other equity row drops to "no edge"
  or "inconclusive". RSI(2) is the most cost-sensitive because its average win
  is small.
- **FX with zero costs and zero financing** is still negative for breakout
  (CAGR −3.04%). Pullback is flat (+0.18%, t 0.35). Capping FX leverage at
  1× does not rescue either. The second half (2016–2026) lost money in every
  FX variant tested.
- **The earnings rule costs return on large caps.** Breakout CAGR is 5.24%
  without it vs 4.12% with it on the same data, and RSI(2) is 2.20% vs 1.97%.
  It does cut the tail. Breakout's worst trade is −5.4R without the rule
  (META, Jul-2018) vs −2.7R with it, and trades worse than −2R fall from 7 to 1.
  The apparent benefit of holding through earnings is inflated by
  survivorship, because today's large caps are the ones whose reports went
  well. The rule stays on by default and can be toggled in the UI.
- **Earnings coverage gap:** dates follow the *current* SEC registrant. A
  company that re-registered has no dates before the change. For example,
  Google → Alphabet in 2015 left GOOGL uncovered, so a Jan-2012 Google earnings
  gap slipped through as a −3.5R pullback loss.

## How to read this honestly

1. **Nothing here beat buy-and-hold SPY on a risk-adjusted basis.** The best
   result, the ETF breakout, earned a consistent and statistically detectable
   +0.2R per trade. It did so with much smaller drawdowns (−23% vs −55%) and a
   0.32 correlation to SPY, but with a lower Sharpe (≈0.51 vs 0.64). That
   profile makes it a possible diversifying sleeve, not a replacement for
   holding the index.
2. **Multiple testing.** Eight configurations were run. The best t-stat (≈2.3)
   is below the ≈2.7 a Bonferroni correction for eight tests would require.
   Treat it as weak evidence.
3. **FX majors showed no edge for these setups**, and the failure is worst in
   the most recent decade. This matches the widely reported fade of FX trend
   returns after the 2000s. The platform's broker (Alpaca) also cannot trade
   spot FX.
4. **The RSI(2) bot looks very different at portfolio level.** Its headline
   86.7% win rate and 2.57 profit factor come from pooled per-symbol trades on
   a survivor set. In one shared account (2005–2026, costs, earnings rule) it
   still wins ~80% of trades, but it earns ≈1.9% a year because it is invested
   in only ~17% of equity on average.
5. **Cash yield is not modelled.** Idle cash would earn T-bill interest, but a
   fair comparison uses excess returns. Subtracting the risk-free rate lowers
   both the strategies' and SPY's Sharpe ratios and does not change the
   ranking.

## Live execution changes that came with this

- **Brackets now use whole shares and are good-til-cancelled.** Previously a
  notional-sized bracket was rejected by Alpaca, which needs whole shares for
  child legs, and silently fell back to an unprotected market order. Even when
  it went through, a `day` bracket's stop leg expired at the close, which is
  exactly when an overnight position needs it.
- **Open orders are cancelled before a software exit.** Leftover GTC legs are
  cancelled first, so the close isn't blocked and an orphaned stop can't open a
  short later.
- **Entry records persist in `run_state.positions`.** They hold the entry date,
  stop, target, and stop-leg id. Stops filled at the broker are reconciled
  (`CLOSED_BY_BROKER`). The time stop is now enforced live, including for
  RSI(2). Trailing stops ratchet the broker leg up and never down.
- **Swing bots** (`kind: swing_breakout | swing_pullback`) size by
  `execution.risk_per_trade_pct` of equity. They skip entries within 7 days of
  earnings and exit the day before a report. When Yahoo's calendar is
  unavailable, the next report is estimated from last year's SEC filing date
  with a ±7-day conservative margin. They are long-only and US equities/ETFs
  only.
- **Optional daily auto-run** (`BOT_AUTORUN=1`, `BOT_AUTORUN_ET=09:45`). Armed
  and active trader bots run once per US trading day with every existing
  guardrail. It is off by default.

## Reproduce

```bash
apps/api/.venv/bin/python scripts/backtest_swing.py                   # full grid → apps/api/artifacts/swing_results.json
apps/api/.venv/bin/python scripts/backtest_swing.py --jitter 20      # medians/ranges over 20 nudged-price runs (the table above)
apps/api/.venv/bin/python scripts/backtest_swing.py --cost-stress 2   # doubled costs
apps/api/.venv/bin/python scripts/backtest_swing.py --setup breakout --universe fx_majors
```

---

# ML signal filter for swing setups (v7)

Module: `app/ml/signal_filter.py`. API: `POST /swing/filter-eval`, plus
`ai_filter` on `POST /swing/scan`. Bot switch: `dsl.ml_filter` (off by default).

## What it is

A second model sits on top of a setup. The setup says "this is a trade". The
filter looks at the context and either takes the trade or skips it. It never
creates trades or changes stops, targets or exits. This design is known as
meta-labeling (López de Prado, *Advances in Financial ML*, ch. 3).

## Pre-declared design

Everything below was fixed before the first result was seen.

- **Events and labels.** Every bar where the setup fires is an event. Its label
  is the R-multiple of that trade under the setup's own exits, net of costs,
  from the same engine. The model learns "R > 0".
- **Features.** 23 inputs, all from bars at or before the signal close:
  - momentum over 5, 21, 63, 126 and 252 days;
  - volatility level and trend, and ATR %;
  - distance from the 50- and 200-day averages and the 55- and 252-day
    extremes;
  - RSI(14), the opening gap, and the volume trend;
  - how many symbols signalled that day;
  - SPY trend, return and volatility; VIX level and change; and the share of
    the universe above its 200-day average.
  FX market inputs are lagged a session because Yahoo's FX dates look shifted.
- **Model.** Shallow gradient boosting: depth 2, 200 trees, heavy
  regularisation. A regularised logistic regression is reported only as a
  cross-check.
- **Walk-forward.** The model is retrained every January 1 from 2010. It trains
  only on trades that had closed at least 10 days before the cut-off, so no
  training label overlaps the test year. It then scores that year only.
- **Rule.** Take a signal if P(win) ≥ the training win rate.
- **Verdict.** "Helps" only if the 90% paired block-bootstrap interval of the
  Sharpe improvement is above 0 *and* the out-of-sample AUC interval is above
  0.5.
- **Primary test.** The ETF breakout, the only setup with an edge worth
  filtering. Every other row is secondary.

Each evaluation also reruns the portfolio with two reference filters, a
**random** one that skips the same share of signals and a **perfect-foresight**
one that knows each outcome, so "no skill" and "the ceiling" sit next to the
model. On the ETF breakout, perfect foresight lifts Sharpe from 0.42 to 0.96,
which confirms the plumbing works. Random skips drop it to 0.29.

## Results — out of sample 2010 → 2026, plus 5 runs on nudged prices

| Setup | Universe | AUC [90%] | Sharpe: setup → filtered | Δ Sharpe [90%] | Δ in 6 runs | Verdicts (6 runs) |
|---|---|---:|---:|---:|---:|---|
| **Breakout (primary)** | **ETFs** | **0.516 [0.491, 0.541]** | **0.42 → 0.37** | **−0.05 [−0.24, 0.15]** | **−0.21 … −0.05** | **no improvement 5, hurts 1** |
| Pullback | ETFs | 0.532 [0.519, 0.544] | 0.07 → 0.27 | +0.20 [−0.04, 0.45] | +0.14 … +0.33 | no improvement 4, helps 2 |
| RSI(2) | ETFs | 0.631 [0.613, 0.649] | 0.26 → 0.28 | +0.02 [−0.13, 0.18] | −0.02 … +0.07 | no improvement 6 |
| Breakout | US large caps† | 0.512 [0.496, 0.530] | 0.56 → 0.63 | +0.07 [−0.21, 0.36] | +0.07 … +0.31 | no improvement 6 |
| Pullback | US large caps† | 0.527 [0.518, 0.536] | 0.45 → 0.39 | −0.06 [−0.28, 0.16] | −0.06 … +0.16 | no improvement 6 |
| RSI(2) | US large caps† | 0.642 [0.627, 0.657] | 0.51 → 0.78 | +0.27 [+0.08, +0.49] | +0.19 … +0.31 | helps 5, no improvement 1 |
| Breakout | FX majors | 0.484 [0.448, 0.517] | −0.48 → −0.43 | +0.06 [−0.20, 0.31] | −0.05 … +0.09 | no improvement 6 |
| Pullback | FX majors | 0.523 [0.499, 0.546] | −0.44 → −0.15 | +0.29 [−0.07, 0.52] | −0.21 … +0.30 | no improvement 6 |

SPY's Sharpe over the same 2010–2026 window is 0.86.
† Survivorship-biased universe.

## How to read this honestly

1. **The primary test failed.** On the ETF breakout the filter cannot tell good
   breakouts from bad ones (AUC 0.52, interval spanning 0.5), and it lowered
   Sharpe in all six runs. At 0.37 it lands between random skipping (0.29) and
   simply taking every signal (0.42). An early explanation, that skipped
   breakouts were re-entered later at worse prices, held on one data snapshot
   and reversed on the next, so it is not claimed here.
2. **The model does see something in RSI(2).** AUC is ≈0.63–0.64 in both
   universes, and both model families agree. Kept trades win ~82–85% of the
   time vs ~67–71% for skipped ones. The inputs it leans on are the depth of
   the dip (RSI, 5-day return) and volatility (ATR %).
3. **That skill only paid off on large caps, and it's weak evidence.** On
   large caps it lifted Sharpe from 0.51 to 0.78, consistently across nudged
   runs, with one-sided p = 0.012. That is one of eight tests, and the
   Bonferroni-adjusted p is ≈0.10. The universe is survivorship-biased. The
   result is still below SPY's 0.86 over the same years. On ETFs the same
   skill produced no portfolio gain.
4. **Where it "helped" a strategy with no edge**, as with the ETF pullback, it
   mostly cut exposure to a losing system. That isn't a reason to trade it.

**Default: off.** The honest next step, if you want one, is a forward paper test:
two paper RSI(2) bots on the same universe, one with `ml_filter` on, compared
after 6–12 months.

## Reproduce

```bash
apps/api/.venv/bin/python scripts/eval_signal_filter.py              # 8 configs → apps/api/artifacts/signal_filter_results.json
apps/api/.venv/bin/python scripts/eval_signal_filter.py --jitter 5   # + nudged-price robustness
```

---

# AI risk-gate forward test (v7)

The LLM gate (`bot_intel.llm_review`) can veto or shrink a quant BUY. It has
been on by default for the RSI(2) bot, but it had never been measured. It
also can't be backtested, because the model has read about the outcomes. Its
vetoes weren't stored anywhere, and the bot's log reset daily.

## What changed

- **Every verdict is logged.** Each gate verdict goes to the `ai_gate_decisions`
  table (migration `0007`, also auto-created) with symbol, signal date,
  strategy, source, decision, size multiplier, rationale and provider.
- **Scoring.** `gate_log.score_pending` replays the trade the quant signal
  would have taken, with the strategy's own exits and costs, once it has
  closed. It also records forward returns at 5, 10 and 20 sessions. This runs
  daily inside the auto-run loop and on every report request.
- **Scorecard.** `GET /bot/gate/report` feeds the **AI gate track record**
  panel on the Trading Bot page.
  - Gate value in R: −R for each vetoed trade, −(1 − m)·R for each downsized
    one. Positive means it saved money.
  - A bootstrap interval around that value.
  - Verdict: "collecting" until 20 vetoes or downsizes are scored, then
    "helping", "hurting" or "inconclusive".
  - Fail-open approvals, logged when the LLM was offline, are excluded.
  - Repeat reviews of the same signal are de-duplicated.
- **The gate prompt now describes swing signals correctly.** It used to tell
  the model every trade was an "oversold dip".
- **Tests can't pollute the log.** A test fixture routes the log to an
  in-memory database.

The log started empty on 2026-09-29. At the bot's signal frequency, expect
months before the verdict leaves "collecting".

---

# Forward paper test: RSI(2) with vs without the ML filter (started 2026-09-29)

The walk-forward result that the filter lifts RSI(2) on US large caps (Sharpe
0.51 → 0.78, 2010–2026) is one of eight tests, on a survivorship-biased
universe. The only way to learn whether it holds is to run it on data that
didn't exist when the design was fixed.

## Setup

Created by `scripts/setup_rsi2_ab_test.py` for the platform user `1@1.com`.
Both bots are tagged `experiment: rsi2_ml_filter_ab`.

| | Control | Treatment |
|---|---|---|
| Strategy | RSI(2) bot rules (RSI(2) < 10, above SMA200, SPY above SMA200, +1% target, SMA20 exit, −25% stop, 40-bar time stop) | same |
| ML filter | off | **on** (trained on the same 50 names, refreshed daily) |
| LLM gate | off | off |
| Universe | US large caps (50) | same |
| Signals | completed daily bars (decide at the close, act at the next 09:45 ET run) | same |
| Account | own simulated $100k ledger | own simulated $100k ledger |
| Sizing | $12.5k per position (1/8 of start), max 8 positions | same |
| Guards | 20 orders/day, $5k daily loss, 15% drawdown halt | same |

**Why simulated ledgers, not the Alpaca paper account.** Two bots on one
account contaminate each other. They share buying power, each would close
the other's positions, and the treatment's trades are a subset of the
control's, so they would collide on the same symbols. `broker_sim.SimBroker`
gives each bot its own ledger and runs the unchanged production code
(`trading_bot.run_live`):

- Market orders fill at the live quote, with 5 bps slippage and 1 bp
  commission.
- The +1% target and −25% stop rest as bracket legs and are settled against
  real 5-minute bars after the fill. A gap through the stop fills at the bar's
  open. If one bar touches both, the stop is assumed to hit first.

## How it runs

- The API's background loop runs both bots once per US trading day at 09:45
  New York time. This needs `BOT_AUTORUN=1`, now set in `apps/api/.env`, and
  the server running at that time.
- Auto-run is now per-bot opt-in (`execution.autorun`). No other bot is
  affected.
- If the server isn't always up, `scripts/run_paper_bots.py` does the same
  pass from cron. A bot never runs twice in one New York day.
- Missed days are missed by **both** arms, so the comparison stays fair.
  Open positions keep their resting target and stop. Those are settled from
  intraday bars at the next run, so no exit is lost.

## How to judge it

- **Where to look.** The Trading Bot page shows an *Experiment* panel with both
  equity curves, and `GET /bot/experiments` returns the same data.
- **When.** Not before ~100 closed trades per arm. At RSI(2)'s pace on 50 names
  that is roughly 6–12 months. Earlier differences are noise.
- **What counts as a win for the filter.** A higher return per unit of risk
  (Sharpe from the daily equity) that persists, and a higher win rate on a
  similar number of trades. Fewer trades with the same total return is also
  a win, since it means less exposure.
- **What doesn't count.** The treatment being ahead after a good month.
- **Known differences from the backtest.**
  - Entries fill at the 09:45 price, not the exact open.
  - The SMA20 exit is acted on the next morning, not at that day's close.
  - The live bot has no earnings rule.
  These affect both arms equally.

---

# Insider cluster buys (v8)

Modules: `app/services/insider_bulk.py` (loader), `app/ml/insider_study.py`
(study). Script: `scripts/backtest_insider_clusters.py`.

## Why a new loader

The old Form 4 adapter reads EDGAR full-text search, which returns only the
100 most recent filings per company. That's about 8 months of history for
JPMorgan, 15 for Pfizer and 2.5 years for Apple. The forecaster's insider
features were zero for every earlier date, so its "insider data earns no
weight" result never tested insider data.

The new loader reads the SEC's quarterly *Insider Transactions Data Sets*
(every Form 3/4/5 since 2006). It keeps open-market purchases and sales from
original Form 4s, one row per owner, keyed to the **filing date**, and deletes
each ~12 MB zip after extraction.

## Pre-declared design (fixed before results)

- **Event.** At least 2 distinct officers or directors file open-market
  purchases (code P) within 30 days, totalling at least $100k. Pre-scheduled
  10b5-1 purchases are excluded; that flag only exists from 2023. The event
  fires on the filing that completes the cluster. Each company gets one event
  per 6 months.
- **Trade.** Buy at the open after the filing date and sell at the open 126
  sessions later. Costs are 10 bps per side. There is no stop.
- **Universe.** The company's SEC CIK must still map to a ticker on
  NYSE/Nasdaq/NYSE American/Cboe today. Price at least $5 and median daily
  dollar volume at least $5M over the prior 60 sessions.
- **Primary test.** Event 6-month return minus SPY, minus the same measure for
  the **same stocks on random dates** with no insider buying within ±6 months.
  The verdict is "edge" only if the 90% interval, bootstrapped by calendar
  month, is above zero.
- **Secondary checks.**
  - A placebo matched on the prior 6-month return.
  - Split halves.
  - Liquidity terciles and cluster size.
  - A 20-slot calendar-time portfolio compared with SPY and with 20 placebo
    portfolios.

## Coverage

4.23 million purchase and sale rows, filed 2006-01-03 → 2026-06-30. They
cover 1.19 million purchases and 3.05 million sales.

| Stage | Events |
|---|---:|
| Cluster buys detected (all companies) | 19,950 |
| No current listing (delisted, acquired or re-registered) | −9,263 |
| Listed only on OTC | −462 |
| Below $5 or under $5M daily dollar volume at the event | −6,559 |
| Not yet 6 months old | −102 |
| **Studied** | **3,564** |

**Almost half the events belong to companies that no longer trade under
their filing identity.** They can't be priced with the free data used here,
and the direction of that bias is unknown. Some were bankruptcies, which would
flatter the results by their absence. Others were takeovers, which would hurt
them. The liquidity floor removes most micro-caps, which is where the academic
literature finds the effect, and also where 10 bps badly understates trading
costs.

## Results (primary test pre-declared; repeated with 3 placebo seeds)

| Measure (6-month, net of costs) | Result |
|---|---:|
| Events, return minus SPY | −0.06% [−1.60, +1.49] |
| Same stocks on random dates, minus SPY | −0.40% |
| **Primary: events minus same-stock placebo** | **+0.63% [−0.92, +2.18]**: not distinguishable (+0.63 / +0.96 / +0.78% over 3 seeds, all intervals span 0) |
| Secondary: events minus placebo matched on prior 6-month return | −1.70% [−3.10, −0.27] (−1.70 / −2.00 / −1.76%, all intervals below 0) |
| First half 2006–2015 / second half 2016–2026 | +0.99% / +0.45% |
| 2 insiders / 3 or more / CEO or CFO involved | +0.21% / +1.84% / +1.24% |

**Calendar-time portfolio** (20 equal slots, 2006-01 → 2026-09):

| | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| Insider cluster buys | 7.52% | 0.41 | −54.5% |
| SPY | 11.06% | 0.64 | −55.2% |
| 20 placebo portfolios (same stocks, random dates) | median 6.14% | median 0.38 (range 0.08–0.49) | |

The strategy beat 16 of the 20 placebo portfolios, short of the 19 of 20 a 95%
test needs. Its correlation to SPY is 0.85: it is mostly the market.

## How to read this honestly

1. **No edge in liquid, still-listed US stocks.** Following officer and
   director cluster buys did not beat SPY and did not beat the same stocks on
   random dates. This held in both halves of the sample and across placebo
   seeds.
2. **Against stocks with similar recent price moves, insider picks did
   worse.** Insiders tend to buy after declines. Compared with the same
   stocks at random moments with similar prior-6-month returns, their picks
   trailed by ≈1.7–2.0% over six months, and that interval excludes zero. So
   whatever raw return insider buys show is explained by buying after drops,
   not by what insiders know.
3. **Exploratory only: the size pattern.** Split by liquidity *after* seeing
   the results, the event-minus-placebo return was:
   - +3.9% [+1.6, +6.4] in the smallest third (median $9M/day traded);
   - +1.7% in the middle third;
   - −3.5% [−5.2, −1.8] in the largest third (median $134M/day).

   That matches the literature's claim that the effect lives in small
   companies. But it was chosen after the fact and is one of three groups.
   The smallest group is also exactly where survivorship bias and real
   spreads are largest. **It is a hypothesis, not a strategy.** Testing it
   properly needs price data that includes delisted companies and a spread
   model for small caps.
4. **The old forecaster result is superseded.** Insider data was never really
   tested there. Now it has been, on 20 years of complete filings, and it
   doesn't help in the universe the platform trades.

## Reproduce

```bash
apps/api/.venv/bin/python scripts/backtest_insider_clusters.py --sync   # first run: ~1 GB of SEC files + ~3,000 price histories
apps/api/.venv/bin/python scripts/backtest_insider_clusters.py          # later runs use the caches (~3 minutes)
```
