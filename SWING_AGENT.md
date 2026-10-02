# Multi-timeframe swing agent

A swing agent (2–10 day holds) that reads the daily chart for context, a 1-hour
regime for the intraday state and 15-minute bars for timing, picks a strategy
per regime combination, and is tested with the same code it will paper-trade
with. Built in phases, each one validated before the next:

1. **Data and universe** ← this document, current phase
2. Regimes (daily macro, intraday micro)
3. Individual strategies
4. Adaptive agent (selector + risk + execution + decision log)
5. Backtest and statistics (walk-forward, Monte Carlo, deflated Sharpe, final test)
6. Paper trading (Alpaca paper, degradation monitor, go-live criteria)

Code: `apps/api/app/swing_agent/`. Configuration: `apps/api/swing_agent.toml`
(the only config file). Tests: `apps/api/tests/test_swing_agent_data.py`.
Phase 1 pipeline: `scripts/swing_agent_data.py`.

---

## Phase 1 — data and universe

Status: **done, waiting for validation** (2026-10-02). Nothing in this phase
looks at strategy returns; the only return numbers below measure the
universe itself (survivorship bias), on the research window only.

### 1.1 Choices that were left open

| Item | Choice | Why |
|---|---|---|
| Market | US stocks + US ETFs | Free consolidated (SIP) intraday history since 2016 that keeps delisted names; the same vendor runs the paper account; regular sessions are what VWAP / opening-range / gap logic needs. Crypto trades 24/7 with 15–25 bps fees on Alpaca; FX majors were rejected in v6. |
| Universe | The 40 most liquid **point-in-time** S&P 500 members, re-ranked monthly, + SPY and QQQ | See 1.3. 40 is inside the 20–50 asked for and fixed before any test. |
| Capital | €10,000 | Account is in USD, so the backtest converts at the start date and reports the EUR/USD effect separately. |
| Tax residence | Portugal | See 1.9. |
| Timeframes | 1d context, 1h regime, 15m timing | 1h is built from 15m, aligned to the 09:30 open. |

### 1.2 Data vendors (checked 2026-10-02)

| Vendor | Cheapest plan with intraday history | Intraday history | Consolidated? | Delisted names | Verdict |
|---|---|---|---|---|---|
| **Alpaca Basic** | **free** (200 req/min) | **2016-01-04 →** (tested) | **Yes, SIP**, when the query ends >15 min ago | **Yes** (tested SIVB, FRC, TWTR, CELG, ATVI); `asof` maps renames (FB→META) | **Chosen** |
| Massive (Polygon.io renamed 2025-10-30) | Starter $29/mo (5y), Developer $79/mo (10y), Advanced $199/mo (20y+) | to 2003 on Advanced | Yes, 100% of volume | Yes, but renames are not stitched | Upgrade path if pre-2016 intraday (2008, 2011) is ever needed |
| Databento | $125 free credit; Standard $199/mo | Consolidated intraday (EQUS.MINI) only from 2023-03; Nasdaq-only from 2018-05 | Not before 2023 | Yes (point-in-time security master) | Best raw quality, too little consolidated history |
| Twelve Data | Grow $29/mo | 1-min from 2020-02 | Unverified | Unverified | No |
| IBKR (TWS API) | ~$10/mo + ≥$500 equity | Undocumented, 60 requests / 10 min | Yes | **No** ("no longer trading" unavailable) | Unusable for backtests |

**Recommendation: Alpaca Basic, €0/month.** It is the only free source of
consolidated 15m bars back to 2016, it keeps delisted symbols, and it is the
paper broker, so backtest and live read the same bars. yfinance is only used
as the fallback for daily series Alpaca lacks (the VIX comes from Cboe's own
file first).

What the free plan does *not* give, and what it costs us:
* nothing before 2016-01-04 intraday, so **2008 and 2011 can only be tested
  on daily bars**;
* intraday pages hold ~1 month of bars whatever `limit` says, so 10 years of
  15m bars is ~130 requests per symbol. The phase-1 download of 110 names
  took ~45 min;
* the last 15 minutes are off-limits, irrelevant for 2–10 day holds.

### 1.3 Point-in-time universe

* **Membership:** `fja05680/sp500` (free, GitHub), 1996 → 2026-08-18, one row
  per change, tickers as they were on the day. 753 membership intervals
  overlap 2016–2026; 250 of them belong to tickers no longer in the index.
  Its last update predates the 2026-09-21 changes, so it needs a refresh
  before live use.
* **Rule** (fixed in `swing_agent.toml`): on the first session of each month,
  take that day's members; keep raw close ≥ $5, bars on ≥90% of the previous
  63 sessions, median dollar volume ≥ $50M; rank by median dollar volume over
  the 63 sessions **before** the rebalance; keep the top 40, counting a
  company once (GOOG and GOOGL share a SEC CIK; correlation alone fails
  because their spread collapsed on 2021-07-28).
* **Result:** 127 months (2016-04 → 2026-10), always 40 names, **110 distinct
  names**. Eight have since left the index: TWTR, AGN, AAL, CELG, ENPH,
  VIAC, APC, and FB (renamed META, a survivor). They are 1.4% of stock-months.
* Delisted names keep their data until their last trading day; a cash
  take-out price comes from Alpaca's merger records when there is one.

### 1.4 Survivorship bias, measured

Same data, same rule, three universes, each held equal-weight and rebalanced
monthly on total-return prices, **research window 2017-01 → 2023-12 only**
(84 months; the final test was not read):

| Universe | CAGR | Vol | Max DD | Bias vs point-in-time |
|---|---:|---:|---:|---:|
| **Point-in-time top 40** (real rule) | **14.17%** | 18.2% | −28.9% | — |
| Survivors only (today's members), same rule | 14.45% | 18.2% | −29.2% | **+0.28 pp/yr** |
| Today's 40 most liquid, held throughout | 27.16% | 22.4% | −35.1% | **+12.99 pp/yr** |
| All S&P 500 members, point-in-time | 11.47% | 18.9% | −27.6% | — |
| All members, survivors only | 14.04% | 17.9% | −24.7% | **+2.57 pp/yr** |

How to read it:
* Classic survivorship bias on the whole index is **+2.6 pp/yr**. On a
  top-40-by-liquidity universe it is small (+0.3 pp/yr), because the most
  traded names rarely die while they are that liquid.
* The dangerous mistake is the **hindsight list**: "pick 40 liquid tickers"
  today (MU, NVDA, PLTR, HOOD, APP…) and backtest them. That adds **+13
  pp/yr** of fake performance. This is the same trap v6 found in the app's
  bundled stock list.
* Delisting returns barely matter here: valuing the five names whose data
  ends after a collapse at −100% moves the all-members CAGR by 0.05 pp.
* **Bias left in our setup ≈ 0** for 2016–2026: delisted names are in, and
  only 3 of 753 membership intervals had no data at first, fixed on retry.
  What remains: the membership file's own accuracy, and daily-only tests
  before 2016 (Yahoo has no delisted names) — those will be labelled.

### 1.5 Data quality: what was found and fixed

| # | Finding | Fix |
|---|---|---|
| 1 | **Alpaca records spin-offs as reverse splits** and applies them: WestRock/Ingevity 2016-05-16 is a "1→0.1666 reverse split", which makes earlier WRK prices 6× too high (a fake −84% day). Same for UTX→Carrier/Otis, EQT→Equitrans, TGNA→Cars.com, CNX, APTV, and small steps on HON, APD, DOC, VFC, JCI, SLG, AIV. | Applied steps are read from the data (raw/adjusted ratio), a step counts as a split only if the raw price really moved by its ratio, and anything else is undone (23 steps). The spin-off is credited as a cash distribution (from the new share's price, or the overnight gap when unknown; 20 estimates, flagged). |
| 2 | Corteva's 2026-10-01 spin-off is not adjusted at all (77.65 → 12.57). | Same mechanism; distributions never show as crashes in signal prices (`tr_factor`). |
| 3 | Alpaca's action list attaches events from **reused tickers** to the wrong company (PARA was reassigned on 2026-08-07; APC is now ARKO). | Each entity's symbol history is anchored at its own `asof` date. |
| 4 | The **Nasdaq earnings calendar** relabels history with today's tickers (Facebook's 2019 reports appear as META) and drops delisted companies (no TWTR, CELG, SIVB). | SEC 8-K Item 2.02 is the primary source, with the acceptance time for before-open / after-close. Nasdaq fills gaps. |
| 5 | EDGAR's ticker lookup only knows some old tickers; a name search matched AGN (Allergan plc) to the pre-2015 Allergan Inc. | Layered CIK resolution, each match validated by earnings 8-Ks filed while the name was in the index; two explicit, verified overrides (APC, AGN). All 110 names resolved. |
| 6 | GOOG −0.3% vs GOOGL +3.2% on 2021-07-28 looked like a bad print. | It is real (Yahoo agrees): the class spread collapsed. Not "fixed". |
| 7 | The 16:00 closing-auction print falls in the post-market 15m bar; regular-hours bars hold ~81% of daily volume. | Fills "at the close" use the official daily close; relative volume is computed on regular-hours bars only. |
| 8 | One daily batch silently dropped 3 symbols (DOW among them). | The step records misses; a re-run fetched them. Coverage is checked in the report. |

**Quality numbers after the fixes** (`phase1_report.json`):

| Check | Result |
|---|---|
| Member-sessions with a daily bar, 2016–2026 | **99.73%** (1,359,177 of 1,362,817); 4 of 753 intervals have no data (AABA, CCEP, KDP and WYND around their mergers), none ever liquid enough for the universe |
| Daily bars: OHLC inconsistencies / non-positive prices | 0 / 0 (1.81M bars, 755 series) |
| Applied splits that passed the raw-price check | 121 kept, 23 undone; raw = adjusted × split factor within 2% everywhere except 2 series never in the universe (POM, STI) |
| 15m bars | 4.27M regular-hours bars, 113 series |
| Missing 15m slots | median 0.07%, worst CCL 0.78% |
| First 15m open vs official open | median 0 bps (p99 of per-name p99s: 62 bps — opening-auction prints) |
| Last 15m close vs official close | median 1.5 bps |
| Regular-hours share of daily volume | median 81% (the rest is the closing auction and extended hours) |
| Daily moves >25% inside the universe | 14, all real news (NFLX 2022-04-20, META 2022-02-03, INTC 2024-08-02, ORCL 2025-09-10, SMCI…) |
| Overnight gaps >8% inside the universe | 476; **only 47% fall on an earnings reaction session** — the rest are market shocks (2020-03, 2024-08-05, 2025-04) and company news (AMD +37.5% on 2025-10-06, BA 2019-03-11). Avoiding earnings removes about half of the gap risk; sizing has to handle the rest. |

### 1.6 Event calendars

* **Earnings:** 4,827 reports for the 110 universe names (~4 a year each):
  4,138 found in both sources, 536 SEC only (mostly delisted names), 153
  Nasdaq only. Timing: 2,283 after close, 1,569 before open, 765 released
  during the session, 210 unknown (both sessions are then avoided). When the
  sources disagree on the date, both dates are kept.
  Coverage: **99.84%** of the 5,080 universe stock-months have a report within
  ±70 days (8 one-month holes). Unknown timing falls from 8.5% of reports in
  2016 to 2.4% in 2025.
* **Macro:** 92 FOMC statements (incl. the 2020 emergency ones), 128 CPI and
  131 employment reports (2016 → 2026; the 2025 shutdown's cancelled
  releases are correctly missing). Future CPI/FOMC dates for live use come in
  phase 6.
* **VIX:** Cboe daily OHLC, 2015 → yesterday. It settles at 16:15, so it is
  only usable after 16:15.

### 1.7 Time conventions (the no-look-ahead contract) and tests

* Intraday bars are indexed by start time and carry `t_close`; a decision at
  time *t* may only read bars with `t_close ≤ t` (`store.known_at`).
* During session D, daily bars stop at D−1; D becomes readable at D's close
  (13:00 on half days).
* 1h bars are built from 15m bars aligned to 09:30; an hour closes on the
  clock even if a 15m bar is missing; the last hour is 30 minutes.
* Liquidity ranks only use bars before the rebalance day — a test poisons
  the future with absurd volumes and checks the ranking does not change.
* Signal prices use a forward-anchored total-return factor, so a level on
  day *t* never depends on later dividends. (Split adjustment does use later
  splits, as usual; that is why the $5 filter and per-share costs use raw
  prices.)
* The calendar is the broker's: holidays, 13:00 half days, one-off closures
  (2018-12-05, 2025-01-09), DST switches.
* `apps/api/tests/test_swing_agent_data.py`: 22 tests, all offline.

### 1.8 Splits of the history

| Segment | Dates | Use |
|---|---|---|
| Warm-up | 2016 | indicators only |
| Train | 2017–2021 | design and fitting; contains Q4-2018 and the 2020 crash |
| Validation | 2022–2023 | model choice; contains the 2022 bear and the 2023 bank failures |
| **Final test** | 2024-01 → 2026-09 | read **once**, through `FinalTestLock`, which writes a receipt and refuses a second opening |

Every reader defaults to the research segment (train + validation), so code
cannot touch the final test by accident.

### 1.9 Rules and taxes (verified 2026-10-02)

* **PDT rule: gone.** The SEC approved FINRA's rewrite of Rule 4210 on
  2026-04-14; since **2026-06-04** the "pattern day trader" designation and
  the $25,000 minimum no longer exist (FINRA Regulatory Notice 26-10; brokers
  may phase in until 2027-10-20). Margin accounts now face an intraday margin
  check instead. Alpaca switched on 2026-06-04. Consequence for the agent: no
  day-trade counter is needed; a cash account still has to respect T+1
  settlement (no selling of shares bought with unsettled cash).
* **Portugal:** gains on shares and ETFs are taxed at 28% (art. 72.º CIRS).
  Gains on positions held under 365 days must be aggregated with other income
  at progressive rates if total taxable income reaches the top bracket
  (€86,634 for 2026). Losses can only be carried forward 5 years if you opt
  for aggregation. US dividends: 15% withheld with a W-8BEN, Portugal's 28%
  applies with a credit for it. A swing agent's gains are all short-term.
  The after-tax estimate comes in phase 5, once there are trades.

### 1.10 Reproduce

```bash
apps/api/.venv/bin/python scripts/swing_agent_data.py calendar membership daily cik universe
apps/api/.venv/bin/python scripts/swing_agent_data.py corporate intraday earnings macro vix
apps/api/.venv/bin/python scripts/swing_agent_data.py report     # → apps/api/artifacts/swing_agent/phase1_report.json
apps/api/.venv/bin/python -m pytest apps/api/tests/test_swing_agent_data.py -q
```

---

## Phase 2 — regimes

Status: **done, waiting for validation** (2026-10-02). No strategy and no
trading return is computed in this phase.

### 2.0 Written before any regime result was computed

* Rules and thresholds: `swing_agent.toml` [regime.*], textbook values.
* Everything is evaluated on the research window only, train (2017–2021) and
  validation (2022–2023) reported separately. The final test is not read.
* **Rules vs hidden-Markov model (decision rule fixed in advance).** The
  Markov-switching model replaces the rules only if, on the **validation**
  years, it beats them on all three of: (1) forward 10-day volatility
  separation between risk-off and risk-on days, (2) mean detection lag on
  drawdowns ≥ 10%, (3) no more than 1.5× the rules' switches per year.
  Otherwise the rules stay. "Clearly better" means all three, not on average.

### 2.1 What was built

| Layer | Where | Decides on | Inputs |
|---|---|---|---|
| Market (macro) daily regime | `regime_daily.market_regime` | each close, usable next morning | SPY, VIX (Cboe), point-in-time breadth |
| Stock daily regime | `regime_daily.stock_regime` | each close | the stock's own SMA50/200, ADX, ATR rank |
| Intraday (micro) regime | `regime_intraday.micro_regime` | each 1h close (10:30 … 15:30, 16:00) | session VWAP, gap vs yesterday, volume and range vs the same time of day, extension from the open |
| Evaluation | `regime_eval` | — | shape, forward information, detection lag |
| ML alternative | `regime_hmm` | each close | SPY returns only, Markov-switching filter |
| Experiment log | `experiments.jsonl` | — | 21 distinct variants so far, 44 rows |

Breadth = share of the S&P 500 *members of that day* above their own 50-day
average (median 63%, from 1% in March 2020 to 98%).

### 2.2 One structural change, made on the train years and logged

The first version (high vol overriding everything) had two defects that show
without looking at any strategy:

* the VIX stayed above 20 until 2021, so the whole 2020 recovery was
  high_vol for 218 sessions: back to bull only on 2021-04-14, **88% of the
  rebound missed**;
* *bear* was almost never used (1–2% of days), because declines lift the VIX.

Version 2 separates **trend** (price only) from **vol**. The label is the
trend, except that high vol *without* a bull trend is high_vol. A bull trend
in high vol stays bull, flagged `vol_state = high` for the risk layer to
shrink positions. 2020: back to bull on 2020-06-08, 45% of the rebound
missed. Version 1 is kept as variant `v1_priority` (93% of days identical).

### 2.3 Market regime — results

| | Train 2017–21 | Validation 2022–23 |
|---|---:|---:|
| Bull / range / bear / high_vol (% of days) | 37 / 39 / 2 / 23 | 18 / 36 / 1 / 45 |
| Switches per year | 14.8 | 9.6 |
| Median spell (sessions): bull · range · high_vol | 11 · 8 · 8 | 18.5 · 14 · 48 |
| Forward 10-day vol, risk-off ÷ risk-on | **2.25×** | **1.70×** |
| Mean detection lag after a ≥10% peak | 10.7 sessions | 11.0 sessions |

Forward 10 days of SPY by regime (from the next open):

| Regime | Train: mean · median · vol | Validation: mean · median · vol |
|---|---|---|
| bull | +0.23% · +0.62% · 11.6% | +0.26% · +0.53% · 11.0% |
| range | +0.72% · +0.95% · 9.7% | −0.50% · −0.03% · 14.9% |
| high_vol | **+1.22%** · +1.70% · 24.5% | +0.57% · +1.51% · 23.4% |
| bear | (20 days, 2 independent obs.) | (6 days) |

**How to read it.** The regime separates *risk* well: forward volatility after
risk-off days is about twice that after calm days, in both periods. It does
**not** separate *returns* the way intuition says: stressed days were
followed by the highest average returns (rebounds), bull days by the lowest.
Going to cash in high_vol therefore buys lower volatility at the price of
missed rebounds. That points the agent toward **sizing down** in high vol,
not switching off — to be tested in phase 5. Observations are few: 28
independent 10-day windows of high_vol in train, 22 in validation.

Detection, per drawdown (risk-off = bear or high_vol):

| Drawdown | Depth | Detected after | Drawdown already taken | Back to bull | Rebound missed |
|---|---:|---:|---:|---|---:|
| 2018-01-26 → 02-08 | −10.1% | 6 sessions | 78% of it | 2018-02-26 | 7.8% |
| 2018-09-20 → 12-24 | −19.3% | 23 sessions | 33% | 2019-02-13 | 17.3% |
| 2020-02-19 → 03-23 | −33.8% | 3 sessions | 14% | 2020-06-08 | 45.1% |
| 2022-01-03 → 10-12 (validation) | −24.5% | 11 sessions | 22% | 2022-12-01 | 14.3% |

Fast crashes are caught early (2020: −4.7% when flagged); the short, sharp
Feb-2018 drop was mostly over before detection. Re-entry is the slow side.

### 2.4 Parameter sensitivity (one at a time, 13 variants + v1)

Share of days with the same label as the baseline, train / validation:
VIX 22/18 → 94% / 85%; VIX 28/23 → 98% / 84%; ADX 15 → **80%** / 92%;
ADX 25 → 85% / 97%; SMA50 slope over 5 or 20 days → ≥97%; confirm 1 or 3
days → ≥95%; no breadth gate → 91% / 98%; breadth 60% → 94% / 98%; SMAs
40/150 or 60/250 → ≥94%; ATR rank 95% → ≥98%. Volatility separation stays
2.17–2.37 (train) and 1.66–1.87 (validation); mean lag stays 10–11 sessions,
except without the breadth gate (21 in train). Nothing flips the picture.
The ADX threshold moves days between bull and range, which is what it
defines; the agent should not depend on that boundary being sharp.

### 2.5 Rules vs hidden-Markov model — decided by the rule in § 2.0

| Validation 2022–23 | Rules | HMM 2 states | HMM 3 states |
|---|---:|---:|---:|
| Forward vol separation | **1.70** | 1.69 | 1.36 |
| Mean detection lag (sessions) | 11 | **2** | 81 |
| Switches per year | **9.6** | 25.2 | 30.2 |
| Adopt? | — | **no** (fails 1 and 3) | no |

The 2-state model sees crashes sooner and misses less of rebounds (2022:
10.6% vs 14.3%), but flips 2.6× as often and does not separate risk better.
**The rules stay.** Its filtered risk-off probability is noted as a
candidate input for position sizing, to be tested (and counted as a variant)
in phase 5.

### 2.6 Stock daily regimes

Same trend rules on each of the 110 universe stocks (and SPY/QQQ/IWM):
during their months in the universe, range 38%, bull 34%, bear 18%,
high_vol 10%; 13.8 switches per stock-year. Bear is common at stock level
(18%) even though it is rare for the index.

### 2.7 Intraday regime — results

Share of hourly closes (stocks): range 75%, trend up/down 8% each, reversal
4% each, gap-and-go 1% each; about 0.9 label changes per session.

Forward move from the hourly close to the session close, **in the label's
direction**, in daily ATRs, relative to the average universe stock at the
same hour (so the market's afternoon move is removed); t-stats clustered by
session:

| Label | Train mean (t) | Validation mean (t) | Reads as |
|---|---:|---:|---|
| trend_up | −0.021 (−6.0) | −0.032 (−6.1) | gives back |
| trend_down | −0.029 (−7.5) | −0.022 (−3.6) | gives back |
| gap_go_up | −0.080 (−5.1) | −0.060 (−2.8) | gap fades |
| gap_go_down | −0.107 (−6.5) | −0.077 (−3.7) | gap fades |
| reversal_up | −0.027 (−4.8) | −0.005 (−0.7) | weak |
| reversal_down | −0.017 (−3.0) | −0.018 (−2.3) | weak |
| range | −0.002 (−2.4) | +0.001 (+1.1) | nothing |

**How to read it.** In these large caps, intraday strength and weakness tend
to **revert** for the rest of the day, in both periods: "trend" and
"gap-and-go" are descriptions, not continuation signals. The effect is small
(0.02–0.1 daily ATR ≈ 4–20 bps for a stock with a 2% ATR), so it is not a
strategy on its own after costs, but it says something useful for 15m
timing: **do not chase an intraday move to enter a swing trade.** SPY and
QQQ labels show no reliable effect except the late-session drift in range.

Volatility state, controlled for the hour of day: after *expanding* hours the
rest-of-day move is 1.15× (train) / 1.11× (validation) the normal for that
hour; after *compressing* hours 0.91× / 0.95×. Expansion persists;
**compression does not announce a same-day breakout**.

### 2.8 No-look-ahead tests (12, `tests/test_swing_agent_regimes.py`)

* Collapsing prices, the VIX and breadth after day k leaves every regime up
  to k unchanged; day k's regime is invisible at 16:00 (VIX settles 16:15)
  and visible the next morning.
* Poisoning the 15m bars after 11:45 and today's daily bar leaves every
  intraday feature and label up to 11:45 unchanged.
* The same-time-of-day baselines ignore today; trailing ranks ignore the
  future; breadth drops a stock the day it leaves the index.
* **Mutation check:** making the intraday layer read today's daily ATR, or
  stamping the market regime at 16:00 instead of 16:15, makes the tests fail.

### 2.9 Chart

`apps/api/artifacts/swing_agent/regimes/regime_chart.html`: SPY with the
regime as background, VIX and breadth panels, hover for the daily values,
table view by year.

### 2.10 Limits of this phase

* Few independent episodes: 4 drawdowns ≥10% in seven years; bear spells are
  too rare at index level to say anything about them.
* Regimes describe the present with a lag of ~2 weeks on the way down and
  more on the way up; they cannot see a regime never met in 2016–2023.
* The intraday effects are statistically clear but economically small;
  whether the intraday layer adds value to swing trades is phase 5's
  question (agent vs daily-only version).

### 2.11 Reproduce

```bash
apps/api/.venv/bin/python scripts/swing_agent_regimes.py daily sensitivity hmm intraday charts
apps/api/.venv/bin/python -m pytest apps/api/tests/test_swing_agent_regimes.py -q
```

---

## Phase 3 — individual strategies

Status: **done, waiting for validation** (2026-10-02). Parameters are the
textbook values in `swing_agent.toml` [strategy.*], fixed before the first
run; nothing was tuned. Research window only.

### 3.1 The engine (`engine.py`, 13 tests in `tests/test_swing_agent_engine.py`)

One cash account of €10,000 converted at the EUR/USD of 2016-12-30 (1.0575 →
$10,575), simulated bar by bar on 15m bars:

* signal on a bar's close, **fill at the next bar's open**; no signals in the
  first or last 30 minutes;
* stops checked before targets; a bar opening through the stop fills **at
  the open** (gap); same-bar stop and target → stop; a take-profit needs a
  trade *through* it; the entry bar only checks the stop;
* trailing stops ratchet on 15m closes, never down; time and strategy exits
  fill in the **closing auction** at the official close;
* 1% of equity at risk per trade, whole raw shares, ≤ 20% of equity per
  position (5 × 20% = 100%, no margin), ≤ 1% of median 15m volume;
* no entry within 5 sessions of an earnings reaction session; held positions
  leave in the auction before it; no entries on FOMC days or before 10:30 on
  CPI / jobs days;
* dividends credited in cash to overnight holders; splits through the
  adjusted share count; delisted stocks closed at the take-out price;
* costs: half-spread max(½¢, 1 bp), slippage 2 bps (market) / 5 bps (stop) /
  1 bp (auction), SEC and FINRA sell fees; limits free.

Tests pin each rule on hand-built bars, and a poisoned-future test (with a
mutation check) shows the strategy signals never read later bars or today's
daily bar. A first run exposed two real bugs, both fixed: split ratios read
from rounded prices (AAPL 3.99996 instead of 4 → fractional shares) and a
25% position cap that let four positions fill the account.

### 3.2 The four strategies (long only)

| Strategy | Setup (yesterday's daily bar) | Trigger (today, 15m) | Exits |
|---|---|---|---|
| Pullback | stock regime bull, RSI(3) ≤ 30 | back above the session VWAP after trading below it | stop 1.5 ATR, target 3 ATR, trail 1.5 ATR after +1.5 ATR, 10 sessions |
| VWAP reversion | above SMA200, RSI(2) ≤ 10 | ≥ 0.5 ATR below the session VWAP | close > 5-day average (auction), stop 2.5 ATR, 5 sessions |
| Compression breakout | Bollinger width in bottom 20% of 6 months, above SMA50 | above the 20-day high on ≥ 1.2× volume | stop 1.5 ATR, target 3 ATR, trail 2 ATR after +1.5 ATR, 10 sessions |
| Gap and go | above a rising SMA50, today's gap ≥ 0.5 ATR | above the open and VWAP on ≥ 1.5× volume, by 11:30 | stop under session low (0.5–1.5 ATR), target 2 ATR, trail 1 ATR, 5 sessions |

Each has a **daily-only twin**: same setup, entry at 10:00 with no trigger,
same exits — the test of whether the 15m layer adds anything.

### 3.3 Results (after costs; train 2017–21 | validation 2022–23)

| Strategy | CAGR | Sharpe | Max DD | Trades | Mean R (t) | Exposure | Costs / gross profit |
|---|---|---|---|---|---|---|---|
| Pullback · 15m | 6.4% \| −7.2% | 0.54 \| −0.55 | −17.8% \| −18.6% | 727 \| 237 | +0.08 (1.9) \| −0.09 (−1.2) | 65% \| 51% | 25% \| — |
| Pullback · daily | 9.3% \| −8.3% | 0.76 \| −0.66 | −16.4% \| −24.3% | 706 \| 226 | +0.12 (2.8) \| −0.10 (−1.4) | 64% \| 52% | 18% \| — |
| VWAP reversion · 15m | 0.6% \| 2.1% | 0.13 \| 0.52 | −14.5% \| −6.0% | 262 \| 84 | +0.02 (0.4) \| +0.05 (0.8) | 9% \| 8% | 40% \| 13% |
| VWAP reversion · daily | 2.8% \| −2.9% | 0.35 \| −0.36 | −18.2% \| −8.6% | 889 \| 278 | +0.02 (1.3) \| −0.03 (−0.9) | 30% \| 23% | 34% \| — |
| Compression · 15m | 0.8% \| 9.7% | 0.16 \| 1.14 | −14.2% \| −5.0% | 311 \| 102 | +0.08 (1.1) \| +0.21 (1.7) | 28% \| 23% | 51% \| 6% |
| Compression · daily | 5.9% \| −2.6% | 0.54 \| −0.11 | −14.4% \| −24.6% | 739 \| 265 | +0.09 (2.1) \| −0.03 (−0.4) | 65% \| 59% | 25% \| — |
| Gap and go · 15m | 0.8% \| 0.1% | 0.19 \| 0.04 | −9.4% \| −4.9% | 304 \| 64 | +0.13 (1.3) \| −0.03 (−0.1) | 9% \| 4% | 54% \| 85% |
| Gap and go · daily | 2.3% \| −3.5% | 0.31 \| −0.52 | −16.4% \| −15.9% | 916 \| 235 | +0.07 (1.2) \| −0.03 (−0.3) | 24% \| 15% | 58% \| — |
| **SPY buy & hold** | **18.1% \| 1.6%** | **1.00 \| 0.18** | −33.8% \| −24.5% | | | 100% | |

20 runs on prices nudged by ±0.001% move CAGRs by at most ±0.3 points: the
numbers are stable for this data. Position sizing is not a detail, though:
with a 25% cap instead of 20%, the same pullback trades (+0.08R each) gave
9.7% instead of 6.4% a year — more risk per trade, same edge.

Stress windows (portfolio return over the window):

| | Q4 2018 | Crash 2020 | Rebound 2020 (03-23 → 08-31) | Bear 2022 (01-03 → 10-12) | Banks 2023 |
|---|---:|---:|---:|---:|---:|
| Pullback · 15m | −4.5% | −16.3% | +19.0% | −16.3% | +0.6% |
| VWAP reversion · 15m | −7.3% | −4.0% | +1.7% | −2.0% | +0.1% |
| Compression · 15m | −5.5% | −1.5% | −1.4% | +1.2% | +4.3% |
| Gap and go · 15m | −0.1% | −0.5% | +0.5% | −0.5% | 0.0% |
| SPY | −19.3% | −33.8% | +57.8% | −24.5% | +3.0% |

### 3.4 How to read it

1. **No strategy beats SPY.** In the train years (a strong bull market) the
   best, the daily-only pullback, made 9.3% a year against 18.1%, with a
   lower Sharpe (0.76 vs 1.00). In validation only three of the eight runs
   made money — VWAP reversion · 15m (+2.1%), compression · 15m (+9.7%) and
   gap and go · 15m (+0.1%), the 15m versions that trade the least.
2. **The edge per trade is small and does not survive validation.** The
   best train result, +0.12R per trade (t 2.8), became −0.10R in 2022–23.
   Costs take 0.01–0.09R per trade, 18–58% of gross profit.
3. **The 15m trigger does not add value.** In train the daily-only twin has
   the higher mean R for three of four strategies. The trigger mostly makes
   strategies trade less (lower exposure, so smaller losses in 2022).
4. **Too good to be true, flagged:** compression · 15m in validation (Sharpe
   1.14, +0.21R) against 0.16 in train, on 102 trades with t = 1.7. Most likely
   noise, and the daily twin lost money over the same years.
5. **What does hold:** in crashes these strategies lose far less than the
   index (pullback −16% vs −34% in 2020), because they are half out of the
   market and stops cut losers — and they catch far less of the rebound.
   That is a risk profile, not an edge.

### 3.5 Performance by regime (the matrix phase 4 would use)

`summary.json` holds, per strategy and segment, mean R by market regime, by
intraday regime and by their combination (cells ≥ 10 trades). The cells
with |t| ≥ 2 in train — pullback entered in the first hour (+0.22R, t 3.1),
VWAP reversion after an intraday reversal down (+0.21R, t 2.9), compression
on an intraday trend up (+0.31R, t 2.1), gap and go in a range market
(+0.42R, t 2.3) — do **not** repeat in validation (−0.02, −0.21, +0.08,
+0.10R). With ~48 cells, 2–3 of them reaching t ≥ 2 by chance is expected.
There is no regime × strategy pairing in this data that a selector could
learn on train and trust on validation.

### 3.6 Reproduce

```bash
apps/api/.venv/bin/python scripts/swing_agent_strategies.py run jitter chart
apps/api/.venv/bin/python -m pytest apps/api/tests/test_swing_agent_engine.py -q
```
Outputs in `apps/api/artifacts/swing_agent/strategies/` (`summary.json`, trades
and equity per run, `jitter_summary.csv`, `stress_windows.json`,
`strategies_chart.html`). Experiment log: 39 distinct variants, 80 rows.

---

## Intraday-only hypotheses (2026-10-02, asked for after phase 3)

"With AI, smart money and the rest, an intraday trader must be able to make a
profit." Four intraday-only hypotheses, flat at the close (no overnight risk;
the PDT rule no longer applies since 2026-06-04), parameters written into
`swing_agent.toml` before the first run, same engine and costs, plus a run at
2× spread and slippage. Research window only.

| Hypothesis | Source | Train: R/trade (t) · CAGR | Validation: R/trade (t) · CAGR | Costs per trade | At 2× costs, train |
|---|---|---|---|---|---|
| H1 gap-down fade: still below open and VWAP on 1.5× volume, 10:30–14:30 → buy, sell in the auction | phase-2 finding | −0.025 (−1.7) · −2.6% | +0.020 (0.9) · +2.2% | 0.017R | −0.041R |
| H2 liquidity sweep ("smart money"): takes out yesterday's low, closes back above it | SMC | −0.098 (−8.0) · −10.7% | −0.083 (−4.3) · −9.0% | 0.095R | −0.192R |
| H3 last half hour, SPY/QQQ: buy 15:30 if the first half hour was up | Gao et al. 2018 | −0.101 (−10.7) · −2.9% | −0.060 (−4.4) · −2.1% | 0.082R | −0.179R |
| H4 30-min opening-range breakout (control) | retail classic | −0.035 (−2.9) · −6.7% | −0.030 (−1.6) · −5.5% | 0.044R | −0.079R |

**How to read it.**

* None makes money after costs. H2 and H3 are roughly flat before costs; the
  costs (a tight stop makes a few bps of spread and slippage worth almost
  0.1R) turn them clearly negative.
* H1 is the phase-2 effect (gap-downs recover about 0.12 daily ATR into the
  close, measured from the hourly close). With a real fill at the next bar's
  open, a stop and costs, it is −0.025R in train and +0.02R in validation:
  the effect is real but smaller than the cost of trading it.
* This matches the evidence on retail day trading: in Taiwan fewer than 1% of
  day traders profit predictably net of fees (Barber, Lee, Liu & Odean,
  2014); in Brazil 97% of those who kept at it for 300+ days lost money
  (Chague, De-Losso & Giovannetti, 2019). In liquid large caps the intraday
  edges are a few basis points wide and are taken by market makers whose
  costs are lower than ours.
* Experiment log: 47 distinct variants so far. Every one counts against any
  future "winner" when its Sharpe is deflated in phase 5.

---

## Studies: which stocks to buy, and new companies (2026-10-02)

Rules in `swing_agent.toml` [study.*], written before the first run;
research window only (data ≤ 2023-12-31); script `scripts/swing_agent_studies.py`,
outputs in `apps/api/artifacts/swing_agent/studies/`.

### A. The platform's stock-selection model, without survivorship bias

The production composite (`app/ml/stock_selection.py`: 50% earnings surprise,
50% momentum, top 10%, monthly, 10 bps a side), code and weights unchanged,
on the **point-in-time S&P 500** (742 names, including those that left or
died) vs the same rule on today's survivors (528). Median over all 21
rebalance-day alignments:

| 2017–21 \| 2022–23 | Model book | Equal weight | SPY | Excess vs equal weight (t) |
|---|---|---|---|---|
| Full composite, point-in-time | 15.2% \| 4.1% | 15.6% \| 0.9% | 17.1% \| 2.8% | +0.5 (0.2) \| +2.4 (0.6) %/yr |
| Full composite, survivors only | 15.1% \| 4.7% | 17.9% \| 3.7% | 17.1% \| 2.8% | −1.8 (−0.6) \| +0.7 (0.1) |
| Momentum only, point-in-time | 8.2% \| 1.8% | 15.6% \| 0.9% | 17.1% \| 2.8% | −6.1 (−1.3) \| +1.5 (0.2) |

* With survivorship removed the model has **no measurable edge**: excess t of
  0.2–0.6, a book below SPY in 2017–21. The earlier replay (18.5% vs 15.2%
  for SPY on today's S&P 500+400) was mostly the bias: survivors-only equal
  weight is +2.3 points a year above point-in-time equal weight.
* Limits: S&P 500 only (no free point-in-time S&P 400), and yfinance has no
  earnings history for dead companies, so their surprise counts as neutral.

### B. Buying new IPOs

All 2,537 US IPOs priced 2016-01 → 2022-12 (Nasdaq calendar), minus SPACs,
units, deals under $50M or $5 → **1,071**, including the ones that later
failed or were taken over. 0.5% round-trip cost. 252-session hold:

| Entry | IPOs 2016–20 (n ≈ 698) | IPOs 2021–22 (n ≈ 370) |
|---|---|---|
| First close | mean +13.3%, **median −4.1%**; beat SPY 35%; lost half 17% | mean −44%, median −55%; beat SPY 15%; **lost half 55%** |
| +21 sessions | median −5.6%; vs SPY −7.2% (t −2.3) | median −56%; beat SPY 13% |
| +126 sessions (lock-up) | median −16.4%; vs SPY −8.9% (t −2.5) | median −44%; beat SPY 22% |

Calendar-time portfolio (equal weight, every IPO in its first year, one
return a day): **5.1%/yr vs 13.5% for SPY**, Sharpe 0.32 vs 0.79, max
drawdown **−75%**. The mean is lifted by a few huge winners while the typical
IPO loses: a lottery ticket. Consistent with Ritter and with Loughran &
Ritter's "new issues puzzle".

### C. Buying spun-off companies

Only **11** spin-offs by S&P 500 members (2016–22) are in Alpaca's records,
three of them really mergers (GE→WAB, PFE→VTRS, T→WBD); many others are
misrecorded as reverse splits (§ 1.5). 12-month excess vs SPY: −4.4%
(train, n = 4), −4.3% (validation, n = 7), t ≈ −0.2. **Inconclusive**: far
too few events; a fair test needs a complete spin-off list.

### Conclusion of the three studies

Nothing here beats simply holding the index once the biases are removed.
The stock-selection edge disappears without survivorship bias; buying new
IPOs loses money on a typical deal and in a portfolio; spin-offs cannot be
judged with free data.
