"""Cross-sectional stock selection: momentum + post-earnings drift.

Why this and not a per-stock return forecast: v5 of the forecaster showed that
predicting a single stock's next 5/21-day return from its own history is noise
(IC ~ 0 on every symbol). Ranking stocks *against each other* is a different
problem — every date contributes ~150 labelled comparisons instead of one, and
a few effects are well documented and survive costs. The research behind the
weights below (scripts/backtest_stock_selection.py, BACKTEST_RESULTS.md "v6")
compared ~20 single factors, pooled XGBoost / ridge rankers and an adaptive
IC-weighted blend on 2017-2022 and kept a 2023-2026 holdout untouched. The
fitted models were unstable across the two periods; this fixed-weight
composite was not. Absolute returns below are survivorship-biased (today's
index members); the excess over the equal-weight universe is the fair number.

Composite (each factor is a cross-sectional percentile rank in [-0.5, 0.5]):
  50%   earn_surprise latest EPS surprise vs consensus, while < 63 sessions old
  1/6   mom_12_1      12-month return skipping the last month (Jegadeesh-Titman)
  1/6   resmom_12_1   same, on market-beta residuals, vol-scaled (Blitz et al.)
  1/6   sec_rel_mom   12-1 momentum minus its sector median

On the unbiased S&P 500+400 universe, earnings surprise was the only factor
positive in both the 2017-2022 design period (+2.9%/yr, t=2.3) and the
2023-2026 holdout (+2.9%/yr); momentum was ~0 then +7%/yr. The 50/50 split was
fixed from the design period and the literature, not tuned on the holdout.
`earn_ear` (announcement-day return) is still computed but carries no weight:
it was ~0 in every cut.

Portfolio in the backtest: at the close every 21 sessions, buy the top 10% at
the next open, equal weight, 10 bps per side on turnover.

Everything here is pure pandas on a (date x symbol) panel so the live ranking
and the backtest run literally the same code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

WEIGHTS: dict[str, float] = {
    "mom_12_1": 1 / 6,
    "resmom_12_1": 1 / 6,
    "sec_rel_mom": 1 / 6,
    "earn_surprise": 0.5,
}
# Optional news factors (app/services/alpaca_news.py), merged in by
# scripts/backtest_stock_selection.py --news. Empty = not part of the composite.
NEWS_WEIGHTS: dict[str, float] = {}
EARNINGS_FRESH_BARS = 63
HOLD_BARS = 21
# Market trend gate (Faber's 10-month SMA — the textbook value, not tuned here):
# hold stocks while SPY closes above its 210-session average, otherwise hold
# short-term Treasuries. Over 2006-2026 (median of all 21 rebalance-day
# offsets) it cut max drawdown from ~-52% to ~-27% at a cost of ~2.5%/yr of
# return. Optional: the ungated book is the max-return profile.
TREND_SMA_BARS = 210
TOP_FRACTION = 0.10
COST_PER_SIDE = 10 / 10_000


@dataclass
class EarningsEvent:
    ts: pd.Timestamp          # announcement timestamp (tz-aware when known)
    surprise_pct: float | None


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centred to [-0.5, 0.5]."""
    return df.rank(axis=1, pct=True) - 0.5


def universe_mask(close: pd.DataFrame, volume: pd.DataFrame,
                  min_dollar_volume: float = 5e6) -> pd.DataFrame:
    """Tradeable on a date: >= ~1y of history, price > $5, and average daily
    traded value above `min_dollar_volume` (consolidated-tape dollars; pass a
    scaled-down threshold for single-venue volume such as Alpaca's free IEX feed)."""
    hist = close.notna().cumsum() >= 260
    return hist & (close > 5) & ((close * volume).rolling(21).mean() > min_dollar_volume)


def _reaction_bar(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """First bar whose close reflects the announcement: same session if it came
    before the 09:30 New York open, otherwise the next session."""
    local = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    day = pd.Timestamp(local.date())
    before_open = local.hour < 9 or (local.hour == 9 and local.minute < 30)
    return int(index.searchsorted(day if before_open else day + pd.Timedelta(days=1)))


def earnings_factors(close: pd.DataFrame,
                     events: dict[str, list[EarningsEvent]]) -> dict[str, pd.DataFrame]:
    idx = close.index
    surprise = np.full(close.shape, np.nan)
    ear = np.full(close.shape, np.nan)
    for j, sym in enumerate(close.columns):
        c = close[sym].to_numpy(dtype=float)
        last = -1
        marks: list[tuple[int, float, float]] = []
        for ev in events.get(sym, []):
            pos = _reaction_bar(idx, pd.Timestamp(ev.ts))
            if pos <= 0 or pos >= len(idx) or not (c[pos - 1] > 0 and c[pos] > 0):
                continue
            sp = float(np.clip(ev.surprise_pct, -100, 100)) if ev.surprise_pct is not None else np.nan
            marks.append((pos, sp, math.log(c[pos] / c[pos - 1])))
        for pos, sp, r in sorted(marks):
            surprise[pos, j], ear[pos, j] = sp, r
        # carry the latest event forward while it is fresh
        for i in range(len(idx)):
            if not np.isnan(ear[i, j]):
                last = i
            elif last >= 0 and i - last <= EARNINGS_FRESH_BARS:
                surprise[i, j], ear[i, j] = surprise[last, j], ear[last, j]
    return {"earn_surprise": pd.DataFrame(surprise, index=idx, columns=close.columns),
            "earn_ear": pd.DataFrame(ear, index=idx, columns=close.columns)}


def compute_factors(close: pd.DataFrame, benchmark: pd.Series, sectors: dict[str, str | None],
                    events: dict[str, list[EarningsEvent]] | None = None) -> dict[str, pd.DataFrame]:
    """All factors at each bar's close, using only data up to that close."""
    lc = np.log(close)
    r1 = lc.diff()
    bench = np.log(benchmark.reindex(close.index)).diff()
    beta = r1.rolling(252, min_periods=126).cov(bench).div(
        bench.rolling(252, min_periods=126).var(), axis=0)
    resid = r1.sub(beta.mul(bench, axis=0))
    mom = lc.shift(21) - lc.shift(252)
    sec = pd.Series({s: sectors.get(s) or "_na" for s in close.columns})
    f = {
        "mom_12_1": mom,
        "resmom_12_1": resid.shift(21).rolling(231).sum() / (resid.shift(21).rolling(231).std() + 1e-9),
        "sec_rel_mom": mom - mom.T.groupby(sec).transform("median").T,
    }
    if events:
        f.update(earnings_factors(close, events))
    return f


def composite(factors: dict[str, pd.DataFrame], mask: pd.DataFrame,
              weights: dict[str, float] = WEIGHTS) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Weighted sum of cross-sectional ranks. A missing factor value (e.g. no
    fresh earnings) counts as neutral (rank 0)."""
    ranks = {k: cs_rank(v.where(mask)) for k, v in factors.items() if k in weights}
    score = sum(weights[k] * r.fillna(0.0) for k, r in ranks.items())
    return score.where(mask), ranks


def market_risk_on(benchmark_close: pd.Series, sma_bars: int = TREND_SMA_BARS) -> pd.Series:
    """True where the benchmark closes above its `sma_bars` simple moving average."""
    return benchmark_close > benchmark_close.rolling(sma_bars).mean()


def backtest(score: pd.DataFrame, open_: pd.DataFrame, mask: pd.DataFrame, *,
             hold: int = HOLD_BARS, top: float = TOP_FRACTION, start: str = "2017-01-01",
             cost_per_side: float = COST_PER_SIDE, min_names: int = 30,
             risk_on: pd.Series | None = None, cash_open: pd.Series | None = None) -> pd.DataFrame:
    """Rebalance every `hold` bars: decide at the close, fill at the next open,
    hold to the open `hold` bars later. Returns one row per rebalance with
    log-returns of the long book (net), the equal-weight universe and the
    long-short spread (net), plus rank IC and turnover.

    With `risk_on` (bool per date) and `cash_open` (open prices of a cash proxy
    such as SHY), a `gated` column is added: the long book while risk-on, the
    cash proxy otherwise, charging a full exit/entry on each switch."""
    fwd = np.log(open_.shift(-(1 + hold)) / open_.shift(-1))
    cash_fwd = (np.log(cash_open.shift(-(1 + hold)) / cash_open.shift(-1))
                if cash_open is not None else None)
    rows, prev = [], set()
    was_on = True
    for d in score.index[score.index >= start][::hold]:
        s = score.loc[d].dropna()
        y = fwd.loc[d].reindex(s.index)
        s, y = s[y.notna()], y[y.notna()]
        if len(s) < min_names:
            continue
        # at least 5 names in a stock universe; small universes (ETFs) use the fraction as-is
        n = max(5 if len(s) >= 25 else 1, int(len(s) * top + 1e-9))
        order = s.sort_values()
        longs, shorts = list(order.index[-n:]), list(order.index[:n])
        turnover = 1.0 if not prev else len(set(longs) ^ prev) / (2 * n)
        prev = set(longs)
        rl = math.log(np.mean(np.exp(y[longs])))
        rs = math.log(np.mean(np.exp(y[shorts])))
        row_gate = {}
        if risk_on is not None and cash_fwd is not None:
            on = bool(risk_on.get(d, True))
            switch_cost = 2 * cost_per_side if on != was_on else 0.0
            was_on = on
            row_gate = {"gated": (rl - 2 * cost_per_side * turnover if on
                                  else float(cash_fwd.get(d, 0.0))) - switch_cost,
                        "risk_on": on}
        rows.append({
            **row_gate,
            "date": d, "long": rl - 2 * cost_per_side * turnover,
            "long_short": rl - rs - 4 * cost_per_side * turnover,
            "equal_weight": math.log(np.mean(np.exp(y))),
            "ic": float(s.rank().corr(y.rank())), "turnover": turnover, "n_long": n,
        })
    return pd.DataFrame(rows).set_index("date")


def summarize(r: pd.DataFrame, hold: int = HOLD_BARS) -> dict:
    per_year = 252 / hold

    def curve(x: pd.Series) -> dict:
        x = x.dropna()
        if len(x) < 3:
            return {}
        eq = np.exp(x.cumsum())
        return {"cagr_pct": round((math.exp(x.sum() / (len(x) / per_year)) - 1) * 100, 1),
                "sharpe": round(x.mean() / (x.std() + 1e-12) * math.sqrt(per_year), 2),
                "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}

    excess = r["long"] - r["equal_weight"]
    return {
        "n_rebalances": len(r),
        "long": curve(r["long"]), "equal_weight": curve(r["equal_weight"]),
        "long_short": curve(r["long_short"]),
        **({"spy": curve(r["benchmark"])} if "benchmark" in r else {}),
        **({"gated": curve(r["gated"]), "time_risk_on": round(float(r["risk_on"].mean()), 3)}
           if "gated" in r else {}),
        "excess_vs_equal_weight_pct_per_year": round(excess.mean() * per_year * 100, 1),
        "excess_t_stat": round(excess.mean() / (excess.std() + 1e-12) * math.sqrt(len(excess)), 2),
        "rank_ic_mean": round(r["ic"].mean(), 4),
        "rank_ic_t_stat": round(r["ic"].mean() / (r["ic"].std() + 1e-12) * math.sqrt(len(r)), 2),
        "avg_turnover": round(r["turnover"].mean(), 3),
    }
