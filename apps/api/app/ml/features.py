"""Feature pipeline. Pure pandas — no extra deps.

Output: a single dataframe where each row is a feature vector for that timestamp
and `y_h<N>` columns are the future returns we want to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import indicators as ind
from app.services.macro import macro_features

# Stable feature → category mapping. Used by SHAP attribution bucketing.
FEATURE_CATEGORIES: dict[str, list[str]] = {
    "technical": [
        "ret_1", "ret_5", "ret_21", "ret_63",
        "vol_5", "vol_21", "vol_63",
        "sma_10", "close_over_sma_10",
        "sma_20", "close_over_sma_20",
        "sma_50", "close_over_sma_50",
        "sma_200", "close_over_sma_200",
        "rsi_14", "macd", "macd_hist",
        "bb_pctb", "atr_14", "atr_pct",
        "vol_z_20", "vol_chg_5",
        "drawdown",
        # scale-free versions of the price-level features above
        "macd_pct", "macd_hist_pct", "drawdown_252",
    ],
    "fundamental": [
        # legacy placeholders kept so old saved models still load
        "fund_pe", "fund_growth",
        # real PIT fundamentals (filled at backtest time when pit_history=True)
        "fund_eps_growth_yoy", "fund_revenue_growth_yoy",
        "fund_net_margin", "fund_debt_to_assets", "fund_eps_ttm",
    ],
    "sentiment_news": [
        "news_sent_7d", "news_count_z_30d",
        # PIT-historical news features (Alpha Vantage)
        "news_sent_mean", "news_sent_wmean", "news_count_z_30",
    ],
    "sentiment_social": ["social_hype"],
    "smc": [
        # Smart Money Concepts features (deterministic per-bar from price_action.analyze)
        "smc_confluence_score",
        "smc_trend_up",          # 1 if uptrend, -1 if down, 0 if range
        "smc_active_fvg_count",
        "smc_unmitigated_ob_count",
        "smc_dist_to_demand_atr",  # distance to nearest demand zone in ATR units
        "smc_dist_to_supply_atr",
        "smc_bos_recency",       # 1.0 if BOS in last 5 bars (decays linearly)
        "smc_choch_recency",
        "smc_sweep_recency",
        # Premium/discount array (ICT): signed position within the dealing range
        # (+ = discount/cheap, - = premium/rich) and OTE-pocket membership.
        "smc_premium_discount",  # +1 deep discount ... -1 deep premium (0 at equilibrium)
        "smc_in_ote",            # +1 in bullish OTE, -1 in bearish OTE, 0 otherwise
    ],
    "smart_money": [
        # Continuous insider density signals — from EDGAR Form 4
        "insider_net_buys_90d",
        "insider_net_value_90d",
    ],
    "macro": [
        # placeholders kept for backward compat
        "macro_yield_10y", "macro_vix",
        # real macro features joined from the macro service
        "vix_level", "vix_chg_5", "vix_z_60",
        "yield_10y_level", "yield_10y_chg_5", "yield_10y_z_60",
        "dxy_level", "dxy_chg_5", "dxy_z_60",
    ],
    "events": ["evt_earnings_window"],
}


def build_features(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    include_macro: bool = True,
    symbol: str | None = None,
    join_live_signals: bool = False,
    pit_history: bool = False,
) -> pd.DataFrame:
    """df must have columns: open, high, low, close, volume — indexed by ts.

    If `symbol` and `join_live_signals=True`, current fundamentals + 7d news
    sentiment + insider net flow are fetched from their respective services and
    broadcast as constant features on the most recent bars. These are non-zero
    only at inference; training keeps them at 0.0 (no point-in-time history).
    """
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # returns — use LOG returns for consistency with realized vol and the y_h*
    # targets below (all log-based). Avoids mixing simple/log return semantics.
    log_close = np.log(close)
    for h in (1, 5, 21, 63):
        f[f"ret_{h}"] = log_close.diff(h)

    # realized vol (log-returns)
    log_ret = log_close.diff()
    for w in (5, 21, 63):
        f[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    # moving averages and gaps
    for p in (10, 20, 50, 200):
        f[f"sma_{p}"] = ind.sma(close, p)
        f[f"close_over_sma_{p}"] = close / f[f"sma_{p}"] - 1

    # RSI / MACD / BB / ATR
    f["rsi_14"] = ind.rsi(close, 14)
    macd_df = ind.macd(close)
    f["macd"] = macd_df["macd"]
    f["macd_hist"] = macd_df["hist"]
    bb = ind.bollinger(close, 20)
    f["bb_pctb"] = bb["pctb"]
    f["atr_14"] = ind.atr(high, low, close, 14)
    f["atr_pct"] = f["atr_14"] / close

    # volume features
    f["vol_z_20"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    f["vol_chg_5"] = volume.pct_change(5)

    # drawdown
    rolling_max = close.cummax()
    f["drawdown"] = close / rolling_max - 1

    # Scale-free twins of the price-level features. Raw SMA/ATR/MACD are in
    # dollars, and `drawdown` depends on where the downloaded history starts,
    # so their ranges drift with the price level — a tree model uses them as a
    # proxy for *calendar time* and cannot extrapolate once price leaves the
    # training range. See NON_STATIONARY_FEATURES.
    f["macd_pct"] = f["macd"] / close
    f["macd_hist_pct"] = f["macd_hist"] / close
    f["drawdown_252"] = close / close.rolling(252, min_periods=1).max() - 1

    # placeholders for non-price features — initialized to neutral 0.0;
    # overwritten by news-service / fundamentals-ingest / macro-ingest in prod.
    f["news_sent_7d"] = 0.0
    f["news_count_z_30d"] = 0.0
    f["social_hype"] = 0.0
    f["fund_pe"] = 0.0
    f["fund_growth"] = 0.0

    # Smart Money Concepts per-bar summary features (deterministic).
    # We initialize to 0.0 (effectively zero-variance during training without
    # symbol context) — the model will treat them as constant and ignore them.
    # When `symbol` is set we compute real values via price_action.analyze().
    for c in ("smc_confluence_score", "smc_trend_up", "smc_active_fvg_count",
              "smc_unmitigated_ob_count", "smc_dist_to_demand_atr",
              "smc_dist_to_supply_atr", "smc_bos_recency",
              "smc_choch_recency", "smc_sweep_recency",
              "smc_premium_discount", "smc_in_ote"):
        f[c] = 0.0

    # Continuous insider density signals (from EDGAR Form 4)
    f["insider_net_buys_90d"] = 0.0
    f["insider_net_value_90d"] = 0.0

    if pit_history and symbol:
        # Per-bar SMC features: walk every ~21 trading days, computing the
        # SMC summary on the data up to that bar. Forward-fill between samples.
        # This is the right trade-off: SMC summaries are stable over 1-2 weeks.
        try:
            from app.services import price_action as pa_svc
            stride = 21
            tracked = ["smc_confluence_score", "smc_trend_up", "smc_active_fvg_count",
                       "smc_unmitigated_ob_count", "smc_dist_to_demand_atr",
                       "smc_dist_to_supply_atr", "smc_bos_recency",
                       "smc_choch_recency", "smc_sweep_recency",
                       "smc_premium_discount", "smc_in_ote"]
            n = len(df)
            for i in range(60, n, stride):
                chunk = df.iloc[: i + 1]
                if len(chunk) < 60:
                    continue
                try:
                    r = pa_svc.analyze(chunk)
                except Exception:
                    continue
                if not r.get("ok"):
                    continue
                conf = r.get("confluence") or {}
                last_close = float(chunk["close"].iloc[-1])
                atr_now = float(r.get("atr") or 1.0) or 1.0
                trend_v = 1.0 if r.get("current_trend") == "up" else (-1.0 if r.get("current_trend") == "down" else 0.0)
                active_fvgs = sum(1 for fv in (r.get("fvgs") or []) if not fv.get("filled"))
                unmit_obs = sum(1 for ob in (r.get("order_blocks") or []) if not ob.get("mitigated"))
                # distance to nearest demand / supply zone (in ATR units; positive = below price)
                demand_zones = [z for z in (r.get("zones") or []) if z["direction"] == "bullish"]
                supply_zones = [z for z in (r.get("zones") or []) if z["direction"] == "bearish"]
                d_dem = float("inf")
                for z in demand_zones:
                    d = max(0.0, last_close - float(z["upper"])) / atr_now
                    d_dem = min(d_dem, d)
                d_sup = float("inf")
                for z in supply_zones:
                    d = max(0.0, float(z["lower"]) - last_close) / atr_now
                    d_sup = min(d_sup, d)
                # event recency: 1.0 if event in last 5 bars; decays linearly to 0 at 20
                events = r.get("structure_events") or []
                bos = next((e for e in reversed(events) if e.get("kind") == "BOS"), None)
                choch = next((e for e in reversed(events) if e.get("kind") == "CHOCH"), None)
                sweep_evt = (r.get("liquidity_sweeps") or [{}])[-1] if r.get("liquidity_sweeps") else None
                def _recency(ev_ts: str | None) -> float:
                    if not ev_ts:
                        return 0.0
                    try:
                        ts = pd.Timestamp(ev_ts)
                        bars_ago = (chunk.index[-1] - ts).days / 1.4  # approx trading days
                        return float(np.clip(1.0 - bars_ago / 20.0, 0.0, 1.0))
                    except Exception:
                        return 0.0
                bos_r = _recency(bos.get("ts") if bos else None) * (1 if bos and bos.get("direction") == "bullish" else -1 if bos else 0)
                choch_r = _recency(choch.get("ts") if choch else None) * (1 if choch and choch.get("direction") == "bullish" else -1 if choch else 0)
                sweep_r = _recency(sweep_evt.get("ts") if sweep_evt else None) * (1 if sweep_evt and sweep_evt.get("direction") == "bullish" else -1 if sweep_evt else 0)
                # write at chunk.index[-1] and a few rows forward (will be ffilled later)
                row_idx = f.index.get_loc(chunk.index[-1])
                f.iloc[row_idx, f.columns.get_loc("smc_confluence_score")] = float(conf.get("score") or 0)
                f.iloc[row_idx, f.columns.get_loc("smc_trend_up")] = trend_v
                f.iloc[row_idx, f.columns.get_loc("smc_active_fvg_count")] = float(active_fvgs)
                f.iloc[row_idx, f.columns.get_loc("smc_unmitigated_ob_count")] = float(unmit_obs)
                f.iloc[row_idx, f.columns.get_loc("smc_dist_to_demand_atr")] = float(min(d_dem, 10.0))
                f.iloc[row_idx, f.columns.get_loc("smc_dist_to_supply_atr")] = float(min(d_sup, 10.0))
                f.iloc[row_idx, f.columns.get_loc("smc_bos_recency")] = float(bos_r)
                f.iloc[row_idx, f.columns.get_loc("smc_choch_recency")] = float(choch_r)
                f.iloc[row_idx, f.columns.get_loc("smc_sweep_recency")] = float(sweep_r)
                # premium/discount array (ICT): signed position within the dealing
                # range (+1 deep discount ... -1 deep premium) and OTE membership.
                dr = r.get("dealing_range") or {}
                pct = dr.get("pct_of_range")
                pd_signed = (0.5 - float(pct)) * 2.0 if pct is not None else 0.0
                in_ote = (1.0 if dr.get("in_bull_ote") else
                          -1.0 if dr.get("in_bear_ote") else 0.0)
                f.iloc[row_idx, f.columns.get_loc("smc_premium_discount")] = float(pd_signed)
                f.iloc[row_idx, f.columns.get_loc("smc_in_ote")] = float(in_ote)
            # forward-fill SMC features so every bar has the most recent value
            for c in tracked:
                f[c] = f[c].replace(0.0, np.nan).ffill().fillna(0.0)
        except Exception:
            pass

        # Per-bar insider density (cheap once cached)
        try:
            from app.services.edgar_form4 import insider_density_features
            dens = insider_density_features(symbol, list(f.index))
            f["insider_net_buys_90d"] = dens["insider_net_buys_90d"]
            f["insider_net_value_90d"] = dens["insider_net_value_90d"]
        except Exception:
            pass

    # legacy placeholder names (kept zero so old saved models still load with the same schema)
    f["macro_yield_10y"] = 0.0
    f["macro_vix"] = 0.0
    # earnings event flag — currently inference-only (zeros at training time
    # since we'd need an earnings-date history table; yfinance only ships ~8 dates)
    f["evt_earnings_window"] = 0.0
    if join_live_signals and symbol:
        try:
            from app.services.earnings import earnings_event_flag
            flag = earnings_event_flag(symbol, days_window=7)
            if flag > 0:
                idx_slice = f.index[-7:]
                f.loc[idx_slice, "evt_earnings_window"] = flag
        except Exception:
            pass

    # real macro features (VIX, 10Y yield, USD index), aligned by date
    if include_macro:
        try:
            mfeat = macro_features(df.index)
            for col in mfeat.columns:
                f[col] = mfeat[col].values
        except Exception:
            # If macro service is unreachable, keep training going on technicals-only.
            for col in ["vix_level", "vix_chg_5", "vix_z_60",
                        "yield_10y_level", "yield_10y_chg_5", "yield_10y_z_60",
                        "dxy_level", "dxy_chg_5", "dxy_z_60"]:
                f[col] = 0.0
    else:
        for col in ["vix_level", "vix_chg_5", "vix_z_60",
                    "yield_10y_level", "yield_10y_chg_5", "yield_10y_z_60",
                    "dxy_level", "dxy_chg_5", "dxy_z_60"]:
            f[col] = 0.0

    # PIT history join: at training time we want historical news sentiment +
    # fundamentals on every row, lagged appropriately to avoid look-ahead bias.
    if pit_history and symbol:
        try:
            from app.services.fundamentals_pit import features_for_index as _fund_pit
            fund = _fund_pit(symbol, f.index)
            for col in fund.columns:
                f[col] = fund[col].values
        except Exception:
            for col in ["fund_eps_growth_yoy", "fund_revenue_growth_yoy",
                        "fund_net_margin", "fund_debt_to_assets", "fund_eps_ttm"]:
                f[col] = 0.0
        try:
            from app.services.historical_news import fetch_history as _news_pit
            news = _news_pit(symbol, f.index)
            for col in news.columns:
                f[col] = news[col].values
        except Exception:
            for col in ["news_sent_mean", "news_sent_wmean", "news_count_z_30"]:
                f[col] = 0.0
    else:
        # Default zeros (live mode broadcasts on top of these for the most recent rows).
        # IMPORTANT: keep this list in sync with the PIT-history branch above so the
        # feature column set is identical regardless of which path runs.
        for col in ["fund_eps_growth_yoy", "fund_revenue_growth_yoy",
                    "fund_net_margin", "fund_debt_to_assets", "fund_eps_ttm",
                    "news_sent_mean", "news_sent_wmean", "news_count", "news_count_z_30"]:
            if col not in f.columns:
                f[col] = 0.0

    # live-join: at inference time we have today's news sentiment, current
    # fundamentals, and recent insider flow. These broadcast onto the most
    # recent ~5 bars and supplement / overwrite the PIT-history values.
    contaminated_idx = None
    if join_live_signals and symbol:
        live = _fetch_live_signals(symbol)
        # also count the earnings-window broadcast (last 7 bars) as contaminated
        n_broadcast = max(min(5, len(f)), min(7, len(f)))
        if n_broadcast > 0:
            contaminated_idx = f.index[-n_broadcast:]
        broadcast_bars = min(5, len(f))
        if broadcast_bars > 0:
            idx_slice = f.index[-broadcast_bars:]
            if live["news_sent_7d"] is not None:
                f.loc[idx_slice, "news_sent_7d"] = live["news_sent_7d"]
            if live["news_count_z_30d"] is not None:
                f.loc[idx_slice, "news_count_z_30d"] = live["news_count_z_30d"]
            if live["fund_pe"] is not None:
                f.loc[idx_slice, "fund_pe"] = live["fund_pe"]
            if live["fund_growth"] is not None:
                f.loc[idx_slice, "fund_growth"] = live["fund_growth"]

    # targets
    for h in horizons:
        f[f"y_h{h}"] = np.log(close.shift(-h) / close)

    # LEAKAGE GUARD: live signals (news/PE/earnings-window) are stamped onto the
    # most recent bars as constants. Those same bars still carry forward-return
    # targets y_h*, so training on them would leak today's info into the past.
    # Null the targets on every contaminated row — these are inference-only rows
    # and must never be used as training labels.
    if contaminated_idx is not None:
        for h in horizons:
            f.loc[contaminated_idx, f"y_h{h}"] = np.nan

    return f


def _fetch_live_signals(symbol: str) -> dict[str, float | None]:
    """Pull current signals; safe to call repeatedly (services cache)."""
    out: dict[str, float | None] = {
        "news_sent_7d": None, "news_count_z_30d": None,
        "fund_pe": None, "fund_growth": None,
    }
    try:
        from app.services import news as news_svc
        agg = news_svc.aggregate_sentiment(symbol, window_n=25)
        out["news_sent_7d"] = float(agg.get("weighted", 0.0))
        # cheap z-score proxy: scale article count to ~[-1, +1]
        n = float(agg.get("n", 0))
        out["news_count_z_30d"] = float(max(-1.0, min(1.0, (n - 8) / 15)))
    except Exception:
        pass
    try:
        from app.services import market_data as md
        ks = md.get_key_stats(symbol)
        pe = ks.get("pe")
        if isinstance(pe, (int, float)):
            # Normalize PE: log1p(pe), clipped — keeps tree splits stable
            out["fund_pe"] = float(np.clip(np.log1p(max(0.0, pe)) - 3.0, -2.0, 3.0))
        eps = ks.get("eps")
        # Use forward_pe / pe as a crude growth proxy
        fpe = ks.get("forward_pe")
        if pe and fpe and pe > 0 and fpe > 0:
            out["fund_growth"] = float(np.clip(np.log(pe / fpe), -1.5, 1.5))
        elif eps:
            out["fund_growth"] = float(np.tanh(eps / 5.0))
    except Exception:
        pass
    return out


def categorize_features() -> dict[str, set[str]]:
    return {k: set(v) for k, v in FEATURE_CATEGORIES.items()}


def category_of(feature_name: str) -> str:
    for cat, names in FEATURE_CATEGORIES.items():
        if feature_name in names:
            return cat
    return "technical"


# Columns whose level trends with price or with the calendar rather than
# describing the current state of the market. Kept in the frame (saved models
# reference them by name) but excluded from `feature_columns(stationary_only=True)`.
NON_STATIONARY_FEATURES: frozenset[str] = frozenset({
    "sma_10", "sma_20", "sma_50", "sma_200",   # dollars
    "atr_14", "macd", "macd_hist",             # dollars -> use atr_pct / macd_pct
    "drawdown",                                # anchored at the first downloaded bar
    "vix_level", "yield_10y_level", "dxy_level",  # multi-year regimes; _chg_5 / _z_60 kept
    "fund_eps_ttm",                            # dollars per share, grows with the company
})


def feature_columns(df: pd.DataFrame, stationary_only: bool = False) -> list[str]:
    # INVARIANT: no feature column may use future data. Exclude targets (y_h*)
    # and any look-ahead column such as the ichimoku lagging span (close.shift(-26)),
    # which is valid for charting but would leak future closes if used as a feature.
    return [
        c for c in df.columns
        if not c.startswith("y_h") and "lagging" not in c.lower()
        and not (stationary_only and c in NON_STATIONARY_FEATURES)
    ]
