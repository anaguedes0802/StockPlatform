"""Walk-forward backtest of the forecaster on real historical data.

Usage (from the repo root):
    python scripts/backtest_forecaster.py AAPL --horizon 5 --range 5y --step 60 --mode ensemble

What it does:
  1. Pulls daily bars for the symbol.
  2. For each anchor t = warmup, warmup+step, warmup+2*step, … :
       - Trains fresh models on bars[:t+1] (targets that would need bars > t
         are NaN, so no label leaks across the anchor).
       - DENSE (default): predicts the h-day-ahead return at EVERY bar in
         [t, t+step) with that fold's models — ~step predictions per fold, the
         same "retrain every `step` bars" cadence production would run.
         Features at bar j only use data up to j.
       - SPARSE (--sparse): the legacy mode — one prediction per fold at bar t.
         16 folds = 16 samples, so one prediction flipping moves directional
         accuracy by 6pp. Too few to tell any two model versions apart; kept
         only to reproduce the historical numbers in BACKTEST_RESULTS.md.
  3. Scores against baselines that are computed from TRAINING data only
     (no hindsight):
       - always_up          — predict "up" every time
       - drift              — point = mean training h-day log-return
       - zero               — point = 0 (random walk)
       - climatology        — quantiles = empirical q10/q50/q90 of training returns
       - base_rate          — P(up) = training up-rate (for the Brier score)
       - naive_trailing     — next h-day return = last h-day return
     Hindsight majority is reported for reference only.
  4. Metrics: directional accuracy (± std error on the effective sample size —
     consecutive h-day targets overlap, so n_eff ≈ n / h), MAE, pinball loss
     (proper scoring rule for quantiles), p10–p90 coverage + width, Brier
     score, information coefficient (Spearman corr of forecast vs realized),
     and a daily-rebalanced long/flat strategy net of costs vs buy & hold.

Honest disclosures:
  - XGBoost params are FIXED here (use_tuned_params=False): the Optuna study
    tunes on the full date range, so loading those params into an earlier fold
    would leak that fold's own future into its hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import pandas as pd

# Allow running from repo root.
sys.path.insert(0, "apps/api")

from app.ml import features as feat_mod  # noqa: E402
from app.ml.arima_model import AutoARIMAForecaster  # noqa: E402
from app.ml.ensemble import REGIME_MODEL_MIX  # noqa: E402
from app.ml.regime import detect  # noqa: E402
from app.ml.xgboost_model import XGBForecaster  # noqa: E402
from app.services import market_data as md  # noqa: E402

QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}


def _fold_predictions(
    df: pd.DataFrame, t: int, positions: list[int], horizon: int, mode: str, symbol: str,
) -> pd.DataFrame:
    """Train on bars[:t+1]; return q10/q50/q90/prob_up for each bar in `positions`."""
    train_df = df.iloc[: t + 1]
    last_pos = positions[-1]
    # Features at bar j depend only on bars <= j, so one build over bars[:last+1]
    # serves every test position of the fold.
    test_feats = feat_mod.build_features(df.iloc[: last_pos + 1], symbol=symbol, pit_history=True)
    test_idx = df.index[positions]

    members: dict[str, pd.DataFrame] = {}
    # use_tuned_params=False — see module docstring.
    m = XGBForecaster(use_tuned_params=False)
    m._training_symbol = symbol
    m.fit(train_df, horizon)
    members["xgboost"] = m.predict_logrets(test_feats).reindex(test_idx)

    if mode == "ensemble" and AutoARIMAForecaster.available():
        try:
            a = AutoARIMAForecaster()
            a.fit(train_df, horizon)
            members["arima"] = a.predict_logrets(df["close"].iloc[: last_pos + 1], positions, horizon)
        except Exception as e:
            print(f"    arima failed: {e}")

    if len(members) == 1:
        return members["xgboost"]

    # Regime-weighted blend in log-return space, mirroring ensemble._blend:
    # median = weighted mean of medians; band edges = blended median plus the
    # weighted mean of each member's offset from its own median.
    out = []
    for pos, ts in zip(positions, test_idx, strict=True):
        mix = REGIME_MODEL_MIX[detect(df["close"].iloc[: pos + 1])]
        rows = [(members[k].loc[ts], mix.get(k, 0.0)) for k in members if not members[k].loc[ts].isna().any()]
        total = sum(w for _, w in rows) or 1.0
        q50 = sum(r["q50"] * w for r, w in rows) / total
        out.append({
            "q10": q50 + sum((r["q10"] - r["q50"]) * w for r, w in rows) / total,
            "q50": q50,
            "q90": q50 + sum((r["q90"] - r["q50"]) * w for r, w in rows) / total,
            "prob_up": sum(r["prob_up"] * w for r, w in rows) / total,
        })
    return pd.DataFrame(out, index=test_idx)


def backtest(
    symbol: str, horizon: int, range_: str, warmup: int, step: int, mode: str = "xgboost",
    *, dense: bool = True, slippage_bps: float = 5.0, commission_bps: float = 1.0,
) -> dict:
    df = md.get_history(symbol, interval="1d", range_=range_)
    if df.empty:
        raise SystemExit(f"no data for {symbol}")
    df = df.sort_index()
    n = len(df)
    if n < warmup + step + horizon:
        raise SystemExit(f"not enough bars: {n} (need at least {warmup + step + horizon})")

    close = df["close"].astype(float)
    log_close = np.log(close)
    anchors = list(range(warmup, n - horizon, step))
    print(f"backtesting {symbol}: {n} bars, {len(anchors)} folds, horizon={horizon}d, "
          f"step={step}d, mode={mode}, {'dense' if dense else 'sparse'}")

    records: list[dict] = []
    for i, t in enumerate(anchors):
        positions = list(range(t, min(t + step, n - horizon))) if dense else [t]
        # Training-only baselines: realized h-day returns whose target bar <= t.
        train_y = (log_close.shift(-horizon) - log_close).iloc[: t + 1].dropna().values
        t0 = time.time()
        try:
            preds = _fold_predictions(df, t, positions, horizon, mode, symbol)
        except Exception as e:
            print(f"  fold {i+1}/{len(anchors)} FAILED: {e}")
            continue
        clim = np.quantile(train_y, [0.10, 0.50, 0.90])
        for pos in positions:
            ts = df.index[pos]
            p = preds.loc[ts]
            if p.isna().any():
                continue
            records.append({
                "ts": ts, "fold": i,
                "actual": float(log_close.iloc[pos + horizon] - log_close.iloc[pos]),
                "next_day": float(log_close.iloc[pos + 1] - log_close.iloc[pos]),
                "naive": float(log_close.iloc[pos] - log_close.iloc[pos - horizon]),
                "drift": float(train_y.mean()),
                "base_rate": float((train_y > 0).mean()),
                "clim_q10": clim[0], "clim_q50": clim[1], "clim_q90": clim[2],
                **{k: float(p[k]) for k in ("q10", "q50", "q90", "prob_up")},
            })
        fold_recs = [r for r in records if r["fold"] == i]
        acc = np.mean([(r["prob_up"] > 0.5) == (r["actual"] > 0) for r in fold_recs]) if fold_recs else float("nan")
        print(f"  fold {i+1}/{len(anchors)} train_end={df.index[t].date()} n={len(fold_recs)} "
              f"dir_acc={acc:.2f} mean_q50={np.mean([r['q50'] for r in fold_recs]):+.4f} "
              f"({time.time() - t0:.1f}s)")

    if not records:
        raise SystemExit("no successful folds")
    res = _summarize(pd.DataFrame(records).set_index("ts"), horizon,
                     slippage_bps=slippage_bps, commission_bps=commission_bps)
    return {"symbol": symbol, "horizon_days": horizon, "mode": mode,
            "dense": dense, "n_folds": len(anchors), **res}


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _summarize(r: pd.DataFrame, horizon: int, *, slippage_bps: float, commission_bps: float) -> dict:
    y = r["actual"].values
    up = y > 0
    n = len(r)
    n_eff = max(1.0, n / horizon)  # overlapping h-day targets

    def acc(pred_up: np.ndarray) -> float:
        return float((pred_up == up).mean())

    model_up = r["prob_up"].values > 0.5
    dir_model = acc(model_up)
    hindsight_major = bool(up.mean() >= 0.5)

    pin_model = np.mean([_pinball(y, r[k].values, tau) for k, tau in QUANTILES.items()])
    pin_clim = np.mean([_pinball(y, r[f"clim_{k}"].values, tau) for k, tau in QUANTILES.items()])
    brier_model = float(np.mean((r["prob_up"].values - up) ** 2))
    brier_base = float(np.mean((r["base_rate"].values - up) ** 2))

    # Daily-rebalanced long/flat: at each bar's close go long for the next day
    # if prob_up > 0.5. Uses only non-overlapping 1-day P&L, so the equity curve
    # is honest even though the h-day forecasts overlap.
    cost = (slippage_bps + commission_bps) / 10_000.0
    pos = model_up.astype(float)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * r["next_day"].values - flips * cost
    bh = r["next_day"].values
    years = n / 252

    def cagr(x: np.ndarray) -> float:
        return float(math.exp(x.sum() / years) - 1) if years > 0 else 0.0

    def sharpe(x: np.ndarray) -> float:
        return float(np.sqrt(252) * x.mean() / (x.std() or 1e-9))

    return {
        "n_predictions": n,
        "n_effective": round(n_eff, 1),
        "first_pred": r.index[0].date().isoformat(),
        "last_pred": r.index[-1].date().isoformat(),
        "directional_accuracy": {
            "model": round(dir_model, 3),
            "std_error": round(math.sqrt(max(dir_model * (1 - dir_model), 1e-9) / n_eff), 3),
            "always_up": round(acc(np.ones(n, dtype=bool)), 3),
            "naive_trailing": round(acc(r["naive"].values > 0), 3),
            "hindsight_majority": round(acc(np.full(n, hindsight_major)), 3),
            "pct_predicted_up": round(float(model_up.mean()), 3),
        },
        "mae_logret": {
            "model": round(float(np.mean(np.abs(r["q50"].values - y))), 4),
            "zero": round(float(np.mean(np.abs(y))), 4),
            "drift": round(float(np.mean(np.abs(r["drift"].values - y))), 4),
            "naive_trailing": round(float(np.mean(np.abs(r["naive"].values - y))), 4),
        },
        "pinball_loss": {
            "model": round(float(pin_model), 5),
            "climatology": round(float(pin_clim), 5),
            "skill_vs_climatology": round(float(1 - pin_model / pin_clim), 4),
        },
        "interval_p10_p90": {
            "coverage": round(float(((y >= r["q10"].values) & (y <= r["q90"].values)).mean()), 3),
            "mean_width": round(float(np.mean(r["q90"].values - r["q10"].values)), 4),
            "climatology_coverage": round(float(((y >= r["clim_q10"].values) & (y <= r["clim_q90"].values)).mean()), 3),
            "climatology_width": round(float(np.mean(r["clim_q90"].values - r["clim_q10"].values)), 4),
        },
        "brier": {
            "model": round(brier_model, 4),
            "base_rate": round(brier_base, 4),
            "skill": round(1 - brier_model / brier_base, 4) if brier_base else 0.0,
        },
        "information_coefficient": round(_spearman(r["q50"].values, y), 4),
        "long_flat_daily_vs_bh": {
            "cagr_pct_model": round(cagr(strat) * 100, 2),
            "cagr_pct_buy_hold": round(cagr(bh) * 100, 2),
            "sharpe_model": round(sharpe(strat), 2),
            "sharpe_buy_hold": round(sharpe(bh), 2),
            "time_in_market": round(float(pos.mean()), 3),
            "n_position_flips": int(flips.sum()),
            "cost_bps_per_fill": round(slippage_bps + commission_bps, 2),
        },
        "actual_up_rate": round(float(up.mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--range", default="5y", dest="range_")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--out", default=None, help="Optional path to dump JSON report")
    parser.add_argument("--mode", choices=["xgboost", "ensemble"], default="xgboost")
    parser.add_argument("--sparse", action="store_true",
                        help="Legacy: one prediction per fold instead of every OOS bar")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="Per-fill slippage + half-spread (default 5bps)")
    parser.add_argument("--commission-bps", type=float, default=1.0,
                        help="Per-fill commission (default 1bp)")
    args = parser.parse_args()

    res = backtest(args.symbol.upper(), args.horizon, args.range_, args.warmup, args.step,
                   mode=args.mode, dense=not args.sparse,
                   slippage_bps=args.slippage_bps, commission_bps=args.commission_bps)
    print()
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
