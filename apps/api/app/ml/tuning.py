"""Per-symbol Bayesian hyperparameter tuning for XGBForecaster via Optuna.

Walk-forward CV (no look-ahead): train on `[0, split_i)`, validate on
`[split_i, split_i + val_size)`. Average the validation objective across N
splits, then minimize.

Objective combines three terms:
  - **CRPS** approx: mean absolute error between predicted quantiles and the
    realized return, averaged across p10/p50/p90 — a proper scoring rule that
    rewards both accuracy and calibration.
  - **Directional log-loss**: −mean(log p_correct_class) on the direction
    classifier, encouraging probabilistic separation.
  - **Interval-width penalty**: small penalty on (p90 − p10) to avoid
    pathologically wide bands that trivially achieve coverage.

Saves best params + study metadata to:
  apps/api/artifacts/tuning/{symbol}_{horizon}d.json

XGBForecaster.fit() reads this file if it exists and overrides the
default hyperparameters before training.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import features as feat_mod


def _study_path(symbol: str, horizon: int) -> str:
    d = os.path.join(settings.model_dir, "..", "tuning")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{symbol}_{horizon}d.json")


def load_best_params(symbol: str, horizon: int) -> dict[str, Any] | None:
    p = _study_path(symbol, horizon)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("best_params")
    except Exception:
        return None


def _walk_forward_splits(n: int, n_splits: int = 4, val_size: int = 120) -> list[tuple[int, int]]:
    """Generate (train_end, val_end) index pairs.

    Reserves a warmup of 300 bars; each fold validates a 120-bar window after
    the train_end. Splits are expanding-window: every subsequent fold's train
    set grows.
    """
    warmup = 300
    if n < warmup + val_size + n_splits * 30:
        # Fall back to whatever we have
        n_splits = max(2, (n - warmup) // val_size)
    out = []
    span = n - warmup - val_size
    if n_splits <= 0 or span <= 0:
        return [(n - val_size, n)]
    step = span // n_splits
    for i in range(n_splits):
        train_end = warmup + i * step
        val_end = train_end + val_size
        out.append((train_end, min(val_end, n)))
    return out


@dataclass
class _Score:
    crps: float
    direction_logloss: float
    interval_width: float

    def total(self) -> float:
        # Lower is better. Width gets a soft penalty so it doesn't dominate.
        return self.crps + 0.3 * self.direction_logloss + 0.05 * self.interval_width


def _train_and_score(
    df: pd.DataFrame,
    horizon: int,
    symbol: str,
    params: dict[str, Any],
    splits: list[tuple[int, int]],
) -> _Score:
    """Fit XGBoost quantile heads + direction head on each split's train window;
    compute CRPS + direction logloss + interval width on the held-out window."""
    import xgboost as xgb

    feats = feat_mod.build_features(df, symbol=symbol, pit_history=True)
    target = f"y_h{horizon}"
    if target not in feats.columns:
        raise ValueError(f"missing target {target}")
    # Same stationary column set XGBForecaster.fit() trains on.
    feature_cols = feat_mod.feature_columns(feats, stationary_only=True)

    full = feats.dropna(subset=feature_cols + [target])
    if len(full) < 200:
        raise ValueError(f"not enough rows ({len(full)})")

    crps_terms: list[float] = []
    dir_terms: list[float] = []
    width_terms: list[float] = []

    for train_end, val_end in splits:
        train_end = min(train_end, len(full))
        val_end = min(val_end, len(full))
        if val_end - train_end < 5:
            continue
        train = full.iloc[:train_end]
        val = full.iloc[train_end:val_end]
        if train.empty or val.empty:
            continue
        X_tr, y_tr = train[feature_cols].values, train[target].values
        X_v,  y_v = val[feature_cols].values,   val[target].values

        common = {
            "n_estimators":   int(params["n_estimators"]),
            "max_depth":      int(params["max_depth"]),
            "learning_rate":  float(params["learning_rate"]),
            "subsample":      float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "reg_lambda":     float(params["reg_lambda"]),
            "tree_method":    "hist",
            "random_state":   42,
            "n_jobs":         2,
            "verbosity":      0,
        }
        m_q50 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **common)
        m_q10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1, **common)
        m_q90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9, **common)
        m_dir = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  **{k: v for k, v in common.items() if k != "n_jobs"})
        m_q50.fit(X_tr, y_tr)
        m_q10.fit(X_tr, y_tr)
        m_q90.fit(X_tr, y_tr)
        m_dir.fit(X_tr, (y_tr > 0).astype(int))

        p50 = m_q50.predict(X_v)
        p10 = m_q10.predict(X_v)
        p90 = m_q90.predict(X_v)
        prob_up = m_dir.predict_proba(X_v)[:, 1]

        # CRPS approximation: average pinball loss across 3 quantiles
        def pinball(pred: np.ndarray, target: np.ndarray, q: float) -> float:
            diff = target - pred
            return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
        crps = (pinball(p10, y_v, 0.1) + pinball(p50, y_v, 0.5) + pinball(p90, y_v, 0.9)) / 3
        crps_terms.append(crps)

        actual_up = (y_v > 0).astype(int)
        # logloss with clipping
        eps = 1e-6
        prob_up_clipped = np.clip(prob_up, eps, 1 - eps)
        logloss = -np.mean(actual_up * np.log(prob_up_clipped) + (1 - actual_up) * np.log(1 - prob_up_clipped))
        dir_terms.append(float(logloss))

        width_terms.append(float(np.mean(p90 - p10)))

    if not crps_terms:
        raise ValueError("no successful CV splits")
    return _Score(
        crps=float(np.mean(crps_terms)),
        direction_logloss=float(np.mean(dir_terms)),
        interval_width=float(np.mean(width_terms)),
    )


def tune(
    symbol: str,
    df: pd.DataFrame,
    horizon: int,
    n_trials: int = 30,
    n_splits: int = 4,
    val_size: int = 120,
    timeout_seconds: int | None = 600,
) -> dict[str, Any]:
    """Run Optuna optimization. Returns a dict with best_params, best_score, n_trials, elapsed."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    splits = _walk_forward_splits(len(df), n_splits=n_splits, val_size=val_size)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        try:
            score = _train_and_score(df, horizon, symbol, params, splits)
        except Exception as e:
            raise optuna.TrialPruned() from e
        return score.total()

    study = optuna.create_study(direction="minimize")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    summary = {
        "symbol": symbol,
        "horizon": horizon,
        "best_params": best.params,
        "best_score": best.value,
        "n_completed_trials": sum(1 for t in study.trials if t.state.is_finished()),
        "n_pruned": sum(1 for t in study.trials if str(t.state) == "TrialState.PRUNED"),
        "elapsed_seconds": round(elapsed, 1),
        "n_splits": len(splits),
        "val_size": val_size,
    }
    # Persist
    path = _study_path(symbol, horizon)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
