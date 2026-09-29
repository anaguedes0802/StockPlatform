"""XGBoost forecaster: predicts the future log-return distribution for a given
horizon. The distribution is a volatility-scaled historical-drift core with a
gated XGBoost overlay (median + direction heads) whose weight is set out of
sample — see ml/calibration.py for why. SHAP explains the overlay when it is on.

This trains on-demand if no cached model exists for (symbol, horizon).
For production: schedule offline training, persist to model registry, load here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from app.config import settings
from app.ml import features as feat_mod
from app.ml.base import ForecastResult, Forecaster
from app.ml.calibration import fit_calibration, horizon_sigma


# Bump whenever features, hyperparameters or calibration change in a way that
# makes previously pickled bundles wrong. load() rejects other versions, so the
# next request retrains instead of serving a stale model forever.
MODEL_VERSION = 2

# Strongly regularized on purpose: daily-return targets are mostly noise and
# consecutive h-day labels overlap, so the effective sample is ~rows / h. The
# old depth-5 / 300-tree setup fit that noise (see BACKTEST_RESULTS.md, v5).
BASE_PARAMS: dict = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=30,
    reg_lambda=5.0,
    tree_method="hist",
    random_state=42,
    n_jobs=2,
)


@dataclass
class _Bundle:
    median: xgb.XGBRegressor
    direction: xgb.XGBClassifier
    feature_cols: list[str]
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    n_train_rows: int = 0
    last_train_bar_ts: str | None = None
    # Columns where training distribution was effectively zero (variance < 1e-9).
    # At inference we MUST NOT broadcast live values onto these — the model
    # never learned what non-zero means for them. Prevents covariate shift.
    zero_variance_cols: list[str] = field(default_factory=list)
    # Out-of-sample calibration (see ml/calibration.py), stored as a plain dict
    # so the bundle stays picklable across versions.
    calibration: dict = field(default_factory=dict)
    horizon: int = 0
    version: int = MODEL_VERSION


class XGBForecaster(Forecaster):
    name = "xgboost"

    def __init__(self, use_tuned_params: bool = True) -> None:
        self.bundle: _Bundle | None = None
        self._trained = False
        self._inference_symbol: str | None = None  # set by ensemble.py before predict()
        self._training_symbol: str | None = None   # if set, enables PIT-history join at fit()
        # When False, fit() will NOT auto-load Optuna-tuned hyperparameters and
        # uses only the fixed base_params below. The walk-forward backtest sets
        # this to False so that tuned params (chosen on the *full* date range,
        # including each fold's own future) cannot leak into earlier folds.
        self.use_tuned_params = use_tuned_params
        # Populated by fit() when an eval split is requested: out-of-sample
        # directional-accuracy / MAE measured on a held-out chronological tail.
        self.eval_metrics: dict | None = None

    # ---------- training ----------

    def fit(
        self,
        df: pd.DataFrame,
        target_horizon_days: int,
        eval_tail_frac: float = 0.0,
    ) -> None:
        # Training: join PIT-historical features if a training symbol is set
        # (backtest-safe — fundamentals are filing-lagged, news sentiment comes
        # from Alpha Vantage with real per-article timestamps).
        feats = feat_mod.build_features(
            df,
            symbol=self._training_symbol,
            pit_history=bool(self._training_symbol),
        )
        target_col = f"y_h{target_horizon_days}"
        if target_col not in feats.columns:
            raise ValueError(f"missing target {target_col}")
        feature_cols = feat_mod.feature_columns(feats, stationary_only=True)
        train = feats.dropna(subset=feature_cols + [target_col])
        if len(train) < 60:
            raise ValueError(f"not enough data to train ({len(train)} rows)")

        X = train[feature_cols].values
        y = train[target_col].values

        base_params: dict = dict(BASE_PARAMS)

        # Optuna-tuned overrides — applied automatically if a tuning file exists,
        # UNLESS use_tuned_params is False. Disabling this is required for an
        # honest walk-forward backtest: tune_hyperparams tunes on the full date
        # range, so loading those params into an earlier fold leaks the fold's
        # own future. The backtest constructs forecasters with this OFF.
        if self.use_tuned_params and self._training_symbol:
            try:
                from app.ml.tuning import load_best_params
                tuned = load_best_params(self._training_symbol, target_horizon_days)
                if tuned:
                    base_params.update(tuned)
            except Exception:
                pass

        def fit_heads(X_: np.ndarray, y_: np.ndarray) -> tuple[xgb.XGBRegressor, xgb.XGBClassifier]:
            med = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=0.5, **base_params
            ).fit(X_, y_)
            dirn = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                **{k: v for k, v in base_params.items() if k != "n_jobs"},
            ).fit(X_, (y_ > 0).astype(int))
            return med, dirn

        # ---- honest out-of-sample eval on a held-out chronological tail ----
        # Before fitting the production models on ALL rows, optionally measure
        # directional accuracy / MAE on a tail split that the eval models never
        # saw. We train *separate* throwaway models on the head only — no
        # shuffling (this is a time series) and the result never feeds back into
        # the production fit, so there is no leakage. Distinct from the calibration
        # tail in ml/calibration.py, which sets the overlay weights.
        self.eval_metrics = None
        if eval_tail_frac and 0.0 < eval_tail_frac < 1.0 and len(train) >= 80:
            self.eval_metrics = self._evaluate_tail(
                X, y, base_params, eval_tail_frac
            )

        # Core distribution + overlay weights, chosen on a purged held-out tail.
        cal = fit_calibration(
            X, y, horizon_sigma(train, target_horizon_days), target_horizon_days, fit_heads,
        )
        median, direction = fit_heads(X, y)

        # Identify columns that were effectively constant during training.
        # The model can't have learned non-zero behaviour for these; at inference
        # we'll suppress any live broadcast attempting to inject non-zero values
        # onto them. This eliminates the train/inference covariate shift.
        col_var = np.var(X, axis=0)
        zero_var_cols = [feature_cols[i] for i, v in enumerate(col_var) if v < 1e-9]

        self.bundle = _Bundle(
            median=median, direction=direction,
            feature_cols=feature_cols,
            n_train_rows=len(train),
            last_train_bar_ts=train.index[-1].isoformat() if not train.empty else None,
            zero_variance_cols=zero_var_cols,
            calibration=cal.to_dict(),
            horizon=target_horizon_days,
        )
        self._trained = True

    def _evaluate_tail(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_params: dict,
        eval_tail_frac: float,
    ) -> dict:
        """Chronological train/validation split for an honest OOS readout.

        Train throwaway median + direction heads on the head (oldest rows),
        evaluate directional accuracy and MAE on the most recent tail. No
        random shuffling; the production models (fit on all rows) are untouched,
        so nothing here leaks into training.
        """
        n = len(y)
        n_val = max(1, int(round(n * eval_tail_frac)))
        n_tr = n - n_val
        if n_tr < 40 or n_val < 5:
            return {"ok": False, "reason": "tail split too small", "n_val": n_val}

        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_v, y_v = X[n_tr:], y[n_tr:]

        eval_median = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.5, **base_params
        )
        eval_dir = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            **{k: v for k, v in base_params.items() if k != "n_jobs"},
        )
        eval_median.fit(X_tr, y_tr)
        eval_dir.fit(X_tr, (y_tr > 0).astype(int))

        pred_med = eval_median.predict(X_v)
        prob_up = eval_dir.predict_proba(X_v)[:, 1]
        actual_up = (y_v > 0).astype(int)
        model_up = (prob_up > 0.5).astype(int)

        mae = float(np.mean(np.abs(pred_med - y_v)))
        dir_acc = float(np.mean(model_up == actual_up))
        return {
            "ok": True,
            "n_train": int(n_tr),
            "n_val": int(n_val),
            "oos_directional_accuracy": round(dir_acc, 4),
            "oos_mae_logret": round(mae, 6),
            "val_up_rate": round(float(actual_up.mean()), 4),
        }

    # ---------- prediction ----------

    def predict_logrets(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Vectorised forecast for every complete row of a prebuilt feature frame.

        Returns a frame indexed like the usable rows of `feats` with columns
        q10 / q25 / q50 / q75 / q90 (h-day log-returns) and prob_up.
        `predict()` uses this for the latest bar; the walk-forward backtest uses
        it to score every out-of-sample bar with exactly the production math.
        """
        if not self.bundle:
            raise RuntimeError("model not trained")
        feature_cols = self.bundle.feature_cols
        feats = feats.copy()
        # Suppress live broadcast on zero-variance columns
        for col in self.bundle.zero_variance_cols:
            if col in feats.columns:
                feats[col] = 0.0
        valid = feats[feature_cols].dropna()
        if valid.empty:
            raise ValueError("no usable feature row")
        cal = self.bundle.calibration
        X = valid.values
        sigma = horizon_sigma(feats.loc[valid.index], self.bundle.horizon)
        anchor, lam = cal["anchor"], cal["median_weight"]
        base, kappa = cal["base_rate"], cal["direction_weight"]

        q50 = np.full(len(valid), anchor)
        if lam > 0:
            q50 = anchor + lam * (self.bundle.median.predict(X).astype(float) - anchor)
        prob_up = np.full(len(valid), base)
        if kappa > 0:
            prob_up = base + kappa * (self.bundle.direction.predict_proba(X)[:, 1] - base)

        out = {"q50": q50, "prob_up": prob_up}
        for k, z in cal["z"].items():
            out[k] = q50 + z * sigma
        return pd.DataFrame(out, index=valid.index)

    def predict(self, df: pd.DataFrame, target_horizon_days: int) -> ForecastResult:
        if not self.bundle:
            raise RuntimeError("model not trained")
        # Inference: join live news sentiment + fundamentals on the most recent
        # bars so the prediction reflects today's information set — BUT only on
        # columns where training had real variance. Otherwise we'd be feeding
        # the model a feature value it has never seen, which is covariate shift.
        # pit_history must match fit(): without it the SMC / insider / PIT
        # fundamentals columns are all zero at inference while training saw
        # real values — a silent train/serve skew.
        feats = feat_mod.build_features(
            df, symbol=self._inference_symbol, join_live_signals=bool(self._inference_symbol),
            pit_history=bool(self._inference_symbol),
        )
        feature_cols = self.bundle.feature_cols
        # take the latest row that has all features present
        row = self.predict_logrets(feats).iloc[-1]
        x = feats.loc[[row.name], feature_cols].values
        for i, col in enumerate(feature_cols):
            if col in self.bundle.zero_variance_cols:
                x[0, i] = 0.0

        median_logret = float(row["q50"])
        lower_logret = float(row["q10"])
        upper_logret = float(row["q90"])
        prob_up = float(row["prob_up"])

        last_price = float(df["close"].iloc[-1])
        point = last_price * float(np.exp(median_logret))
        p10 = last_price * float(np.exp(lower_logret))
        p90 = last_price * float(np.exp(upper_logret))

        p25 = last_price * float(np.exp(float(row["q25"])))
        p75 = last_price * float(np.exp(float(row["q75"])))

        sigma_ret = abs(upper_logret - lower_logret) / (2 * 1.2816) if upper_logret != lower_logret else 0.05
        expected_vol_pct = float(sigma_ret * 100)

        # confidence: inverse of normalized interval width, clipped
        rel_width = abs(upper_logret - lower_logret) / max(abs(median_logret), 0.01 + 1e-9)
        confidence = float(np.clip(1.0 / (1.0 + rel_width), 0.1, 0.95))

        contributions, drivers = self._explain(x, feature_cols)
        if self.bundle.calibration.get("median_weight", 0.0) <= 0:
            # The overlay earned no weight, so SHAP would explain a model that
            # isn't driving the number. Say what actually is.
            contributions = {"technical": 1.0}
            drivers = [
                {"name": f"Historical median {self.bundle.horizon}-day return",
                 "direction": "up" if median_logret > 0 else "down", "weight": 0.5},
                {"name": "Current realized volatility (band width)",
                 "direction": "neutral", "weight": 0.5},
            ]

        return ForecastResult(
            point=point,
            p10=p10, p25=p25, p50=point, p75=p75, p90=p90,
            direction_prob_up=prob_up,
            expected_volatility_pct=expected_vol_pct,
            confidence=confidence,
            contributions=contributions,
            drivers=drivers,
        )

    # ---------- SHAP explainability ----------

    def _explain(self, x: np.ndarray, feature_cols: list[str]) -> tuple[dict[str, float], list[dict]]:
        if not self.bundle:
            return {}, []
        explainer = shap.TreeExplainer(self.bundle.median)
        shap_values = explainer.shap_values(x)[0]  # shape: (n_features,)

        # bucket by category
        category_sums: dict[str, float] = {}
        for col, sv in zip(feature_cols, shap_values, strict=True):
            cat = feat_mod.category_of(col)
            category_sums[cat] = category_sums.get(cat, 0.0) + abs(float(sv))
        total = sum(category_sums.values()) or 1.0
        contributions = {k: round(v / total, 4) for k, v in category_sums.items()}

        # top drivers
        ranked = sorted(
            zip(feature_cols, shap_values, strict=True),
            key=lambda t: -abs(t[1]),
        )[:6]
        drivers = [
            {
                "name": _humanize(col),
                "direction": "up" if sv > 0 else ("down" if sv < 0 else "neutral"),
                "weight": round(abs(float(sv)) / total, 4),
            }
            for col, sv in ranked
        ]
        return contributions, drivers

    # ---------- persistence ----------

    @staticmethod
    def _model_path(symbol: str, horizon: int) -> str:
        os.makedirs(settings.model_dir, exist_ok=True)
        return os.path.join(settings.model_dir, f"xgb_{symbol}_{horizon}d.joblib")

    def save(self, symbol: str, horizon: int) -> str:
        if not self.bundle:
            raise RuntimeError("not trained")
        path = self._model_path(symbol, horizon)
        joblib.dump(self.bundle, path)
        return path

    def load(self, symbol: str, horizon: int) -> bool:
        path = self._model_path(symbol, horizon)
        if not os.path.exists(path):
            return False
        bundle = joblib.load(path)
        if getattr(bundle, "version", 1) != MODEL_VERSION:
            return False  # stale schema/calibration — caller retrains and overwrites
        self.bundle = bundle
        self._trained = True
        return True


def _humanize(col: str) -> str:
    mapping = {
        "ret_1": "1-day return",
        "ret_5": "5-day return",
        "ret_21": "1-month return",
        "ret_63": "3-month return",
        "vol_5": "5-day realized volatility",
        "vol_21": "1-month realized volatility",
        "vol_63": "3-month realized volatility",
        "rsi_14": "RSI(14)",
        "macd": "MACD",
        "macd_hist": "MACD histogram",
        "bb_pctb": "Bollinger %B",
        "atr_14": "ATR(14)",
        "atr_pct": "ATR as % of price",
        "vol_z_20": "Volume z-score (20d)",
        "drawdown": "Drawdown from peak",
    }
    if col in mapping:
        return mapping[col]
    if col.startswith("close_over_sma_"):
        return f"Close vs {col.split('_')[-1]}-day SMA"
    if col.startswith("sma_"):
        return f"{col.split('_')[-1]}-day SMA"
    return col.replace("_", " ")
