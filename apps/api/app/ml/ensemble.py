"""Regime-aware ensemble.

Active members:
  - XGBoost (always)
  - LSTM (when PyTorch is installed)

Stubs (still typed; implement later by adding fit/predict):
  - Transformer, TFT, Prophet, ARIMA
"""
from __future__ import annotations

import math

import pandas as pd

from app.ml.arima_model import AutoARIMAForecaster
from app.ml.base import ForecastResult, Forecaster
from app.ml.lstm_model import LSTMForecaster
from app.ml.regime import REGIME_WEIGHTS, Regime, detect  # noqa: F401  (REGIME_WEIGHTS re-exported)
from app.ml.xgboost_model import XGBForecaster


class _Stub(Forecaster):
    name = "stub"

    def fit(self, df: pd.DataFrame, target_horizon_days: int) -> None:
        self._trained = False

    def predict(self, df: pd.DataFrame, target_horizon_days: int) -> ForecastResult:
        raise NotImplementedError(f"{self.name} not implemented in v0")


class TransformerForecaster(_Stub):
    """Reference: HuggingFace `time_series_transformer`."""
    name = "transformer"


class TFTForecaster(_Stub):
    """Use Nixtla `neuralforecast.TFT`."""
    name = "tft"


class ProphetForecaster(_Stub):
    """Use prophet library or Nixtla `statsforecast.AutoARIMA`."""
    name = "prophet"


# AutoARIMA is now real (see arima_model.py); kept here only as a typed re-export.
ARIMAForecaster = AutoARIMAForecaster


# ---- mix tables ----
#
# Weights per regime. Components missing at runtime (e.g. torch not installed) are
# dropped and the remaining weights renormalized when blending.

REGIME_MODEL_MIX: dict[Regime, dict[str, float]] = {
    "bull_low_vol":  {"xgboost": 0.50, "lstm": 0.25, "arima": 0.25},
    "bull_high_vol": {"xgboost": 0.40, "lstm": 0.35, "arima": 0.25},
    "bear_low_vol":  {"xgboost": 0.50, "lstm": 0.20, "arima": 0.30},
    "bear_high_vol": {"xgboost": 0.35, "lstm": 0.35, "arima": 0.30},
    "sideways":      {"xgboost": 0.45, "lstm": 0.20, "arima": 0.35},
}


def _load_meta_weights() -> dict[Regime, dict[str, float]] | None:
    """Load fitted regime weights from scripts/fit_ensemble_weights.py.
    Returns None if the file doesn't exist (fall back to hand-tuned table)."""
    import json
    import os
    from app.config import settings as _settings
    path = os.path.join(_settings.model_dir, "..", "ensemble", "regime_weights.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data.get("ok"):
            return None
        return data.get("regimes") or None
    except Exception:
        return None


# Override hand-tuned weights with empirically-fit ones if available.
_META = _load_meta_weights()
if _META:
    for regime_name, weights in _META.items():
        if regime_name in REGIME_MODEL_MIX and weights:
            REGIME_MODEL_MIX[regime_name] = weights


def _blend(results: list[tuple[str, ForecastResult, float]]) -> ForecastResult:
    """Weight-blend N forecast results. Inputs: [(name, result, weight), ...].
    Weights are renormalized to sum to 1 across the present results.

    Price quantiles (p10/p25/p50/p75/p90 and point) are blended in LOG-RETURN
    space, not in raw price levels: each model's quantile is converted to a
    log-return relative to that model's own p50 (its implied last price drops
    out), weight-averaged, then converted back to a price level around the
    weight-averaged median. This is the statistically correct way to combine
    return forecasts — a plain mean of price levels biases the blend toward the
    higher-priced model's quantiles and distorts interval symmetry.
    direction_prob_up, expected_volatility_pct and confidence stay simple
    weighted (arithmetic) means.
    """
    if not results:
        raise ValueError("no forecasters produced a result")
    total = sum(w for _, _, w in results) or 1.0
    norm = [(n, r, w / total) for n, r, w in results]

    def lerp(field: str) -> float:
        return float(sum(getattr(r, field) * w for _, r, w in norm))

    # --- log-return-space blend of price quantiles ---
    # Recover a shared reference last price: all members share the same input
    # bars, so each member's last_price == p50 / exp(median_logret). We don't
    # know median_logret directly, but the weighted GEOMETRIC mean of the p50s
    # equals last_price * exp(weighted_mean(median_logret)) — exactly the
    # log-return blend of the median. Band offsets are blended as ln(q / p50).
    def _safe_ln(num: float, den: float) -> float:
        if num <= 0.0 or den <= 0.0:
            return 0.0
        return math.log(num / den)

    blended_p50 = math.exp(sum(_safe_ln(r.p50, 1.0) * w for _, r, w in norm))

    def blend_quantile(field: str) -> float:
        # weighted mean of each model's log offset of this quantile from its p50
        offset = sum(_safe_ln(getattr(r, field), r.p50) * w for _, r, w in norm)
        return float(blended_p50 * math.exp(offset))

    keys: set[str] = set()
    for _, r, _ in norm:
        keys |= set(r.contributions)
    contributions = {
        k: round(sum(r.contributions.get(k, 0.0) * w for _, r, w in norm), 4)
        for k in keys
    }
    drivers: list[dict] = []
    for _, r, _ in norm:
        drivers.extend(r.drivers[:3])
    return ForecastResult(
        point=float(blended_p50),
        p10=blend_quantile("p10"), p25=blend_quantile("p25"), p50=float(blended_p50),
        p75=blend_quantile("p75"), p90=blend_quantile("p90"),
        direction_prob_up=lerp("direction_prob_up"),
        expected_volatility_pct=lerp("expected_volatility_pct"),
        confidence=lerp("confidence"),
        contributions=contributions,
        drivers=drivers[:6],
    )


def ensemble_forecast(
    df: pd.DataFrame,
    horizon_days: int,
    symbol: str,
) -> tuple[ForecastResult, Regime, dict[str, float], dict]:
    """Single entry point for AI forecast endpoint.

    Trains/loads each available model, predicts, then weight-blends per the regime
    mix table. Models that fail at predict-time are dropped and remaining weights
    are renormalized.

    Returns (forecast, regime, model_mix_actually_used, freshness).
    """
    regime = detect(df["close"])
    target_mix = dict(REGIME_MODEL_MIX[regime])
    freshness: dict = {}
    results: list[tuple[str, ForecastResult, float]] = []

    # --- XGBoost (always)
    try:
        xgb = XGBForecaster()
        if not xgb.load(symbol, horizon_days):
            xgb._training_symbol = symbol  # enables PIT-history join at fit time
            xgb.fit(df, horizon_days)
            try: xgb.save(symbol, horizon_days)
            except Exception: pass
        xgb._inference_symbol = symbol  # enables live news/fundamentals join in predict()
        xgb_pred = xgb.predict(df, horizon_days)
        results.append(("xgboost", xgb_pred, target_mix.get("xgboost", 0.5)))
        if xgb.bundle:
            freshness["xgboost"] = {
                "trained_at": xgb.bundle.trained_at.isoformat() if xgb.bundle.trained_at else None,
                "n_train_rows": xgb.bundle.n_train_rows,
                "last_train_bar_ts": xgb.bundle.last_train_bar_ts,
            }
    except Exception as e:
        freshness["xgboost"] = {"error": str(e)}

    # --- LSTM (if torch available)
    if LSTMForecaster.available():
        try:
            lstm = LSTMForecaster()
            if not lstm.load(symbol, horizon_days):
                lstm.fit(df, horizon_days)
                try: lstm.save(symbol, horizon_days)
                except Exception: pass
            lstm_pred = lstm.predict(df, horizon_days)
            results.append(("lstm", lstm_pred, target_mix.get("lstm", 0.25)))
            freshness["lstm"] = {"available": True}
        except Exception as e:
            freshness["lstm"] = {"error": str(e)}

    # --- AutoARIMA (if statsforecast available)
    if AutoARIMAForecaster.available():
        try:
            arima = AutoARIMAForecaster()
            arima.fit(df, horizon_days)
            arima_pred = arima.predict(df, horizon_days)
            results.append(("arima", arima_pred, target_mix.get("arima", 0.25)))
            freshness["arima"] = {"available": True}
        except Exception as e:
            freshness["arima"] = {"error": str(e)}

    if not results:
        raise RuntimeError("no forecasters produced a result")

    blended = _blend(results)
    actual_mix = {name: round(w / sum(r[2] for r in results), 3) for name, _, w in results}
    return blended, regime, actual_mix, freshness
