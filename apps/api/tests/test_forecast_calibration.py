from __future__ import annotations

import math

import numpy as np
import xgboost as xgb

from app.ml.arima_model import _cumulative
from app.ml.calibration import fit_calibration
from app.ml.xgboost_model import BASE_PARAMS, MODEL_VERSION, XGBForecaster


def _fit_heads(X, y):
    med = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE_PARAMS).fit(X, y)
    dirn = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE_PARAMS).fit(
        X, (y > 0).astype(int)
    )
    return med, dirn


def test_overlay_gets_no_weight_on_pure_noise():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 6))
    y = 0.001 + 0.02 * rng.standard_normal(n)
    cal = fit_calibration(X, y, np.full(n, 0.02), horizon=5, fit_heads=_fit_heads)
    assert cal.median_weight == 0.0
    assert cal.direction_weight == 0.0
    assert abs(cal.anchor - np.median(y)) < 1e-12
    # standardized residual quantiles ~ N(0,1) quantiles for Gaussian noise
    assert -1.5 < cal.z["q10"] < -1.1 and 1.1 < cal.z["q90"] < 1.5


def test_overlay_earns_weight_when_features_carry_signal():
    rng = np.random.default_rng(1)
    n = 900
    X = rng.normal(size=(n, 6))
    y = 0.02 * X[:, 0] + 0.005 * rng.standard_normal(n)
    cal = fit_calibration(X, y, np.full(n, 0.02), horizon=5, fit_heads=_fit_heads)
    assert cal.median_weight >= 0.5
    assert cal.direction_weight > 0.0


def test_short_history_falls_back_to_core_only():
    rng = np.random.default_rng(2)
    n = 150
    X = rng.normal(size=(n, 3))
    y = 0.02 * X[:, 0]
    cal = fit_calibration(X, y, np.full(n, 0.02), horizon=21, fit_heads=_fit_heads)
    assert cal.median_weight == 0.0 and cal.n_calibration == 0


def test_arima_horizon_sigma_scales_with_sqrt_h():
    h, step_sigma = 21, 0.01
    mean = np.zeros(h)
    half = 1.2816 * step_sigma
    mu, sigma = _cumulative(mean, mean - half, mean + half)
    assert mu == 0.0
    assert math.isclose(sigma, step_sigma * math.sqrt(h), rel_tol=1e-9)


class _OldBundle:
    version = MODEL_VERSION - 1


def test_stale_bundle_version_is_rejected(tmp_path, monkeypatch):
    import joblib

    from app.config import settings

    monkeypatch.setattr(settings, "model_dir", str(tmp_path))

    joblib.dump(_OldBundle(), XGBForecaster._model_path("TEST", 5))
    assert XGBForecaster().load("TEST", 5) is False
