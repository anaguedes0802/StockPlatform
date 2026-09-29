"""AutoARIMA forecaster powered by Nixtla StatsForecast.

Classical methods often beat deep models on noisy financial series, especially
at short-to-mid horizons where the signal-to-noise ratio is brutal.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.ml.base import ForecastResult, Forecaster


def _statsforecast_available() -> bool:
    try:
        import statsforecast  # noqa: F401
        return True
    except Exception:
        return False


def _cumulative(mean: np.ndarray, lo80: np.ndarray, hi80: np.ndarray) -> tuple[float, float]:
    """Aggregate per-step return forecasts to an h-day (mean, sigma).

    Means add. Standard deviations do NOT: summing the per-step 80% bounds
    scales the band by h instead of sqrt(h), which made the old 5d interval
    ~2.2x and the 21d interval ~4.6x too wide (that was the "100% coverage" in
    the v2 backtest). Treat per-step errors as ~uncorrelated — true to first
    order for daily returns, whose fitted ARMA terms are tiny — and add
    variances.
    """
    mean_cum = float(np.sum(mean))
    step_sigma = (np.asarray(hi80, dtype=float) - np.asarray(lo80, dtype=float)) / (2 * 1.2816)
    sigma = float(np.sqrt(np.sum(step_sigma ** 2)))
    return mean_cum, sigma


class AutoARIMAForecaster(Forecaster):
    """AutoARIMA on log-returns, with native prediction intervals."""

    name = "arima"

    def __init__(self) -> None:
        self._fitted: dict | None = None  # store input log-returns; refit at predict
        self._trained = False

    @staticmethod
    def available() -> bool:
        return _statsforecast_available()

    def fit(self, df: pd.DataFrame, target_horizon_days: int) -> None:
        if not _statsforecast_available():
            raise RuntimeError("statsforecast not installed")
        log_ret = np.log(df["close"]).diff().dropna()
        # Use the last 500 observations — AutoARIMA is O(n^2) in parameter search.
        if len(log_ret) > 500:
            log_ret = log_ret.iloc[-500:]
        self._fitted = {"log_ret": log_ret.values.astype(float), "h": target_horizon_days}
        self._trained = True

    def predict(self, df: pd.DataFrame, target_horizon_days: int) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("not trained")
        from statsforecast.models import AutoARIMA

        log_ret = self._fitted["log_ret"]
        h = target_horizon_days

        model = AutoARIMA(season_length=1, max_p=3, max_q=3, max_d=1, stepwise=True)
        model.fit(log_ret)
        fc = model.predict(h=h, level=[80])
        # statsforecast returns dict with 'mean' (and 'lo-80', 'hi-80')
        mean = np.asarray(fc["mean"])
        lo80 = np.asarray(fc.get("lo-80", mean))
        hi80 = np.asarray(fc.get("hi-80", mean))

        mean_cum, sigma_ret = _cumulative(mean, lo80, hi80)
        lo_cum = mean_cum - 1.2816 * sigma_ret
        hi_cum = mean_cum + 1.2816 * sigma_ret

        last_price = float(df["close"].iloc[-1])
        point = last_price * math.exp(mean_cum)
        p10 = last_price * math.exp(lo_cum)
        p90 = last_price * math.exp(hi_cum)
        # symmetric inner quantiles
        p25 = last_price * math.exp(mean_cum - 0.6745 * sigma_ret)
        p75 = last_price * math.exp(mean_cum + 0.6745 * sigma_ret)

        # direction probability from gaussian assumption on cumulative mean
        sigma_ret = sigma_ret if sigma_ret > 0 else 0.01
        prob_up = float(0.5 * (1 + math.erf(mean_cum / (sigma_ret * math.sqrt(2)))))
        prob_up = max(0.05, min(0.95, prob_up))

        rel_width = abs(hi_cum - lo_cum) / max(abs(mean_cum), 0.01 + 1e-9)
        confidence = float(np.clip(1.0 / (1.0 + rel_width), 0.1, 0.95))

        return ForecastResult(
            point=point,
            p10=p10, p25=p25, p50=point, p75=p75, p90=p90,
            direction_prob_up=prob_up,
            expected_volatility_pct=float(sigma_ret * 100),
            confidence=confidence,
            contributions={"technical": 1.0, "fundamental": 0.0, "sentiment_news": 0.0,
                           "sentiment_social": 0.0, "macro": 0.0, "events": 0.0},
            drivers=[
                {"name": "AutoARIMA seasonal/trend component", "direction": "up" if mean_cum > 0 else "down",
                 "weight": min(1.0, abs(mean_cum) / max(sigma_ret, 1e-6))},
            ],
        )

    def predict_logrets(self, close: pd.Series, positions: list[int], horizon: int) -> pd.DataFrame:
        """Backtest helper: fit AutoARIMA once on the training returns, then apply
        the fitted model (no re-estimation) at each later bar in `positions`.
        Returns q10/q50/q90/prob_up in h-day log-return space, indexed by bar ts.
        """
        if not self._fitted:
            raise RuntimeError("not trained")
        from statsforecast.models import AutoARIMA

        model = AutoARIMA(season_length=1, max_p=3, max_q=3, max_d=1, stepwise=True)
        model.fit(self._fitted["log_ret"])
        log_ret_all = np.log(close.astype(float)).diff().values
        rows = []
        for pos in positions:
            hist = log_ret_all[max(1, pos + 1 - 500): pos + 1]
            fc = model.forward(y=hist, h=horizon, level=[80])
            mean = np.asarray(fc["mean"])
            mu, sigma = _cumulative(mean, np.asarray(fc.get("lo-80", mean)), np.asarray(fc.get("hi-80", mean)))
            sigma = sigma if sigma > 0 else 0.01
            prob_up = max(0.05, min(0.95, 0.5 * (1 + math.erf(mu / (sigma * math.sqrt(2))))))
            rows.append((mu - 1.2816 * sigma, mu, mu + 1.2816 * sigma, prob_up))
        return pd.DataFrame(rows, columns=["q10", "q50", "q90", "prob_up"], index=close.index[positions])

    # AutoARIMA has tiny state — we just stash the input. No on-disk persistence needed.
    def save(self, *_args, **_kwargs) -> str: return ""
    def load(self, *_args, **_kwargs) -> bool: return False
