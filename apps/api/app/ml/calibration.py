"""Out-of-sample calibration for the XGBoost forecaster.

Why this exists (see BACKTEST_RESULTS.md, "v5"): on ~950 dense walk-forward
predictions per symbol, the raw XGBoost median had an information coefficient
of ~0 (often negative) against realized h-day returns, its quantile heads
covered only 49-67% of outcomes for a nominal 80% band, and its direction
classifier scored ~45% worse (Brier) than simply quoting the historical
up-rate. The model was confidently fitting noise.

The forecast is therefore built as:

    core     = volatility-scaled drift  (always on)
                 median  = median historical h-day log-return
                 q_tau   = median + z_tau * sigma_t
               where sigma_t is today's realized-vol estimate scaled to h days
               and z_tau are empirical quantiles of the standardized training
               residuals — fat tails and skew come from data, not a Gaussian.
    overlay  = XGBoost, weighted by lambda (median) and kappa (direction)

lambda and kappa are chosen on a *purged* held-out tail: throwaway models are
fit on the head, scored on the tail (with an h-bar gap so no training label
overlaps the tail), and the overlay only gets weight if it beats the core by a
statistically significant margin. With today's features it gets 0 — the
forecast degrades to the honest heteroscedastic baseline. When better features land
(per-bar news, earnings surprises, ...) the overlay earns weight automatically.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# Quantile levels the forecast exposes.
TAUS: dict[str, float] = {"q10": 0.10, "q25": 0.25, "q75": 0.75, "q90": 0.90}
# Candidate overlay weights. The best one is only used if its held-out loss
# beats the core's with t-stat >= MIN_T over non-overlapping h-bar blocks —
# consecutive h-day labels overlap, so a 250-row tail at h=21 holds only ~12
# independent observations and a raw "x% better" rule mostly selects noise.
_GRID = np.linspace(0.0, 1.0, 11)
MIN_T = 2.0
CALIB_FRAC = 0.25
MIN_HEAD_ROWS = 120      # below this the held-out fit is too noisy to trust


def horizon_sigma(feats: pd.DataFrame, horizon: int) -> np.ndarray:
    """Expected h-day log-return std per row from realized vol (annualized
    vol_21 / vol_63 blend -> h days). Floored so a quiet stretch can't
    collapse the band."""
    v = 0.5 * feats["vol_21"].to_numpy(dtype=float) + 0.5 * feats["vol_63"].to_numpy(dtype=float)
    return np.maximum(np.nan_to_num(v, nan=0.25), 0.05) * math.sqrt(horizon / 252)


@dataclass
class Calibration:
    anchor: float                 # median training h-day log-return
    base_rate: float              # training P(return > 0)
    z: dict[str, float]           # standardized residual quantiles, keyed like TAUS
    median_weight: float = 0.0    # lambda — weight on the XGBoost median overlay
    direction_weight: float = 0.0  # kappa — weight on the XGBoost direction classifier
    n_calibration: int = 0
    heldout_mae_core: float | None = None
    heldout_mae_overlay: float | None = None
    heldout_logloss_core: float | None = None
    heldout_logloss_overlay: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _logloss(p: np.ndarray, up: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(np.where(up, p, 1.0 - p), 1e-6, 1.0))


def _pick_weight(row_losses: np.ndarray, horizon: int) -> float:
    """Best grid weight if it beats weight 0 significantly, else 0.

    `row_losses` has shape (len(_GRID), n_rows): per-row held-out loss for each
    candidate weight. Loss differences vs weight 0 are averaged within
    non-overlapping blocks of `horizon` rows and t-tested across blocks.
    """
    best = int(np.argmin(row_losses.mean(axis=1)))
    if best == 0:
        return 0.0
    diff = row_losses[0] - row_losses[best]          # > 0 where the overlay helps
    n_blocks = len(diff) // horizon
    if n_blocks < 5:
        return 0.0
    blocks = diff[: n_blocks * horizon].reshape(n_blocks, horizon).mean(axis=1)
    se = blocks.std(ddof=1) / math.sqrt(n_blocks)
    if se > 0 and blocks.mean() / se >= MIN_T:
        return float(_GRID[best])
    return 0.0


def fit_calibration(
    X: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    fit_heads,
) -> Calibration:
    """Fit the core distribution and choose overlay weights on a purged tail.

    `fit_heads(X_train, y_train)` must return `(median_model, direction_model)`
    trained with the production hyperparameters.
    """
    anchor = float(np.median(y))
    z_all = (y - anchor) / sigma
    cal = Calibration(
        anchor=anchor,
        base_rate=float((y > 0).mean()),
        z={k: float(np.quantile(z_all, tau)) for k, tau in TAUS.items()},
    )

    n = len(y)
    n_cal = max(60, int(n * CALIB_FRAC))
    n_head = n - n_cal - horizon      # purge: head labels end before the tail starts
    if n_head < MIN_HEAD_ROWS:
        return cal

    median_m, direction_m = fit_heads(X[:n_head], y[:n_head])
    X_c, y_c = X[n - n_cal:], y[n - n_cal:]
    head_anchor = float(np.median(y[:n_head]))
    head_base = float((y[:n_head] > 0).mean())
    ml_med = median_m.predict(X_c)
    ml_p = direction_m.predict_proba(X_c)[:, 1]
    up = y_c > 0

    abs_err = np.array([np.abs(head_anchor + w * (ml_med - head_anchor) - y_c) for w in _GRID])
    ll = np.array([_logloss(head_base + w * (ml_p - head_base), up) for w in _GRID])
    cal.median_weight = _pick_weight(abs_err, horizon)
    cal.direction_weight = _pick_weight(ll, horizon)
    cal.n_calibration = int(n_cal)
    mae, mll = abs_err.mean(axis=1), ll.mean(axis=1)
    cal.heldout_mae_core = round(float(mae[0]), 6)
    cal.heldout_mae_overlay = round(float(mae.min()), 6)
    cal.heldout_logloss_core = round(float(mll[0]), 6)
    cal.heldout_logloss_overlay = round(float(mll.min()), 6)
    return cal
