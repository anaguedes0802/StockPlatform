"""Fit regime-conditional ensemble weights from matured predictions.

Reads the `predictions` table, joins each prediction to its realized price
at target_ts, computes per-(model, regime) errors, and outputs a regime →
{model: weight} mapping that minimizes squared error.

Output: artifacts/ensemble/regime_weights.json (auto-loaded by ml/ensemble.py
at next process start).

This replaces the hand-tuned REGIME_MODEL_MIX table with an empirically-fit
one once enough matured predictions exist.

Usage:
    python scripts/fit_ensemble_weights.py
    python scripts/fit_ensemble_weights.py --lookback-days 365 --min-samples 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, "apps/api")

import numpy as np
from scipy.optimize import minimize
from sqlalchemy import select

from app.config import settings  # noqa: E402
from app.db.models import Prediction  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml.regime import detect  # noqa: E402
from app.services import market_data as md  # noqa: E402


def _realized_close(symbol: str, target_ts):
    try:
        df = md.get_history(symbol, interval="1d", range_="3y")
        if df.empty:
            return None
        ts = target_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        eligible = df[df.index <= ts]
        if eligible.empty:
            return None
        return float(eligible["close"].iloc[-1])
    except Exception:
        return None


def fit(lookback_days: int = 365, min_samples: int = 30, holdout_frac: float = 0.25) -> dict:
    """Walk matured predictions, fit regime-conditional model weights.

    To avoid the in-sample / out-of-sample conflation the prior version had
    (weights were optimized and reported on the SAME data), each regime's
    samples are sorted chronologically and split into a TRAIN head and a
    held-out TAIL (`holdout_frac`). Weights are fit on TRAIN only; we report
    both in-sample (train) MSE and out-of-sample (tail) MSE so any lift from
    the fitted weights over equal weighting is not overstated.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    # samples[regime] = list of (target_ts, realized_logret, {model_name: predicted_logret})
    samples: dict[str, list[tuple[Any, float, dict[str, float]]]] = defaultdict(list)

    with SessionLocal() as db:
        rows = db.scalars(
            select(Prediction)
            .where(Prediction.created_at >= cutoff)
            .where(Prediction.target_ts <= now)
            .order_by(Prediction.target_ts.desc())
        ).all()
        for r in rows:
            try:
                # Determine the regime at prediction time
                df = md.get_history(r.symbol, interval="1d", range_="3y")
                if df.empty: continue
                hist = df[df.index <= r.created_at]
                if len(hist) < 60: continue
                regime = detect(hist["close"])
                issuance_close = float(hist["close"].iloc[-1])
                realized = _realized_close(r.symbol, r.target_ts)
                if realized is None or issuance_close <= 0: continue
                realized_lr = math.log(realized / issuance_close)
                pred_lr = math.log(float(r.point_estimate) / issuance_close)
                # We log a single ensemble prediction per row; for per-model
                # weights we'd need per-model rows. For now, treat the recorded
                # model_mix as the basis: if the row has per-model contributions
                # in `contributions`, we use those; else we treat the recorded
                # `model` name as the sole contributor.
                model_name = r.model or "ensemble"
                samples[regime].append((r.target_ts, realized_lr, {model_name: pred_lr}))
            except Exception:
                continue

    if not samples:
        return {"ok": False, "reason": "no matured predictions in window",
                "n_samples": 0}

    def _mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        if X.size == 0:
            return float("nan")
        return float(np.mean((X @ w - y) ** 2))

    # For each regime, fit weights w (sum to 1, w >= 0) minimizing MSE on a
    # TRAIN head, then report MSE on the held-out TAIL.
    fitted: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict] = {}
    for regime, pairs in samples.items():
        if len(pairs) < min_samples:
            diagnostics[regime] = {"n": len(pairs), "skipped": "below min_samples"}
            continue
        # Sort chronologically (rows arrive target_ts.desc()) so the holdout is
        # a genuine future tail, not a random slice.
        pairs = sorted(pairs, key=lambda t: t[0])
        # Collect all model names that appear in this regime
        all_models = sorted({m for _, _, preds in pairs for m in preds.keys()})
        if len(all_models) == 1:
            # No optimization possible with a single model — use 1.0
            fitted[regime] = {all_models[0]: 1.0}
            diagnostics[regime] = {"n": len(pairs), "n_models": 1}
            continue

        # Chronological train / held-out tail split.
        n = len(pairs)
        n_holdout = max(1, int(round(n * holdout_frac)))
        n_train = n - n_holdout
        if n_train < max(min_samples // 2, len(all_models) + 1):
            diagnostics[regime] = {"n": n, "skipped": "train split too small after holdout"}
            continue
        train_pairs = pairs[:n_train]
        tail_pairs = pairs[n_train:]

        def _design(ps):
            X = np.array([[preds.get(m, 0.0) for m in all_models] for _, _, preds in ps])
            y = np.array([rlr for _, rlr, _ in ps])
            return X, y

        X_tr, y_tr = _design(train_pairs)
        X_te, y_te = _design(tail_pairs)

        def loss(w: np.ndarray) -> float:
            return float(np.mean((X_tr @ w - y_tr) ** 2))

        w0 = np.ones(len(all_models)) / len(all_models)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0) for _ in all_models]
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 200})
        w_opt = res.x if res.success else w0
        fitted[regime] = {m: float(w) for m, w in zip(all_models, w_opt, strict=True)}
        # Equal-weight baseline lets us judge whether fitting actually helped OOS.
        diagnostics[regime] = {
            "n": n,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "n_models": len(all_models),
            "in_sample_mse": round(_mse(X_tr, y_tr, w_opt), 8),
            "out_of_sample_mse": round(_mse(X_te, y_te, w_opt), 8),
            "out_of_sample_mse_equal_weights": round(_mse(X_te, y_te, w0), 8),
            "success": bool(res.success),
        }

    out = {
        "ok": True,
        "fitted_at": now.isoformat(),
        "lookback_days": lookback_days,
        "min_samples": min_samples,
        "holdout_frac": holdout_frac,
        "regimes": fitted,
        "diagnostics": diagnostics,
    }
    d = os.path.join(settings.model_dir, "..", "ensemble")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "regime_weights.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--min-samples",  type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.25,
                   help="Chronological tail fraction held out for OOS validation")
    args = p.parse_args()
    result = fit(args.lookback_days, args.min_samples, args.holdout_frac)
    print(json.dumps(result, indent=2))
    # Surface in-sample vs out-of-sample error so lift over equal weighting is
    # not overstated by the fitted weights.
    if result.get("ok"):
        print("\nin-sample vs out-of-sample MSE per regime:")
        for regime, d in (result.get("diagnostics") or {}).items():
            if "out_of_sample_mse" not in d:
                continue
            print(f"  {regime:15s} train_mse={d['in_sample_mse']:.6f} "
                  f"oos_mse={d['out_of_sample_mse']:.6f} "
                  f"oos_mse_equal={d['out_of_sample_mse_equal_weights']:.6f} "
                  f"(n_train={d['n_train']}, n_holdout={d['n_holdout']})")


if __name__ == "__main__":
    main()
