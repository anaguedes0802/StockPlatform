"""LSTM forecaster with three quantile heads (p10/p50/p90) and a direction head.

Only active if PyTorch is installed. The ensemble checks `LSTMForecaster.available()`
and falls back to XGBoost-only if torch is missing — keeps the default container light.

Training shape:
  x: (batch, seq_len, n_features)  using last `seq_len` rows of the feature dataframe
  y: (batch, 4)  → [q10, q50, q90, prob_up_logit]
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import features as feat_mod
from app.ml.base import ForecastResult, Forecaster


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


SEQ_LEN = 60
VAL_FRAC = 0.15  # chronological holdout for early stopping / honest val metric


@dataclass
class _Norm:
    mean: np.ndarray
    std: np.ndarray


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(self) -> None:
        self.model = None
        self.norm: _Norm | None = None
        self.feature_cols: list[str] = []
        self.val_loss: float | None = None
        self._trained = False

    @staticmethod
    def available() -> bool:
        return _torch_available()

    # ---------- helpers ----------

    def _build_xy(
        self, df: pd.DataFrame, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build chronologically-split (train, val) windows.

        The scaler is fit on the train portion ONLY — fitting it over the full
        series leaks the future distribution into every window. A window is
        assigned to train/val by the index of its *target* bar, so val targets
        are always strictly later in time than train targets.
        """
        feats = feat_mod.build_features(df)
        target = f"y_h{horizon}"
        if target not in feats.columns:
            raise ValueError(f"missing target {target}")
        feature_cols = feat_mod.feature_columns(feats)
        train = feats.dropna(subset=feature_cols + [target])
        if len(train) < SEQ_LEN + 30:
            raise ValueError(f"not enough rows ({len(train)}) for seq_len={SEQ_LEN}")

        X_full = train[feature_cols].values.astype(np.float32)
        y_full = train[target].values.astype(np.float32)
        n = len(X_full)

        # Chronological split point: target bars at/after `split` are validation.
        split = max(SEQ_LEN + 1, int(n * (1.0 - VAL_FRAC)))
        if split >= n:  # series too short for a holdout — train on everything
            split = n

        # Fit normalization on train rows only, then apply to the whole series.
        mean = X_full[:split].mean(axis=0)
        std = X_full[:split].std(axis=0) + 1e-6
        self.norm = _Norm(mean=mean, std=std)
        X_norm = (X_full - mean) / std

        Xtr, ytr, Xval, yval = [], [], [], []
        for i in range(SEQ_LEN, n):
            window = X_norm[i - SEQ_LEN : i]
            if i < split:
                Xtr.append(window)
                ytr.append(y_full[i])
            else:
                Xval.append(window)
                yval.append(y_full[i])

        X_train = np.stack(Xtr)
        y_train = np.array(ytr, dtype=np.float32)
        X_val = np.stack(Xval) if Xval else np.empty((0, SEQ_LEN, X_norm.shape[1]), dtype=np.float32)
        y_val = np.array(yval, dtype=np.float32)
        return X_train, y_train, X_val, y_val, feature_cols

    # ---------- training ----------

    def fit(self, df: pd.DataFrame, target_horizon_days: int) -> None:
        if not _torch_available():
            raise RuntimeError("PyTorch not installed; install the 'ml-deep' extras to enable LSTM.")
        import torch
        import torch.nn as nn

        X, y, X_val, y_val, feature_cols = self._build_xy(df, target_horizon_days)
        self.feature_cols = feature_cols
        n_features = X.shape[2]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=1, batch_first=True)
                self.head = nn.Linear(64, 4)  # q10, q50, q90, direction_logit

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        X_t = torch.from_numpy(X).to(device)
        y_t = torch.from_numpy(y).to(device)
        direction_t = (y_t > 0).float()
        bce = nn.BCEWithLogitsLoss()

        has_val = len(X_val) > 0
        if has_val:
            Xval_t = torch.from_numpy(X_val).to(device)
            yval_t = torch.from_numpy(y_val).to(device)
            dval_t = (yval_t > 0).float()

        def pinball(pred: "torch.Tensor", target: "torch.Tensor", q: float) -> "torch.Tensor":
            diff = target - pred
            return torch.maximum(q * diff, (q - 1) * diff).mean()

        def total_loss(pred: "torch.Tensor", yb: "torch.Tensor", db: "torch.Tensor") -> "torch.Tensor":
            q10, q50, q90, dlog = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            return (
                pinball(q10, yb, 0.1)
                + pinball(q50, yb, 0.5)
                + pinball(q90, yb, 0.9)
                + 0.5 * bce(dlog, db)
            )

        n_epochs = 40
        batch = 64
        patience = 8  # epochs without val improvement before early stop
        best_val = float("inf")
        best_state = None
        stale = 0
        # Shuffling minibatches within the train split is safe — the leak was the
        # scaler, not the SGD order. Val windows are strictly later in time.
        idx = np.arange(len(X))
        for _ in range(n_epochs):
            net.train()
            np.random.shuffle(idx)
            for s in range(0, len(idx), batch):
                b = idx[s : s + batch]
                pred = net(X_t[b])
                loss = total_loss(pred, y_t[b], direction_t[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

            if has_val:
                net.eval()
                with torch.no_grad():
                    vloss = float(total_loss(net(Xval_t), yval_t, dval_t).item())
                if vloss < best_val - 1e-5:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if has_val and best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        self.val_loss = best_val if has_val else None
        self._trained = True

    # ---------- prediction ----------

    def predict(self, df: pd.DataFrame, target_horizon_days: int) -> ForecastResult:
        if not self.model or not self.norm:
            raise RuntimeError("model not trained")
        import torch

        feats = feat_mod.build_features(df)
        feature_cols = self.feature_cols
        valid = feats[feature_cols].dropna()
        if len(valid) < SEQ_LEN:
            raise ValueError("not enough rows for inference window")

        X = valid.values[-SEQ_LEN:].astype(np.float32)
        X = (X - self.norm.mean) / self.norm.std
        device = next(self.model.parameters()).device
        with torch.no_grad():
            xb = torch.from_numpy(X).unsqueeze(0).to(device)
            out = self.model(xb)[0].cpu().numpy()
        q10_logret, q50_logret, q90_logret, dlog = float(out[0]), float(out[1]), float(out[2]), float(out[3])
        prob_up = 1.0 / (1.0 + math.exp(-dlog))

        last_price = float(df["close"].iloc[-1])
        point = last_price * math.exp(q50_logret)
        p10 = last_price * math.exp(q10_logret)
        p90 = last_price * math.exp(q90_logret)
        spread = (q90_logret - q10_logret) / 2
        p25 = last_price * math.exp(q50_logret - 0.5 * spread)
        p75 = last_price * math.exp(q50_logret + 0.5 * spread)
        sigma_ret = abs(q90_logret - q10_logret) / (2 * 1.2816) if q90_logret != q10_logret else 0.05
        exp_vol_pct = float(sigma_ret * 100)
        rel_width = abs(q90_logret - q10_logret) / max(abs(q50_logret), 0.01 + 1e-9)
        confidence = float(np.clip(1.0 / (1.0 + rel_width), 0.1, 0.95))

        # Contributions via simple gradient magnitudes (lightweight IG-style).
        contributions, drivers = self._explain(X, feature_cols)

        return ForecastResult(
            point=point, p10=p10, p25=p25, p50=point, p75=p75, p90=p90,
            direction_prob_up=prob_up,
            expected_volatility_pct=exp_vol_pct,
            confidence=confidence,
            contributions=contributions,
            drivers=drivers,
        )

    def _explain(self, X: np.ndarray, feature_cols: list[str]) -> tuple[dict[str, float], list[dict]]:
        """Integrated Gradients attribution.

        IG(x) = (x - baseline) · ∫₀¹ ∂F(baseline + α(x - baseline))/∂x dα

        We approximate the integral with Riemann sum over 20 steps from a
        zero baseline. Result is integrated over the time dimension and
        summed per feature to give a single attribution per feature column.
        """
        import torch

        device = next(self.model.parameters()).device
        steps = 20
        baseline = np.zeros_like(X, dtype=np.float32)

        # Per-step gradient accumulation
        grads_sum = np.zeros_like(X, dtype=np.float32)
        for k in range(1, steps + 1):
            alpha = k / steps
            interp = baseline + alpha * (X - baseline)
            xt = torch.from_numpy(interp).unsqueeze(0).to(device).requires_grad_(True)
            self.model.zero_grad()
            out = self.model(xt)[0, 1]  # q50 head
            out.backward()
            g = xt.grad.detach().cpu().numpy()[0]  # (seq, features)
            grads_sum += g

        avg_grad = grads_sum / steps
        ig = (X - baseline) * avg_grad           # element-wise attribution
        attribution = np.abs(ig).sum(axis=0)     # collapse the time axis

        category_sums: dict[str, float] = {}
        signed: dict[str, float] = {}
        for col, abs_a, signed_a in zip(feature_cols, attribution, ig.sum(axis=0), strict=True):
            cat = feat_mod.category_of(col)
            category_sums[cat] = category_sums.get(cat, 0.0) + float(abs_a)
            signed[col] = float(signed_a)
        total = sum(category_sums.values()) or 1.0
        contributions = {k: round(v / total, 4) for k, v in category_sums.items()}

        ranked = sorted(zip(feature_cols, attribution, strict=True), key=lambda t: -t[1])[:6]
        drivers = []
        for col, abs_a in ranked:
            direction = "up" if signed.get(col, 0.0) > 0 else ("down" if signed.get(col, 0.0) < 0 else "neutral")
            drivers.append({
                "name": col.replace("_", " "),
                "direction": direction,
                "weight": round(float(abs_a) / total, 4),
            })
        return contributions, drivers

    # ---------- persistence ----------

    @staticmethod
    def _path(symbol: str, horizon: int) -> str:
        os.makedirs(settings.model_dir, exist_ok=True)
        return os.path.join(settings.model_dir, f"lstm_{symbol}_{horizon}d.pt")

    def save(self, symbol: str, horizon: int) -> str:
        if not _torch_available() or not self.model or not self.norm:
            raise RuntimeError("not trained")
        import torch
        path = self._path(symbol, horizon)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "mean": self.norm.mean.tolist(),
                "std": self.norm.std.tolist(),
                "feature_cols": self.feature_cols,
            },
            path,
        )
        return path

    def load(self, symbol: str, horizon: int) -> bool:
        if not _torch_available():
            return False
        import torch
        import torch.nn as nn

        path = self._path(symbol, horizon)
        if not os.path.exists(path):
            return False
        blob = torch.load(path, map_location="cpu", weights_only=False)
        feature_cols = blob["feature_cols"]
        n_features = len(feature_cols)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=1, batch_first=True)
                self.head = nn.Linear(64, 4)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = Net()
        net.load_state_dict(blob["model_state"])
        net.eval()
        self.model = net
        self.norm = _Norm(
            mean=np.array(blob["mean"], dtype=np.float32),
            std=np.array(blob["std"], dtype=np.float32),
        )
        self.feature_cols = feature_cols
        self._trained = True
        return True
