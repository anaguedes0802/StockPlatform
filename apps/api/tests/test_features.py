from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml import features as feat


def _series(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + rng.normal(0, 1, n).cumsum())
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.integers(1_000, 10_000, n).astype(int))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_build_features_emits_targets():
    df = _series()
    f = feat.build_features(df)
    for h in (1, 5, 21, 63):
        assert f"y_h{h}" in f.columns
    # at least one usable row (no NaNs across feature cols + target)
    target = "y_h5"
    usable = f.dropna(subset=feat.feature_columns(f) + [target])
    assert not usable.empty
