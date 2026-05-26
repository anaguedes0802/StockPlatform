from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    point: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    direction_prob_up: float
    expected_volatility_pct: float
    confidence: float
    contributions: dict[str, float] = field(default_factory=dict)
    drivers: list[dict] = field(default_factory=list)


class Forecaster(ABC):
    """Common forecaster interface. All models train on price+features dataframes."""

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_horizon_days: int) -> None: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, target_horizon_days: int) -> ForecastResult: ...

    def is_trained(self) -> bool:
        return getattr(self, "_trained", False)


def widen_with_volatility(point: float, sigma_pct: float) -> tuple[float, float, float, float, float]:
    """Heuristic interval when a model doesn't natively output quantiles."""
    # Treat sigma_pct as the std of return over the horizon (in percent).
    s = sigma_pct / 100.0
    factors = {
        "p10": -1.2816,
        "p25": -0.6745,
        "p50": 0.0,
        "p75": 0.6745,
        "p90": 1.2816,
    }
    out = [point * np.exp(z * s) for z in factors.values()]
    return tuple(out)  # type: ignore[return-value]
