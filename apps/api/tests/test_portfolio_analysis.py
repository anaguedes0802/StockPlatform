"""Tests for portfolio analysis math (optimizers, HHI, risk metrics).

These tests use synthetic return data so they don't hit yfinance / EDGAR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.portfolio_analysis import (
    _black_litterman_weights,
    _hhi,
    _risk_parity_weights,
    _shrinkage_covariance,
)


def _synthetic_returns(n_days: int = 252, n_symbols: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Three correlated assets + one independent
    base = rng.normal(0.0005, 0.012, n_days)
    cols = {}
    for i in range(n_symbols):
        corr = 0.6 if i < n_symbols - 1 else 0.0
        noise = rng.normal(0.0005, 0.012, n_days)
        cols[f"S{i}"] = corr * base + (1 - corr) * noise
    return pd.DataFrame(cols)


def test_hhi_bounds() -> None:
    assert _hhi({}) == 0.0
    assert _hhi({"A": 1.0}) == 1.0
    assert abs(_hhi({"A": 0.5, "B": 0.5}) - 0.5) < 1e-9
    assert _hhi({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == 0.25


def test_shrinkage_covariance_psd() -> None:
    R = _synthetic_returns()
    cov = _shrinkage_covariance(R)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-9).all(), "shrinkage cov should be PSD"


def test_risk_parity_weights_sum_to_one() -> None:
    R = _synthetic_returns()
    w = _risk_parity_weights(R)
    assert abs(sum(w.values()) - 1.0) < 1e-4
    for v in w.values():
        assert 0.0 <= v <= 1.0


def test_risk_parity_distributes_risk() -> None:
    """In a risk-parity portfolio, every asset should get a non-trivial weight."""
    R = _synthetic_returns()
    w = _risk_parity_weights(R)
    # No symbol should be > 60% of the portfolio (we cap at 50% in the code)
    assert max(w.values()) <= 0.55
    # No symbol should be < 1% (we floor at 0.5%)
    assert min(w.values()) >= 0.004


def test_black_litterman_runs() -> None:
    R = _synthetic_returns()
    w = _black_litterman_weights(R, market_weights={"S0": 0.4, "S1": 0.3, "S2": 0.2, "S3": 0.1},
                                  views={"S0": 0.12, "S1": -0.05})
    assert abs(sum(w.values()) - 1.0) < 1e-4
    # With a positive view on S0 and negative on S1, S0 should weight ≥ S1.
    assert w["S0"] >= w["S1"] - 0.01
