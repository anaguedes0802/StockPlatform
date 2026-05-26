"""Portfolio risk and performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    s = excess.std()
    if s == 0 or np.isnan(s):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / s)


def sortino(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside)


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return 0.0
    running_max = prices.cummax()
    dd = (prices - running_max) / running_max
    return float(dd.min())


def cagr(prices: pd.Series, periods_per_year: int = 252) -> float:
    if len(prices) < 2:
        return 0.0
    total_periods = len(prices) - 1
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.empty or df.iloc[:, 1].var() == 0:
        return 0.0
    cov = df.cov().iloc[0, 1]
    return float(cov / df.iloc[:, 1].var())


def value_at_risk_hist(returns: pd.Series, level: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(-np.quantile(returns, 1 - level))


def hhi(weights: dict[str, float]) -> float:
    """Herfindahl–Hirschman concentration; 1 = single asset, 0 = perfectly diversified."""
    if not weights:
        return 0.0
    total = sum(weights.values()) or 1.0
    return float(sum((w / total) ** 2 for w in weights.values()))
