from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml import stock_selection as ss


def _panel(n_days: int = 700, n_stocks: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    mkt = rng.normal(0.0003, 0.01, n_days)
    rets = mkt[:, None] + rng.normal(0, 0.015, (n_days, n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    vol = pd.DataFrame(1e6, index=idx, columns=cols)
    spy = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx)
    return close, vol, spy


def test_factors_do_not_look_ahead():
    close, vol, spy = _panel()
    sectors = {c: "Tech" if i % 2 else "Energy" for i, c in enumerate(close.columns)}
    t = 500
    f_full = ss.compute_factors(close, spy, sectors)
    tampered = close.copy()
    tampered.iloc[t + 1:] *= np.random.default_rng(1).uniform(0.5, 2.0, tampered.iloc[t + 1:].shape)
    f_tamp = ss.compute_factors(tampered, spy, sectors)
    for k in f_full:
        pd.testing.assert_series_equal(f_full[k].iloc[t], f_tamp[k].iloc[t], check_names=False)


def test_planted_momentum_is_picked_up():
    close, vol, spy = _panel(n_days=900, n_stocks=100, seed=3)
    # stocks 0-19 get persistent extra drift -> high 12-1 momentum AND higher future returns
    boost = np.zeros(close.shape)
    boost[:, :20] = 0.003
    close = close * np.exp(np.cumsum(boost, axis=0))
    f = ss.compute_factors(close, spy, {c: None for c in close.columns})
    mask = ss.universe_mask(close, vol)
    score, _ = ss.composite(f, mask, {"mom_12_1": 1.0})
    r = ss.backtest(score, close, mask, start="2019-03-01")
    s = ss.summarize(r)
    assert s["excess_vs_equal_weight_pct_per_year"] > 10
    assert s["rank_ic_mean"] > 0.1


def test_earnings_reaction_bar_respects_announcement_time():
    idx = pd.bdate_range("2024-01-01", periods=10)
    before_open = pd.Timestamp("2024-01-03 07:00", tz="America/New_York")
    after_close = pd.Timestamp("2024-01-03 16:05", tz="America/New_York")
    assert idx[ss._reaction_bar(idx, before_open)] == pd.Timestamp("2024-01-03")
    assert idx[ss._reaction_bar(idx, after_close)] == pd.Timestamp("2024-01-04")


def test_earnings_factor_expires_after_fresh_window():
    idx = pd.bdate_range("2024-01-01", periods=120)
    close = pd.DataFrame({"A": np.linspace(100, 120, 120)}, index=idx)
    ev = {"A": [ss.EarningsEvent(pd.Timestamp("2024-01-10 07:00", tz="America/New_York"), 12.0)]}
    f = ss.earnings_factors(close, ev)
    pos = idx.get_loc(pd.Timestamp("2024-01-10"))
    assert np.isnan(f["earn_surprise"]["A"].iloc[pos - 1])
    assert f["earn_surprise"]["A"].iloc[pos] == 12.0
    assert f["earn_surprise"]["A"].iloc[pos + ss.EARNINGS_FRESH_BARS] == 12.0
    assert np.isnan(f["earn_surprise"]["A"].iloc[pos + ss.EARNINGS_FRESH_BARS + 1])


def test_gate_holds_cash_when_risk_off():
    close, vol, spy = _panel(n_days=700, n_stocks=60, seed=5)
    f = ss.compute_factors(close, spy, {c: None for c in close.columns})
    mask = ss.universe_mask(close, vol)
    score, _ = ss.composite(f, mask, {"mom_12_1": 1.0})
    cash = pd.Series(100 * np.exp(np.arange(len(close)) * 0.0001), index=close.index)
    off = pd.Series(False, index=close.index)
    r = ss.backtest(score, close, mask, start="2019-03-01", risk_on=off, cash_open=cash)
    expected = np.log(cash.shift(-(1 + ss.HOLD_BARS)) / cash.shift(-1)).reindex(r.index)
    # first row pays the switch from the initial risk-on state, the rest are pure cash
    np.testing.assert_allclose(r["gated"].iloc[1:], expected.iloc[1:], atol=1e-12)
    on = pd.Series(True, index=close.index)
    r_on = ss.backtest(score, close, mask, start="2019-03-01", risk_on=on, cash_open=cash)
    np.testing.assert_allclose(r_on["gated"], r_on["long"], atol=1e-12)
