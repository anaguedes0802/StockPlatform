"""Hidden-Markov (Markov-switching) alternative to the rule-based market regime.

Hamilton-style Markov-switching model on SPY daily log returns, with a
regime-dependent mean and variance (statsmodels `MarkovRegression`). It is
fitted on the warm-up + train years only, then run as a **filter** over the
whole research window with those fixed parameters: the probability on day t
uses returns up to t, never the smoothed (two-sided) estimate.

The highest-variance state is called risk-off. With 3 states the middle one
is "range". The comparison with the rules is decided by a rule written down
before running it (SWING_AGENT.md § 2).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def fit_filter(ret: pd.Series, fit_end: pd.Timestamp, k: int = 2, seed: int = 0) -> dict[str, Any]:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = (100 * np.log1p(ret)).dropna()
    train = y[y.index <= fit_end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        mod = MarkovRegression(train, k_regimes=k, trend="c", switching_variance=True)
        res = mod.fit(search_reps=20, disp=False)
        full = MarkovRegression(y, k_regimes=k, trend="c", switching_variance=True)
        filt = full.filter(res.params)
    probs = pd.DataFrame(np.asarray(filt.filtered_marginal_probabilities), index=y.index)
    names = list(res.model.param_names)
    pv = np.asarray(res.params)
    sig = np.array([pv[names.index(f"sigma2[{i}]")] for i in range(k)])
    mu = np.array([pv[names.index(f"const[{i}]")] for i in range(k)])
    order = np.argsort(sig)  # calm → wild
    labels = {int(order[-1]): "high_vol"}
    labels[int(order[0])] = "bull"  # calmest state
    if k == 3:
        labels[int(order[1])] = "range"
    state = probs.to_numpy().argmax(axis=1)
    regime = pd.Series([labels[int(s)] for s in state], index=y.index, name="regime")
    return {"regime": regime, "p_risk_off": probs[int(order[-1])],
            "params": {"mu_daily_pct": mu[order].round(4).tolist(),
                       "vol_annual_pct": (np.sqrt(sig[order]) * np.sqrt(252)).round(1).tolist(),
                       "loglike_train": round(float(res.llf), 1)}}
