from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import paper_replay as pr


def _panel(spy_drift: float, n_days: int = 700, n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n)), axis=0)), index=idx, columns=cols)
    close["SPY"] = 100 * np.exp(np.cumsum(np.full(n_days, spy_drift)))
    close["SHY"] = 80 * np.exp(np.cumsum(np.full(n_days, 0.0001)))
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return {"open": close.copy(), "close": close, "volume": vol, "sectors": {c: "X" for c in cols}}


def _run(monkeypatch, panel):
    monkeypatch.setattr(pr, "load_panel", lambda: panel)
    monkeypatch.setattr(pr, "load_events", lambda syms: {})
    return pr.run(start="2020-03-02", capital=10_000.0, save=False)


def test_risk_off_holds_shy(monkeypatch):
    res = _run(monkeypatch, _panel(spy_drift=-0.001))       # SPY in a steady downtrend -> gate off
    assert res["time_risk_on_pct"] == 0.0
    assert [h["symbol"] for h in res["holdings_end"]] == ["SHY"]
    # SHY drifts up slowly; only costs and the cash buffer separate us from it
    assert 9_500 < res["final_value"] < 11_500


def test_risk_on_buys_top_decile(monkeypatch):
    res = _run(monkeypatch, _panel(spy_drift=0.001))        # uptrend -> gate on
    assert res["time_risk_on_pct"] == 100.0
    assert 3 <= len(res["holdings_end"]) <= 10 and "SHY" not in {h["symbol"] for h in res["holdings_end"]}
    assert res["n_rebalances"] >= 10 and res["n_orders"] > res["n_rebalances"]
    assert res["final_value"] > 0
