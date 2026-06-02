"""Tests for the long-horizon investment bot (app.services.investing).

All offline: rebalance math is pure, and execution uses a fake in-memory broker.
Timing is monkeypatched so we never hit the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import investing as inv


# --------------------------------------------------------------------------
# allocation + cash-flow rebalancing math
# --------------------------------------------------------------------------

def test_normalize_weights_sums_to_one() -> None:
    w = inv.normalize_weights({"SPY": 2, "BND": 2})  # arbitrary, non-normalised
    assert w == {"SPY": 0.5, "BND": 0.5}
    assert abs(sum(inv.normalize_weights(inv.DEFAULT_ALLOCATION).values()) - 1.0) < 1e-9


def test_normalize_drops_nonpositive_and_falls_back() -> None:
    assert inv.normalize_weights({"SPY": 1, "BND": 0, "GLD": -1}) == {"SPY": 1.0}
    assert inv.normalize_weights({}) == inv.normalize_weights(inv.DEFAULT_ALLOCATION)


def test_fresh_deploy_hits_target_weights() -> None:
    plan = inv.rebalance_plan([], cash=10_000.0, contribution=0.0,
                              timing={"suggested_equity_tilt": 0.0})
    by_sym = {o["symbol"]: o["notional"] for o in plan["orders"]}
    assert by_sym["SPY"] == pytest.approx(3500.0, abs=1.0)   # 35%
    assert by_sym["BND"] == pytest.approx(2000.0, abs=1.0)   # 20%
    assert plan["leftover"] == pytest.approx(0.0, abs=1.0)
    assert sum(by_sym.values()) == pytest.approx(10_000.0, abs=1.0)


def test_contribution_skips_overweight_sleeve() -> None:
    # Already 80% SPY / 20% BND. New money must NOT buy more SPY (overweight);
    # it should flow to the underweight sleeves instead.
    pos = [{"symbol": "SPY", "market_value": 8000.0},
           {"symbol": "BND", "market_value": 2000.0}]
    plan = inv.rebalance_plan(pos, cash=0.0, contribution=5000.0,
                              timing={"suggested_equity_tilt": 0.0})
    by_sym = {o["symbol"]: o["notional"] for o in plan["orders"]}
    assert "SPY" not in by_sym            # SPY is overweight -> no buy
    assert by_sym  # something was bought
    assert sum(by_sym.values()) == pytest.approx(5000.0, abs=1.0)


def test_dip_tilt_favours_equities() -> None:
    pos = [{"symbol": "SPY", "market_value": 1000.0}]
    flat = inv.rebalance_plan(pos, cash=0.0, contribution=6000.0,
                              timing={"suggested_equity_tilt": 0.0})
    tilt = inv.rebalance_plan(pos, cash=0.0, contribution=6000.0,
                              timing={"suggested_equity_tilt": 0.20})
    def eq_share(plan):
        eq = sum(o["notional"] for o in plan["orders"]
                 if o["symbol"] in inv.DEFAULT_INVEST_CONFIG["equity_sleeves"])
        return eq / sum(o["notional"] for o in plan["orders"])
    assert eq_share(tilt) > eq_share(flat)
    assert tilt["equity_tilt_applied"] == 0.20


def test_nothing_to_deploy_returns_empty_plan() -> None:
    plan = inv.rebalance_plan([], cash=0.0, contribution=0.0, timing={})
    assert plan["orders"] == []


# --------------------------------------------------------------------------
# quality stock-picker mode
# --------------------------------------------------------------------------

def _factors(sym, level):
    """A full factor row; `level` (0..1) scales every factor so ranking is
    deterministic (higher level → better on value, quality, growth, momentum)."""
    return {
        "symbol": sym, "earnings_yield": 0.02 + 0.06 * level, "margin": 0.05 + 0.30 * level,
        "roe": 0.05 + 0.40 * level, "low_debt": -2.0 + 1.8 * level,
        "rev_growth": 0.50 * level, "eps_growth": 0.60 * level, "momentum": 0.80 * level,
        "last_price": 100.0,
    }


@pytest.fixture
def no_picks_cache(monkeypatch):
    """Neutralise the Redis result-cache so picker tests are deterministic and
    don't depend on (or pollute) a running Redis. get → always miss, set → no-op."""
    monkeypatch.setattr(inv, "_cache_get", lambda *a, **k: None)
    monkeypatch.setattr(inv, "_cache_set", lambda *a, **k: None)


def test_quality_picks_equal_weights_and_order(no_picks_cache, monkeypatch) -> None:
    data = {s: _factors(s, lv) for s, lv in
            [("NVDA", 1.0), ("JPM", 0.7), ("GE", 0.5), ("KO", 0.3), ("XOM", 0.1)]}
    monkeypatch.setattr(inv, "_pick_factors", lambda s: data.get(s))
    qp = inv.quality_picks(universe=list(data), top_n=3)
    assert [p["symbol"] for p in qp["picks"]] == ["NVDA", "JPM", "GE"]  # best factors first
    assert set(qp["weights"]) == {"NVDA", "JPM", "GE"}
    assert all(abs(w - 1 / 3) < 1e-9 for w in qp["weights"].values())


def test_quality_picks_dedupes_share_classes(no_picks_cache, monkeypatch) -> None:
    data = {s: _factors(s, lv) for s, lv in
            [("GOOGL", 1.0), ("GOOG", 0.99), ("MSFT", 0.6), ("AAPL", 0.4)]}
    monkeypatch.setattr(inv, "_pick_factors", lambda s: data.get(s))
    qp = inv.quality_picks(universe=["GOOGL", "GOOG", "MSFT", "AAPL"], top_n=3)
    # GOOG (secondary class) must be dropped so Alphabet takes only one slot.
    assert "GOOG" not in qp["weights"]
    assert "GOOGL" in qp["weights"]


def test_quality_picks_failure_falls_back_to_etf(no_picks_cache, monkeypatch) -> None:
    monkeypatch.setattr(inv, "_pick_factors", lambda s: None)  # nothing scoreable
    target, picks = inv._resolve_target({**inv.DEFAULT_INVEST_CONFIG, "pick_mode": "quality"})
    assert picks is None
    assert set(target) == set(inv.normalize_weights(inv.DEFAULT_ALLOCATION))


def test_rebalance_plan_quality_mode_buys_picks(no_picks_cache, monkeypatch) -> None:
    data = {s: _factors(s, lv) for s, lv in [("AAA", 1.0), ("BBB", 0.6), ("CCC", 0.2)]}
    monkeypatch.setattr(inv, "_pick_factors", lambda s: data.get(s))
    dsl = {**inv.DEFAULT_INVEST_CONFIG, "pick_mode": "quality", "top_n": 2,
           "pick_universe": list(data)}
    plan = inv.rebalance_plan([], cash=10_000.0, dsl=dsl, contribution=0.0,
                              timing={"suggested_equity_tilt": 0.0})
    by = {o["symbol"]: o["notional"] for o in plan["orders"]}
    assert set(by) == {"AAA", "BBB"}                       # bought the top picks, not ETFs
    assert by["AAA"] == pytest.approx(5000.0, abs=1.0)     # equal weight
    assert plan["pick_mode"] == "quality"
    assert [p["symbol"] for p in plan["picks"]] == ["AAA", "BBB"]


# --------------------------------------------------------------------------
# quality_picks Redis result cache (fake in-memory cache layer)
# --------------------------------------------------------------------------

def _fake_cache(monkeypatch):
    """Install an in-memory stand-in for the Redis cache helpers and return the
    backing dict. Works whether or not a real Redis is running."""
    store: dict[str, object] = {}
    monkeypatch.setattr(inv, "_cache_get", lambda key: store.get(key))
    monkeypatch.setattr(inv, "_cache_set",
                        lambda key, value, ttl: store.__setitem__(key, value))
    return store


def test_quality_picks_caches_result(monkeypatch) -> None:
    store = _fake_cache(monkeypatch)
    data = {s: _factors(s, lv) for s, lv in
            [("NVDA", 1.0), ("JPM", 0.7), ("GE", 0.5), ("KO", 0.3)]}

    calls: list[str] = []

    def counting_factors(sym):
        calls.append(sym)
        return data.get(sym)

    monkeypatch.setattr(inv, "_pick_factors", counting_factors)

    first = inv.quality_picks(universe=list(data), top_n=3)
    n_after_first = len(calls)
    assert n_after_first > 0                       # computed on the cold path
    assert store                                   # result was written to cache

    second = inv.quality_picks(universe=list(data), top_n=3)
    assert len(calls) == n_after_first             # NOT recomputed — served from cache
    assert second == first                         # identical cached dict


def test_quality_picks_cache_key_varies_with_args() -> None:
    base = ["NVDA", "JPM", "GE"]
    # Order-independent: same set of symbols -> same key.
    assert inv._picks_cache_key(base, 3) == inv._picks_cache_key(list(reversed(base)), 3)
    # top_n is part of the key.
    assert inv._picks_cache_key(base, 3) != inv._picks_cache_key(base, 5)
    # Different universe -> different key.
    assert inv._picks_cache_key(base, 3) != inv._picks_cache_key(base + ["KO"], 3)


def test_quality_picks_failure_not_cached(monkeypatch) -> None:
    store = _fake_cache(monkeypatch)
    monkeypatch.setattr(inv, "_pick_factors", lambda s: None)  # nothing scoreable
    out = inv.quality_picks(universe=["AAA", "BBB", "CCC"], top_n=2)
    assert out == {"weights": {}, "picks": []}
    assert store == {}   # empty/failure result is not pinned, so it can retry


def test_cache_helpers_fail_open(monkeypatch) -> None:
    # When the underlying Redis client raises, the helpers must swallow it:
    # _cache_get returns None (miss) and _cache_set is a silent no-op. This is
    # what guarantees quality_picks degrades to a normal compute when Redis is
    # unavailable.
    class BoomRedis:
        def get(self, *a, **k):
            raise RuntimeError("redis down")
        def setex(self, *a, **k):
            raise RuntimeError("redis down")
    monkeypatch.setattr(inv, "_cache", lambda: BoomRedis())
    assert inv._cache_get("qpicks:anything") is None
    inv._cache_set("qpicks:anything", {"x": 1}, ttl=10)  # must not raise


def test_quality_picks_computes_when_cache_unavailable(monkeypatch) -> None:
    # End-to-end: with a Redis that always errors, quality_picks still returns a
    # correctly-ranked result rather than propagating the error.
    class BoomRedis:
        def get(self, *a, **k):
            raise RuntimeError("redis down")
        def setex(self, *a, **k):
            raise RuntimeError("redis down")
    monkeypatch.setattr(inv, "_cache", lambda: BoomRedis())
    data = {s: _factors(s, lv) for s, lv in
            [("NVDA", 1.0), ("JPM", 0.7), ("GE", 0.5)]}
    monkeypatch.setattr(inv, "_pick_factors", lambda s: data.get(s))
    out = inv.quality_picks(universe=list(data), top_n=2)
    assert [p["symbol"] for p in out["picks"]] == ["NVDA", "JPM"]


# --------------------------------------------------------------------------
# timing levels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dd,reg_score,expected", [
    (0.0, 30, "NORMAL"),        # at highs, risk-on
    (0.11, -15, "ACCUMULATE"),  # moderate pullback
    (0.25, -60, "STRONG_BUY"),  # deep drawdown + fear
])
def test_invest_timing_levels(monkeypatch, dd, reg_score, expected) -> None:
    monkeypatch.setattr(inv, "_drawdown_from_high", lambda *a, **k: dd)
    monkeypatch.setattr(inv.regime_svc, "assess",
                        lambda *a, **k: {"score": reg_score, "regime": "x", "headline": "h"})
    monkeypatch.setattr(inv.llm, "is_available", lambda: False)
    t = inv.invest_timing(use_llm=False)
    assert t["level"] == expected
    assert 0.0 <= t["opportunity"] <= 1.0
    # The dip tilt only engages once we're at least ACCUMULATE.
    if expected == "NORMAL":
        assert t["suggested_equity_tilt"] == 0.0
    else:
        assert t["suggested_equity_tilt"] > 0.0


# --------------------------------------------------------------------------
# live execution via a fake broker
# --------------------------------------------------------------------------

class FakeBroker:
    def __init__(self, *, cash=10_000.0, configured=True, market_open=True, positions=None):
        self._cash = cash
        self._configured = configured
        self._open = market_open
        self._positions = positions or []
        self.orders: list[dict] = []

    def is_configured(self): return self._configured
    def is_paper(self): return True
    def is_market_open(self): return self._open
    def get_account(self): return {"cash": self._cash, "buying_power": self._cash,
                                   "equity": self._cash, "portfolio_value": self._cash}
    def list_positions(self): return list(self._positions)

    def submit_market_order(self, symbol, *, notional=None, qty=None, side="buy"):
        o = {"symbol": symbol.upper(), "notional": notional, "side": side, "id": f"o{len(self.orders)}"}
        self.orders.append(o)
        return o


def _fixed_timing(monkeypatch, level="NORMAL", tilt=0.0):
    monkeypatch.setattr(inv, "invest_timing",
                        lambda *a, **k: {"level": level, "suggested_equity_tilt": tilt,
                                         "opportunity": 0.1, "headline": "x"})


def test_run_invest_blocked_when_market_closed(monkeypatch) -> None:
    _fixed_timing(monkeypatch)
    fb = FakeBroker(market_open=False)
    res = inv.run_invest(dsl=None, execution={"market_hours_only": True},
                         contribution=0.0, broker_mod=fb, use_llm=False)
    assert res["blocked"] == "market_closed"
    assert fb.orders == []


def test_run_invest_blocked_when_unconfigured(monkeypatch) -> None:
    _fixed_timing(monkeypatch)
    fb = FakeBroker(configured=False)
    res = inv.run_invest(dsl=None, execution={}, broker_mod=fb, use_llm=False)
    assert res["blocked"] == "broker_not_configured"


def test_run_invest_deploys_cash_buy_only(monkeypatch) -> None:
    _fixed_timing(monkeypatch)
    fb = FakeBroker(cash=10_000.0, market_open=True)
    res = inv.run_invest(dsl=None, execution={"market_hours_only": True},
                         contribution=0.0, broker_mod=fb, use_llm=False)
    assert res["blocked"] is None
    assert all(o["side"] == "buy" for o in fb.orders)            # never sells
    assert len(fb.orders) == len(inv.DEFAULT_ALLOCATION)         # one per sleeve
    assert sum(o["notional"] for o in fb.orders) == pytest.approx(10_000.0, abs=2.0)


def test_run_invest_adds_contribution_on_top_of_cash(monkeypatch) -> None:
    _fixed_timing(monkeypatch)
    fb = FakeBroker(cash=2_000.0, market_open=True)
    res = inv.run_invest(dsl=None, execution={"market_hours_only": True},
                         contribution=3_000.0, broker_mod=fb, use_llm=False)
    # buying_power caps at cash (2k) in this fake, so it can't spend the full 5k;
    # but the *plan* should target 5k deployable.
    assert res["plan"]["deployable"] == pytest.approx(5_000.0, abs=1.0)


# --------------------------------------------------------------------------
# DCA backtest math (offline, synthetic prices)
# --------------------------------------------------------------------------

def test_irr_recovers_known_rate() -> None:
    # 12 monthly $100 outflows; final value chosen so the annual IRR is ~0.
    flows = [-100.0] * 12 + [1200.0]
    irr = inv._annualised_irr(flows)
    assert irr == pytest.approx(0.0, abs=0.01)


def test_irr_positive_when_value_grows() -> None:
    flows = [-100.0] * 12 + [1400.0]  # put in 1200, ended at 1400
    assert inv._annualised_irr(flows) > 0.0


def _flat_px(days: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame({"SPY": np.full(days, price)}, index=idx)


def test_simulate_dca_flat_price_no_profit() -> None:
    px = _flat_px(800)
    r = inv._simulate_dca(px, lambda _ts: {"SPY": 1.0}, monthly=1000.0)
    # Flat prices: you get back exactly what you put in, ~0% return / IRR.
    assert r["final_value"] == pytest.approx(r["contributed"], rel=1e-6)
    assert r["total_return_pct"] == pytest.approx(0.0, abs=0.01)
    assert r["irr_annual_pct"] == pytest.approx(0.0, abs=0.5)
    assert r["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)


def test_simulate_dca_rising_price_profits() -> None:
    idx = pd.date_range("2015-01-01", periods=800, freq="D", tz="UTC")
    px = pd.DataFrame({"SPY": np.linspace(100, 200, 800)}, index=idx)  # +100%
    r = inv._simulate_dca(px, lambda _ts: {"SPY": 1.0}, monthly=1000.0)
    assert r["final_value"] > r["contributed"]
    assert r["irr_annual_pct"] > 0.0


def test_level_from_drawdown_thresholds() -> None:
    assert inv._level_from_drawdown(0.0)[0] == "NORMAL"
    assert inv._level_from_drawdown(0.12)[0] == "ACCUMULATE"
    assert inv._level_from_drawdown(0.30)[0] == "STRONG_BUY"
    # Deeper drawdown → larger equity tilt.
    assert inv._level_from_drawdown(0.30)[1] > inv._level_from_drawdown(0.12)[1]


def test_backtest_picks_growth_and_tilt(monkeypatch) -> None:
    # Synthetic adjusted closes: a 3-year daily series that roughly triples.
    idx = pd.date_range("2021-01-01", periods=1100, freq="D", tz="UTC")
    def series(mult):  # ends `mult`x where it started
        return pd.Series(np.linspace(100, 100 * mult, len(idx)), index=idx)
    prices = {"SPY": series(2.0), "QQQ": series(3.0), "BND": series(1.1),
              "VXUS": series(1.5), "SCHD": series(1.8), "GLD": series(2.2)}
    monkeypatch.setattr(inv, "_full_history_close", lambda s: prices.get(s))

    r = inv.backtest_picks("2021-06-01", amount=10_000)
    assert r["amount"] == 10_000
    # Every holding bought at ~start and grown to ~end -> positive growth.
    assert all(h["growth_pct"] > 0 for h in r["holdings"])
    # Holdings are sorted best-grower first.
    g = [h["growth_pct"] for h in r["holdings"]]
    assert g == sorted(g, reverse=True)
    # Portfolio value = sum of holding current values, and dollars deployed = amount.
    assert sum(h["invested"] for h in r["holdings"]) == pytest.approx(10_000, abs=1.0)
    assert r["portfolio"]["current_value"] == pytest.approx(
        sum(h["current_value"] for h in r["holdings"]), abs=1.0)
    assert "level" in r["timing_at_start"]
