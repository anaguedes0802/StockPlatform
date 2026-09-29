"""ML signal filter: no look-ahead, purged walk-forward, engine plumbing, stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest import swing_engine as eng
from app.ml import signal_filter as sf
from app.services import swing
from app.services import trading_bot as bot


def _walk(n=900, seed=1, drift=0.0004, start="2004-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


def _ctx(setup="breakout", n_syms=6, n=1800, allow_short=False) -> dict:
    data = {f"S{i}": _walk(n=n, seed=10 + i) for i in range(n_syms)}
    p = swing._p(setup, None)
    cfg = swing.config_for("equity")
    cfg.allow_short = allow_short
    return {"setup": setup, "params": p, "rules": swing.setup_rules(setup, p), "cfg": cfg,
            "data": data, "errors": {}, "signals": {k: swing.build_signals(v, setup, p) for k, v in data.items()},
            "earnings": None, "regime": None, "benchmark": _walk(n=n, seed=99), "benchmark_symbol": "SPY",
            "asset_class": "equity", "universe": "custom",
            "universe_meta": {"label": "t", "note": "t", "asset_class": "equity"}, "load_from": None}


# ---------------------------------------------------------------------------
# no look-ahead
# ---------------------------------------------------------------------------

def test_symbol_features_have_no_lookahead() -> None:
    df = _walk()
    full = sf.symbol_features(df)
    cut = sf.symbol_features(df.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[:600], cut, check_exact=False, rtol=1e-9)


def test_market_features_have_no_lookahead() -> None:
    data = {k: _walk(seed=s) for k, s in (("A", 1), ("B", 2))}
    spy, vix = _walk(seed=3), _walk(seed=4)
    cal = data["A"].index
    full = sf.market_features(cal, data, spy, vix)
    cut_data = {k: v.iloc[:600] for k, v in data.items()}
    cut = sf.market_features(cal[:600], cut_data, spy.iloc[:600], vix.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[:600], cut, check_exact=False, rtol=1e-9)


def test_fx_market_features_are_lagged_one_session() -> None:
    data = {"A": _walk(seed=1)}
    spy = _walk(seed=3)
    cal = data["A"].index
    now = sf.market_features(cal, data, spy, None)
    lagged = sf.market_features(cal, data, spy, None, lag=1)
    pd.testing.assert_series_equal(lagged["spy_ret_63"].iloc[1:], now["spy_ret_63"].iloc[:-1],
                                   check_index=False, check_names=False)


def test_short_events_flip_directional_features() -> None:
    df = _walk()
    feats = sf.symbol_features(df)
    sig = swing.build_signals(df, "breakout")
    sig.loc[:, ["long_entry", "short_entry"]] = False
    d = df.index[500]
    sig.loc[d, "short_entry"] = True
    cal = df.index
    row = sf.event_rows("A", feats, sig, pd.DataFrame(index=cal, columns=sf.MARKET_FEATURES, dtype=float),
                        pd.Series(1.0, index=cal), allow_short=True).iloc[0]
    assert row["side"] == -1
    assert row["ret_21"] == pytest.approx(-feats.loc[d, "ret_21"])
    assert row["dist_ext55_atr"] == pytest.approx(-feats.loc[d, "dist_low55_atr"])
    assert row["rsi14"] == pytest.approx(100 - feats.loc[d, "rsi14"])


# ---------------------------------------------------------------------------
# events, labels, purged walk-forward
# ---------------------------------------------------------------------------

def test_labels_are_trades_starting_the_next_session() -> None:
    ctx = _ctx()
    ev = sf.build_events(ctx, spy=None, vix=None)
    lab = ev[ev["label_r"].notna()]
    assert len(lab) > 20
    assert (lab["exit_date"] > lab["signal_date"]).all()
    assert ev["label_r"].isna().any()      # overlapping signals stay unlabeled
    assert set(sf.FEATURES) <= set(ev.columns)


def test_walk_forward_is_purged(monkeypatch) -> None:
    ctx = _ctx(n=2400)
    ev = sf.build_events(ctx, spy=None, vix=None)
    seen = []

    def spy_fit(X, y, kind="gbm"):
        seen.append(X["exit_date"].max())
        return sf.fit_model(X, y, kind)

    out, folds = sf.walk_forward(ev, first_test_year=2008, min_train=40, fit=spy_fit)
    ran = [f for f in folds if "skipped" not in f]
    assert ran, "expected at least one fold to train"
    for f, last_exit in zip(ran, seen):
        cutoff = pd.Timestamp(f"{f['year']}-01-01", tz="UTC") - pd.Timedelta(days=sf.EMBARGO_DAYS)
        assert last_exit < cutoff
    scored = out[out["p"].notna()]
    assert (scored["signal_date"].dt.year >= 2008).all()
    assert (scored["signal_date"].dt.year == scored["fold"]).all()
    assert out.loc[out["signal_date"].dt.year < 2008, "p"].isna().all()


def test_filter_map_blocks_signals_in_engine() -> None:
    ctx = _ctx()
    ev = sf.build_events(ctx, spy=None, vix=None)
    ev["p"], ev["threshold"] = 0.0, 0.5            # reject everything
    res = eng.simulate(ctx["data"], ctx["signals"], ctx["rules"], ctx["cfg"],
                       entry_filter=sf.filter_map(ev), mc_sims=0)
    assert res.trades == []
    assert res.diagnostics["skipped_filter"] > 0
    assert res.diagnostics["filter_missing"] == 0


def test_evaluation_runs_and_oracle_beats_baseline() -> None:
    ctx = _ctx(n=2400)
    out = sf.evaluate_on_context(ctx, spy=None, vix=None, first_test_year=2008, mc_sims=0, min_train=40)
    c = out["comparison"]
    assert out["verdict"]["label"] in {"helps", "hurts", "no_improvement", "insufficient_data"}
    assert c["oracle_filter"]["sharpe"] > c["baseline"]["sharpe"]
    assert c["filter_missing"] == 0
    assert out["classification"]["auc"]["n"] > 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_sharpe_diff_ci_identical_series_is_zero() -> None:
    rng = np.random.default_rng(1)
    eq = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 1500)))
    ci = sf.sharpe_diff_ci(eq, eq)
    assert ci["diff"] == 0 and ci["lo"] <= 0 <= ci["hi"]


def test_sharpe_diff_ci_detects_clear_improvement() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, 2500)
    a = pd.Series(100 * np.cumprod(1 + noise))
    b = pd.Series(100 * np.cumprod(1 + noise + 0.002))
    assert sf.sharpe_diff_ci(a, b)["lo"] > 0


def test_filter_verdict_rules() -> None:
    good_auc = {"auc": 0.6, "lo": 0.55, "hi": 0.65}
    assert sf.filter_verdict({"lo": 0.1, "hi": 0.3}, good_auc)["label"] == "helps"
    assert sf.filter_verdict({"lo": 0.1, "hi": 0.3}, {"auc": 0.51, "lo": 0.49, "hi": 0.53})["label"] == "no_improvement"
    assert sf.filter_verdict({"lo": -0.4, "hi": -0.1}, good_auc)["label"] == "hurts"
    assert sf.filter_verdict({"lo": None, "hi": None}, good_auc)["label"] == "insufficient_data"


# ---------------------------------------------------------------------------
# bot integration
# ---------------------------------------------------------------------------

def test_bot_ml_filter_skips_low_score(monkeypatch) -> None:
    monkeypatch.setattr(sf, "score_latest", lambda *a, **k: {"p": 0.31, "threshold": 0.42, "pass": False})
    out = bot._apply_ml_filter("AAA", {"symbol": "AAA", "action": "BUY", "reason": "x"},
                               {"kind": "swing_breakout", "ml_filter": True}, ["AAA", "BBB"])
    assert out["action"] == "FLAT" and "ML filter skip" in out["reason"]


def test_bot_ml_filter_fails_open(monkeypatch) -> None:
    def boom(*a, **k):
        raise ValueError("not enough trades")
    monkeypatch.setattr(sf, "score_latest", boom)
    out = bot._apply_ml_filter("AAA", {"symbol": "AAA", "action": "BUY", "reason": "x"},
                               {"kind": "rsi2_meanrev", "ml_filter": True}, None)
    assert out["action"] == "BUY" and "unavailable" in out["reason"]


def test_run_live_reports_ml_skip(monkeypatch) -> None:
    from tests.test_trading_bot import FakeBroker
    fb = FakeBroker()
    monkeypatch.setattr(bot, "latest_signal", lambda sym, dsl=None, **k: {
        "symbol": sym, "action": "FLAT", "reason": "ML filter skip",
        "ml_filter": {"p": 0.3, "threshold": 0.4, "pass": False}})
    res = bot.run_live(universe=["AAA"], dsl=None, execution=None, run_state=None, broker_mod=fb)
    assert [a["action"] for a in res["actions"]] == ["ML_SKIPPED"]
    assert fb.orders == []
