"""ML signal filter for swing setups ("meta-labeling").

The swing setup (the *primary* model) decides that a trade is possible. This
module trains a *secondary* model that looks at the context of each signal and
predicts whether that particular trade will end profitable, then skips the
ones it rates below average. It never creates trades and never changes stops,
targets or exits — it can only decline signals. (López de Prado, *Advances in
Financial Machine Learning*, ch. 3.)

Pre-declared design (fixed before looking at results, so the evaluation is a
test and not a search):

* **Events**: every bar on which the setup fires an entry signal.
* **Label**: the R-multiple of that trade under the setup's own exit rules,
  from the same engine, net of costs, simulated one symbol at a time. Target
  = R > 0. Signals that overlap an open trade in the same symbol have no label
  (they are still scored).
* **Features** (all from bars at or before the signal close): momentum over
  5/21/63/126/252 days, volatility level and trend, ATR %, distance from the
  50/200-day averages and from the 55/252-day extremes in ATR units, RSI(14),
  opening gap, volume trend, how many symbols signalled the same day, and the
  market backdrop (SPY trend/return/volatility, VIX level/change, % of the
  universe above its 200-day average). Directional features are sign-flipped
  for shorts.
* **Model**: shallow gradient boosting (depth 2, 200 trees, strong
  regularisation). A regularised logistic regression is reported alongside as
  a sanity check, not as a second chance.
* **Walk-forward**: retrain every January 1 from `first_test_year`, on events
  whose trade had *closed* at least `embargo_days` before the cut-off (purged:
  no training label overlaps the test year). Predict that year only.
* **Decision rule**: take the signal if P(win) ≥ the training base rate.
* **Verdict**: "helps" only if the paired block-bootstrap 90% interval of the
  Sharpe improvement is above zero *and* the out-of-sample AUC interval is
  above 0.5.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.backtest import swing_engine as eng
from app.services import indicators as ind
from app.services import swing
from app.services import swing_data as sd

SYMBOL_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "vol_20", "vol_ratio", "atr_pct",
    "dist_sma200_atr", "dist_sma50_atr", "dist_ext55_atr", "dist_ext252", "rsi14",
    "gap_atr", "volume_ratio",
]
MARKET_FEATURES = ["spy_above_200", "spy_ret_63", "spy_vol_20", "vix", "vix_chg_20", "breadth_200"]
EVENT_FEATURES = ["side", "signals_today"]
FEATURES = SYMBOL_FEATURES + MARKET_FEATURES + EVENT_FEATURES
# Momentum-like features that mean the opposite for a short.
_DIRECTIONAL = ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "dist_sma200_atr",
                "dist_sma50_atr", "gap_atr"]

GBM_PARAMS = dict(max_depth=2, n_estimators=200, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  eval_metric="logloss", random_state=0, n_jobs=2)
EMBARGO_DAYS = 10
MIN_TRAIN = 150

_MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "swing_filter"
_MODEL_TTL_S = 24 * 3600


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features from bars at or before each bar (no look-ahead).

    Returns both long- and short-oriented extreme distances; `event_rows`
    picks the right one per side.
    """
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    lr = np.log(c).diff()
    atr = ind.atr(h, l, c, 20)
    f = pd.DataFrame(index=df.index)
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    sd20, sd100 = lr.rolling(20).std(), lr.rolling(100).std()
    f["vol_20"] = sd20 * math.sqrt(252)
    f["vol_ratio"] = sd20 / sd100
    f["atr_pct"] = atr / c
    f["dist_sma200_atr"] = (c - ind.sma(c, 200)) / atr
    f["dist_sma50_atr"] = (c - ind.sma(c, 50)) / atr
    f["dist_high55_atr"] = (c - h.rolling(55).max().shift(1)) / atr
    f["dist_low55_atr"] = (c - l.rolling(55).min().shift(1)) / atr
    f["dist_high252"] = c / h.rolling(252).max() - 1
    f["dist_low252"] = c / l.rolling(252).min() - 1
    f["rsi14"] = ind.rsi(c, 14)
    f["gap_atr"] = (o - c.shift(1)) / atr
    v20, v100 = v.rolling(20).mean(), v.rolling(100).mean()
    f["volume_ratio"] = (v20 / v100).where(v100 > 0)
    return f.replace([np.inf, -np.inf], np.nan)


def market_features(cal: pd.DatetimeIndex, data: dict[str, pd.DataFrame],
                    spy: pd.DataFrame | None, vix: pd.DataFrame | None, *, lag: int = 0) -> pd.DataFrame:
    """Market backdrop per calendar date, forward-filled (backward-looking).

    `lag` shifts everything by N sessions. FX uses lag=1: Yahoo's FX daily
    bars appear to be keyed one session late, so same-date US data could leak.
    """
    m = pd.DataFrame(index=cal)
    if spy is not None and not spy.empty:
        sc = spy["close"]
        slr = np.log(sc).diff()
        spyf = pd.DataFrame({
            "spy_above_200": (sc > ind.sma(sc, 200)).astype(float).where(ind.sma(sc, 200).notna()),
            "spy_ret_63": np.log(sc / sc.shift(63)),
            "spy_vol_20": slr.rolling(20).std() * math.sqrt(252),
        })
        m = m.join(spyf.reindex(cal, method="ffill", limit=5) if not spyf.index.equals(cal)
                   else spyf)
    if vix is not None and not vix.empty:
        vc = vix["close"]
        vf = pd.DataFrame({"vix": vc, "vix_chg_20": np.log(vc / vc.shift(20))})
        m = m.join(vf.reindex(cal, method="ffill", limit=5))
    above = []
    for df in data.values():
        c = df["close"]
        s200 = ind.sma(c, 200)
        above.append((c > s200).astype(float).where(s200.notna()).reindex(cal, method="ffill", limit=5))
    if above:
        m["breadth_200"] = pd.concat(above, axis=1).mean(axis=1, skipna=True)
    for col in MARKET_FEATURES:
        if col not in m:
            m[col] = np.nan
    m = m[MARKET_FEATURES]
    return m.shift(lag) if lag else m


def _signals_today(signals: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, allow_short: bool) -> pd.Series:
    cnt = pd.Series(0.0, index=cal)
    for sig in signals.values():
        fired = sig["long_entry"] | (sig["short_entry"] if allow_short else False)
        cnt = cnt.add(fired.astype(float).reindex(cal).fillna(0.0), fill_value=0.0)
    return cnt


def event_rows(sym: str, feats: pd.DataFrame, sig: pd.DataFrame, market: pd.DataFrame,
               crowd: pd.Series, allow_short: bool) -> pd.DataFrame:
    """One row per entry signal of `sym` with its side-oriented features."""
    long_ = sig["long_entry"]
    short_ = sig["short_entry"] & ~long_ if allow_short else pd.Series(False, index=sig.index)
    idx = sig.index[long_ | short_]
    if len(idx) == 0:
        return pd.DataFrame(columns=["symbol", "signal_date", *FEATURES])
    side = pd.Series(np.where(long_.loc[idx], 1.0, -1.0), index=idx)
    f = feats.loc[idx].copy()
    f["dist_ext55_atr"] = np.where(side > 0, f["dist_high55_atr"], -f["dist_low55_atr"])
    f["dist_ext252"] = np.where(side > 0, f["dist_high252"], -f["dist_low252"])
    for col in _DIRECTIONAL:
        f[col] = f[col] * side
    f["rsi14"] = np.where(side > 0, f["rsi14"], 100 - f["rsi14"])
    f = f[SYMBOL_FEATURES]
    f = f.join(market.reindex(idx))
    f["side"] = side
    f["signals_today"] = crowd.reindex(idx).to_numpy()
    f.insert(0, "signal_date", idx)
    f.insert(0, "symbol", sym)
    return f.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Events + labels
# ---------------------------------------------------------------------------

def _label_config(cfg: eng.EngineConfig) -> eng.EngineConfig:
    """Single-symbol, unconstrained sizing: R depends on the price path only."""
    return replace(cfg, max_positions=1, max_position_pct=1e9, max_gross_leverage=1e9,
                   max_heat_pct=1e9, whole_units=False, min_notional=0.0, start=None, end=None)


def build_events(ctx: dict[str, Any], *, spy: pd.DataFrame | None, vix: pd.DataFrame | None) -> pd.DataFrame:
    """All signal events with features and (where resolvable) trade labels."""
    data, signals, cfg = ctx["data"], ctx["signals"], ctx["cfg"]
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    lag = 1 if ctx["asset_class"] == "fx" else 0
    market = market_features(cal, data, spy, vix, lag=lag)
    crowd = _signals_today(signals, cal, cfg.allow_short)
    lcfg = _label_config(cfg)
    rows = []
    for sym, df in data.items():
        ev = event_rows(sym, symbol_features(df), signals[sym], market, crowd, cfg.allow_short)
        if ev.empty:
            continue
        try:
            res = eng.simulate({sym: df}, {sym: signals[sym]}, ctx["rules"], lcfg,
                               earnings={sym: (ctx["earnings"] or {}).get(sym)},
                               long_regime=ctx["regime"], mc_sims=0)
            trades = res.trades
        except ValueError:
            trades = []
        pos = {d: i for i, d in enumerate(df.index)}
        lab = {}
        for t in trades:
            if t.exit_reason == "end_of_data":
                continue  # unresolved: the trade is still open
            i = pos.get(pd.Timestamp(t.entry_date, tz="UTC"))
            if i is None or i == 0:
                continue
            lab[df.index[i - 1]] = (t.r_multiple, pd.Timestamp(t.exit_date, tz="UTC"))
        ev["label_r"] = [lab.get(d, (np.nan, pd.NaT))[0] for d in ev["signal_date"]]
        ev["exit_date"] = [lab.get(d, (np.nan, pd.NaT))[1] for d in ev["signal_date"]]
        rows.append(ev)
    if not rows:
        return pd.DataFrame(columns=["symbol", "signal_date", *FEATURES, "label_r", "exit_date"])
    events = pd.concat(rows, ignore_index=True)
    events["label_win"] = np.where(events["label_r"].notna(), (events["label_r"] > 0).astype(float), np.nan)
    return events.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Models + walk-forward
# ---------------------------------------------------------------------------

def fit_model(X: pd.DataFrame, y: np.ndarray, kind: str = "gbm"):
    if kind == "gbm":
        from xgboost import XGBClassifier
        m = XGBClassifier(**GBM_PARAMS)
        m.fit(X[FEATURES].to_numpy(dtype=float), y.astype(int))
        return m
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X[FEATURES].to_numpy(dtype=float), y.astype(int))
        return m
    raise ValueError(kind)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES].to_numpy(dtype=float))[:, 1]


def walk_forward(events: pd.DataFrame, *, first_test_year: int, kind: str = "gbm",
                 embargo_days: int = EMBARGO_DAYS, min_train: int = MIN_TRAIN,
                 fit=fit_model) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Yearly expanding-window refits with purged training labels.

    Returns (events + columns p / threshold / fold, fold summaries).
    """
    ev = events.copy()
    ev["p"], ev["threshold"], ev["fold"] = np.nan, np.nan, np.nan
    if ev.empty:
        return ev, []
    years = sorted({d.year for d in ev["signal_date"] if d.year >= first_test_year})
    folds = []
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01", tz="UTC") - pd.Timedelta(days=embargo_days)
        train = ev[ev["label_r"].notna() & (ev["exit_date"] < cutoff)]
        test_mask = ev["signal_date"].dt.year == y
        info = {"year": y, "n_train": int(len(train)), "n_test": int(test_mask.sum()),
                "train_last_exit": train["exit_date"].max().date().isoformat() if len(train) else None}
        if len(train) < min_train or train["label_win"].nunique() < 2:
            info["skipped"] = "not enough resolved training trades"
            folds.append(info)
            continue
        model = fit(train, train["label_win"].to_numpy(), kind)
        thr = float(train["label_win"].mean())
        ev.loc[test_mask, "p"] = predict(model, ev.loc[test_mask])
        ev.loc[test_mask, "threshold"] = thr
        ev.loc[test_mask, "fold"] = y
        info["threshold"] = round(thr, 4)
        folds.append(info)
    return ev, folds


def filter_map(ev: pd.DataFrame) -> dict[str, pd.Series]:
    """symbol → bool Series over signal dates (True = take the signal)."""
    scored = ev[ev["p"].notna()]
    out = {}
    for sym, g in scored.groupby("symbol"):
        out[sym] = pd.Series((g["p"] >= g["threshold"]).to_numpy(), index=pd.DatetimeIndex(g["signal_date"]))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 11) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    if len(y) < 30 or len(np.unique(y)) < 2:
        return {"auc": None, "lo": None, "hi": None, "n": int(len(y))}
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
    p_le = float((np.asarray(boots) <= 0.5).mean()) if boots else None
    return {"auc": round(auc, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_le_0_5": round(p_le, 4) if p_le is not None else None, "n": int(len(y))}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, *, block: int = 20, n_boot: int = 1000,
                   seed: int = 5) -> dict[str, Any]:
    """Paired circular block bootstrap of Sharpe(b) − Sharpe(a) on daily returns."""
    ra, rb = a.pct_change(), b.pct_change()
    df = pd.concat([ra, rb], axis=1, keys=["a", "b"]).dropna()
    n = len(df)
    if n < 250:
        return {"diff": None, "lo": None, "hi": None}

    def sh(x: np.ndarray) -> float:
        s = x.std()
        return float(np.sqrt(252) * x.mean() / s) if s > 0 else 0.0

    A, B = df["a"].to_numpy(), df["b"].to_numpy()
    point = sh(B) - sh(A)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for j in range(n_boot):
        starts = rng.integers(0, n, k)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        diffs[j] = sh(B[idx]) - sh(A[idx])
    lo, hi = np.percentile(diffs, [5, 95])
    return {"diff": round(point, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "p_le_0": round(float((diffs <= 0).mean()), 4),  # one-sided bootstrap p
            "block_days": block, "n_days": n}


def _kept_vs_skipped(ev: pd.DataFrame) -> dict[str, Any]:
    lab = ev[ev["p"].notna() & ev["label_r"].notna()]
    out = {}
    for name, g in (("kept", lab[lab["p"] >= lab["threshold"]]), ("skipped", lab[lab["p"] < lab["threshold"]])):
        out[name] = {"n": int(len(g)),
                     "win_rate_pct": round(float(g["label_win"].mean() * 100), 1) if len(g) else None,
                     "mean_r": round(float(g["label_r"].mean()), 3) if len(g) else None}
    return out


def _quintiles(ev: pd.DataFrame) -> list[dict[str, Any]]:
    lab = ev[ev["p"].notna() & ev["label_r"].notna()]
    if len(lab) < 50:
        return []
    q = pd.qcut(lab["p"].rank(method="first"), 5, labels=False)
    out = []
    for k, g in lab.groupby(q):
        out.append({"quintile": int(k) + 1, "n": int(len(g)),
                    "p_mean": round(float(g["p"].mean()), 3),
                    "win_rate_pct": round(float(g["label_win"].mean() * 100), 1),
                    "mean_r": round(float(g["label_r"].mean()), 3)})
    return out


def _importance(events: pd.DataFrame, top: int = 8) -> list[dict[str, Any]]:
    lab = events[events["label_r"].notna()]
    if len(lab) < MIN_TRAIN:
        return []
    m = fit_model(lab, lab["label_win"].to_numpy(), "gbm")
    imp = getattr(m, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(imp)[::-1][:top]
    return [{"feature": FEATURES[i], "importance": round(float(imp[i]), 4)} for i in order]


def filter_verdict(sharpe_ci: dict[str, Any], auc: dict[str, Any]) -> dict[str, Any]:
    lo, hi = sharpe_ci.get("lo"), sharpe_ci.get("hi")
    auc_lo = auc.get("lo")
    if lo is None or auc.get("auc") is None:
        return {"label": "insufficient_data", "summary": "Not enough out-of-sample data to judge."}
    if lo > 0 and auc_lo is not None and auc_lo > 0.5:
        return {"label": "helps", "summary": (
            "The filter improved risk-adjusted returns out of sample, and its predictions "
            "ranked winners above losers better than chance.")}
    if hi < 0:
        return {"label": "hurts", "summary": "The filter made the strategy worse out of sample."}
    return {"label": "no_improvement", "summary": (
        "Out of sample, the filtered strategy is not distinguishable from the unfiltered one. "
        "Keep the filter off.")}


def evaluate_on_context(ctx: dict[str, Any], *, spy: pd.DataFrame | None, vix: pd.DataFrame | None,
                        first_test_year: int = 2010, kind: str = "gbm", mc_sims: int = 1000,
                        max_trades: int = 200, min_train: int = MIN_TRAIN) -> dict[str, Any]:
    t0 = time.time()
    events = build_events(ctx, spy=spy, vix=vix)
    ev, folds = walk_forward(events, first_test_year=first_test_year, kind=kind, min_train=min_train)
    cfg_eval = replace(ctx["cfg"], start=f"{first_test_year}-01-01")
    common = dict(earnings=ctx["earnings"], long_regime=ctx["regime"],
                  benchmark=ctx["benchmark"], benchmark_symbol=ctx["benchmark_symbol"], mc_sims=mc_sims)
    base = eng.simulate(ctx["data"], ctx["signals"], ctx["rules"], cfg_eval, **common)
    filt = eng.simulate(ctx["data"], ctx["signals"], ctx["rules"], cfg_eval,
                        entry_filter=filter_map(ev), **common)

    # Reference points for the same plumbing: a filter that knows each
    # trade's outcome (the ceiling) and one that keeps the same fraction of
    # signals at random (what "no skill" looks like).
    scored = ev[ev["p"].notna()]
    keep_frac = float((scored["p"] >= scored["threshold"]).mean()) if len(scored) else 1.0
    rng = np.random.default_rng(0)
    oracle_map, random_map = {}, {}
    for sym, g in scored.groupby("symbol"):
        idx = pd.DatetimeIndex(g["signal_date"])
        oracle_map[sym] = pd.Series(((g["label_r"] > 0) | g["label_r"].isna()).to_numpy(), index=idx)
        random_map[sym] = pd.Series(rng.random(len(g)) < keep_frac, index=idx)
    oracle = eng.simulate(ctx["data"], ctx["signals"], ctx["rules"], cfg_eval,
                          entry_filter=oracle_map, **{**common, "mc_sims": 0})
    rand = eng.simulate(ctx["data"], ctx["signals"], ctx["rules"], cfg_eval,
                        entry_filter=random_map, **{**common, "mc_sims": 0})

    oos = ev[ev["p"].notna() & ev["label_r"].notna()]
    auc = _auc_ci(oos["label_win"].to_numpy(), oos["p"].to_numpy())
    ic = (float(pd.Series(oos["p"]).corr(pd.Series(oos["label_r"]), method="spearman"))
          if len(oos) >= 30 else None)
    s_ci = sharpe_diff_ci(base.daily_equity, filt.daily_equity)

    # Sanity check with a different model family (reported, not used).
    ev_lr, _ = walk_forward(events, first_test_year=first_test_year, kind="logistic", min_train=min_train)
    oos_lr = ev_lr[ev_lr["p"].notna() & ev_lr["label_r"].notna()]
    logistic = {"auc": _auc_ci(oos_lr["label_win"].to_numpy(), oos_lr["p"].to_numpy()),
                "kept_vs_skipped": _kept_vs_skipped(ev_lr)}

    def brief(r: eng.SwingResult) -> dict[str, Any]:
        m = r.metrics
        return {k: m.get(k) for k in ("cagr_pct", "sharpe", "sharpe_t_stat", "max_drawdown_pct",
                                      "n_trades", "win_rate_pct", "expectancy_r", "profit_factor")}

    return {
        **swing.report_meta(ctx),
        "model": kind, "first_test_year": first_test_year,
        "n_events": int(len(events)), "n_labeled": int(events["label_r"].notna().sum()),
        "folds": folds,
        "classification": {"auc": auc, "rank_ic": round(ic, 4) if ic is not None else None,
                           "kept_vs_skipped": _kept_vs_skipped(ev), "quintiles": _quintiles(ev)},
        "logistic_check": logistic,
        "comparison": {"baseline": brief(base), "filtered": brief(filt), "sharpe_diff": s_ci,
                       "oracle_filter": brief(oracle), "random_filter": brief(rand),
                       "keep_fraction": round(keep_frac, 3),
                       "skipped_by_filter": filt.diagnostics.get("skipped_filter"),
                       "filter_missing": filt.diagnostics.get("filter_missing")},
        "verdict": filter_verdict(s_ci, auc),
        "top_features": _importance(events),
        "baseline": base.as_dict(max_trades=max_trades),
        "filtered": filt.as_dict(max_trades=max_trades),
        "runtime_s": round(time.time() - t0, 1),
    }


def _market_series(load_from: str | None, max_age_s: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw = {"start": load_from} if max_age_s is None else {"start": load_from, "max_age_s": max_age_s}
    return sd.daily_bars("SPY", **kw), sd.daily_bars("^VIX", **kw)


def evaluate(*, setup: str, universe: str | None = None, symbols: list[str] | None = None,
             params: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
             first_test_year: int = 2010, use_earnings: bool = True, kind: str = "gbm") -> dict[str, Any]:
    """Load data (same as swing.backtest) and run the walk-forward evaluation."""
    ctx = swing.prepare(setup=setup, universe=universe, symbols=symbols, params=params,
                        config=config, start="2005-01-01", use_earnings=use_earnings)
    spy, vix = _market_series(ctx["load_from"])
    spy, vix = swing.completed_bars(spy, "SPY"), swing.completed_bars(vix, "^VIX")
    return evaluate_on_context(ctx, spy=spy, vix=vix, first_test_year=first_test_year, kind=kind)


# ---------------------------------------------------------------------------
# Live scoring
# ---------------------------------------------------------------------------

_MEM: dict[str, tuple[float, dict[str, Any]]] = {}


def _key(setup: str, symbols: list[str], params: dict[str, Any] | None) -> str:
    raw = f"{setup}|{','.join(sorted(s.upper() for s in symbols))}|{sorted((params or {}).items())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def live_model(setup: str, symbols: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model trained on every resolved event up to now (cached 24h)."""
    key = _key(setup, symbols, params)
    hit = _MEM.get(key)
    if hit and time.time() - hit[0] < _MODEL_TTL_S:
        return hit[1]
    path = _MODEL_DIR / f"{setup}_{key}.pkl"
    try:
        if path.exists() and time.time() - path.stat().st_mtime < _MODEL_TTL_S:
            with path.open("rb") as f:
                bundle = pickle.load(f)
            _MEM[key] = (time.time(), bundle)
            return bundle
    except Exception:  # noqa: BLE001 — a bad cache file is just a miss
        pass
    ctx = swing.prepare(setup=setup, symbols=symbols, params=params, start="2005-01-01")
    spy, vix = _market_series(ctx["load_from"])
    events = build_events(ctx, spy=swing.completed_bars(spy, "SPY"), vix=swing.completed_bars(vix, "^VIX"))
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=EMBARGO_DAYS)
    train = events[events["label_r"].notna() & (events["exit_date"] < cutoff)]
    if len(train) < MIN_TRAIN or train["label_win"].nunique() < 2:
        raise ValueError(f"only {len(train)} resolved trades to learn from (need {MIN_TRAIN})")
    bundle = {"model": fit_model(train, train["label_win"].to_numpy(), "gbm"),
              "threshold": float(train["label_win"].mean()), "n_train": int(len(train)),
              "trained_at": datetime.now(timezone.utc).isoformat(), "setup": setup,
              "symbols": sorted(ctx["data"]), "params": params or {}}
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(bundle, f)
    except Exception:  # noqa: BLE001
        pass
    _MEM[key] = (time.time(), bundle)
    return bundle


def score_latest(symbol: str, setup: str, symbols: list[str], params: dict[str, Any] | None = None,
                 side: int = 1) -> dict[str, Any]:
    """P(win) for a signal on `symbol`'s latest completed bar, with pass/fail."""
    bundle = live_model(setup, symbols, params)
    ctx = swing.prepare(setup=setup, symbols=sorted(set(symbols) | {symbol.upper()}), params=params,
                        start=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=500)).date().isoformat(),
                        use_earnings=False, max_age_s=3600)
    sym = symbol.upper()
    data = {k: swing.completed_bars(v, k) for k, v in ctx["data"].items()}
    df = data[sym]
    cal = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    spy, vix = _market_series(ctx["load_from"], max_age_s=3600)
    lag = 1 if ctx["asset_class"] == "fx" else 0
    market = market_features(cal, data, swing.completed_bars(spy, "SPY"),
                             swing.completed_bars(vix, "^VIX"), lag=lag)
    signals = {k: swing.build_signals(v, setup, bundle["params"] or None) for k, v in data.items()}
    crowd = _signals_today(signals, cal, ctx["cfg"].allow_short)
    last = df.index[-1]
    sig = signals[sym].copy()
    sig.loc[:, ["long_entry", "short_entry"]] = False
    sig.loc[last, "long_entry" if side > 0 else "short_entry"] = True
    row = event_rows(sym, symbol_features(df), sig, market, crowd, ctx["cfg"].allow_short)
    p = float(predict(bundle["model"], row)[0])
    return {"p": round(p, 4), "threshold": round(bundle["threshold"], 4), "pass": p >= bundle["threshold"],
            "as_of": last.date().isoformat(), "n_train": bundle["n_train"],
            "trained_at": bundle["trained_at"]}
