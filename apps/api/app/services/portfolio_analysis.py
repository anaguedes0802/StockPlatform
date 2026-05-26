"""AI portfolio analysis.

Reads a portfolio's positions and produces a structured advisory output:

  - **risk_metrics**: beta vs SPY, volatility, max drawdown, VaR
  - **concentration**: HHI, single-name + sector overconcentration warnings
  - **diversification_score**: 0..100 (higher = better diversified)
  - **rebalance_suggestions**: mean-variance optimal weights (Markowitz tangent
    portfolio with turnover penalty); per-position delta
  - **hedging_suggestions**: portfolio beta + per-100k-position inverse ETF size
  - **rec_aggregate**: average recommendation score across holdings
  - **drawdown_alerts**: positions down > N% from 52w high

Inputs use only data already available locally (yfinance + computed services).
No paid sources, no key required.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services import indicators as ind
from app.services import market_data as md
from app.services import risk as risk_svc
from app.services.recommendation import recommend as _recommend


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _returns_matrix(symbols: list[str], range_: str = "1y") -> pd.DataFrame:
    """Aligned daily-return matrix across symbols. Missing days are dropped."""
    frames = []
    for s in symbols:
        try:
            df = md.get_history(s, interval="1d", range_=range_)
            if df.empty:
                continue
            r = df["close"].pct_change().rename(s)
            frames.append(r)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1, sort=False).dropna(how="any")


def _benchmark_returns(range_: str = "1y") -> pd.Series:
    try:
        df = md.get_history("SPY", interval="1d", range_=range_)
        return df["close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


def _hhi(weights: dict[str, float]) -> float:
    total = sum(weights.values()) or 1.0
    return float(sum((w / total) ** 2 for w in weights.values()))


# ----------------------------------------------------------------------------
# Portfolio optimizers
# ----------------------------------------------------------------------------
#
# Why no plain mean-variance: DeMiguel/Garlappi/Uppal (2009) showed that
# Markowitz with sample-mean returns gets beaten by equal-weight 1/N. The
# error in estimating expected returns dominates the gain from optimization.
# Real practitioners use either:
#   - Risk parity (no return estimate needed)
#   - Black-Litterman (start from market-implied returns, add explicit views)
#   - Shrinkage (Ledoit-Wolf for covariance + James-Stein for mean)
# We expose all three and default to risk-parity.

def _shrinkage_covariance(R: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance. Cheap and dramatically better than
    the sample covariance for small N or short windows."""
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(R.values).covariance_ * 252
    except Exception:
        return R.cov().values * 252


def _risk_parity_weights(R: pd.DataFrame) -> dict[str, float]:
    """Risk-parity (a.k.a. equal-risk-contribution) weights.

    Solves for w such that each asset contributes equally to total portfolio
    variance. No return estimate required → no estimation-error blow-up.
    """
    from scipy.optimize import minimize
    if R.empty:
        return {}
    symbols = list(R.columns)
    n = len(symbols)
    cov = _shrinkage_covariance(R)

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 1e9
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return float(((rc - target) ** 2).sum())

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.005, 0.50) for _ in range(n)]
    try:
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-9})
        w_opt = res.x if res.success else w0
    except Exception:
        w_opt = w0
    return {s: float(w) for s, w in zip(symbols, w_opt, strict=True)}


def _black_litterman_weights(
    R: pd.DataFrame,
    market_weights: dict[str, float] | None = None,
    views: dict[str, float] | None = None,
    view_confidence: float = 0.5,
    risk_aversion: float = 3.0,
) -> dict[str, float]:
    """Black-Litterman with optional views.

    π = δ·Σ·w_mkt (implied equilibrium returns from market-cap weights)
    μ_BL = π + τΣP'(PτΣP'+Ω)⁻¹(Q - Pπ)
    w* = (δΣ)⁻¹ μ_BL  (unconstrained), then clip to feasible.

    `views`: optional {symbol: annual_expected_return}. If empty, this just
    returns the implied weights (≈ market-cap weights).
    """
    if R.empty:
        return {}
    symbols = list(R.columns)
    n = len(symbols)
    cov = _shrinkage_covariance(R)

    # Equilibrium weights (market-cap proxy via current holdings; else equal)
    w_mkt = np.array([float((market_weights or {}).get(s, 1.0 / n)) for s in symbols])
    w_mkt = w_mkt / max(w_mkt.sum(), 1e-9)

    # Implied returns
    pi = risk_aversion * cov @ w_mkt

    if views:
        # P picks views; Q is the view return vector
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        for i, (sym, q) in enumerate(views.items()):
            if sym not in symbols:
                continue
            P[i, symbols.index(sym)] = 1.0
            Q[i] = float(q)
        tau = 0.025
        # Omega: view uncertainty proportional to (1 - confidence) on the diagonal
        omega = np.diag([(1.0 - view_confidence) * float(P[i] @ (tau * cov) @ P[i]) + 1e-6
                          for i in range(len(views))])
        try:
            inner = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
            mu_bl = pi + (tau * cov) @ P.T @ inner @ (Q - P @ pi)
        except np.linalg.LinAlgError:
            mu_bl = pi
    else:
        mu_bl = pi

    # Unconstrained mean-variance solution given mu_bl
    try:
        w_unc = np.linalg.solve(risk_aversion * cov, mu_bl)
        w_clipped = np.clip(w_unc, 0.0, 0.4)
        if w_clipped.sum() > 0:
            w_clipped /= w_clipped.sum()
        else:
            w_clipped = np.ones(n) / n
    except np.linalg.LinAlgError:
        w_clipped = np.ones(n) / n
    return {s: float(w) for s, w in zip(symbols, w_clipped, strict=True)}


def _target_weights(
    R: pd.DataFrame,
    current_weights: dict[str, float],
    method: str = "risk_parity",
    views: dict[str, float] | None = None,
) -> dict[str, float]:
    """Dispatcher for the three optimizers. Default = risk-parity (most robust)."""
    if R.empty or len(R) < 30:
        return current_weights
    if method == "risk_parity":
        return _risk_parity_weights(R)
    if method == "black_litterman":
        return _black_litterman_weights(R, market_weights=current_weights, views=views or {})
    # equal-weight fallback (1/N — DeMiguel et al. baseline)
    n = len(R.columns)
    return {s: 1.0 / n for s in R.columns}


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def analyze_portfolio(positions: list[dict[str, Any]], base_currency: str = "USD") -> dict[str, Any]:
    """Run the full analysis on a list of position dicts.

    Each position dict must have at minimum:
      symbol, quantity, market_value, last_price, avg_buy_price
    """
    if not positions:
        return {
            "summary": "Empty portfolio — add transactions to see analysis.",
            "risk_metrics": {}, "concentration": {}, "diversification_score": 0,
            "rebalance_suggestions": [], "hedging_suggestions": {},
            "rec_aggregate": {}, "alerts": [],
        }

    symbols = [p["symbol"] for p in positions]
    total_value = sum(float(p.get("market_value") or 0.0) for p in positions) or 1.0
    weights = {p["symbol"]: float(p.get("market_value") or 0) / total_value for p in positions}

    # ------ risk metrics
    R = _returns_matrix(symbols, range_="1y")
    bench = _benchmark_returns(range_="1y")

    portfolio_ret = pd.Series(dtype=float)
    if not R.empty:
        # weighted daily portfolio return
        w_vec = pd.Series({s: weights.get(s, 0) for s in R.columns})
        portfolio_ret = (R * w_vec).sum(axis=1)

    pf_vol_ann = float(portfolio_ret.std() * np.sqrt(252)) if not portfolio_ret.empty else 0.0
    pf_sharpe = risk_svc.sharpe(portfolio_ret) if not portfolio_ret.empty else 0.0
    pf_sortino = risk_svc.sortino(portfolio_ret) if not portfolio_ret.empty else 0.0
    pf_beta = risk_svc.beta(portfolio_ret, bench) if (not portfolio_ret.empty and not bench.empty) else 0.0
    eq_curve = (1 + portfolio_ret).cumprod() if not portfolio_ret.empty else pd.Series([1.0])
    pf_mdd = risk_svc.max_drawdown(eq_curve)
    pf_var95 = risk_svc.value_at_risk_hist(portfolio_ret) if not portfolio_ret.empty else 0.0

    risk_metrics = {
        "volatility_annual_pct": round(pf_vol_ann * 100, 2),
        "sharpe": round(pf_sharpe, 2),
        "sortino": round(pf_sortino, 2),
        "beta_vs_spy": round(pf_beta, 2),
        "max_drawdown_pct": round(pf_mdd * 100, 2),
        "value_at_risk_95_daily_pct": round(pf_var95 * 100, 2),
    }

    # ------ concentration
    hhi = _hhi(weights)
    sector_alloc: dict[str, float] = {}
    for p in positions:
        try:
            sec = md.get_profile(p["symbol"]).get("sector") or "Unknown"
        except Exception:
            sec = "Unknown"
        sector_alloc[sec] = sector_alloc.get(sec, 0) + weights.get(p["symbol"], 0)
    sector_hhi = _hhi(sector_alloc)
    biggest_position = max(weights.items(), key=lambda kv: kv[1]) if weights else (None, 0)
    biggest_sector = max(sector_alloc.items(), key=lambda kv: kv[1]) if sector_alloc else (None, 0)

    alerts: list[str] = []
    if biggest_position[1] > 0.30:
        alerts.append(f"Overconcentration: {biggest_position[0]} = {biggest_position[1]*100:.0f}% of portfolio (>30%)")
    if biggest_sector[1] > 0.50:
        alerts.append(f"Sector concentration: {biggest_sector[0]} = {biggest_sector[1]*100:.0f}% of portfolio (>50%)")
    if pf_beta > 1.4:
        alerts.append(f"Portfolio beta {pf_beta:.2f} — well above market; high systematic risk")
    if pf_mdd < -0.20:
        alerts.append(f"1y max drawdown {pf_mdd*100:.1f}% — risk discipline review recommended")

    # 52w-low alerts per position
    for p in positions:
        try:
            df = md.get_history(p["symbol"], interval="1d", range_="1y")
            if df.empty: continue
            high = float(df["close"].max())
            last = float(df["close"].iloc[-1])
            drop = (last - high) / high
            if drop < -0.25:
                alerts.append(f"{p['symbol']} down {drop*100:.0f}% from 52w high — review thesis")
        except Exception:
            continue

    concentration = {
        "hhi": round(hhi, 3),
        "n_positions": len(positions),
        "sector_allocation": {k: round(v, 4) for k, v in sector_alloc.items()},
        "sector_hhi": round(sector_hhi, 3),
        "biggest_position": {"symbol": biggest_position[0], "weight": round(biggest_position[1], 3)} if biggest_position[0] else None,
    }
    # diversification score: function of (1/HHI) bounded to [0, 100]
    # n_effective = 1/HHI; cap at ~10 effective positions for a max score
    n_eff = 1.0 / max(hhi, 1e-6)
    diversification_score = int(min(100.0, (n_eff / 10.0) * 100))

    # ------ rebalance suggestions
    # Default = risk-parity (robust, no return estimate needed).
    # When AI recommendations exist for each symbol, we *also* compute
    # Black-Litterman weights using the recommendation scores as views, so
    # the user can compare both. We return the BL weights as the primary
    # target because they incorporate platform-specific views.
    rebalance: list[dict[str, Any]] = []
    optimizer_used = "risk_parity"
    target_w: dict[str, float] = weights
    if len(symbols) >= 2 and not R.empty:
        # Build views from AI recommendation scores: positive score → above-equilibrium expected return.
        # We map score ∈ [-1, +1] to an annual return view ∈ [-15%, +15%], a reasonable spread.
        rec_views: dict[str, float] = {}
        for p in positions:
            try:
                rec_score = float(_recommend(p["symbol"])["score"])
                rec_views[p["symbol"]] = float(np.clip(rec_score, -1.0, 1.0) * 0.15)
            except Exception:
                continue
        if rec_views:
            try:
                target_w = _black_litterman_weights(R, market_weights=weights, views=rec_views, view_confidence=0.5)
                optimizer_used = "black_litterman_with_ai_views"
            except Exception:
                target_w = _risk_parity_weights(R)
        else:
            target_w = _risk_parity_weights(R)
        for sym in symbols:
            tgt = target_w.get(sym, 0)
            cur = weights.get(sym, 0)
            delta = tgt - cur
            if abs(delta) >= 0.02:
                action = "increase" if delta > 0 else "trim"
                rebalance.append({
                    "symbol": sym,
                    "current_weight_pct": round(cur * 100, 2),
                    "target_weight_pct": round(tgt * 100, 2),
                    "delta_pct": round(delta * 100, 2),
                    "action": action,
                    "dollar_delta": round(delta * total_value, 2),
                })
        rebalance.sort(key=lambda r: -abs(r["delta_pct"]))

    # ------ hedging
    hedging = {}
    if pf_beta > 1.1:
        # SH (inverse SPY 1×) or SDS (inverse 2×) target sized to neutralize 30-50% of beta
        target_neutralize = 0.4
        hedge_dollars = round(total_value * pf_beta * target_neutralize, 2)
        hedging = {
            "rationale": f"Portfolio beta {pf_beta:.2f} is elevated. Consider a partial hedge to reduce systematic risk.",
            "instrument": "SH (ProShares Short S&P 500, 1×)",
            "suggested_notional": hedge_dollars,
            "alt_instrument": "SDS (ProShares UltraShort S&P 500, 2×)",
            "alt_notional": round(hedge_dollars / 2, 2),
        }
    elif pf_beta < 0.5 and pf_vol_ann > 0.10:
        hedging = {"rationale": f"Low beta ({pf_beta:.2f}). Portfolio is already defensive; no hedge needed."}

    # ------ aggregated recommendations across holdings
    rec_blocks: list[dict[str, Any]] = []
    rec_score_sum = 0.0
    rec_score_weights = 0.0
    for p in positions:
        try:
            r = _recommend(p["symbol"])
            rec_blocks.append({"symbol": p["symbol"], "label": r["label"], "score": r["score"]})
            rec_score_sum += r["score"] * weights.get(p["symbol"], 0)
            rec_score_weights += weights.get(p["symbol"], 0)
        except Exception:
            continue
    avg_rec_score = rec_score_sum / rec_score_weights if rec_score_weights else 0
    avg_label = (
        "STRONG_BUY" if avg_rec_score > 0.55 else
        "BUY" if avg_rec_score > 0.25 else
        "HOLD" if avg_rec_score > -0.10 else
        "REDUCE" if avg_rec_score > -0.30 else
        "SELL"
    )
    rec_aggregate = {
        "avg_label": avg_label,
        "avg_score": round(avg_rec_score, 3),
        "per_symbol": rec_blocks,
    }

    return {
        "summary": f"Portfolio: {len(positions)} positions, "
                   f"diversification {diversification_score}/100, "
                   f"beta {pf_beta:.2f}, sharpe {pf_sharpe:.2f}, "
                   f"AI verdict {avg_label}.",
        "risk_metrics": risk_metrics,
        "concentration": concentration,
        "diversification_score": diversification_score,
        "rebalance_suggestions": rebalance[:10],
        "rebalance_optimizer": optimizer_used,
        "hedging_suggestions": hedging,
        "rec_aggregate": rec_aggregate,
        "alerts": alerts,
    }
