"""Swing-trading evidence run: every setup × every universe, same engine.

Usage (from the repo root):
    apps/api/.venv/bin/python scripts/backtest_swing.py                # full grid
    apps/api/.venv/bin/python scripts/backtest_swing.py --setup breakout --universe fx_majors
    apps/api/.venv/bin/python scripts/backtest_swing.py --cost-stress 2  # double all costs

Parameters are the fixed defaults in app/services/swing.py — nothing here is
tuned. Writes a JSON summary to apps/api/artifacts/swing_results.json.

Read the output with the multiple-testing problem in mind: with 8 configs,
one of them looking good by luck is not surprising. A t-stat near 2 on one
config out of 8 is weak evidence; the Bonferroni-style bar is ~2.7.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

from app.services import swing  # noqa: E402

GRID = [
    ("breakout", "etfs"), ("pullback", "etfs"), ("rsi2", "etfs"),
    ("breakout", "us_large_caps"), ("pullback", "us_large_caps"), ("rsi2", "us_large_caps"),
    ("breakout", "fx_majors"), ("pullback", "fx_majors"),
]


def _row(setup: str, uni: str, r: dict) -> dict:
    m, b, v = r["metrics"], r.get("benchmark_metrics") or {}, r["verdict"]
    h = {x["label"]: x for x in r["halves"]}
    return {
        "setup": setup, "universe": uni, "start": m.get("start"), "end": m.get("end"),
        "cagr_pct": m.get("cagr_pct"), "sharpe": m.get("sharpe"),
        "max_dd_pct": m.get("max_drawdown_pct"), "sharpe_t": m.get("sharpe_t_stat"),
        "trades": m.get("n_trades"), "win_pct": m.get("win_rate_pct"),
        "exp_r": m.get("expectancy_r"), "pf": m.get("profit_factor"),
        "avg_bars": m.get("avg_bars_held"), "corr_bench": m.get("correlation_to_benchmark"),
        "h1_cagr": h.get("first_half", {}).get("cagr_pct"),
        "h2_cagr": h.get("second_half", {}).get("cagr_pct"),
        "h1_exp_r": h.get("first_half", {}).get("expectancy_r"),
        "h2_exp_r": h.get("second_half", {}).get("expectancy_r"),
        "mc_p_loss": r["monte_carlo"].get("prob_total_r_le_0_pct"),
        "bench_cagr": b.get("cagr_pct"), "bench_sharpe": b.get("sharpe"),
        "bench_dd": b.get("max_drawdown_pct"),
        "time_in_mkt": r["diagnostics"].get("time_in_market_pct"),
        "avg_lev": r["diagnostics"].get("avg_gross_leverage"),
        "earn_cov": f'{r["diagnostics"].get("earnings_coverage")}/{r["diagnostics"].get("symbols")}',
        "verdict": v["label"],
        "yearly": r["yearly"],
        "exit_reasons": m.get("exit_reasons"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup")
    ap.add_argument("--universe")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--cost-stress", type=float, default=1.0,
                    help="multiply commission, slippage and financing by this factor")
    ap.add_argument("--jitter", type=int, default=0,
                    help="also re-run on N copies with prices nudged by 0.001%% (robustness)")
    ap.add_argument("--out", default="apps/api/artifacts/swing_results.json")
    a = ap.parse_args()

    grid = [(s, u) for s, u in GRID
            if (not a.setup or s == a.setup) and (not a.universe or u == a.universe)]
    rows = []
    for setup, uni in grid:
        t0 = time.time()
        aclass = swing.UNIVERSES[uni]["asset_class"]
        base = swing.config_for(aclass)
        cfg = {"commission_bps": base.commission_bps * a.cost_stress,
               "slippage_bps": base.slippage_bps * a.cost_stress,
               "financing_bps_per_year": base.financing_bps_per_year * a.cost_stress}
        r = swing.backtest(setup=setup, universe=uni, start=a.start, config=cfg)
        row = _row(setup, uni, r)
        if a.jitter:
            from app.backtest import swing_engine as eng
            ctx = swing.prepare(setup=setup, universe=uni, start=a.start, config=cfg)
            jit = []
            for seed in range(1, a.jitter + 1):
                j = swing.jitter_context(ctx, seed)
                res = eng.simulate(j["data"], j["signals"], j["rules"], j["cfg"], earnings=j["earnings"],
                                   long_regime=j["regime"], benchmark=j["benchmark"],
                                   benchmark_symbol=j["benchmark_symbol"], mc_sims=500)
                jit.append({"cagr_pct": res.metrics.get("cagr_pct"), "sharpe": res.metrics.get("sharpe"),
                            "n_trades": res.metrics.get("n_trades"), "verdict": res.verdict["label"]})
            row["jitter"] = jit
            cg = sorted(x["cagr_pct"] for x in jit)
            sh = sorted(x["sharpe"] for x in jit)
            row["jitter_cagr_range"] = [cg[0], cg[len(cg) // 2], cg[-1]]      # min, median, max
            row["jitter_sharpe_range"] = [sh[0], sh[len(sh) // 2], sh[-1]]
            row["jitter_verdicts"] = {v: sum(1 for x in jit if x["verdict"] == v) for v in {x["verdict"] for x in jit}}
        rows.append(row)
        print(f"[{time.time() - t0:5.1f}s] {setup:9s} {uni:14s} "
              f"CAGR {row['cagr_pct']:6.2f}%  Sharpe {row['sharpe']:5.2f} (t {row['sharpe_t']:5.2f})  "
              f"MDD {row['max_dd_pct']:7.2f}%  n={row['trades']:5d}  win {row['win_pct']:5.1f}%  "
              f"E[R] {row['exp_r']:+.3f}  PF {row['pf']}  halves {row['h1_cagr']}/{row['h2_cagr']}%  "
              f"MC p(≤0) {row['mc_p_loss']}%  bench {row['bench_cagr']}%/{row['bench_sharpe']}  "
              f"→ {row['verdict']}"
              + (f"  | jitter CAGR {row['jitter_cagr_range']} Sharpe {row['jitter_sharpe_range']} "
                 f"{row['jitter_verdicts']}" if a.jitter else ""),
              flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"cost_stress": a.cost_stress, "start": a.start, "rows": rows}, f, indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
