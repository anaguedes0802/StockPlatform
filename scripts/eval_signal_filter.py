"""Walk-forward evaluation of the ML swing signal filter (meta-labeling).

Usage (repo root):
    apps/api/.venv/bin/python scripts/eval_signal_filter.py                 # full grid
    apps/api/.venv/bin/python scripts/eval_signal_filter.py --setup breakout --universe etfs

The PRIMARY, pre-declared test is breakout × etfs (the only setup with an edge
worth filtering). Every other row is secondary evidence; with 8 configurations
one "helps" by luck would not be surprising.
Writes apps/api/artifacts/signal_filter_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

from app.ml import signal_filter as sf  # noqa: E402
from app.services import swing  # noqa: E402

GRID = [
    ("breakout", "etfs"), ("pullback", "etfs"), ("rsi2", "etfs"),
    ("breakout", "us_large_caps"), ("pullback", "us_large_caps"), ("rsi2", "us_large_caps"),
    ("breakout", "fx_majors"), ("pullback", "fx_majors"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup")
    ap.add_argument("--universe")
    ap.add_argument("--first-test-year", type=int, default=2010)
    ap.add_argument("--jitter", type=int, default=0,
                    help="also re-run on N copies with prices nudged by 0.001%% (robustness)")
    ap.add_argument("--out", default="apps/api/artifacts/signal_filter_results.json")
    a = ap.parse_args()
    rows = []
    for setup, uni in GRID:
        if (a.setup and setup != a.setup) or (a.universe and uni != a.universe):
            continue
        t0 = time.time()
        r = sf.evaluate(setup=setup, universe=uni, first_test_year=a.first_test_year)
        c, cl = r["comparison"], r["classification"]
        b, f = c["baseline"], c["filtered"]
        row = {"setup": setup, "universe": uni, "primary": (setup, uni) == ("breakout", "etfs"),
               "auc": cl["auc"], "logistic_auc": r["logistic_check"]["auc"], "rank_ic": cl["rank_ic"],
               "kept_vs_skipped": cl["kept_vs_skipped"], "baseline": b, "filtered": f,
               "oracle": c["oracle_filter"], "random": c["random_filter"],
               "keep_fraction": c["keep_fraction"], "sharpe_diff": c["sharpe_diff"],
               "verdict": r["verdict"]["label"], "n_labeled": r["n_labeled"]}
        if a.jitter:
            ctx = swing.prepare(setup=setup, universe=uni, start="2005-01-01")
            spy, vix = sf._market_series(ctx["load_from"])
            spy, vix = swing.completed_bars(spy, "SPY"), swing.completed_bars(vix, "^VIX")
            jit = []
            for seed in range(1, a.jitter + 1):
                j = sf.evaluate_on_context(swing.jitter_context(ctx, seed), spy=spy, vix=vix,
                                           first_test_year=a.first_test_year, mc_sims=0)
                jit.append({"seed": seed, "sharpe_diff": j["comparison"]["sharpe_diff"],
                            "baseline_sharpe": j["comparison"]["baseline"]["sharpe"],
                            "filtered_sharpe": j["comparison"]["filtered"]["sharpe"],
                            "verdict": j["verdict"]["label"]})
            row["jitter"] = jit
            row["jitter_verdicts"] = {v: sum(1 for x in jit if x["verdict"] == v) for v in {x["verdict"] for x in jit}}
        rows.append(row)
        print(f"[{time.time() - t0:5.1f}s] {'*' if row['primary'] else ' '} {setup:9s} {uni:14s} "
              f"AUC {cl['auc']['auc']} [{cl['auc']['lo']}, {cl['auc']['hi']}]  "
              f"Sharpe {b['sharpe']:.2f} → {f['sharpe']:.2f} (Δ {c['sharpe_diff']['diff']} "
              f"[{c['sharpe_diff']['lo']}, {c['sharpe_diff']['hi']}])  "
              f"CAGR {b['cagr_pct']} → {f['cagr_pct']}%  random {c['random_filter']['sharpe']:.2f}  "
              f"oracle {c['oracle_filter']['sharpe']:.2f}  → {row['verdict']}"
              + (f"  | jitter Δ {[x['sharpe_diff']['diff'] for x in row['jitter']]} {row['jitter_verdicts']}"
                 if a.jitter else ""), flush=True)
    with open(a.out, "w") as fh:
        json.dump({"first_test_year": a.first_test_year, "rows": rows}, fh, indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
