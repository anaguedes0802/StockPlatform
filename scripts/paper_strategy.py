"""Run the stock-selection forward test on the Alpaca PAPER account.

Usage (from the repo root):
    python scripts/paper_strategy.py plan                 # dry run: show this month's target book + orders
    python scripts/paper_strategy.py rebalance --execute  # submit orders if a rebalance is due (market hours)
    python scripts/paper_strategy.py snapshot             # record today's value vs SPY
    python scripts/paper_strategy.py status
    python scripts/paper_strategy.py replay --start 2016-01-04   # historical replay of the same rules

Cron suggestion (weekdays, 30 min after the US open; it only trades when due,
i.e. every ~28 days, and snapshots daily):
    0 14 * * 1-5  cd /app && .venv/bin/python scripts/paper_strategy.py rebalance --execute
    5 21 * * 1-5  cd /app && .venv/bin/python scripts/paper_strategy.py snapshot

Refuses to run against a real-money Alpaca endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "apps/api")

from app.services import paper_strategy as ps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "rebalance", "snapshot", "status", "replay"])
    ap.add_argument("--start", default="2016-01-04", help="replay start date")
    ap.add_argument("--no-gate", action="store_true", help="replay without the SPY trend gate")
    ap.add_argument("--execute", action="store_true", help="actually submit paper orders")
    ap.add_argument("--force", action="store_true", help="rebalance even if not due")
    ap.add_argument("--capital", type=float, default=None, help="first run only: capital to allocate")
    args = ap.parse_args()
    if args.cmd == "plan":
        out = ps.plan(args.capital)
    elif args.cmd == "rebalance":
        out = ps.rebalance(execute=args.execute, force=args.force, initial_capital=args.capital)
        if args.execute and out.get("executed"):
            ps.snapshot()
    elif args.cmd == "snapshot":
        out = ps.snapshot()
    elif args.cmd == "replay":
        from app.services import paper_replay
        res = paper_replay.run(start=args.start, gated=not args.no_gate)
        out = {k: v for k, v in res.items() if k not in ("curve", "holdings_end")}
    else:
        out = ps.status()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
