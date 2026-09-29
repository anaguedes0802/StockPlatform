"""Refresh the House STOCK Act archive (official House Clerk PTRs) and the
copy-trading scorecards shown on the Congress tracker.

Usage (from the repo root):
    python scripts/refresh_congress.py                 # current + previous year (daily cron)
    python scripts/refresh_congress.py --since 2020    # initial backfill
    python scripts/refresh_congress.py --scorecards-only

Cron suggestion (after US close, disclosures post during business hours):
    30 22 * * 1-5  cd /app && .venv/bin/python scripts/refresh_congress.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

sys.path.insert(0, "apps/api")

from app.services import congress_ptr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=None, help="first filing year to backfill")
    ap.add_argument("--scorecards-only", action="store_true")
    args = ap.parse_args()
    if not args.scorecards_only:
        years = list(range(args.since, date.today().year + 1)) if args.since else None
        print(json.dumps(congress_ptr.sync(years), indent=2), flush=True)
    print("scorecards:", congress_ptr.compute_scorecards(), flush=True)
    print(json.dumps(congress_ptr.status(), indent=2))


if __name__ == "__main__":
    main()
