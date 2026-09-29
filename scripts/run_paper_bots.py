"""Run every due auto-run bot once, without the API server.

The API's background loop does this when BOT_AUTORUN=1 and the server is up.
If the server isn't always running, schedule this script instead, e.g. with
cron on weekdays shortly after 09:45 New York time:

    50 9 * * 1-5  cd /path/to/StockPlatform && apps/api/.venv/bin/python scripts/run_paper_bots.py

(cron uses the machine's local time zone: 09:50 ET is 14:50 in Lisbon.)
A bot runs at most once per New York trading day, so running this and the
server loop together is safe.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("WARMUP_DISABLED", "1")
_API = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "api")
sys.path.insert(0, os.path.abspath(_API))
os.chdir(_API)

from app.services import bot_autorun  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(bot_autorun.run_due(), indent=2, default=str))
    print(json.dumps(bot_autorun.score_gate_daily() or {}, default=str))
