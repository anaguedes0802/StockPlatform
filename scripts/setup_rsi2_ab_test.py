"""Create the RSI(2) ML-filter A/B paper test: two identical bots, filter on/off.

Usage (repo root):
    apps/api/.venv/bin/python scripts/setup_rsi2_ab_test.py --email you@example.com
    apps/api/.venv/bin/python scripts/setup_rsi2_ab_test.py --email you@example.com --dry-run

Idempotent: bots already tagged with the experiment for that user are left
alone. Both bots:
  * trade the 50-name US large-cap universe the filter was evaluated on,
  * use the RSI(2) bot's tuned rules, deciding on completed daily bars
    (signal at the close, act at the next morning's run),
  * have the LLM gate OFF (so the only difference is the ML filter),
  * run on their own simulated $100k paper ledger (no shared account),
  * size 1/8 of the starting equity per position, max 8 positions,
  * opt in to the daily auto-run (needs BOT_AUTORUN=1 on the API server).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("WARMUP_DISABLED", "1")
_API = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "api")
sys.path.insert(0, os.path.abspath(_API))
os.chdir(_API)  # the API's .env / sqlite path are relative to apps/api

from sqlalchemy import select  # noqa: E402

from app.db.models import TradingBot, User  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import swing  # noqa: E402
from app.services import trading_bot as botsvc  # noqa: E402

EXPERIMENT = "rsi2_ml_filter_ab"
NOTE = ("Forward paper test started to check the walk-forward result that the ML signal filter "
        "improves RSI(2) on US large caps (Sharpe 0.51 → 0.78 in 2010–2026 backtests). "
        "Only difference between the bots: ml_filter. Judge after 6–12 months, not before.")

EXECUTION = {
    **botsvc.DEFAULT_EXECUTION,
    "broker": "sim",
    "sim_initial_cash": 100_000.0,
    "max_position_usd": 12_500.0,        # 1/8 of starting equity, as in the backtest
    "max_open_positions": 8,
    "max_gross_exposure_usd": 100_000.0,
    "max_daily_orders": 20,
    "max_daily_loss_usd": 5_000.0,       # 5% — a guard, not a strategy rule
    "max_total_drawdown_pct": 0.15,
    "market_hours_only": True,
    "use_broker_bracket": True,          # +1% target / −25% stop rest at the (simulated) broker
    "autorun": True,
}


def dsl(arm: str) -> dict:
    return {**botsvc.DEFAULT_STRATEGY, "llm_gate": False, "completed_bars": True,
            "ml_filter": arm == "treatment", "experiment": EXPERIMENT,
            "experiment_arm": arm, "experiment_note": NOTE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="platform login that should own the bots")
    ap.add_argument("--dry-run", action="store_true",
                    help="run one pass per arm on throwaway ledgers; create nothing")
    a = ap.parse_args()
    universe = list(swing.UNIVERSES["us_large_caps"]["symbols"])

    if a.dry_run:
        for arm in ("control", "treatment"):
            res = botsvc.execute_pass(universe=universe, dsl=dsl(arm),
                                      execution={**EXECUTION, "market_hours_only": False},
                                      run_state=None, context={"source": "dry_run"})
            acts = [(x["action"], x.get("symbol"), x.get("reason", "")[:70]) for x in res["actions"]]
            led = res["run_state"]["sim_ledger"]
            print(f"{arm}: blocked={res['blocked']} actions={len(acts)} cash={led['cash']:.0f} "
                  f"positions={list(led['positions'])}")
            for act in acts:
                print("   ", act)
        return

    with SessionLocal() as db:
        user = db.scalars(select(User).where(User.email == a.email)).first()
        if not user:
            sys.exit(f"no user with email {a.email}")
        existing = [b for b in db.scalars(select(TradingBot).where(TradingBot.user_id == user.id)).all()
                    if (b.dsl or {}).get("experiment") == EXPERIMENT]
        have = {(b.dsl or {}).get("experiment_arm") for b in existing}
        for arm, name in (("control", "RSI(2) A/B · control (no ML filter)"),
                          ("treatment", "RSI(2) A/B · treatment (ML filter on)")):
            if arm in have:
                print(f"{arm}: already exists, left unchanged")
                continue
            # Start both arms together at the next scheduled morning run
            # rather than whenever the server next happens to restart.
            today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
            b = TradingBot(user_id=user.id, name=name, bot_type="trader", dsl=dsl(arm),
                           universe=universe, status="active", mode="paper", armed=True,
                           execution=dict(EXECUTION), run_state={"last_autorun_day": today_et})
            db.add(b)
            db.flush()
            print(f"{arm}: created {b.id}")
        db.commit()


if __name__ == "__main__":
    main()
