"""Trading bot API — backtest, configure, scan, and live-execute a strategy.

Two layers:
  * Signals/backtest (`/bot/backtest`, `/bot/signal`, `/bot/scan`) — read-only,
    never touch a broker.
  * Live execution (`/bot/account`, `/bot/bots/{id}/arm`, `.../run`) — places
    real orders through Alpaca. Defaults to the PAPER endpoint (fake money,
    real-time fills); the endpoint is chosen server-side so a client can never
    force real money. A bot only trades while `armed`, and a daily kill-switch
    (max orders / max loss) plus a per-position cap bound the downside.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import TradingBot, User
from app.db.session import get_db
from app.deps import get_current_user
from app.services import broker_alpaca as broker
from app.services import broker_sim
from app.services import gate_log
from app.services import investing as invest
from app.services import llm
from app.services import notifications as notif
from app.services import swing as swing_mod
from app.services import swing_data
from app.services import trading_bot as botsvc

router = APIRouter(prefix="/bot", tags=["bot"])


# ---------- schemas ----------

class BacktestIn(BaseModel):
    symbol: str | None = None
    symbols: list[str] | None = None
    dsl: dict | None = None
    range_: str = Field(default="10y", alias="range")
    initial_cash: float = 10_000.0

    model_config = {"populate_by_name": True}


class BotIn(BaseModel):
    name: str = "My Bot"
    bot_type: str | None = None  # "trader" (default) | "investor"
    dsl: dict | None = None
    universe: list[str] | None = None


class InvestRunIn(BaseModel):
    """One investment pass. `contribution` is new money the user opts to add on
    top of whatever idle cash is already in the account (0 = just deploy cash)."""
    contribution: float = Field(default=0.0, ge=0)


class DcaBacktestIn(BaseModel):
    """Backtest dollar-cost-averaging into an allocation."""
    monthly_contribution: float = Field(default=1_000.0, gt=0)
    allocation: dict[str, float] | None = None


class PicksBacktestIn(BaseModel):
    """Point-in-time buy list → growth-to-today for an allocation."""
    start_date: str
    amount: float = Field(default=10_000.0, gt=0)
    allocation: dict[str, float] | None = None


class ExecutionIn(BaseModel):
    """Per-bot live-execution guardrails."""
    max_position_usd: float | None = Field(default=None, gt=0)
    max_daily_orders: int | None = Field(default=None, ge=1)
    max_daily_loss_usd: float | None = Field(default=None, gt=0)
    market_hours_only: bool | None = None
    max_open_positions: int | None = Field(default=None, ge=1, le=50)
    autorun: bool | None = None
    max_gross_exposure_usd: float | None = Field(default=None, gt=0)
    risk_per_trade_pct: float | None = Field(default=None, gt=0, le=5)


# ---------- defaults / config ----------

@router.get("/config", response_model=dict)
def get_config() -> dict:
    """The bot's default tuned strategy and recommended universe."""
    return {
        "default_strategy": botsvc.DEFAULT_STRATEGY,
        "default_universe": botsvc.DEFAULT_UNIVERSE,
        "default_execution": botsvc.DEFAULT_EXECUTION,
        "swing": {
            "default_strategy": botsvc.DEFAULT_SWING_STRATEGY,
            "default_universe": botsvc.SWING_DEFAULT_UNIVERSE,
            "kinds": list(swing_mod.BOT_KINDS),
        },
        "broker": {
            "configured": broker.is_configured(),
            "paper": broker.is_paper(),
        },
        "llm": {
            "available": llm.is_available(),
            "providers": llm.available_providers(),
        },
        "description": (
            "Trend-filtered RSI(2) mean-reversion. Buys brief oversold dips in "
            "names trading above their 200-day SMA; takes the bounce via a small "
            "profit target with an SMA backstop. Tuned for high win rate."
        ),
    }


# ---------- backtest ----------

@router.post("/backtest", response_model=dict)
def backtest(payload: BacktestIn) -> dict:
    """Backtest the bot strategy on a single symbol or across a basket.

    Pass `symbol` for a per-symbol report (with trades + equity curve), or
    `symbols` (or neither, to use the default universe) for a pooled report.
    """
    try:
        if payload.symbol:
            rep = botsvc.backtest_symbol(
                payload.symbol, payload.dsl, range_=payload.range_, initial_cash=payload.initial_cash
            )
            return {"mode": "single", **rep.as_dict()}
        return {
            "mode": "portfolio",
            **botsvc.backtest_portfolio(
                payload.symbols, payload.dsl, range_=payload.range_, initial_cash=payload.initial_cash
            ),
        }
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/signal/{symbol}", response_model=dict)
def signal(symbol: str, llm: bool = True) -> dict:
    """The action the bot would take today for `symbol` (BUY / FLAT).

    With `llm=true` (default) a quant BUY is pressure-tested by the LLM risk-gate
    (news + smart-money review), which may veto or downsize it. Pass `llm=false`
    for the raw systematic signal only.
    """
    return botsvc.latest_signal(symbol, llm_gate=llm, context={"source": "signal"})


# ---------- saved bots (CRUD + paper scan) ----------

def _is_sim(b: TradingBot) -> bool:
    return (b.execution or {}).get("broker") == "sim"


def _serialize(b: TradingBot) -> dict:
    rs = dict(b.run_state or {})
    ledger = rs.pop("sim_ledger", None)
    return {
        "id": str(b.id),
        "name": b.name,
        "bot_type": getattr(b, "bot_type", "trader") or "trader",
        "dsl": b.dsl,
        "universe": b.universe,
        "status": b.status,
        "mode": b.mode,
        "armed": bool(b.armed),
        "execution": {**botsvc.DEFAULT_EXECUTION, **(b.execution or {})},
        "run_state": rs,
        "last_signals": b.last_signals or [],
        "last_run_at": b.last_run_at.isoformat() if b.last_run_at else None,
        "sim": broker_sim.summary(ledger) if _is_sim(b) else None,
    }


@router.post("/bots", status_code=201, response_model=dict)
def create_bot(payload: BotIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    bot_type = (payload.bot_type or "trader").lower()
    if bot_type not in ("trader", "investor"):
        raise HTTPException(400, "bot_type must be 'trader' or 'investor'")
    if bot_type == "investor":
        dsl = {**invest.DEFAULT_INVEST_CONFIG, **(payload.dsl or {})}
        if dsl.get("pick_mode") == "quality":
            # Holdings are chosen dynamically each run — no fixed universe.
            universe = payload.universe or []
        else:
            # An investor bot's "universe" is the symbols in its target allocation.
            universe = payload.universe or list(invest.normalize_weights(dsl["allocation"]).keys())
    else:
        dsl = botsvc.merge_dsl(payload.dsl)
        if botsvc.is_swing(dsl):
            universe = [s.upper() for s in (payload.universe or botsvc.SWING_DEFAULT_UNIVERSE)]
            _check_tradable(universe)
        else:
            universe = payload.universe or list(botsvc.DEFAULT_UNIVERSE)
    b = TradingBot(
        user_id=user.id,
        name=payload.name,
        bot_type=bot_type,
        dsl=dsl,
        universe=universe,
        status="paused",
        mode="paper",
        armed=False,
        execution=dict(botsvc.DEFAULT_EXECUTION),
        run_state={},
    )
    db.add(b)
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.get("/bots", response_model=list[dict])
def list_bots(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    rows = db.scalars(select(TradingBot).where(TradingBot.user_id == user.id)).all()
    return [_serialize(b) for b in rows]


def _check_tradable(universe: list[str]) -> None:
    """Swing bots execute through Alpaca: US equities/ETFs only."""
    bad = [s for s in universe if swing_data.asset_class(s) != "equity"]
    if bad:
        raise HTTPException(400, f"not tradable through Alpaca (FX/crypto are research-only): {bad}")


def _get_owned(bid: uuid.UUID, db: Session, user: User) -> TradingBot:
    b = db.get(TradingBot, bid)
    if not b or b.user_id != user.id:
        raise HTTPException(404, "bot not found")
    return b


@router.patch("/bots/{bid}", response_model=dict)
def update_bot(
    bid: uuid.UUID, payload: BotIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    b = _get_owned(bid, db, user)
    if payload.name:
        b.name = payload.name
    if payload.dsl is not None:
        if (b.bot_type or "trader") == "investor":
            b.dsl = {**invest.DEFAULT_INVEST_CONFIG, **payload.dsl}
        else:
            b.dsl = botsvc.merge_dsl(payload.dsl)
    if payload.universe is not None:
        if botsvc.is_swing(b.dsl):
            _check_tradable(payload.universe)
        b.universe = payload.universe
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.post("/bots/{bid}/status/{status}", response_model=dict)
def set_status(
    bid: uuid.UUID, status: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    if status not in ("active", "paused"):
        raise HTTPException(400, "status must be 'active' or 'paused'")
    b = _get_owned(bid, db, user)
    b.status = status
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.delete("/bots/{bid}", status_code=204)
def delete_bot(bid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    b = _get_owned(bid, db, user)
    db.delete(b)
    db.commit()


@router.post("/bots/{bid}/scan", response_model=dict)
def scan_bot(bid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    """Run the bot once: evaluate today's signal for every symbol in its universe.

    Paper-only — records the actions but places no orders. Persists the result
    so the UI can show "last run" without recomputing.
    """
    b = _get_owned(bid, db, user)
    if (b.bot_type or "trader") == "investor":
        # An investor "scan" is the timing read, not per-symbol entry signals.
        timing = invest.invest_timing()
        b.last_signals = [timing]
        b.last_run_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(b)
        return {**_serialize(b), "timing": timing}
    signals = []
    for sym in (b.universe or []):
        try:
            signals.append(botsvc.latest_signal(sym, b.dsl, universe=b.universe,
                                                context={"source": "scan", "bot_id": str(b.id)}))
        except Exception as e:  # noqa: BLE001
            signals.append({"symbol": sym.upper(), "action": "ERROR", "reason": str(e)})
    b.last_signals = signals
    b.last_run_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(b)
    buys = [s["symbol"] for s in signals if s.get("action") == "BUY"]
    return {**_serialize(b), "buys": buys}


# ---------- live execution (Alpaca) ----------

@router.get("/account", response_model=dict)
def account(user: User = Depends(get_current_user)) -> dict:
    """Broker account snapshot + open positions + market clock.

    Returns `configured: false` (no 500) when no Alpaca trading credentials are
    set, so the UI can prompt the user to add keys rather than erroring.
    """
    if not broker.is_configured():
        return {"configured": False, "paper": broker.is_paper(),
                "account": None, "positions": [], "market_open": None}
    try:
        return {
            "configured": True,
            "paper": broker.is_paper(),
            "account": broker.get_account(),
            "positions": broker.list_positions(),
            "market_open": broker.is_market_open(),
        }
    except broker.BrokerError as e:
        raise HTTPException(502, f"broker error: {e}") from e


@router.patch("/bots/{bid}/execution", response_model=dict)
def update_execution(
    bid: uuid.UUID, payload: ExecutionIn, db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Update a bot's live-execution guardrails (only the fields provided)."""
    b = _get_owned(bid, db, user)
    current = {**botsvc.DEFAULT_EXECUTION, **(b.execution or {})}
    patch = {k: v for k, v in payload.model_dump().items() if v is not None}
    b.execution = {**current, **patch}
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.post("/bots/{bid}/arm", response_model=dict)
def arm_bot(bid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    """Arm the bot so it may place orders. Refuses if the broker isn't set up."""
    b = _get_owned(bid, db, user)
    if not _is_sim(b) and not broker.is_configured():
        raise HTTPException(400, "Alpaca trading credentials are not configured")
    b.armed = True
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.post("/bots/{bid}/disarm", response_model=dict)
def disarm_bot(bid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    """Disarm the bot — it can no longer place any orders until re-armed."""
    b = _get_owned(bid, db, user)
    b.armed = False
    db.commit()
    db.refresh(b)
    return _serialize(b)


@router.post("/bots/{bid}/run", response_model=dict)
def run_bot(bid: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    """Execute one live trading pass: close exits and open new entries.

    Hard requirements: the bot must be `armed` and the broker configured. The
    daily kill-switch and per-position cap are enforced inside `run_live`.
    """
    b = _get_owned(bid, db, user)
    if (b.bot_type or "trader") == "investor":
        raise HTTPException(400, "investor bots run via /bot/bots/{id}/invest")
    if not b.armed:
        raise HTTPException(409, "bot is disarmed — arm it before running live")
    if not _is_sim(b) and not broker.is_configured():
        raise HTTPException(400, "Alpaca trading credentials are not configured")

    result = botsvc.execute_pass(
        universe=b.universe or [],
        dsl=b.dsl,
        execution=b.execution,
        run_state=b.run_state,
        alpaca=broker,
        context={"bot_id": str(b.id)},
    )
    b.run_state = result["run_state"]
    b.last_run_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(b)
    return {
        "bot": _serialize(b),
        "blocked": result.get("blocked"),
        "market_open": result.get("market_open"),
        "actions": result.get("actions", []),
        "account": result.get("account"),
        "mode": result.get("mode"),
        "resting_fills": result.get("resting_fills", []),
    }


# ---------- investment bot (long-horizon accumulator) ----------

@router.get("/invest/timing", response_model=dict)
def invest_timing(
    notify: bool = True, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    """Is now a good time to add long-term money? (NORMAL / ACCUMULATE / STRONG_BUY)

    When `notify` is set and the read is better than NORMAL, drop a (daily-deduped)
    in-app notification so the user is nudged to consider adding money — exactly
    the "tell me the right time to invest" behaviour. The user always chooses
    whether to actually contribute; this only informs.
    """
    timing = invest.invest_timing()
    if notify and timing.get("level") in ("ACCUMULATE", "STRONG_BUY"):
        sev = "warning" if timing["level"] == "STRONG_BUY" else "info"
        notif._create_if_new(
            db, user_id=user.id, kind="invest_opportunity", severity=sev,
            symbol="SPY", title=timing["headline"],
            body=timing.get("rationale") or timing.get("regime", {}).get("headline"),
            payload=timing,
        )
    return timing


@router.post("/invest/backtest", response_model=dict)
def invest_backtest(payload: DcaBacktestIn) -> dict:
    """Long-term DCA backtest: what monthly contributions into the allocation
    would have grown to (dividend-adjusted), vs a 100%-SPY benchmark. Read-only."""
    dsl = {"allocation": payload.allocation} if payload.allocation else None
    try:
        return invest.backtest_dca(dsl, monthly_contribution=payload.monthly_contribution)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/invest/quality-picks", response_model=dict)
def invest_quality_picks(top_n: int = 10) -> dict:
    """The stocks the quality picker would buy right now (top-N, equal-weight),
    ranked by the value+quality+growth+momentum factor model. The dynamic
    'what to buy' list."""
    return invest.quality_picks(top_n=top_n)


@router.post("/invest/picks-backtest", response_model=dict)
def invest_picks_backtest(payload: PicksBacktestIn) -> dict:
    """What the bot would have said to buy on a past date, and how each holding
    has grown since (vs all-in SPY). Read-only."""
    dsl = {"allocation": payload.allocation} if payload.allocation else None
    try:
        return invest.backtest_picks(payload.start_date, amount=payload.amount, dsl=dsl)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/bots/{bid}/rebalance-preview", response_model=dict)
def rebalance_preview(
    bid: uuid.UUID, payload: InvestRunIn, db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Preview the buy plan for an investor bot — no orders placed.

    Reads live positions + cash and computes the cash-flow rebalance toward the
    bot's target allocation (plus any contribution the user is considering)."""
    b = _get_owned(bid, db, user)
    if (b.bot_type or "trader") != "investor":
        raise HTTPException(400, "rebalance preview is only for investor bots")
    timing = invest.invest_timing()
    positions = broker.list_positions() if broker.is_configured() else []
    acct = broker.get_account() if broker.is_configured() else {}
    cash = float(acct.get("cash", 0.0)) if acct else 0.0
    plan = invest.rebalance_plan(positions, cash, b.dsl, contribution=payload.contribution, timing=timing)
    return {"timing": timing, "plan": plan, "broker_configured": broker.is_configured(),
            "cash": cash}


@router.post("/bots/{bid}/invest", response_model=dict)
def run_invest(
    bid: uuid.UUID, payload: InvestRunIn, db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Execute one investment pass: deploy idle cash (+ optional contribution).

    Buy-only — never sells. Requires the bot to be armed and the broker set up,
    same safety model as the trading bot."""
    b = _get_owned(bid, db, user)
    if (b.bot_type or "trader") != "investor":
        raise HTTPException(400, "this endpoint is only for investor bots")
    if not b.armed:
        raise HTTPException(409, "bot is disarmed — arm it before investing live")
    if not broker.is_configured():
        raise HTTPException(400, "Alpaca trading credentials are not configured")

    result = invest.run_invest(
        dsl=b.dsl, execution=b.execution, contribution=payload.contribution, broker_mod=broker,
    )
    b.last_run_at = datetime.now(timezone.utc)
    # Persist a bounded log of investment actions on run_state for the UI.
    state = dict(b.run_state or {})
    log_entries = (state.get("log") or []) + [a for a in result.get("actions", []) if a.get("action") == "BUY"]
    state["log"] = log_entries[-100:]
    state["last_plan"] = result.get("plan")
    state["last_timing"] = result.get("timing")
    b.run_state = state
    db.commit()
    db.refresh(b)
    return {
        "bot": _serialize(b),
        "blocked": result.get("blocked"),
        "market_open": result.get("market_open"),
        "timing": result.get("timing"),
        "plan": result.get("plan"),
        "actions": result.get("actions", []),
        "account": result.get("account"),
        "mode": "paper" if broker.is_paper() else "live",
    }


# ---------- AI risk-gate forward test ----------

@router.get("/gate/report", response_model=dict)
def gate_report(limit: int = 50) -> dict:
    """Forward-test scorecard of the LLM risk-gate.

    Scores any decisions whose hypothetical trade has since closed, then
    reports whether vetoed/downsized trades did worse than approved ones.
    """
    return gate_log.report(limit_recent=max(1, min(limit, 200)))


# ---------- A/B experiments between bots ----------

@router.get("/experiments", response_model=list[dict])
def experiments(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    """Bots grouped by `dsl.experiment`, with their simulated ledgers side by side."""
    rows = db.scalars(select(TradingBot).where(TradingBot.user_id == user.id)).all()
    groups: dict[str, list[TradingBot]] = {}
    for b in rows:
        tag = (b.dsl or {}).get("experiment")
        if tag:
            groups.setdefault(tag, []).append(b)
    out = []
    for tag, bots in groups.items():
        arms = []
        for b in sorted(bots, key=lambda x: str((x.dsl or {}).get("experiment_arm", ""))):
            ledger = (b.run_state or {}).get("sim_ledger")
            arms.append({"bot_id": str(b.id), "name": b.name, "arm": (b.dsl or {}).get("experiment_arm"),
                         "ml_filter": bool((b.dsl or {}).get("ml_filter")), "armed": bool(b.armed),
                         "status": b.status, "last_run_at": b.last_run_at.isoformat() if b.last_run_at else None,
                         "sim": broker_sim.summary(ledger) if ledger else None})
        started = min((b.created_at for b in bots if b.created_at), default=None)
        out.append({"experiment": tag, "started_at": started.isoformat() if started else None,
                    "description": (bots[0].dsl or {}).get("experiment_note"), "arms": arms})
    return out
