from __future__ import annotations

# Load .env into os.environ BEFORE any service imports. pydantic-settings
# reads .env into the Settings object but not into os.environ, so feature-flag
# vars like USE_TOPIC_CLASSIFIER / USE_FINBERT / USE_GDELT_HISTORICAL that are
# checked via `os.environ.get(...)` in service modules would never see them.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # python-dotenv is optional; fall back to whatever the shell exported.
    pass

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.audit import AuditMiddleware
from app.core.logging import configure_logging, log
from app.routers import ai, auth, bot, chat, discovery, news, notifications, opportunities, paper_strategy, portfolio, pro, screener, smart_money, stocks, strategy, swing, watchlists, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.env)
    log.info("startup", env=settings.env)
    # Zero-setup local mode: Alembic migrations target Postgres/TimescaleDB, so
    # on SQLite the schema is created straight from the ORM models.
    if settings.database_url.startswith("sqlite"):
        from app.db import models  # noqa: F401  (registers every table)
        from app.db.base import Base
        from app.db.session import engine
        Base.metadata.create_all(engine)
    # Background warmer: keeps screeners + opinion signals + notifications hot
    # so the UI doesn't have to wait on cold scans. Disable with
    # WARMUP_DISABLED=1 in .env (useful for tests).
    from app.services import warmup as warmup_svc
    warmup_svc.start_scheduler()
    # Opt-in (BOT_AUTORUN=1): run armed swing/trader bots once per trading day.
    from app.services import bot_autorun
    bot_autorun.start()
    try:
        yield
    finally:
        await bot_autorun.stop()
        await warmup_svc.stop_scheduler()
        log.info("shutdown")


app = FastAPI(
    title="StockPlatform API",
    version="0.1.0",
    summary="AI-driven stock analysis, forecasting, and portfolio platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditMiddleware)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/readyz")
def readyz() -> dict:
    # In prod: check DB + Redis + provider reachability here.
    return {"ok": True}


app.include_router(auth.router)
app.include_router(stocks.router)
app.include_router(ai.router)
app.include_router(portfolio.router)
app.include_router(watchlists.router)
app.include_router(strategy.router)
app.include_router(bot.router)
app.include_router(swing.router)
app.include_router(news.router)
app.include_router(ws.router)
app.include_router(chat.router)
app.include_router(screener.router)
app.include_router(smart_money.router)
app.include_router(notifications.router)
app.include_router(opportunities.router)
app.include_router(pro.router)
app.include_router(discovery.router)
app.include_router(paper_strategy.router)
