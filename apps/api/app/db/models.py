from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)


# Cross-dialect autoincrement primary key. sqlite needs an `INTEGER PRIMARY KEY`
# (the rowid alias) to autoincrement; BigInteger maps to BIGINT which won't.
# Postgres still gets a 64-bit type via the variant.
def AutoIntPK():
    return BigInteger().with_variant(Integer(), "sqlite")
from sqlalchemy import CHAR, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.dialects.postgresql import UUID as _PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


# Cross-dialect JSON: JSONB on Postgres (for GIN indexes), generic JSON otherwise.
JSONB = JSON().with_variant(_PG_JSONB(astext_type=Text()), "postgresql")


class _UUID(TypeDecorator):
    """Cross-dialect UUID — native UUID on Postgres, CHAR(36) string on sqlite.

    The Postgres-only `with_variant(CHAR(36), "sqlite")` swap only changes the
    column type; it doesn't add a bind processor, so sqlite's driver still sees
    `uuid.UUID` instances at bind time and dies with "type 'UUID' is not
    supported". This TypeDecorator coerces both ways explicitly.
    """
    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


def UUID(as_uuid: bool = True):
    """Backwards-compat shim — `as_uuid` is honored by always returning UUID objects."""
    return _UUID()


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    totp_secret: Mapped[str | None] = mapped_column(String(64))
    # 2FA is enforced only once a code from the authenticator has been verified;
    # until then totp_secret is a pending enrolment.
    totp_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false(), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="user")
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    watchlists: Mapped[list["Watchlist"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Instrument(Base):
    __tablename__ = "instruments"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(32))
    asset_class: Mapped[str] = mapped_column(String(16), default="stock")
    sector: Mapped[str | None] = mapped_column(String(64))
    industry: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(8))
    currency: Mapped[str | None] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    extra: Mapped[dict | None] = mapped_column(JSONB, default=dict)


class PriceBar(Base):
    """OHLCV. In production this is a TimescaleDB hypertable on (ts)."""

    __tablename__ = "price_bars"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    resolution: Mapped[str] = mapped_column(String(8), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    high: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    low: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    close: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    volume: Mapped[int] = mapped_column(BigInteger, default=0)


class Watchlist(Base, TimestampMixin):
    __tablename__ = "watchlists"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    user: Mapped[User] = relationship(back_populates="watchlists")
    items: Mapped[list["WatchlistItem"]] = relationship(
        back_populates="watchlist", cascade="all, delete-orphan"
    )


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    watchlist_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("watchlists.id", ondelete="CASCADE"), primary_key=True
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    watchlist: Mapped[Watchlist] = relationship(back_populates="items")


class Portfolio(Base, TimestampMixin):
    __tablename__ = "portfolios"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(8), default="USD")

    user: Mapped[User] = relationship(back_populates="portfolios")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", order_by="Transaction.occurred_at"
    )


class Transaction(Base, TimestampMixin):
    __tablename__ = "transactions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(8))  # buy | sell
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    note: Mapped[str | None] = mapped_column(Text)

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    model: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    point_estimate: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    lower_80: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    upper_80: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    lower_95: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    upper_95: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    confidence: Mapped[float | None] = mapped_column()
    contributions: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    drivers: Mapped[list | None] = mapped_column(JSONB, default=list)
    direction_prob_up: Mapped[float | None] = mapped_column()


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    label: Mapped[str] = mapped_column(String(16))  # strong_buy|buy|hold|sell|strong_sell
    score: Mapped[float] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    reasoning: Mapped[dict] = mapped_column(JSONB, default=dict)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str | None] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    entities: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    sentiment: Mapped[float | None] = mapped_column()
    sentiment_confidence: Mapped[float | None] = mapped_column()
    topics: Mapped[list | None] = mapped_column(JSONB, default=list)
    impact_score: Mapped[float | None] = mapped_column()


class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128))
    dsl: Mapped[dict] = mapped_column(JSONB, default=dict)


class Backtest(Base):
    __tablename__ = "backtests"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    equity_curve: Mapped[list] = mapped_column(JSON, default=list)
    trades: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TradingBot(Base, TimestampMixin):
    """A user's automated trading bot — a saved strategy config + a universe it
    watches. The bot is paper-only: it computes signals and a paper trade log,
    but never places real broker orders (the platform does not execute trades).

    `dsl` is the strategy config consumed by `app.services.trading_bot`.
    `universe` is the list of symbols the bot scans.
    `last_signals` caches the most recent scan so the UI can render instantly.
    """

    __tablename__ = "trading_bots"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), default="My Bot")
    # "trader" = the RSI-2 dip-buying strategy (buys dips, sells bounces).
    # "investor" = the long-horizon accumulator (holds a diversified target
    # allocation, dollar-cost-averages, never takes profit). The same row/CRUD/
    # arm/run plumbing serves both; the type selects which engine runs.
    bot_type: Mapped[str] = mapped_column(String(16), default="trader")  # trader | investor
    dsl: Mapped[dict] = mapped_column(JSONB, default=dict)
    universe: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(16), default="paused")  # active | paused
    last_signals: Mapped[list | None] = mapped_column(JSONB, default=list)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- Live execution (Alpaca) ---
    # `mode` is "paper" (default) or "live"; the actual broker endpoint is set
    # server-side (alpaca_trading_base_url) so a client can never force real
    # money. `armed` is the manual safety toggle: the bot can only place orders
    # while armed, and it never auto-arms. `execution` holds the per-bot
    # guardrails (position cap, daily kill-switch). `run_state` tracks the
    # rolling daily counters + a bounded order log.
    mode: Mapped[str] = mapped_column(String(8), default="paper")  # paper | live
    armed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    run_state: Mapped[dict | None] = mapped_column(JSONB, default=dict)


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))   # user | assistant | tool
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list | None] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(64))
    ua: Mapped[str | None] = mapped_column(Text)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    payload: Mapped[dict | None] = mapped_column(JSONB)
    __table_args__ = (UniqueConstraint("id"),)


class Notification(Base):
    """In-app notification — surfaced via the nav bell.

    `kind` examples:
      - "new_rising_star"     — a symbol surfaced in today's rising_stars that wasn't in yesterday's
      - "underpriced_value"   — symbol newly classified as deep-value
      - "portfolio_alert"     — portfolio-review surfaced a critical/warning
      - "watchlist_surge"     — watchlist symbol moved >X% in a session

    `payload` carries kind-specific structured detail for the UI to render.
    """
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(AutoIntPK(), primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")  # info | warning | critical
    symbol: Mapped[str | None] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # dedupe — same (user, kind, symbol) within a 24h window collapses to one row
    dedupe_key: Mapped[str | None] = mapped_column(String(255), index=True)


class ScreenerSnapshot(Base):
    """Per-strategy daily snapshot. Used to diff today vs yesterday so we can
    notify on *newly* surfacing picks. Anonymized — not per-user."""
    __tablename__ = "screener_snapshots"
    id: Mapped[int] = mapped_column(AutoIntPK(), primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    symbols: Mapped[list | None] = mapped_column(JSONB)          # ranked list of symbol strings
    full_results: Mapped[list | None] = mapped_column(JSONB)     # full hit details
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Thesis(Base, TimestampMixin):
    """A committed investment thesis — the core of the professional process loop.

    A pro never buys without writing down WHY, what would invalidate it, the
    target, the catalyst, and the time horizon. Then they review the position
    against the thesis rather than reacting to price. This table makes that
    discipline first-class: the platform tracks each holding against its
    thesis and flags when the invalidation level is breached or the horizon
    elapses without the thesis playing out.
    """
    __tablename__ = "theses"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8), default="long")   # long | short
    thesis: Mapped[str] = mapped_column(Text)                            # the written rationale
    catalyst: Mapped[str | None] = mapped_column(Text)                   # what should move it
    entry_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    invalidation_price: Mapped[float | None] = mapped_column(Numeric(20, 8))  # thesis is wrong below/above this
    target_1: Mapped[float | None] = mapped_column(Numeric(20, 8))
    target_2: Mapped[float | None] = mapped_column(Numeric(20, 8))
    horizon_days: Mapped[int | None] = mapped_column(BigInteger)         # expected time to play out
    conviction: Mapped[int] = mapped_column(BigInteger, default=3)       # 1-5 (sizing input)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    # active | invalidated | target_hit | closed | expired
    close_reason: Mapped[str | None] = mapped_column(Text)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_price: Mapped[float | None] = mapped_column(Numeric(20, 8))


class EntryTrigger(Base, TimestampMixin):
    """A user-defined watch condition. The background scanner evaluates these
    every cycle and fires a notification ONLY when the condition is met —
    turning the watchlist into a personalized, level-aware hunt.

    `condition` examples:
      - price_below / price_above   (threshold = price)
      - rsi_below / rsi_above       (threshold = RSI level, e.g. 40)
      - pct_change_day              (threshold = % move today, e.g. -5)
      - breakout_volume             (threshold = price; also needs volume spike)
    """
    __tablename__ = "entry_triggers"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    condition: Mapped[str] = mapped_column(String(32))
    threshold: Mapped[float] = mapped_column(Numeric(20, 8))
    note: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fire_count: Mapped[int] = mapped_column(BigInteger, default=0)


class OpportunityRecord(Base):
    """Track record of every opportunity the engine surfaces, so we can measure
    the platform's OWN batting average and down-weight strategies that don't
    work. Recorded at surface time; the forward returns are backfilled by a
    scoring job at 1/4/12-week marks.
    """
    __tablename__ = "opportunity_records"
    id: Mapped[int] = mapped_column(AutoIntPK(), primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str | None] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(Numeric(8, 4))
    entry_quality: Mapped[float | None] = mapped_column(Numeric(8, 4))
    price_at_surface: Mapped[float | None] = mapped_column(Numeric(20, 8))
    surfaced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    # Forward returns — filled in by the scoring backfill (None until matured)
    ret_1w: Mapped[float | None] = mapped_column(Numeric(10, 4))
    ret_4w: Mapped[float | None] = mapped_column(Numeric(10, 4))
    ret_12w: Mapped[float | None] = mapped_column(Numeric(10, 4))
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # dedupe — one record per (symbol, strategy, day)
    dedupe_key: Mapped[str | None] = mapped_column(String(255), index=True)


class AiGateDecision(Base):
    """Every verdict the LLM risk-gate gives on a quant BUY — the forward test.

    LLM trading judgement can't be backtested honestly (the model has read
    about the outcomes), so the only valid evidence is what happens *after*
    each decision. Rows are written when the gate rules; the `trade_*` and
    `ret_*` columns are backfilled by `gate_log.score_pending` once the
    hypothetical trade (taken under the strategy's own exit rules) has closed.
    """
    __tablename__ = "ai_gate_decisions"
    id: Mapped[int] = mapped_column(AutoIntPK(), primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    as_of: Mapped[str] = mapped_column(String(32))            # signal bar date (YYYY-MM-DD)
    kind: Mapped[str] = mapped_column(String(32))             # strategy kind (rsi2_meanrev, swing_breakout…)
    source: Mapped[str] = mapped_column(String(16))           # run | scan | signal | autorun
    bot_id: Mapped[str | None] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16), index=True)  # APPROVE | DOWNSIZE | VETO
    size_multiplier: Mapped[float | None] = mapped_column(Numeric(8, 4))
    conviction: Mapped[float | None] = mapped_column(Numeric(8, 4))
    provider: Mapped[str | None] = mapped_column(String(64))
    rationale: Mapped[str | None] = mapped_column(Text)
    key_risks: Mapped[list | None] = mapped_column(JSONB, default=list)
    signal_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    stop_dist: Mapped[float | None] = mapped_column(Numeric(20, 8))
    # --- outcome (backfilled) ---
    entry_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ret_5: Mapped[float | None] = mapped_column(Numeric(12, 6))
    ret_10: Mapped[float | None] = mapped_column(Numeric(12, 6))
    ret_20: Mapped[float | None] = mapped_column(Numeric(12, 6))
    trade_r: Mapped[float | None] = mapped_column(Numeric(12, 6))
    trade_return_pct: Mapped[float | None] = mapped_column(Numeric(12, 6))
    trade_exit_reason: Mapped[str | None] = mapped_column(String(32))
    trade_bars: Mapped[int | None] = mapped_column(Integer)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
