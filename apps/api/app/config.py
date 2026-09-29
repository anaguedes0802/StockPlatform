from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = "postgresql+psycopg://stockplatform:stockplatform@localhost:5432/stockplatform"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret: str = "dev-only-change-me"
    jwt_access_minutes: int = 15
    jwt_refresh_days: int = 30

    cors_origins: str = "http://localhost:3000"

    polygon_api_key: str | None = None
    finnhub_api_key: str | None = None
    alphavantage_api_key: str | None = None
    newsapi_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Alpaca Markets — real-time market data via WebSocket (free tier = IEX feed).
    # When set, /ws/quotes streams Alpaca ticks; otherwise it falls back to
    # yfinance polling on a 5-second cadence.
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_feed: str = "iex"  # "iex" (free) or "sip" (paid, full consolidated tape)

    # Alpaca trading (order execution) for the live bot. Defaults to the PAPER
    # endpoint — orders execute in real time against live prices but with fake
    # money, zero capital at risk. Point this at https://api.alpaca.markets only
    # to trade real money. Trading keys may differ from data keys (paper trading
    # uses its own key pair); if unset, the data keys above are reused.
    alpaca_trading_base_url: str = "https://paper-api.alpaca.markets"
    # Stock-selection forward test (app/services/paper_strategy.py): "sim" fills
    # internally at live prices with modeled costs; "alpaca" sends orders to the
    # Alpaca PAPER account above. Neither touches real money.
    paper_strategy_broker: str = "sim"
    alpaca_trading_api_key: str | None = None
    alpaca_trading_api_secret: str | None = None

    # LLM model used by /ai/opinion and /ai/chat. Override via LLM_MODEL env var.
    # Common choices:
    #   claude-haiku-4-5    — fastest, $0.80/M input, $4/M output    (~$0.03/opinion)
    #   claude-sonnet-4-6   — default, $3/M input, $15/M output      (~$0.10/opinion)
    #   claude-opus-4-7     — highest quality, $15/M input, $75/M out (~$0.50/opinion)
    llm_model: str = "claude-sonnet-4-6"

    model_dir: str = Field(default="./artifacts/models")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
