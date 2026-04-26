"""
Application configuration loaded from environment variables via .env file.
All values are configurable — nothing is hardcoded in business logic.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 2048

    # ── Data ──────────────────────────────────────────────────────────────────
    # Relative to the financial_advisor_agent/ folder.
    # "../" means the parent dir that contains all 6 JSON mock files.
    data_dir: str = "../"

    # ── Risk Thresholds ───────────────────────────────────────────────────────
    # Configurable — never hardcoded inside PortfolioAnalyzer or nodes.
    sector_concentration_warning_pct: float = 30.0
    sector_concentration_critical_pct: float = 60.0
    single_stock_warning_pct: float = 15.0
    high_beta_threshold: float = 1.2
    negative_sentiment_threshold: float = -0.5

    # ── LangGraph ─────────────────────────────────────────────────────────────
    max_chat_history: int = 20
    graph_recursion_limit: int = 25

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


# Singleton — import this everywhere
settings = Settings()
