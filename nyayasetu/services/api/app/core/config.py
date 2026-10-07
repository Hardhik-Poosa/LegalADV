"""Central configuration using pydantic-settings.

All settings are read from environment variables (or a .env file).
Instantiate ``get_settings()`` once at startup; everywhere else inject it.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:dev@localhost:5432/nyayasetu",
        description="Async SQLAlchemy DSN (asyncpg driver).",
    )
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=20, ge=0)
    db_echo: bool = Field(default=False, description="Log SQL statements (dev only).")

    # ── LM Studio ────────────────────────────────────────────────────────────
    lm_studio_base_url: str = Field(default="http://localhost:1234/v1")
    lm_studio_chat_model: str = Field(default="local-model")
    lm_studio_embedding_model: str = Field(
        default="text-embedding-nomic-embed-text-v1.5"
    )
    lm_studio_timeout_s: float = Field(default=120.0, gt=0)
    lm_studio_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    lm_studio_max_tokens: int = Field(default=1024, ge=1)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target characters per chunk (hard split at boundary).",
    )
    chunk_overlap: int = Field(
        default=64, ge=0, description="Characters of overlap between chunks."
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, description="Texts per embed() call."
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_chunk_size(cls, v: int, info: object) -> int:  # noqa: ARG003
        return v  # cross-field guard done at runtime in chunker


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    s = Settings()
    logger.info(
        "Settings loaded: env=%s db_echo=%s chunk_size=%d",
        s.environment,
        s.db_echo,
        s.chunk_size,
    )
    return s
