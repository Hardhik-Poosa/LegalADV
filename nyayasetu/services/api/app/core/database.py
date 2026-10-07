"""Async SQLAlchemy engine + session factory.

Import ``async_session_factory`` to get a session inside a ``async with`` block,
or depend on ``get_db`` in FastAPI route handlers.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    s = get_settings()
    logger.info("Creating async engine: pool_size=%d", s.db_pool_size)
    return create_async_engine(
        str(s.database_url),
        pool_size=s.db_pool_size,
        max_overflow=s.db_max_overflow,
        echo=s.db_echo,
        future=True,
        connect_args={"server_settings": {"jit": "off"}},  # safer for pgvector
    )


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, Any]:
    """FastAPI dependency — yields a session, auto-closes on exit."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
