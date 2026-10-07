# Alembic environment configuration
# Generated for NyayaSetu — async SQLAlchemy 2 + pgvector

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Alembic can detect schema changes.
from app.core.config import get_settings
from app.models.legal import Base  # noqa: F401 — registers all tables

# --------------------------------------------------------------------------- #
# Alembic config object (gives access to alembic.ini values)
# --------------------------------------------------------------------------- #
config = context.config

# Apply Python logging config from alembic.ini if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Provide the ORM metadata for autogenerate.
target_metadata = Base.metadata

# Override sqlalchemy.url from application settings so we never hard-code creds.
settings = get_settings()
config.set_main_option("sqlalchemy.url", str(settings.database_url))


# --------------------------------------------------------------------------- #
# Offline migrations (generate SQL without a live DB connection)
# --------------------------------------------------------------------------- #
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# --------------------------------------------------------------------------- #
# Online migrations (async engine)
# --------------------------------------------------------------------------- #
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    logger.info("Running migrations offline.")
    run_migrations_offline()
else:
    logger.info("Running migrations online.")
    run_migrations_online()
