"""Async database engine and session management."""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent_rag.config import settings

logger = structlog.get_logger()


def create_engine() -> AsyncEngine:
    """Create an async SQLAlchemy engine with connection pooling."""
    logger.info(
        "creating_database_engine",
        url=settings.DATABASE_URL.split("@")[-1],  # Log without credentials
    )
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


engine = create_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session.

    Usage:
        async for session in get_session():
            result = await session.execute(...)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("database_session_error")
            raise
