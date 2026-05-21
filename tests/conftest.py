"""Pytest fixtures."""

import pytest
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient
from db.repositories import get_all_section_embeddings

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="session")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
        
    await engine.dispose()

@pytest.fixture(scope="session")
def groq_client() -> GroqClient:
    return GroqClient()

@pytest.fixture(scope="session")
async def section_embeddings(db_session: AsyncSession) -> list[tuple[int, list[float]]]:
    return await get_all_section_embeddings(db_session)
