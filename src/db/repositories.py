"""Database repositories for CRUD operations."""

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.db.models import (
    Chunk,
    DocType,
    Document,
    Page,
    RegistrySection,
)

logger = structlog.get_logger()


async def create_registry_section(
    session: AsyncSession,
    name: str,
    description: str,
    description_embedding: list[float] | None = None,
) -> RegistrySection:
    """Create a new registry section."""
    section = RegistrySection(
        name=name,
        description=description,
        description_embedding=description_embedding,
    )
    session.add(section)
    await session.flush()
    logger.info("created_registry_section", id=section.id, name=name)
    return section


async def create_document(
    session: AsyncSession,
    registry_section_id: int,
    title: str,
    doc_type: DocType,
    url: str | None = None,
) -> Document:
    """Create a new document."""
    doc = Document(
        registry_section_id=registry_section_id,
        title=title,
        doc_type=doc_type,
        url=url,
    )
    session.add(doc)
    await session.flush()
    logger.info("created_document", id=doc.id, title=title)
    return doc


async def create_page(
    session: AsyncSession,
    document_id: int,
    page_number: int,
    raw_text: str,
) -> Page:
    """Create a new page."""
    page = Page(
        document_id=document_id,
        page_number=page_number,
        raw_text=raw_text,
    )
    session.add(page)
    await session.flush()
    return page


async def create_chunks_batch(
    session: AsyncSession,
    chunks_data: list[dict],
) -> list[Chunk]:
    """Create chunks in batch.

    Args:
        chunks_data: List of dicts with keys:
            page_id, registry_section_id, content, embedding, chunk_index
    """
    chunks = [Chunk(**data) for data in chunks_data]
    session.add_all(chunks)
    await session.flush()
    logger.info("created_chunks_batch", count=len(chunks))
    return chunks


async def get_all_sections(session: AsyncSession) -> list[RegistrySection]:
    """Get all registry sections."""
    result = await session.execute(select(RegistrySection))
    return list(result.scalars().all())
