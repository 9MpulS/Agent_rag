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


async def get_all_section_embeddings(
    session: AsyncSession,
) -> list[tuple[int, list[float]]]:
    """Get all section IDs and their description embeddings."""
    result = await session.execute(
        select(RegistrySection.id, RegistrySection.description_embedding)
        .where(RegistrySection.description_embedding.is_not(None))
    )
    return [(row.id, row.description_embedding) for row in result.all()]


async def get_pages_by_ids(
    session: AsyncSession,
    page_ids: list[int],
) -> list[Page]:
    """Get pages by their IDs."""
    if not page_ids:
        return []
    result = await session.execute(
        select(Page).where(Page.id.in_(page_ids))
    )
    return list(result.scalars().all())


from sqlalchemy import func, text

async def vector_search_chunks(
    session: AsyncSession,
    vec: list[float],
    section_id: int | None,
    top_k: int,
) -> list[Chunk]:
    """Search chunks using pgvector HNSW cosine distance."""
    # We select Chunk.id, page_id, registry_section_id, content and distance.
    # Actually it's easier to just return the Chunk objects and we can attach distance
    # or just return Chunk since Chunk model has what we need.
    
    stmt = select(Chunk).order_by(Chunk.embedding.cosine_distance(vec)).limit(top_k)
    if section_id is not None:
        stmt = stmt.where(Chunk.registry_section_id == section_id)
        
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def fts_search_chunks(
    session: AsyncSession,
    tsquery_str: str,
    section_id: int | None,
    top_k: int,
) -> list[Chunk]:
    """Search chunks using Full-Text Search (GIN)."""
    # Use func.to_tsquery('simple', tsquery_str)
    tsq = func.to_tsquery('simple', tsquery_str)
    
    stmt = select(Chunk).where(
        Chunk.content_tsv.bool_op('@@')(tsq)
    ).order_by(
        func.ts_rank(Chunk.content_tsv, tsq).desc()
    ).limit(top_k)
    
    if section_id is not None:
        stmt = stmt.where(Chunk.registry_section_id == section_id)
        
    result = await session.execute(stmt)
    return list(result.scalars().all())

