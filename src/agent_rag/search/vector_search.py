"""Vector search module."""

from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.config import settings
from db.repositories import vector_search_chunks

@dataclass
class ChunkResult:
    chunk_id: int
    page_id: int
    section_id: int
    content: str
    score: float

async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    section_id: int | None,
    top_k: int = settings.VECTOR_TOP_K,
) -> list[ChunkResult]:
    """Search vector database."""
    chunks = await vector_search_chunks(session, query_embedding, section_id, top_k)
    results = []
    
    # We didn't fetch distance directly in the simple ORM query, so we'll leave score=0.0
    # or we can refactor vector_search_chunks to return distance if we need it. 
    # For now, we will return 0.0 or just trust the ordering.
    for chunk in chunks:
        results.append(
            ChunkResult(
                chunk_id=chunk.id,
                page_id=chunk.page_id,
                section_id=chunk.registry_section_id,
                content=chunk.content,
                score=0.0,
            )
        )
    return results
