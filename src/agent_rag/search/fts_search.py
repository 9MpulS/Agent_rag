"""Full-Text Search module."""

from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient, LLMUsage
from agent_rag.llm.prompts import FTS_QUERY_SYSTEM
from agent_rag.db.repositories import fts_search_chunks
from agent_rag.search.vector_search import ChunkResult

from agent_rag.search.ukrainian_stemmer import stem_ukrainian_text

async def fts_query_enhance(groq: GroqClient, query: str) -> tuple[str, LLMUsage]:
    """Enhance user query for Full-Text Search."""
    # Complete text using Groq
    response = await groq.complete(FTS_QUERY_SYSTEM, query)
    fts_query = stem_ukrainian_text(response.content.strip())
    return fts_query, response.usage

async def fts_search(
    session: AsyncSession,
    fts_query: str,
    section_id: int | None,
    top_k: int = settings.FTS_TOP_K,
) -> list[ChunkResult]:
    """Search vector database."""
    chunks = await fts_search_chunks(session, fts_query, section_id, top_k)
    results = []
    
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
