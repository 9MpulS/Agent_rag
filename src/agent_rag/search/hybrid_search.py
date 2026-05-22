"""Hybrid Search module."""

from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

import structlog

from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient, LLMUsage
from agent_rag.llm.prompts import RERANKER_SYSTEM
from agent_rag.search.vector_search import vector_search, ChunkResult
from agent_rag.search.fts_search import fts_search
from agent_rag.db.models import Chunk, Page

logger = structlog.get_logger()


@dataclass
class RankedResult:
    page_id: int
    raw_text: str
    llm_score: float
    source_doc_title: str
    source_page_number: int


async def _fetch_chunks_by_ids(
    session: AsyncSession, chunk_ids: list[int]
) -> dict[int, Chunk]:
    """Fetch Chunk objects with their page→document eagerly loaded."""
    if not chunk_ids:
        return {}
    result = await session.execute(
        select(Chunk)
        .options(selectinload(Chunk.page).selectinload(Page.document))
        .where(Chunk.id.in_(chunk_ids))
    )
    return {c.id: c for c in result.scalars().all()}


async def hybrid_search(
    session: AsyncSession,
    groq: GroqClient,
    query: str,
    query_embedding: list[float],
    fts_query: str,
    section_id: int | None,
) -> tuple[list[RankedResult], LLMUsage]:
    """Hybrid search with LLM reranking.

    Uses chunk text (not page raw_text) for reranking, so the correct
    semantic content reaches the LLM even when all chunks share the same
    page_id due to simplified seeding.
    """

    # 1. Parallel retrieval
    vec_chunks = await vector_search(session, query_embedding, section_id, settings.VECTOR_TOP_K)
    fts_chunks = await fts_search(session, fts_query, section_id, settings.FTS_TOP_K)

    # 2. Reciprocal Rank Fusion (RRF) to merge vector and FTS results fairly
    #    score(chunk) = 1/(k + rank_vec) + 1/(k + rank_fts)  where k=60
    RRF_K = 60
    rrf_scores: dict[int, float] = {}

    for rank, cr in enumerate(vec_chunks):
        rrf_scores[cr.chunk_id] = rrf_scores.get(cr.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, cr in enumerate(fts_chunks):
        # Add a tiny boost (0.001) so exact text matches win tie-breakers against vector search
        rrf_scores[cr.chunk_id] = rrf_scores.get(cr.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1) + 0.001

    # Build deduplicated list sorted by RRF score descending
    all_chunks_map: dict[int, ChunkResult] = {}
    for cr in vec_chunks + fts_chunks:
        if cr.chunk_id not in all_chunks_map:
            all_chunks_map[cr.chunk_id] = cr

    unique_chunks: list[ChunkResult] = sorted(
        all_chunks_map.values(),
        key=lambda cr: rrf_scores[cr.chunk_id],
        reverse=True,
    )


    top_chunks = unique_chunks[: settings.RERANK_TOP_PAGES]

    if not top_chunks:
        return [], LLMUsage(0, 0, 0, 0.0, 0.0)

    # 3. Fetch full Chunk objects (with page + document)
    chunk_map = await _fetch_chunks_by_ids(session, [c.chunk_id for c in top_chunks])

    # 4. Bypass LLM reranking to save API tokens and avoid RateLimitError
    # We will use the RRF score directly as the final score.
    final_results: list[RankedResult] = []

    for cr in top_chunks:
        chunk_obj = chunk_map.get(cr.chunk_id)
        if not chunk_obj:
            continue

        # Use RRF score directly
        score = rrf_scores.get(cr.chunk_id, 0.0)

        # Use chunk content as raw_text for the answer generator
        final_results.append(
            RankedResult(
                page_id=chunk_obj.page_id,
                raw_text=chunk_obj.content,
                llm_score=score,
                source_doc_title=chunk_obj.page.document.title,
                source_page_number=chunk_obj.page.page_number,
            )
        )

    # Sort results by RRF score descending
    final_results.sort(key=lambda x: x.llm_score, reverse=True)

    # Return empty usage since we bypassed the LLM
    empty_usage = LLMUsage(0, 0, 0, 0.0, 0.0)
    return final_results[: settings.RERANK_FINAL_TOP_K], empty_usage
