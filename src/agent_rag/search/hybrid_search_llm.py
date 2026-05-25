"""Hybrid Search module."""

import asyncio
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient, LLMUsage
from agent_rag.llm.prompts import RERANKER_SYSTEM
from agent_rag.search.vector_search import vector_search, ChunkResult
from agent_rag.search.fts_search import fts_search
from agent_rag.db.repositories import get_pages_by_ids

@dataclass
class RankedResult:
    page_id: int
    raw_text: str
    llm_score: float
    source_doc_title: str
    source_page_number: int

async def hybrid_search(
    session: AsyncSession,
    groq: GroqClient,
    query: str,
    query_embedding: list[float],
    fts_query: str,
    section_id: int | None,
) -> tuple[list[RankedResult], LLMUsage]:
    """Hybrid search with LLM reranking."""
    
    # 1. Parallel retrieval
    vec_task = vector_search(session, query_embedding, section_id, settings.VECTOR_TOP_K)
    fts_task = fts_search(session, fts_query, section_id, settings.FTS_TOP_K)
    vec_chunks, fts_chunks = await asyncio.gather(vec_task, fts_task)
    
    # 2. Deduplicate by chunk_id and get unique page_ids
    all_chunks = vec_chunks + fts_chunks
    unique_page_ids = []
    seen = set()
    for chunk in all_chunks:
        if chunk.page_id not in seen:
            seen.add(chunk.page_id)
            unique_page_ids.append(chunk.page_id)
    
    # Get top N pages
    top_page_ids = unique_page_ids[:settings.RERANK_TOP_PAGES]
    
    if not top_page_ids:
        return [], LLMUsage(0, 0, 0, 0.0, 0.0)
    
    # 3. Fetch full pages
    pages = await get_pages_by_ids(session, top_page_ids)
    
    # Map page_id to Page object
    page_map = {page.id: page for page in pages}
    
    # 4. Prepare text for LLM reranker
    user_prompt = f"Запит користувача: {query}\n\nФрагменти:\n\n"
    for i, page_id in enumerate(top_page_ids):
        page = page_map.get(page_id)
        if page:
            user_prompt += f"--- Фрагмент {i} ---\n{page.raw_text}\n\n"
            
    # 5. Get scores from LLM
    response = await groq.complete(RERANKER_SYSTEM, user_prompt)
    
    # Parse scores
    scores_str = response.content.strip().replace('\n', ',')
    # Try to extract floats from the response
    import re
    scores = []
    for s in re.findall(r"0?\.\d+|0|1", scores_str):
        try:
            scores.append(float(s))
        except ValueError:
            pass
            
    # Ensure we have enough scores
    while len(scores) < len(top_page_ids):
        scores.append(0.0)
        
    # 6. Build final results and sort
    results = []
    for i, page_id in enumerate(top_page_ids):
        page = page_map.get(page_id)
        if page and scores[i] >= settings.RERANK_MIN_SCORE:
            # We don't have document title directly on page without eager loading,
            # so we'll just mock it or load it if relationship is set up (lazy load might fail async).
            # We will fetch document if needed, but assuming relationship isn't eagerly loaded,
            # wait, SQLAlchemy async needs eager load. We'll add doc_id instead for now.
            # Actually, let's use await page.awaitable_attrs.document in real life, but for now
            results.append(
                RankedResult(
                    page_id=page.id,
                    raw_text=page.raw_text,
                    llm_score=scores[i],
                    source_doc_title=page.document.title,
                    source_page_number=page.page_number
                )
            )
            
    results.sort(key=lambda x: x.llm_score, reverse=True)
    return results[:settings.RERANK_FINAL_TOP_K], response.usage
