"""Agentic RAG tools: search and refine_query.

Provides:
- TOOL_DEFINITIONS: JSON schemas for Groq native tool calling API
- execute_search(): full search pipeline (embed → route → FTS enhance → hybrid)
- execute_refine_query(): LLM-based query refinement
"""

import json
import time
import math
import structlog

from agent_rag.config import settings
from agent_rag.embeddings import get_embedding
from agent_rag.search.fts_search import fts_query_enhance, fts_search
from agent_rag.search.hybrid_search import hybrid_search, RankedResult
from agent_rag.search.hybrid_search_llm import hybrid_search as hybrid_search_llm
from agent_rag.search.vector_search import vector_search
from agent_rag.db.repositories import get_all_section_embeddings
from agent_rag.llm.groq_client import GroqClient, LLMUsage
from agent_rag.llm.prompts import REFINE_QUERY_SYSTEM

from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

# ── Native Tool Definitions (OpenAI function calling format) ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Пошук у базі знань університету СумДУ. "
                "Використовуй цей інструмент, щоб знайти інформацію з "
                "документів, положень та наказів університету."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Пошуковий запит українською мовою",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_query",
            "description": (
                "Уточнення пошукового запиту. Використовуй, якщо попередній "
                "пошук дав нерелевантні або недостатні результати. "
                "Генерує покращений запит з синонімами та переформулюванням."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original_query": {
                        "type": "string",
                        "description": "Початковий пошуковий запит",
                    },
                    "problem": {
                        "type": "string",
                        "description": "Опис проблеми з результатами попереднього пошуку",
                    },
                },
                "required": ["original_query", "problem"],
            },
        },
    },
]


# ── Helper ──

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def _route_to_section(
    session: AsyncSession,
    query_embedding: list[float],
) -> int | None:
    """Determine the best matching section for the query."""
    section_embeddings = await get_all_section_embeddings(session)

    best_section_id = None
    best_score = -1.0

    for sid, semb in section_embeddings:
        score = cosine_similarity(query_embedding, semb)
        if score > best_score:
            best_score = score
            best_section_id = sid

    if best_score < settings.ROUTING_MIN_SIMILARITY:
        best_section_id = None

    return best_section_id


# ── Tool Implementations ──

async def execute_search(
    session: AsyncSession,
    groq: GroqClient,
    query: str,
) -> tuple[list[RankedResult], dict[str, float], dict[str, LLMUsage]]:
    """Execute the full search pipeline: embed → route → FTS enhance → hybrid search.

    Returns:
        Tuple of (results, timings, token_usage)
    """
    timings: dict[str, float] = {}
    token_usage: dict[str, LLMUsage] = {}

    # 1. Embed query
    t0 = time.perf_counter()
    query_embedding = await get_embedding(query)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # 2. Route to section
    t0 = time.perf_counter()
    section_id = await _route_to_section(session, query_embedding)
    timings["route_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # 3. Enhance FTS query
    t0 = time.perf_counter()
    fts_q, fts_usage = await fts_query_enhance(groq, query)
    timings["fts_enhance_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    token_usage["fts_enhance"] = fts_usage

    # 4. Hybrid search (original without LLM reranker)
    # t0 = time.perf_counter()
    # results, rerank_usage = await hybrid_search(
    #     session=session,
    #     groq=groq,
    #     query=query,
    #     query_embedding=query_embedding,
    #     fts_query=fts_q,
    #     section_id=section_id,
    # )
    # timings["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # token_usage["rerank"] = rerank_usage

    # 4. Hybrid search with LLM reranker
    t0 = time.perf_counter()
    results, rerank_usage = await hybrid_search_llm(
        session=session,
        groq=groq,
        query=query,
        query_embedding=query_embedding,
        fts_query=fts_q,
        section_id=section_id,
    )
    timings["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    token_usage["rerank"] = rerank_usage

    logger.info(
        "search_tool_executed",
        query=query[:80],
        section_id=section_id,
        results_count=len(results),
    )

    return results, timings, token_usage


async def execute_refine_query(
    groq: GroqClient,
    original_query: str,
    problem: str,
) -> tuple[dict, LLMUsage]:
    """Refine a search query based on problem description.

    Returns:
        Tuple of (parsed JSON with refined_query and reasoning, usage)
    """
    user_prompt = (
        f"Оригінальний запит: {original_query}\n"
        f"Проблема: {problem}"
    )
    parsed, usage = await groq.complete_json(REFINE_QUERY_SYSTEM, user_prompt)

    refined_query = parsed.get("refined_query", original_query)
    reasoning = parsed.get("reasoning", "")

    logger.info(
        "refine_query_tool_executed",
        original=original_query[:80],
        refined=refined_query[:80],
        reasoning=reasoning[:100],
    )

    return {"refined_query": refined_query, "reasoning": reasoning}, usage
