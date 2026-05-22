"""Agent Nodes."""

import time
import math
from agent_rag.config import settings
from agent_rag.agent.state import AgentState
from agent_rag.embeddings import get_embedding
from agent_rag.search.fts_search import fts_query_enhance
from agent_rag.search.hybrid_search import hybrid_search
from agent_rag.llm.prompts import ANSWER_SYSTEM
from agent_rag.db.repositories import get_all_section_embeddings


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def embed_query(state: AgentState) -> dict:
    t0 = time.perf_counter()
    embedding = await get_embedding(state["query"])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "query_embedding": embedding,
        "timings": {**state.get("timings", {}), "embed_ms": round(elapsed_ms, 1)},
    }


async def route_to_section(state: AgentState) -> dict:
    t0 = time.perf_counter()
    section_embeddings = await get_all_section_embeddings(state["session"])
    
    best_section_id = None
    best_score = -1.0
    
    query_emb = state["query_embedding"]
    for sid, semb in section_embeddings:
        score = cosine_similarity(query_emb, semb)
        if score > best_score:
            best_score = score
            best_section_id = sid
            
    if best_score < settings.ROUTING_MIN_SIMILARITY:
        best_section_id = None
        
    elapsed_ms = (time.perf_counter() - t0) * 1000
    search_mode = "narrowed" if best_section_id else "global"
    
    return {
        "section_id": best_section_id,
        "search_mode": search_mode,
        "timings": {**state.get("timings", {}), "route_ms": round(elapsed_ms, 1)},
    }


async def enhance_fts_query(state: AgentState) -> dict:
    t0 = time.perf_counter()
    fts_q, usage = await fts_query_enhance(state["groq_client"], state["query"])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    token_usage = state.get("token_usage", {})
    token_usage["fts_enhance"] = usage
    
    return {
        "fts_query": fts_q,
        "token_usage": token_usage,
        "timings": {**state.get("timings", {}), "fts_enhance_ms": round(elapsed_ms, 1)},
    }


async def search(state: AgentState) -> dict:
    t0 = time.perf_counter()
    results, usage = await hybrid_search(
        session=state["session"],
        groq=state["groq_client"],
        query=state["query"],
        query_embedding=state["query_embedding"],
        fts_query=state["fts_query"],
        section_id=state["section_id"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    token_usage = state.get("token_usage", {})
    token_usage["rerank"] = usage
    
    return {
        "search_results": results,
        "token_usage": token_usage,
        "timings": {**state.get("timings", {}), "search_ms": round(elapsed_ms, 1)},
    }


async def check_sufficiency(state: AgentState) -> dict:
    t0 = time.perf_counter()
    results = state.get("search_results", [])

    if not results:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "context_sufficient": False,
            "timings": {**state.get("timings", {}), "check_ms": round(elapsed_ms, 1)},
        }

    # Since we bypassed LLM reranker and use RRF scores directly (which are small fractions ~0.03),
    # any valid results returned by the search are considered sufficient.
    is_sufficient = len(results) > 0

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "context_sufficient": is_sufficient,
        "timings": {**state.get("timings", {}), "check_ms": round(elapsed_ms, 1)},
    }


async def expand_search(state: AgentState) -> dict:
    return {
        "section_id": None,
        "search_mode": "global",
        "iteration": state.get("iteration", 0) + 1,
    }


async def generate_answer(state: AgentState) -> dict:
    t0 = time.perf_counter()
    results = state.get("search_results", [])
    
    if not results or not state.get("context_sufficient"):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": "Інформації не знайдено в базі знань університету.",
            "sources": [],
            "timings": {**state.get("timings", {}), "answer_ms": round(elapsed_ms, 1)},
        }
        
    context_text = "\n\n".join(f"[{i+1}] {r.raw_text}" for i, r in enumerate(results))
    user_prompt = f"Запит: {state['query']}\n\nКонтекст:\n{context_text}"
    
    response = await state["groq_client"].complete(ANSWER_SYSTEM, user_prompt)
    
    sources = []
    for r in results:
        src = f"{r.source_doc_title}, сторінка {r.source_page_number}"
        if src not in sources:
            sources.append(src)
            
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    token_usage = state.get("token_usage", {})
    token_usage["answer"] = response.usage
    
    return {
        "answer": response.content,
        "sources": sources,
        "token_usage": token_usage,
        "timings": {**state.get("timings", {}), "answer_ms": round(elapsed_ms, 1)},
    }
