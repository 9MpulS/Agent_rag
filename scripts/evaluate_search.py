import asyncio
import json
import time
import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from agent_rag.db.models import Page
from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient
from agent_rag.embeddings import get_embedding
from agent_rag.search.vector_search import vector_search
from agent_rag.search.fts_search import fts_query_enhance, fts_search
from agent_rag.search.hybrid_search_llm import hybrid_search
from agent_rag.search.ukrainian_stemmer import stem_ukrainian_text

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    groq_client = GroqClient()
    
    with open("tests/benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    metrics = {
        "vector": {"doc_hits": 0, "page_hits": 0, "latencies": [], "queries": 0},
        "fts": {"doc_hits": 0, "page_hits": 0, "latencies": [], "queries": 0},
        "hybrid": {"doc_hits": 0, "page_hits": 0, "latencies": [], "queries": 0},
    }
    
    async with async_session() as session:
        page_doc_map = {}
        pages_res = await session.execute(select(Page))
        for p in pages_res.scalars().all():
            page_doc_map[p.id] = p.document_id
            
        for i, case in enumerate(benchmark):
            if i >= 5: break
            query = case.get("query", "")
            if not query:
                continue
                
            expected_doc_id = case.get("expected_document_id")
            expected_page_id = case.get("expected_page_id")
            if not expected_doc_id or not expected_page_id:
                continue
                
            # 1. Vector Search
            t0 = time.perf_counter()
            query_embedding = await get_embedding(query)
            vec_res = await vector_search(session, query_embedding, section_id=None, top_k=settings.VECTOR_TOP_K)
            t_vec = time.perf_counter() - t0
            metrics["vector"]["latencies"].append(t_vec)
            metrics["vector"]["queries"] += 1
            
            vec_doc_found = any(page_doc_map.get(r.page_id) == expected_doc_id for r in vec_res)
            vec_page_found = any(r.page_id == expected_page_id for r in vec_res)
            if vec_doc_found: metrics["vector"]["doc_hits"] += 1
            if vec_page_found: metrics["vector"]["page_hits"] += 1
            
            # 2. FTS Search
            t0 = time.perf_counter()
            try:
                fts_query, _ = await fts_query_enhance(groq_client, query)
                fts_res = await fts_search(session, fts_query, section_id=None, top_k=settings.FTS_TOP_K)
            except Exception as e:
                await session.rollback()
                try:
                    fts_query = stem_ukrainian_text(re.sub(r'[^a-zA-Zа-яА-Я0-9їієґЇІЄҐ\s]', ' ', query).strip())
                    fts_res = await fts_search(session, fts_query, section_id=None, top_k=settings.FTS_TOP_K)
                except Exception:
                    await session.rollback()
                    fts_res = []
            
            t_fts = time.perf_counter() - t0
            metrics["fts"]["latencies"].append(t_fts)
            metrics["fts"]["queries"] += 1
            
            fts_doc_found = any(page_doc_map.get(r.page_id) == expected_doc_id for r in fts_res)
            fts_page_found = any(r.page_id == expected_page_id for r in fts_res)
            if fts_doc_found: metrics["fts"]["doc_hits"] += 1
            if fts_page_found: metrics["fts"]["page_hits"] += 1
            
            # 3. Hybrid Search
            t0 = time.perf_counter()
            try:
                hyb_res, _ = await hybrid_search(session, groq_client, query, query_embedding, fts_query, section_id=None)
            except Exception as e:
                await session.rollback()
                hyb_res = []
                
            t_hyb = time.perf_counter() - t0
            metrics["hybrid"]["latencies"].append(t_hyb)
            metrics["hybrid"]["queries"] += 1
            
            hyb_doc_found = any(page_doc_map.get(r.page_id) == expected_doc_id for r in hyb_res)
            hyb_page_found = any(r.page_id == expected_page_id for r in hyb_res)
            if hyb_doc_found: metrics["hybrid"]["doc_hits"] += 1
            if hyb_page_found: metrics["hybrid"]["page_hits"] += 1

    report = "# Результати бенчмаркінгу методів пошуку\n\n"
    report += "| Метод пошуку | К-сть запитів | Hit Rate (Документ) | Hit Rate (Сторінка) | Середній час (с) |\n"
    report += "|---|---|---|---|---|\n"
    
    for method, data in metrics.items():
        q = data["queries"]
        if q == 0: continue
        doc_hr = (data["doc_hits"] / q) * 100
        page_hr = (data["page_hits"] / q) * 100
        avg_lat = sum(data["latencies"]) / q
        report += f"| {method.upper()} | {q} | {doc_hr:.1f}% | {page_hr:.1f}% | {avg_lat:.3f} |\n"
        
    with open("C:/Users/VladK/.gemini/antigravity/brain/b6632344-15a5-411d-a37a-ac14e1ec3fcd/benchmark_metrics_results.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Done")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
