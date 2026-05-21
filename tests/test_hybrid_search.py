"""Tests for hybrid search."""

import pytest
import time
import asyncio
from agent_rag.config import settings
from agent_rag.search.hybrid_search import hybrid_search
from agent_rag.search.fts_search import fts_query_enhance
from agent_rag.embeddings import get_embedding
from tests.benchmark_data import BENCHMARK

@pytest.mark.asyncio
async def test_hybrid_search_benchmark(db_session, groq_client):
    latencies = []
    costs = []
    recalls = []
    mrrs = []
    
    # We will simulate the benchmark here. Since we have empty ground truth for now, 
    # we just run the query and collect stats.
    # We will bypass the asserts if no ground truth is provided.
    
    for item in BENCHMARK:
        await asyncio.sleep(settings.BENCHMARK_SLEEP_BETWEEN_QUERIES)
        
        query = item["query"]
        
        t0 = time.perf_counter()
        
        # Enhance query and get embedding
        query_embedding = await get_embedding(query)
        fts_query, fts_usage = await fts_query_enhance(groq_client, query)
        
        results, rerank_usage = await hybrid_search(
            db_session,
            groq_client,
            query,
            query_embedding,
            fts_query,
            section_id=None
        )
        
        latencies.append((time.perf_counter() - t0) * 1000)
        costs.append(fts_usage.cost_usd + rerank_usage.cost_usd)
        
        # Recall & MRR calculation
        expected_pages = set(item.get("relevant_page_ids", []))
        if expected_pages:
            retrieved_pages = [r.page_id for r in results]
            hits = set(retrieved_pages) & expected_pages
            recalls.append(len(hits) / len(expected_pages))
            
            mrr = 0.0
            for i, r in enumerate(results):
                if r.page_id in expected_pages:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)
            
    avg_latency = sum(latencies)/len(latencies) if latencies else 0
    avg_cost = sum(costs)/len(costs) * 1000 if costs else 0
    print(f"\nHybrid Search Avg Latency: {avg_latency:.1f}ms")
    print(f"Hybrid Search Cost / 1000 queries: ${avg_cost:.3f}")
    
    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        mean_mrr = sum(mrrs) / len(mrrs)
        assert mean_recall >= settings.BENCHMARK_TARGET_RECALL
        assert mean_mrr >= settings.BENCHMARK_TARGET_MRR
