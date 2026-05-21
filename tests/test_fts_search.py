"""Tests for full-text search."""

import pytest
import time
from agent_rag.search.fts_search import fts_query_enhance, fts_search
from tests.benchmark_data import BENCHMARK

@pytest.mark.asyncio
async def test_fts_search_benchmark(db_session, groq_client):
    latencies = []
    
    for item in BENCHMARK:
        query = item["query"]
        t0 = time.perf_counter()
        
        fts_q, usage = await fts_query_enhance(groq_client, query)
        results = await fts_search(
            db_session, 
            fts_q, 
            section_id=None
        )
        
        latencies.append((time.perf_counter() - t0) * 1000)
        
        # We just measure here, no assert on recall since it's just FTS
        assert results is not None
        
    print(f"\nFTS Search Avg Latency: {sum(latencies)/len(latencies):.1f}ms")
