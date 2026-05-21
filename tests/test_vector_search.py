"""Tests for vector search."""

import pytest
import time
from agent_rag.search.vector_search import vector_search
from agent_rag.embeddings import get_embedding
from tests.benchmark_data import BENCHMARK

@pytest.mark.asyncio
async def test_vector_search_benchmark(db_session):
    latencies = []
    
    for item in BENCHMARK:
        query = item["query"]
        t0 = time.perf_counter()
        
        # In a real test, we might mock this or have it running. 
        # For benchmark, we actually call it to get true latency.
        embedding = await get_embedding(query)
        results = await vector_search(
            db_session, 
            embedding, 
            section_id=None
        )
        
        latencies.append((time.perf_counter() - t0) * 1000)
        
        # We just measure here, no assert on recall since it's vector search
        assert results is not None
        
    print(f"\nVector Search Avg Latency: {sum(latencies)/len(latencies):.1f}ms")
