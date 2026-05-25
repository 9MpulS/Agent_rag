import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from agent_rag.config import settings
from agent_rag.llm.groq_client import GroqClient
from agent_rag.search.fts_search import fts_query_enhance
from agent_rag.search.hybrid_search import hybrid_search
from agent_rag.embeddings import get_embedding

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    groq_client = GroqClient()
    
    query = "Що регулює Положення про вчену раду?"
    with open("fts_test_output.txt", "w", encoding="utf-8") as f:
        f.write(f"--- Original Query ---\n{query}\n\n")
        f.write("--- Testing FTS Enhancement ---\n")
        
        try:
            fts_query, usage = await fts_query_enhance(groq_client, query)
            f.write(f"Enhanced FTS Query:\n{fts_query}\n\n")
        except Exception as e:
            f.write(f"Error during enhancement: {e}\n")
            return
            
        f.write("--- Running Hybrid Search ---\n")
        query_embedding = await get_embedding(query)
        
        async with async_session() as session:
            try:
                results, hyb_usage = await hybrid_search(
                    session, groq_client, query, query_embedding, fts_query, section_id=None
                )
                f.write(f"Hybrid Search Returned {len(results)} results\n")
                for i, r in enumerate(results[:3]):
                    f.write(f"[{i}] Score: {r.llm_score:.4f} | Page: {r.source_page_number} | Doc: {r.source_doc_title}\n")
            except Exception as e:
                f.write(f"Error during hybrid search: {e}\n")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
