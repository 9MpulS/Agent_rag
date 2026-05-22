import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, func
from groq import AsyncGroq
from agent_rag.db.models import Chunk, RegistrySection
from agent_rag.config import settings

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
    benchmark_data = []
    
    async with async_session() as session:
        # Get all sections
        result = await session.execute(select(RegistrySection.id).order_by(RegistrySection.id))
        section_ids = [row[0] for row in result.all()]
        
        for sid in section_ids:
            # Get 4 random chunks for each section
            # We want chunks that have at least some reasonable text length
            stmt = select(Chunk).where(
                Chunk.registry_section_id == sid,
                func.length(Chunk.content) > 200
            ).order_by(func.random()).limit(4)
            
            chunks_result = await session.execute(stmt)
            chunks = chunks_result.scalars().all()
            
            for chunk in chunks:
                prompt = f"На основі наступного тексту сформулюй одне коротке і чітке запитання українською мовою, відповіддю на яке буде цей текст. Поверни ТІЛЬКИ запитання, ніяких інших слів:\n\nТекст:\n{chunk.content}"
                
                try:
                    response = await groq_client.chat.completions.create(
                        model=settings.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=60
                    )
                    question = response.choices[0].message.content.strip().strip('"\'')
                    
                    benchmark_data.append({
                        "query": question,
                        "expected_section_id": chunk.registry_section_id,
                        "relevant_chunk_ids": [chunk.id],
                        "relevant_page_ids": [chunk.page_id]
                    })
                except Exception as e:
                    print(f"Error calling Groq for chunk {chunk.id}: {e}")
                    
    # Write to tests/benchmark_data.py
    with open("tests/benchmark_data.py", "w", encoding="utf-8") as f:
        f.write('"""Benchmark ground truth dataset."""\n\n')
        f.write('BENCHMARK: list[dict] = [\n')
        for i, item in enumerate(benchmark_data):
            f.write('    {\n')
            f.write(f'        "query": "{item["query"].replace("\"", "")}",\n')
            f.write(f'        "expected_section_id": {item["expected_section_id"]},\n')
            f.write(f'        "relevant_chunk_ids": {item["relevant_chunk_ids"]},\n')
            f.write(f'        "relevant_page_ids": {item["relevant_page_ids"]},\n')
            f.write('    }')
            if i < len(benchmark_data) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')

    print(f"Generated {len(benchmark_data)} benchmark queries in tests/benchmark_data.py")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
