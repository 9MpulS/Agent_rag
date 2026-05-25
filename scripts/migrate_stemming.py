import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text, select, update
from agent_rag.db.models import Chunk, TSV_TRIGGER_FUNCTION
from agent_rag.config import settings
from agent_rag.search.ukrainian_stemmer import stem_ukrainian_text

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with engine.begin() as conn:
        print("Adding column content_stemmed...")
        try:
            await conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_stemmed TEXT"))
        except Exception as e:
            print(f"Column might already exist: {e}")
            
        print("Updating trigger function...")
        # Since TSV_TRIGGER_FUNCTION is a DDL object, we can just execute its statement
        await conn.execute(text("""
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('simple', COALESCE(NEW.content_stemmed, NEW.content));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """))
        
    async with async_session() as session:
        print("Fetching chunks to update stemming...")
        result = await session.execute(select(Chunk))
        chunks = result.scalars().all()
        print(f"Found {len(chunks)} chunks to process.")
        
        for c in chunks:
            if not c.content_stemmed:
                c.content_stemmed = stem_ukrainian_text(c.content)
                
        await session.commit()
        
    # Wait, SQLAlchemy ORM updates will trigger the new trigger, updating content_tsv!
    # But just in case, we can run an explicit UPDATE chunks SET content_tsv = to_tsvector('simple', COALESCE(content_stemmed, content))
    async with engine.begin() as conn:
        print("Forcing content_tsv update for all rows...")
        await conn.execute(text("UPDATE chunks SET content_tsv = to_tsvector('simple', COALESCE(content_stemmed, content))"))
        
    print("Migration complete!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
