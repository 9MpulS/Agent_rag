import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from agent_rag.config import settings
from sqlalchemy import text

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.connect() as conn:
        res = await conn.execute(text('SHOW sharedir;'))
        print(res.scalar())
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
