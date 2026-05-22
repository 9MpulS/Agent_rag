"""Async Ollama embeddings client."""

import httpx
import structlog
from agent_rag.config import settings

logger = structlog.get_logger()

async def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text string."""
    url = f"{settings.OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": text,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except Exception as e:
            logger.error("embedding_error", text=text[:50], error=str(e))
            raise

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    url = f"{settings.OLLAMA_BASE_URL}/api/embed"
    embeddings = []
    
    # Process in batches to avoid overloading the local model
    for i in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + settings.EMBEDDING_BATCH_SIZE]
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "input": batch,
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                embeddings.extend(data["embeddings"])
            except Exception as e:
                logger.error("embedding_batch_error", batch_size=len(batch), error=str(e))
                raise
                
    return embeddings
