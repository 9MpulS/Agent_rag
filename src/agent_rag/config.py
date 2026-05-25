"""Global settings configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ── Ollama / Embeddings ───────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 5

    # ── Groq / LLM ────────────────────────────────────────────
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_TOKENS: int = 1024

    # ── Groq Pricing (USD per token) ──────────────────────────
    GROQ_PRICE_INPUT_PER_TOKEN: float = 0.59 / 1_000_000
    GROQ_PRICE_OUTPUT_PER_TOKEN: float = 0.79 / 1_000_000

    # ── Vector Search ─────────────────────────────────────────
    VECTOR_TOP_K: int = 15

    # ── Full-Text Search ──────────────────────────────────────
    FTS_TOP_K: int = 15
    FTS_LANGUAGE: str = "simple"

    # ── Hybrid Search / Re-ranking ────────────────────────────
    RERANK_TOP_PAGES: int = 15
    RERANK_MIN_SCORE: float = 0.0
    RERANK_FINAL_TOP_K: int = 5

    # ── Routing ───────────────────────────────────────────────
    ROUTING_MIN_SIMILARITY: float = 0.3
    ROUTING_SECONDARY_THRESHOLD: float = 0.7

    # ── Agent / Graph ─────────────────────────────────────────
    MAX_RETRY_ITERATIONS: int = 5

    # ── Benchmark / Testing ───────────────────────────────────
    BENCHMARK_SLEEP_BETWEEN_QUERIES: float = 15.0
    BENCHMARK_TARGET_RECALL: float = 0.75
    BENCHMARK_TARGET_MRR: float = 0.60

settings = Settings()
