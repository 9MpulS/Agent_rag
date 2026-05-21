"""Agent state definition."""

from typing import TypedDict
from agent_rag.llm.groq_client import LLMUsage
from agent_rag.search.hybrid_search import RankedResult
from sqlalchemy.ext.asyncio import AsyncSession
from agent_rag.llm.groq_client import GroqClient

class AgentState(TypedDict):
    # Session and client
    session: AsyncSession
    groq_client: GroqClient
    
    # State data
    query: str
    query_embedding: list[float]
    section_id: int | None
    fts_query: str
    search_results: list[RankedResult]
    context_sufficient: bool
    answer: str
    sources: list[str]
    iteration: int
    search_mode: str
    
    # Metrics
    timings: dict[str, float]
    token_usage: dict[str, LLMUsage]
