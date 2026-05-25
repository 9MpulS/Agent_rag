"""Agent state definition for Agentic RAG."""

from typing import TypedDict
from agent_rag.llm.groq_client import LLMUsage
from agent_rag.search.hybrid_search import RankedResult
from sqlalchemy.ext.asyncio import AsyncSession
from agent_rag.llm.groq_client import GroqClient


class ToolCallRecord(TypedDict):
    """Record of a single tool call for observability."""
    tool: str
    params: dict
    result_summary: str


class AgentState(TypedDict):
    # Session and client
    session: AsyncSession
    groq_client: GroqClient

    # Core
    query: str                              # Original user question
    messages: list[dict]                    # Conversation history [{role, content}, ...]
    iteration: int                          # Tool call counter (safety limit)

    # Search results
    search_results: list[RankedResult]      # Accumulated search results
    accumulated_context: str                # Combined text from all search results

    # Agent decision
    last_action: str                        # Last action decided by agent
    last_action_params: dict                # Params of the last action

    # Output
    answer: str
    sources: list[str]

    # Observability
    tool_calls_log: list[ToolCallRecord]    # Log of all tool calls
    timings: dict[str, float]
    token_usage: dict[str, LLMUsage]
