"""E2E tests for the Agentic RAG pipeline."""

import pytest
from agent_rag.agent.graph import graph
from agent_rag.agent.state import AgentState
from agent_rag.config import settings


@pytest.mark.asyncio
async def test_graph_compiles():
    """Test that the agentic graph compiles without errors."""
    assert graph is not None
    # Graph should have 2 nodes: agent and tools
    assert "agent" in graph.nodes
    assert "tools" in graph.nodes


@pytest.mark.asyncio
async def test_agentic_rag_e2e(db_session, groq_client):
    """E2E test: 'За що мене можуть відрахувати?'

    Verifies:
    1. Agent produces a non-empty answer
    2. Answer contains keywords about expulsion
    3. Agent used at least one tool call
    4. Safety limit is respected
    """
    query = "За що мене можуть відрахувати?"
    initial_state: AgentState = {
        "session": db_session,
        "groq_client": groq_client,
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "iteration": 0,
        "search_results": [],
        "accumulated_context": "",
        "last_action": "",
        "last_action_params": {},
        "answer": "",
        "sources": [],
        "tool_calls_log": [],
        "timings": {},
        "token_usage": {},
    }

    result = await graph.ainvoke(initial_state)

    # 1. Answer exists and is non-empty
    answer = result.get("answer", "")
    assert answer, "Answer should not be empty"
    assert len(answer) > 50, f"Answer too short: {len(answer)} chars"

    # 2. Answer contains keywords about expulsion
    answer_lower = answer.lower()
    expulsion_keywords = ["відрахування", "відрахувати", "виключення", "припинення"]
    assert any(kw in answer_lower for kw in expulsion_keywords), (
        f"Answer should contain expulsion keywords. Got: {answer[:200]}"
    )

    # 3. Agent used at least one tool call
    tool_calls = result.get("tool_calls_log", [])
    assert len(tool_calls) >= 1, "Agent should have made at least one tool call"

    # Verify the first tool call was 'search'
    assert tool_calls[0]["tool"] == "search", "First tool call should be 'search'"

    # 4. Safety limit is respected
    assert result["iteration"] <= settings.MAX_RETRY_ITERATIONS, (
        f"Iterations ({result['iteration']}) exceeded max ({settings.MAX_RETRY_ITERATIONS})"
    )

    # 5. Timings are recorded
    assert result.get("timings"), "Timings should be recorded"


@pytest.mark.asyncio
async def test_agentic_rag_simple_question(db_session, groq_client):
    """Test with a simpler question to verify the basic flow."""
    query = "Як взяти академічну відпустку?"
    initial_state: AgentState = {
        "session": db_session,
        "groq_client": groq_client,
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "iteration": 0,
        "search_results": [],
        "accumulated_context": "",
        "last_action": "",
        "last_action_params": {},
        "answer": "",
        "sources": [],
        "tool_calls_log": [],
        "timings": {},
        "token_usage": {},
    }

    result = await graph.ainvoke(initial_state)

    assert result.get("answer"), "Answer should not be empty"
    assert result.get("tool_calls_log"), "Tool calls log should not be empty"
