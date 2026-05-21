"""Tests for the agent pipeline."""

import pytest
from agent_rag.agent.graph import graph
from agent_rag.agent.state import AgentState

@pytest.mark.asyncio
async def test_agent_pipeline_e2e(db_session, groq_client):
    # Pass dependencies via state
    initial_state = {
        "session": db_session,
        "groq_client": groq_client,
        "query": "Які підстави для відрахування студента?",
        "iteration": 0,
        "timings": {},
        "token_usage": {}
    }
    
    result = await graph.ainvoke(initial_state)
    
    assert result["answer"] is not None
    assert isinstance(result["sources"], list)
    assert "timings" in result
    assert "embed_ms" in result["timings"]
    assert "answer_ms" in result["timings"]
