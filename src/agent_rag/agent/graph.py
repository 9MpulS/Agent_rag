"""Agentic RAG graph definition.

Reactive agent loop:
  agent_decide ──→ (tool call?) ──→ execute_tool ──→ agent_decide
                 └─ (final_answer or max iter) ──→ END
"""

from langgraph.graph import StateGraph, END
from agent_rag.config import settings
from agent_rag.agent.state import AgentState
from agent_rag.agent.nodes import agent_decide, execute_tool

builder = StateGraph(AgentState)

# Two nodes in the agent loop
builder.add_node("agent", agent_decide)
builder.add_node("tools", execute_tool)

builder.set_entry_point("agent")


def should_continue(state: AgentState) -> str:
    """Route based on agent's decision.

    - tool call (search / refine_query) → "tools"
    - final_answer or max iterations   → END
    """
    action = state.get("last_action", "final_answer")
    iteration = state.get("iteration", 0)

    # Safety limit
    if iteration >= settings.MAX_RETRY_ITERATIONS:
        return END

    if action in ("search", "refine_query"):
        return "tools"

    # final_answer or any unknown action
    return END


builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()
