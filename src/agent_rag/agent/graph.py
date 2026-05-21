"""Agent graph definition."""

from langgraph.graph import StateGraph, END
from agent_rag.config import settings
from agent_rag.agent.state import AgentState
from agent_rag.agent.nodes import (
    embed_query,
    route_to_section,
    enhance_fts_query,
    search,
    check_sufficiency,
    expand_search,
    generate_answer,
)

builder = StateGraph(AgentState)

builder.add_node("embed_query", embed_query)
builder.add_node("route_to_section", route_to_section)
builder.add_node("enhance_fts_query", enhance_fts_query)
builder.add_node("search", search)
builder.add_node("check_sufficiency", check_sufficiency)
builder.add_node("expand_search", expand_search)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("embed_query")
builder.add_edge("embed_query", "route_to_section")
builder.add_edge("route_to_section", "enhance_fts_query")
builder.add_edge("enhance_fts_query", "search")
builder.add_edge("search", "check_sufficiency")

def sufficiency_router(state: AgentState) -> str:
    if state.get("context_sufficient"):
        return "generate_answer"
    if state.get("iteration", 0) >= settings.MAX_RETRY_ITERATIONS:
        return "generate_answer"
    return "expand_search"

builder.add_conditional_edges("check_sufficiency", sufficiency_router)
builder.add_edge("expand_search", "enhance_fts_query")
builder.add_edge("generate_answer", END)

graph = builder.compile()
