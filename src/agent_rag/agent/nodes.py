"""Agentic RAG nodes: agent_decide and execute_tool.

Uses native Groq tool calling (OpenAI-compatible function calling).
The agent loop:
1. agent_decide: LLM sees conversation history + tool defs → returns tool_calls or text
2. execute_tool: runs the selected tool, adds tool result to messages
3. Back to agent_decide until text response or max iterations
"""

import json
import time
import structlog

from agent_rag.config import settings
from agent_rag.agent.state import AgentState
from agent_rag.agent.tools import (
    TOOL_DEFINITIONS,
    execute_search,
    execute_refine_query,
)
from agent_rag.llm.prompts import AGENT_SYSTEM

logger = structlog.get_logger()


async def agent_decide(state: AgentState) -> dict:
    """LLM decides the next action via JSON output.

    Sends the conversation history to the LLM and parses the JSON response
    to determine the next action.
    """
    t0 = time.perf_counter()
    groq = state["groq_client"]

    # Build user message from conversation history
    messages = state.get("messages", [])
    user_content = "\n\n".join(m["content"] for m in messages if m["role"] == "user")

    # Include tool results in conversation context
    assistant_and_tool_parts = []
    for m in messages:
        if m["role"] == "assistant":
            assistant_and_tool_parts.append(f"[Попереднє рішення]: {m['content']}")
        elif m["role"] == "tool_result":
            assistant_and_tool_parts.append(f"[Результат інструменту]: {m['content']}")

    full_user_prompt = user_content
    if assistant_and_tool_parts:
        full_user_prompt += "\n\n--- ІСТОРІЯ ВИКОНАННЯ ---\n" + "\n\n".join(assistant_and_tool_parts)

    parsed, usage = await groq.complete_json(AGENT_SYSTEM, full_user_prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    action = parsed.get("action", "final_answer")
    params = parsed.get("params", {})

    # Update token usage
    token_usage = dict(state.get("token_usage", {}))
    decide_key = f"agent_decide_{state.get('iteration', 0)}"
    token_usage[decide_key] = usage

    # Add assistant message to history
    new_messages = list(messages) + [
        {"role": "assistant", "content": str(parsed)}
    ]

    logger.info(
        "agent_decide",
        action=action,
        params=params,
        iteration=state.get("iteration", 0),
        latency_ms=round(elapsed_ms, 1),
    )

    result: dict = {
        "messages": new_messages,
        "last_action": action,
        "last_action_params": params,
        "token_usage": token_usage,
        "timings": {**state.get("timings", {}), f"decide_{state.get('iteration', 0)}_ms": round(elapsed_ms, 1)},
    }

    # If final_answer, extract answer and sources directly
    if action == "final_answer":
        result["answer"] = parsed.get("answer", "Інформації не знайдено в базі знань університету.")
        result["sources"] = parsed.get("sources", [])

    return result


async def execute_tool(state: AgentState) -> dict:
    """Execute the tool chosen by agent_decide.

    Runs either search or refine_query, then adds the result to the
    conversation history as a tool_result message.
    """
    t0 = time.perf_counter()
    action = state["last_action"]
    params = state.get("last_action_params", {})
    messages = list(state.get("messages", []))
    tool_calls_log = list(state.get("tool_calls_log", []))
    timings = dict(state.get("timings", {}))
    token_usage = dict(state.get("token_usage", {}))
    iteration = state.get("iteration", 0)

    if action == "search":
        query = params.get("query", state["query"])

        results, search_timings, search_token_usage = await execute_search(
            session=state["session"],
            groq=state["groq_client"],
            query=query,
        )

        # Merge timings and token usage
        for k, v in search_timings.items():
            timings[f"iter{iteration}_{k}"] = v
        for k, v in search_token_usage.items():
            token_usage[f"iter{iteration}_{k}"] = v

        # Build context from results
        if results:
            context_parts = []
            for i, r in enumerate(results):
                src = f"{r.source_doc_title}, сторінка {r.source_page_number}"
                context_parts.append(f"[{i+1}] (Джерело: {src})\n{r.raw_text}")
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Пошук не дав результатів."

        # Accumulate search results
        existing_results = list(state.get("search_results", []))
        existing_results.extend(results)

        # Accumulate context
        prev_context = state.get("accumulated_context", "")
        if prev_context:
            accumulated = f"{prev_context}\n\n--- Результати пошуку (ітерація {iteration + 1}) ---\n{context_text}"
        else:
            accumulated = context_text

        result_summary = f"Знайдено {len(results)} фрагментів для запиту '{query[:50]}'"
        messages.append({"role": "tool_result", "content": f"Результати пошуку:\n{context_text}"})

        tool_calls_log.append({
            "tool": "search",
            "params": {"query": query},
            "result_summary": result_summary,
        })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings[f"tool_exec_{iteration}_ms"] = round(elapsed_ms, 1)

        return {
            "messages": messages,
            "search_results": existing_results,
            "accumulated_context": accumulated,
            "iteration": iteration + 1,
            "tool_calls_log": tool_calls_log,
            "timings": timings,
            "token_usage": token_usage,
        }

    elif action == "refine_query":
        original_query = params.get("original_query", state["query"])
        problem = params.get("problem", "Результати нерелевантні")

        refine_result, refine_usage = await execute_refine_query(
            groq=state["groq_client"],
            original_query=original_query,
            problem=problem,
        )

        token_usage[f"iter{iteration}_refine"] = refine_usage

        refined_query = refine_result["refined_query"]
        reasoning = refine_result["reasoning"]

        result_summary = f"Запит уточнено: '{refined_query[:50]}' (причина: {reasoning[:50]})"
        messages.append({
            "role": "tool_result",
            "content": f"Уточнений запит: {refined_query}\nОбґрунтування: {reasoning}",
        })

        tool_calls_log.append({
            "tool": "refine_query",
            "params": {"original_query": original_query, "problem": problem},
            "result_summary": result_summary,
        })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings[f"tool_exec_{iteration}_ms"] = round(elapsed_ms, 1)

        return {
            "messages": messages,
            "iteration": iteration + 1,
            "tool_calls_log": tool_calls_log,
            "timings": timings,
            "token_usage": token_usage,
        }

    else:
        # Unknown action — treat as done
        logger.warning("unknown_tool_action", action=action)
        return {"iteration": iteration + 1}



