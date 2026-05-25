"""Run the Agentic RAG pipeline."""

import asyncio
from agent_rag.db.engine import get_session
from agent_rag.llm.groq_client import GroqClient
from agent_rag.agent.graph import graph
from agent_rag.agent.state import AgentState


async def run_rag(query: str):
    async for session in get_session():
        groq_client = GroqClient()

        initial_state: AgentState = {
            "query": query,
            "session": session,
            "groq_client": groq_client,
            "messages": [{"role": "user", "content": query}],
            "iteration": 0,
            "search_results": [],
            "accumulated_context": "",
            "last_action": "",
            "last_action_params": {},
            "answer": "",
            "sources": [],
            "tool_calls_log": [],
            "token_usage": {},
            "timings": {},
        }

        result_state = await graph.ainvoke(initial_state)

        with open("rag_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Запит: '{query}'\n")
            f.write(f"Ітерацій: {result_state['iteration']}\n")
            f.write("=" * 50 + "\n")

            # Tool calls trace
            f.write("TOOL CALLS TRACE:\n")
            for i, tc in enumerate(result_state.get("tool_calls_log", []), 1):
                f.write(f"  [{i}] {tc['tool']}: {tc['params']}\n")
                f.write(f"       → {tc['result_summary']}\n")
            f.write("=" * 50 + "\n")

            # Search results
            f.write(f"Знайдено фрагментів: {len(result_state.get('search_results', []))}\n")
            if result_state.get("search_results"):
                for r in result_state["search_results"]:
                    f.write(f"  page_id={r.page_id}, score={r.llm_score}, стор={r.source_page_number}\n")
                    f.write(f"  Текст: {r.raw_text[:150].replace(chr(10), ' ')}\n")
            f.write("=" * 50 + "\n")

            # Answer
            f.write("ВІДПОВІДЬ:\n")
            f.write(result_state.get("answer", "Немає відповіді") + "\n")
            f.write("=" * 50 + "\n")

            # Sources
            f.write("ДЖЕРЕЛА:\n")
            sources = result_state.get("sources", [])
            if sources:
                for i, src in enumerate(sources, 1):
                    f.write(f"[{i}] {src}\n")
            else:
                f.write("Не знайдено.\n")

            # Timings
            f.write("=" * 50 + "\n")
            f.write("TIMINGS:\n")
            for k, v in sorted(result_state.get("timings", {}).items()):
                f.write(f"  {k}: {v}ms\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "За що мене можуть відрахувати?"

    asyncio.run(run_rag(query))
