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
            "query_embedding": [],
            "fts_query": "",
            "section_id": None,
            "search_mode": "global",
            "search_results": [],
            "context_sufficient": False,
            "iteration": 0,
            "answer": "",
            "sources": [],
            "token_usage": {},
            "timings": {}
        }
        
        result_state = await graph.ainvoke(initial_state)
        
        with open("rag_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Запит: '{query}'\n")
            f.write(f"Знайдено фрагментів: {len(result_state['search_results'])}\n")
            if result_state['search_results']:
                for r in result_state['search_results']:
                    f.write(f"  page_id={r.page_id}, score={r.llm_score}, стор={r.source_page_number}\n")
                    f.write(f"  Текст: {r.raw_text[:150].replace(chr(10), ' ')}\n")
            f.write(f"Достатність контексту: {result_state['context_sufficient']}\n")
            f.write("=" * 50 + "\n")
            f.write("ВІДПОВІДЬ:\n")
            f.write(result_state["answer"] + "\n")
            f.write("=" * 50 + "\n")
            f.write("ДЖЕРЕЛА:\n")
            if result_state["sources"]:
                for i, src in enumerate(result_state["sources"], 1):
                    f.write(f"[{i}] {src}\n")
            else:
                f.write("Не знайдено.\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "Як взяти академ відпустку?"

    asyncio.run(run_rag(query))
