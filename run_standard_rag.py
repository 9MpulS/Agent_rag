import asyncio
import time
from agent_rag.db.engine import get_session
from agent_rag.llm.groq_client import GroqClient
from agent_rag.agent.tools import execute_search
from agent_rag.llm.prompts import ANSWER_SYSTEM

async def run_standard_rag(query: str):
    async for session in get_session():
        t0 = time.perf_counter()
        groq_client = GroqClient()

        # 1. Search for context
        results, search_timings, search_token_usage = await execute_search(
            session=session,
            groq=groq_client,
            query=query,
        )

        # 2. Format context
        if results:
            context_parts = []
            for i, r in enumerate(results):
                src = f"{r.source_doc_title}, сторінка {r.source_page_number}"
                context_parts.append(f"[{i+1}] (Джерело: {src})\n{r.raw_text}")
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Пошук не дав результатів."

        # 3. Generate answer
        user_prompt = f"Запит користувача: {query}\n\nКонтекст:\n{context_text}"
        response = await groq_client.complete(ANSWER_SYSTEM, user_prompt)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Output results
        with open("standard_rag_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Запит: '{query}'\n")
            f.write("=" * 50 + "\n")
            
            f.write(f"Знайдено фрагментів: {len(results)}\n")
            if results:
                for r in results:
                    f.write(f"  page_id={r.page_id}, score={r.llm_score}, стор={r.source_page_number}\n")
            f.write("=" * 50 + "\n")
            
            f.write("ВІДПОВІДЬ:\n")
            f.write(response.content + "\n")
            f.write("=" * 50 + "\n")
            
            f.write("TIMINGS:\n")
            for k, v in sorted(search_timings.items()):
                f.write(f"  {k}: {v}ms\n")
            f.write(f"  total_time_ms: {round(elapsed_ms, 1)}ms\n")
            
        print(f"Готово. Результат збережено у standard_rag_output.txt")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "як мені вступити до сумду"

    asyncio.run(run_standard_rag(query))
