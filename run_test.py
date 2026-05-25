import asyncio
from agent_rag.db.engine import get_session
from agent_rag.llm.groq_client import GroqClient
from agent_rag.agent.graph import graph
from agent_rag.agent.state import AgentState

async def run():
    async for session in get_session():
        groq_client = GroqClient()
        query = "За що мене можуть відрахувати?"
        
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
        
        print("Starting graph...")
        async for event in graph.astream(initial_state):
            for k, v in event.items():
                print(f"NODE FIRED: {k}")
                if "last_action" in v:
                    print(f"  Action: {v['last_action']}")
                if "iteration" in v:
                    print(f"  Iteration: {v['iteration']}")
        print("Graph finished!")

if __name__ == "__main__":
    asyncio.run(run())
