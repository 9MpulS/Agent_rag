# University RAG Agent (СумДУ)

MVP для дослідження методів оптимізації пошуку в україномовних нормативно-юридичних документах.
Порівняння трьох режимів: Vector Search, Full-Text Search, Hybrid + LLM Rerank.

## Архітектура
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Embeddings**: Ollama (paraphrase-multilingual-mpnet-base-v2)
- **БД**: PostgreSQL 16 + pgvector
- **Оркестрація**: LangGraph
