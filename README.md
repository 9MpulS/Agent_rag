# University RAG Agent (СумДУ)

MVP для дослідження методів оптимізації пошуку в україномовних нормативно-юридичних документах.
Порівняння трьох режимів: Vector Search, Full-Text Search, Hybrid + LLM Rerank.

## Архітектура
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Embeddings**: Ollama (paraphrase-multilingual-mpnet-base-v2)
- **БД**: PostgreSQL 16 + pgvector
- **Оркестрація**: LangGraph

## Benchmark Очікування

| Method                | Recall@5 |  MRR  | Avg time |
|-----------------------|----------|-------|----------|
| Vector Search         |   ?.?%   |  ?.?? |   ??ms   |
| Full-Text Search      |   ?.?%   |  ?.?? |   ??ms   |
| Hybrid + Rerank       |  ≥ 75%   | ≥ 0.60| < 2000ms |
| Routing Accuracy      |  ≥ 90%   |   —   |  < 20ms  |
