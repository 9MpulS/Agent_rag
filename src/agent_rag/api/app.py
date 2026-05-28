"""FastAPI application initialization."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_rag.api.routes import router

app = FastAPI(
    title="Agentic RAG API",
    description="API for querying, adding, and deleting documents in the RAG system.",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
