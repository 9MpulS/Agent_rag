"""Pydantic schemas for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query for the RAG agent.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer.")
    sources: list[str] = Field(default_factory=list, description="List of source document names/URLs.")
    iteration: int = Field(..., description="Number of agent iterations taken.")
    timings: dict[str, float] | None = Field(None, description="Timing for internal steps.")
