"""API routes for Agentic RAG."""

import os
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.api.schemas import QueryRequest, QueryResponse
from agent_rag.agent.graph import graph
from agent_rag.agent.state import AgentState
from agent_rag.db.engine import get_session
from agent_rag.db.models import DocType
from agent_rag.db.repositories import (
    create_chunks_batch,
    create_document,
    create_page,
    create_registry_section,
    delete_document,
    get_section_by_name,
)
from agent_rag.embeddings import get_embedding, get_embeddings_batch
from agent_rag.llm.groq_client import GroqClient

# Dynamically import pdf_parser from scripts/
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir / "scripts") not in sys.path:
    sys.path.append(str(root_dir / "scripts"))
try:
    from pdf_parser import chunk_elements, extract_pages_text, parse_pdf
except ImportError:
    pass

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, session: AsyncSession = Depends(get_session)):
    """Run a query against the RAG agent."""
    groq_client = GroqClient()

    initial_state: AgentState = {
        "query": request.query,
        "session": session,
        "groq_client": groq_client,
        "messages": [{"role": "user", "content": request.query}],
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

    return QueryResponse(
        answer=result_state.get("answer", "Немає відповіді"),
        sources=result_state.get("sources", []),
        iteration=result_state.get("iteration", 0),
        timings=result_state.get("timings", {})
    )

@router.post("/documents")
async def add_document(
    section_name: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session)
):
    """Add a new document to the database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Check if section exists
    section = await get_section_by_name(session, section_name)
    if not section:
        # Create new section
        description = f"Підрозділ «{section_name}» містить нормативні документи: {file.filename}."
        description_embedding = await get_embedding(description)
        section = await create_registry_section(
            session,
            name=section_name,
            description=description,
            description_embedding=description_embedding
        )

    # Save file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse and process
        doc_title = temp_path.stem
        name_lower = file.filename.lower()
        if "положення" in name_lower:
            doc_type = DocType.regulation
        elif "наказ" in name_lower:
            doc_type = DocType.order
        elif "інструкція" in name_lower or "порядок" in name_lower:
            doc_type = DocType.instruction
        elif "методика" in name_lower or "пам'ятка" in name_lower:
            doc_type = DocType.manual
        else:
            doc_type = DocType.regulation

        doc = await create_document(
            session,
            registry_section_id=section.id,
            title=doc_title,
            doc_type=doc_type,
        )

        pages_text = extract_pages_text(temp_path)
        page_ids: list[int] = []
        for page_num, raw_text in pages_text.items():
            page = await create_page(session, doc.id, page_number=page_num, raw_text=raw_text)
            page_ids.append(page.id)

        elements = parse_pdf(temp_path)
        text_chunks = chunk_elements(elements, max_characters=750, overlap=50)

        if not text_chunks:
            await session.rollback()
            raise HTTPException(status_code=400, detail="Could not extract text chunks from PDF.")

        chunk_embeddings = await get_embeddings_batch(text_chunks)

        first_page_id = page_ids[0] if page_ids else 1
        chunks_data = [
            {
                "page_id": first_page_id,
                "registry_section_id": section.id,
                "content": content,
                "embedding": embedding,
                "chunk_index": idx,
            }
            for idx, (content, embedding) in enumerate(zip(text_chunks, chunk_embeddings))
        ]

        await create_chunks_batch(session, chunks_data)
        await session.commit()

        return {"message": "Document uploaded and processed successfully", "document_id": doc.id}
    finally:
        if temp_path.exists():
            temp_path.unlink()

@router.delete("/documents/{document_id}")
async def delete_doc(document_id: int, session: AsyncSession = Depends(get_session)):
    """Delete a document by ID."""
    success = await delete_document(session, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.commit()
    return {"message": "Document deleted successfully"}
