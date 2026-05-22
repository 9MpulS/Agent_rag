"""Partial reseed: re-parse and re-embed a single document by doc_id.

Usage:
    uv run python scripts/reseed_doc.py --doc-id 97

This script:
1. Deletes all chunks and pages for the specified document.
2. Re-parses the PDF using the LangChain-based parser (with OCR fallback).
3. Re-embeds and inserts fresh chunks, each linked to its correct page_id.

Unlike seed_db.py (which truncates everything), this only touches one document.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog
from sqlalchemy import delete, select

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_rag.db.engine import async_session_factory
from agent_rag.db.models import Chunk, Document, Page
from agent_rag.embeddings import get_embeddings_batch
from pdf_parser import chunk_markdown, parse_pdf_to_markdown

logger = structlog.get_logger()


async def _get_or_create_page(
    session,
    doc_id: int,
    page_number: int,
    raw_text: str,
) -> int:
    """Insert a new Page row and return its id."""
    from sqlalchemy import insert

    from agent_rag.db.models import Page

    result = await session.execute(
        insert(Page)
        .values(document_id=doc_id, page_number=page_number, raw_text=raw_text)
        .returning(Page.id)
    )
    return result.scalar_one()


async def reseed_document(doc_id: int) -> None:
    """Delete and re-seed a single document."""
    async with async_session_factory() as session:
        # 1. Fetch document record
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            logger.error("document_not_found", doc_id=doc_id)
            return

        logger.info("reseeding_document", doc_id=doc_id, title=doc.title[:70])

        # 2. Locate the PDF on disk (search recursively by filename stem)
        pdf_root = Path("pdf_documents")
        matches = list(pdf_root.rglob(f"{doc.title}.pdf"))
        if not matches:
            logger.error("pdf_not_found", title=doc.title, root=str(pdf_root))
            return
        pdf_path = matches[0]
        logger.info("found_pdf", path=str(pdf_path))

        # 3. Delete existing chunks and pages for this document
        page_ids_result = await session.execute(
            select(Page.id).where(Page.document_id == doc_id)
        )
        page_ids = [row[0] for row in page_ids_result.all()]
        if page_ids:
            await session.execute(delete(Chunk).where(Chunk.page_id.in_(page_ids)))
            await session.execute(delete(Page).where(Page.id.in_(page_ids)))
            logger.info("deleted_old_data", pages=len(page_ids))
        await session.flush()

        # 4. Parse PDF → {page_number: markdown_text}
        logger.info("parsing_pdf", path=pdf_path.name)
        pages_md = parse_pdf_to_markdown(pdf_path)
        logger.info("pages_parsed", count=len(pages_md))

        # 5. Create Page rows and build page_number → db_page_id map
        page_id_map: dict[int, int] = {}
        for page_number, raw_text in sorted(pages_md.items()):
            db_page_id = await _get_or_create_page(
                session, doc_id, page_number, raw_text
            )
            page_id_map[page_number] = db_page_id

        # 6. Chunk the markdown
        chunks = chunk_markdown(pages_md, chunk_size=750, chunk_overlap=100)
        logger.info("chunks_created", count=len(chunks))

        if not chunks:
            logger.warning("no_chunks", pdf=pdf_path.name)
            await session.commit()
            return

        # 7. Generate embeddings in one batch
        texts = [c["content"] for c in chunks]
        logger.info("generating_embeddings", count=len(texts))
        embeddings = await get_embeddings_batch(texts)

        # 8. Build and insert Chunk rows
        first_page_id = list(page_id_map.values())[0]
        chunk_rows = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            pg_num = chunk["page_number"]
            db_page_id = page_id_map.get(pg_num, first_page_id)
            chunk_rows.append(
                {
                    "page_id": db_page_id,
                    "registry_section_id": doc.registry_section_id,
                    "content": chunk["content"],
                    "embedding": embedding,
                    "chunk_index": idx,
                }
            )

        from agent_rag.db.repositories import create_chunks_batch

        await create_chunks_batch(session, chunk_rows)
        await session.commit()

        logger.info(
            "reseed_complete",
            doc_id=doc_id,
            pages=len(page_id_map),
            chunks=len(chunk_rows),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-seed a single document by doc_id using the LangChain parser"
    )
    parser.add_argument(
        "--doc-id", type=int, required=True, help="Document DB ID to reseed"
    )
    args = parser.parse_args()
    await reseed_document(args.doc_id)


if __name__ == "__main__":
    asyncio.run(main())
