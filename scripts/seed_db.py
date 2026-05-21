"""Seed the database with real PDF documents from pdf_documents/ folder."""

import asyncio
import re
from pathlib import Path

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.db.engine import async_session_factory
from agent_rag.db.models import DocType
from agent_rag.db.repositories import (
    create_chunks_batch,
    create_document,
    create_page,
    create_registry_section,
)
from agent_rag.embeddings import get_embedding, get_embeddings_batch

# pyrefly: ignore [missing-import]
from pdf_parser import chunk_elements, extract_pages_text, parse_pdf

logger = structlog.get_logger()

PDF_ROOT = Path("pdf_documents")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def guess_doc_type(filename: str) -> DocType:
    """Guess document type from filename using Ukrainian keywords.

    Args:
        filename: Name of the PDF file (e.g. "Положення про Наглядову раду.pdf").

    Returns:
        Matching DocType enum value.
    """
    name_lower = filename.lower()
    if "положення" in name_lower:
        return DocType.regulation
    if "наказ" in name_lower:
        return DocType.order
    if "інструкція" in name_lower or "порядок" in name_lower:
        return DocType.instruction
    if "методика" in name_lower or "пам'ятка" in name_lower:
        return DocType.manual
    return DocType.regulation


def build_section_description(section_name: str, pdf_files: list[Path]) -> str:
    """Build a section description from its name and contained PDF filenames.

    Example output:
        Підрозділ «Академічна доброчесність» містить нормативні документи:
        Кодекс академічної доброчесності; Положення про групу сприяння...

    Args:
        section_name: Cleaned section name without numeric prefix.
        pdf_files: Sorted list of PDF file paths in the section folder.

    Returns:
        Description string for the RegistrySection.description field.
    """
    doc_names = [f.stem for f in sorted(pdf_files)]
    docs_list = "; ".join(doc_names)
    return (
        f"Підрозділ «{section_name}» містить нормативні документи: "
        f"{docs_list}."
    )


def clean_section_name(folder_name: str) -> str:
    """Remove leading numbering from folder name.

    '1.1. Загальні питання' -> 'Загальні питання'
    '2.2. Академічна доброчесність' -> 'Академічна доброчесність'

    Args:
        folder_name: Original folder name with numeric prefix.

    Returns:
        Cleaned section name.
    """
    return re.sub(r"^\d+\.\d+\.\s*", "", folder_name).strip()


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


async def truncate_all(session: AsyncSession) -> None:
    """Truncate all data tables before re-seeding.

    Uses CASCADE to handle foreign key constraints automatically.
    Schema (tables, indexes, triggers) is preserved.
    """
    logger.warning("truncating_all_tables")
    await session.execute(
        text("TRUNCATE chunks, pages, documents, registry_sections CASCADE")
    )
    await session.commit()


async def seed(session: AsyncSession) -> None:
    """Seed the database from real PDF documents in pdf_documents/."""
    logger.info("seed_started", pdf_root=str(PDF_ROOT))

    # Discover section folders sorted alphabetically
    section_folders = sorted(
        [d for d in PDF_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not section_folders:
        logger.error("no_section_folders_found", path=str(PDF_ROOT))
        return

    total_docs = 0
    total_chunks = 0

    for folder in section_folders:
        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning("empty_section_folder", folder=folder.name)
            continue

        # --- 1. Create RegistrySection ---
        section_name = clean_section_name(folder.name)
        description = build_section_description(section_name, pdf_files)
        description_embedding = await get_embedding(description)

        section = await create_registry_section(
            session,
            name=section_name,
            description=description,
            description_embedding=description_embedding,
        )
        logger.info(
            "section_created",
            section_id=section.id,
            name=section_name,
            pdf_count=len(pdf_files),
        )

        # --- 2. Process each PDF in the section ---
        for pdf_path in pdf_files:
            try:
                doc_type = guess_doc_type(pdf_path.name)
                doc_title = pdf_path.stem  # Filename without .pdf extension

                # Create Document
                doc = await create_document(
                    session,
                    registry_section_id=section.id,
                    title=doc_title,
                    doc_type=doc_type,
                )
                total_docs += 1

                # Extract page-level text and create Page records
                pages_text = extract_pages_text(pdf_path)
                page_ids: list[int] = []
                for page_num, raw_text in pages_text.items():
                    page = await create_page(
                        session, doc.id, page_number=page_num, raw_text=raw_text
                    )
                    page_ids.append(page.id)

                # Parse PDF into elements and chunk semantically
                elements = parse_pdf(pdf_path)
                text_chunks = chunk_elements(
                    elements, max_characters=500, overlap=50
                )

                if not text_chunks:
                    logger.warning("no_chunks_produced", pdf=pdf_path.name)
                    continue

                # Batch-generate embeddings via Ollama
                chunk_embeddings = await get_embeddings_batch(text_chunks)

                # Build chunk data dicts for batch insert
                # Simplified page mapping: assign all chunks to the first page
                # (chunks may span multiple pages due to title-based chunking)
                first_page_id = page_ids[0] if page_ids else 1
                chunks_data = [
                    {
                        "page_id": first_page_id,
                        "registry_section_id": section.id,
                        "content": content,
                        "embedding": embedding,
                        "chunk_index": idx,
                    }
                    for idx, (content, embedding) in enumerate(
                        zip(text_chunks, chunk_embeddings)
                    )
                ]

                await create_chunks_batch(session, chunks_data)
                total_chunks += len(chunks_data)

                logger.info(
                    "document_seeded",
                    doc_id=doc.id,
                    title=doc_title[:50],
                    pages=len(pages_text),
                    chunks=len(chunks_data),
                )

            except Exception as exc:
                logger.error(
                    "document_seed_error",
                    pdf=pdf_path.name,
                    error=str(exc),
                )
                continue

    await session.commit()
    logger.info(
        "seed_completed",
        sections=len(section_folders),
        documents=total_docs,
        chunks=total_chunks,
    )


async def main() -> None:
    """Entry point for seed script."""
    async with async_session_factory() as session:
        await truncate_all(session)
        await seed(session)


if __name__ == "__main__":
    asyncio.run(main())
