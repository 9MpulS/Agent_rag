"""PDF document parser using the unstructured library.

This module provides functions for:
- Parsing PDF files into structured elements (titles, paragraphs, lists, tables)
- Chunking elements by title boundaries for RAG embedding
- Extracting raw page-level text for database storage
"""

from pathlib import Path

import structlog
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element,
    ListItem,
    NarrativeText,
    Table,
    Title,
)
from unstructured.partition.pdf import partition_pdf

logger = structlog.get_logger()

# Element types that carry meaningful content for RAG retrieval.
# We exclude Headers, Footers, PageBreaks, FigureCaption, etc.
_KEEP_TYPES = (NarrativeText, Title, ListItem, Table)


def parse_pdf(path: Path) -> list[Element]:
    """Parse a PDF file and return filtered structural elements.

    Uses unstructured's partition_pdf with "fast" strategy (no OCR, no
    deep-learning layout models) — suitable for digitally-created PDFs
    of SumDU normative documents.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of Element objects filtered to NarrativeText, Title,
        ListItem, and Table types only.

    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
        ValueError: If unstructured fails to parse the PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    size_mb = round(path.stat().st_size / 1024 / 1024, 2)
    logger.info("parsing_pdf", path=str(path), size_mb=size_mb)

    try:
        elements = partition_pdf(
            filename=str(path),
            strategy="fast",
            languages=["ukr", "eng"],
        )
    except Exception as exc:
        logger.error("pdf_parse_error", path=str(path), error=str(exc))
        raise ValueError(f"Failed to parse {path.name}: {exc}") from exc

    filtered = [el for el in elements if isinstance(el, _KEEP_TYPES)]

    logger.info(
        "pdf_parsed",
        path=path.name,
        total_elements=len(elements),
        kept_elements=len(filtered),
    )
    return filtered


def chunk_elements(
    elements: list[Element],
    max_characters: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Chunk parsed elements using title-based semantic chunking.

    How chunk_by_title works:
    1. Each Title element acts as a hard boundary — starts a new chunk.
    2. Consecutive NarrativeText/ListItem elements are merged until
       they hit max_characters.
    3. If a single element exceeds max_characters, it is split with
       overlap to preserve context.
    4. Elements shorter than combine_text_under_n_chars are merged
       with their neighbors.

    Args:
        elements: List of Element objects from parse_pdf().
        max_characters: Maximum character count per chunk (default: 500).
        overlap: Character overlap between split chunks (default: 50).

    Returns:
        List of non-empty text strings ready for embedding.
    """
    if not elements:
        return []

    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        overlap=overlap,
        combine_text_under_n_chars=100,
    )

    texts = [str(chunk) for chunk in chunks if str(chunk).strip()]

    avg_len = round(sum(len(t) for t in texts) / max(len(texts), 1))
    logger.info(
        "elements_chunked",
        input_elements=len(elements),
        output_chunks=len(texts),
        avg_chunk_len=avg_len,
    )
    return texts


def extract_pages_text(path: Path) -> dict[int, str]:
    """Extract raw text from a PDF file grouped by page number.

    Each page's text is a concatenation of all element texts on that page,
    joined by newlines. Used to populate the `pages` table with raw text.

    Args:
        path: Path to the PDF file.

    Returns:
        Dict mapping 1-indexed page_number to concatenated raw text.

    Raises:
        ValueError: If unstructured fails to parse the PDF.
    """
    try:
        elements = partition_pdf(
            filename=str(path),
            strategy="fast",
            languages=["ukr", "eng"],
        )
    except Exception as exc:
        logger.error("pdf_extract_pages_error", path=str(path), error=str(exc))
        raise ValueError(f"Failed to extract pages from {path.name}: {exc}") from exc

    pages: dict[int, list[str]] = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        pages.setdefault(page_num, []).append(str(el))

    return {
        page_num: "\n".join(texts)
        for page_num, texts in sorted(pages.items())
    }
