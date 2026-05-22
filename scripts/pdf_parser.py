# -*- coding: utf-8 -*-
"""PDF parser using LangChain + PyMuPDF with OCR fallback via Tesseract.

Pipeline:
  1. ``pymupdf4llm`` renders each page to Markdown (tables, headings, lists).
  2. If the extracted text looks garbled (non-standard font encoding —
     common in Ukrainian PDFs), fall back to OCR:
     - PyMuPDF renders each page to a high-resolution image (300 DPI).
     - ``pytesseract`` runs Tesseract with ``ukr+eng`` language on the image.
  3. LangChain splitters chunk the resulting Markdown into overlapping chunks
     for vector embedding.

Usage from other scripts::

    from pdf_parser import parse_pdf_to_markdown, chunk_markdown, extract_pages_text

Requires:
    uv add pymupdf4llm langchain-community langchain-text-splitters pytesseract
    tessdata/ukr.traineddata  (project-local Tesseract Ukrainian model)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Tesseract / tessdata setup
# ---------------------------------------------------------------------------

def _setup_tessdata() -> str:
    """Return the tessdata directory and ensure TESSDATA_PREFIX is set.

    Prefers the project-local ``./tessdata/`` folder so no admin rights are
    needed to install language packs.
    """
    local = Path("tessdata").resolve()
    if local.is_dir() and (local / "ukr.traineddata").exists():
        tessdata = str(local)
    else:
        tessdata = os.environ.get("TESSDATA_PREFIX", "")

    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata
    return tessdata


# Run once at module import so subsequent pytesseract calls pick it up.
_TESSDATA_DIR = _setup_tessdata()

# ---------------------------------------------------------------------------
# Garbled-text detection
# ---------------------------------------------------------------------------

_GARBLED_RE = re.compile(r"[\ufffd\uf000-\uf8ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _is_garbled(text: str) -> bool:
    """Detect if extracted text is garbled (broken font mapping).

    Checks two conditions:
    1. If the text has a high percentage of words composed entirely of '?' or '\ufffd' (Unicode replacement char).
    2. If the text lacks standard Cyrillic letters (for Ukrainian documents).

    Returns:
        True if text appears garbled and needs OCR.
    """
    if not text.strip():
        return True

    # 1. Check for garbage tokens like "?", "???", or "\ufffd\ufffd"
    words = text.split()
    if not words:
        return True

    bad_words = 0
    for w in words:
        clean_w = w.strip(".,;")
        if not clean_w:
            continue
        # A word is bad if it's entirely '?' or entirely '\ufffd'
        is_q = (clean_w == "?" * len(clean_w))
        is_ufffd = (clean_w == "\ufffd" * len(clean_w))
        if is_q or is_ufffd:
            bad_words += 1

    bad_ratio = bad_words / len(words)
    if bad_ratio > 0.05:  # more than 5% of words are just '?' or '\ufffd'
        return True

    # 2. Check for Cyrillic density
    # A Ukrainian document should have a reasonable amount of Cyrillic chars.
    cyrillic_chars = sum(1 for c in text if "\u0400" <= c <= "\u04FF" or c in "іІїЇєЄґҐ")
    alpha_chars = sum(1 for c in text if c.isalpha())

    # If there are alphabet characters, but less than 10% are Cyrillic,
    # it's likely a broken mapping that turned Cyrillic into Latin/symbols.
    if alpha_chars > 50 and (cyrillic_chars / alpha_chars) < 0.1:
        return True

    return False


# ---------------------------------------------------------------------------
# OCR backend  (PyMuPDF render → Tesseract)
# ---------------------------------------------------------------------------

def _ocr_page_text(page) -> str:  # page: fitz.Page
    """Render *page* at 300 DPI and return Tesseract OCR text (Ukrainian)."""
    import fitz  # PyMuPDF

    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")

    # Write to temp file for tesseract CLI (most reliable on Windows)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        env = {**os.environ, "TESSDATA_PREFIX": _TESSDATA_DIR}
        proc = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", "ukr+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        if proc.returncode != 0:
            logger.warning("tesseract_error", stderr=proc.stderr[:200])
        return proc.stdout.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _ocr_all_pages(pdf_path: Path) -> dict[int, str]:
    """OCR every page of *pdf_path*, return {page_num: text} (1-indexed)."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages: dict[int, str] = {}
    total = doc.page_count
    logger.info("ocr_start", path=pdf_path.name, total_pages=total)
    for i in range(total):
        text = _ocr_page_text(doc[i])
        if text:
            pages[i + 1] = text
        if (i + 1) % 10 == 0:
            logger.info("ocr_progress", page=i + 1, total=total)
    logger.info("ocr_done", path=pdf_path.name, pages_extracted=len(pages))
    return pages


def _fast_native_pages(pdf_path: Path) -> dict[int, str]:
    """Extract raw text per page using fitz directly (no OCR, ever).

    Returns {page_number: text} (1-indexed).  Uses fitz.get_text('text')
    which only reads embedded font data - no fallback to Tesseract.
    If the PDF has broken font encoding the result will contain replacement
    characters which _is_garbled() will detect.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    pages: dict[int, str] = {}
    for i in range(doc.page_count):
        text = doc[i].get_text("text").strip()
        if text:
            pages[i + 1] = text
    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf_to_markdown(pdf_path: Path) -> dict[int, str]:
    """Parse a PDF and return {page_number: text} (1-indexed).

    Strategy:
    1. Extract raw text via fitz (native, zero OCR).
    2. If garbled - OCR via PyMuPDF render + Tesseract ukr+eng.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping 1-based page numbers to extracted text strings.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If both extraction methods fail.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    size_mb = round(pdf_path.stat().st_size / 1024 / 1024, 2)
    logger.info("parsing_pdf", path=pdf_path.name, size_mb=size_mb)

    # ── Step 1: fitz native text (no OCR whatsoever) ──────────────────────
    try:
        native_pages = _fast_native_pages(pdf_path)
        combined = " ".join(native_pages.values())

        if not _is_garbled(combined):
            # Text is clean! pymupdf4llm keeps maliciously triggering OCR
            # and ruining perfect text, so we return native text directly.
            logger.info("pdf_parsed_native_clean", path=pdf_path.name, pages=len(native_pages))
            return native_pages

        logger.warning(
            "fast_extraction_garbled_switching_to_ocr",
            path=pdf_path.name,
            sample=combined[:80],
        )
    except Exception as exc:
        logger.warning("fast_extraction_failed", path=pdf_path.name, error=str(exc))


    # ── Step 2: OCR via PyMuPDF render + Tesseract ────────────────────────
    try:
        pages = _ocr_all_pages(pdf_path)
        logger.info("pdf_parsed_ocr", path=pdf_path.name, pages=len(pages))
        return pages
    except Exception as exc:
        logger.error("ocr_failed", path=pdf_path.name, error=str(exc))
        raise RuntimeError(f"Failed to parse {pdf_path.name}") from exc


def chunk_markdown(
    pages: dict[int, str],
    chunk_size: int = 750,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Chunk Markdown text using LangChain splitters.

    Uses ``MarkdownHeaderTextSplitter`` to respect heading boundaries, then
    ``RecursiveCharacterTextSplitter`` to enforce *chunk_size*.

    Args:
        pages: Dict from :func:`parse_pdf_to_markdown`.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with keys:
        - ``content`` (str): chunk text
        - ``page_number`` (int): source page (1-indexed)
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Headers to split on (Markdown ATX style)
    header_splits = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=header_splits,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    result: list[dict] = []
    for page_number, text in sorted(pages.items()):
        if not text.strip():
            continue
        try:
            md_docs = md_splitter.split_text(text)
            for doc in md_docs:
                sub_docs = char_splitter.split_text(doc.page_content)
                for sub in sub_docs:
                    if sub.strip():
                        result.append(
                            {"content": sub.strip(), "page_number": page_number}
                        )
        except Exception as exc:
            logger.warning(
                "chunk_error", page=page_number, error=str(exc)
            )
            # Fallback: just split by characters
            for sub in char_splitter.split_text(text):
                if sub.strip():
                    result.append({"content": sub.strip(), "page_number": page_number})

    logger.info("chunks_created", total=len(result))
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility shims (used by seed_db.py and reseed_doc.py)
# ---------------------------------------------------------------------------

def extract_pages_text(path: Path) -> dict[int, str]:
    """Legacy API: extract raw text per page (used by seed_db.py)."""
    return parse_pdf_to_markdown(path)


def parse_pdf(path: Path) -> list:
    """Legacy API: return list of fake Element-like objects for compatibility."""

    class _FakePage:
        """Minimal stand-in for an unstructured Element."""

        def __init__(self, text: str, page_number: int) -> None:
            self._text = text

            class _Meta:
                pass

            self.metadata = _Meta()
            self.metadata.page_number = page_number  # type: ignore[attr-defined]

        def __str__(self) -> str:
            return self._text

    pages = parse_pdf_to_markdown(path)
    return [_FakePage(text, pg) for pg, text in sorted(pages.items()) if text.strip()]


def chunk_elements(elements: list, max_characters: int = 750, overlap: int = 100) -> list[str]:
    # Legacy API: chunk elements into strings (used by seed_db.py).
    # Re-assemble pages dict from _FakePage objects
    pages: dict[int, str] = {}
    for el in elements:
        pg = getattr(getattr(el, "metadata", None), "page_number", 1) or 1
        pages[pg] = pages.get(pg, "") + "\n" + str(el)

    chunks = chunk_markdown(pages, chunk_size=max_characters, chunk_overlap=overlap)
    return [c["content"] for c in chunks]
