"""PDF text extraction and chunking using PyMuPDF.

The extractor preserves per-page provenance so that every chunk can be
traced back to its source document and page number. This provenance is
what makes chunk-level source attribution possible downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    text: str
    source: str          # original file name
    page: int            # 1-indexed page number
    chunk_id: int        # running index within the document
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def citation(self) -> str:
        """Human-readable citation label, e.g. 'report.pdf p.4'."""
        return f"{self.source} p.{self.page}"


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip common PDF extraction artefacts."""
    # Join words split across line breaks with a hyphen: "exam-\nple" -> "example"
    text = re.sub(r"-\n", "", text)
    # Collapse newlines/tabs into spaces, then squash repeated spaces.
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_pages(file_bytes: bytes, source_name: str) -> List[tuple[int, str]]:
    """Extract cleaned text from each page of a PDF.

    Args:
        file_bytes: Raw bytes of the PDF file.
        source_name: Display name used for citations (usually the filename).

    Returns:
        A list of (page_number, page_text) tuples for pages that contain text.
    """
    import fitz  # PyMuPDF (lazy import keeps the heavy dep optional for tests)

    pages: List[tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            raw = page.get_text("text")
            cleaned = _clean_text(raw)
            if cleaned:
                pages.append((page_index + 1, cleaned))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """Split text into overlapping, word-boundary-aware character windows.

    Overlap preserves context that would otherwise be severed at a hard cut,
    which improves retrieval quality for facts that straddle a boundary.

    Args:
        text: The text to split.
        chunk_size: Target maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for the joining space.
        if current_len + len(word) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Build the overlap tail from the end of the current chunk.
            tail: List[str] = []
            tail_len = 0
            for w in reversed(current):
                if tail_len + len(w) + 1 > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            current = tail
            current_len = tail_len
        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def process_pdf(
    file_bytes: bytes,
    source_name: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[Chunk]:
    """Full pipeline: extract a PDF's text and return provenance-tagged chunks.

    Args:
        file_bytes: Raw PDF bytes.
        source_name: Filename used for citation labels.
        chunk_size: Target max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        A list of :class:`Chunk` objects ready for embedding.
    """
    chunks: List[Chunk] = []
    chunk_id = 0
    for page_number, page_text in extract_pages(file_bytes, source_name):
        for piece in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source_name,
                    page=page_number,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1
    return chunks
