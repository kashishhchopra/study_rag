"""
Document chunking utilities.

Splits raw document text into overlapping chunks so that each chunk is small
enough to embed meaningfully and large enough to retain context. Splitting is
paragraph-aware where possible, falling back to a hard character-window split
for very long paragraphs.
"""

from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    """Hard character-window split for a single paragraph longer than chunk_size."""
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(paragraph):
        end = start + chunk_size
        chunks.append(paragraph[start:end])
        if end >= len(paragraph):
            break
        start += step
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split `text` into overlapping chunks.

    Strategy: group whole paragraphs together until adding the next paragraph
    would exceed chunk_size, then start a new chunk that overlaps with the
    tail of the previous one (by re-including the last `overlap` characters).
    Paragraphs themselves longer than chunk_size are hard-split.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_paragraph(para, chunk_size, overlap))
            continue

        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            # start new chunk, seeded with overlap tail of the previous chunk
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + "\n\n" + para).strip()

    if current:
        chunks.append(current.strip())

    return chunks
