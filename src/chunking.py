"""Split long documents into overlapping chunks for embedding and retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """One segment of the corpus stored in the vector DB."""

    chunk_id: str
    text: str
    start_char: int
    end_char: int


def _infer_page_hint(chunk_text: str) -> str | None:
    """If this chunk contains a '--- Page N ---' marker, surface N for metadata."""
    m = re.search(r"---\s*Page\s+(\d+)\s*---", chunk_text)
    return m.group(1) if m else None


def chunk_marked_document(
    document: str,
    *,
    chunk_size: int,
    overlap: int,
    id_prefix: str = "chunk",
) -> list[TextChunk]:
    """
    Character-based sliding window with overlap.

    Why overlap: a fact split across two windows can still appear whole in one chunk.
    Trade-off: more chunks → more storage/API calls at ingest time.

    chunk_size/overlap are in characters (rough proxy for tokens; good enough for a demo).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    document = document.strip()
    if not document:
        return []

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        piece = document[start:end].strip()
        if piece:
            cid = f"{id_prefix}-{idx:05d}"
            chunks.append(TextChunk(chunk_id=cid, text=piece, start_char=start, end_char=end))
            idx += 1
        if end >= len(document):
            break
        start = end - overlap

    return chunks


def chunk_metadata(chunk: TextChunk) -> dict[str, str]:
    """Optional metadata stored alongside vectors in Chroma."""
    hint = _infer_page_hint(chunk.text)
    meta: dict[str, str] = {
        "start_char": str(chunk.start_char),
        "end_char": str(chunk.end_char),
    }
    if hint:
        meta["page_hint"] = hint
    return meta
