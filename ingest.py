"""
CLI: ingest a PDF into the Chroma vector store.

Usage (from project root):
  python ingest.py --pdf path/to/book.pdf --collection mybook
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python ingest.py` without installing the package as editable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.chunking import chunk_marked_document, chunk_metadata
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    require_api_key,
)
from src.embeddings import (
    EmbeddingBackendError,
    GeminiQuotaError,
    embed_texts,
    embedding_backend,
    embeddings_require_api_key,
)
from src.pdf_extract import extract_pages, pages_to_marked_document
from src.vector_store import BookVectorStore


def main() -> None:
    try:
        if embeddings_require_api_key():
            require_api_key()
    except EmbeddingBackendError as e:
        raise SystemExit(str(e)) from e

    p = argparse.ArgumentParser(description="Ingest a PDF into Chroma for RAG")
    p.add_argument("--pdf", required=True, help="Path to a .pdf file")
    p.add_argument("--collection", default="default_book", help="Chroma collection name")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="If >0, only embed the first N chunks (quota saver).",
    )
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to collection instead of clearing it first (may fail if chunk IDs collide)",
    )
    args = p.parse_args()

    pages = extract_pages(args.pdf)
    doc = pages_to_marked_document(pages)
    if not doc.strip():
        raise SystemExit("No text extracted from PDF (scanned book? try OCR tools).")

    chunks = chunk_marked_document(
        doc,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        id_prefix=args.collection,
    )
    if not chunks:
        raise SystemExit("Chunking produced zero chunks.")
    if args.max_chunks > 0 and len(chunks) > args.max_chunks:
        print(f"Quota saver: using first {args.max_chunks} of {len(chunks)} chunks.")
        chunks = chunks[: args.max_chunks]

    print(f"Chunks: {len(chunks)} — embedding with backend '{embedding_backend()}'...")
    texts = [c.text for c in chunks]
    try:
        embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    except GeminiQuotaError as e:
        raise SystemExit(str(e)) from e
    except EmbeddingBackendError as e:
        raise SystemExit(str(e)) from e

    store = BookVectorStore(args.collection)
    if not args.no_reset:
        store.reset()

    store.add_chunks(
        ids=[c.chunk_id for c in chunks],
        texts=texts,
        embeddings=embeddings,
        metadatas=[chunk_metadata(c) for c in chunks],
    )
    print(f"Done. Stored {store.count()} vectors in collection '{args.collection}'.")


if __name__ == "__main__":
    main()
