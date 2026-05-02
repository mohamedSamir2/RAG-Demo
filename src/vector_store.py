"""Chroma persistent vector store: add chunks + query by embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_DIR


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    distance: float | None
    metadata: dict[str, Any]


class BookVectorStore:
    """
    Thin wrapper around Chroma so the rest of the code stays readable.

    distance: lower is closer for cosine space in Chroma's default interpretation;
    exact meaning depends on Chroma's configured space (we use cosine).
    """

    def __init__(self, collection_name: str, *, persist_root: Path | None = None) -> None:
        root = Path(persist_root) if persist_root is not None else CHROMA_DIR
        root.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(root))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete all vectors in this collection (re-ingest after this)."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        *,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not (len(ids) == len(texts) == len(embeddings)):
            raise ValueError("ids, texts, embeddings must have the same length")
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for i, cid in enumerate(ids):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=docs[i] or "",
                    distance=float(dists[i]) if dists and i < len(dists) else None,
                    metadata=dict(metas[i] or {}),
                )
            )
        return retrieved

    def count(self) -> int:
        return int(self._collection.count())
