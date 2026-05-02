"""Embedding backends for retrieval: Gemini API or local sentence-transformers."""

from __future__ import annotations

import time
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import EMBEDDING_BACKEND, GEMINI_EMBED_MODEL, LOCAL_EMBED_MODEL, require_api_key

# Task types for gemini-embedding-001 (see Gemini embeddings docs).
EmbedTask = Literal["RETRIEVAL_DOCUMENT", "QUESTION_ANSWERING"]


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini rejects requests due to exhausted quota."""


class EmbeddingBackendError(RuntimeError):
    """Raised when embedding backend is misconfigured or unavailable."""


def _is_quota_error(exc: Exception) -> bool:
    if not isinstance(exc, genai_errors.ClientError):
        return False
    # `status_code` is reliable; message fallback helps when wrappers change.
    code = getattr(exc, "status_code", None)
    text = str(exc).upper()
    return code == 429 or "RESOURCE_EXHAUSTED" in text


def _client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


def embedding_backend() -> str:
    if EMBEDDING_BACKEND not in {"gemini", "local"}:
        raise EmbeddingBackendError(
            f"Unsupported EMBEDDING_BACKEND='{EMBEDDING_BACKEND}'. Use 'gemini' or 'local'."
        )
    return EMBEDDING_BACKEND


def embeddings_require_api_key() -> bool:
    return embedding_backend() == "gemini"


_LOCAL_MODEL = None


def _local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingBackendError(
                "Local embedding backend selected but sentence-transformers is unavailable. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        _LOCAL_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _LOCAL_MODEL


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # Convert numpy arrays to plain lists for Chroma compatibility.
    return [v.tolist() for v in vectors]


def _embed_texts_gemini(
    texts: list[str],
    *,
    task_type: EmbedTask,
    batch_size: int,
    sleep_s: float,
) -> list[list[float]]:
    """
    Gemini embedding path.

    The API accepts a list in one call; we still batch to limit payload size.
    """
    client = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            result = client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except Exception as exc:
            if _is_quota_error(exc):
                raise GeminiQuotaError(
                    "Gemini API quota exhausted (HTTP 429 RESOURCE_EXHAUSTED). "
                    "Switch EMBEDDING_BACKEND=local, wait for reset, or ingest fewer chunks."
                ) from exc
            raise
        embs = result.embeddings or []
        for e in embs:
            out.append(list(e.values))
        if sleep_s:
            time.sleep(sleep_s)
    return out


def embed_texts(
    texts: list[str],
    *,
    task_type: EmbedTask,
    batch_size: int = 16,
    sleep_s: float = 0.1,
) -> list[list[float]]:
    """
    Embed many strings using configured backend (local or Gemini).
    """
    if not texts:
        return []

    backend = embedding_backend()
    if backend == "local":
        # task_type is ignored in local mode because one local encoder handles both sides.
        return _embed_texts_local(texts)
    return _embed_texts_gemini(texts, task_type=task_type, batch_size=batch_size, sleep_s=sleep_s)


def embed_query(question: str) -> list[float]:
    vecs = embed_texts([question], task_type="QUESTION_ANSWERING", batch_size=1)
    return vecs[0]
