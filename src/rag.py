"""Retrieve relevant chunks, then call Gemini twice: book-grounded + supplemental."""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import DEFAULT_TOP_K, GEMINI_CHAT_MODEL, require_api_key
from src.embeddings import GeminiQuotaError, embed_query
from src.vector_store import BookVectorStore, RetrievedChunk


BOOK_SYSTEM = """You are a careful assistant answering questions about a supplied book (CONTEXT).
Rules:
- Treat CONTEXT as the only source of factual claims about what the book says.
- If CONTEXT does not contain enough information, say so clearly and answer only what you can support.
- Quote or paraphrase closely when possible; do not invent page numbers.
- Write clearly under a heading: "From the book".
"""


SUPPLEMENT_SYSTEM = """You complement a book-grounded answer with general knowledge.
Rules:
- You will receive the USER QUESTION and the BOOK ANSWER already derived from the book.
- Under the heading "Beyond the book", add concise context: definitions, background, related ideas, or caveats.
- Do not contradict the BOOK ANSWER when it is specific to the book; if your general knowledge differs, note that the book may take a specific stance.
- If nothing useful can be added, say so briefly."""


def _client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


def _raise_if_quota_error(exc: Exception) -> None:
    if not isinstance(exc, genai_errors.ClientError):
        return
    code = getattr(exc, "status_code", None)
    text = str(exc).upper()
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        raise GeminiQuotaError(
            "Gemini API quota exhausted (HTTP 429 RESOURCE_EXHAUSTED). "
            "Try again later or check plan/billing limits."
        ) from exc


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        page = c.metadata.get("page_hint")
        loc = f" (approx. page {page})" if page else ""
        parts.append(f"[Passage {i}{loc}]\n{c.text}")
    return "\n\n".join(parts)


def _retrieval_only_fallback(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "From the book\n\n"
            "Gemini is currently unavailable and no passages were retrieved from the book."
        )
    lines = [
        "From the book (retrieval-only fallback)",
        "",
        "Gemini is currently unavailable, so here are the most relevant passages directly from the book:",
        "",
    ]
    for i, c in enumerate(chunks[:3], start=1):
        page = c.metadata.get("page_hint")
        loc = f" (approx. page {page})" if page else ""
        preview = c.text[:900] + ("..." if len(c.text) > 900 else "")
        lines.append(f"{i}) Passage{loc}:\n{preview}\n")
    return "\n".join(lines)


def answer_book_only(*, question: str, context: str) -> str:
    model = _client()
    prompt = f"""CONTEXT (excerpts from the book):
{context}

USER QUESTION:
{question}
"""
    try:
        resp = model.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=BOOK_SYSTEM),
        )
    except Exception as exc:
        _raise_if_quota_error(exc)
        raise
    return (resp.text or "").strip()


def answer_supplement(*, question: str, book_answer: str) -> str:
    model = _client()
    prompt = f"""USER QUESTION:
{question}

BOOK ANSWER:
{book_answer}
"""
    try:
        resp = model.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SUPPLEMENT_SYSTEM),
        )
    except Exception as exc:
        _raise_if_quota_error(exc)
        raise
    return (resp.text or "").strip()


def run_rag_turn(
    *,
    store: BookVectorStore,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, list[RetrievedChunk]]:
    """
    Full RAG turn: embed question → retrieve → Gemini (book) → Gemini (supplement).

    Returns the composed reply and the retrieved chunks (for UI/debug).
    """
    q_emb = embed_query(question)
    chunks = store.query(q_emb, k=top_k)
    context = format_context(chunks)
    try:
        book = answer_book_only(question=question, context=context)
    except GeminiQuotaError:
        # Hard fallback when Gemini is unavailable even for the first call.
        return _retrieval_only_fallback(chunks), chunks

    try:
        extra = answer_supplement(question=question, book_answer=book)
    except GeminiQuotaError:
        extra = (
            "Beyond the book\n\n"
            "Additional Gemini context is temporarily unavailable due to quota limits."
        )

    reply = f"{book}\n\n---\n\n{extra}"
    return reply, chunks
