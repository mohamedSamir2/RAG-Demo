"""
Interactive chat UI: ingest a PDF and ask questions (book + supplemental answers).

Run from project root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.chunking import chunk_marked_document, chunk_metadata
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
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
from src.rag import run_rag_turn
from src.vector_store import BookVectorStore


st.set_page_config(page_title="Book RAG + Gemini", layout="wide")
st.title("Book RAG demo (PDF + Gemini)")

with st.sidebar:
    st.header("Setup")
    st.markdown("Add `GOOGLE_API_KEY` to a `.env` file in the project root.")
    st.caption(f"Embedding backend: `{embedding_backend()}`")
    collection = st.text_input("Collection name", value="default_book")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, value=DEFAULT_CHUNK_SIZE, step=100)
    overlap = st.number_input("Overlap (chars)", min_value=0, value=DEFAULT_CHUNK_OVERLAP, step=50)
    max_chunks = st.number_input(
        "Max chunks to ingest (quota saver)",
        min_value=10,
        value=200,
        step=10,
        help="If your PDF creates more chunks than this, only the first N are embedded.",
    )
    top_k = st.number_input("Chunks to retrieve (top-k)", min_value=1, value=DEFAULT_TOP_K, step=1)
    reset = st.checkbox("Reset collection on ingest", value=True)

    if st.button("Ingest uploaded PDF"):
        try:
            if embeddings_require_api_key():
                require_api_key()
        except RuntimeError as e:
            st.error(str(e))
        except EmbeddingBackendError as e:
            st.error(str(e))
        else:
            if not uploaded:
                st.warning("Upload a PDF first.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    pages = extract_pages(tmp_path)
                    doc = pages_to_marked_document(pages)
                    if not doc.strip():
                        st.error("No text extracted (image-only PDF?).")
                    else:
                        chunks = chunk_marked_document(
                            doc,
                            chunk_size=int(chunk_size),
                            overlap=int(overlap),
                            id_prefix=collection,
                        )
                        if len(chunks) > int(max_chunks):
                            st.info(
                                f"Quota saver active: using first {int(max_chunks)} of {len(chunks)} chunks."
                            )
                            chunks = chunks[: int(max_chunks)]
                        with st.spinner(f"Embedding {len(chunks)} chunks..."):
                            embeddings = embed_texts(
                                [c.text for c in chunks],
                                task_type="RETRIEVAL_DOCUMENT",
                            )
                        store = BookVectorStore(collection)
                        if reset:
                            store.reset()
                        store.add_chunks(
                            ids=[c.chunk_id for c in chunks],
                            texts=[c.text for c in chunks],
                            embeddings=embeddings,
                            metadatas=[chunk_metadata(c) for c in chunks],
                        )
                        st.success(f"Ingested {store.count()} chunks into '{collection}'.")
                except GeminiQuotaError as e:
                    st.error(str(e))
                    st.warning(
                        "Try smaller chunk count/size, then retry; or check your Gemini usage dashboard."
                    )
                except EmbeddingBackendError as e:
                    st.error(str(e))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about your book...")
if prompt:
    try:
        require_api_key()
    except RuntimeError as e:
        st.error(str(e))
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply: str | None = None
        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating..."):
                store = BookVectorStore(collection)
                if store.count() == 0:
                    reply = "No vectors in this collection. Ingest a PDF in the sidebar."
                    st.error(reply)
                else:
                    try:
                        reply, chunks = run_rag_turn(store=store, question=prompt, top_k=int(top_k))
                        st.markdown(reply)
                        with st.expander("Retrieved passages (debug)"):
                            for i, c in enumerate(chunks, 1):
                                st.caption(f"{i}. id={c.chunk_id} distance={c.distance}")
                                st.text(c.text[:1200] + ("…" if len(c.text) > 1200 else ""))
                    except GeminiQuotaError as e:
                        reply = str(e)
                        st.error(reply)

        if reply is not None:
            st.session_state.messages.append({"role": "assistant", "content": reply})
