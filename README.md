# Book RAG + Gemini (PDF, two-layer answers)

Small **retrieval-augmented generation (RAG)** demo: ingest a **PDF**, then chat with **Gemini** using (1) **book-grounded** context and (2) a **supplemental** answer that rounds out the topic.

## What you need installed


| Tool                           | Why                                                                      |
| ------------------------------ | ------------------------------------------------------------------------ |
| **Python 3.10+**               | Runtime                                                                  |
| **pip**                        | Install dependencies                                                     |
| A **Google AI Studio API key** | Gemini chat generation ([get a key](https://aistudio.google.com/apikey)) |
| **Streamlit**                  | Installed via `requirements.txt`; used for the chat UI                   |


The code uses Google’s current `google-genai` Python SDK (not the deprecated `google-generativeai` package).

Optional but recommended: **Git** for version control (not required to run the app).

## Setup

```powershell
cd C:\Users\mhmds\Projects\book-rag-gemini
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

```cmd
cd C:\Users\mhmds\Projects\book-rag-gemini
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` in an editor and set `GOOGLE_API_KEY=...` (and optional `EMBEDDING_BACKEND=local`).

## Run the interactive app (Streamlit)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

```cmd
.\.venv\Scripts\activate.bat
python -m streamlit run streamlit_app.py
```

1. Open the local URL Streamlit prints (usually `http://localhost:8501`).
2. In the sidebar: upload a PDF, choose a **collection name**, click **Ingest**.
3. Ask questions in the chat. Replies have **From the book** and **Beyond the book** sections (two Gemini calls).

## Embedding backend modes (new)

Set `EMBEDDING_BACKEND` in `.env`:

- `local` (default): embeddings are computed on your machine via `sentence-transformers`; no Gemini embedding quota is used.
- `gemini`: embeddings are computed via Gemini API; uses embedding quota.

Note: chat generation still uses Gemini in both modes, so `GOOGLE_API_KEY` is still required.

## Ingest from the command line (no UI)

```powershell
python ingest.py --pdf "C:\path\to\book.pdf" --collection mybook
```

By default this **clears** the collection first. To append (advanced), use `--no-reset` (risk: duplicate chunk IDs if you re-ingest the same PDF).
To reduce quota usage during testing, use `--max-chunks 120` (or any smaller number).

Then run Streamlit and use the **same collection name**.

## Common error: Gemini 429 RESOURCE_EXHAUSTED

If you see `429 RESOURCE_EXHAUSTED`, your Gemini API quota/rate limit is exhausted. This is not a code bug.

Try these fixes:

- Wait for quota reset and retry later.
- Check usage/limits in [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) and [usage dashboard](https://ai.dev/rate-limit).
- In Streamlit, lower **Max chunks to ingest**, increase chunk size, and retry.
- In CLI ingest, pass `--max-chunks` to embed fewer chunks while testing.

## Suggested reading order (to learn the code)

1. `src/config.py` — environment variables and model names.
2. `src/pdf_extract.py` — PDF → text (what “corpus” means here).
3. `src/chunking.py` — why overlap exists; what a **chunk** is.
4. `src/embeddings.py` — **RETRIEVAL_DOCUMENT** vs **QUESTION_ANSWERING** task types for `gemini-embedding-001`.
5. `src/vector_store.py` — Chroma: where vectors and text live.
6. `ingest.py` — wires ingestion end-to-end.
7. `src/rag.py` — retrieval + two Gemini calls (book vs supplement).
8. `streamlit_app.py` — UI and session state.

## Project map (read this while studying the code)


| File                  | Role                                                                |
| --------------------- | ------------------------------------------------------------------- |
| `src/config.py`       | API key, model names, chunk defaults                                |
| `src/pdf_extract.py`  | PDF → text (PyMuPDF)                                                |
| `src/chunking.py`     | Long text → overlapping chunks + metadata                           |
| `src/embeddings.py`   | Text → vectors (local transformer or Gemini embedding API)          |
| `src/vector_store.py` | Vectors + text in **Chroma** (on-disk)                              |
| `src/rag.py`          | Retrieve top-k chunks → **Gemini** (book) → **Gemini** (supplement) |
| `ingest.py`           | CLI pipeline: PDF → chunks → embed → store                          |
| `streamlit_app.py`    | Web UI                                                              |


## Concepts this demo encodes

- **Chunking**: split the book so each piece fits embedding/context limits; overlap reduces “cut” facts.
- **Embedding (bi-encoder style)**: the same family of model maps **passages** and **questions** into one vector space so **nearest neighbors** ≈ **semantically related**.
- **Vector DB**: fast approximate nearest neighbor search (here: Chroma + HNSW, cosine space).
- **RAG**: retrieve passages, paste into prompt, ask the LLM to **ground** answers in those passages.
- **Two-call pattern**: first answer is **faithful to the book**; second adds **general context** without pretending it came from the PDF.

## Interview prep: theory

See `docs/INTERVIEW_DEEP_DIVE.md` for transformers, how they show up in embeddings vs generation, and ML terminology tied to this repo.

## Limits of this demo

- **Scanned PDFs** (images only) need OCR; this code uses text extraction only.
- **Rate limits**: large books mean many embedding API calls; ingest may take time.
- **Legal / privacy**: only ingest content you are allowed to process and store.

