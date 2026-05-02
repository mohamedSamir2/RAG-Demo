# Interview deep dive: ML, deep learning, transformers, and this RAG

This document ties **theory** to **this repository** so you can defend a demo in a technical interview.

## 1. Machine learning vs deep learning (how you should phrase it)

- **Machine learning (ML)** is the broad field: learn a mapping from data instead of hand-writing rules.
- **Deep learning** is ML using **deep neural networks** (many layers). It powers modern **language models** and **embedding models**.

In this project:

- **Embeddings** come from a **deep model** (transformer-based encoder) via either:
  - local `sentence-transformers` on your machine, or
  - Gemini embedding API.
- **Answers** come from a **large language model (LLM)** (transformer-based decoder stack, depending on the exact Gemini variant), exposed via Gemini’s generate API.

You are not training either model here; you **call pretrained models** (inference). Your “learning” artifact is the **vector index** (stored chunks + vectors), which is **data engineering for retrieval**, not weight updates.

## 2. Supervised vs self-supervised (what pretrained LLMs are)

- **Supervised learning**: inputs paired with explicit labels (spam/not spam).
- **Self-supervised learning** (common in NLP): the model creates its own training signal from raw text (e.g. predict the next token, masked word, or contrastive pairs). **LLMs** are typically pretrained with **next-token prediction** on huge corpora, then optionally aligned with human feedback.

For interviews: RAG does not replace pretraining; it **grounds** the pretrained model on **your** document at query time.

## 3. Transformers: the core idea

A **transformer** is an architecture built on **attention**, especially **self-attention**, which lets each token in a sequence **look at** other tokens and weigh their influence.

Key terms:

- **Token**: a substring piece after **tokenization** (not always a whole word).
- **Self-attention**: for each position, build a weighted combination of representations of all positions.
- **Multi-head attention**: several attention mechanisms in parallel to capture different relationship types.
- **Positional information**: added via positional encodings or rotary embeddings so “order” is not lost.

Two big families:

- **Encoder** (e.g. BERT-style): good for **understanding** / **representation** tasks; produces contextual embeddings.
- **Decoder** (e.g. GPT-style): **autoregressive** generation—predict next token conditioned on previous tokens.

Many modern LLMs use **decoder-only** stacks for chat, but may include mixture-of-experts, etc.

## 4. Where transformers appear in *this* RAG

### A) Embeddings (`src/embeddings.py`)

Both embedding backends are transformer-based encoders that map variable-length text into fixed-length vectors:

- **Local mode**: `sentence-transformers/all-MiniLM-L6-v2` runs fully on CPU/GPU on your machine.
- **Gemini mode**: `gemini-embedding-001` via API.

**Why vectors?** Because we can measure **similarity** (e.g. cosine similarity) in that space: texts with “similar meaning” often land nearby—**not perfectly**, but usefully.

In **Gemini mode**, this repo uses **`gemini-embedding-001`** with two **task types** from the API: **`RETRIEVAL_DOCUMENT`** for chunks and **`QUESTION_ANSWERING`** for the user’s question. That is a mild form of **asymmetric retrieval**: the model is nudged to put “questions” and “passages you might answer with” in a shared geometry optimized for QA-style search, not raw string equality.

In **local mode**, one encoder embeds both documents and query in the same learned space (symmetric setup).

Trade-off summary:

- **Local embeddings**: no embedding API quota/cost, lower latency for small-medium corpora, but model quality may be lower than frontier hosted embeddings on some domains.
- **Gemini embeddings**: potentially stronger semantic quality and multilingual breadth, but network latency and quota limits apply.

This is the classic **bi-encoder** retrieval idea:

- Encode all **document chunks** once at ingest.
- Encode the **user question** at query time.
- Retrieve chunks whose vectors are **nearest** to the question vector.

You are not doing cross-attention between every chunk and the query at retrieval time (that would be a **cross-encoder**, heavier but sometimes more accurate). This demo uses **bi-encoder + ANN search** for speed.

### B) Generation (`src/rag.py`)

You call Gemini’s **chat model**. For answering, the model **conditions** on the prompt (context + question) and generates tokens **autoregressively** (each new token depends on prior tokens), using transformer layers.

**Important nuance:** retrieval happens **outside** the transformer as a separate step. The transformer does not “magically know” your PDF; you **inject** retrieved text into its input context.

## 5. The RAG loop as a systems diagram

1. **Offline**: PDF → text → chunks → **embedding model** → vectors → **vector DB**.
2. **Online**: question → **embedding model** → vector search (top-k) → build prompt with chunks → **LLM** → answer.

Failure modes interviewers like:

- **Chunking** splits a fact across two chunks → neither ranks high → **missed retrieval**.
- **Embedding mismatch** (domain shift) → retrieves irrelevant passages → **grounded but wrong context**.
- **Prompt not strict** → model ignores context → **hallucination under the guise of RAG**.
- **Context too long** → must truncate or compress → **lost detail**.

## 6. Vector search is not “magic retrieval”

Chroma here uses an **approximate nearest neighbor (ANN)** index (HNSW is common). It trades a little accuracy for speed.

**Cosine similarity** measures the angle between vectors (orientation), often used when magnitude is less important than direction.

## 7. Why two Gemini calls (book + supplement)

This is **orchestration**, not a second ML model by default:

- Call 1: **constrain** the model to the retrieved passages (faithfulness to the book).
- Call 2: **allow** broader knowledge, explicitly labeled, to **complete the circle** without pretending it is from the PDF.

Alternatives: one call with two sections (less separation), tool use / search grounding (different product features), or a small classifier to decide when to augment.

## 8. What you should be able to say in “physics / fundamentals” terms

- **Representation learning**: embeddings map text to a space where geometric nearness ≈ semantic relatedness (learned from data).
- **Conditional probability** in generation: LLMs implement something like \(P(\text{token}_t \mid \text{prior tokens}, \text{prompt})\) approximated by the network.
- **Attention** routes information across tokens; **feed-forward** layers transform per-token features.
- **RAG** reduces reliance on parametric memory (weights) for facts by adding **non-parametric memory** (your chunks) at inference time.

## 9. Honest scope statement (shows maturity)

This demo teaches **core RAG**. Production systems add: evaluation sets, reranking, query expansion, hybrid sparse+dense retrieval, caching, auth, PII handling, observability, cost controls, and continuous ingestion.
