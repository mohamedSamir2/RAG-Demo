"""Central configuration: API key, model names, paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env in the project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

# Gemini model for chat (fast, good for demos). Swap for gemini-1.5-pro if you need depth.
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

# Text embeddings (document vs question task types — see src/embeddings.py).
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local").strip().lower()
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Where Chroma persists vectors on disk
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(_PROJECT_ROOT / "chroma_data")))

# Ingestion defaults
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "6"))


def require_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Copy .env.example to .env and add your key "
            "(https://aistudio.google.com/apikey)."
        )
    return GOOGLE_API_KEY
