"""Extract plain text from a PDF with lightweight page markers for traceability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """
    Read each page of the PDF and return structured text.

    Why PyMuPDF: fast, no external binaries, good default text order for many PDFs.
    Scanned PDFs (images only) need OCR — out of scope for this demo.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        pages: list[PageText] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = text.strip()
            pages.append(PageText(page_number=i + 1, text=text))
        return pages
    finally:
        doc.close()


def pages_to_marked_document(pages: list[PageText]) -> str:
    """
    Join pages with explicit markers so chunks can still hint at page location.

    The model does not need these markers to answer; they help you debug sources.
    """
    parts: list[str] = []
    for p in pages:
        if not p.text:
            continue
        parts.append(f"--- Page {p.page_number} ---\n{p.text}")
    return "\n\n".join(parts)
