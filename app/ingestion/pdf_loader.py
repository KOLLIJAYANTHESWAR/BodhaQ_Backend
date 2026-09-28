"""
PDF text extraction.
Uses PyMuPDF (fitz) for reliable page-level text with page numbers.
"""
from __future__ import annotations

import pymupdf as fitz  # PyMuPDF 1.24+ uses 'pymupdf'; aliased to 'fitz' for compatibility


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Returns a list of page dicts:
        [{"page": 1, "text": "..."}, ...]

    Pages with no extractable text (e.g. scanned images) are skipped.
    """
    pages: list[dict] = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{file_path}': {exc}") from exc

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})

    doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in '{file_path}'. "
            "The PDF may be scanned / image-only."
        )

    return pages
