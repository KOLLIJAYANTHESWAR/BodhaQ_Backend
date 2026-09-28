"""
Word document text extraction.
Uses python-docx to extract text from each paragraph, grouped
into logical page approximations (every N paragraphs).
"""
from __future__ import annotations

import math

from docx import Document

# Group every PARAS_PER_PAGE paragraphs into a logical "page"
PARAS_PER_PAGE = 20


def load_docx(file_path: str) -> list[dict]:
    """
    Extract text from a DOCX file.

    DOCX files don't have true page breaks accessible via python-docx,
    so paragraphs are grouped into logical pages of PARAS_PER_PAGE each.

    Returns:
        [{"page": <logical_page_number>, "text": "..."}, ...]
    """
    try:
        doc = Document(file_path)
    except Exception as exc:
        raise ValueError(f"Cannot open DOCX '{file_path}': {exc}") from exc

    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paras:
        raise ValueError(f"No extractable text found in '{file_path}'.")

    total_pages = math.ceil(len(paras) / PARAS_PER_PAGE)
    pages: list[dict] = []

    for page_num in range(1, total_pages + 1):
        start = (page_num - 1) * PARAS_PER_PAGE
        end = start + PARAS_PER_PAGE
        chunk_text = "\n".join(paras[start:end])
        pages.append({"page": page_num, "text": chunk_text})

    return pages
