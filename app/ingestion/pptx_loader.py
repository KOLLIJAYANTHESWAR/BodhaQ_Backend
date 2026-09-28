"""
PowerPoint text extraction.
Uses python-pptx to extract text from each slide with slide numbers.
"""
from __future__ import annotations

from pptx import Presentation


def load_pptx(file_path: str) -> list[dict]:
    """
    Extract text slide-by-slide from a PPTX file.

    Returns:
        [{"page": <slide_number>, "text": "..."}, ...]

    Slides with no text content are skipped.
    """
    pages: list[dict] = []
    try:
        prs = Presentation(file_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PPTX '{file_path}': {exc}") from exc

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        texts.append(line)

        if texts:
            pages.append({"page": slide_num, "text": "\n".join(texts)})

    if not pages:
        raise ValueError(f"No extractable text found in '{file_path}'.")

    return pages
