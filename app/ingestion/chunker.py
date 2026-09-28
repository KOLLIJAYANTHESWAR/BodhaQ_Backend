"""
Text chunker.

Splits each document page into overlapping fixed-size character chunks
and attaches source metadata to every chunk.

Chunk metadata schema:
    {
        "source": "Operating_Systems.pdf",
        "page": 12,
        "chunk_id": "os_12_03"
    }
"""
from __future__ import annotations

import re
import uuid

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def _slug(filename: str) -> str:
    """Create a short slug from a filename for use in chunk IDs."""
    name = filename.rsplit(".", 1)[0]           # strip extension
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name) # non-alphanum → underscore
    return name[:12].lower()


def chunk_pages(
    pages: list[dict],
    source_filename: str,
    document_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a list of page dicts into overlapping character-level chunks.

    Args:
        pages:            Output of pdf_loader / pptx_loader / docx_loader.
        source_filename:  Original filename (used in metadata).
        document_id:      The DB-level document ID (used in chunk_id prefix).
        chunk_size:       Maximum characters per chunk.
        overlap:          Character overlap between successive chunks.

    Returns:
        List of chunk dicts:
            {
                "chunk_id": str,
                "text": str,
                "metadata": {
                    "source": str,
                    "document_id": str,
                    "page": int,
                    "chunk_index": int
                }
            }
    """
    chunks: list[dict] = []
    slug = _slug(source_filename)

    for page_dict in pages:
        page_num: int = page_dict["page"]
        text: str = page_dict["text"]

        # Walk through the page text with a sliding window
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{slug}_{page_num:03d}_{chunk_index:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source_filename,
                        "document_id": document_id,
                        "page": page_num,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1

            if end >= len(text):
                break
            start = end - overlap   # slide forward with overlap

    return chunks
