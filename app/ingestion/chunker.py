"""
Text chunker.

Splits each document page into overlapping fixed-size character chunks
and attaches source metadata to every chunk.

Chunk metadata schema:

    {
        "source": "Operating_Systems.pdf",
        "document_id": "document-uuid",
        "page": 12,
        "chunk_index": 3
    }
"""

from __future__ import annotations

import hashlib
import re


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Number of hexadecimal characters used from the SHA-256 document hash
# in generated chunk IDs.
DOCUMENT_ID_HASH_LENGTH = 16


# ============================================================================
# HELPERS
# ============================================================================


def _slug(
    filename: str,
) -> str:
    """
    Create a short slug from a filename for use in chunk IDs.

    The slug is intentionally limited in length so filenames cannot
    make vector-store chunk IDs unnecessarily large.
    """
    if not isinstance(
        filename,
        str,
    ):
        raise ValueError(
            "Filename must be a string."
        )

    name = filename.strip()

    if not name:
        raise ValueError(
            "Filename cannot be empty."
        )

    # Strip extension.
    if "." in name:
        name = name.rsplit(
            ".",
            1,
        )[0]

    # Replace non-alphanumeric characters.
    name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        name,
    )

    # Remove leading/trailing underscores.
    name = name.strip("_")

    if not name:
        name = "document"

    return name[:12].lower()


def _document_id_hash(
    document_id: str,
) -> str:
    """
    Create a stable short hash for a document ID.

    The raw document ID is deliberately not embedded directly into
    chunk IDs. This keeps generated IDs compact and prevents arbitrary
    document ID content from affecting the vector-store identifier.
    """
    digest = hashlib.sha256(
        document_id.encode(
            "utf-8"
        )
    ).hexdigest()

    return digest[:DOCUMENT_ID_HASH_LENGTH]


def _validate_chunk_parameters(
    chunk_size: int,
    overlap: int,
) -> None:
    """
    Validate chunking configuration.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(
            chunk_size,
            int,
        )
        or chunk_size <= 0
    ):
        raise ValueError(
            "chunk_size must be a positive integer."
        )

    if (
        isinstance(overlap, bool)
        or not isinstance(
            overlap,
            int,
        )
        or overlap < 0
    ):
        raise ValueError(
            "overlap must be a non-negative integer."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )


def _normalize_page_text(
    text: str,
) -> str:
    """
    Normalize extracted page text before chunking.

    This preserves the actual textual content while removing
    unnecessary leading/trailing whitespace and collapsing repeated
    whitespace characters.
    """
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================================
# CHUNK PAGES
# ============================================================================


def chunk_pages(
    pages: list[dict],
    source_filename: str,
    document_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a list of page dictionaries into overlapping
    character-level chunks.

    Args:
        pages:
            Output of PDF/PPTX/DOCX loaders.

            Each page must contain:

                {
                    "page": int,
                    "text": str
                }

        source_filename:
            Original filename used in metadata.

        document_id:
            Database-level document ID.

        chunk_size:
            Maximum number of characters per chunk.

        overlap:
            Number of overlapping characters between
            consecutive chunks on the same page.

    Returns:
        List of chunk dictionaries:

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

    # ------------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------------

    if not isinstance(
        pages,
        list,
    ):
        raise ValueError(
            "pages must be a list."
        )

    if (
        not isinstance(
            source_filename,
            str,
        )
        or not source_filename.strip()
    ):
        raise ValueError(
            "source_filename cannot be empty."
        )

    if (
        not isinstance(
            document_id,
            str,
        )
        or not document_id.strip()
    ):
        raise ValueError(
            "document_id cannot be empty."
        )

    _validate_chunk_parameters(
        chunk_size,
        overlap,
    )

    normalized_filename = (
        source_filename.strip()
    )

    normalized_document_id = (
        document_id.strip()
    )

    slug = _slug(
        normalized_filename
    )

    document_hash = _document_id_hash(
        normalized_document_id
    )

    chunks: list[dict] = []

    # Track page numbers so duplicate page entries cannot silently
    # generate duplicate chunk IDs.
    seen_pages: set[int] = set()

    # ------------------------------------------------------------------------
    # PROCESS EACH PAGE
    # ------------------------------------------------------------------------

    for page_position, page_dict in enumerate(
        pages
    ):
        if not isinstance(
            page_dict,
            dict,
        ):
            raise ValueError(
                f"Page entry at index {page_position} "
                "must be an object."
            )

        page_num = page_dict.get(
            "page"
        )

        text = page_dict.get(
            "text"
        )

        if (
            isinstance(page_num, bool)
            or not isinstance(
                page_num,
                int,
            )
            or page_num < 1
        ):
            raise ValueError(
                f"Invalid page number at index "
                f"{page_position}."
            )

        if page_num in seen_pages:
            raise ValueError(
                f"Duplicate page number detected: "
                f"{page_num}."
            )

        seen_pages.add(
            page_num
        )

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                f"Text for page {page_num} "
                "must be a string."
            )

        text = _normalize_page_text(
            text
        )

        if not text:
            continue

        # --------------------------------------------------------------------
        # SLIDING WINDOW
        # --------------------------------------------------------------------

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + chunk_size,
                text_length,
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                # The raw document ID is kept in metadata, but a stable
                # short hash is used in the chunk ID.
                #
                # This prevents arbitrary document ID contents from
                # producing unnecessarily large or awkward vector-store IDs.
                chunk_id = (
                    f"{slug}_"
                    f"{document_hash}_"
                    f"{page_num:03d}_"
                    f"{chunk_index:03d}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source": normalized_filename,
                            "document_id": normalized_document_id,
                            "page": page_num,
                            "chunk_index": chunk_index,
                        },
                    }
                )

                chunk_index += 1

            # Last chunk reached.
            if end >= text_length:
                break

            start = end - overlap

    return chunks