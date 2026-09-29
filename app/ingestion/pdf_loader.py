"""
PDF text extraction.

Uses PyMuPDF (pymupdf) for reliable page-level text extraction
with physical PDF page numbers.

Pages without extractable text, such as scanned/image-only pages,
are skipped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf as fitz


logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================


def _normalize_page_text(
    text: str,
) -> str:
    """
    Normalize extracted PDF text before returning it.

    PyMuPDF can produce excessive whitespace depending on the
    source PDF. Collapse repeated horizontal/vertical whitespace
    while preserving paragraph-like line breaks where possible.
    """
    if not isinstance(text, str):
        return ""

    # Normalize carriage returns.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Remove trailing whitespace from individual lines.
    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Collapse repeated spaces/tabs inside lines.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def _validate_file_path(
    file_path: str,
) -> Path:
    """
    Validate and resolve the supplied PDF path.
    """
    if (
        not isinstance(
            file_path,
            str,
        )
        or not file_path.strip()
    ):
        raise ValueError(
            "PDF file path cannot be empty."
        )

    path = Path(
        file_path.strip()
    )

    try:
        path = path.resolve()
    except (
        OSError,
        RuntimeError,
    ) as exc:
        logger.warning(
            "[PDF Loader] Unable to resolve PDF path."
        )

        raise ValueError(
            "Unable to access the PDF file."
        ) from exc

    if not path.exists():
        raise ValueError(
            "PDF file was not found."
        )

    if not path.is_file():
        raise ValueError(
            "PDF path does not point to a file."
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Expected a PDF file."
        )

    return path


# ============================================================================
# PDF LOADER
# ============================================================================


def load_pdf(
    file_path: str,
) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:

        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]

    Pages with no extractable text are skipped.

    Page numbers represent the physical page number within
    the PDF document.

    Raises:
        ValueError:
            If the path is invalid, the PDF cannot be opened,
            or no extractable text is found.
    """

    # ------------------------------------------------------------------------
    # VALIDATE PATH
    # ------------------------------------------------------------------------

    path = _validate_file_path(
        file_path
    )

    # ------------------------------------------------------------------------
    # OPEN PDF
    # ------------------------------------------------------------------------

    pages: list[dict] = []

    try:
        with fitz.open(
            str(path)
        ) as doc:

            # A valid PDF may technically contain zero pages.
            if doc.page_count <= 0:
                raise ValueError(
                    "The PDF document contains no pages."
                )

            # ---------------------------------------------------------------
            # EXTRACT PAGE TEXT
            # ---------------------------------------------------------------

            for page_num, page in enumerate(
                doc,
                start=1,
            ):
                try:
                    raw_text = page.get_text(
                        "text"
                    )
                except Exception as exc:
                    # One problematic page should not expose an internal
                    # PyMuPDF exception to the user. Log the page number
                    # for debugging and continue with other pages.
                    logger.warning(
                        "[PDF Loader] Failed to extract text "
                        "from page %s.",
                        page_num,
                        exc_info=exc,
                    )
                    continue

                text = _normalize_page_text(
                    raw_text
                )

                if not text:
                    continue

                pages.append(
                    {
                        "page": page_num,
                        "text": text,
                    }
                )

    except ValueError:
        # Preserve our intentional validation/extraction errors.
        raise

    except (
        fitz.FileDataError,
        fitz.EmptyFileError,
        OSError,
    ) as exc:
        logger.exception(
            "[PDF Loader] Failed to open PDF document."
        )

        raise ValueError(
            "Unable to open the PDF document. "
            "The file may be corrupted or invalid."
        ) from exc

    except Exception as exc:
        logger.exception(
            "[PDF Loader] Unexpected PDF extraction failure."
        )

        raise ValueError(
            "Unable to read the PDF document."
        ) from exc

    # ------------------------------------------------------------------------
    # VALIDATE EXTRACTION
    # ------------------------------------------------------------------------

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF document. "
            "The PDF may be scanned or image-only."
        )

    return pages