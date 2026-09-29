"""
BodhaQ RAG Retriever.

Responsibilities:
    - Convert a user question into an embedding.
    - Query the session-isolated ChromaDB collection.
    - Validate and rank retrieved chunks.
    - Build document-grounded context for Gemini.

Gemini API keys are request-scoped and are never stored or logged.

Session isolation:
    - Every retrieval operation requires a valid session_id.
    - ChromaDB is queried using both session_id and document_id.
    - A document from another session cannot be retrieved through
      this service.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Any

from app.rag.embeddings import embed_query
from app.rag.vector_store import query_collection


logger = logging.getLogger(__name__)


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Cosine distance:
# lower = more similar
RELEVANCE_THRESHOLD = 1.0

# Prevent unexpectedly large context blocks from being constructed
# from malformed or unusually large vector-store records.
MAX_CONTEXT_CHUNKS = 20
MAX_CHUNK_TEXT_LENGTH = 12000
MAX_SOURCE_LENGTH = 500
MAX_PAGE_LENGTH = 50

MAX_SESSION_ID_LENGTH = 100
MAX_DOCUMENT_ID_LENGTH = 200
MAX_QUESTION_LENGTH = 5000


# ============================================================================
# INTERNAL VALIDATION HELPERS
# ============================================================================


def _validate_session_id(
    session_id: str,
) -> str:
    """
    Validate and normalize an anonymous BodhaQ session ID.

    The session token itself is validated by the FastAPI session
    dependency. This additional validation protects this lower-level
    service if it is called directly.
    """

    if not isinstance(
        session_id,
        str,
    ):
        raise ValueError(
            "Session ID must be a string."
        )

    normalized = session_id.strip()

    if not normalized:
        raise ValueError(
            "Session ID cannot be empty."
        )

    if len(normalized) > MAX_SESSION_ID_LENGTH:
        raise ValueError(
            "Session ID is too long."
        )

    try:
        uuid.UUID(
            normalized
        )
    except (
        ValueError,
        AttributeError,
        TypeError,
    ) as exc:
        raise ValueError(
            "Invalid session ID."
        ) from exc

    return normalized


def _validate_api_key(
    api_key: str,
) -> str:
    """
    Validate a request-scoped Gemini API key.

    The key is never logged or persisted.
    """

    if not isinstance(
        api_key,
        str,
    ):
        raise ValueError(
            "Gemini API key must be a string."
        )

    normalized = api_key.strip()

    if not normalized:
        raise ValueError(
            "Gemini API key is required."
        )

    return normalized


def _validate_document_id(
    document_id: str,
) -> str:
    """
    Validate and normalize a document ID.
    """

    if not isinstance(
        document_id,
        str,
    ):
        raise ValueError(
            "Document ID must be a string."
        )

    normalized = document_id.strip()

    if not normalized:
        raise ValueError(
            "Document ID cannot be empty."
        )

    if len(normalized) > MAX_DOCUMENT_ID_LENGTH:
        raise ValueError(
            "Document ID is too long."
        )

    return normalized


def _validate_question(
    question: str,
) -> str:
    """
    Validate and normalize a user question.
    """

    if not isinstance(
        question,
        str,
    ):
        raise ValueError(
            "Question must be a string."
        )

    normalized = question.strip()

    if not normalized:
        raise ValueError(
            "Question cannot be empty."
        )

    if len(normalized) > MAX_QUESTION_LENGTH:
        raise ValueError(
            "Question is too long."
        )

    return normalized


def _validate_top_k(
    top_k: int,
) -> int:
    """
    Validate and bound the requested retrieval count.
    """

    # bool is a subclass of int in Python.
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    return min(
        top_k,
        MAX_TOP_K,
    )


# ============================================================================
# RETRIEVE CONTEXT
# ============================================================================


def retrieve_context(
    session_id: str,
    document_id: str,
    question: str,
    api_key: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant text chunks for a question from a
    session-owned document.

    Args:
        session_id:
            Anonymous BodhaQ session identifier.

        document_id:
            The document to search within.

        question:
            The user's natural-language question.

        api_key:
            Request-scoped Gemini API key supplied by the user.

        top_k:
            Maximum number of chunks to retrieve.

    Returns:
        List of relevant chunk dictionaries:

            {
                "text": str,
                "metadata": {
                    "source": str,
                    "page": int,
                    ...
                },
                "distance": float,
            }

    Raises:
        ValueError:
            If the session ID, document ID, question, API key,
            or top_k is invalid.

        RuntimeError:
            If embedding or vector retrieval fails.
    """

    # ------------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------------

    normalized_session_id = _validate_session_id(
        session_id
    )

    normalized_document_id = _validate_document_id(
        document_id
    )

    normalized_question = _validate_question(
        question
    )

    validated_api_key = _validate_api_key(
        api_key
    )

    normalized_top_k = _validate_top_k(
        top_k
    )

    # ------------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------------

    started_at = time.perf_counter()

    try:

        query_vector = embed_query(
            normalized_question,
            api_key=validated_api_key,
        )

        results = query_collection(
            session_id=normalized_session_id,
            document_id=normalized_document_id,
            query_embedding=query_vector,
            top_k=normalized_top_k,
        )

    except ValueError:
        raise

    except Exception as exc:

        logger.exception(
            "[RAG Retriever] Context retrieval failed | "
            "document_id=%s",
            normalized_document_id,
        )

        raise RuntimeError(
            "Document retrieval failed."
        ) from exc

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    logger.info(
        "[RAG Retriever] document=%s results=%s elapsed_ms=%.2f",
        normalized_document_id,
        len(results) if results else 0,
        elapsed_ms,
    )

    if not results:
        return []

    # ------------------------------------------------------------------------
    # VALIDATE AND FILTER RESULTS
    # ------------------------------------------------------------------------

    relevant: list[dict] = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        text = result.get(
            "text"
        )

        distance = result.get(
            "distance"
        )

        if not isinstance(
            text,
            str,
        ):
            continue

        normalized_text = text.strip()

        if not normalized_text:
            continue

        if isinstance(
            distance,
            bool,
        ) or not isinstance(
            distance,
            (int, float),
        ):
            continue

        normalized_distance = float(
            distance
        )

        # Reject NaN and infinity.
        if not math.isfinite(
            normalized_distance
        ):
            continue

        if normalized_distance < 0:
            continue

        if normalized_distance > RELEVANCE_THRESHOLD:
            continue

        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        # --------------------------------------------------------------------
        # DEFENSIVE OWNERSHIP CHECK
        # --------------------------------------------------------------------
        #
        # vector_store.py is responsible for enforcing collection-level
        # isolation. This additional check prevents a malformed or stale
        # metadata record from being returned by the retrieval layer.
        #

        result_session_id = str(
            metadata.get(
                "session_id",
                "",
            )
            or ""
        ).strip()

        result_document_id = str(
            metadata.get(
                "document_id",
                "",
            )
            or ""
        ).strip()

        if result_session_id != normalized_session_id:
            continue

        if result_document_id != normalized_document_id:
            continue

        relevant.append(
            {
                **result,
                "metadata": metadata,
                "text": normalized_text,
                "distance": normalized_distance,
            }
        )

    # ------------------------------------------------------------------------
    # SORT BY RELEVANCE
    # ------------------------------------------------------------------------

    relevant.sort(
        key=lambda item: item["distance"]
    )

    return relevant[:normalized_top_k]


# ============================================================================
# BUILD GEMINI CONTEXT
# ============================================================================


def build_context_string(
    chunks: list[dict],
) -> str:
    """
    Format retrieved chunks into a single context block for Gemini.

    Each chunk is labelled with its source and page for traceability.

    The function does not modify the original chunk objects.
    """

    if not isinstance(
        chunks,
        list,
    ) or not chunks:
        return ""

    parts: list[str] = []

    for index, chunk in enumerate(
        chunks[:MAX_CONTEXT_CHUNKS],
        start=1,
    ):

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        text = chunk.get(
            "text"
        )

        if not isinstance(
            text,
            str,
        ):
            continue

        normalized_text = text.strip()

        if not normalized_text:
            continue

        # Bound individual chunks before placing them into the
        # Gemini context.
        if len(normalized_text) > MAX_CHUNK_TEXT_LENGTH:

            normalized_text = (
                normalized_text[
                    :MAX_CHUNK_TEXT_LENGTH
                ].rstrip()
                + "\n[Content truncated]"
            )

        metadata = chunk.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "?",
        )

        # Prevent malformed metadata from creating unexpectedly
        # large context strings.
        source = str(
            source
        )[:MAX_SOURCE_LENGTH]

        page = str(
            page
        )[:MAX_PAGE_LENGTH]

        parts.append(
            (
                f"[{index}] "
                f"Source: {source}, "
                f"Page: {page}\n"
                f"{normalized_text}"
            )
        )

    return "\n\n---\n\n".join(
        parts
    )