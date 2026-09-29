"""
Embedding generation using the Gemini Embedding model.

This module is responsible only for generating embeddings.

Document embeddings use RETRIEVAL_DOCUMENT.

Query embeddings use RETRIEVAL_QUERY.

Security:
    Gemini API keys are request-scoped and are never loaded from
    environment variables, stored globally, or persisted by this module.
"""

from __future__ import annotations

import logging
from typing import Literal

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


# ============================================================================
# GEMINI EMBEDDING CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "gemini-embedding-001"

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

EmbeddingTask = Literal[
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _validate_api_key(
    api_key: str,
) -> str:
    """
    Validate a request-scoped Gemini API key.

    The key is intentionally not logged, persisted, or included
    in exception messages.
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


def _create_client(
    api_key: str,
) -> genai.Client:
    """
    Create a Gemini client using the request-scoped API key.

    No global client is retained so different browser sessions can
    safely provide different BYOK keys.
    """
    validated_key = _validate_api_key(
        api_key
    )

    try:
        return genai.Client(
            api_key=validated_key,
        )
    except Exception as exc:
        logger.exception(
            "[Embeddings] Failed to initialize Gemini client."
        )

        raise RuntimeError(
            "Unable to initialize the embedding service."
        ) from exc


# ============================================================================
# EMBEDDINGS
# ============================================================================


def embed_texts(
    texts: list[str],
    api_key: str,
    task_type: EmbeddingTask = TASK_DOCUMENT,
) -> list[list[float]]:
    """
    Generate embeddings for a list of text inputs.

    Args:
        texts:
            Text strings to embed.

        api_key:
            Request-scoped Gemini API key supplied by the user.

        task_type:
            Gemini retrieval task type.

            RETRIEVAL_DOCUMENT:
                Used when embedding document chunks.

            RETRIEVAL_QUERY:
                Used when embedding user queries.

    Returns:
        One embedding vector for each input text.

    Raises:
        ValueError:
            If the task type, API key, or input is invalid.

        RuntimeError:
            If embedding generation fails or Gemini returns
            an invalid response.
    """

    # ------------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------------

    if not isinstance(
        texts,
        list,
    ):
        raise ValueError(
            "Embedding input must be a list of strings."
        )

    if not texts:
        return []

    if task_type not in {
        TASK_DOCUMENT,
        TASK_QUERY,
    }:
        raise ValueError(
            f"Unsupported embedding task type: {task_type}"
        )

    cleaned_texts: list[str] = []

    for text in texts:
        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                "All embedding inputs must be strings."
            )

        cleaned = text.strip()

        if not cleaned:
            raise ValueError(
                "Embedding input cannot contain empty text."
            )

        cleaned_texts.append(
            cleaned
        )

    # ------------------------------------------------------------------------
    # CREATE REQUEST-SCOPED GEMINI CLIENT
    # ------------------------------------------------------------------------

    client = _create_client(
        api_key
    )

    # ------------------------------------------------------------------------
    # GENERATE EMBEDDINGS
    # ------------------------------------------------------------------------

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=cleaned_texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )

    except Exception as exc:
        # Never include the API key in the log or exception message.
        logger.exception(
            "[Embeddings] Gemini embedding generation failed."
        )

        raise RuntimeError(
            "Embedding generation failed."
        ) from exc

    # ------------------------------------------------------------------------
    # VALIDATE RESPONSE
    # ------------------------------------------------------------------------

    if (
        response is None
        or not getattr(
            response,
            "embeddings",
            None,
        )
    ):
        raise RuntimeError(
            "Embedding service returned no embeddings."
        )

    embeddings: list[list[float]] = []

    for embedding in response.embeddings:

        if embedding is None:
            raise RuntimeError(
                "Embedding service returned an invalid embedding."
            )

        values = getattr(
            embedding,
            "values",
            None,
        )

        if not values:
            raise RuntimeError(
                "Embedding service returned an empty embedding."
            )

        try:
            vector = [
                float(value)
                for value in values
            ]
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Embedding service returned invalid vector values."
            ) from exc

        if not vector:
            raise RuntimeError(
                "Embedding service returned an empty embedding."
            )

        embeddings.append(
            vector
        )

    # ------------------------------------------------------------------------
    # RESPONSE COUNT VALIDATION
    # ------------------------------------------------------------------------

    if len(embeddings) != len(
        cleaned_texts
    ):
        logger.error(
            "[Embeddings] Unexpected embedding count: "
            "expected=%s received=%s",
            len(cleaned_texts),
            len(embeddings),
        )

        raise RuntimeError(
            "Embedding service returned an unexpected "
            "number of embedding vectors."
        )

    # ------------------------------------------------------------------------
    # VECTOR DIMENSION VALIDATION
    # ------------------------------------------------------------------------

    expected_dimension = len(
        embeddings[0]
    )

    if expected_dimension <= 0:
        raise RuntimeError(
            "Embedding service returned an invalid vector dimension."
        )

    for vector in embeddings:
        if len(vector) != expected_dimension:
            raise RuntimeError(
                "Embedding service returned vectors "
                "with inconsistent dimensions."
            )

    return embeddings


# ============================================================================
# QUERY EMBEDDINGS
# ============================================================================


def embed_query(
    text: str,
    api_key: str,
) -> list[float]:
    """
    Generate an embedding for a user search/query.

    Query embeddings use RETRIEVAL_QUERY so that they are optimized
    for retrieval against document embeddings.
    """

    if not isinstance(
        text,
        str,
    ):
        raise ValueError(
            "Query text must be a string."
        )

    normalized_text = text.strip()

    if not normalized_text:
        raise ValueError(
            "Query text cannot be empty."
        )

    vectors = embed_texts(
        [normalized_text],
        api_key=api_key,
        task_type=TASK_QUERY,
    )

    if len(vectors) != 1:
        raise RuntimeError(
            "Query embedding generation returned "
            "an unexpected result."
        )

    return vectors[0]


# ============================================================================
# DOCUMENT EMBEDDINGS
# ============================================================================


def embed_documents(
    texts: list[str],
    api_key: str,
) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Document embeddings use RETRIEVAL_DOCUMENT so they can be
    compared against RETRIEVAL_QUERY embeddings during RAG retrieval.

    The API key is request-scoped and is not persisted.
    """

    return embed_texts(
        texts,
        api_key=api_key,
        task_type=TASK_DOCUMENT,
    )