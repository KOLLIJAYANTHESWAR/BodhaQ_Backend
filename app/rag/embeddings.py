"""
Embedding generation using the Gemini Embedding model.

This module is responsible only for generating embeddings.
Document embeddings use RETRIEVAL_DOCUMENT.
Query embeddings use RETRIEVAL_QUERY.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY


# ── Gemini embedding configuration ────────────────────────────────────────────

EMBEDDING_MODEL = "gemini-embedding-001"

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


# ── Gemini client ─────────────────────────────────────────────────────────────

_client = genai.Client(
    api_key=GEMINI_API_KEY,
)


# ── Internal helpers ──────────────────────────────────────────────────────────


def embed_texts(
    texts: list[str],
    task_type: str = TASK_DOCUMENT,
) -> list[list[float]]:
    """
    Generate embeddings for a list of text inputs.

    Args:
        texts:
            Text strings to embed.

        task_type:
            Gemini retrieval task type.

            RETRIEVAL_DOCUMENT:
                Used when embedding document chunks.

            RETRIEVAL_QUERY:
                Used when embedding user queries.

    Returns:
        One embedding vector for each input text.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    if task_type not in {
        TASK_DOCUMENT,
        TASK_QUERY,
    }:
        raise ValueError(
            f"Unsupported embedding task type: {task_type}"
        )

    try:
        response = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=cleaned_texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )

    except Exception as exc:
        raise RuntimeError(
            f"Embedding generation failed: {exc}"
        ) from exc

    if not response.embeddings:
        raise RuntimeError(
            "Embedding service returned no embeddings."
        )

    embeddings = [
        embedding.values
        for embedding in response.embeddings
        if embedding.values
    ]

    if len(embeddings) != len(cleaned_texts):
        raise RuntimeError(
            "Embedding service returned an unexpected number "
            "of embedding vectors."
        )

    return embeddings


# ── Query embeddings ──────────────────────────────────────────────────────────


def embed_query(
    text: str,
) -> list[float]:
    """
    Generate an embedding for a user search/query.

    Query embeddings use RETRIEVAL_QUERY so that they are optimized
    for retrieval against document embeddings.
    """

    if not text or not text.strip():
        raise ValueError(
            "Query text cannot be empty."
        )

    vectors = embed_texts(
        [text],
        task_type=TASK_QUERY,
    )

    if not vectors:
        raise RuntimeError(
            "Query embedding generation returned no vector."
        )

    return vectors[0]


# ── Document embeddings ───────────────────────────────────────────────────────


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Document embeddings use RETRIEVAL_DOCUMENT so they can be
    compared against RETRIEVAL_QUERY embeddings during RAG retrieval.
    """

    return embed_texts(
        texts,
        task_type=TASK_DOCUMENT,
    )