"""
BodhaQ ChromaDB Vector Store.

Provides a single persistent ChromaDB client used across the application.

Collections are namespaced by both session_id and document_id so vector
data belonging to one anonymous BodhaQ session cannot be addressed through
another session's document namespace.

Security:
    - No API keys or other secrets are stored in this module.
    - Session tokens are never stored in ChromaDB.
    - Only the validated session identifier is used for data ownership.
    - Every retrieved chunk is checked against its session/document owner.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

_COLLECTION_PREFIX = "doc_"

_HASH_LENGTH = 32

_MAX_SESSION_ID_LENGTH = 100

_MAX_DOCUMENT_ID_LENGTH = 100

_MAX_CHUNK_ID_LENGTH = 512

_MAX_TOP_K = 20


_client: chromadb.ClientAPI | None = None

_client_lock = threading.Lock()


# ============================================================================
# CHROMA CLIENT
# ============================================================================


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Return or lazily create the singleton ChromaDB persistent client.

    The client is shared across the application because ChromaDB's
    persistent client manages access to the local vector database.
    """

    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            try:
                CHROMA_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                _client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(
                        anonymized_telemetry=False,
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "[Vector Store] Failed to initialize ChromaDB."
                )

                raise RuntimeError(
                    "Unable to initialize the document vector store."
                ) from exc

    return _client


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _normalize_session_id(
    session_id: str,
) -> str:
    """
    Validate and normalize an anonymous BodhaQ session UUID.

    The signed session token is validated by the FastAPI dependency.
    This additional validation protects this lower-level persistence layer.
    """

    if not isinstance(
        session_id,
        str,
    ):
        raise ValueError(
            "session_id must be a string."
        )

    normalized = session_id.strip()

    if not normalized:
        raise ValueError(
            "session_id cannot be empty."
        )

    if len(normalized) > _MAX_SESSION_ID_LENGTH:
        raise ValueError(
            "session_id is too long."
        )

    try:
        parsed = uuid.UUID(
            normalized
        )
    except (
        ValueError,
        AttributeError,
        TypeError,
    ) as exc:
        raise ValueError(
            "Invalid session_id."
        ) from exc

    return str(
        parsed
    )


def _normalize_document_id(
    document_id: str,
) -> str:
    """
    Validate and normalize a document UUID.
    """

    if not isinstance(
        document_id,
        str,
    ):
        raise ValueError(
            "document_id must be a string."
        )

    normalized = document_id.strip()

    if not normalized:
        raise ValueError(
            "document_id cannot be empty."
        )

    if len(normalized) > _MAX_DOCUMENT_ID_LENGTH:
        raise ValueError(
            "document_id is too long."
        )

    try:
        parsed = uuid.UUID(
            normalized
        )
    except (
        ValueError,
        AttributeError,
        TypeError,
    ) as exc:
        raise ValueError(
            "Invalid document_id."
        ) from exc

    return str(
        parsed
    )


def _validate_embedding(
    embedding: list[float],
    field_name: str,
) -> list[float]:
    """
    Validate an embedding vector.

    Ensures every value is numeric, finite, and usable by ChromaDB.
    """

    if not isinstance(
        embedding,
        list,
    ) or not embedding:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    normalized: list[float] = []

    for index, value in enumerate(
        embedding
    ):
        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ValueError(
                f"{field_name}[{index}] must be numeric."
            )

        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):
            raise ValueError(
                f"{field_name}[{index}] must be finite."
            )

        normalized.append(
            numeric_value
        )

    return normalized


def _validate_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate metadata before sending it to ChromaDB.

    Chroma metadata should contain primitive values rather than
    arbitrary nested objects/lists.
    """

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "Chunk metadata must be an object."
        )

    validated: dict[str, Any] = {}

    for key, value in metadata.items():
        if not isinstance(
            key,
            str,
        ):
            raise ValueError(
                "Chunk metadata keys must be strings."
            )

        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError(
                "Chunk metadata keys cannot be empty."
            )

        if value is None:
            # Chroma metadata does not reliably accept None.
            # Omit absent metadata instead.
            continue

        if isinstance(
            value,
            bool,
        ):
            validated[normalized_key] = value
            continue

        if isinstance(
            value,
            str,
        ):
            validated[normalized_key] = value
            continue

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
        ):
            validated[normalized_key] = value
            continue

        if isinstance(
            value,
            float,
        ):
            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"Metadata value for '{normalized_key}' "
                    "must be finite."
                )

            validated[normalized_key] = value
            continue

        raise ValueError(
            f"Unsupported metadata value type for "
            f"'{normalized_key}'."
        )

    return validated


def _validate_top_k(
    top_k: int,
) -> int:
    """
    Validate and bound the retrieval count.
    """

    if (
        isinstance(
            top_k,
            bool,
        )
        or not isinstance(
            top_k,
            int,
        )
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )

    return min(
        top_k,
        _MAX_TOP_K,
    )


def _is_collection_not_found_error(
    exc: Exception,
) -> bool:
    """
    Determine whether a Chroma exception represents a missing collection.

    Chroma's exact exception wording can vary between versions.
    """

    message = str(
        exc
    ).lower()

    return any(
        phrase in message
        for phrase in (
            "does not exist",
            "not found",
            "doesn't exist",
            "no such collection",
            "collection not found",
        )
    )


# ============================================================================
# COLLECTION HELPERS
# ============================================================================


def _collection_name(
    session_id: str,
    document_id: str,
) -> str:
    """
    Build a deterministic collision-resistant ChromaDB collection name.

    Both session_id and document_id are included in the namespace.

    Therefore:

        session A + document X

    and:

        session B + document X

    always resolve to different ChromaDB collections.
    """

    normalized_session_id = _normalize_session_id(
        session_id
    )

    normalized_document_id = _normalize_document_id(
        document_id
    )

    namespace = (
        f"{normalized_session_id}:"
        f"{normalized_document_id}"
    )

    namespace_hash = hashlib.sha256(
        namespace.encode(
            "utf-8"
        )
    ).hexdigest()[
        :_HASH_LENGTH
    ]

    return (
        f"{_COLLECTION_PREFIX}"
        f"{namespace_hash}"
    )


def get_or_create_collection(
    session_id: str,
    document_id: str,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for a session/document pair.

    Used when ingesting a new document.
    """

    collection_name = _collection_name(
        session_id,
        document_id,
    )

    client = get_chroma_client()

    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    except Exception as exc:
        logger.exception(
            "[Vector Store] Failed to create/access collection."
        )

        raise RuntimeError(
            "Unable to initialize the document vector store."
        ) from exc


def get_existing_collection(
    session_id: str,
    document_id: str,
) -> chromadb.Collection | None:
    """
    Return an existing ChromaDB collection for a session/document pair.

    Unlike get_or_create_collection(), this function never creates
    a collection.

    Returns:
        Existing collection, or None if the collection does not exist.
    """

    collection_name = _collection_name(
        session_id,
        document_id,
    )

    client = get_chroma_client()

    try:
        return client.get_collection(
            name=collection_name,
        )

    except Exception as exc:
        if _is_collection_not_found_error(
            exc
        ):
            return None

        logger.exception(
            "[Vector Store] Failed to access existing collection."
        )

        raise RuntimeError(
            "Unable to access the document vector store."
        ) from exc


# ============================================================================
# INGESTION
# ============================================================================


def upsert_chunks(
    session_id: str,
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert text chunks and their embeddings into the session-owned
    ChromaDB collection.

    Args:
        session_id:
            Owner of the document/vector data.

        document_id:
            ID of the document.

        chunks:
            Output of the document chunking pipeline.

            Each chunk must contain:
                - chunk_id
                - text
                - metadata

        embeddings:
            Pre-computed embedding vectors in the same order as chunks.
    """

    normalized_session_id = _normalize_session_id(
        session_id
    )

    normalized_document_id = _normalize_document_id(
        document_id
    )

    if (
        not isinstance(
            chunks,
            list,
        )
        or not chunks
    ):
        raise ValueError(
            "Cannot upsert chunks because the chunk list is empty."
        )

    if not isinstance(
        embeddings,
        list,
    ):
        raise ValueError(
            "Embeddings must be provided as a list."
        )

    if len(chunks) != len(
        embeddings
    ):
        raise ValueError(
            "The number of chunks must match "
            "the number of embeddings."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------
    # VALIDATE CHUNKS
    # ------------------------------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):
        if not isinstance(
            chunk,
            dict,
        ):
            raise ValueError(
                f"Chunk at index {index} must be an object."
            )

        chunk_id = chunk.get(
            "chunk_id"
        )

        text = chunk.get(
            "text"
        )

        metadata = chunk.get(
            "metadata"
        )

        if not isinstance(
            chunk_id,
            str,
        ):
            raise ValueError(
                f"Chunk at index {index} has an invalid chunk_id."
            )

        normalized_chunk_id = chunk_id.strip()

        if not normalized_chunk_id:
            raise ValueError(
                f"Chunk at index {index} has an invalid chunk_id."
            )

        if len(
            normalized_chunk_id
        ) > _MAX_CHUNK_ID_LENGTH:
            raise ValueError(
                f"Chunk at index {index} has an excessively long "
                "chunk_id."
            )

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                f"Chunk at index {index} has invalid text."
            )

        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                f"Chunk at index {index} has empty text."
            )

        validated_metadata = _validate_metadata(
            metadata
        )

        # Store ownership metadata with every chunk.
        #
        # The collection namespace is the primary isolation boundary.
        # Metadata ownership provides an additional consistency check.
        validated_metadata["session_id"] = (
            normalized_session_id
        )

        validated_metadata["document_id"] = (
            normalized_document_id
        )

        ids.append(
            normalized_chunk_id
        )

        documents.append(
            normalized_text
        )

        metadatas.append(
            validated_metadata
        )

    # ------------------------------------------------------------------------
    # VALIDATE CHUNK ID UNIQUENESS
    # ------------------------------------------------------------------------

    if len(
        ids
    ) != len(
        set(ids)
    ):
        raise ValueError(
            "Chunk IDs must be unique within an upsert operation."
        )

    # ------------------------------------------------------------------------
    # VALIDATE EMBEDDINGS
    # ------------------------------------------------------------------------

    validated_embeddings: list[list[float]] = []

    expected_dimension: int | None = None

    for index, embedding in enumerate(
        embeddings
    ):
        validated_embedding = _validate_embedding(
            embedding,
            f"Embedding at index {index}",
        )

        if expected_dimension is None:
            expected_dimension = len(
                validated_embedding
            )

        elif len(
            validated_embedding
        ) != expected_dimension:
            raise ValueError(
                "All embeddings must have the same vector dimension."
            )

        validated_embeddings.append(
            validated_embedding
        )

    if expected_dimension is None:
        raise ValueError(
            "No valid embeddings were provided."
        )

    # ------------------------------------------------------------------------
    # UPSERT
    # ------------------------------------------------------------------------

    collection = get_or_create_collection(
        session_id=normalized_session_id,
        document_id=normalized_document_id,
    )

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=validated_embeddings,
            metadatas=metadatas,
        )

    except Exception as exc:
        logger.exception(
            "[Vector Store] Failed to upsert document chunks."
        )

        raise RuntimeError(
            "Unable to store document embeddings."
        ) from exc


# ============================================================================
# RETRIEVAL
# ============================================================================


def query_collection(
    session_id: str,
    document_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-K most similar chunks from a session-owned
    document collection.
    """

    normalized_session_id = _normalize_session_id(
        session_id
    )

    normalized_document_id = _normalize_document_id(
        document_id
    )

    validated_embedding = _validate_embedding(
        query_embedding,
        "query_embedding",
    )

    normalized_top_k = _validate_top_k(
        top_k
    )

    collection = get_existing_collection(
        session_id=normalized_session_id,
        document_id=normalized_document_id,
    )

    if collection is None:
        raise ValueError(
            f"Document '{normalized_document_id}' was not found."
        )

    # ------------------------------------------------------------------------
    # CHECK COLLECTION CONTENT
    # ------------------------------------------------------------------------

    try:
        chunk_count = collection.count()

    except Exception as exc:
        logger.exception(
            "[Vector Store] Failed to count document chunks."
        )

        raise RuntimeError(
            "Unable to inspect document vector data."
        ) from exc

    if chunk_count <= 0:
        raise ValueError(
            f"Document '{normalized_document_id}' "
            "contains no indexed content."
        )

    result_count = min(
        normalized_top_k,
        chunk_count,
    )

    # ------------------------------------------------------------------------
    # QUERY CHROMA
    # ------------------------------------------------------------------------

    try:
        results = collection.query(
            query_embeddings=[
                validated_embedding
            ],
            n_results=result_count,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as exc:
        logger.exception(
            "[Vector Store] ChromaDB query failed."
        )

        raise RuntimeError(
            "Unable to retrieve document context."
        ) from exc

    # ------------------------------------------------------------------------
    # SAFELY EXTRACT RESULTS
    # ------------------------------------------------------------------------

    if not isinstance(
        results,
        dict,
    ):
        raise RuntimeError(
            "Vector store returned an invalid retrieval response."
        )

    documents = results.get(
        "documents"
    )

    metadatas = results.get(
        "metadatas"
    )

    distances = results.get(
        "distances"
    )

    if not documents:
        return []

    if not isinstance(
        documents,
        list,
    ):
        raise RuntimeError(
            "Vector store returned invalid document results."
        )

    documents = (
        documents[0]
        if documents
        else []
    )

    metadatas = (
        metadatas[0]
        if isinstance(
            metadatas,
            list,
        ) and metadatas
        else []
    )

    distances = (
        distances[0]
        if isinstance(
            distances,
            list,
        ) and distances
        else []
    )

    if not isinstance(
        documents,
        list,
    ):
        return []

    if not isinstance(
        metadatas,
        list,
    ):
        metadatas = []

    if not isinstance(
        distances,
        list,
    ):
        distances = []

    # ------------------------------------------------------------------------
    # RESULT ALIGNMENT
    # ------------------------------------------------------------------------
    #
    # Chroma should return aligned arrays. If the metadata or distance
    # arrays are unexpectedly shorter, do not fabricate values.
    #

    output: list[dict] = []

    for index, doc in enumerate(
        documents
    ):
        if not isinstance(
            doc,
            str,
        ):
            continue

        normalized_doc = doc.strip()

        if not normalized_doc:
            continue

        if index >= len(
            metadatas
        ):
            continue

        if index >= len(
            distances
        ):
            continue

        metadata = metadatas[
            index
        ]

        distance = distances[
            index
        ]

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        if (
            isinstance(
                distance,
                bool,
            )
            or not isinstance(
                distance,
                (int, float),
            )
        ):
            continue

        normalized_distance = float(
            distance
        )

        if not math.isfinite(
            normalized_distance
        ):
            continue

        if normalized_distance < 0:
            continue

        # --------------------------------------------------------------------
        # OWNERSHIP CONSISTENCY CHECK
        # --------------------------------------------------------------------

        metadata_session_id = str(
            metadata.get(
                "session_id",
                "",
            )
            or ""
        ).strip()

        metadata_document_id = str(
            metadata.get(
                "document_id",
                "",
            )
            or ""
        ).strip()

        if (
            metadata_session_id
            != normalized_session_id
            or metadata_document_id
            != normalized_document_id
        ):
            logger.warning(
                "[Vector Store] Ownership metadata mismatch "
                "during session/document retrieval."
            )

            continue

        output.append(
            {
                "text": normalized_doc,
                "metadata": metadata,
                "distance": normalized_distance,
            }
        )

    return output


# ============================================================================
# DELETION
# ============================================================================


def delete_collection(
    session_id: str,
    document_id: str,
) -> None:
    """
    Delete the ChromaDB collection for a session-owned document.

    If the collection does not exist, deletion is treated as successful.
    Other ChromaDB failures are surfaced as RuntimeError.
    """

    normalized_session_id = _normalize_session_id(
        session_id
    )

    normalized_document_id = _normalize_document_id(
        document_id
    )

    collection_name = _collection_name(
        session_id=normalized_session_id,
        document_id=normalized_document_id,
    )

    client = get_chroma_client()

    try:
        client.delete_collection(
            name=collection_name,
        )

    except Exception as exc:
        if _is_collection_not_found_error(
            exc
        ):
            return

        logger.exception(
            "[Vector Store] Failed to delete collection."
        )

        raise RuntimeError(
            "Unable to delete document vector data."
        ) from exc