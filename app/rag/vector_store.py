"""
ChromaDB vector store wrapper.

Provides a single persistent ChromaDB client used across the application.

Collections are namespaced per document_id so they remain isolated.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings


# ── Persistent storage ───────────────────────────────────────────────────────

_CHROMA_PATH = "data/chromadb"

_client: chromadb.ClientAPI | None = None


# ── Client ────────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.ClientAPI:
    """Return (or lazily create) the singleton ChromaDB persistent client."""
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=_CHROMA_PATH,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

    return _client


# ── Collection helpers ────────────────────────────────────────────────────────

def _collection_name(document_id: str) -> str:
    """Build the ChromaDB collection name for a document."""
    return f"doc_{document_id.replace('-', '_')}"


def get_or_create_collection(
    document_id: str,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for the given document_id.

    Used when ingesting a new document.
    """
    client = get_chroma_client()

    collection_name = _collection_name(document_id)

    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
        },
    )


def get_existing_collection(
    document_id: str,
) -> chromadb.Collection | None:
    """
    Return an existing ChromaDB collection for a document.

    Unlike get_or_create_collection(), this function never creates
    a collection.

    Returns:
        Existing collection, or None if it does not exist.
    """
    client = get_chroma_client()

    collection_name = _collection_name(document_id)

    try:
        return client.get_collection(
            name=collection_name,
        )
    except Exception:
        return None


# ── Ingestion ─────────────────────────────────────────────────────────────────

def upsert_chunks(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert text chunks and their embeddings into ChromaDB.

    Args:
        document_id:
            Used to select the right collection.

        chunks:
            Output of chunker.chunk_pages().
            Each chunk contains chunk_id, text, and metadata.

        embeddings:
            Pre-computed embedding vectors in the same order as chunks.
    """
    if not chunks:
        raise ValueError(
            "Cannot upsert chunks because the chunk list is empty."
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    collection = get_or_create_collection(
        document_id,
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query_collection(
    document_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-K most similar chunks for a query embedding.

    Returns:
        List of result dictionaries:

        {
            "text": str,
            "metadata": dict,
            "distance": float,
        }

    Raises:
        ValueError:
            If the document collection does not exist or contains
            no chunks.
    """
    if not document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    if not query_embedding:
        raise ValueError(
            "query_embedding cannot be empty."
        )

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    collection = get_existing_collection(
        document_id,
    )

    if collection is None:
        raise ValueError(
            f"Document '{document_id}' was not found."
        )

    chunk_count = collection.count()

    if chunk_count == 0:
        raise ValueError(
            f"Document '{document_id}' contains no indexed content."
        )

    result_count = min(
        top_k,
        chunk_count,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding,
        ],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    output: list[dict] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(
        documents,
        metadatas,
        distances,
    ):
        output.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )

    return output


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_collection(
    document_id: str,
) -> None:
    """
    Delete the ChromaDB collection for a document.

    Used when a document is deleted.
    """
    client = get_chroma_client()

    collection_name = _collection_name(
        document_id,
    )

    try:
        client.delete_collection(
            name=collection_name,
        )
    except Exception:
        # Collection may already have been deleted.
        pass