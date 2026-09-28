"""
Retriever — the R in RAG.

Converts a user question into an embedding, queries ChromaDB,
and returns ranked context chunks ready to feed into Gemini.
"""
from __future__ import annotations

from app.rag.embeddings import embed_query
from app.rag.vector_store import query_collection

DEFAULT_TOP_K = 5
RELEVANCE_THRESHOLD = 1.0   # cosine distance; lower = more similar


def retrieve_context(
    document_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant text chunks for a question from a document.

    Args:
        document_id: The document to search within.
        question:    The user's natural-language question.
        top_k:       Maximum number of chunks to return.

    Returns:
        List of chunk dicts filtered by relevance threshold:
            {"text": str, "metadata": {"source": str, "page": int, ...}, "distance": float}
    """
    query_vector = embed_query(question)
    results = query_collection(document_id, query_vector, top_k=top_k)

    # Filter out chunks that are too distant (not relevant enough)
    relevant = [r for r in results if r["distance"] <= RELEVANCE_THRESHOLD]

    # Sort ascending by distance (most similar first)
    relevant.sort(key=lambda r: r["distance"])

    return relevant


def build_context_string(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context block for Gemini.

    Each chunk is labelled with its source and page for traceability.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(
            f"[{i}] Source: {source}, Page: {page}\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)
