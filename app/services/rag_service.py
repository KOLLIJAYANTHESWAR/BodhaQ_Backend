"""
RAG Service — bridges document retrieval and Gemini.

Responsibilities:
    - Verify that the requested document exists.
    - Retrieve relevant chunks from ChromaDB.
    - Build context for document-grounded generation.
    - Return source references for retrieved content.

This service does NOT call Gemini directly.
GeminiService is responsible for generation.

SQLite is the authoritative source for document existence.
ChromaDB is the vector-storage layer.
"""

from __future__ import annotations

import logging
import time

from app.models.responses import SourceReference

logger = logging.getLogger(__name__)
from app.rag.retriever import (
    build_context_string,
    retrieve_context,
)
from app.services.document_service import document_service


class RAGService:
    """
    Service responsible for document-grounded retrieval.

    Document lifecycle:

        SQLite
            ↓
        Document exists?
            ↓ yes
        ChromaDB retrieval
            ↓
        Relevant chunks
            ↓
        Context for Gemini

    SQLite is checked first so deleted documents cannot
    accidentally reach an empty or stale ChromaDB collection.
    """

    def get_context_for_question(
        self,
        document_id: str,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, list[SourceReference]]:
        """
        Retrieve relevant document chunks for a user question.

        Args:
            document_id:
                ID of the document to search.

            question:
                User's natural-language question.

            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            Tuple containing:

                context_string:
                    Retrieved chunks formatted for Gemini.

                sources:
                    Source metadata for the retrieved chunks.

        Raises:
            ValueError:
                If the document does not exist or contains no
                indexed content.

            RuntimeError:
                If an unexpected retrieval failure occurs.
        """

        document_id = document_id.strip()
        question = question.strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

        # ── Verify document existence in SQLite ─────────────────────────────

        document = document_service.get_document(
            document_id
        )

        if document is None:
            raise ValueError(
                f"Document '{document_id}' was not found."
            )

        # ── Retrieve document chunks from ChromaDB ──────────────────────────

        try:
            chunks = retrieve_context(
                document_id=document_id,
                question=question,
                top_k=top_k,
            )

        except ValueError:
            # Preserve document/vector-state errors so the route
            # can return an appropriate client error.
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Document retrieval failed: {exc}"
            ) from exc

        if not chunks:
            raise RuntimeError(
                "No relevant content found in the document for "
                "this question. Try rephrasing the question or "
                "ask a topic-based question instead."
            )

        context_string = build_context_string(
            chunks
        )

        if not context_string.strip():
            raise RuntimeError(
                "Relevant document chunks were found, but no "
                "usable context could be constructed."
            )

        sources = self._build_sources(
            chunks
        )

        return context_string, sources

    def get_context_for_quiz(
        self,
        document_id: str,
        topic_hint: str = "",
        top_k: int = 15,
    ) -> str:
        """
        Retrieve document context for quiz generation.

        The document remains the authoritative source for the quiz.
        topic_hint is used as the retrieval query when provided.

        Args:
            document_id:
                ID of the source document.

            topic_hint:
                Optional topic used to focus retrieval.

            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            Formatted document context.

        Raises:
            ValueError:
                If the document does not exist.

            RuntimeError:
                If retrieval fails or no usable content is found.
        """

        document_id = document_id.strip()
        topic_hint = topic_hint.strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if not isinstance(top_k, int) or top_k < 1 or top_k > 30:
            raise ValueError(
                "top_k must be between 1 and 30."
            )

        # ── Verify document existence in SQLite ─────────────────────────────

        t0 = time.time()
        document = document_service.get_document(
            document_id
        )
        lookup_time = time.time() - t0

        if document is None:
            raise ValueError(
                f"Document '{document_id}' was not found."
            )

        query = (
            topic_hint
            if topic_hint
            else "key concepts, definitions, and important facts"
        )

        try:
            t1 = time.time()
            chunks = retrieve_context(
                document_id=document_id,
                question=query,
                top_k=top_k,
            )
            rag_time = time.time() - t1

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Document retrieval failed: {exc}"
            ) from exc

        if not chunks:
            raise RuntimeError(
                "Could not retrieve relevant content from the "
                "document for quiz generation."
            )

        context_string = build_context_string(
            chunks
        )

        if not context_string.strip():
            raise RuntimeError(
                "Retrieved document chunks contained no usable "
                "content for quiz generation."
            )

        logger.info(f"[RAG] Document lookup: {lookup_time:.3f}s | RAG retrieval: {rag_time:.3f}s")

        return context_string

    @staticmethod
    def _build_sources(
        chunks: list[dict],
    ) -> list[SourceReference]:
        """
        Convert retrieved chunk metadata into unique source references.

        Multiple chunks from the same document/page are represented
        by one source reference.
        """

        sources: list[SourceReference] = []

        seen: set[tuple[str, str]] = set()

        for chunk in chunks:
            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            source = str(
                metadata.get(
                    "source",
                    "unknown",
                )
                or "unknown"
            )

            page = metadata.get(
                "page"
            )

            page_key = (
                str(page)
                if page is not None
                else ""
            )

            source_key = (
                source,
                page_key,
            )

            if source_key in seen:
                continue

            seen.add(
                source_key
            )

            sources.append(
                SourceReference(
                    document=source,
                    page=page,
                )
            )

        return sources


# Module-level singleton.
rag_service = RAGService()