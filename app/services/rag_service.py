"""
BodhaQ RAG Service — bridges document retrieval and Gemini embeddings.

Responsibilities:
    - Verify that the requested document belongs to the current session.
    - Retrieve relevant chunks from ChromaDB.
    - Build context for document-grounded generation.
    - Return source references for retrieved content.

This service does NOT call Gemini directly.
GeminiService is responsible for generation.
The retriever/embedding layer uses the request-scoped Gemini API key.

SQLite is the authoritative source for document existence and ownership.
ChromaDB is the vector-storage layer.

Security:
    - Gemini API keys are request-scoped.
    - API keys are never persisted.
    - API keys are never logged.
    - Documents are isolated by anonymous session.
    - Cross-session document access is rejected.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.models.responses import SourceReference
from app.rag.retriever import (
    build_context_string,
    retrieve_context,
)
from app.services.document_service import document_service
from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_SESSION_ID_LENGTH = 100
MAX_DOCUMENT_ID_LENGTH = 200
MAX_QUESTION_LENGTH = 2000
MAX_TOPIC_HINT_LENGTH = 500
MAX_SOURCE_REFERENCES = 20


# ============================================================================
# SERVICE
# ============================================================================


class RAGService:
    """
    Service responsible for document-grounded retrieval.

    Document lifecycle:

        Session
            ↓
        SQLite
            ↓
        Document belongs to session?
            ↓ yes
        ChromaDB retrieval
            ↓
        Relevant chunks
            ↓
        Context for Gemini

    SQLite is checked first so deleted documents or documents belonging
    to another anonymous session cannot accidentally reach ChromaDB.
    """

    # ========================================================================
    # QUESTION RETRIEVAL
    # ========================================================================

    def get_context_for_question(
        self,
        session_id: str,
        document_id: str,
        question: str,
        api_key: str,
        top_k: int = 5,
    ) -> tuple[str, list[SourceReference]]:
        """
        Retrieve relevant document chunks for a user question.

        Args:
            session_id:
                Anonymous BodhaQ session identifier.

            document_id:
                ID of the document to search.

            question:
                User's natural-language question.

            api_key:
                Request-scoped Gemini API key used by the
                embedding/retrieval layer.

            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            (
                context_string,
                source_references,
            )

        Raises:
            ValueError:
                Invalid input or document does not exist for the session.

            RuntimeError:
                Retrieval failed or no usable content was found.

        Security:
            The API key is passed only to the embedding layer.
            It is never persisted, logged, or returned.
        """

        normalized_session_id = self._normalize_text(
            session_id
        )

        normalized_document_id = self._normalize_text(
            document_id
        )

        normalized_question = self._normalize_text(
            question
        )

        normalized_api_key = self._validate_api_key(
            api_key
        )

        # --------------------------------------------------------------------
        # VALIDATE INPUT
        # --------------------------------------------------------------------

        self._validate_session_id(
            normalized_session_id
        )

        self._validate_text_length(
            normalized_document_id,
            field_name="document_id",
            maximum=MAX_DOCUMENT_ID_LENGTH,
        )

        self._validate_text_length(
            normalized_question,
            field_name="question",
            maximum=MAX_QUESTION_LENGTH,
        )

        if not normalized_document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if not normalized_question:
            raise ValueError(
                "question cannot be empty."
            )

        self._validate_top_k(
            top_k=top_k,
            maximum=20,
        )

        # --------------------------------------------------------------------
        # VERIFY DOCUMENT IN SQLITE
        # --------------------------------------------------------------------

        document = document_service.get_document(
            session_id=normalized_session_id,
            document_id=normalized_document_id,
        )

        if document is None:
            raise ValueError(
                f"Document '{normalized_document_id}' was not found."
            )

        # --------------------------------------------------------------------
        # RETRIEVE FROM CHROMADB
        # --------------------------------------------------------------------

        retrieval_start = time.perf_counter()

        try:

            chunks = retrieve_context(
                session_id=normalized_session_id,
                document_id=normalized_document_id,
                question=normalized_question,
                api_key=normalized_api_key,
                top_k=top_k,
            )

        except (
            GeminiAuthenticationError,
            GeminiQuotaError,
        ):
            # Preserve AI authentication/rate-limit errors so the route
            # can return the appropriate status code.
            raise

        except ValueError:
            # Preserve known RAG/document validation errors.
            raise

        except Exception as exc:

            logger.exception(
                "[RAG] Question retrieval failed | "
                "session_id=%s | document_id=%s",
                normalized_session_id,
                normalized_document_id,
            )

            raise RuntimeError(
                "Document retrieval failed."
            ) from exc

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # --------------------------------------------------------------------
        # VALIDATE RETRIEVAL RESULT
        # --------------------------------------------------------------------

        if not isinstance(
            chunks,
            list,
        ):
            raise RuntimeError(
                "Document retrieval returned an invalid result."
            )

        if not chunks:
            raise RuntimeError(
                "No relevant content was found in the document "
                "for this question. Try rephrasing the question "
                "or ask a topic-based question instead."
            )

        context_string = build_context_string(
            chunks
        )

        if not isinstance(
            context_string,
            str,
        ) or not context_string.strip():
            raise RuntimeError(
                "Relevant document chunks were found, but no "
                "usable context could be constructed."
            )

        # --------------------------------------------------------------------
        # BUILD SOURCES
        # --------------------------------------------------------------------

        sources = self._build_sources(
            chunks
        )

        logger.info(
            "[RAG] Question retrieval | "
            "session_id=%s | document_id=%s | "
            "chunks=%d | sources=%d | time=%.3fs",
            normalized_session_id,
            normalized_document_id,
            len(chunks),
            len(sources),
            retrieval_time,
        )

        return (
            context_string,
            sources,
        )

    # ========================================================================
    # QUIZ RETRIEVAL
    # ========================================================================

    def get_context_for_quiz(
        self,
        session_id: str,
        document_id: str,
        api_key: str,
        topic_hint: str = "",
        top_k: int = 15,
    ) -> str:
        """
        Retrieve document context for quiz generation.

        The document remains the authoritative source for the quiz.

        Args:
            session_id:
                Anonymous BodhaQ session identifier.

            document_id:
                ID of the source document.

            api_key:
                Request-scoped Gemini API key used by the
                embedding/retrieval layer.

            topic_hint:
                Optional topic used to focus semantic retrieval.

            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            Formatted document context.

        Raises:
            ValueError:
                Invalid input or document does not exist for the session.

            RuntimeError:
                Retrieval failed or no usable content was found.
        """

        normalized_session_id = self._normalize_text(
            session_id
        )

        normalized_document_id = self._normalize_text(
            document_id
        )

        normalized_api_key = self._validate_api_key(
            api_key
        )

        normalized_topic_hint = self._normalize_text(
            topic_hint
        )

        # --------------------------------------------------------------------
        # VALIDATE INPUT
        # --------------------------------------------------------------------

        self._validate_session_id(
            normalized_session_id
        )

        self._validate_text_length(
            normalized_document_id,
            field_name="document_id",
            maximum=MAX_DOCUMENT_ID_LENGTH,
        )

        self._validate_text_length(
            normalized_topic_hint,
            field_name="topic_hint",
            maximum=MAX_TOPIC_HINT_LENGTH,
        )

        if not normalized_document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        self._validate_top_k(
            top_k=top_k,
            maximum=30,
        )

        # --------------------------------------------------------------------
        # VERIFY DOCUMENT IN SQLITE
        # --------------------------------------------------------------------

        lookup_start = time.perf_counter()

        document = document_service.get_document(
            session_id=normalized_session_id,
            document_id=normalized_document_id,
        )

        lookup_time = (
            time.perf_counter()
            - lookup_start
        )

        if document is None:
            raise ValueError(
                f"Document '{normalized_document_id}' was not found."
            )

        # --------------------------------------------------------------------
        # BUILD RETRIEVAL QUERY
        # --------------------------------------------------------------------

        query = (
            normalized_topic_hint
            if normalized_topic_hint
            else "key concepts, definitions, and important facts"
        )

        # --------------------------------------------------------------------
        # RETRIEVE FROM CHROMADB
        # --------------------------------------------------------------------

        retrieval_start = time.perf_counter()

        try:

            chunks = retrieve_context(
                session_id=normalized_session_id,
                document_id=normalized_document_id,
                question=query,
                api_key=normalized_api_key,
                top_k=top_k,
            )

        except (
            GeminiAuthenticationError,
            GeminiQuotaError,
        ):
            # Preserve AI authentication/rate-limit errors.
            raise

        except ValueError:
            raise

        except Exception as exc:

            logger.exception(
                "[RAG] Quiz retrieval failed | "
                "session_id=%s | document_id=%s",
                normalized_session_id,
                normalized_document_id,
            )

            raise RuntimeError(
                "Document retrieval failed."
            ) from exc

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # --------------------------------------------------------------------
        # VALIDATE RETRIEVAL RESULT
        # --------------------------------------------------------------------

        if not isinstance(
            chunks,
            list,
        ):
            raise RuntimeError(
                "Document retrieval returned an invalid result."
            )

        if not chunks:
            raise RuntimeError(
                "Could not retrieve relevant content from the "
                "document for quiz generation."
            )

        context_string = build_context_string(
            chunks
        )

        if not isinstance(
            context_string,
            str,
        ) or not context_string.strip():
            raise RuntimeError(
                "Retrieved document chunks contained no usable "
                "content for quiz generation."
            )

        logger.info(
            "[RAG] Quiz retrieval | "
            "session_id=%s | document_id=%s | "
            "chunks=%d | lookup=%.3fs | retrieval=%.3fs",
            normalized_session_id,
            normalized_document_id,
            len(chunks),
            lookup_time,
            retrieval_time,
        )

        return context_string

    # ========================================================================
    # SOURCE REFERENCES
    # ========================================================================

    @staticmethod
    def _build_sources(
        chunks: list[dict[str, Any]],
    ) -> list[SourceReference]:
        """
        Convert retrieved chunk metadata into unique source references.

        Multiple chunks from the same document/page are represented
        by one source reference.
        """

        sources: list[
            SourceReference
        ] = []

        seen: set[
            tuple[str, str]
        ] = set()

        if not isinstance(
            chunks,
            list,
        ):
            return sources

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            source = str(
                metadata.get(
                    "source",
                    "",
                )
                or ""
            ).strip()

            if not source:
                source = "unknown"

            page = metadata.get(
                "page"
            )

            # Page numbers should normally be integers from the ingestion
            # pipeline. Preserve valid values and avoid exposing arbitrary
            # malformed metadata.
            if isinstance(
                page,
                bool,
            ):
                page = None

            elif isinstance(
                page,
                int,
            ):
                if page < 1:
                    page = None

            elif page is not None:
                try:
                    page = int(
                        str(page).strip()
                    )

                    if page < 1:
                        page = None

                except (
                    TypeError,
                    ValueError,
                ):
                    page = None

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

            if len(sources) >= MAX_SOURCE_REFERENCES:
                break

        return sources

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> None:
        """
        Validate the anonymous session identifier.

        The session token itself is validated by the FastAPI dependency.
        This additional validation protects the service if it is called
        directly from another backend component.
        """

        if not session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        if len(session_id) > MAX_SESSION_ID_LENGTH:
            raise ValueError(
                "session_id exceeds the maximum allowed length."
            )

        try:
            uuid.UUID(
                session_id
            )
        except (
            ValueError,
            AttributeError,
            TypeError,
        ) as exc:
            raise ValueError(
                "Invalid session_id."
            ) from exc

    @staticmethod
    def _validate_api_key(
        api_key: Any,
    ) -> str:
        """
        Validate the request-scoped Gemini API key.

        The key is never logged or persisted.
        """

        if not isinstance(
            api_key,
            str,
        ):
            raise ValueError(
                "Gemini API key cannot be empty."
            )

        normalized = api_key.strip()

        if not normalized:
            raise ValueError(
                "Gemini API key cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_text_length(
        value: str,
        field_name: str,
        maximum: int,
    ) -> None:
        """
        Validate maximum length for text inputs.
        """

        if len(value) > maximum:
            raise ValueError(
                f"{field_name} exceeds the maximum allowed "
                f"length of {maximum} characters."
            )

    @staticmethod
    def _validate_top_k(
        top_k: int,
        maximum: int,
    ) -> None:
        """
        Validate the requested number of retrieved chunks.
        """

        # bool is a subclass of int in Python, so explicitly reject it.
        if isinstance(
            top_k,
            bool,
        ):
            raise ValueError(
                f"top_k must be between 1 and {maximum}."
            )

        if not isinstance(
            top_k,
            int,
        ):
            raise ValueError(
                f"top_k must be between 1 and {maximum}."
            )

        if top_k < 1 or top_k > maximum:
            raise ValueError(
                f"top_k must be between 1 and {maximum}."
            )

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Safely normalize a text input.
        """

        if value is None:
            return ""

        return str(
            value
        ).strip()


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

rag_service = RAGService()