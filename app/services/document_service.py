"""
BodhaQ Document Service.

Responsibilities:
- Load PDF, PPTX, and DOCX documents.
- Extract readable content.
- Split content into chunks.
- Generate embeddings.
- Store vectors in session-isolated ChromaDB.
- Store document metadata in session-scoped SQLite.
- List documents belonging to the current session.
- Retrieve document metadata belonging to the current session.
- Delete documents and their vector data.

Persistence:
    SQLite via the built-in sqlite3 module.

Vector storage:
    ChromaDB via the vector_store module.

Security:
    - Gemini API keys are request-scoped.
    - API keys are never persisted.
    - API keys are never logged.
    - Documents are isolated by anonymous session.
    - Cross-session document access is rejected.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from pathlib import Path

from app.config import DB_PATH

from app.ingestion.chunker import chunk_pages
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.pptx_loader import load_pptx
from app.models.responses import DocumentListItem
from app.rag.embeddings import embed_documents
from app.rag.vector_store import (
    delete_collection,
    upsert_chunks,
)


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".docx",
}

DOCUMENT_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F-]{36}$"
)

SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

SQLITE_BUSY_TIMEOUT_SECONDS = 10


# ============================================================================
# DATABASE
# ============================================================================


def _get_db() -> sqlite3.Connection:
    """
    Return a SQLite connection and ensure the database schema exists.

    SQLite is configured for:
        - WAL journaling
        - foreign-key enforcement
        - busy timeout

    The documents table is migrated from the previous MVP schema when
    necessary.

    Existing legacy documents receive a NULL session_id and therefore
    cannot be accessed through session-scoped operations. They are not
    silently assigned to a new session.
    """

    db_path = Path(DB_PATH)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        str(db_path),
        timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        f"PRAGMA busy_timeout = "
        f"{SQLITE_BUSY_TIMEOUT_SECONDS * 1000}"
    )

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.execute(
        "PRAGMA journal_mode = WAL"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            session_id TEXT,
            filename TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now'))
        )
        """
    )

    # ------------------------------------------------------------------------
    # SAFE MIGRATION FROM THE PREVIOUS SCHEMA
    # ------------------------------------------------------------------------
    #
    # Older BodhaQ databases did not have session_id.
    #
    # SQLite does not support ADD COLUMN IF NOT EXISTS consistently across
    # all supported versions, so inspect the schema first.
    #

    columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()
    }

    if "session_id" not in columns:

        conn.execute(
            """
            ALTER TABLE documents
            ADD COLUMN session_id TEXT
            """
        )

    # ------------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------------

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_documents_session_created
        ON documents (
            session_id,
            created_at DESC
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_documents_session_document
        ON documents (
            session_id,
            document_id
        )
        """
    )

    conn.commit()

    return conn


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _normalize_filename(
    filename: str,
) -> str:
    """
    Normalize an original document filename.

    This does not alter the actual temporary file path. It only normalizes
    the metadata stored in SQLite and passed to the ingestion pipeline.
    """

    if not isinstance(
        filename,
        str,
    ):
        raise ValueError(
            "Original filename must be a string."
        )

    normalized = filename.strip()

    if not normalized:
        raise ValueError(
            "Original filename cannot be empty."
        )

    if "\x00" in normalized:
        raise ValueError(
            "Original filename contains an invalid character."
        )

    return normalized


def _validate_session_id(
    session_id: str,
) -> str:
    """
    Validate an anonymous BodhaQ session UUID.

    The signed session token is validated at the FastAPI dependency layer.
    This additional validation protects the service if called directly.
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

    if not SESSION_ID_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Invalid session ID."
        )

    return normalized


def _validate_document_id(
    document_id: str,
) -> str:
    """
    Validate a document UUID.

    Document IDs are generated by this service, so requiring UUID format
    prevents malformed identifiers from reaching persistence layers.
    """

    if not isinstance(
        document_id,
        str,
    ):
        raise ValueError(
            "Document ID must be a string."
        )

    normalized = document_id.strip()

    if not DOCUMENT_ID_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Invalid document ID."
        )

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
            "Gemini API key cannot be empty."
        )

    normalized = api_key.strip()

    if not normalized:
        raise ValueError(
            "Gemini API key cannot be empty."
        )

    return normalized


# ============================================================================
# SERVICE
# ============================================================================


class DocumentService:
    """
    Service responsible for document ingestion and metadata management.

    Every document belongs to exactly one anonymous session.
    """

    # ========================================================================
    # INGESTION
    # ========================================================================

    def ingest(
        self,
        session_id: str,
        file_path: str,
        original_filename: str,
        api_key: str,
    ) -> dict:
        """
        Run the complete document ingestion pipeline.

        Pipeline:

            Session
              ↓
            Load
              ↓
            Chunk
              ↓
            Embed
              ↓
            Store in session-isolated ChromaDB
              ↓
            Store session-scoped metadata in SQLite

        Args:
            session_id:
                Anonymous BodhaQ session identifier.

            file_path:
                Path to the temporarily saved document.

            original_filename:
                Original filename supplied by the user.

            api_key:
                Request-scoped Gemini API key used only for
                embedding generation.

        Returns:
            Dictionary containing:

                document_id
                filename
                chunk_count

        Security:
            The API key is used only for this request and is never
            stored in SQLite, ChromaDB, logs, or the response.
        """

        # --------------------------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------------------------

        normalized_session_id = _validate_session_id(
            session_id
        )

        if not isinstance(
            file_path,
            str,
        ) or not file_path.strip():
            raise ValueError(
                "Document file path cannot be empty."
            )

        normalized_filename = _normalize_filename(
            original_filename
        )

        normalized_api_key = _validate_api_key(
            api_key
        )

        source_path = Path(
            file_path
        ).resolve()

        if not source_path.is_file():
            raise ValueError(
                "The uploaded document could not be found."
            )

        # Use the supplied original filename to determine
        # the document type.
        ext = Path(
            normalized_filename
        ).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(
                sorted(
                    SUPPORTED_EXTENSIONS
                )
            )

            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {supported}"
            )

        # --------------------------------------------------------------------
        # 1. EXTRACT TEXT
        # --------------------------------------------------------------------

        pages = self._load_pages(
            str(source_path),
            ext,
        )

        if not pages:
            raise ValueError(
                "No readable content could be extracted from the document."
            )

        # --------------------------------------------------------------------
        # 2. GENERATE DOCUMENT ID
        # --------------------------------------------------------------------

        document_id = str(
            uuid.uuid4()
        )

        # --------------------------------------------------------------------
        # 3. CHUNK DOCUMENT
        # --------------------------------------------------------------------

        chunks = chunk_pages(
            pages,
            normalized_filename,
            document_id,
        )

        if not chunks:
            raise ValueError(
                "No text chunks could be extracted from the document."
            )

        # Make sure every chunk contains usable text and the expected
        # metadata structure.
        valid_chunks: list[dict] = []

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            valid_chunks.append(
                {
                    **chunk,
                    "text": text,
                    "metadata": metadata,
                }
            )

        if not valid_chunks:
            raise ValueError(
                "No usable text was found in the document chunks."
            )

        chunks = valid_chunks

        # --------------------------------------------------------------------
        # 4. GENERATE EMBEDDINGS
        # --------------------------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embed_documents(
            texts,
            api_key=normalized_api_key,
        )

        if not embeddings:
            raise RuntimeError(
                "No embeddings were generated for the document."
            )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # --------------------------------------------------------------------
        # 5. STORE VECTOR DATA
        # --------------------------------------------------------------------

        vector_stored = False

        try:

            upsert_chunks(
                session_id=normalized_session_id,
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
            )

            vector_stored = True

            # ---------------------------------------------------------------
            # 6. STORE SQLITE METADATA
            # ---------------------------------------------------------------

            with _get_db() as conn:

                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        session_id,
                        filename,
                        chunk_count
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        normalized_session_id,
                        normalized_filename,
                        len(chunks),
                    ),
                )

        except Exception:

            # ---------------------------------------------------------------
            # ROLLBACK VECTOR DATA IF SQLITE PERSISTENCE FAILS
            # ---------------------------------------------------------------

            if vector_stored:

                try:

                    delete_collection(
                        session_id=normalized_session_id,
                        document_id=document_id,
                    )

                except Exception:
                    logger.exception(
                        "[DocumentService] Failed to clean up "
                        "vector data after metadata persistence failure."
                    )

            raise

        logger.info(
            "[DocumentService] Document ingested successfully: "
            "session_id=%s document_id=%s filename=%s chunks=%d",
            normalized_session_id,
            document_id,
            normalized_filename,
            len(chunks),
        )

        return {
            "document_id": document_id,
            "filename": normalized_filename,
            "chunk_count": len(chunks),
        }

    # ========================================================================
    # LIST
    # ========================================================================

    def list_documents(
        self,
        session_id: str,
    ) -> list[DocumentListItem]:
        """
        Return all documents belonging to the current session.

        Legacy documents with NULL session_id are intentionally excluded.
        """

        normalized_session_id = _validate_session_id(
            session_id
        )

        with _get_db() as conn:

            rows = conn.execute(
                """
                SELECT
                    document_id,
                    filename,
                    chunk_count
                FROM documents
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (
                    normalized_session_id,
                ),
            ).fetchall()

        return [
            DocumentListItem(
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_count=row["chunk_count"],
            )
            for row in rows
        ]

    # ========================================================================
    # GET
    # ========================================================================

    def get_document(
        self,
        session_id: str,
        document_id: str,
    ) -> DocumentListItem | None:
        """
        Return a document's metadata only when it belongs to the session.

        Returns:
            DocumentListItem when the document exists and belongs
            to the session.

            None when the document does not exist or belongs to
            another session.
        """

        try:
            normalized_session_id = _validate_session_id(
                session_id
            )
        except ValueError:
            return None

        if not isinstance(
            document_id,
            str,
        ):
            return None

        normalized_id = document_id.strip()

        if not normalized_id:
            return None

        if not DOCUMENT_ID_PATTERN.fullmatch(
            normalized_id
        ):
            return None

        with _get_db() as conn:

            row = conn.execute(
                """
                SELECT
                    document_id,
                    filename,
                    chunk_count
                FROM documents
                WHERE session_id = ?
                  AND document_id = ?
                """,
                (
                    normalized_session_id,
                    normalized_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return DocumentListItem(
            document_id=row["document_id"],
            filename=row["filename"],
            chunk_count=row["chunk_count"],
        )

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete_document(
        self,
        session_id: str,
        document_id: str,
    ) -> bool:
        """
        Delete a session-owned document's vector data and SQLite metadata.

        Vector data is deleted first.

        If vector deletion fails:
            SQLite metadata is preserved.

        A document belonging to another session is treated as not found.

        Returns:
            True if the document existed and was deleted.
            False if the document did not exist for the session.
        """

        try:
            normalized_session_id = _validate_session_id(
                session_id
            )
        except ValueError:
            return False

        if not isinstance(
            document_id,
            str,
        ):
            return False

        normalized_id = document_id.strip()

        if not normalized_id:
            return False

        if not DOCUMENT_ID_PATTERN.fullmatch(
            normalized_id
        ):
            return False

        with _get_db() as conn:

            existing = conn.execute(
                """
                SELECT document_id
                FROM documents
                WHERE session_id = ?
                  AND document_id = ?
                """,
                (
                    normalized_session_id,
                    normalized_id,
                ),
            ).fetchone()

            if existing is None:
                return False

            # ---------------------------------------------------------------
            # DELETE VECTOR DATA FIRST
            # ---------------------------------------------------------------

            delete_collection(
                session_id=normalized_session_id,
                document_id=normalized_id,
            )

            # ---------------------------------------------------------------
            # DELETE SQLITE METADATA
            # ---------------------------------------------------------------

            conn.execute(
                """
                DELETE FROM documents
                WHERE session_id = ?
                  AND document_id = ?
                """,
                (
                    normalized_session_id,
                    normalized_id,
                ),
            )

        logger.info(
            "[DocumentService] Document deleted successfully: "
            "session_id=%s document_id=%s",
            normalized_session_id,
            normalized_id,
        )

        return True

    # ========================================================================
    # LOADERS
    # ========================================================================

    def _load_pages(
        self,
        file_path: str,
        ext: str,
    ) -> list[dict]:
        """
        Load document content using the appropriate file loader.
        """

        if ext == ".pdf":
            return load_pdf(file_path)

        if ext == ".pptx":
            return load_pptx(file_path)

        if ext == ".docx":
            return load_docx(file_path)

        raise ValueError(
            f"Unhandled file extension: {ext}"
        )


# ============================================================================
# SINGLETON
# ============================================================================

document_service = DocumentService()