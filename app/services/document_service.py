"""
DocumentService — manages document ingestion, metadata persistence,
and document listing/deletion.

Persistence:
    SQLite via the built-in sqlite3 module.

Vector storage:
    ChromaDB via the vector_store module.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.pptx_loader import load_pptx
from app.models.responses import DocumentListItem
from app.rag.embeddings import embed_documents
from app.rag.vector_store import delete_collection, upsert_chunks


DB_PATH = "data/bodhaq.db"
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx"}


def _get_db() -> sqlite3.Connection:
    """Return a SQLite connection and ensure the database schema exists."""

    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    conn.commit()

    return conn


class DocumentService:
    """Service responsible for document ingestion and metadata management."""

    def ingest(
        self,
        file_path: str,
        original_filename: str,
    ) -> dict:
        """
        Run the complete document ingestion pipeline.

        Pipeline:
            Load
              ↓
            Chunk
              ↓
            Embed
              ↓
            Store in ChromaDB
              ↓
            Store metadata in SQLite

        Args:
            file_path:
                Path to the temporarily saved document.

            original_filename:
                Original filename supplied by the user.

        Returns:
            Dictionary containing:
                document_id
                filename
                chunk_count
        """

        ext = Path(original_filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {supported}"
            )

        # 1. Extract text from the document.
        pages = self._load_pages(file_path, ext)

        if not pages:
            raise ValueError(
                "No readable content could be extracted from the document."
            )

        # 2. Generate a unique document ID.
        document_id = str(uuid.uuid4())

        # 3. Split extracted content into chunks.
        chunks = chunk_pages(
            pages,
            original_filename,
            document_id,
        )

        if not chunks:
            raise ValueError(
                "No text chunks could be extracted from the document."
            )

        # 4. Generate embeddings for all chunks.
        texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_documents(texts)

        if not embeddings:
            raise RuntimeError(
                "No embeddings were generated for the document."
            )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # 5. Store chunks and embeddings in ChromaDB.
        upsert_chunks(
            document_id,
            chunks,
            embeddings,
        )

        # 6. Persist document metadata in SQLite.
        try:
            with _get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        filename,
                        chunk_count
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        document_id,
                        original_filename,
                        len(chunks),
                    ),
                )
        except Exception:
            # Avoid leaving orphaned vector data if metadata
            # persistence fails.
            try:
                delete_collection(document_id)
            except Exception:
                pass

            raise

        return {
            "document_id": document_id,
            "filename": original_filename,
            "chunk_count": len(chunks),
        }

    def list_documents(self) -> list[DocumentListItem]:
        """Return all ingested documents."""

        with _get_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    document_id,
                    filename,
                    chunk_count
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            DocumentListItem(
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_count=row["chunk_count"],
            )
            for row in rows
        ]

    def get_document(
        self,
        document_id: str,
    ) -> DocumentListItem | None:
        """Return a document's metadata, or None if it does not exist."""

        with _get_db() as conn:
            row = conn.execute(
                """
                SELECT
                    document_id,
                    filename,
                    chunk_count
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentListItem(
            document_id=row["document_id"],
            filename=row["filename"],
            chunk_count=row["chunk_count"],
        )

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document's vector data and SQLite metadata.

        Returns:
            True if the document existed and was deleted.
            False if the document did not exist.
        """

        with _get_db() as conn:
            existing = conn.execute(
                """
                SELECT document_id
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if existing is None:
                return False

            # Delete vector data first. If this fails, keep the
            # SQLite metadata so the document is not falsely
            # reported as completely deleted.
            delete_collection(document_id)

            conn.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            )

        return True

    def _load_pages(
        self,
        file_path: str,
        ext: str,
    ) -> list[dict]:
        """Load document content using the appropriate file loader."""

        if ext == ".pdf":
            return load_pdf(file_path)

        if ext == ".pptx":
            return load_pptx(file_path)

        if ext == ".docx":
            return load_docx(file_path)

        raise ValueError(
            f"Unhandled file extension: {ext}"
        )


# Module-level singleton.
document_service = DocumentService()