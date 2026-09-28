"""
Document routes.

POST /api/documents/upload  — upload and ingest a PDF, PPTX, or DOCX.
GET  /api/documents         — list all ingested documents.
DELETE /api/documents/{id}  — delete a document and its vector data.
"""

import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.responses import (
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_service import document_service


UPLOADS_DIR = "uploads"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".docx"}

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a study document (PDF, PPTX, DOCX).

    The document is extracted, chunked, embedded, and stored in ChromaDB.
    Returns a document_id used to reference this document in other API calls.
    """

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    f"Unsupported file type '{ext}'. "
                    "Allowed: PDF, PPTX, DOCX."
                ),
                "code": "UNSUPPORTED_FILE_TYPE",
            },
        )

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(
        UPLOADS_DIR,
        temp_filename,
    )

    try:
        total_size = 0

        with open(temp_path, "wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "error": "File exceeds the 50 MB size limit.",
                            "code": "FILE_TOO_LARGE",
                        },
                    )

                destination.write(chunk)

        result = document_service.ingest(
            temp_path,
            filename,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": str(exc),
                "code": "DOCUMENT_PROCESSING_ERROR",
            },
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Unexpected error while processing the document.",
                "code": "DOCUMENT_UPLOAD_ERROR",
            },
        ) from exc

    finally:
        await file.close()

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return DocumentUploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        message=(
            f"Document ingested successfully. "
            f"{result['chunk_count']} chunks stored."
        ),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents():
    """Return all uploaded and ingested documents."""

    docs = document_service.list_documents()

    return DocumentListResponse(
        documents=docs,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
)
def delete_document(document_id: str):
    """Delete a document and its associated vector embeddings."""

    deleted = document_service.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Document not found.",
                "code": "DOCUMENT_NOT_FOUND",
            },
        )

    return {
        "message": f"Document '{document_id}' deleted successfully."
    }