"""
Document routes.

Endpoints:
    POST /api/documents/upload
        Upload and ingest a PDF, PPTX, or DOCX.

    GET /api/documents
        List documents belonging to the current anonymous session.

    DELETE /api/documents/{document_id}
        Delete a session-owned document and its vector data.

Security:
    - Every operation requires a valid BodhaQ anonymous session.
    - Uploaded files are streamed to disk.
    - Files are limited to 50 MB.
    - Temporary filenames are generated server-side.
    - Original filenames are never used as filesystem paths.
    - Gemini API keys are request-scoped and are never logged or persisted.
    - Cross-session document access is rejected by the service layer.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from app.config import UPLOADS_DIR
from app.dependencies.session import get_session_id
from app.models.responses import (
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_service import document_service


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".docx",
}


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


# ============================================================================
# HELPERS
# ============================================================================


def _get_gemini_api_key(
    api_key: str | None,
) -> str:
    """
    Validate the request-scoped Gemini API key.

    The key is intentionally never logged, persisted, or included
    in an error response.
    """

    if not isinstance(
        api_key,
        str,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Gemini API key is required.",
                "code": "GEMINI_API_KEY_REQUIRED",
            },
        )

    normalized = api_key.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Gemini API key is required.",
                "code": "GEMINI_API_KEY_REQUIRED",
            },
        )

    return normalized


def _normalize_filename(
    filename: str | None,
) -> tuple[str, str]:
    """
    Validate and normalize an uploaded filename.

    Returns:
        (
            safe_display_filename,
            normalized_extension,
        )
    """

    if not isinstance(
        filename,
        str,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "A valid filename is required.",
                "code": "INVALID_FILENAME",
            },
        )

    normalized = filename.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "A valid filename is required.",
                "code": "INVALID_FILENAME",
            },
        )

    # Only retain the final filename component.
    #
    # The uploaded filename is metadata, not a filesystem path.
    safe_filename = Path(
        normalized
    ).name

    if not safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "A valid filename is required.",
                "code": "INVALID_FILENAME",
            },
        )

    extension = Path(
        safe_filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    f"Unsupported file type "
                    f"'{extension or 'unknown'}'. "
                    "Allowed: PDF, PPTX, DOCX."
                ),
                "code": "UNSUPPORTED_FILE_TYPE",
            },
        )

    return (
        safe_filename,
        extension,
    )


def _create_temporary_path(
    extension: str,
) -> Path:
    """
    Create a server-generated temporary upload path.

    The original filename is never used to construct this path.
    """

    upload_directory = Path(
        UPLOADS_DIR
    ).resolve()

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not upload_directory.is_dir():
        raise OSError(
            "Configured upload path is not a directory."
        )

    temporary_name = (
        f"{uuid.uuid4().hex}{extension}"
    )

    temporary_path = (
        upload_directory
        / temporary_name
    ).resolve()

    # Defensive containment check.
    if (
        temporary_path.parent
        != upload_directory
    ):
        raise OSError(
            "Generated temporary upload path is invalid."
        )

    return temporary_path


# ============================================================================
# UPLOAD
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Upload a study document.

    Supported formats:
        - PDF
        - PPTX
        - DOCX

    Processing pipeline:

        Upload
            ↓
        Temporary file
            ↓
        DocumentService
            ↓
        Text extraction
            ↓
        Chunking
            ↓
        Embeddings
            ↓
        Session-isolated ChromaDB + SQLite

    The returned document_id is used by downstream RAG endpoints.

    Gemini embedding requests use the request-scoped
    X-Gemini-API-Key header.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    filename, extension = _normalize_filename(
        file.filename
    )

    temp_path: Path | None = None

    total_size = 0

    try:

        # --------------------------------------------------------------------
        # TEMPORARY FILE
        # --------------------------------------------------------------------

        temp_path = _create_temporary_path(
            extension
        )

        # --------------------------------------------------------------------
        # STREAM FILE TO DISK
        # --------------------------------------------------------------------

        with temp_path.open(
            "wb"
        ) as destination:

            while True:

                chunk = await file.read(
                    UPLOAD_CHUNK_SIZE_BYTES
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_FILE_SIZE_BYTES
                ):
                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail={
                            "error": (
                                "File exceeds the "
                                "50 MB size limit."
                            ),
                            "code": "FILE_TOO_LARGE",
                        },
                    )

                destination.write(
                    chunk
                )

        # --------------------------------------------------------------------
        # EMPTY FILE VALIDATION
        # --------------------------------------------------------------------

        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "The uploaded file is empty.",
                    "code": "EMPTY_FILE",
                },
            )

        # --------------------------------------------------------------------
        # DOCUMENT INGESTION
        # --------------------------------------------------------------------

        result = document_service.ingest(
            session_id=session_id,
            file_path=str(temp_path),
            original_filename=filename,
            api_key=api_key,
        )

    except HTTPException:
        raise

    except ValueError as exc:

        logger.warning(
            "[Document Upload] Invalid document: %s",
            exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "error": str(exc),
                "code": "DOCUMENT_PROCESSING_ERROR",
            },
        ) from exc

    except RuntimeError as exc:

        logger.exception(
            "[Document Upload] Document service failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "error": (
                    "Document processing service "
                    "is currently unavailable."
                ),
                "code": "DOCUMENT_PROCESSING_UNAVAILABLE",
            },
        ) from exc

    except OSError as exc:

        logger.exception(
            "[Document Upload] File system error."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "error": (
                    "Unable to temporarily store "
                    "the uploaded document."
                ),
                "code": "DOCUMENT_STORAGE_ERROR",
            },
        ) from exc

    except Exception:

        logger.exception(
            "[Document Upload] Unexpected error."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "error": (
                    "Unexpected error while "
                    "processing the document."
                ),
                "code": "DOCUMENT_UPLOAD_ERROR",
            },
        )

    finally:

        await file.close()

        # --------------------------------------------------------------------
        # ALWAYS REMOVE TEMPORARY FILE
        # --------------------------------------------------------------------

        if temp_path is not None:

            try:

                if temp_path.exists():
                    temp_path.unlink()

            except OSError:

                logger.warning(
                    "[Document Upload] Could not remove "
                    "temporary upload file."
                )

    # ------------------------------------------------------------------------
    # BUILD RESPONSE
    # ------------------------------------------------------------------------

    return DocumentUploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        message=(
            "Document ingested successfully. "
            f"{result['chunk_count']} chunks stored."
        ),
    )


# ============================================================================
# LIST DOCUMENTS
# ============================================================================


@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents(
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Return documents belonging to the current anonymous session.
    """

    try:

        documents = (
            document_service.list_documents(
                session_id=session_id
            )
        )

        return DocumentListResponse(
            documents=documents,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_SESSION",
            },
        ) from exc

    except Exception:

        logger.exception(
            "[Document List] Failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "error": (
                    "Unable to retrieve documents."
                ),
                "code": "DOCUMENT_LIST_ERROR",
            },
        )


# ============================================================================
# DELETE DOCUMENT
# ============================================================================


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: str,
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Delete a session-owned document and its associated vector embeddings.
    """

    normalized_document_id = (
        document_id.strip()
        if isinstance(
            document_id,
            str,
        )
        else ""
    )

    if not normalized_document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Document ID is required.",
                "code": "INVALID_DOCUMENT_ID",
            },
        )

    try:

        deleted = (
            document_service.delete_document(
                session_id=session_id,
                document_id=normalized_document_id,
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_DOCUMENT_ID",
            },
        ) from exc

    except Exception:

        logger.exception(
            "[Document Delete] Failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "error": (
                    "Unable to delete the document."
                ),
                "code": "DOCUMENT_DELETE_ERROR",
            },
        )

    if not deleted:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Document not found.",
                "code": "DOCUMENT_NOT_FOUND",
            },
        )

    return {
        "message": "Document deleted successfully."
    }