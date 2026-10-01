"""
Resume routes.

Resume Prep:

    - Upload and extract a resume.
    - Track resume preparation progress.
    - Generate interview assessments for resume items.
    - Submit resume assessments using the shared evaluation engine.

Security:

    - Gemini API keys are supplied per request.
    - API keys are never persisted or logged.
    - Resume data is isolated by anonymous BodhaQ session.
    - Deterministic resume evaluation does not require Gemini.
"""

from __future__ import annotations

import logging
from pathlib import Path
import uuid

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
from app.models.requests import (
    QuizSubmitRequest,
    ResumeQuizGenerateRequest,
)
from app.models.responses import (
    DocumentUploadResponse,
    QuizEvaluationResponse,
    QuizGenerateResponse,
    ResumeProgressResponse,
)
from app.services.evaluation_service import (
    evaluation_service,
)
from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
)
from app.services.quiz_service import quiz_service
from app.services.resume_service import resume_service


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/resume",
    tags=["Resume"],
)


# ============================================================================
# HELPERS
# ============================================================================


def _get_gemini_api_key(
    api_key: str | None,
) -> str:
    """
    Validate the request-scoped Gemini API key.

    The key is intentionally never logged, persisted, or returned.
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


def _raise_gemini_error(
    exc: Exception,
    *,
    feature: str,
) -> None:
    """
    Convert Gemini-related exceptions into safe HTTP responses.

    This function always raises HTTPException.
    """

    if isinstance(
        exc,
        GeminiAuthenticationError,
    ):
        logger.error(
            "[Resume] Gemini authentication failed. feature=%s",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Gemini authentication failed. "
                    "Please check your API key."
                ),
                "code": "AI_AUTHENTICATION_FAILED",
            },
        ) from exc

    if isinstance(
        exc,
        GeminiQuotaError,
    ):
        logger.warning(
            "[Resume] Gemini quota/rate limit reached. feature=%s",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": (
                    "AI service rate limit reached. "
                    "Please try again later."
                ),
                "code": "AI_RATE_LIMIT",
            },
        ) from exc

    if isinstance(
        exc,
        ValueError,
    ):
        logger.warning(
            "[Resume] Invalid request. feature=%s type=%s",
            feature,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_RESUME_REQUEST",
            },
        ) from exc

    if isinstance(
        exc,
        RuntimeError,
    ):
        logger.exception(
            "[Resume] Gemini service failed. feature=%s",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "AI service is currently unavailable."
                ),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    logger.exception(
        "[Resume] Unexpected Gemini failure. feature=%s",
        feature,
    )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": (
                "Unable to complete the AI operation."
            ),
            "code": "AI_SERVICE_UNAVAILABLE",
        },
    ) from exc


# ============================================================================
# RESUME UPLOAD
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Upload a resume and extract structured information from it.

    Supported formats:
        - PDF
        - DOCX

    Resume processing uses the request-scoped Gemini API key.

    The resulting resume belongs exclusively to the authenticated
    anonymous BodhaQ session.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    # ------------------------------------------------------------------------
    # FILENAME VALIDATION
    # ------------------------------------------------------------------------

    filename = (
        file.filename.strip()
        if file.filename
        else ""
    )

    if not filename:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "A valid filename is required.",
                "code": "INVALID_FILENAME",
            },
        )

    # ------------------------------------------------------------------------
    # EXTENSION VALIDATION
    # ------------------------------------------------------------------------

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    "Unsupported file type. "
                    "Please upload a PDF or DOCX file."
                ),
                "code": "UNSUPPORTED_FILE_TYPE",
            },
        )

    # ------------------------------------------------------------------------
    # PREPARE TEMPORARY UPLOAD
    # ------------------------------------------------------------------------

    upload_directory = Path(
        UPLOADS_DIR
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        upload_directory
        / f"{uuid.uuid4().hex}{extension}"
    )

    resume_id: str | None = None

    try:
        # --------------------------------------------------------------------
        # STREAM FILE TO DISK
        # --------------------------------------------------------------------

        total_size = 0

        with temp_path.open(
            "wb"
        ) as destination:

            while True:

                chunk = await file.read(
                    1024 * 1024
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
                                "File exceeds the maximum "
                                "size limit of 50 MB."
                            ),
                            "code": "FILE_TOO_LARGE",
                        },
                    )

                destination.write(
                    chunk
                )

        # --------------------------------------------------------------------
        # RESUME INGESTION
        # --------------------------------------------------------------------

        logger.info(
            "[Resume] Processing uploaded resume."
        )

        resume_id = (
            resume_service.ingest_resume(
                session_id=session_id,
                file_path=str(temp_path),
                original_filename=filename,
                api_key=api_key,
            )
        )

        logger.info(
            "[Resume] Resume processed successfully."
        )

    except HTTPException:
        raise

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:

        _raise_gemini_error(
            exc,
            feature="resume upload",
        )

    except OSError as exc:

        logger.exception(
            "[Resume] File system error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Unable to temporarily store "
                    "the uploaded resume."
                ),
                "code": "RESUME_STORAGE_ERROR",
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "[Resume] Unexpected resume processing error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Failed to process the resume. "
                    "Please try again."
                ),
                "code": "RESUME_UPLOAD_ERROR",
            },
        ) from exc

    finally:
        await file.close()

        # --------------------------------------------------------------------
        # ALWAYS REMOVE TEMPORARY FILE
        # --------------------------------------------------------------------

        try:
            if temp_path.exists():
                temp_path.unlink()

        except OSError:
            logger.warning(
                "[Resume] Could not remove temporary file."
            )

    # ------------------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------------------

    if not resume_id:
        logger.error(
            "[Resume] Ingestion returned no resume ID."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Resume processing completed without "
                    "creating a resume record."
                ),
                "code": "RESUME_ID_MISSING",
            },
        )

    return DocumentUploadResponse(
        document_id=resume_id,
        filename=filename,
        chunk_count=0,
        message="Resume processed successfully.",
    )


# ============================================================================
# RESUME PROGRESS
# ============================================================================


@router.get(
    "/progress",
    response_model=ResumeProgressResponse,
)
def get_resume_progress(
    session_id: str = Depends(get_session_id),
):
    """
    Return the latest resume belonging to the current session
    and its preparation progress.
    """

    try:
        progress = (
            resume_service.get_latest_resume_progress(
                session_id=session_id,
            )
        )

    except ValueError as exc:

        logger.warning(
            "[Resume] Invalid session/progress request: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_RESUME_PROGRESS_REQUEST",
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "[Resume] Failed to load resume progress."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Failed to load resume progress."
                ),
                "code": "RESUME_PROGRESS_ERROR",
            },
        ) from exc

    if not progress:
        return ResumeProgressResponse(
            resume_id=None,
            filename=None,
        )

    return ResumeProgressResponse(
        **progress
    )


# ============================================================================
# GENERATE RESUME INTERVIEW QUIZ
# ============================================================================


@router.post(
    "/quiz/generate",
    response_model=QuizGenerateResponse,
)
def generate_resume_quiz(
    request: ResumeQuizGenerateRequest,
    session_id: str = Depends(get_session_id),
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Generate an interview assessment for a resume item.

    The resume item is already stored by the backend, so the frontend
    only provides the item ID and quiz configuration.

    Gemini uses the request-scoped API key supplied by the user.

    The resume item and generated quiz must belong to the current
    anonymous BodhaQ session.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        logger.info(
            "[Resume] Generating assessment | "
            "difficulty=%s | questions=%s",
            request.difficulty,
            request.number_of_questions,
        )

        return (
            resume_service.generate_quiz_for_item(
                session_id=session_id,
                item_id=request.item_id,
                difficulty=request.difficulty,
                num_questions=request.number_of_questions,
                api_key=api_key,
            )
        )

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:

        _raise_gemini_error(
            exc,
            feature="resume quiz",
        )

    except Exception as exc:

        logger.exception(
            "[Resume] Unexpected resume quiz generation error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Unable to generate the "
                    "resume assessment."
                ),
                "code": "RESUME_QUIZ_GENERATION_FAILED",
            },
        ) from exc


# ============================================================================
# SUBMIT RESUME QUIZ
# ============================================================================


@router.post(
    "/quiz/submit",
    response_model=QuizEvaluationResponse,
)
def submit_resume_quiz(
    request: QuizSubmitRequest,
    session_id: str = Depends(get_session_id),
):
    """
    Submit a resume interview assessment.

    Uses the same deterministic evaluation engine as normal quizzes.

    If the quiz belongs to a resume item, the item's progress is
    updated automatically.

    No Gemini API key is required.
    """

    try:
        # --------------------------------------------------------------------
        # DETERMINISTIC EVALUATION
        # --------------------------------------------------------------------

        evaluation = (
            evaluation_service.evaluate(
                session_id=session_id,
                quiz_id=request.quiz_id,
                user_answers=request.answers,
            )
        )

        # --------------------------------------------------------------------
        # DETERMINE WHETHER THIS QUIZ BELONGS TO A RESUME ITEM
        # --------------------------------------------------------------------

        stored = (
            quiz_service.get_stored_quiz(
                session_id=session_id,
                quiz_id=request.quiz_id,
            )
        )

        if (
            stored
            and stored.get("source_type")
            == "resume_item"
        ):

            item_id = stored.get(
                "source_id"
            )

            if item_id:

                resume_service.mark_item_completed(
                    session_id=session_id,
                    item_id=item_id,
                    score=evaluation.score,
                    percentage=evaluation.percentage,
                )

                logger.info(
                    "[Resume] Item completion recorded | "
                    "score=%s/%s | percentage=%.2f",
                    evaluation.score,
                    evaluation.total,
                    evaluation.percentage,
                )

        return evaluation

    except ValueError as exc:

        logger.warning(
            "[Resume] Invalid quiz submission: %s",
            type(exc).__name__,
        )

        message = str(exc)

        if "not found" in message.lower():
            response_status = (
                status.HTTP_404_NOT_FOUND
            )
            error_code = "QUIZ_NOT_FOUND"

        else:
            response_status = (
                status.HTTP_400_BAD_REQUEST
            )
            error_code = "INVALID_QUIZ_SUBMISSION"

        raise HTTPException(
            status_code=response_status,
            detail={
                "error": message,
                "code": error_code,
            },
        ) from exc

    except RuntimeError as exc:

        logger.exception(
            "[Resume] Resume quiz evaluation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Resume assessment service "
                    "is currently unavailable."
                ),
                "code": "RESUME_QUIZ_EVALUATION_UNAVAILABLE",
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "[Resume] Unexpected resume quiz submission error."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Unable to submit the "
                    "resume assessment."
                ),
                "code": "RESUME_QUIZ_SUBMISSION_FAILED",
            },
        ) from exc