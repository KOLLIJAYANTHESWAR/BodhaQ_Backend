"""
Resume routes.

Resume Prep:
    - Upload and extract a resume.
    - Track resume preparation progress.
    - Generate interview assessments for resume items.
    - Submit resume assessments using the shared evaluation engine.

Gemini authentication is handled exclusively by the backend
through GEMINI_API_KEY in backend/.env.
"""

import logging
import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import UPLOADS_DIR
from app.models.requests import (
    ResumeQuizGenerateRequest,
    QuizSubmitRequest,
)
from app.models.responses import (
    DocumentUploadResponse,
    ResumeProgressResponse,
    QuizGenerateResponse,
    QuizEvaluationResponse,
)
from app.services.evaluation_service import evaluation_service
from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
)
from app.services.resume_service import resume_service


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/resume",
    tags=["Resume"],
)


# ---------------------------------------------------------------------------
# Resume upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_resume(
    file: UploadFile = File(...),
):
    """
    Upload a resume and extract structured information from it.

    Supported formats:
        - PDF
        - DOCX

    Gemini authentication is handled by the backend.
    The frontend never sends a Gemini API key.
    """

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    # -----------------------------------------------------------------------
    # Validate extension
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Prepare temporary upload
    # -----------------------------------------------------------------------

    os.makedirs(
        UPLOADS_DIR,
        exist_ok=True,
    )

    temp_filename = f"{uuid.uuid4()}{extension}"

    temp_path = os.path.join(
        UPLOADS_DIR,
        temp_filename,
    )

    try:

        total_size = 0

        with open(
            temp_path,
            "wb",
        ) as destination:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                # -----------------------------------------------------------
                # File size protection
                # -----------------------------------------------------------

                if total_size > MAX_FILE_SIZE_BYTES:

                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail={
                            "error": (
                                "File exceeds the maximum size "
                                "limit of 50 MB."
                            ),
                            "code": "FILE_TOO_LARGE",
                        },
                    )

                destination.write(chunk)

        logger.info(
            "[Resume] Processing uploaded resume: %s",
            filename,
        )

        # -------------------------------------------------------------------
        # Resume ingestion
        #
        # IMPORTANT:
        # Do not pass an API key from the frontend.
        # resume_service uses the centralized GeminiService.
        # -------------------------------------------------------------------

        resume_id = resume_service.ingest_resume(
            temp_path,
            filename,
        )

        logger.info(
            "[Resume] Resume processed successfully: %s",
            resume_id,
        )

    except HTTPException:
        raise

    except GeminiAuthenticationError as exc:

        logger.error(
            "[Resume] Gemini authentication failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": (
                    "Gemini authentication failed. "
                    "Check GEMINI_API_KEY in backend/.env."
                ),
                "code": "INVALID_API_KEY",
            },
        ) from exc

    except GeminiQuotaError as exc:

        logger.warning(
            "[Resume] Gemini quota/rate limit reached: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": str(exc),
                "code": "AI_RATE_LIMIT",
            },
        ) from exc

    except ValueError as exc:

        logger.warning(
            "[Resume] Invalid resume: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_RESUME",
            },
        ) from exc

    except RuntimeError as exc:

        logger.exception(
            "[Resume] Resume processing failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "RESUME_AI_SERVICE_UNAVAILABLE",
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

        # -------------------------------------------------------------------
        # Always remove temporary uploaded file.
        # -------------------------------------------------------------------

        if os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:

                logger.warning(
                    "[Resume] Could not remove temporary file: %s",
                    temp_path,
                )

    return DocumentUploadResponse(
        document_id=resume_id,
        filename=filename,
        chunk_count=0,
        message="Resume processed successfully.",
    )


# ---------------------------------------------------------------------------
# Resume progress
# ---------------------------------------------------------------------------

@router.get(
    "/progress",
    response_model=ResumeProgressResponse,
)
def get_resume_progress():
    """
    Return the latest resume and its preparation progress.
    """

    try:

        progress = resume_service.get_latest_resume_progress()

    except Exception as exc:

        logger.exception(
            "[Resume] Failed to load resume progress."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to load resume progress.",
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


# ---------------------------------------------------------------------------
# Generate resume interview quiz
# ---------------------------------------------------------------------------

@router.post(
    "/quiz/generate",
    response_model=QuizGenerateResponse,
)
def generate_resume_quiz(
    request: ResumeQuizGenerateRequest,
):
    """
    Generate an interview assessment for a resume item.

    The resume item itself is already stored by the backend, so the
    frontend only needs to provide the item ID and quiz configuration.
    """

    try:

        logger.info(
            "[Resume] Generating assessment | item_id=%s | "
            "difficulty=%s | questions=%s",
            request.item_id,
            request.difficulty,
            request.number_of_questions,
        )

        return resume_service.generate_quiz_for_item(
            item_id=request.item_id,
            difficulty=request.difficulty,
            num_questions=request.number_of_questions,
        )

    except GeminiAuthenticationError as exc:

        logger.error(
            "[Resume] Gemini authentication failed during quiz generation: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": (
                    "Gemini authentication failed. "
                    "Check GEMINI_API_KEY in backend/.env."
                ),
                "code": "INVALID_API_KEY",
            },
        ) from exc

    except GeminiQuotaError as exc:

        logger.warning(
            "[Resume] Gemini quota/rate limit reached during quiz generation."
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": str(exc),
                "code": "AI_RATE_LIMIT",
            },
        ) from exc

    except ValueError as exc:

        logger.warning(
            "[Resume] Invalid resume quiz request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_RESUME_QUIZ_REQUEST",
            },
        ) from exc

    except RuntimeError as exc:

        logger.exception(
            "[Resume] Resume quiz generation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "RESUME_AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "[Resume] Unexpected resume quiz generation error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to generate the interview assessment.",
                "code": "QUIZ_GENERATE_ERROR",
            },
        ) from exc


# ---------------------------------------------------------------------------
# Submit resume quiz
# ---------------------------------------------------------------------------

@router.post(
    "/quiz/submit",
    response_model=QuizEvaluationResponse,
)
def submit_resume_quiz(
    request: QuizSubmitRequest,
):
    """
    Submit a resume interview assessment.

    Uses the same deterministic evaluation engine as normal quizzes.

    If the quiz belongs to a resume item, the item's progress is
    updated automatically.
    """

    try:

        # -------------------------------------------------------------------
        # Deterministic evaluation
        # -------------------------------------------------------------------

        evaluation = evaluation_service.evaluate(
            request.quiz_id,
            request.answers,
        )

        # -------------------------------------------------------------------
        # Determine whether this quiz belongs to a resume item.
        # -------------------------------------------------------------------

        from app.services.quiz_service import quiz_service

        stored = quiz_service.get_stored_quiz(
            request.quiz_id
        )

        if stored and stored.get("source_type") == "resume_item":

            item_id = stored.get("source_id")

            if item_id:

                resume_service.mark_item_completed(
                    item_id,
                    evaluation.score,
                    evaluation.percentage,
                )

                logger.info(
                    "[Resume] Item completed | item_id=%s | "
                    "score=%s/%s | percentage=%.2f",
                    item_id,
                    evaluation.score,
                    evaluation.total_questions,
                    evaluation.percentage,
                )

        return evaluation

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": str(exc),
                "code": "QUIZ_NOT_FOUND",
            },
        ) from exc