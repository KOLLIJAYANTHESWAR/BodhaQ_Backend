"""
Quiz routes.

POST /api/quiz/generate
    Generate a quiz from a topic or document.

POST /api/quiz/submit
    Submit answers and receive scored evaluation.

POST /api/quiz/practice
    Generate targeted practice for a weak topic.

    Supports:
        - Topic-only practice
        - Document-grounded practice using RAG

GET /api/quiz/gaps
    Analyse submitted quizzes and return aggregated learning gaps.

Correct answers are always stored server-side and are never returned
to the frontend.

Security:
    - A valid BodhaQ anonymous session is required.
    - Gemini API keys are supplied per request for AI generation.
    - API keys are never persisted or logged.
    - Quiz data and evaluations are isolated by session.
    - Quiz submission and learning-gap analysis do not require Gemini.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    status,
)

from app.dependencies.session import get_session_id
from app.models.requests import (
    PracticeRequest,
    QuizGenerateRequest,
    QuizSubmitRequest,
)
from app.models.responses import (
    QuizEvaluationResponse,
    QuizGenerateResponse,
    WeakTopicsResponse,
)
from app.services.evaluation_service import evaluation_service
from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
)
from app.services.quiz_service import quiz_service


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/quiz",
    tags=["Quiz"],
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


def _raise_generation_error(
    exc: Exception,
    *,
    feature: str,
) -> None:
    """
    Convert AI-generation exceptions into safe HTTP responses.

    This function always raises HTTPException.
    """

    if isinstance(
        exc,
        GeminiAuthenticationError,
    ):
        logger.error(
            "[Quiz %s] Gemini authentication failed.",
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
            "[Quiz %s] Gemini quota/rate limit reached.",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": (
                    "Gemini rate limit or quota reached. "
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
            "[Quiz %s] Invalid request: %s",
            feature,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_QUIZ_REQUEST",
            },
        ) from exc

    if isinstance(
        exc,
        RuntimeError,
    ):
        logger.exception(
            "[Quiz %s] Service failure.",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Quiz generation service "
                    "is currently unavailable."
                ),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    logger.exception(
        "[Quiz %s] Unexpected failure.",
        feature,
    )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": (
                "Unable to generate the quiz."
            ),
            "code": "QUIZ_GENERATION_FAILED",
        },
    ) from exc


# ============================================================================
# QUIZ GENERATION
# ============================================================================


@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
)
def generate_quiz(
    request: QuizGenerateRequest,
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Generate a multiple-choice quiz.

    source_type='topic':
        source_id is the topic text.

    source_type='document':
        source_id is the document_id returned from
        /api/documents/upload.

    Correct answers are stored server-side and are NOT
    included in the response.

    Gemini uses the request-scoped API key supplied by the user.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        return quiz_service.generate_quiz(
            session_id=session_id,
            source_type=request.source_type,
            source_id=request.source_id,
            num_questions=request.number_of_questions,
            difficulty=request.difficulty,
            api_key=api_key,
        )

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:
        _raise_generation_error(
            exc,
            feature="Generate",
        )

    except Exception as exc:
        _raise_generation_error(
            exc,
            feature="Generate",
        )


# ============================================================================
# QUIZ SUBMISSION
# ============================================================================


@router.post(
    "/submit",
    response_model=QuizEvaluationResponse,
)
def submit_quiz(
    request: QuizSubmitRequest,
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Submit quiz answers for deterministic backend scoring.

    The score is calculated algorithmically.
    The LLM is not involved in scoring.

    Only the session that generated the quiz can submit it.

    Returns:
        - score
        - percentage
        - mistake breakdown
        - explanations
    """

    try:
        return evaluation_service.evaluate(
            session_id=session_id,
            quiz_id=request.quiz_id,
            user_answers=request.answers,
        )

    except ValueError as exc:
        logger.warning(
            "[Quiz Submit] Invalid submission: %s",
            type(exc).__name__,
        )

        message = str(exc)

        if "not found" in message.lower():
            error_code = "QUIZ_NOT_FOUND"
            response_status = (
                status.HTTP_404_NOT_FOUND
            )

        else:
            error_code = "INVALID_QUIZ_SUBMISSION"
            response_status = (
                status.HTTP_400_BAD_REQUEST
            )

        raise HTTPException(
            status_code=response_status,
            detail={
                "error": message,
                "code": error_code,
            },
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "[Quiz Submit] Evaluation service failure."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Quiz evaluation service "
                    "is currently unavailable."
                ),
                "code": "QUIZ_EVALUATION_UNAVAILABLE",
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "[Quiz Submit] Unexpected failure."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Unable to evaluate the quiz."
                ),
                "code": "QUIZ_EVALUATION_FAILED",
            },
        ) from exc


# ============================================================================
# AGGREGATED LEARNING GAPS
# ============================================================================


@router.get(
    "/gaps",
    response_model=WeakTopicsResponse,
)
def get_aggregated_gaps(
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Analyse submitted quizzes to detect aggregated weak topics.

    Only completed quizzes belonging to the current session
    are included.

    The evaluation service returns the latest 10 valid completed
    quizzes for this session.
    """

    try:
        return evaluation_service.get_aggregated_gaps(
            session_id=session_id,
            limit=10,
        )

    except ValueError as exc:
        logger.warning(
            "[Quiz Gaps] Invalid gap request: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_GAP_REQUEST",
            },
        ) from exc

    except Exception:
        logger.exception(
            "[Quiz Gaps] Failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": (
                    "Unable to retrieve learning gaps."
                ),
                "code": "LEARNING_GAPS_ERROR",
            },
        )


# ============================================================================
# TARGETED PRACTICE
# ============================================================================


@router.post(
    "/practice",
    response_model=QuizGenerateResponse,
)
def generate_practice(
    request: PracticeRequest,
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
    session_id: str = Depends(
        get_session_id
    ),
):
    """
    Generate targeted practice for a weak topic.

    Two modes are supported.

    Topic-only practice:
        {
            "topic": "HashMap ordering",
            "document_id": null
        }

    Document-grounded practice:
        {
            "topic": "Key Uniqueness",
            "document_id": "original-document-id"
        }

    When document_id is supplied, RAG retrieves relevant chunks
    from the original study material belonging to the current session.

    When document_id is omitted, the topic itself is used as
    the practice source.

    Correct answers remain server-side.

    Gemini uses the request-scoped API key supplied by the user.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        return quiz_service.generate_targeted_practice(
            session_id=session_id,
            topic=request.topic,
            document_id=request.document_id,
            num_questions=request.question_count,
            difficulty=request.difficulty,
            api_key=api_key,
        )

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:
        _raise_generation_error(
            exc,
            feature="Practice",
        )

    except Exception as exc:
        _raise_generation_error(
            exc,
            feature="Practice",
        )
        