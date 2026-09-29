"""
Doubts routes.

POST /api/doubts/ask

Supports:
    - Topic mode:
        Answer a question using Gemini without document context.

    - Document mode:
        Retrieve relevant chunks from the selected document using
        session-isolated RAG and answer the question using the
        retrieved context.

Security:
    - A valid BodhaQ anonymous session is required.
    - Gemini credentials are supplied per request by the user.
    - API keys are never persisted by this route.
    - API keys are never logged.
    - Document retrieval is restricted to the current session.
    - Internal provider/runtime errors are not exposed to clients.
"""

from __future__ import annotations

import logging
import time

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    status,
)

from app.dependencies.session import get_session_id
from app.models.requests import DoubtRequest
from app.models.responses import DoubtResponse
from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
    gemini_service,
)
from app.services.rag_service import rag_service


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/doubts",
    tags=["Doubts"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_HISTORY_TURNS = 8


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


def _normalize_history(
    history,
) -> list[dict[str, str]]:
    """
    Convert validated Pydantic conversation turns into the compact
    representation expected by the Gemini service.

    The request model already limits history to eight turns. Slicing
    defensively here keeps the service boundary bounded as well.
    """

    return [
        turn.model_dump()
        for turn in history[
            -MAX_HISTORY_TURNS:
        ]
    ]


def _handle_gemini_error(
    exc: Exception,
    *,
    feature: str,
) -> None:
    """
    Convert internal Gemini exceptions into safe HTTP errors.

    This helper always raises HTTPException.
    """

    if isinstance(
        exc,
        GeminiAuthenticationError,
    ):
        logger.error(
            "[Doubts] Gemini authentication failed. feature=%s",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "AI authentication is currently unavailable."
                ),
                "code": "AI_AUTHENTICATION_FAILED",
            },
        ) from exc

    if isinstance(
        exc,
        GeminiQuotaError,
    ):
        logger.warning(
            "[Doubts] Gemini quota/rate limit reached. feature=%s",
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
            "[Doubts] Invalid Gemini request. feature=%s",
            feature,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Unable to process the question.",
                "code": "INVALID_AI_REQUEST",
            },
        ) from exc

    if isinstance(
        exc,
        RuntimeError,
    ):
        logger.exception(
            "[Doubts] Gemini generation failed. feature=%s",
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
        "[Doubts] Unexpected Gemini failure. feature=%s",
        feature,
    )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": (
                "Unable to generate an AI answer."
            ),
            "code": "AI_SERVICE_UNAVAILABLE",
        },
    ) from exc


# ============================================================================
# MAIN ENDPOINT
# ============================================================================


@router.post(
    "/ask",
    response_model=DoubtResponse,
    status_code=status.HTTP_200_OK,
)
def ask_doubt(
    request: DoubtRequest,
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
    session_id: str = Depends(
        get_session_id
    ),
) -> DoubtResponse:
    """
    Answer a user's question.

    If document_id is provided:
        Use session-isolated RAG to retrieve relevant document chunks.

    If document_id is not provided:
        Answer using Gemini's general knowledge.

    The Gemini API key is supplied by the user for this request only.
    """

    # ------------------------------------------------------------------------
    # GEMINI API KEY
    # ------------------------------------------------------------------------

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    # ------------------------------------------------------------------------
    # NORMALIZE REQUEST VALUES
    # ------------------------------------------------------------------------

    question = (
        request.question.strip()
        if isinstance(
            request.question,
            str,
        )
        else ""
    )

    document_id = (
        request.document_id.strip()
        if isinstance(
            request.document_id,
            str,
        )
        else ""
    )

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Question cannot be empty.",
                "code": "INVALID_DOUBT_REQUEST",
            },
        )

    history = _normalize_history(
        request.history
    )

    # ------------------------------------------------------------------------
    # LIGHTWEIGHT DETERMINISTIC RESPONSES
    # ------------------------------------------------------------------------

    lower_question = question.lower()

    if lower_question in {
        "hi",
        "hello",
        "hey",
    }:
        return DoubtResponse(
            answer="Hi! What would you like to learn?",
            sources=[],
        )

    if lower_question in {
        "thanks",
        "thank you",
    }:
        return DoubtResponse(
            answer=(
                "You're welcome! "
                "Let me know if you have any more questions."
            ),
            sources=[],
        )

    logger.info(
        "[Doubts] Request received. document_mode=%s",
        bool(document_id),
    )

    start_time = time.perf_counter()

    # ------------------------------------------------------------------------
    # DOCUMENT MODE
    # ------------------------------------------------------------------------

    if document_id:
        return _ask_document_doubt(
            session_id=session_id,
            question=question,
            document_id=document_id,
            history=history,
            api_key=api_key,
            start_time=start_time,
        )

    # ------------------------------------------------------------------------
    # TOPIC MODE
    # ------------------------------------------------------------------------

    return _ask_topic_doubt(
        question=question,
        history=history,
        api_key=api_key,
        start_time=start_time,
    )


# ============================================================================
# DOCUMENT MODE
# ============================================================================


def _ask_document_doubt(
    *,
    session_id: str,
    question: str,
    document_id: str,
    history: list[dict[str, str]],
    api_key: str,
    start_time: float,
) -> DoubtResponse:
    """
    Answer a question using session-isolated document-grounded RAG.
    """

    # ------------------------------------------------------------------------
    # STEP 1 — RETRIEVE RELEVANT DOCUMENT CONTEXT
    # ------------------------------------------------------------------------

    retrieval_start = time.perf_counter()

    try:
        context, sources = (
            rag_service.get_context_for_question(
                session_id=session_id,
                document_id=document_id,
                question=question,
                api_key=api_key,
            )
        )

    except ValueError as exc:
        logger.warning(
            "[Doubts] Invalid document request: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Document not found.",
                "code": "DOCUMENT_NOT_FOUND",
            },
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "[Doubts] RAG retrieval failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Document retrieval service "
                    "is currently unavailable."
                ),
                "code": "RAG_RETRIEVAL_FAILED",
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "[Doubts] Unexpected RAG failure."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": (
                    "Unable to retrieve document context."
                ),
                "code": "RAG_RETRIEVAL_FAILED",
            },
        ) from exc

    rag_time = (
        time.perf_counter()
        - retrieval_start
    )

    # ------------------------------------------------------------------------
    # STEP 2 — GENERATE GROUNDED ANSWER
    # ------------------------------------------------------------------------

    gemini_start = time.perf_counter()

    try:
        answer = gemini_service.answer_doubt(
            question=question,
            context=context,
            history=history,
            api_key=api_key,
        )

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:
        _handle_gemini_error(
            exc,
            feature="document",
        )

    except Exception as exc:
        _handle_gemini_error(
            exc,
            feature="document",
        )

    gemini_time = (
        time.perf_counter()
        - gemini_start
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "[Doubts] Document mode completed. "
        "RAG=%.2fs Gemini=%.2fs Total=%.2fs",
        rag_time,
        gemini_time,
        total_time,
    )

    return DoubtResponse(
        answer=answer,
        sources=sources,
    )


# ============================================================================
# TOPIC MODE
# ============================================================================


def _ask_topic_doubt(
    *,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
    start_time: float,
) -> DoubtResponse:
    """
    Answer a question without document grounding.
    """

    gemini_start = time.perf_counter()

    try:
        answer = gemini_service.answer_doubt(
            question=question,
            history=history,
            api_key=api_key,
        )

    except (
        GeminiAuthenticationError,
        GeminiQuotaError,
        ValueError,
        RuntimeError,
    ) as exc:
        _handle_gemini_error(
            exc,
            feature="topic",
        )

    except Exception as exc:
        _handle_gemini_error(
            exc,
            feature="topic",
        )

    gemini_time = (
        time.perf_counter()
        - gemini_start
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "[Doubts] Topic mode completed. "
        "Gemini=%.2fs Total=%.2fs",
        gemini_time,
        total_time,
    )

    return DoubtResponse(
        answer=answer,
        sources=[],
    )