"""
Doubts routes.

POST /api/doubts/ask

Supports:
    - Topic mode:
        Answer a question using Gemini without document context.

    - Document mode:
        Retrieve relevant chunks from the selected document using RAG
        and answer the question using the retrieved context.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.models.requests import DoubtRequest
from app.models.responses import DoubtResponse
from app.services.gemini_service import (
    gemini_service,
    GeminiAuthenticationError,
    GeminiQuotaError,
)
from app.services.rag_service import rag_service


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/doubts",
    tags=["Doubts"],
)


@router.post(
    "/ask",
    response_model=DoubtResponse,
    status_code=status.HTTP_200_OK,
)
def ask_doubt(request: DoubtRequest) -> DoubtResponse:
    """
    Answer a user's question.

    If document_id is provided:
        Use RAG to retrieve relevant document chunks.

    If document_id is not provided:
        Answer using Gemini's general knowledge.

    Gemini API keys are handled only by the backend.
    They are never accepted from the frontend.
    """

    lower_q = request.question.strip().lower()

    # ---------------------------------------------------------
    # Lightweight deterministic responses
    # Avoid unnecessary Gemini calls for simple messages.
    # ---------------------------------------------------------

    if lower_q in {"hi", "hello", "hey"}:
        return DoubtResponse(
            answer="Hi! What would you like to learn?",
            sources=[],
        )

    if lower_q in {"thanks", "thank you"}:
        return DoubtResponse(
            answer="You're welcome! Let me know if you have any more questions.",
            sources=[],
        )

    logger.info(
        "[Doubts] Request received: '%s' | document_id=%s",
        request.question,
        request.document_id,
    )

    start_time = time.time()

    if request.document_id:
        return _ask_document_doubt(
            request=request,
            start_time=start_time,
        )

    return _ask_topic_doubt(
        request=request,
        start_time=start_time,
    )


def _ask_document_doubt(
    request: DoubtRequest,
    start_time: float,
) -> DoubtResponse:
    """
    Answer a question using document-grounded RAG.
    """

    # ---------------------------------------------------------
    # Step 1: Retrieve relevant document context
    # ---------------------------------------------------------

    try:
        retrieval_start = time.time()

        context, sources = rag_service.get_context_for_question(
            document_id=request.document_id,
            question=request.question,
        )

        rag_time = time.time() - retrieval_start

    except ValueError as exc:
        logger.warning(
            "[Doubts] Document retrieval failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": str(exc),
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
                "error": str(exc),
                "code": "RAG_RETRIEVAL_FAILED",
            },
        ) from exc

    # ---------------------------------------------------------
    # Step 2: Generate answer using Gemini
    # ---------------------------------------------------------

    try:
        gemini_start = time.time()

        answer = gemini_service.answer_doubt(
            question=request.question,
            context=context,
            history=[
                turn.model_dump()
                for turn in request.history[-8:]
            ],
        )

        gemini_time = time.time() - gemini_start

    except GeminiAuthenticationError as exc:
        logger.error(
            "[Doubts] Gemini authentication error: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": str(exc),
                "code": "INVALID_API_KEY",
            },
        ) from exc

    except GeminiQuotaError as exc:
        logger.warning(
            "[Doubts] Gemini quota/rate limit error: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": str(exc),
                "code": "AI_RATE_LIMIT",
            },
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "[Doubts] Gemini generation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    # ---------------------------------------------------------
    # Step 3: Performance logging
    # ---------------------------------------------------------

    total_time = time.time() - start_time

    logger.info(
        "[Doubts] RAG retrieval: %.2fs | "
        "Gemini generation: %.2fs | "
        "Total: %.2fs",
        rag_time,
        gemini_time,
        total_time,
    )

    return DoubtResponse(
        answer=answer,
        sources=sources,
    )


def _ask_topic_doubt(
    request: DoubtRequest,
    start_time: float,
) -> DoubtResponse:
    """
    Answer a question without document grounding.
    """

    try:
        gemini_start = time.time()

        answer = gemini_service.answer_doubt(
            question=request.question,
            history=[
                turn.model_dump()
                for turn in request.history[-8:]
            ],
        )

        gemini_time = time.time() - gemini_start

    except GeminiAuthenticationError as exc:
        logger.error(
            "[Doubts] Gemini authentication error: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": str(exc),
                "code": "INVALID_API_KEY",
            },
        ) from exc

    except GeminiQuotaError as exc:
        logger.warning(
            "[Doubts] Gemini quota/rate limit error: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": str(exc),
                "code": "AI_RATE_LIMIT",
            },
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "[Doubts] Gemini generation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    # ---------------------------------------------------------
    # Performance logging
    # ---------------------------------------------------------

    total_time = time.time() - start_time

    logger.info(
        "[Doubts] Gemini generation: %.2fs | Total: %.2fs",
        gemini_time,
        total_time,
    )

    return DoubtResponse(
        answer=answer,
        sources=[],
    )