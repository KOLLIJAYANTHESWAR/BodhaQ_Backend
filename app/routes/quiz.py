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

GET /api/quiz/{quiz_id}/weak-topics
    Analyse weak topics from a submitted quiz.
"""

from fastapi import APIRouter, HTTPException, status

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
from app.services.quiz_service import quiz_service


router = APIRouter(
    prefix="/api/quiz",
    tags=["Quiz"],
)


# ── Quiz generation ───────────────────────────────────────────────────────────


@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
)
def generate_quiz(
    request: QuizGenerateRequest,
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
    """

    try:
        return quiz_service.generate_quiz(
            source_type=request.source_type,
            source_id=request.source_id,
            num_questions=request.number_of_questions,
            difficulty=request.difficulty,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_QUIZ_REQUEST",
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


# ── Quiz submission ──────────────────────────────────────────────────────────


@router.post(
    "/submit",
    response_model=QuizEvaluationResponse,
)
def submit_quiz(
    request: QuizSubmitRequest,
):
    """
    Submit quiz answers for deterministic backend scoring.

    The score is calculated algorithmically.
    The LLM is not involved in scoring.

    Returns:
        - score
        - percentage
        - mistake breakdown
        - explanations
    """

    try:
        return evaluation_service.evaluate(
            request.quiz_id,
            request.answers,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": str(exc),
                "code": "QUIZ_NOT_FOUND",
            },
        ) from exc


# ── Weak-topic detection ─────────────────────────────────────────────────────


@router.get(
    "/gaps",
    response_model=WeakTopicsResponse,
)
def get_aggregated_gaps():
    """
    Analyse submitted quizzes to detect aggregated weak topics.
    Returns the latest 10 quizzes and their combined gaps.
    """
    return evaluation_service.get_aggregated_gaps(limit=10)


# ── Targeted practice ────────────────────────────────────────────────────────


@router.post(
    "/practice",
    response_model=QuizGenerateResponse,
)
def generate_practice(
    request: PracticeRequest,
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
    from the original study material.

    When document_id is omitted, the topic itself is used as
    the practice source.

    Correct answers remain server-side.
    """

    try:
        return quiz_service.generate_targeted_practice(
            topic=request.topic,
            document_id=request.document_id,
            num_questions=request.question_count,
            difficulty=request.difficulty,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_PRACTICE_REQUEST",
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