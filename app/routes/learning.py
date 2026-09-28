"""
Learning routes.

POST /api/learn/topic — generate structured learning content for a topic.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.requests import TopicRequest
from app.models.responses import LearningContent
from app.services.gemini_service import gemini_service


router = APIRouter(
    prefix="/api/learn",
    tags=["Learning"],
)


@router.post(
    "/topic",
    response_model=LearningContent,
    status_code=status.HTTP_200_OK,
)
def learn_topic(request: TopicRequest):
    """
    Generate structured learning content for a topic.
    """

    try:
        result = gemini_service.generate_learning_content(
            request.topic
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": str(exc),
                "code": "AI_SERVICE_UNAVAILABLE",
            },
        ) from exc

    return LearningContent(
        topic=result.topic,
        definition=result.definition,
        key_concepts=result.key_concepts,
        example=result.example.model_dump(),
        important_points=result.important_points,
    )