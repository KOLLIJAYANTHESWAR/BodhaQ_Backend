"""
Anonymous session routes.

Responsibilities:
    - Create a new anonymous BodhaQ session.
    - Return the session identifier and signed session token.

Security:
    - No user API keys are handled here.
    - Session tokens are signed server-side.
    - Session data is not persisted by this route.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.models.session import SessionResponse
from app.services.session_service import create_session


router = APIRouter(
    prefix="/api/session",
    tags=["Session"],
)


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_anonymous_session() -> SessionResponse:
    """
    Create a new anonymous BodhaQ session.

    Returns:
        A UUID session identifier and its signed session token.
    """

    session_id, session_token = create_session()

    return SessionResponse(
        session_id=session_id,
        session_token=session_token,
    )