from __future__ import annotations

from fastapi import Header, HTTPException

from app.services.session_service import SessionError, validate_session_token


SESSION_HEADER = "X-BodhaQ-Session"


def get_session_id(
    x_bodhaq_session: str | None = Header(
        default=None,
        alias=SESSION_HEADER,
    ),
) -> str:
    try:
        return validate_session_token(x_bodhaq_session or "")
    except SessionError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_SESSION",
                "message": str(exc),
            },
        ) from exc
