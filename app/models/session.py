from __future__ import annotations

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str
    session_token: str
