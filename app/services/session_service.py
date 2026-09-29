from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid

from app.config import SESSION_SECRET


SESSION_TOKEN_VERSION = "v1"
SESSION_TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


class SessionError(ValueError):
    pass


def _sign(payload: str) -> str:
    return hmac.new(
        SESSION_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_session() -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    issued_at = str(int(time.time()))
    nonce = secrets.token_hex(16)

    payload = ".".join((
        SESSION_TOKEN_VERSION,
        session_id,
        issued_at,
        nonce,
    ))

    signature = _sign(payload)
    token = f"{payload}.{signature}"

    return session_id, token


def validate_session_token(token: str) -> str:
    token = str(token or "").strip()

    if not token:
        raise SessionError("Session token is required.")

    parts = token.split(".")

    if len(parts) != 5:
        raise SessionError("Invalid session token.")

    version, session_id, issued_at, nonce, signature = parts

    if version != SESSION_TOKEN_VERSION:
        raise SessionError("Unsupported session token.")

    try:
        uuid.UUID(session_id)
    except ValueError as exc:
        raise SessionError("Invalid session identifier.") from exc

    if not issued_at.isdigit():
        raise SessionError("Invalid session timestamp.")

    issued_timestamp = int(issued_at)
    now = int(time.time())

    if issued_timestamp > now + 60:
        raise SessionError("Invalid session timestamp.")

    if now - issued_timestamp > SESSION_TOKEN_MAX_AGE_SECONDS:
        raise SessionError("Session has expired.")

    if not nonce or len(nonce) != 32:
        raise SessionError("Invalid session token.")

    payload = ".".join((
        version,
        session_id,
        issued_at,
        nonce,
    ))

    expected_signature = _sign(payload)

    if not hmac.compare_digest(signature, expected_signature):
        raise SessionError("Invalid session token.")

    return session_id
