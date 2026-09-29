"""
BodhaQ FastAPI application entry point.

Responsibilities:
    - Create the FastAPI application.
    - Register all route modules.
    - Configure CORS.
    - Apply basic HTTP security headers.

Business logic lives in services/.
AI calls live in services/gemini_service.py.
RAG logic lives in rag/ and services/rag_service.py.

API keys are request-scoped and are never stored by the application.
Anonymous session tokens are signed server-side and are never persisted
as authentication credentials.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import FRONTEND_URL
from app.routes import (
    coding,
    documents,
    doubts,
    health,
    learning,
    quiz,
    resume,
    session,
    settings,
)


logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="BodhaQ API",
    description=(
        "AI-powered learning workspace. "
        "Turn study materials and topics into interactive "
        "learning experiences."
    ),
    version="0.2.0",
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Accept",
        "Content-Type",
        "X-BodhaQ-Session",
        "X-Gemini-API-Key",
        "X-Tavily-API-Key",
    ],
)


# ============================================================================
# SECURITY HEADERS
# ============================================================================


@app.middleware("http")
async def security_headers(
    request: Request,
    call_next,
):
    """
    Apply basic security-related HTTP response headers.

    These headers do not contain secrets and do not alter application
    business logic.
    """

    response = await call_next(
        request
    )

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    return response


# ============================================================================
# STARTUP
# ============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """
    Log non-sensitive application startup information.
    """

    logger.info(
        "BodhaQ API started successfully."
    )

    logger.info(
        "Configured frontend origin: %s",
        FRONTEND_URL,
    )


# ============================================================================
# ROUTES
# ============================================================================

# Health
app.include_router(
    health.router
)

# Anonymous sessions
app.include_router(
    session.router
)

# Learning
app.include_router(
    learning.router
)

# Documents / RAG
app.include_router(
    documents.router
)

# Quiz
app.include_router(
    quiz.router
)

# Doubts
app.include_router(
    doubts.router
)

# Settings
app.include_router(
    settings.router
)

# Resume preparation
app.include_router(
    resume.router
)

# Coding
app.include_router(
    coding.router,
    prefix="/api/coding",
)


# ============================================================================
# ROOT
# ============================================================================


@app.get("/")
def root() -> dict[str, str]:
    """
    Basic API status endpoint.
    """

    return {
        "name": "BodhaQ API",
        "version": "0.2.0",
        "status": "running",
    }