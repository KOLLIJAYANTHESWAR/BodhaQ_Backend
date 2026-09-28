"""
BodhaQ FastAPI application entry point.

Responsibilities:
    - Create the FastAPI application.
    - Register all route modules.
    - Configure CORS.

Business logic lives in services/.
AI calls live in services/gemini_service.py.
RAG logic lives in rag/ and services/rag_service.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import (
    documents,
    doubts,
    health,
    learning,
    quiz,
    settings,
    resume,
    coding,
)


app = FastAPI(
    title="BodhaQ API",
    description=(
        "AI-powered learning workspace. "
        "Turn study materials and topics into interactive "
        "learning experiences."
    ),
    version="0.2.0",
)


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(learning.router)
app.include_router(documents.router)
app.include_router(quiz.router)
app.include_router(doubts.router)
app.include_router(settings.router)
app.include_router(resume.router)
app.include_router(coding.router, prefix="/api/coding")