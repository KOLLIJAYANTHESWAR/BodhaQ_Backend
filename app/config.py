"""
Application configuration.

Loads non-secret environment variables from backend/.env and exposes
centralized configuration values used throughout BodhaQ.

API keys are intentionally NOT loaded from environment variables.

BodhaQ uses a BYOK (Bring Your Own Key) architecture:

    Browser sessionStorage
        ↓
    Request headers
        ↓
    FastAPI
        ↓
    Gemini / Tavily

Gemini and Tavily API keys are request-scoped and must never be:

    - stored in this configuration module
    - persisted in SQLite
    - written to files
    - stored in localStorage
    - logged
    - returned in API responses

The session secret is different from user API keys. It is a server-side
secret used only to sign and validate anonymous BodhaQ session tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================================
# BASE DIRECTORIES
# ============================================================================

# backend/app/config.py
#       ↓ parent
# backend/app
#       ↓ parent
# backend
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

load_dotenv(dotenv_path=ENV_FILE)


# ----------------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------------

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173",
).strip()


# ----------------------------------------------------------------------------
# Session security
# ----------------------------------------------------------------------------
#
# This is a server-side signing secret for anonymous session tokens.
#
# It is NOT a Gemini/Tavily API key and must never be exposed to the browser.
#
# Production/local setup:
#   BODHAQ_SESSION_SECRET=<long random secret>
#
# The application intentionally fails to start if the secret is missing.
# This prevents session tokens from becoming invalid after a restart due to
# an automatically regenerated secret.
#

SESSION_SECRET = os.getenv(
    "BODHAQ_SESSION_SECRET",
    "",
).strip()

if not SESSION_SECRET:
    raise RuntimeError(
        "BODHAQ_SESSION_SECRET is required."
    )


# ============================================================================
# APPLICATION DIRECTORIES
# ============================================================================

UPLOADS_DIR = BASE_DIR / "uploads"

DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = DATA_DIR / "chroma"


# ============================================================================
# DATABASE
# ============================================================================

DB_PATH = DATA_DIR / "bodhaq.db"


# ============================================================================
# ENSURE REQUIRED DIRECTORIES EXIST
# ============================================================================

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)