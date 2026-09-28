"""
Application configuration.

Loads environment variables from backend/.env and exposes
centralized configuration values used throughout BodhaQ.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ── Base directories ──────────────────────────────────────────────────────────

# backend/app/config.py
#        ↓ parent
# backend/app
#        ↓ parent
# backend
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"


# ── Environment variables ─────────────────────────────────────────────────────

load_dotenv(
    dotenv_path=ENV_FILE,
)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()


if not GEMINI_API_KEY:
    import logging
    logging.getLogger("uvicorn.error").warning(
        "GEMINI_API_KEY is not configured in backend/.env. "
        "Requests will require X-Gemini-API-Key header or a valid key."
    )


# ── Application directories ───────────────────────────────────────────────────

UPLOADS_DIR = BASE_DIR / "uploads"

DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = DATA_DIR / "chroma"


# ── Database ───────────────────────────────────────────────────────────────────

DB_PATH = DATA_DIR / "bodhaq.db"


# ── Ensure required directories exist ─────────────────────────────────────────

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