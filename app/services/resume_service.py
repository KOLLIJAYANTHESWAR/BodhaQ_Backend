"""
ResumeService — manages resume ingestion, parsing, and progress tracking.

Gemini authentication is handled centrally by GeminiService using the
backend GEMINI_API_KEY from .env. The frontend never provides an API key.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.services.gemini_service import gemini_service
from app.services.quiz_service import quiz_service


DB_PATH = "data/bodhaq.db"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _get_db() -> sqlite3.Connection:
    """
    Get a SQLite connection and ensure the resume tables exist.
    """
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_items (
            id TEXT PRIMARY KEY,
            resume_id TEXT NOT NULL,
            item_type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            technologies TEXT,
            status TEXT DEFAULT 'not_started',
            score INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(resume_id) REFERENCES resumes(id)
        )
        """
    )

    conn.commit()

    return conn


class ResumeService:
    """
    Service responsible for:

    - extracting text from resumes
    - using Gemini to structure resume information
    - storing resume data and interview items
    - tracking interview preparation progress
    - generating resume-specific quizzes
    """

    # ──────────────────────────────────────────────────────────────────────
    # Resume ingestion
    # ──────────────────────────────────────────────────────────────────────

    def ingest_resume(
        self,
        file_path: str,
        original_filename: str,
    ) -> str:
        """
        Process a resume and create a new persistent resume preparation queue.

        Gemini authentication comes exclusively from backend/.env through
        GeminiService.
        """

        ext = Path(original_filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {supported}"
            )

        # ── 1. Extract text ────────────────────────────────────────────────

        if ext == ".pdf":
            pages = load_pdf(file_path)

        elif ext == ".docx":
            pages = load_docx(file_path)

        else:
            raise ValueError("Unhandled resume extension.")

        if not pages:
            raise ValueError(
                "No readable content could be extracted from the resume."
            )

        full_text = "\n".join(
            page.get("text", "")
            for page in pages
            if page.get("text")
        ).strip()

        if not full_text:
            raise ValueError(
                "The resume does not contain readable text."
            )

        # ── 2. Extract structured resume information using Gemini ─────────

        # IMPORTANT:
        # No API key is passed from the frontend or route.
        # GeminiService uses GEMINI_API_KEY from backend/.env.
        extracted = gemini_service.extract_resume(full_text)

        # ── 3. Store resume and extracted interview items ─────────────────

        resume_id = str(uuid.uuid4())

        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO resumes (
                    id,
                    filename,
                    raw_text
                )
                VALUES (?, ?, ?)
                """,
                (
                    resume_id,
                    original_filename,
                    full_text,
                ),
            )

            # ── Skills ────────────────────────────────────────────────────

            for skill in extracted.skills:
                conn.execute(
                    """
                    INSERT INTO resume_items (
                        id,
                        resume_id,
                        item_type,
                        name,
                        description,
                        technologies
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        resume_id,
                        "skill",
                        skill.name,
                        skill.description,
                        None,
                    ),
                )

            # ── Projects ──────────────────────────────────────────────────

            for project in extracted.projects:
                technologies_json = json.dumps(
                    project.technologies
                )

                conn.execute(
                    """
                    INSERT INTO resume_items (
                        id,
                        resume_id,
                        item_type,
                        name,
                        description,
                        technologies
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        resume_id,
                        "project",
                        project.name,
                        project.description,
                        technologies_json,
                    ),
                )

            # ── Certifications ────────────────────────────────────────────

            for certification in extracted.certifications:
                conn.execute(
                    """
                    INSERT INTO resume_items (
                        id,
                        resume_id,
                        item_type,
                        name,
                        description,
                        technologies
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        resume_id,
                        "certification",
                        certification.name,
                        None,
                        None,
                    ),
                )

        return resume_id

    # ──────────────────────────────────────────────────────────────────────
    # Progress
    # ──────────────────────────────────────────────────────────────────────

    def get_latest_resume_progress(self) -> dict:
        """
        Return the latest resume and its persistent interview preparation
        progress.
        """

        with _get_db() as conn:
            # Get latest uploaded resume.
            resume_row = conn.execute(
                """
                SELECT *
                FROM resumes
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

            if not resume_row:
                return {}

            resume_id = resume_row["id"]
            filename = resume_row["filename"]

            items_rows = conn.execute(
                """
                SELECT *
                FROM resume_items
                WHERE resume_id = ?
                ORDER BY created_at ASC
                """,
                (resume_id,),
            ).fetchall()

        skills = []
        projects = []
        certifications = []

        completed_count = 0

        for row in items_rows:
            item = {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "status": row["status"],
                "score": row["score"],
            }

            if row["status"] == "completed":
                completed_count += 1

            if row["item_type"] == "skill":
                skills.append(item)

            elif row["item_type"] == "project":
                technologies = []

                if row["technologies"]:
                    try:
                        technologies = json.loads(
                            row["technologies"]
                        )
                    except (TypeError, json.JSONDecodeError):
                        technologies = []

                item["technologies"] = technologies

                projects.append(item)

            elif row["item_type"] == "certification":
                certifications.append(item)

        total_items = (
            len(skills)
            + len(projects)
            + len(certifications)
        )

        overall_progress = (
            (completed_count / total_items) * 100
            if total_items > 0
            else 0.0
        )

        return {
            "resume_id": resume_id,
            "filename": filename,
            "skills": skills,
            "projects": projects,
            "certifications": certifications,
            "overall_progress": overall_progress,
            "items_completed": completed_count,
            "total_items": total_items,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Resume item quiz generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_quiz_for_item(
        self,
        item_id: str,
        difficulty: str,
        num_questions: int,
    ):
        """
        Generate a quiz for one resume item.

        The resume item itself is already stored in SQLite, so the full
        resume does not need to be sent to Gemini again.

        Gemini authentication comes exclusively from backend/.env.
        """

        # ── 1. Load resume item ────────────────────────────────────────────

        with _get_db() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM resume_items
                WHERE id = ?
                """,
                (item_id,),
            ).fetchone()

            if not row:
                raise ValueError("Resume item not found.")

            item_type = row["item_type"]

            item_name = row["name"]
            item_description = row["description"]

            technologies = []

            if row["technologies"]:
                try:
                    technologies = json.loads(
                        row["technologies"]
                    )
                except (TypeError, json.JSONDecodeError):
                    technologies = []

        item_data = {
            "name": item_name,
            "description": item_description,
            "technologies": technologies,
        }

        # ── 2. Generate quiz through centralized Gemini service ────────────

        # IMPORTANT:
        # No api_key argument.
        # GeminiService gets GEMINI_API_KEY from backend/.env.
        raw_quiz = gemini_service.generate_resume_quiz(
            item_type=item_type,
            item_data=item_data,
            num_questions=num_questions,
            difficulty=difficulty,
        )

        # ── 3. Create quiz through existing quiz engine ───────────────────

        quiz_id = str(uuid.uuid4())

        response = quiz_service._build_quiz_response(
            raw_quiz=raw_quiz,
            quiz_id=quiz_id,
            source_type="resume_item",
            source_id=item_id,
            num_questions=num_questions,
            topic_hint=item_name,
        )

        # Only mark the item in progress after successful quiz generation.
        with _get_db() as conn:
            conn.execute(
                """
                UPDATE resume_items
                SET status = 'in_progress'
                WHERE id = ?
                  AND status = 'not_started'
                """,
                (item_id,),
            )

        return response

    # ──────────────────────────────────────────────────────────────────────
    # Completion
    # ──────────────────────────────────────────────────────────────────────

    def mark_item_completed(
        self,
        item_id: str,
        score: int,
        percentage: float,
    ):
        """
        Update persistent resume item progress after quiz evaluation.

        >= 80%  -> completed
        < 80%   -> needs_review
        """

        status = (
            "completed"
            if percentage >= 80
            else "needs_review"
        )

        percentage_int = int(round(percentage))

        with _get_db() as conn:
            result = conn.execute(
                """
                UPDATE resume_items
                SET
                    status = ?,
                    score = ?
                WHERE id = ?
                """,
                (
                    status,
                    percentage_int,
                    item_id,
                ),
            )

            if result.rowcount == 0:
                raise ValueError(
                    "Resume item not found."
                )


# ──────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────────

resume_service = ResumeService()