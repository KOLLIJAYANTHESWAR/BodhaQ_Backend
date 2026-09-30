"""
ResumeService — manages resume ingestion, parsing, and progress tracking.

Responsibilities:
    - Extract text from resumes.
    - Use GeminiService to structure resume information.
    - Store resume versions and extracted preparation items.
    - Track persistent interview-preparation progress.
    - Generate resume-specific quizzes.
    - Reuse the existing QuizService for secure quiz generation.

Gemini authentication:
    - Gemini API keys are request-scoped.
    - The API key is supplied by the route for the current request only.
    - The key is never persisted or logged by this service.

Persistence:
    - Resume data and progress are stored in SQLite.
    - Each uploaded resume gets its own resume_id.
    - Each resume belongs to exactly one anonymous BodhaQ session.
    - Resume items are permanently associated with that resume version.
    - Legacy resumes without a session_id remain inaccessible to new sessions.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app.config import DB_PATH, DATA_DIR, UPLOADS_DIR
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.services.gemini_service import gemini_service
from app.services.quiz_service import quiz_service


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


VALID_ITEM_TYPES = {
    "skill",
    "project",
    "certification",
}


MIN_QUESTIONS = 1
MAX_QUESTIONS = 20

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_session_id(session_id: str) -> str:
    """
    Validate an anonymous BodhaQ session identifier.

    The session ID is validated independently from the signed session token
    because services receive the already-resolved session identifier from
    FastAPI dependencies.
    """

    normalized = str(
        session_id or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "Session ID cannot be empty."
        )

    try:
        uuid.UUID(normalized)
    except ValueError as exc:
        raise ValueError(
            "Invalid session ID."
        ) from exc

    return normalized


def _validate_resume_id(resume_id: str) -> str:
    """
    Validate a resume UUID.
    """

    normalized = str(
        resume_id or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "Resume ID cannot be empty."
        )

    try:
        uuid.UUID(normalized)
    except ValueError as exc:
        raise ValueError(
            "Invalid resume ID."
        ) from exc

    return normalized


def _validate_item_id(item_id: str) -> str:
    """
    Validate a resume-item UUID.
    """

    normalized = str(
        item_id or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "Resume item ID cannot be empty."
        )

    try:
        uuid.UUID(normalized)
    except ValueError as exc:
        raise ValueError(
            "Invalid resume item ID."
        ) from exc

    return normalized


def _validate_api_key(api_key: str) -> str:
    """
    Validate a request-scoped Gemini API key.

    The key is intentionally never logged, persisted, or returned.
    """

    normalized = str(
        api_key or ""
    ).strip()

    if not normalized:
        raise ValueError(
            "Gemini API key cannot be empty."
        )

    return normalized


def _validate_upload_path(file_path: str) -> Path:
    """
    Validate that the temporary resume file exists inside UPLOADS_DIR.

    Resume uploads are expected to be temporary files created by the
    resume route. Restricting the path prevents this service from being
    accidentally reused as an arbitrary local-file reader.
    """

    normalized_path = str(
        file_path or ""
    ).strip()

    if not normalized_path:
        raise ValueError(
            "Resume file path cannot be empty."
        )

    source_path = Path(
        normalized_path
    ).resolve()

    if not source_path.is_file():
        raise ValueError(
            "Resume file could not be found."
        )

    uploads_root = Path(
        UPLOADS_DIR
    ).resolve()

    try:
        source_path.relative_to(
            uploads_root
        )
    except ValueError as exc:
        raise ValueError(
            "Resume file is outside the allowed upload directory."
        ) from exc

    return source_path


# ============================================================================
# DATABASE
# ============================================================================


def _get_db() -> sqlite3.Connection:
    """
    Get a SQLite connection and ensure the resume tables exist.

    Session ownership is stored on the resumes table. Resume items inherit
    ownership through their resume_id foreign key.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        str(DB_PATH)
    )

    conn.row_factory = sqlite3.Row

    # SQLite foreign keys must be enabled per connection.
    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    # ------------------------------------------------------------------------
    # RESUMES TABLE
    # ------------------------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            filename TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # ------------------------------------------------------------------------
    # SAFE MIGRATION
    # ------------------------------------------------------------------------

    # Existing BodhaQ installations may already have a resumes table without
    # session_id. Add the column without modifying existing resume contents.
    #
    # Existing rows remain NULL and therefore cannot be returned to any new
    # anonymous session. This prevents accidental cross-session exposure.
    columns = {
        str(row["name"])
        for row in conn.execute(
            "PRAGMA table_info(resumes)"
        ).fetchall()
    }

    if "session_id" not in columns:
        conn.execute(
            """
            ALTER TABLE resumes
            ADD COLUMN session_id TEXT
            """
        )

    # ------------------------------------------------------------------------
    # RESUME ITEMS TABLE
    # ------------------------------------------------------------------------

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

            FOREIGN KEY(resume_id)
                REFERENCES resumes(id)
                ON DELETE CASCADE
        )
        """
    )

    # ------------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------------

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_resume_items_resume_id
        ON resume_items(resume_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_resumes_created_at
        ON resumes(created_at)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_resumes_session_created
        ON resumes(session_id, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_resumes_session_id
        ON resumes(session_id)
        """
    )

    conn.commit()

    return conn


# ============================================================================
# SERVICE
# ============================================================================


class ResumeService:
    """
    Service responsible for:

    - extracting text from resumes
    - using Gemini to structure resume information
    - storing resume data and interview items
    - tracking interview preparation progress
    - generating resume-specific quizzes
    """

    # ========================================================================
    # RESUME INGESTION
    # ========================================================================

    def ingest_resume(
        self,
        session_id: str,
        file_path: str,
        original_filename: str,
        api_key: str,
    ) -> str:
        """
        Process a resume and create a new persistent resume-preparation queue.

        Args:
            session_id:
                Anonymous BodhaQ session that owns this resume.

            file_path:
                Path to the temporarily uploaded resume.

            original_filename:
                Original filename supplied by the user.

            api_key:
                Request-scoped Gemini API key.

        Returns:
            Newly created resume_id.

        Security:
            The API key is used only for the current Gemini request.
            It is never stored in SQLite or logged.
        """

        session_id = _validate_session_id(
            session_id
        )

        original_filename = str(
            original_filename or ""
        ).strip()

        if not original_filename:
            raise ValueError(
                "Resume filename cannot be empty."
            )

        source_path = _validate_upload_path(
            file_path
        )

        api_key = _validate_api_key(
            api_key
        )

        # --------------------------------------------------------------------
        # FILE TYPE
        # --------------------------------------------------------------------

        ext = Path(
            original_filename
        ).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(
                sorted(
                    SUPPORTED_EXTENSIONS
                )
            )

            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {supported}"
            )

        # --------------------------------------------------------------------
        # 1. EXTRACT TEXT
        # --------------------------------------------------------------------

        try:

            if ext == ".pdf":
                pages = load_pdf(
                    str(source_path)
                )

            elif ext == ".docx":
                pages = load_docx(
                    str(source_path)
                )

            else:
                raise ValueError(
                    "Unhandled resume extension."
                )

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                "Failed to extract readable content from the resume."
            ) from exc

        if not pages:
            raise ValueError(
                "No readable content could be extracted from the resume."
            )

        full_text = "\n".join(
            str(
                page.get(
                    "text",
                    "",
                )
                or ""
            )
            for page in pages
            if isinstance(
                page,
                dict,
            )
        ).strip()

        if not full_text:
            raise ValueError(
                "The resume does not contain readable text."
            )

        # --------------------------------------------------------------------
        # 2. EXTRACT STRUCTURED RESUME INFORMATION
        # --------------------------------------------------------------------

        extracted = gemini_service.extract_resume(
            full_text,
            api_key=api_key,
        )

        if extracted is None:
            raise RuntimeError(
                "AI service returned no structured resume information."
            )

        # --------------------------------------------------------------------
        # 3. CREATE RESUME ID
        # --------------------------------------------------------------------

        resume_id = str(
            uuid.uuid4()
        )

        # --------------------------------------------------------------------
        # 4. STORE RESUME + ITEMS ATOMICALLY
        # --------------------------------------------------------------------

        try:

            with _get_db() as conn:

                conn.execute(
                    """
                    INSERT INTO resumes (
                        id,
                        session_id,
                        filename,
                        raw_text
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        resume_id,
                        session_id,
                        original_filename,
                        full_text,
                    ),
                )

                # ------------------------------------------------------------
                # SKILLS
                # ------------------------------------------------------------

                for skill in extracted.skills:

                    name = str(
                        skill.name or ""
                    ).strip()

                    if not name:
                        continue

                    description = (
                        str(
                            skill.description or ""
                        ).strip()
                        or None
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
                            "skill",
                            name,
                            description,
                            None,
                        ),
                    )

                # ------------------------------------------------------------
                # PROJECTS
                # ------------------------------------------------------------

                for project in extracted.projects:

                    name = str(
                        project.name or ""
                    ).strip()

                    if not name:
                        continue

                    description = (
                        str(
                            project.description or ""
                        ).strip()
                        or None
                    )

                    technologies = [
                        str(
                            technology
                        ).strip()
                        for technology in (
                            project.technologies or []
                        )
                        if str(
                            technology
                        ).strip()
                    ]

                    technologies_json = json.dumps(
                        technologies
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
                            name,
                            description,
                            technologies_json,
                        ),
                    )

                # ------------------------------------------------------------
                # CERTIFICATIONS
                # ------------------------------------------------------------

                for certification in (
                    extracted.certifications
                ):

                    name = str(
                        certification.name or ""
                    ).strip()

                    if not name:
                        continue

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
                            name,
                            None,
                            None,
                        ),
                    )

        except sqlite3.Error as exc:
            raise RuntimeError(
                "Failed to store the extracted resume data."
            ) from exc

        return resume_id

    # ========================================================================
    # PROGRESS
    # ========================================================================

    def get_latest_resume_progress(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Return the latest resume belonging to the supplied session.

        Only items belonging to that session's latest resume version
        are returned.
        """

        session_id = _validate_session_id(
            session_id
        )

        with _get_db() as conn:

            resume_row = conn.execute(
                """
                SELECT *
                FROM resumes
                WHERE session_id = ?
                ORDER BY
                    created_at DESC,
                    id DESC
                LIMIT 1
                """,
                (
                    session_id,
                ),
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
                ORDER BY
                    created_at ASC,
                    id ASC
                """,
                (
                    resume_id,
                ),
            ).fetchall()

        skills: list[dict[str, Any]] = []
        projects: list[dict[str, Any]] = []
        certifications: list[dict[str, Any]] = []

        completed_count = 0

        for row in items_rows:

            item_status = (
                str(
                    row["status"]
                    or "not_started"
                )
                .strip()
                .lower()
            )

            item = {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "status": item_status,
                "score": row["score"],
            }

            if item_status == "completed":
                completed_count += 1

            # ---------------------------------------------------------------
            # SKILL
            # ---------------------------------------------------------------

            if row["item_type"] == "skill":

                skills.append(
                    item
                )

            # ---------------------------------------------------------------
            # PROJECT
            # ---------------------------------------------------------------

            elif row["item_type"] == "project":

                technologies: list[str] = []

                if row["technologies"]:

                    try:

                        decoded = json.loads(
                            row["technologies"]
                        )

                        if isinstance(
                            decoded,
                            list,
                        ):
                            technologies = [
                                str(value).strip()
                                for value in decoded
                                if str(value).strip()
                            ]

                    except (
                        TypeError,
                        json.JSONDecodeError,
                    ):
                        technologies = []

                item["technologies"] = (
                    technologies
                )

                projects.append(
                    item
                )

            # ---------------------------------------------------------------
            # CERTIFICATION
            # ---------------------------------------------------------------

            elif row["item_type"] == "certification":

                certifications.append(
                    item
                )

        total_items = (
            len(skills)
            + len(projects)
            + len(certifications)
        )

        overall_progress = (
            (
                completed_count
                / total_items
            )
            * 100
            if total_items > 0
            else 0.0
        )

        return {
            "resume_id": resume_id,
            "filename": filename,
            "skills": skills,
            "projects": projects,
            "certifications": certifications,
            "overall_progress": round(
                overall_progress,
                2,
            ),
            "items_completed": completed_count,
            "total_items": total_items,
        }

    # ========================================================================
    # RESUME ITEM QUIZ GENERATION
    # ========================================================================

    def generate_quiz_for_item(
        self,
        session_id: str,
        item_id: str,
        difficulty: str,
        num_questions: int,
        api_key: str,
    ):
        """
        Generate a quiz for one resume item.

        The resume item is loaded only when it belongs to the supplied
        anonymous session.

        Args:
            session_id:
                Anonymous BodhaQ session that owns the resume item.

            item_id:
                Stored resume-item ID.

            difficulty:
                easy, medium, or hard.

            num_questions:
                Number of questions to generate.

            api_key:
                Request-scoped Gemini API key.

        Security:
            The API key is used only for the current Gemini request.
            It is never persisted or logged.
        """

        session_id = _validate_session_id(
            session_id
        )

        item_id = _validate_item_id(
            item_id
        )

        api_key = _validate_api_key(
            api_key
        )

        difficulty = str(
            difficulty or ""
        ).strip().lower()

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                "Invalid difficulty. Expected 'easy', 'medium', or 'hard'."
            )

        if isinstance(
            num_questions,
            bool,
        ) or not isinstance(
            num_questions,
            int,
        ):
            raise ValueError(
                "num_questions must be an integer."
            )

        if (
            num_questions < MIN_QUESTIONS
            or num_questions > MAX_QUESTIONS
        ):
            raise ValueError(
                f"num_questions must be between "
                f"{MIN_QUESTIONS} and {MAX_QUESTIONS}."
            )

        # --------------------------------------------------------------------
        # 1. LOAD RESUME ITEM WITH SESSION OWNERSHIP CHECK
        # --------------------------------------------------------------------

        with _get_db() as conn:

            row = conn.execute(
                """
                SELECT
                    ri.*,
                    r.session_id AS resume_session_id
                FROM resume_items AS ri
                INNER JOIN resumes AS r
                    ON r.id = ri.resume_id
                WHERE
                    ri.id = ?
                    AND r.session_id = ?
                """,
                (
                    item_id,
                    session_id,
                ),
            ).fetchone()

            if not row:
                raise ValueError(
                    "Resume item not found."
                )

            item_type = (
                str(
                    row["item_type"]
                    or ""
                ).strip().lower()
            )

            if item_type not in VALID_ITEM_TYPES:
                raise RuntimeError(
                    "Stored resume item has an invalid item type."
                )

            item_name = (
                str(
                    row["name"]
                    or ""
                ).strip()
            )

            item_description = (
                str(
                    row["description"]
                    or ""
                ).strip()
            )

            technologies: list[str] = []

            if row["technologies"]:

                try:

                    decoded = json.loads(
                        row["technologies"]
                    )

                    if isinstance(
                        decoded,
                        list,
                    ):
                        technologies = [
                            str(value).strip()
                            for value in decoded
                            if str(value).strip()
                        ]

                except (
                    TypeError,
                    json.JSONDecodeError,
                ):
                    technologies = []

        if not item_name:
            raise RuntimeError(
                "Resume item has no usable name."
            )

        item_data = {
            "name": item_name,
            "description": item_description,
            "technologies": technologies,
        }

        # --------------------------------------------------------------------
        # 2. GENERATE QUIZ THROUGH CENTRALIZED GEMINI SERVICE
        # --------------------------------------------------------------------

        raw_quiz = gemini_service.generate_resume_quiz(
            item_type=item_type,
            item_data=item_data,
            num_questions=num_questions,
            difficulty=difficulty,
            api_key=api_key,
        )

        if raw_quiz is None:
            raise RuntimeError(
                "AI service returned no resume quiz."
            )

        # --------------------------------------------------------------------
        # 3. REUSE EXISTING SECURE QUIZ ENGINE
        # --------------------------------------------------------------------

        quiz_id = str(
            uuid.uuid4()
        )

        response = quiz_service._build_quiz_response(
            raw_quiz=raw_quiz,
            quiz_id=quiz_id,
            source_type="resume_item",
            source_id=item_id,
            num_questions=num_questions,
            topic_hint=item_name,
            session_id=session_id,
        )

        # --------------------------------------------------------------------
        # 4. MARK ITEM IN PROGRESS
        # --------------------------------------------------------------------

        # Only change not_started → in_progress.
        #
        # If the item is already completed or needs_review, don't
        # accidentally erase that state merely because another quiz
        # was generated.

        with _get_db() as conn:

            conn.execute(
                """
                UPDATE resume_items
                SET status = 'in_progress'
                WHERE id = ?
                  AND status = 'not_started'
                  AND resume_id IN (
                      SELECT id
                      FROM resumes
                      WHERE session_id = ?
                  )
                """,
                (
                    item_id,
                    session_id,
                ),
            )

        return response

    # ========================================================================
    # COMPLETION
    # ========================================================================

    def mark_item_completed(
        self,
        session_id: str,
        item_id: str,
        score: int,
        percentage: float,
    ) -> None:
        """
        Update persistent resume item progress after quiz evaluation.

        >= 80%:
            completed

        < 80%:
            needs_review
        """

        session_id = _validate_session_id(
            session_id
        )

        item_id = _validate_item_id(
            item_id
        )

        # --------------------------------------------------------------------
        # Validate score
        # --------------------------------------------------------------------

        if isinstance(
            score,
            bool,
        ) or not isinstance(
            score,
            int,
        ):
            raise ValueError(
                "score must be an integer."
            )

        if score < 0:
            raise ValueError(
                "score cannot be negative."
            )

        # --------------------------------------------------------------------
        # Validate percentage
        # --------------------------------------------------------------------

        if isinstance(
            percentage,
            bool,
        ) or not isinstance(
            percentage,
            (int, float),
        ):
            raise ValueError(
                "percentage must be a number."
            )

        percentage = float(
            percentage
        )

        if percentage < 0 or percentage > 100:
            raise ValueError(
                "percentage must be between 0 and 100."
            )

        # --------------------------------------------------------------------
        # Determine persistent status
        # --------------------------------------------------------------------

        item_status = (
            "completed"
            if percentage >= 80
            else "needs_review"
        )

        percentage_int = int(
            round(
                percentage
            )
        )

        # --------------------------------------------------------------------
        # Update item only when it belongs to this session
        # --------------------------------------------------------------------

        with _get_db() as conn:

            result = conn.execute(
                """
                UPDATE resume_items
                SET
                    status = ?,
                    score = ?
                WHERE
                    id = ?
                    AND resume_id IN (
                        SELECT id
                        FROM resumes
                        WHERE session_id = ?
                    )
                """,
                (
                    item_status,
                    percentage_int,
                    item_id,
                    session_id,
                ),
            )

            if result.rowcount == 0:
                raise ValueError(
                    "Resume item not found."
                )


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

resume_service = ResumeService()