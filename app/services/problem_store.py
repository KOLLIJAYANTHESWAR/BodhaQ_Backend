"""
BodhaQ Problem Store.

In-memory storage for generated coding problems.

Security goals:
- Problems are isolated by anonymous BodhaQ session.
- Hidden tests remain backend-only.
- Stored objects are protected from caller-side mutation.
- Access is thread-safe.
- Storage is bounded to prevent unbounded memory growth.

Persistence is intentionally in-memory for the current MVP.
"""

from __future__ import annotations

import copy
import re
import threading
import uuid
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

PROBLEM_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F-]{36}$"
)

MAX_PROBLEMS_PER_SESSION = 50
MAX_TOTAL_PROBLEMS = 500


# ============================================================================
# STORE
# ============================================================================


class ProblemStore:
    """
    Thread-safe, session-isolated in-memory coding problem store.

    Storage format:

        problem_id -> (session_id, problem_object)

    The stored problem object is expected to contain:
        public_tests
        hidden_tests
        statement
        and other coding-problem fields.

    Hidden tests remain entirely inside the backend.
    """

    def __init__(self) -> None:
        self._store: dict[
            str,
            tuple[str, Any],
        ] = {}

        self._lock = threading.RLock()

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _normalize_session_id(
        session_id: str,
    ) -> str:
        """
        Validate and normalize a session ID.
        """

        if not isinstance(
            session_id,
            str,
        ):
            raise ValueError(
                "Session ID must be a string."
            )

        normalized = session_id.strip()

        if not normalized:
            raise ValueError(
                "Session ID cannot be empty."
            )

        try:
            parsed = uuid.UUID(
                normalized
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid session ID."
            ) from exc

        return str(parsed)

    @staticmethod
    def _normalize_problem_id(
        problem_id: str,
    ) -> str:
        """
        Validate and normalize a stored problem ID.
        """

        if not isinstance(
            problem_id,
            str,
        ):
            raise ValueError(
                "Problem ID must be a string."
            )

        normalized = problem_id.strip()

        if not normalized:
            raise ValueError(
                "Problem ID cannot be empty."
            )

        if not PROBLEM_ID_PATTERN.fullmatch(
            normalized
        ):
            raise ValueError(
                "Invalid problem ID."
            )

        try:
            parsed = uuid.UUID(
                normalized
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid problem ID."
            ) from exc

        return str(parsed)

    @staticmethod
    def _validate_test_collection(
        tests: Any,
        *,
        name: str,
    ) -> None:
        """
        Validate that a test collection is a list.

        Individual test schemas are validated by the coding
        service/model layer. This store only guarantees that
        collections have the expected container type.
        """

        if not isinstance(
            tests,
            list,
        ):
            raise ValueError(
                f"{name} must be provided as a list."
            )

    @staticmethod
    def _copy_problem(
        problem_data: Any,
    ) -> Any:
        """
        Return an isolated copy of stored problem data.

        This prevents callers from mutating the store's internal
        object through a previously returned reference.
        """

        try:
            return copy.deepcopy(
                problem_data
            )

        except Exception as exc:
            raise ValueError(
                "Problem data could not be safely copied."
            ) from exc

    def _count_session_problems(
        self,
        session_id: str,
    ) -> int:
        """
        Count problems owned by a session.

        Caller must hold self._lock.
        """

        return sum(
            1
            for owner_session_id, _ in self._store.values()
            if owner_session_id == session_id
        )

    # ========================================================================
    # SAVE
    # ========================================================================

    def save_problem(
        self,
        session_id: str,
        problem_data: Any,
    ) -> str:
        """
        Save a generated coding problem for a specific session.

        Returns:
            Newly generated problem ID.

        Raises:
            ValueError:
                If the session/problem data is invalid or storage
                limits have been reached.
        """

        normalized_session_id = (
            self._normalize_session_id(
                session_id
            )
        )

        if problem_data is None:
            raise ValueError(
                "Problem data cannot be empty."
            )

        if not hasattr(
            problem_data,
            "public_tests",
        ):
            raise ValueError(
                "Problem data must contain public tests."
            )

        if not hasattr(
            problem_data,
            "hidden_tests",
        ):
            raise ValueError(
                "Problem data must contain hidden tests."
            )

        self._validate_test_collection(
            problem_data.public_tests,
            name="Public tests",
        )

        self._validate_test_collection(
            problem_data.hidden_tests,
            name="Hidden tests",
        )

        stored_problem = self._copy_problem(
            problem_data
        )

        problem_id = str(
            uuid.uuid4()
        )

        with self._lock:

            if len(self._store) >= MAX_TOTAL_PROBLEMS:
                raise ValueError(
                    "Problem storage capacity has been reached."
                )

            if (
                self._count_session_problems(
                    normalized_session_id
                )
                >= MAX_PROBLEMS_PER_SESSION
            ):
                raise ValueError(
                    "This session has reached its problem limit."
                )

            self._store[
                problem_id
            ] = (
                normalized_session_id,
                stored_problem,
            )

        return problem_id

    # ========================================================================
    # GET
    # ========================================================================

    def get_problem(
        self,
        session_id: str,
        problem_id: str,
    ) -> Any | None:
        """
        Retrieve a problem belonging to the specified session.

        Returns:
            An isolated copy when the problem exists and belongs
            to the session.

            None when:
            - the problem does not exist
            - the problem belongs to another session
            - the identifiers are invalid
        """

        try:
            normalized_session_id = (
                self._normalize_session_id(
                    session_id
                )
            )

            normalized_problem_id = (
                self._normalize_problem_id(
                    problem_id
                )
            )

        except ValueError:
            return None

        with self._lock:

            stored = self._store.get(
                normalized_problem_id
            )

            if stored is None:
                return None

            owner_session_id, problem = stored

            if owner_session_id != normalized_session_id:
                return None

            return self._copy_problem(
                problem
            )

    # ========================================================================
    # APPEND TESTS
    # ========================================================================

    def append_tests(
        self,
        session_id: str,
        problem_id: str,
        new_public: list,
        new_hidden: list,
    ) -> None:
        """
        Append additional public and hidden tests to a problem
        owned by the specified session.

        Raises:
            ValueError:
                If the problem does not exist, belongs to another
                session, or test collections are invalid.
        """

        normalized_session_id = (
            self._normalize_session_id(
                session_id
            )
        )

        normalized_problem_id = (
            self._normalize_problem_id(
                problem_id
            )
        )

        self._validate_test_collection(
            new_public,
            name="Public tests",
        )

        self._validate_test_collection(
            new_hidden,
            name="Hidden tests",
        )

        with self._lock:

            stored = self._store.get(
                normalized_problem_id
            )

            if stored is None:
                raise ValueError(
                    f"Problem '{normalized_problem_id}' not found."
                )

            owner_session_id, problem = stored

            if owner_session_id != normalized_session_id:
                raise ValueError(
                    f"Problem '{normalized_problem_id}' not found."
                )

            # ----------------------------------------------------------------
            # Validate stored problem structure.
            # ----------------------------------------------------------------

            if not hasattr(
                problem,
                "public_tests",
            ):
                raise ValueError(
                    f"Problem '{normalized_problem_id}' "
                    "does not contain public tests."
                )

            if not hasattr(
                problem,
                "hidden_tests",
            ):
                raise ValueError(
                    f"Problem '{normalized_problem_id}' "
                    "does not contain hidden tests."
                )

            self._validate_test_collection(
                problem.public_tests,
                name="Stored public tests",
            )

            self._validate_test_collection(
                problem.hidden_tests,
                name="Stored hidden tests",
            )

            # ----------------------------------------------------------------
            # Append caller-owned data only after making defensive copies.
            # ----------------------------------------------------------------

            if new_public:
                problem.public_tests.extend(
                    copy.deepcopy(
                        new_public
                    )
                )

            if new_hidden:
                problem.hidden_tests.extend(
                    copy.deepcopy(
                        new_hidden
                    )
                )

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete_problem(
        self,
        session_id: str,
        problem_id: str,
    ) -> bool:
        """
        Delete a problem belonging to the specified session.

        Returns:
            True if deleted.
            False if the problem does not exist or belongs
            to another session.
        """

        try:
            normalized_session_id = (
                self._normalize_session_id(
                    session_id
                )
            )

            normalized_problem_id = (
                self._normalize_problem_id(
                    problem_id
                )
            )

        except ValueError:
            return False

        with self._lock:

            stored = self._store.get(
                normalized_problem_id
            )

            if stored is None:
                return False

            owner_session_id, _ = stored

            if owner_session_id != normalized_session_id:
                return False

            del self._store[
                normalized_problem_id
            ]

        return True

    # ========================================================================
    # CLEAR
    # ========================================================================

    def clear(self) -> None:
        """
        Clear all stored coding problems.

        Primarily useful for tests and development.
        """

        with self._lock:
            self._store.clear()


# ============================================================================
# SINGLETON
# ============================================================================

problem_store = ProblemStore()
