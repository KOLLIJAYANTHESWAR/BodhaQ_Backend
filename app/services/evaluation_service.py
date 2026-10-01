"""
BodhaQ Evaluation Service.

Deterministic quiz scoring.

Scoring is entirely algorithmic.

The LLM is NOT involved in:
- calculating scores
- identifying mistakes
- calculating topic accuracy
- determining learning-gap status

These operations are application logic.

Security:
    - Evaluations are isolated by anonymous BodhaQ session.
    - A quiz can only be evaluated by the session that created it.
    - Learning gaps are calculated only from the current session.
    - No API keys are handled or stored here.

Persistence:
    - Evaluation storage is currently in-memory for the MVP.
    - Evaluations disappear when the backend restarts.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any

from app.models.responses import (
    MistakeDetail,
    QuizEvaluationResponse,
    QuizHistoryItem,
    WeakTopicItem,
    WeakTopicsResponse,
)
from app.services.document_service import document_service
from app.services.quiz_service import quiz_service


# ============================================================================
# CONFIGURATION
# ============================================================================

WEAK_TOPIC_THRESHOLD = 60.0
IMPROVING_TOPIC_THRESHOLD = 80.0

DEFAULT_GAP_HISTORY_LIMIT = 10
MAX_GAP_HISTORY_LIMIT = 100

# Bound in-memory evaluation storage.
MAX_EVALUATIONS_PER_SESSION = 100
MAX_TOTAL_EVALUATIONS = 1000

VALID_ANSWER_LETTERS = {
    "A",
    "B",
    "C",
    "D",
}


# ============================================================================
# SERVICE
# ============================================================================


class EvaluationService:
    """
    Service responsible for deterministic, session-isolated quiz evaluation.
    """

    def __init__(self) -> None:
        # --------------------------------------------------------------------
        # SESSION-ISOLATED EVALUATIONS
        # --------------------------------------------------------------------

        # session_id -> quiz_id -> most recent evaluation

        # This remains in-memory for the current MVP.
        # A server restart intentionally clears evaluation history.

        self._evaluations: dict[
            str,
            dict[str, QuizEvaluationResponse],
        ] = {}

        self._lock = threading.RLock()

    # ========================================================================
    # SESSION HELPERS
    # ========================================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:
        """
        Validate and normalize a BodhaQ session identifier.
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
            uuid.UUID(normalized)
        except ValueError as exc:
            raise ValueError(
                "Invalid session ID."
            ) from exc

        return normalized

    @staticmethod
    def _validate_quiz_id(
        quiz_id: str,
    ) -> str:
        """
        Validate and normalize a quiz identifier.
        """

        if not isinstance(
            quiz_id,
            str,
        ):
            raise ValueError(
                "Quiz ID must be a string."
            )

        normalized = quiz_id.strip()

        if not normalized:
            raise ValueError(
                "Quiz ID cannot be empty."
            )

        try:
            uuid.UUID(normalized)
        except ValueError as exc:
            raise ValueError(
                "Invalid quiz ID."
            ) from exc

        return normalized

    def _get_session_evaluations(
        self,
        session_id: str,
    ) -> dict[str, QuizEvaluationResponse]:
        """
        Return the evaluation store for one session.

        Caller must hold self._lock.
        """

        return self._evaluations.setdefault(
            session_id,
            {},
        )

    # ========================================================================
    # OPTION HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_option_text(
        value: Any,
    ) -> str:
        """
        Convert a stored option value into displayable text.

        Quiz generation may represent options as strings or as small
        dictionaries. This helper keeps evaluation output deterministic
        without involving an LLM.
        """

        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, dict):
            for key in (
                "text",
                "label",
                "value",
                "option",
                "answer",
            ):
                candidate = value.get(key)

                if isinstance(
                    candidate,
                    str,
                ):
                    normalized = candidate.strip()

                    if normalized:
                        return normalized

            return ""

        return str(value).strip()

    @classmethod
    def _get_option_text(
        cls,
        metadata: dict[str, Any],
        answer_letter: str,
    ) -> str:
        """
        Resolve the text belonging to an answer letter.

        Supported stored quiz shapes include:

            {
                "options": {
                    "A": "update",
                    "B": "insert",
                    ...
                }
            }

        and:

            {
                "options": [
                    {"letter": "A", "text": "update"},
                    {"letter": "B", "text": "insert"},
                    ...
                ]
            }

        The helper also supports common alternative option containers so
        evaluation remains compatible with existing generated quiz data.

        Returns an empty string when no matching option text is available.
        """

        normalized_letter = (
            str(answer_letter)
            .strip()
            .upper()
        )

        if (
            not normalized_letter
            or normalized_letter
            not in VALID_ANSWER_LETTERS
        ):
            return ""

        # --------------------------------------------------------------------
        # DIRECT LETTER-SUFFIX FIELDS
        # --------------------------------------------------------------------

        direct_keys = (
            f"option_{normalized_letter.lower()}",
            f"option_{normalized_letter}",
            f"{normalized_letter.lower()}",
            normalized_letter,
        )

        for key in direct_keys:
            if key in metadata:
                text = cls._normalize_option_text(
                    metadata.get(key)
                )

                if text:
                    return text

        # --------------------------------------------------------------------
        # OPTION CONTAINERS
        # --------------------------------------------------------------------

        option_container = None

        for key in (
            "options",
            "choices",
            "answer_options",
            "answers",
        ):
            candidate = metadata.get(key)

            if candidate is not None:
                option_container = candidate
                break

        if option_container is None:
            return ""

        # --------------------------------------------------------------------
        # DICTIONARY:
        #
        # {
        #     "A": "text",
        #     "B": "text"
        # }
        # --------------------------------------------------------------------

        if isinstance(
            option_container,
            dict,
        ):
            direct = option_container.get(
                normalized_letter
            )

            if direct is None:
                direct = option_container.get(
                    normalized_letter.lower()
                )

            if direct is None:
                direct = option_container.get(
                    normalized_letter.upper()
                )

            text = cls._normalize_option_text(
                direct
            )

            if text:
                return text

            # Some generators may use numeric indexes.
            index = (
                ord(normalized_letter)
                - ord("A")
            )

            numeric_candidate = option_container.get(
                index
            )

            text = cls._normalize_option_text(
                numeric_candidate
            )

            if text:
                return text

            return ""

        # --------------------------------------------------------------------
        # LIST:
        #
        # [
        #     {"letter": "A", "text": "..."},
        #     {"letter": "B", "text": "..."}
        # ]
        #
        # or:
        #
        # ["text A", "text B", "text C", "text D"]
        # --------------------------------------------------------------------

        if isinstance(
            option_container,
            list,
        ):
            index = (
                ord(normalized_letter)
                - ord("A")
            )

            # First prefer explicit letter metadata.
            for option in option_container:

                if not isinstance(
                    option,
                    dict,
                ):
                    continue

                option_letter = str(
                    option.get(
                        "letter",
                        option.get(
                            "key",
                            option.get(
                                "id",
                                "",
                            ),
                        ),
                    )
                ).strip().upper()

                if option_letter != normalized_letter:
                    continue

                text = cls._normalize_option_text(
                    option
                )

                if text:
                    return text

            # Fall back to positional options.
            if (
                0 <= index
                < len(option_container)
            ):
                text = cls._normalize_option_text(
                    option_container[index]
                )

                if text:
                    return text

        return ""

    # ========================================================================
    # QUIZ EVALUATION
    # ========================================================================

    def evaluate(
        self,
        session_id: str,
        quiz_id: str,
        user_answers: dict[str, str],
    ) -> QuizEvaluationResponse:
        """
        Compare user answers against stored correct answers.

        The quiz must belong to the supplied session.

        Args:
            session_id:
                Current anonymous BodhaQ session.

            quiz_id:
                The quiz being submitted.

            user_answers:
                Mapping of question ID to selected answer letter.

        Returns:
            QuizEvaluationResponse containing:
                - score
                - total
                - percentage
                - mistakes
                - timestamp

        Raises:
            ValueError:
                If the session, quiz, or stored quiz data is invalid.
        """

        normalized_session_id = (
            self._validate_session_id(
                session_id
            )
        )

        normalized_quiz_id = (
            self._validate_quiz_id(
                quiz_id
            )
        )

        # --------------------------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------------------------

        if not isinstance(
            user_answers,
            dict,
        ):
            raise ValueError(
                "User answers must be a dictionary."
            )

        # --------------------------------------------------------------------
        # LOAD PRIVATE QUIZ DATA
        # --------------------------------------------------------------------

        stored = quiz_service.get_stored_quiz(
            session_id=normalized_session_id,
            quiz_id=normalized_quiz_id,
        )

        if stored is None:
            raise ValueError(
                f"Quiz '{normalized_quiz_id}' not found. "
                "It may have expired because the server restarted "
                "or reached its in-memory storage limit."
            )

        # Defense in depth.
        if stored.get(
            "session_id"
        ) != normalized_session_id:
            raise ValueError(
                "Quiz does not belong to the current session."
            )

        private_answers = stored.get(
            "answers",
            {},
        )

        if not isinstance(
            private_answers,
            dict,
        ):
            raise ValueError(
                f"Quiz '{normalized_quiz_id}' contains invalid answer data."
            )

        total = len(private_answers)

        if total == 0:
            raise ValueError(
                f"Quiz '{normalized_quiz_id}' contains no questions."
            )

        # --------------------------------------------------------------------
        # SCORE QUIZ
        # --------------------------------------------------------------------

        score = 0

        mistakes: list[MistakeDetail] = []

        for question_id, metadata in private_answers.items():

            # ---------------------------------------------------------------
            # QUESTION ID VALIDATION
            # ---------------------------------------------------------------

            try:
                normalized_question_id = int(
                    question_id
                )

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Quiz '{normalized_quiz_id}' contains "
                    "an invalid question ID."
                ) from exc

            if normalized_question_id < 1:
                raise ValueError(
                    f"Quiz '{normalized_quiz_id}' contains "
                    "an invalid question ID."
                )

            # ---------------------------------------------------------------
            # QUESTION METADATA VALIDATION
            # ---------------------------------------------------------------

            if not isinstance(
                metadata,
                dict,
            ):
                raise ValueError(
                    f"Quiz '{normalized_quiz_id}' contains "
                    "invalid question metadata."
                )

            correct_answer = str(
                metadata.get(
                    "correct_answer",
                    "",
                )
            ).strip().upper()

            if (
                not correct_answer
                or correct_answer
                not in VALID_ANSWER_LETTERS
            ):
                raise ValueError(
                    f"Quiz '{normalized_quiz_id}' contains "
                    "an invalid correct answer."
                )

            question = str(
                metadata.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                raise ValueError(
                    f"Quiz '{normalized_quiz_id}' contains "
                    "a question with no question text."
                )

            explanation = str(
                metadata.get(
                    "explanation",
                    "",
                )
            ).strip()

            topic = str(
                metadata.get(
                    "topic",
                    "General",
                )
            ).strip() or "General"

            # ---------------------------------------------------------------
            # OPTION TEXT
            # ---------------------------------------------------------------

            correct_answer_text = (
                self._get_option_text(
                    metadata=metadata,
                    answer_letter=correct_answer,
                )
            )

            # ---------------------------------------------------------------
            # USER ANSWER
            # ---------------------------------------------------------------

            raw_user_answer = user_answers.get(
                str(question_id),
                "",
            )

            user_answer = str(
                raw_user_answer
            ).strip().upper()

            # Missing answers are treated as unanswered.
            if (
                user_answer
                and user_answer
                not in VALID_ANSWER_LETTERS
            ):
                raise ValueError(
                    f"Invalid answer for question "
                    f"{normalized_question_id}."
                )

            user_answer_text = ""

            if user_answer:
                user_answer_text = (
                    self._get_option_text(
                        metadata=metadata,
                        answer_letter=user_answer,
                    )
                )

            # ---------------------------------------------------------------
            # CORRECT
            # ---------------------------------------------------------------

            if user_answer == correct_answer:
                score += 1
                continue

            # ---------------------------------------------------------------
            # INCORRECT / UNANSWERED
            # ---------------------------------------------------------------

            mistakes.append(
                MistakeDetail(
                    question_id=normalized_question_id,
                    question=question,
                    correct_answer=correct_answer,
                    correct_answer_text=(
                        correct_answer_text
                        or None
                    ),
                    user_answer=(
                        user_answer
                        or "(no answer)"
                    ),
                    user_answer_text=(
                        user_answer_text
                        or None
                    ),
                    explanation=explanation,
                    topic=topic,
                )
            )

        # --------------------------------------------------------------------
        # CALCULATE SCORE
        # --------------------------------------------------------------------

        percentage = round(
            (score / total) * 100,
            1,
        )

        evaluation = QuizEvaluationResponse(
            quiz_id=normalized_quiz_id,
            score=score,
            total=total,
            percentage=percentage,
            mistakes=mistakes,
            timestamp=time.time(),
        )

        # --------------------------------------------------------------------
        # STORE SESSION-ISOLATED EVALUATION
        # --------------------------------------------------------------------

        with self._lock:
            session_evaluations = (
                self._get_session_evaluations(
                    normalized_session_id
                )
            )

            session_evaluations[
                normalized_quiz_id
            ] = evaluation

            self._enforce_storage_limits_locked(
                preferred_session_id=normalized_session_id,
            )

        return evaluation

    # ========================================================================
    # EVALUATION STORAGE LIMITS
    # ========================================================================

    def _enforce_storage_limits_locked(
        self,
        preferred_session_id: str,
    ) -> None:
        """
        Bound in-memory evaluation storage.

        Caller must hold self._lock.
        """

        def total_count() -> int:
            return sum(
                len(session_evaluations)
                for session_evaluations
                in self._evaluations.values()
            )

        # --------------------------------------------------------------------
        # PER-SESSION LIMIT
        # --------------------------------------------------------------------

        session_evaluations = self._evaluations.get(
            preferred_session_id
        )

        if session_evaluations is not None:

            while (
                len(session_evaluations)
                > MAX_EVALUATIONS_PER_SESSION
            ):
                oldest_quiz_id = min(
                    session_evaluations,
                    key=lambda quiz_id: (
                        session_evaluations[
                            quiz_id
                        ].timestamp,
                        quiz_id,
                    ),
                )

                del session_evaluations[
                    oldest_quiz_id
                ]

        # --------------------------------------------------------------------
        # GLOBAL LIMIT
        # --------------------------------------------------------------------

        while total_count() > MAX_TOTAL_EVALUATIONS:

            oldest_session_id: str | None = None
            oldest_quiz_id: str | None = None
            oldest_timestamp = float(
                "inf"
            )

            for session_id, session_evaluations in (
                self._evaluations.items()
            ):
                for quiz_id, evaluation in (
                    session_evaluations.items()
                ):
                    if (
                        evaluation.timestamp
                        < oldest_timestamp
                        or (
                            evaluation.timestamp
                            == oldest_timestamp
                            and (
                                oldest_session_id is None
                                or (
                                    session_id,
                                    quiz_id,
                                )
                                < (
                                    oldest_session_id,
                                    oldest_quiz_id or "",
                                )
                            )
                        )
                    ):
                        oldest_timestamp = (
                            evaluation.timestamp
                        )
                        oldest_session_id = session_id
                        oldest_quiz_id = quiz_id

            if (
                oldest_session_id is None
                or oldest_quiz_id is None
            ):
                break

            session_evaluations = self._evaluations.get(
                oldest_session_id
            )

            if session_evaluations is None:
                continue

            session_evaluations.pop(
                oldest_quiz_id,
                None,
            )

            if not session_evaluations:
                self._evaluations.pop(
                    oldest_session_id,
                    None,
                )

    # ========================================================================
    # GET EVALUATION
    # ========================================================================

    def get_evaluation(
        self,
        session_id: str,
        quiz_id: str,
    ) -> QuizEvaluationResponse | None:
        """
        Return the most recent evaluation for a session-owned quiz.
        """

        try:
            normalized_session_id = (
                self._validate_session_id(
                    session_id
                )
            )

            normalized_quiz_id = (
                self._validate_quiz_id(
                    quiz_id
                )
            )

        except ValueError:
            return None

        with self._lock:
            session_evaluations = (
                self._evaluations.get(
                    normalized_session_id
                )
            )

            if not session_evaluations:
                return None

            evaluation = session_evaluations.get(
                normalized_quiz_id
            )

            if evaluation is None:
                return None

            return copy.deepcopy(
                evaluation
            )

    # ========================================================================
    # AGGREGATED LEARNING GAPS
    # ========================================================================

    def get_aggregated_gaps(
        self,
        session_id: str,
        limit: int = DEFAULT_GAP_HISTORY_LIMIT,
    ) -> WeakTopicsResponse:
        """
        Aggregate learning gaps from the most recent completed quizzes
        belonging only to the supplied session.
        """

        normalized_session_id = (
            self._validate_session_id(
                session_id
            )
        )

        # --------------------------------------------------------------------
        # LIMIT VALIDATION
        # --------------------------------------------------------------------

        if isinstance(
            limit,
            bool,
        ) or not isinstance(
            limit,
            int,
        ):
            raise ValueError(
                "Gap history limit must be an integer."
            )

        if limit < 1:
            raise ValueError(
                "Gap history limit must be at least 1."
            )

        limit = min(
            limit,
            MAX_GAP_HISTORY_LIMIT,
        )

        # --------------------------------------------------------------------
        # LOAD SESSION EVALUATIONS
        # --------------------------------------------------------------------

        with self._lock:
            session_evaluations = (
                self._evaluations.get(
                    normalized_session_id,
                    {},
                )
            )

            evaluations = list(
                session_evaluations.values()
            )

        evaluations.sort(
            key=lambda evaluation: evaluation.timestamp,
            reverse=True,
        )

        # --------------------------------------------------------------------
        # BUILD HISTORY + TOPIC STATISTICS
        # --------------------------------------------------------------------

        recent_quizzes: list[QuizHistoryItem] = []

        topic_stats: dict[
            str,
            dict[str, Any],
        ] = {}

        for evaluation in evaluations:

            # Stop after collecting the requested number of valid
            # completed quizzes.
            if len(recent_quizzes) >= limit:
                break

            # Do not count evaluations whose underlying quiz is no longer
            # available.
            stored = quiz_service.get_stored_quiz(
                session_id=normalized_session_id,
                quiz_id=evaluation.quiz_id,
            )

            if not stored:
                continue

            # Defense in depth.
            if stored.get(
                "session_id"
            ) != normalized_session_id:
                continue

            # ---------------------------------------------------------------
            # DETERMINE QUIZ TITLE
            # ---------------------------------------------------------------

            source_type = str(
                stored.get(
                    "source_type",
                    "",
                )
            ).strip()

            source_id = str(
                stored.get(
                    "source_id",
                    "",
                )
            ).strip()

            title = self._get_quiz_title(
                session_id=normalized_session_id,
                source_type=source_type,
                source_id=source_id,
            )

            recent_quizzes.append(
                QuizHistoryItem(
                    quiz_id=evaluation.quiz_id,
                    title=title,
                    score=evaluation.score,
                    total=evaluation.total,
                    percentage=evaluation.percentage,
                    timestamp=evaluation.timestamp,
                )
            )

            # ---------------------------------------------------------------
            # PRIVATE ANSWERS
            # ---------------------------------------------------------------

            private_answers = stored.get(
                "answers",
                {},
            )

            if not isinstance(
                private_answers,
                dict,
            ):
                continue

            # Build a lookup of incorrectly answered question IDs from the
            # deterministic evaluation result.
            mistake_ids = {
                mistake.question_id
                for mistake in evaluation.mistakes
            }

            # ---------------------------------------------------------------
            # TOPIC STATISTICS
            # ---------------------------------------------------------------

            for question_id, metadata in private_answers.items():

                if not isinstance(
                    metadata,
                    dict,
                ):
                    continue

                topic = str(
                    metadata.get(
                        "topic",
                        "General",
                    )
                ).strip() or "General"

                if topic not in topic_stats:
                    topic_stats[topic] = {
                        "correct": 0,
                        "total": 0,
                        "quizzes": set(),
                        "sources": set(),
                    }

                stats = topic_stats[
                    topic
                ]

                stats["total"] += 1

                stats["quizzes"].add(
                    evaluation.quiz_id
                )

                if source_type or source_id:
                    stats["sources"].add(
                        (
                            source_type,
                            source_id,
                        )
                    )

                try:
                    normalized_question_id = int(
                        question_id
                    )
                except (TypeError, ValueError):
                    continue

                if (
                    normalized_question_id
                    not in mistake_ids
                ):
                    stats["correct"] += 1

        # --------------------------------------------------------------------
        # BUILD WEAK TOPICS
        # --------------------------------------------------------------------

        weak_topics: list[WeakTopicItem] = []

        for topic, stats in topic_stats.items():

            total = stats["total"]
            correct = stats["correct"]

            accuracy = (
                round(
                    (correct / total) * 100,
                    1,
                )
                if total > 0
                else 0.0
            )

            # ---------------------------------------------------------------
            # STATUS
            # ---------------------------------------------------------------

            if accuracy < WEAK_TOPIC_THRESHOLD:
                topic_status = "Needs Practice"

            elif accuracy < IMPROVING_TOPIC_THRESHOLD:
                topic_status = "Improving"

            else:
                topic_status = "Learned"

            # ---------------------------------------------------------------
            # SOURCE
            # ---------------------------------------------------------------

            source_type, source_id = (
                self._select_topic_source(
                    stats["sources"]
                )
            )

            weak_topics.append(
                WeakTopicItem(
                    topic=topic,
                    accuracy=accuracy,
                    status=topic_status,
                    source_type=source_type,
                    source_id=source_id,
                    quiz_count=len(
                        stats["quizzes"]
                    ),
                )
            )

        # --------------------------------------------------------------------
        # SORT
        # --------------------------------------------------------------------

        weak_topics.sort(
            key=lambda item: (
                item.accuracy,
                -item.quiz_count,
                item.topic.lower(),
            )
        )

        # --------------------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------------------

        return WeakTopicsResponse(
            weak_topics=weak_topics,
            recent_quizzes=recent_quizzes,
        )

    # ========================================================================
    # QUIZ TITLE
    # ========================================================================

    @staticmethod
    def _get_quiz_title(
        session_id: str,
        source_type: str,
        source_id: str,
    ) -> str:
        """
        Resolve a user-friendly title for a completed quiz.

        Document lookup is restricted to the current session.

        Resume-item quizzes use their stored source ID as a safe fallback.
        """

        if source_type == "document":

            document = document_service.get_document(
                session_id=session_id,
                document_id=source_id,
            )

            if document is not None:
                return document.filename

        if source_type == "resume_item":
            return "Resume Interview Assessment"

        return source_id or "Quiz"

    # ========================================================================
    # TOPIC SOURCE
    # ========================================================================

    @staticmethod
    def _select_topic_source(
        sources: set[tuple[str, str]],
    ) -> tuple[str, str]:
        """
        Select a deterministic representative source for a topic.

        Preference:
            1. Document source
            2. Resume-item source
            3. Other source
            4. Empty source
        """

        if not sources:
            return "", ""

        document_sources = sorted(
            (
                source
                for source in sources
                if source[0] == "document"
            ),
            key=lambda item: item[1],
        )

        if document_sources:
            return document_sources[0]

        resume_sources = sorted(
            (
                source
                for source in sources
                if source[0] == "resume_item"
            ),
            key=lambda item: item[1],
        )

        if resume_sources:
            return resume_sources[0]

        return sorted(
            sources,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )[0]


# ============================================================================
# SINGLETON
# ============================================================================

evaluation_service = EvaluationService()