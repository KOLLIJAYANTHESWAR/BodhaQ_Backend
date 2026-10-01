"""
BodhaQ Quiz Service.

Responsibilities:
- Generate quizzes from topics or documents.
- Generate targeted practice from weak topics.
- Generate resume-item interview assessments.
- Store correct answers server-side only.
- Return public quiz data without answer leakage.
- Provide session-owned quiz data to the deterministic evaluation service.

Key design decisions:
- Correct answers are never returned to the frontend.
- Scoring is performed deterministically by EvaluationService.
- Gemini is used only for quiz/content generation.
- Document-based quizzes use session-isolated RAG context.
- Targeted practice supports:
    1. Weak topic + original document using session-isolated RAG.
    2. Weak topic alone using Gemini-generated topic context.
- Quiz storage is isolated by anonymous BodhaQ session.
- Resume-item quizzes are also session-isolated.

Persistence:
- Quiz storage is currently in-memory for the MVP.
- Quizzes disappear when the backend restarts.
- API keys are never stored in quiz data.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from typing import Any

from app.models.responses import (
    QuizGenerateResponse,
    QuizOption,
    QuizQuestionPublic,
)
from app.services.gemini_service import gemini_service
from app.services.rag_service import rag_service


logger = logging.getLogger(__name__)


# ============================================================================
# IN-MEMORY QUIZ STORAGE
# ============================================================================

# session_id -> quiz_id -> complete private quiz data
#
# IMPORTANT:
# This dictionary contains correct answers.
# It must never be returned directly by an API endpoint.
#
# For the current MVP this is intentionally in-memory.
_quiz_store: dict[
    str,
    dict[str, dict[str, Any]],
] = {}

_quiz_store_lock = threading.RLock()


# ============================================================================
# CONSTANTS
# ============================================================================

OPTION_LETTERS = (
    "A",
    "B",
    "C",
    "D",
)

VALID_SOURCE_TYPES = {
    "topic",
    "document",
    "resume_item",
}

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

MIN_QUESTIONS = 1
MAX_QUESTIONS = 20

MAX_SOURCE_ID_LENGTH = 500
MAX_TOPIC_LENGTH = 500

# Prevent unbounded in-memory growth during long-running deployments.
MAX_QUIZZES_PER_SESSION = 50
MAX_TOTAL_QUIZZES = 500

# Defensive limits for model-generated content stored in memory.
MAX_QUESTION_LENGTH = 5000
MAX_OPTION_LENGTH = 2000
MAX_EXPLANATION_LENGTH = 5000
MAX_TOPIC_VALUE_LENGTH = 500


# ============================================================================
# SERVICE
# ============================================================================


class QuizService:
    """
    Service responsible for quiz generation and private,
    session-isolated quiz storage.
    """

    # ========================================================================
    # STANDARD QUIZ GENERATION
    # ========================================================================

    def generate_quiz(
        self,
        session_id: str,
        source_type: str,
        source_id: str,
        num_questions: int,
        difficulty: str,
        api_key: str,
    ) -> QuizGenerateResponse:
        """
        Generate a quiz from a topic or document.

        source_type='topic':
            source_id is the topic text.

        source_type='document':
            source_id is the document_id and session-isolated RAG is used.

        Correct answers are stored privately and never returned
        in the public response.
        """

        normalized_session_id = (
            self._validate_session_id(
                session_id
            )
        )

        source_type = self._normalize_type(
            source_type
        )

        source_id = self._normalize_source_id(
            source_id
        )

        difficulty = self._normalize_type(
            difficulty
        )

        normalized_api_key = self._validate_api_key(
            api_key
        )

        self._validate_request(
            source_type=source_type,
            source_id=source_id,
            num_questions=num_questions,
            difficulty=difficulty,
        )

        logger.info(
            "[Quiz] Request received | "
            "source_type=%s | questions=%d | difficulty=%s",
            source_type,
            num_questions,
            difficulty,
        )

        start_time = time.perf_counter()

        # --------------------------------------------------------------------
        # 1. BUILD QUIZ CONTEXT
        # --------------------------------------------------------------------

        context_start = time.perf_counter()

        if source_type == "topic":

            context = self._get_topic_context(
                source_id,
                api_key=normalized_api_key,
            )

            topic_hint = source_id

        else:

            context = rag_service.get_context_for_quiz(
                session_id=normalized_session_id,
                document_id=source_id,
                api_key=normalized_api_key,
            )

            topic_hint = ""

        context_time = (
            time.perf_counter()
            - context_start
        )

        if not context or not context.strip():
            raise RuntimeError(
                "No study material was available to generate the quiz."
            )

        # --------------------------------------------------------------------
        # 2. GENERATE QUIZ USING GEMINI
        # --------------------------------------------------------------------

        gemini_start = time.perf_counter()

        raw_quiz = gemini_service.generate_quiz(
            context=context,
            topic_hint=topic_hint,
            num_questions=num_questions,
            difficulty=difficulty,
            api_key=normalized_api_key,
        )

        gemini_time = (
            time.perf_counter()
            - gemini_start
        )

        if not raw_quiz.questions:
            raise RuntimeError(
                "AI service returned no quiz questions."
            )

        # --------------------------------------------------------------------
        # 3. VALIDATE, STORE, AND RETURN
        # --------------------------------------------------------------------

        response, validation_time, storage_time = (
            self._build_quiz_response_with_timing(
                raw_quiz=raw_quiz,
                quiz_id=str(uuid.uuid4()),
                session_id=normalized_session_id,
                source_type=source_type,
                source_id=source_id,
                num_questions=num_questions,
                topic_hint=topic_hint,
            )
        )

        total_time = (
            time.perf_counter()
            - start_time
        )

        logger.info(
            "[Quiz] Context=%.2fs | Gemini=%.2fs | "
            "Validation=%.2fs | Storage=%.2fs | Total=%.2fs",
            context_time,
            gemini_time,
            validation_time,
            storage_time,
            total_time,
        )

        return response

    # ========================================================================
    # TARGETED PRACTICE
    # ========================================================================

    def generate_targeted_practice(
        self,
        session_id: str,
        topic: str,
        document_id: str | None,
        num_questions: int,
        difficulty: str,
        api_key: str,
    ) -> QuizGenerateResponse:
        """
        Generate targeted practice for a weak topic.

        DOCUMENT MODE
        -------------
        topic + document_id

        The weak topic is used as a semantic retrieval query against
        the original document belonging to the current session.

        TOPIC MODE
        ----------
        topic without document_id

        The weak topic is used to generate a short study context,
        which is then passed to Gemini.

        Correct answers remain server-side in both modes.
        """

        normalized_session_id = (
            self._validate_session_id(
                session_id
            )
        )

        topic = self._normalize_topic(
            topic
        )

        if len(topic) > MAX_TOPIC_LENGTH:
            raise ValueError(
                "topic exceeds the maximum allowed length."
            )

        if document_id is not None:

            document_id = self._normalize_source_id(
                document_id
            )

            if not document_id:
                document_id = None

            elif len(document_id) > MAX_SOURCE_ID_LENGTH:
                raise ValueError(
                    "document_id exceeds the maximum allowed length."
                )

        difficulty = self._normalize_type(
            difficulty
        )

        normalized_api_key = self._validate_api_key(
            api_key
        )

        # --------------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------------

        if not topic:
            raise ValueError(
                "topic cannot be empty."
            )

        self._validate_question_count(
            num_questions
        )

        self._validate_difficulty(
            difficulty
        )

        # --------------------------------------------------------------------
        # DOCUMENT-SOURCED PRACTICE
        # --------------------------------------------------------------------

        if document_id:

            context, _sources = (
                rag_service.get_context_for_question(
                    session_id=normalized_session_id,
                    document_id=document_id,
                    question=topic,
                    api_key=normalized_api_key,
                )
            )

            if not context or not context.strip():
                raise RuntimeError(
                    "No relevant study material was found for the "
                    "requested weak topic."
                )

            raw_quiz = gemini_service.generate_quiz(
                context=context,
                topic_hint=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                api_key=normalized_api_key,
            )

            if not raw_quiz.questions:
                raise RuntimeError(
                    "AI service returned no practice questions."
                )

            return self._build_quiz_response(
                raw_quiz=raw_quiz,
                quiz_id=str(uuid.uuid4()),
                session_id=normalized_session_id,
                source_type="document",
                source_id=document_id,
                num_questions=num_questions,
                topic_hint=topic,
            )

        # --------------------------------------------------------------------
        # TOPIC-ONLY PRACTICE
        # --------------------------------------------------------------------

        context = self._get_topic_context(
            topic,
            api_key=normalized_api_key,
        )

        if not context or not context.strip():
            raise RuntimeError(
                "No study material was available for the requested "
                "practice topic."
            )

        raw_quiz = gemini_service.generate_quiz(
            context=context,
            topic_hint=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            api_key=normalized_api_key,
        )

        if not raw_quiz.questions:
            raise RuntimeError(
                "AI service returned no practice questions."
            )

        return self._build_quiz_response(
            raw_quiz=raw_quiz,
            quiz_id=str(uuid.uuid4()),
            session_id=normalized_session_id,
            source_type="topic",
            source_id=topic,
            num_questions=num_questions,
            topic_hint=topic,
        )

    # ========================================================================
    # PRIVATE QUIZ RETRIEVAL
    # ========================================================================

    def get_stored_quiz(
        self,
        session_id: str,
        quiz_id: str,
    ) -> dict[str, Any] | None:
        """
        Return complete server-side quiz data belonging to one session.

        This method is intended for internal services such as
        EvaluationService.

        IMPORTANT:
        Never expose this object directly through an API response because
        it contains correct answers.

        A defensive deep copy is returned so callers cannot accidentally
        mutate the private store.
        """

        try:
            normalized_session_id = (
                self._validate_session_id(
                    session_id
                )
            )

        except ValueError:
            return None

        normalized_id = self._normalize_quiz_id(
            quiz_id
        )

        if not normalized_id:
            return None

        with _quiz_store_lock:

            session_store = _quiz_store.get(
                normalized_session_id
            )

            if session_store is None:
                return None

            stored = session_store.get(
                normalized_id
            )

            if stored is None:
                return None

            # Defense in depth: verify ownership even though the dictionary
            # is already session-scoped.
            if stored.get(
                "session_id"
            ) != normalized_session_id:
                return None

            return copy.deepcopy(
                stored
            )

    # ========================================================================
    # REQUEST VALIDATION
    # ========================================================================

    def _validate_request(
        self,
        source_type: str,
        source_id: str,
        num_questions: int,
        difficulty: str,
    ) -> None:
        """
        Validate quiz-generation parameters.
        """

        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type '{source_type}'. "
                "Expected 'topic' or 'document'."
            )

        if source_type == "resume_item":
            raise ValueError(
                "resume_item quizzes must be generated "
                "through the resume service."
            )

        if not source_id:
            raise ValueError(
                "source_id cannot be empty."
            )

        if len(source_id) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(
                "source_id exceeds the maximum allowed length."
            )

        self._validate_question_count(
            num_questions
        )

        self._validate_difficulty(
            difficulty
        )

    @staticmethod
    def _validate_question_count(
        num_questions: int,
    ) -> None:
        """
        Validate requested question count.
        """

        if isinstance(
            num_questions,
            bool,
        ):
            raise ValueError(
                "number_of_questions must be an integer."
            )

        if not isinstance(
            num_questions,
            int,
        ):
            raise ValueError(
                "number_of_questions must be an integer."
            )

        if (
            num_questions < MIN_QUESTIONS
            or num_questions > MAX_QUESTIONS
        ):
            raise ValueError(
                "number_of_questions must be between "
                f"{MIN_QUESTIONS} and {MAX_QUESTIONS}."
            )

    @staticmethod
    def _validate_difficulty(
        difficulty: str,
    ) -> None:
        """
        Validate quiz difficulty.
        """

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                "Expected 'easy', 'medium', or 'hard'."
            )

    @staticmethod
    def _validate_api_key(
        api_key: str,
    ) -> str:
        """
        Validate the request-scoped Gemini API key.

        The key is never logged or persisted.
        """

        if not isinstance(
            api_key,
            str,
        ):
            raise ValueError(
                "Gemini API key cannot be empty."
            )

        normalized = api_key.strip()

        if not normalized:
            raise ValueError(
                "Gemini API key cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:
        """
        Validate and normalize an anonymous BodhaQ session ID.
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

    # ========================================================================
    # TOPIC CONTEXT
    # ========================================================================

    def _get_topic_context(
        self,
        topic: str,
        api_key: str,
    ) -> str:
        """
        Generate a short topic explanation to provide Gemini
        with source material for quiz generation.
        """

        explanation = gemini_service.generate_explanation(
            topic,
            api_key=api_key,
        )

        explanation = (
            explanation or ""
        ).strip()

        if not explanation:
            return ""

        return (
            f"Topic: {topic}\n\n"
            f"{explanation}"
        )

    # ========================================================================
    # QUIZ RESPONSE CONSTRUCTION
    # ========================================================================

    def _build_quiz_response(
        self,
        raw_quiz: Any,
        quiz_id: str,
        session_id: str,
        source_type: str,
        source_id: str,
        num_questions: int,
        topic_hint: str,
    ) -> QuizGenerateResponse:
        """
        Build a public quiz response and store private answers.
        """

        response, _, _ = (
            self._build_quiz_response_with_timing(
                raw_quiz=raw_quiz,
                quiz_id=quiz_id,
                session_id=session_id,
                source_type=source_type,
                source_id=source_id,
                num_questions=num_questions,
                topic_hint=topic_hint,
            )
        )

        return response

    def _build_quiz_response_with_timing(
        self,
        raw_quiz: Any,
        quiz_id: str,
        session_id: str,
        source_type: str,
        source_id: str,
        num_questions: int,
        topic_hint: str,
    ) -> tuple[
        QuizGenerateResponse,
        float,
        float,
    ]:
        """
        Validate Gemini's quiz output, store private answers,
        and construct the public response.

        Returns:
            (
                public_response,
                validation_time_seconds,
                storage_time_seconds,
            )
        """

        normalized_session_id = (
            self._validate_session_id(
                session_id
            )
        )

        normalized_quiz_id = self._validate_quiz_id(
            quiz_id
        )

        normalized_source_type = (
            self._normalize_type(
                source_type
            )
        )

        if normalized_source_type not in VALID_SOURCE_TYPES:
            raise RuntimeError(
                "Invalid quiz source type."
            )

        normalized_source_id = (
            self._normalize_source_id(
                source_id
            )
        )

        if not normalized_source_id:
            raise RuntimeError(
                "Quiz source ID cannot be empty."
            )

        if len(normalized_source_id) > MAX_SOURCE_ID_LENGTH:
            raise RuntimeError(
                "Quiz source ID exceeds the maximum allowed length."
            )

        validation_start = time.perf_counter()

        if not raw_quiz or not raw_quiz.questions:
            raise RuntimeError(
                "AI service returned no quiz questions."
            )

        questions = list(
            raw_quiz.questions
        )

        # --------------------------------------------------------------------
        # QUESTION COUNT
        # --------------------------------------------------------------------

        if len(questions) != num_questions:
            raise RuntimeError(
                f"AI service returned {len(questions)} "
                f"questions instead of the requested "
                f"{num_questions}."
            )

        public_questions: list[
            QuizQuestionPublic
        ] = []

        private_answers: dict[
            str,
            dict[str, Any],
        ] = {}

        seen_question_ids: set[int] = set()

        # --------------------------------------------------------------------
        # PROCESS QUESTIONS
        # --------------------------------------------------------------------

        for expected_id, question in enumerate(
            questions,
            start=1,
        ):

            # ---------------------------------------------------------------
            # QUESTION ID
            # ---------------------------------------------------------------

            try:
                question_id = int(
                    question.id
                )

            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "AI service returned an invalid question ID."
                ) from exc

            if question_id <= 0:
                raise RuntimeError(
                    "AI service returned an invalid question ID."
                )

            if question_id in seen_question_ids:
                raise RuntimeError(
                    "AI service returned duplicate question IDs."
                )

            if question_id != expected_id:
                raise RuntimeError(
                    "AI service returned non-sequential question IDs."
                )

            seen_question_ids.add(
                question_id
            )

            # ---------------------------------------------------------------
            # QUESTION TEXT
            # ---------------------------------------------------------------

            question_text = (
                str(
                    question.question or ""
                ).strip()
            )

            if not question_text:
                raise RuntimeError(
                    f"Question {question_id} has empty question text."
                )

            if len(question_text) > MAX_QUESTION_LENGTH:
                raise RuntimeError(
                    f"Question {question_id} exceeds the maximum "
                    "allowed length."
                )

            # ---------------------------------------------------------------
            # OPTIONS
            # ---------------------------------------------------------------

            options = self._parse_options(
                question.options
            )

            option_texts = [
                option.text.strip().lower()
                for option in options
            ]

            if len(set(option_texts)) != len(
                option_texts
            ):
                raise RuntimeError(
                    f"Question {question_id} contains duplicate options."
                )

            # ---------------------------------------------------------------
            # CORRECT ANSWER
            # ---------------------------------------------------------------

            correct_answer = (
                str(
                    question.correct_answer or ""
                ).strip().upper()
            )

            if correct_answer not in OPTION_LETTERS:
                raise RuntimeError(
                    f"Question {question_id} has an invalid correct answer."
                )

            option_letters = {
                option.letter
                for option in options
            }

            if correct_answer not in option_letters:
                raise RuntimeError(
                    f"Question {question_id} has a correct answer "
                    "that does not match the available options."
                )

            # ---------------------------------------------------------------
            # EXPLANATION
            # ---------------------------------------------------------------

            explanation = (
                str(
                    question.explanation or ""
                ).strip()
            )

            if not explanation:
                raise RuntimeError(
                    f"Question {question_id} has an empty explanation."
                )

            if len(explanation) > MAX_EXPLANATION_LENGTH:
                raise RuntimeError(
                    f"Question {question_id} explanation exceeds "
                    "the maximum allowed length."
                )

            # ---------------------------------------------------------------
            # TOPIC
            # ---------------------------------------------------------------

            topic = (
                str(
                    question.topic or ""
                ).strip()
            )

            if not topic:
                topic = (
                    topic_hint.strip()
                    if topic_hint
                    else "General"
                )

            if len(topic) > MAX_TOPIC_VALUE_LENGTH:
                topic = topic[:MAX_TOPIC_VALUE_LENGTH].rstrip()

            # ---------------------------------------------------------------
            # PUBLIC QUESTION
            # ---------------------------------------------------------------

            public_questions.append(
                QuizQuestionPublic(
                    id=question_id,
                    question=question_text,
                    options=options,
                    topic=topic,
                )
            )

            # ---------------------------------------------------------------
            # PRIVATE ANSWER STORAGE
            # ---------------------------------------------------------------

            # IMPORTANT:
            #
            # correct_answer is stored ONLY here.
            #
            # The validated option text is also stored ONLY here so that
            # EvaluationService can deterministically translate answer
            # letters such as "B" into their corresponding option text.
            #
            # This remains safe because private_answers is contained inside
            # the session-isolated server-side quiz store and is never used
            # to construct QuizGenerateResponse.
            #
            # Example private record:
            #
            # {
            #     "correct_answer": "C",
            #     "options": [
            #         {"letter": "A", "text": "..."},
            #         {"letter": "B", "text": "..."},
            #         {"letter": "C", "text": "..."},
            #         {"letter": "D", "text": "..."},
            #     ],
            #     "explanation": "...",
            #     "topic": "...",
            #     "question": "...",
            # }

            private_answers[
                str(question_id)
            ] = {
                "correct_answer": correct_answer,
                "options": [
                    {
                        "letter": option.letter,
                        "text": option.text,
                    }
                    for option in options
                ],
                "explanation": explanation,
                "topic": topic,
                "question": question_text,
            }

        validation_time = (
            time.perf_counter()
            - validation_start
        )

        # --------------------------------------------------------------------
        # FINAL VALIDATION
        # --------------------------------------------------------------------

        if len(public_questions) != num_questions:
            raise RuntimeError(
                "Validated question count does not match "
                "the requested question count."
            )

        if len(private_answers) != num_questions:
            raise RuntimeError(
                "Private answer count does not match "
                "the requested question count."
            )

        # --------------------------------------------------------------------
        # STORE ONLY AFTER EVERYTHING PASSES VALIDATION
        # --------------------------------------------------------------------

        storage_start = time.perf_counter()

        private_quiz = {
            "quiz_id": normalized_quiz_id,
            "session_id": normalized_session_id,
            "source_type": normalized_source_type,
            "source_id": normalized_source_id,
            "answers": private_answers,
            "created_at": time.time(),
        }

        with _quiz_store_lock:

            session_store = _quiz_store.setdefault(
                normalized_session_id,
                {},
            )

            session_store[
                normalized_quiz_id
            ] = private_quiz

            self._enforce_storage_limits_locked(
                preferred_session_id=normalized_session_id,
            )

        storage_time = (
            time.perf_counter()
            - storage_start
        )

        # --------------------------------------------------------------------
        # PUBLIC RESPONSE
        # --------------------------------------------------------------------

        response = QuizGenerateResponse(
            quiz_id=normalized_quiz_id,
            source_type=normalized_source_type,
            source_id=normalized_source_id,
            questions=public_questions,
        )

        return (
            response,
            validation_time,
            storage_time,
        )

    # ========================================================================
    # STORAGE LIMITS
    # ========================================================================

    @staticmethod
    def _enforce_storage_limits_locked(
        preferred_session_id: str,
    ) -> None:
        """
        Bound in-memory quiz storage.

        Caller must hold _quiz_store_lock.

        Oldest quizzes are removed first. A quiz from another session
        is never returned or exposed; eviction only removes private
        server-side state that is no longer needed.
        """

        def total_count() -> int:
            return sum(
                len(session_quizzes)
                for session_quizzes in _quiz_store.values()
            )

        # --------------------------------------------------------------------
        # PER-SESSION LIMIT
        # --------------------------------------------------------------------

        session_store = _quiz_store.get(
            preferred_session_id
        )

        if session_store is not None:

            while (
                len(session_store)
                > MAX_QUIZZES_PER_SESSION
            ):

                oldest_quiz_id = min(
                    session_store,
                    key=lambda quiz_id: (
                        float(
                            session_store[
                                quiz_id
                            ].get(
                                "created_at",
                                0,
                            )
                        ),
                        quiz_id,
                    ),
                )

                del session_store[
                    oldest_quiz_id
                ]

        # --------------------------------------------------------------------
        # GLOBAL LIMIT
        # --------------------------------------------------------------------

        while total_count() > MAX_TOTAL_QUIZZES:

            oldest_session_id: str | None = None
            oldest_quiz_id: str | None = None
            oldest_timestamp = float(
                "inf"
            )

            for session_id, session_quizzes in (
                _quiz_store.items()
            ):

                for quiz_id, quiz in (
                    session_quizzes.items()
                ):

                    timestamp = float(
                        quiz.get(
                            "created_at",
                            0,
                        )
                    )

                    if (
                        timestamp < oldest_timestamp
                        or (
                            timestamp == oldest_timestamp
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
                        oldest_timestamp = timestamp
                        oldest_session_id = session_id
                        oldest_quiz_id = quiz_id

            if (
                oldest_session_id is None
                or oldest_quiz_id is None
            ):
                break

            session_quizzes = _quiz_store.get(
                oldest_session_id
            )

            if session_quizzes is None:
                continue

            session_quizzes.pop(
                oldest_quiz_id,
                None,
            )

            if not session_quizzes:
                _quiz_store.pop(
                    oldest_session_id,
                    None,
                )

    # ========================================================================
    # OPTION PARSING
    # ========================================================================

    def _parse_options(
        self,
        raw_options: list[str],
    ) -> list[QuizOption]:
        """
        Parse Gemini option strings.

        Expected formats:

            A. option text
            B. option text
            C. option text
            D. option text

        Plain option text is also handled defensively.
        """

        if not isinstance(
            raw_options,
            list,
        ):
            raise RuntimeError(
                "AI service returned invalid quiz options."
            )

        if len(raw_options) != 4:
            raise RuntimeError(
                "AI service returned a question without exactly "
                "four options."
            )

        parsed: list[
            QuizOption
        ] = []

        for index, raw_option in enumerate(
            raw_options
        ):

            letter = OPTION_LETTERS[
                index
            ]

            text = str(
                raw_option or ""
            ).strip()

            if (
                len(text) >= 2
                and text[0].upper() == letter
                and text[1] == "."
            ):
                text = text[2:].strip()

            if not text:
                raise RuntimeError(
                    f"AI service returned an empty option {letter}."
                )

            if len(text) > MAX_OPTION_LENGTH:
                raise RuntimeError(
                    f"AI service returned an option that is too long "
                    f"for option {letter}."
                )

            parsed.append(
                QuizOption(
                    letter=letter,
                    text=text,
                )
            )

        return parsed

    # ========================================================================
    # QUIZ ID VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_quiz_id(
        quiz_id: str,
    ) -> str:
        """
        Validate a quiz UUID.
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

    @staticmethod
    def _normalize_quiz_id(
        quiz_id: str | None,
    ) -> str:
        """
        Normalize and validate a quiz ID for lookup.

        Invalid IDs return an empty string so internal lookup behaves
        like a missing quiz rather than exposing validation details.
        """

        if not isinstance(
            quiz_id,
            str,
        ):
            return ""

        normalized = quiz_id.strip()

        if not normalized:
            return ""

        try:
            uuid.UUID(normalized)
        except ValueError:
            return ""

        return normalized

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_type(
        value: str | None,
    ) -> str:
        """
        Normalize enum-like values.
        """

        return str(
            value or ""
        ).strip().lower()

    @staticmethod
    def _normalize_source_id(
        value: str | None,
    ) -> str:
        """
        Normalize source identifiers without altering their case.
        """

        return str(
            value or ""
        ).strip()

    @staticmethod
    def _normalize_topic(
        value: str | None,
    ) -> str:
        """
        Normalize human-readable topic text without lowercasing it.
        """

        return str(
            value or ""
        ).strip()


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

quiz_service = QuizService()