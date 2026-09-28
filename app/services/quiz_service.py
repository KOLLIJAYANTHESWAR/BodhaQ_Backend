"""
QuizService — manages quiz generation and in-memory quiz storage.

Key design decisions:
- Correct answers are stored server-side only.
- Quiz-generation responses never include correct answers.
- Scoring is performed deterministically by the evaluation service.
- Gemini is used only for quiz generation and content generation.
- Document-based quizzes use RAG context.
- Targeted practice can be generated from either:
    1. A weak topic + original document using RAG.
    2. A weak topic alone using Gemini-generated topic context.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

from app.models.responses import (
    QuizGenerateResponse,
    QuizOption,
    QuizQuestionPublic,
)
from app.services.gemini_service import gemini_service
from app.services.rag_service import rag_service


# ── In-memory quiz storage ────────────────────────────────────────────────────

# quiz_id -> complete quiz data, including correct answers.
#
# For the MVP this is sufficient.
# Later this can be moved to SQLite for persistence across restarts.
_quiz_store: dict[str, dict[str, Any]] = {}


# ── Constants ─────────────────────────────────────────────────────────────────

OPTION_LETTERS = (
    "A",
    "B",
    "C",
    "D",
)

VALID_SOURCE_TYPES = {
    "topic",
    "document",
}

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}


class QuizService:
    """Service responsible for quiz generation and private quiz storage."""

    # ── Standard quiz generation ──────────────────────────────────────────────

    def generate_quiz(
        self,
        source_type: str,
        source_id: str,
        num_questions: int,
        difficulty: str,
    ) -> QuizGenerateResponse:
        """
        Generate a quiz from a topic or document.

        source_type='topic':
            source_id is the topic text.

        source_type='document':
            source_id is the document_id and RAG is used.

        Correct answers are stored privately and never returned
        in the public response.
        """

        source_type = source_type.strip().lower()
        source_id = source_id.strip()
        difficulty = difficulty.strip().lower()

        self._validate_request(
            source_type=source_type,
            source_id=source_id,
            num_questions=num_questions,
            difficulty=difficulty,
        )

        logger.info(f"[Quiz] request received for {source_type}")
        start_time = time.time()

        # ── 1. Build quiz context ────────────────────────────────────────────

        t0 = time.time()
        if source_type == "topic":
            context = self._get_topic_context(source_id)
            topic_hint = source_id
        else:
            context = rag_service.get_context_for_quiz(source_id)
            topic_hint = ""
            
        context_time = time.time() - t0

        if not context.strip():
            raise RuntimeError(
                "No study material was available to generate the quiz."
            )

        # ── 2. Generate quiz using Gemini ────────────────────────────────────

        t1 = time.time()
        raw_quiz = gemini_service.generate_quiz(
            context=context,
            topic_hint=topic_hint,
            num_questions=num_questions,
            difficulty=difficulty,
        )
        gemini_time = time.time() - t1

        if not raw_quiz.questions:
            raise RuntimeError(
                "AI service returned no quiz questions."
            )

        # ── 3. Store and return quiz ──────────────────────────────────────────

        t2 = time.time()
        res, val_time, store_time = self._build_quiz_response_with_timing(
            raw_quiz=raw_quiz,
            quiz_id=str(uuid.uuid4()),
            source_type=source_type,
            source_id=source_id,
            num_questions=num_questions,
            topic_hint=topic_hint,
        )
        
        total_time = time.time() - start_time
        logger.info(f"[Quiz] Context prep: {context_time:.2f}s | Gemini generation: {gemini_time:.2f}s | Validation: {val_time:.2f}s | Storage: {store_time:.2f}s | Total: {total_time:.2f}s")
        return res

    # ── Targeted practice ─────────────────────────────────────────────────────

    def generate_targeted_practice(
        self,
        topic: str,
        document_id: str | None,
        num_questions: int,
        difficulty: str,
    ) -> QuizGenerateResponse:
        """
        Generate targeted practice for a weak topic.

        Two modes are supported.

        DOCUMENT MODE
        -------------
        topic + document_id

        The weak topic is used as a semantic retrieval query against
        the original document.

        Flow:

            weak topic
                ↓
            document RAG retrieval
                ↓
            relevant source chunks
                ↓
            Gemini
                ↓
            targeted practice


        TOPIC MODE
        ----------
        topic without document_id

        The weak topic itself is used to generate a short study context,
        which is then passed to Gemini for targeted practice.

        Flow:

            weak topic
                ↓
            Gemini explanation
                ↓
            targeted practice


        Correct answers remain server-side in both modes.
        """

        topic = topic.strip()

        if document_id is not None:
            document_id = document_id.strip()

            if not document_id:
                document_id = None

        difficulty = difficulty.strip().lower()

        # ── Validate topic ───────────────────────────────────────────────────

        if not topic:
            raise ValueError(
                "topic cannot be empty."
            )

        # ── Validate question count ─────────────────────────────────────────

        if not isinstance(num_questions, int):
            raise ValueError(
                "question_count must be an integer."
            )

        if num_questions < 1 or num_questions > 20:
            raise ValueError(
                "question_count must be between 1 and 20."
            )

        # ── Validate difficulty ──────────────────────────────────────────────

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                "Expected 'easy', 'medium', or 'hard'."
            )

        # ── DOCUMENT-SOURCED PRACTICE ────────────────────────────────────────

        if document_id:

            # Use the weak topic as the semantic retrieval query.
            #
            # Example:
            #
            # weak topic = "Key Uniqueness"
            #
            # Instead of asking the vector database for generic
            # information about "Key Uniqueness", retrieval is performed
            # against the actual uploaded document.

            context, _sources = rag_service.get_context_for_question(
                document_id=document_id,
                question=topic,
            )

            if not context.strip():
                raise RuntimeError(
                    "No relevant study material was found for the "
                    "requested weak topic."
                )

            raw_quiz = gemini_service.generate_quiz(
                context=context,
                topic_hint=topic,
                num_questions=num_questions,
                difficulty=difficulty,
            )

            if not raw_quiz.questions:
                raise RuntimeError(
                    "AI service returned no practice questions."
                )

            return self._build_quiz_response(
                raw_quiz=raw_quiz,
                quiz_id=str(uuid.uuid4()),
                source_type="document",
                source_id=document_id,
                num_questions=num_questions,
                topic_hint=topic,
            )

        # ── TOPIC-ONLY PRACTICE ──────────────────────────────────────────────

        # No document was supplied.
        #
        # Generate a concise topic explanation first so Gemini has
        # explicit source material rather than receiving only a short
        # topic label.

        context = self._get_topic_context(
            topic
        )

        if not context.strip():
            raise RuntimeError(
                "No study material was available for the requested "
                "practice topic."
            )

        raw_quiz = gemini_service.generate_quiz(
            context=context,
            topic_hint=topic,
            num_questions=num_questions,
            difficulty=difficulty,
        )

        if not raw_quiz.questions:
            raise RuntimeError(
                "AI service returned no practice questions."
            )

        return self._build_quiz_response(
            raw_quiz=raw_quiz,
            quiz_id=str(uuid.uuid4()),
            source_type="topic",
            source_id=topic,
            num_questions=num_questions,
            topic_hint=topic,
        )

    # ── Storage ───────────────────────────────────────────────────────────────

    def get_stored_quiz(
        self,
        quiz_id: str,
    ) -> dict[str, Any] | None:
        """
        Return the complete server-side quiz data.

        This method is used internally by evaluation logic.

        It must never be exposed directly through an API response.
        """

        return _quiz_store.get(
            quiz_id
        )

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_request(
        self,
        source_type: str,
        source_id: str,
        num_questions: int,
        difficulty: str,
    ) -> None:
        """Validate quiz-generation parameters."""

        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type '{source_type}'. "
                "Expected 'topic' or 'document'."
            )

        if not source_id:
            raise ValueError(
                "source_id cannot be empty."
            )

        if not isinstance(num_questions, int):
            raise ValueError(
                "number_of_questions must be an integer."
            )

        if num_questions < 1 or num_questions > 20:
            raise ValueError(
                "number_of_questions must be between 1 and 20."
            )

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                "Expected 'easy', 'medium', or 'hard'."
            )

    # ── Topic context ─────────────────────────────────────────────────────────

    def _get_topic_context(
        self,
        topic: str,
    ) -> str:
        """
        Generate a short topic explanation to provide Gemini
        with source material for quiz generation.
        """

        explanation = gemini_service.generate_explanation(
            topic
        )

        return (
            f"Topic: {topic}\n\n"
            f"{explanation}"
        )

    # ── Quiz response construction ────────────────────────────────────────────

    def _build_quiz_response(
        self,
        raw_quiz: Any,
        quiz_id: str,
        source_type: str,
        source_id: str,
        num_questions: int,
        topic_hint: str,
    ) -> QuizGenerateResponse:
        res, _, _ = self._build_quiz_response_with_timing(raw_quiz, quiz_id, source_type, source_id, num_questions, topic_hint)
        return res

    def _build_quiz_response_with_timing(
        self,
        raw_quiz: Any,
        quiz_id: str,
        source_type: str,
        source_id: str,
        num_questions: int,
        topic_hint: str,
    ) -> tuple[QuizGenerateResponse, float, float]:
        """
        Validate Gemini's quiz output, store private answers,
        and construct the public response, returning timings.
        """
        
        t0 = time.time()
        public_questions: list[QuizQuestionPublic] = []

        private_answers: dict[str, dict[str, Any]] = {}

        seen_question_ids: set[int] = set()

        for question in raw_quiz.questions:

            question_id = int(
                question.id
            )

            if question_id <= 0:
                raise RuntimeError(
                    "AI service returned an invalid question ID."
                )

            if question_id in seen_question_ids:
                raise RuntimeError(
                    "AI service returned duplicate question IDs."
                )

            seen_question_ids.add(
                question_id
            )

            options = self._parse_options(
                question.options
            )

            correct_answer = (
                question.correct_answer
                .strip()
                .upper()
            )

            if correct_answer not in OPTION_LETTERS:
                raise RuntimeError(
                    "AI service returned an invalid correct answer."
                )

            option_letters = {
                option.letter
                for option in options
            }

            if correct_answer not in option_letters:
                raise RuntimeError(
                    "AI service returned a correct answer "
                    "that does not match the available options."
                )

            topic = (
                question.topic.strip()
                if question.topic
                else topic_hint or "General"
            )

            question_text = (
                question.question.strip()
            )

            explanation = (
                question.explanation.strip()
            )

            if not question_text:
                raise RuntimeError(
                    "AI service returned an empty question."
                )

            if not explanation:
                raise RuntimeError(
                    "AI service returned an empty explanation."
                )

            public_questions.append(
                QuizQuestionPublic(
                    id=question_id,
                    question=question_text,
                    options=options,
                    topic=topic,
                )
            )

            # IMPORTANT:
            # correct_answer is stored only here.
            #
            # It is never included in QuizGenerateResponse.

            private_answers[str(question_id)] = {
                "correct_answer": correct_answer,
                "explanation": explanation,
                "topic": topic,
                "question": question_text,
            }

        if len(public_questions) != num_questions:
            raise RuntimeError(
                f"AI service returned {len(public_questions)} "
                f"questions instead of the requested "
                f"{num_questions}."
            )
            
        val_time = time.time() - t0

        t1 = time.time()
        _quiz_store[quiz_id] = {
            "quiz_id": quiz_id,
            "source_type": source_type,
            "source_id": source_id,
            "answers": private_answers,
        }
        store_time = time.time() - t1

        res = QuizGenerateResponse(
            quiz_id=quiz_id,
            source_type=source_type,
            source_id=source_id,
            questions=public_questions,
        )
        return res, val_time, store_time

    # ── Option parsing ────────────────────────────────────────────────────────

    def _parse_options(
        self,
        raw_options: list[str],
    ) -> list[QuizOption]:
        """
        Parse Gemini option strings.

        Expected formats include:

            A. option text
            B. option text
            C. option text
            D. option text

        Plain option text is also handled defensively.
        """

        if len(raw_options) != 4:
            raise RuntimeError(
                "AI service returned a question without exactly "
                "four options."
            )

        parsed: list[QuizOption] = []

        for index, raw_option in enumerate(
            raw_options
        ):
            letter = OPTION_LETTERS[index]

            text = str(
                raw_option
            ).strip()

            if (
                len(text) >= 3
                and text[0].upper() == letter
                and text[1] == "."
            ):
                text = text[2:].strip()

            if not text:
                raise RuntimeError(
                    f"AI service returned an empty option {letter}."
                )

            parsed.append(
                QuizOption(
                    letter=letter,
                    text=text,
                )
            )

        return parsed


# ── Module-level singleton ────────────────────────────────────────────────────

quiz_service = QuizService()