"""
GeminiService — the single point of contact with the Gemini API.

All other services call methods here. No other module imports the
google.genai client directly. This keeps AI logic isolated and testable.
"""

from __future__ import annotations

import logging
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import GEMINI_API_KEY


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

# This is the model that was previously working in BodhaQ.
GEMINI_MODEL = "gemini-3.5-flash-lite"

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GeminiAuthenticationError(RuntimeError):
    """Raised when Gemini API authentication fails."""
    pass


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini API quota or rate limit is exceeded."""
    pass


# ---------------------------------------------------------------------------
# Internal Gemini response schemas
# ---------------------------------------------------------------------------


class _CodeExample(BaseModel):
    code: str
    explanation: str


class _LearningContentSchema(BaseModel):
    topic: str
    definition: str
    key_concepts: list[str]
    example: _CodeExample
    important_points: list[str]


class _QuizQuestionSchema(BaseModel):
    id: int
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    topic: str


class _QuizSchema(BaseModel):
    questions: list[_QuizQuestionSchema]


class _ExtractedSkill(BaseModel):
    name: str
    description: str


class _ExtractedProject(BaseModel):
    name: str
    description: str
    technologies: list[str]


class _ExtractedCertification(BaseModel):
    name: str


class _ResumeExtractionSchema(BaseModel):
    skills: list[_ExtractedSkill]
    projects: list[_ExtractedProject]
    certifications: list[_ExtractedCertification]


StructuredModel = TypeVar(
    "StructuredModel",
    bound=BaseModel,
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GeminiService:
    """Centralized Gemini API service for BodhaQ."""

    def __init__(self) -> None:
        """
        Initialize the Gemini Developer API client.

        The API key is loaded only from backend configuration.
        """

        self._client: genai.Client | None = None

        if not GEMINI_API_KEY:
            logger.warning(
                "[Gemini] GEMINI_API_KEY is not configured."
            )
            return

        try:
            self._client = genai.Client(
                api_key=GEMINI_API_KEY,
            )

            logger.info(
                "[Gemini] Client initialized successfully. Model=%s",
                GEMINI_MODEL,
            )

        except Exception as exc:
            logger.exception(
                "[Gemini] Failed to initialize Gemini client: %s",
                exc,
            )
            self._client = None

    # -----------------------------------------------------------------------
    # Client
    # -----------------------------------------------------------------------

    def _get_client(self) -> genai.Client:
        """
        Return the configured Gemini client.

        Gemini credentials remain entirely on the backend.
        """

        if self._client is not None:
            return self._client

        if not GEMINI_API_KEY:
            raise GeminiAuthenticationError(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to backend/.env."
            )

        try:
            self._client = genai.Client(
                api_key=GEMINI_API_KEY,
            )

            return self._client

        except Exception as exc:
            raise GeminiAuthenticationError(
                "Unable to initialize the Gemini client. "
                "Check GEMINI_API_KEY in backend/.env."
            ) from exc

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------

    def generate_learning_content(
        self,
        topic: str,
    ) -> _LearningContentSchema:
        """
        Generate structured learning content for a topic.
        """

        prompt = f"""
You are BodhaQ, an expert AI learning assistant.

Create a clear, beginner-friendly lesson for this topic:

{topic}

Requirements:

- Definition: 2-3 clear sentences.
- Key concepts: 3 to 5 concise items.
- Example: one practical, beginner-friendly code example.
- Important points: 3 to 5 concise items.
- Keep the content technically accurate and concise.
- Use terminology appropriate for a student learning the topic.

CODE EXAMPLE REQUIREMENTS:

- The example must be syntactically valid code.
- The example must be directly related to the requested topic.
- Use a real programming language appropriate for the topic.
- Include required imports when the language requires them.
- Use correct variable declarations, types, literals, method calls,
  quotation marks, brackets, and punctuation.
- Strings must be represented as actual string literals.
- Never place natural-language words directly into code where a
  string literal, number, boolean, or valid identifier is required.
- Do not use untranslated words or non-code prose as code values.
- Do not use placeholders such as <value>, YOUR_VALUE, or ...
  unless they are explicitly part of valid syntax.
- Prefer a simple example that a beginner can copy and run.
- Do not include Markdown fences inside the code field.
- Return only the raw source code in the code field.
- The explanation must describe what the code actually does.

FORMATTING REQUIREMENTS — CRITICAL:

- The code field MUST contain real newline characters.
- NEVER return the entire program on a single line.
- Put each import on its own separate line.
- Put each class declaration on its own separate line.
- Put each method declaration on its own separate line.
- Put each statement on its own separate line.
- Put closing braces on their own appropriate separate lines.
- Indent code blocks consistently using spaces.
- Do NOT minify, compress, or concatenate statements onto one line.
- Do NOT omit line breaks between logical sections of the code.
- Return actual line breaks inside the code string.

IMPORTANT:

If the example uses text values such as country names, city names,
student names, messages, or other natural-language values, represent
those values using valid string literals for the selected language.

For example, in Java:

capitalCities.put("England", "London");

NOT:

capitalCities.put("England", London);

Do not translate natural-language values into bare identifiers.
"""

        result = self._generate_structured(
            prompt=prompt,
            schema=_LearningContentSchema,
        )

        self._validate_learning_content(result)

        return result

    # -----------------------------------------------------------------------
    # Quiz
    # -----------------------------------------------------------------------

    def generate_quiz(
        self,
        context: str,
        topic_hint: str,
        num_questions: int,
        difficulty: str,
    ) -> _QuizSchema:
        """
        Generate a multiple-choice quiz from provided study material.
        """

        prompt = f"""
You are BodhaQ, an expert quiz designer.

Generate exactly {num_questions} {difficulty}-difficulty
multiple-choice questions based on the following study material.

Topic hint:

{topic_hint}

Study material:

{context}

Rules:

- Generate exactly the requested number of questions.
- Each question must have exactly 4 options.
- Options must be formatted as:
  "A. ..."
  "B. ..."
  "C. ..."
  "D. ..."
- correct_answer must be exactly one of:
  "A", "B", "C", or "D".
- Explanation must be 1-2 sentences explaining why the answer is correct.
- topic must contain the specific sub-topic being tested.
- IDs must be sequential integers starting from 1.
- Questions must be based on the provided study material.
- Avoid duplicate or nearly identical questions.
- Make incorrect options plausible and related to the topic.
- Do not include information that contradicts the provided study material.
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_QuizSchema,
        )

    # -----------------------------------------------------------------------
    # Doubts
    # -----------------------------------------------------------------------

    def answer_doubt(
        self,
        question: str,
        context: str = "",
        history: list[dict] | None = None,
    ) -> str:
        """
        Answer a user question.

        When context is supplied, the answer is grounded in the
        retrieved document context.

        When history is supplied, previous turns are included.
        """

        history = history or []

        history_block = ""

        if history:
            turns = []

            for turn in history:
                role_label = (
                    "Student"
                    if turn.get("role") == "user"
                    else "BodhaQ"
                )

                content = str(
                    turn.get("content", "")
                ).strip()

                if content:
                    turns.append(
                        f"{role_label}: {content}"
                    )

            if turns:
                history_block = (
                    "\n\nConversation so far:\n\n"
                    + "\n\n".join(turns)
                    + "\n\n---"
                )

        if context:
            prompt = f"""
You are BodhaQ, a helpful learning assistant.

Answer the following question using ONLY the provided document context.

If the context does not contain sufficient information to answer
the question, clearly say that the provided material does not contain
enough information.

Do not invent information that is not supported by the context.

Document context:

{context}

{history_block}

Student: {question}

Provide a clear and educational answer.

When useful, refer to the relevant information from the context.
"""

        else:
            prompt = f"""
You are BodhaQ, a helpful learning assistant.

Answer the following question clearly and accurately.

Use the conversation history when it helps understand the student's
current question.

{history_block}

Student: {question}
"""

        return self._generate_text(prompt)

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def generate_explanation(
        self,
        topic: str,
        context: str = "",
    ) -> str:
        """
        Generate a concise explanation for a concept.
        """

        if context:
            prompt = f"""
You are BodhaQ, an expert learning assistant.

Explain the following topic using the provided document excerpts.

Topic:

{topic}

Document excerpts:

{context}

Keep the explanation concise, technically accurate,
and easy for a student to understand.
"""

        else:
            prompt = f"""
You are BodhaQ, an expert learning assistant.

Explain the following topic in 3-5 clear and accurate sentences:

{topic}
"""

        return self._generate_text(prompt)

    # -----------------------------------------------------------------------
    # Practice
    # -----------------------------------------------------------------------

    def generate_practice(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        context: str = "",
    ) -> _QuizSchema:
        """
        Generate targeted practice questions for a weak topic.
        """

        material = (
            context
            if context
            else f"Study topic: {topic}"
        )

        return self.generate_quiz(
            context=material,
            topic_hint=topic,
            num_questions=num_questions,
            difficulty=difficulty,
        )

    # -----------------------------------------------------------------------
    # Resume extraction
    # -----------------------------------------------------------------------

    def extract_resume(
        self,
        text: str,
    ) -> _ResumeExtractionSchema:
        """
        Extract structured information from resume text.
        """

        prompt = f"""
You are BodhaQ, an expert technical recruiter and resume analyzer.

Extract the following information from the provided resume text:

1. Technical Skills
   Programming languages, frameworks, tools, databases, cloud,
   DevOps tools, libraries, and other technical skills.

2. Projects
   Extract the project title, description, and technologies used.

3. Certifications
   Extract certification names.

If any section is empty, return an empty list.

Do not invent information that is not present in the resume.

Resume Text:

{text}
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_ResumeExtractionSchema,
        )

    # -----------------------------------------------------------------------
    # Resume quizzes
    # -----------------------------------------------------------------------

    def generate_resume_quiz(
        self,
        item_type: str,
        item_data: dict,
        num_questions: int,
        difficulty: str,
    ) -> _QuizSchema:
        """
        Generate a quiz specifically for a resume item.
        """

        if item_type == "skill":

            prompt = f"""
You are BodhaQ, an expert technical interviewer.

Generate {num_questions} multiple-choice questions for the following
technical skill.

Skill:
{item_data["name"]}

Difficulty:
{difficulty}

Difficulty guidance:

Easy:
- Fundamentals
- Definitions
- Basic concepts
- Differences between concepts

Medium:
- Practical understanding
- Implementation decisions
- Common scenarios
- Debugging

Hard:
- Interview-level questions
- Internals
- Trade-offs
- Optimization
- Architecture

Rules:

- Generate exactly {num_questions} questions.
- Exactly 4 options per question.
- Options must use A., B., C., D.
- correct_answer must be A, B, C, or D.
- Provide a clear 1-2 sentence explanation.
- Set topic to the relevant sub-topic.
- IDs must be sequential starting from 1.
"""

        elif item_type == "project":

            technologies = ", ".join(
                item_data.get("technologies", [])
            )

            prompt = f"""
You are BodhaQ, an expert technical interviewer.

Generate {num_questions} multiple-choice interview questions based
on the candidate's project.

Project Name:
{item_data["name"]}

Description:
{item_data["description"]}

Technologies Used:
{technologies}

Difficulty:
{difficulty}

Do NOT ask generic questions such as:
"What is Python?"
"What is Java?"

Ask questions grounded in the actual project context.

Focus on areas such as:

- Architecture
- Technology choices
- Data flow
- APIs
- Database design
- Failure handling
- Security
- Performance
- Scalability
- Deployment
- Trade-offs
- Debugging
- Reliability

If the project does not mention a technology,
do not invent it.

Rules:

- Generate exactly {num_questions} questions.
- Exactly 4 options per question.
- Options must use A., B., C., D.
- correct_answer must be A, B, C, or D.
- Provide a clear explanation.
- IDs must be sequential starting from 1.
"""

        elif item_type == "certification":

            prompt = f"""
You are BodhaQ, an expert technical interviewer
and certification-domain specialist.

Generate {num_questions} multiple-choice questions based on:

Certification:
{item_data["name"]}

Difficulty:
{difficulty}

Test knowledge relevant to the specific certification domain.

Include:

- Core concepts
- Terminology
- Practical scenarios
- Troubleshooting
- Architecture where applicable
- Interview-level understanding

Rules:

- Generate exactly {num_questions} questions.
- Exactly 4 options per question.
- Options must use A., B., C., D.
- correct_answer must be A, B, C, or D.
- Provide a clear explanation.
- IDs must be sequential starting from 1.
"""

        else:
            raise ValueError(
                f"Unknown item type: {item_type}"
            )

        return self._generate_structured(
            prompt=prompt,
            schema=_QuizSchema,
        )

    # -----------------------------------------------------------------------
    # Learning content validation
    # -----------------------------------------------------------------------

    def _validate_learning_content(
        self,
        content: _LearningContentSchema,
    ) -> None:
        """
        Perform lightweight validation on AI-generated learning content.
        """

        if not content.topic.strip():
            raise RuntimeError(
                "AI service returned an invalid learning topic."
            )

        if not content.definition.strip():
            raise RuntimeError(
                "AI service returned an empty definition."
            )

        if not content.key_concepts:
            raise RuntimeError(
                "AI service returned no key concepts."
            )

        if not content.important_points:
            raise RuntimeError(
                "AI service returned no important points."
            )

        code = content.example.code.strip()

        if not code:
            raise RuntimeError(
                "AI service returned an empty code example."
            )

        if code.startswith("```") or code.endswith("```"):
            raise RuntimeError(
                "AI service returned a code example containing Markdown fences."
            )

        suspicious_placeholders = (
            "<value>",
            "<your_value>",
            "YOUR_VALUE",
            "...",
        )

        if any(
            placeholder in code
            for placeholder in suspicious_placeholders
        ):
            raise RuntimeError(
                "AI service returned a code example containing unsupported placeholders."
            )

        if not content.example.explanation.strip():
            raise RuntimeError(
                "AI service returned an empty code explanation."
            )

        if "\n" not in code and len(code) > 60:
            raise RuntimeError(
                "AI service returned code as a single line. "
                "Expected properly formatted, multi-line source code."
            )

    # -----------------------------------------------------------------------
    # Error classification
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_authentication_error(exc: Exception) -> bool:
        """
        Detect Gemini authentication failures.
        """

        error_text = str(exc).upper()

        markers = (
            "401",
            "UNAUTHENTICATED",
            "UNAUTHORIZED",
            "INVALID API KEY",
            "INVALID_API_KEY",
            "API_KEY_INVALID",
            "API KEY INVALID",
            "ACCESS_TOKEN",
        )

        return any(
            marker in error_text
            for marker in markers
        )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """
        Detect Gemini quota/rate-limit failures.
        """

        error_text = str(exc).upper()

        markers = (
            "429",
            "RESOURCE_EXHAUSTED",
            "QUOTA",
            "RATE LIMIT",
            "RATE_LIMIT",
        )

        return any(
            marker in error_text
            for marker in markers
        )

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """
        Determine whether a Gemini exception is likely temporary.
        """

        error_text = str(exc).upper()

        transient_markers = (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
            "SERVICE_UNAVAILABLE",
            "TEMPORARILY_UNAVAILABLE",
            "500",
            "502",
            "504",
        )

        return any(
            marker in error_text
            for marker in transient_markers
        )

    # -----------------------------------------------------------------------
    # Structured generation
    # -----------------------------------------------------------------------

    def _generate_structured(
        self,
        prompt: str,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        """
        Generate structured JSON content and parse it into a Pydantic model.

        Temporary Gemini service failures are retried with exponential
        backoff.
        """

        client = self._get_client()

        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):

            try:
                logger.info(
                    "[Gemini] Structured request | model=%s | attempt=%d/%d",
                    GEMINI_MODEL,
                    attempt + 1,
                    MAX_RETRIES,
                )

                chat = client.chats.create(
                    model=GEMINI_MODEL,
                )

                response = chat.send_message(
                    message=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                if response.parsed is None:
                    raise RuntimeError(
                        "AI service returned an empty structured response."
                    )

                logger.info(
                    "[Gemini] Structured request succeeded."
                )

                return response.parsed

            except Exception as exc:

                last_exception = exc

                logger.warning(
                    "[Gemini] Structured request failed "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )

                if self._is_authentication_error(exc):
                    raise GeminiAuthenticationError(
                        "Gemini API authentication failed. "
                        "Check GEMINI_API_KEY in backend/.env."
                    ) from exc

                if self._is_quota_error(exc):
                    if attempt == MAX_RETRIES - 1:
                        raise GeminiQuotaError(
                            "Gemini API rate limit or quota exceeded. "
                            "Please wait and try again."
                        ) from exc

                if not self._is_transient_error(exc):
                    raise RuntimeError(
                        "AI service request failed during structured generation: "
                        f"{exc}"
                    ) from exc

                if attempt == MAX_RETRIES - 1:
                    break

                delay = INITIAL_RETRY_DELAY * (2 ** attempt)

                logger.info(
                    "[Gemini] Retrying structured request in %.1fs...",
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            "AI service request failed during structured generation "
            f"after {MAX_RETRIES} attempts: {last_exception}"
        ) from last_exception

    # -----------------------------------------------------------------------
    # Text generation
    # -----------------------------------------------------------------------

    def _generate_text(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a plain-text Gemini response.

        Chat.send_message is used here instead of direct
        Models.generate_content.
        """

        client = self._get_client()

        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):

            try:
                logger.info(
                    "[Gemini] Text request | model=%s | attempt=%d/%d",
                    GEMINI_MODEL,
                    attempt + 1,
                    MAX_RETRIES,
                )

                chat = client.chats.create(
                    model=GEMINI_MODEL,
                )

                response = chat.send_message(
                    message=prompt,
                )

                if not response.text:
                    raise RuntimeError(
                        "AI service returned an empty response."
                    )

                logger.info(
                    "[Gemini] Text request succeeded."
                )

                return response.text.strip()

            except Exception as exc:

                last_exception = exc

                logger.warning(
                    "[Gemini] Text request failed "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )

                if self._is_authentication_error(exc):
                    raise GeminiAuthenticationError(
                        "Gemini API authentication failed. "
                        "Check GEMINI_API_KEY in backend/.env."
                    ) from exc

                if self._is_quota_error(exc):
                    if attempt == MAX_RETRIES - 1:
                        raise GeminiQuotaError(
                            "Gemini API rate limit or quota exceeded. "
                            "Please wait and try again."
                        ) from exc

                if not self._is_transient_error(exc):
                    raise RuntimeError(
                        "AI service request failed during text generation: "
                        f"{exc}"
                    ) from exc

                if attempt == MAX_RETRIES - 1:
                    break

                delay = INITIAL_RETRY_DELAY * (2 ** attempt)

                logger.info(
                    "[Gemini] Retrying text request in %.1fs...",
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            "AI service request failed during text generation "
            f"after {MAX_RETRIES} attempts: {last_exception}"
        ) from last_exception


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

gemini_service = GeminiService()