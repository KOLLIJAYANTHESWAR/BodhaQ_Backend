"""
Request models for BodhaQ API.

All incoming payloads are validated here via Pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# SHARED VALIDATION
# ============================================================================


def _strip_required_text(value: str) -> str:
    """
    Strip surrounding whitespace and reject empty values.
    """
    value = value.strip()

    if not value:
        raise ValueError(
            "Value cannot be empty or whitespace only."
        )

    return value


def _strip_optional_text(value: str) -> str:
    """
    Strip surrounding whitespace from optional text.
    """
    return value.strip()


# ============================================================================
# LEARNING
# ============================================================================


class TopicRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Topic to learn, e.g. 'Java HashMap'",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(
        cls,
        value: str,
    ) -> str:
        value = _strip_required_text(value)

        if len(value) < 2:
            raise ValueError(
                "Topic must contain at least 2 characters."
            )

        return value


# ============================================================================
# DOCUMENTS
# ============================================================================


class DocumentDeleteRequest(BaseModel):
    document_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)


# ============================================================================
# QUIZ
# ============================================================================


class QuizGenerateRequest(BaseModel):
    source_type: Literal[
        "topic",
        "document",
    ] = Field(
        ...,
        description="'topic' or 'document'",
    )

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Topic string when source_type='topic'; "
            "document_id when source_type='document'"
        ),
    )

    number_of_questions: int = Field(
        5,
        ge=1,
        le=30,
    )

    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ] = "medium"

    @field_validator("source_id")
    @classmethod
    def validate_source_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)


class QuizSubmitRequest(BaseModel):
    quiz_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    answers: dict[
        str,
        Literal["A", "B", "C", "D"],
    ] = Field(
        ...,
        max_length=30,
        description=(
            "Map of question id (str) → selected option letter, "
            "e.g. {'1':'A','2':'C'}"
        ),
    )

    @field_validator("quiz_id")
    @classmethod
    def validate_quiz_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("answers")
    @classmethod
    def validate_answers(
        cls,
        value: dict[
            str,
            Literal["A", "B", "C", "D"],
        ],
    ) -> dict[
        str,
        Literal["A", "B", "C", "D"],
    ]:
        normalized: dict[
            str,
            Literal["A", "B", "C", "D"],
        ] = {}

        for question_id, answer in value.items():
            normalized_question_id = (
                question_id.strip()
            )

            if not normalized_question_id:
                raise ValueError(
                    "Question IDs cannot be empty."
                )

            normalized[normalized_question_id] = answer

        return normalized


# ============================================================================
# TARGETED PRACTICE
# ============================================================================


class PracticeRequest(BaseModel):
    """
    Request model for targeted practice.

    document_id is optional.

    If document_id is provided:
        Practice is grounded using RAG against the original document.

    If document_id is omitted:
        Practice is generated from the weak topic itself.
    """

    topic: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Weak topic to practice.",
    )

    document_id: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description=(
            "Optional document ID. If provided, targeted practice "
            "is grounded in the original document using RAG."
        ),
    )

    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ] = "medium"

    question_count: int = Field(
        5,
        ge=1,
        le=20,
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(
        cls,
        value: str,
    ) -> str:
        value = _strip_required_text(value)

        if len(value) < 2:
            raise ValueError(
                "Topic must contain at least 2 characters."
            )

        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return _strip_required_text(value)


# ============================================================================
# DOUBTS
# ============================================================================


class ConversationTurn(BaseModel):
    """A single turn in a conversation history."""

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)


class DoubtRequest(BaseModel):
    """
    Request model for the continuous Doubts conversation.

    Questions can be very short, including messages such as:
        - "hi"
        - "why?"
        - "help"
        - "Java?"
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Question or message from the student.",
    )

    document_id: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description=(
            "If provided, answers are grounded in that document via RAG."
        ),
    )

    history: list[
        ConversationTurn
    ] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Optional conversation history for multi-turn context. "
            "Each turn has 'role' ('user' or 'assistant') and 'content'. "
            "Maximum 8 turns are accepted."
        ),
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return _strip_required_text(value)


# ============================================================================
# RESUME PREP
# ============================================================================


class ResumeQuizGenerateRequest(BaseModel):
    item_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ] = "medium"

    number_of_questions: int = Field(
        5,
        ge=1,
        le=20,
    )

    @field_validator("item_id")
    @classmethod
    def validate_item_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)


# ============================================================================
# CODING
# ============================================================================


class CodeExecutionRequest(BaseModel):
    language: Literal[
        "java",
        "python",
    ] = Field(
        ...,
        description="Programming language.",
    )

    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The source code to execute.",
    )

    stdin: str = Field(
        "",
        max_length=20000,
        description="Standard input for the program.",
    )

    @field_validator("code")
    @classmethod
    def validate_code(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("stdin")
    @classmethod
    def normalize_stdin(
        cls,
        value: str,
    ) -> str:
        return _strip_optional_text(value)


class CodingProblemGenerateRequest(BaseModel):
    """
    Request for AI Learn Code problem generation.

    At least one of:
        - title
        - statement

    must contain meaningful content.

    Examples:

        {"title": "Two Sum"}

        {"statement": "Given an integer array..."}

    An empty request is rejected by the coding route.
    """

    title: str = Field(
        default="",
        max_length=300,
        description="Optional coding problem title.",
    )

    statement: str = Field(
        default="",
        max_length=10000,
        description="Optional full coding problem statement.",
    )

    constraints: str = Field(
        default="",
        max_length=5000,
        description="Optional problem constraints.",
    )

    sample: str = Field(
        default="",
        max_length=5000,
        description="Optional sample test case.",
    )

    @field_validator(
        "title",
        "statement",
        "constraints",
        "sample",
    )
    @classmethod
    def normalize_text_fields(
        cls,
        value: str,
    ) -> str:
        return _strip_optional_text(value)


class CodeAnalyzeRequest(BaseModel):
    problem_statement: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    sample_test_case: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    constraints: str = Field(
        default="",
        max_length=5000,
    )

    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    language: Literal[
        "java",
        "python",
    ]

    @field_validator(
        "problem_statement",
        "sample_test_case",
        "code",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(
        cls,
        value: str,
    ) -> str:
        return _strip_optional_text(value)


class CodeImproveRequest(BaseModel):
    problem_statement: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    sample_test_case: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    constraints: str = Field(
        default="",
        max_length=5000,
    )

    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    language: Literal[
        "java",
        "python",
    ]

    @field_validator(
        "problem_statement",
        "sample_test_case",
        "code",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(
        cls,
        value: str,
    ) -> str:
        return _strip_optional_text(value)


class TestCaseGenerateRequest(BaseModel):
    problem_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    problem_statement: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    sample_test_case: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    constraints: str = Field(
        default="",
        max_length=5000,
    )

    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    language: Literal[
        "java",
        "python",
    ]

    @field_validator("problem_id")
    @classmethod
    def validate_problem_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator(
        "problem_statement",
        "sample_test_case",
        "code",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(
        cls,
        value: str,
    ) -> str:
        return _strip_optional_text(value)


class CodeSubmitRequest(BaseModel):
    problem_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    language: Literal[
        "java",
        "python",
    ]

    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )

    test_suite: Literal[
        "samples",
        "public",
        "all",
    ] = "all"

    @field_validator("problem_id")
    @classmethod
    def validate_problem_id(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)

    @field_validator("code")
    @classmethod
    def validate_code(
        cls,
        value: str,
    ) -> str:
        return _strip_required_text(value)