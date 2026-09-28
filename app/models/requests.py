"""
Request models for BodhaQ API.

All incoming payloads are validated here via Pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ── Learning ──────────────────────────────────────────────────────────────────


class TopicRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Topic to learn, e.g. 'Java HashMap'",
    )


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentDeleteRequest(BaseModel):
    document_id: str


# ── Quiz ──────────────────────────────────────────────────────────────────────


class QuizGenerateRequest(BaseModel):
    source_type: Literal["topic", "document"] = Field(
        ...,
        description="'topic' or 'document'",
    )

    source_id: str = Field(
        ...,
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

    difficulty: Literal["easy", "medium", "hard"] = "medium"


class QuizSubmitRequest(BaseModel):
    quiz_id: str

    answers: dict[str, str] = Field(
        ...,
        description=(
            "Map of question id (str) → selected option letter, "
            "e.g. {'1':'A','2':'C'}"
        ),
    )


# ── Targeted Practice ─────────────────────────────────────────────────────────


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
        description=(
            "Optional document ID. If provided, targeted practice "
            "is grounded in the original document using RAG."
        ),
    )

    difficulty: Literal["easy", "medium", "hard"] = "medium"

    question_count: int = Field(
        5,
        ge=1,
        le=20,
    )


# ── Doubts ────────────────────────────────────────────────────────────────────


class ConversationTurn(BaseModel):
    """A single turn in a conversation history."""

    role: Literal["user", "assistant"]

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


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
        description=(
            "If provided, answers are grounded in that document via RAG."
        ),
    )

    history: list[ConversationTurn] = Field(
        default_factory=list,
        description=(
            "Optional conversation history for multi-turn context. "
            "Each turn has 'role' ('user' or 'assistant') and 'content'. "
            "Omit or pass [] for single-turn behaviour."
        ),
    )


# ── Resume Prep ───────────────────────────────────────────────────────────────

class ResumeQuizGenerateRequest(BaseModel):
    item_id: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    number_of_questions: int = Field(5, ge=1, le=20)


# ── Coding ────────────────────────────────────────────────────────────────────

class CodeExecutionRequest(BaseModel):
    language: str = Field(..., description="Programming language (e.g., 'java', 'python')")
    code: str = Field(..., description="The source code to execute")
    stdin: str = Field("", description="Standard input for the program")