"""
Response models for BodhaQ API.

These models define the data returned to the frontend.

Important security rule:
    Correct quiz answers are NEVER included in quiz-generation
    responses. They are returned only as part of quiz evaluation.

Coding security rule:
    Hidden coding tests are NEVER returned to the frontend.
    Only hidden test counts are exposed.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING
# ─────────────────────────────────────────────────────────────────────────────


class CodeExample(BaseModel):
    """A code example included in learning content."""

    code: str = Field(
        ...,
        max_length=20000,
    )

    explanation: str = Field(
        ...,
        max_length=10000,
    )


class LearningResource(BaseModel):
    """
    External learning resource returned by the resource-search service.

    URLs are obtained from the backend resource-search pipeline.
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    url: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    description: str = Field(
        default="",
        max_length=5000,
    )

    resource_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    domain: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )


class LearningVideo(BaseModel):
    """
    Learning video returned by the video-search pipeline.

    Video metadata is optional because the external search provider
    may not expose fields such as thumbnail, channel, or duration.
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    url: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    thumbnail: str | None = Field(
        default=None,
        max_length=2000,
    )

    channel: str | None = Field(
        default=None,
        max_length=300,
    )

    description: str = Field(
        default="",
        max_length=5000,
    )

    duration: str | None = Field(
        default=None,
        max_length=100,
    )


class LearningContent(BaseModel):
    """Structured learning content generated for a topic."""

    topic: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    definition: str = Field(
        ...,
        min_length=1,
        max_length=20000,
    )

    key_concepts: list[str] = Field(
        ...,
        max_length=50,
    )

    example: CodeExample

    important_points: list[str] = Field(
        ...,
        max_length=50,
    )

    resources: list[LearningResource] = Field(
        default_factory=list,
        max_length=30,
    )

    videos: list[LearningVideo] = Field(
        default_factory=list,
        max_length=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────


class DocumentUploadResponse(BaseModel):
    """Response returned after successfully ingesting a document."""

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    filename: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    chunk_count: int = Field(
        ...,
        ge=0,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class DocumentListItem(BaseModel):
    """Summary information for one stored document."""

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    filename: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    chunk_count: int = Field(
        ...,
        ge=0,
    )


class DocumentListResponse(BaseModel):
    """List of stored documents."""

    documents: list[DocumentListItem] = Field(
        default_factory=list,
        max_length=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ — GENERATION
# ─────────────────────────────────────────────────────────────────────────────


class QuizOption(BaseModel):
    """A single answer option shown to the student."""

    letter: Literal[
        "A",
        "B",
        "C",
        "D",
    ]

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class QuizQuestionPublic(BaseModel):
    """
    Question returned to the frontend.

    IMPORTANT:
        correct_answer is intentionally NOT included.
    """

    id: int = Field(
        ...,
        ge=1,
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    options: list[QuizOption] = Field(
        ...,
        min_length=2,
        max_length=4,
    )

    # Used by the backend for weak-topic detection.
    topic: str = Field(
        default="",
        max_length=300,
    )

    @field_validator("options")
    @classmethod
    def validate_options(
        cls,
        value: list[QuizOption],
    ) -> list[QuizOption]:
        letters = [
            option.letter
            for option in value
        ]

        if len(set(letters)) != len(letters):
            raise ValueError(
                "Quiz option letters must be unique."
            )

        return value


class QuizGenerateResponse(BaseModel):
    """
    Quiz returned after generation.

    Supported quiz sources:
        - topic: Topic-based assessment.
        - document: Document-grounded assessment.
        - resume_item: Resume interview assessment
          generated from an extracted resume item.

    Correct answers are intentionally excluded.
    """

    quiz_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    source_type: Literal[
        "topic",
        "document",
        "resume_item",
    ]

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    questions: list[QuizQuestionPublic] = Field(
        ...,
        min_length=1,
        max_length=30,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


class MistakeDetail(BaseModel):
    """Details of a question answered incorrectly."""

    question_id: int = Field(
        ...,
        ge=1,
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    correct_answer: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    user_answer: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    explanation: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    topic: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )


class QuizEvaluationResponse(BaseModel):
    """Deterministic evaluation result for a submitted quiz."""

    quiz_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    score: int = Field(
        ...,
        ge=0,
    )

    total: int = Field(
        ...,
        ge=0,
    )

    percentage: float = Field(
        ...,
        ge=0,
        le=100,
    )

    mistakes: list[MistakeDetail] = Field(
        default_factory=list,
        max_length=30,
    )

    # Unix timestamp.
    timestamp: float = Field(
        default=0.0,
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WEAK TOPICS
# ─────────────────────────────────────────────────────────────────────────────


class WeakTopicItem(BaseModel):
    """
    Weak-topic information.

    Status values used by the frontend:
        Needs Practice
        Improving
        Learned
    """

    topic: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    accuracy: float = Field(
        ...,
        ge=0,
        le=100,
    )

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    source_type: Literal[
        "topic",
        "document",
    ]

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    quiz_count: int = Field(
        ...,
        ge=0,
    )


class QuizHistoryItem(BaseModel):
    """Summary of a completed quiz stored in quiz history."""

    quiz_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    score: int = Field(
        ...,
        ge=0,
    )

    total: int = Field(
        ...,
        ge=0,
    )

    percentage: float = Field(
        ...,
        ge=0,
        le=100,
    )

    timestamp: float = Field(
        ...,
        ge=0,
    )


class WeakTopicsResponse(BaseModel):
    """
    Legacy/backend response containing weak topics
    and recent quizzes.
    """

    weak_topics: list[WeakTopicItem] = Field(
        default_factory=list,
        max_length=100,
    )

    recent_quizzes: list[QuizHistoryItem] = Field(
        default_factory=list,
        max_length=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOUBTS
# ─────────────────────────────────────────────────────────────────────────────


class SourceReference(BaseModel):
    """Source used by RAG when answering a document-grounded doubt."""

    document: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    page: int | None = Field(
        default=None,
        ge=1,
    )


class DoubtResponse(BaseModel):
    """Response from the Doubts feature."""

    answer: str = Field(
        ...,
        min_length=1,
        max_length=30000,
    )

    sources: list[SourceReference] = Field(
        default_factory=list,
        max_length=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ERRORS
# ─────────────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    code: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESUME PREP
# ─────────────────────────────────────────────────────────────────────────────


class ResumeSkill(BaseModel):
    """A skill extracted from a resume."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    description: str | None = Field(
        default=None,
        max_length=5000,
    )

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )


class ResumeProject(BaseModel):
    """A project extracted from a resume."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    description: str | None = Field(
        default=None,
        max_length=10000,
    )

    technologies: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )


class ResumeCertification(BaseModel):
    """A certification extracted from a resume."""

    id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )


class ResumeProgressResponse(BaseModel):
    """Current Resume Prep progress."""

    resume_id: str | None = Field(
        default=None,
        max_length=200,
    )

    filename: str | None = Field(
        default=None,
        max_length=500,
    )

    skills: list[ResumeSkill] = Field(
        default_factory=list,
        max_length=100,
    )

    projects: list[ResumeProject] = Field(
        default_factory=list,
        max_length=100,
    )

    certifications: list[ResumeCertification] = Field(
        default_factory=list,
        max_length=100,
    )

    overall_progress: float = Field(
        default=0.0,
        ge=0,
        le=100,
    )

    items_completed: int = Field(
        default=0,
        ge=0,
    )

    total_items: int = Field(
        default=0,
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — EXECUTION
# ─────────────────────────────────────────────────────────────────────────────


class CodeExecutionResponse(BaseModel):
    """
    Result of executing code.

    Possible status values include:
        success
        compilation_error
        runtime_error
        timeout
        memory_limit
        output_limit
        execution_error
        execution_service_unavailable
        unsupported_language
    """

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Execution result status.",
    )

    stdout: str = Field(
        default="",
        max_length=100000,
    )

    stderr: str = Field(
        default="",
        max_length=100000,
    )

    exit_code: int = Field(
        default=0,
    )

    execution_time_ms: int = Field(
        default=0,
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — TEST CASES
# ─────────────────────────────────────────────────────────────────────────────


class CodingTestCase(BaseModel):
    """
    Public coding test case.

    This model is safe to return to the frontend.
    """

    input: str = Field(
        ...,
        max_length=20000,
    )

    output: str = Field(
        ...,
        max_length=20000,
    )


class CodingProblemExample(BaseModel):
    """Visible problem example."""

    input: str = Field(
        ...,
        max_length=20000,
    )

    output: str = Field(
        ...,
        max_length=20000,
    )

    explanation: str = Field(
        default="",
        max_length=5000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — PROBLEM GENERATION
# ─────────────────────────────────────────────────────────────────────────────


class CodingProblemGenerateResponse(BaseModel):
    """
    Generated coding workspace.

    IMPORTANT:
        Hidden test inputs and expected outputs are NOT returned.

        Only:
            hidden_test_count

        is exposed.
    """

    problem_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    statement: str = Field(
        ...,
        min_length=1,
        max_length=20000,
    )

    input_format: str = Field(
        ...,
        max_length=5000,
    )

    output_format: str = Field(
        ...,
        max_length=5000,
    )

    constraints: str = Field(
        ...,
        max_length=10000,
    )

    examples: list[CodingProblemExample] = Field(
        default_factory=list,
        max_length=20,
    )

    difficulty: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    topics: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    starter_code_java: str = Field(
        ...,
        max_length=50000,
    )

    starter_code_python: str = Field(
        ...,
        max_length=50000,
    )

    # Visible tests only.
    public_tests: list[CodingTestCase] = Field(
        default_factory=list,
        max_length=50,
    )

    # NEVER expose hidden test data.
    hidden_test_count: int = Field(
        ...,
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — AI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────


class CodeAnalyzeResponse(BaseModel):
    """AI explanation of the submitted code."""

    explanation: str = Field(
        ...,
        min_length=1,
        max_length=30000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — AI IMPROVEMENT
# ─────────────────────────────────────────────────────────────────────────────


class CodeImproveResponse(BaseModel):
    """
    AI improvement analysis.

    The optimized code is returned as a suggestion.
    It does not automatically overwrite the user's editor.
    """

    explanation: str = Field(
        ...,
        min_length=1,
        max_length=30000,
    )

    current_complexity: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    possible_complexity: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    optimized_code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — GENERATED TEST CASES
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseGenerateResponse(BaseModel):
    """
    Additional tests generated by AI.

    Public tests are returned.
    Hidden tests remain backend-only.
    """

    public_tests: list[CodingTestCase] = Field(
        default_factory=list,
        max_length=50,
    )

    hidden_test_count: int = Field(
        ...,
        ge=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CODING — SUBMISSION / JUDGE
# ─────────────────────────────────────────────────────────────────────────────


class CodeSubmitResponse(BaseModel):
    """
    Coding submission result.

    Hidden-test information is intentionally restricted.

    For hidden failures:
        - hidden_test_failed is true
        - failed_test_type is "hidden"

    Hidden input, expected output and actual output are NEVER returned.
    """

    status: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    details: str = Field(
        default="",
        max_length=5000,
    )

    passed_tests: int = Field(
        default=0,
        ge=0,
    )

    total_tests: int = Field(
        default=0,
        ge=0,
    )

    passed_samples: int = Field(
        default=0,
        ge=0,
    )

    total_samples: int = Field(
        default=0,
        ge=0,
    )

    passed_public: int = Field(
        default=0,
        ge=0,
    )

    total_public: int = Field(
        default=0,
        ge=0,
    )

    passed_hidden: int = Field(
        default=0,
        ge=0,
    )

    total_hidden: int = Field(
        default=0,
        ge=0,
    )

    # Public/sample failure information.
    failed_test_type: Literal[
        "sample",
        "public",
        "hidden",
    ] | None = None

    failed_test_input: str | None = Field(
        default=None,
        max_length=20000,
    )

    failed_test_expected: str | None = Field(
        default=None,
        max_length=20000,
    )

    failed_test_actual: str | None = Field(
        default=None,
        max_length=20000,
    )

    # Hidden test failure indicator.
    hidden_test_failed: bool = False
    