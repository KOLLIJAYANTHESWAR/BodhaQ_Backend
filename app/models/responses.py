"""
Response models for BodhaQ API.
These are what the frontend receives. Correct quiz answers are NEVER
included in the quiz-generation response — only in the evaluation response.
"""
from pydantic import BaseModel, Field


# ── Learning ──────────────────────────────────────────────────────────────────

class CodeExample(BaseModel):
    code: str
    explanation: str


class LearningContent(BaseModel):
    topic: str
    definition: str
    key_concepts: list[str]
    example: CodeExample
    important_points: list[str]


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


# ── Quiz (generation — answers hidden) ───────────────────────────────────────

class QuizOption(BaseModel):
    """A single answer option shown to the user."""
    letter: str        # "A", "B", "C", "D"
    text: str


class QuizQuestionPublic(BaseModel):
    """Question as sent to the frontend — no correct_answer field."""
    id: int
    question: str
    options: list[QuizOption]
    topic: str = ""   # used later for weak-topic detection


class QuizGenerateResponse(BaseModel):
    quiz_id: str
    source_type: str
    source_id: str
    questions: list[QuizQuestionPublic]


# ── Quiz (evaluation) ────────────────────────────────────────────────────────

class MistakeDetail(BaseModel):
    question_id: int
    question: str
    correct_answer: str
    user_answer: str
    explanation: str
    topic: str


class QuizEvaluationResponse(BaseModel):
    quiz_id: str
    score: int
    total: int
    percentage: float
    mistakes: list[MistakeDetail]
    timestamp: float = 0.0


# ── Weak Topics ───────────────────────────────────────────────────────────────

class WeakTopicItem(BaseModel):
    topic: str
    accuracy: float     # 0–100 %
    status: str         # "Needs Practice", "Improving", "Learned"
    source_type: str
    source_id: str
    quiz_count: int


class QuizHistoryItem(BaseModel):
    quiz_id: str
    title: str
    score: int
    total: int
    percentage: float
    timestamp: float


class WeakTopicsResponse(BaseModel):
    weak_topics: list[WeakTopicItem]
    recent_quizzes: list[QuizHistoryItem]


# ── Doubts ────────────────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    document: str
    page: int | None = None


class DoubtResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = []


# ── Errors ────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    code: str


# ── Resume Prep ───────────────────────────────────────────────────────────────

class ResumeSkill(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: str
    score: int | None = None

class ResumeProject(BaseModel):
    id: str
    name: str
    description: str | None = None
    technologies: list[str] = []
    status: str
    score: int | None = None

class ResumeCertification(BaseModel):
    id: str
    name: str
    status: str
    score: int | None = None

class ResumeProgressResponse(BaseModel):
    resume_id: str | None
    filename: str | None
    skills: list[ResumeSkill] = []
    projects: list[ResumeProject] = []
    certifications: list[ResumeCertification] = []
    overall_progress: float = 0.0
    items_completed: int = 0
    total_items: int = 0


# ── Coding ────────────────────────────────────────────────────────────────────

class CodeExecutionResponse(BaseModel):
    status: str = Field(..., description="E.g., success, compilation_error, runtime_error, timeout, memory_limit, output_limit, execution_error, unsupported_language")
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int

