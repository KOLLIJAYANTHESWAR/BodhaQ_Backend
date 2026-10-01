"""
GeminiService — the single point of contact with the Gemini API.

All other services call methods here. No other module imports the
google.genai client directly. This keeps AI logic isolated and testable.

Important:
    - Gemini credentials are supplied per request and are never persisted.
    - Structured AI responses are validated with Pydantic.
    - Coding problems are validated before they enter ProblemStore.
    - Hidden coding tests are never returned to the frontend.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError



logger = logging.getLogger(__name__)


# ============================================================================
# Gemini configuration
# ============================================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0


# ============================================================================
# Custom exceptions
# ============================================================================


class GeminiAuthenticationError(RuntimeError):
    """Raised when Gemini API authentication fails."""

    pass


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini API quota or rate limit is exceeded."""

    pass


class GeminiInvalidResponseError(RuntimeError):
    """Raised when Gemini returns unusable structured data."""

    pass


# ============================================================================
# Internal Gemini response schemas
# ============================================================================


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


# ============================================================================
# Coding schemas
# ============================================================================


class _CodingTestCase(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = ""


class _CodingProblemExample(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = ""
    explanation: str = ""


class _CodingProblemSchema(BaseModel):
    title: str
    statement: str
    input_format: str
    output_format: str
    constraints: str

    examples: list[_CodingProblemExample]

    difficulty: str
    topics: list[str]

    starter_code_java: str
    starter_code_python: str

    public_tests: list[_CodingTestCase]
    hidden_tests: list[_CodingTestCase]


class _CodeAnalyzeSchema(BaseModel):
    explanation: str


class _CodeImproveSchema(BaseModel):
    explanation: str
    current_complexity: str
    possible_complexity: str
    optimized_code: str


class _TestCaseSchema(BaseModel):
    public_tests: list[_CodingTestCase]
    hidden_tests: list[_CodingTestCase]


StructuredModel = TypeVar(
    "StructuredModel",
    bound=BaseModel,
)


# ============================================================================
# Service
# ============================================================================


class GeminiService:
    """Centralized Gemini API service for BodhaQ."""

    def __init__(self) -> None:
        """Initialize the request-scoped Gemini service."""

        # Do not keep a Gemini client or API key globally. Each request
        # supplies its own user-provided key.
        pass

    # ========================================================================
    # Client
    # ========================================================================

    @staticmethod
    def _validate_api_key(
        api_key: str,
    ) -> str:
        """Validate and normalize a request-scoped Gemini API key."""

        if not isinstance(api_key, str):
            raise GeminiAuthenticationError(
                "Gemini API key is required."
            )

        normalized = api_key.strip()

        if not normalized:
            raise GeminiAuthenticationError(
                "Gemini API key is required."
            )

        return normalized

    @classmethod
    def _get_client(
        cls,
        api_key: str,
    ) -> genai.Client:
        """Create a Gemini client using the request-scoped API key."""

        validated_key = cls._validate_api_key(api_key)

        try:
            return genai.Client(
                api_key=validated_key,
            )

        except Exception as exc:
            logger.error(
                "[Gemini] Failed to initialize request client: %s",
                type(exc).__name__,
            )

            raise GeminiAuthenticationError(
                "Unable to initialize the Gemini client. "
                "Please verify the supplied API key."
            ) from exc

    # ========================================================================
    # Learning
    # ========================================================================

    def generate_learning_content(
        self,
        topic: str,
        *,
        api_key: str,
    ) -> _LearningContentSchema:
        """Generate structured learning content for a topic."""

        topic = (topic or "").strip()

        if not topic:
            raise ValueError(
                "Learning topic cannot be empty."
            )

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
- Return only raw source code in the code field.
- The explanation must describe what the code actually does.

FORMATTING REQUIREMENTS:

- The code field MUST contain real newline characters.
- NEVER return the entire program on a single line.
- Put each import on its own line.
- Put each class declaration on its own line.
- Put each method declaration on its own line.
- Put each statement on its own line.
- Put closing braces on appropriate separate lines.
- Indent code consistently.
- Do not minify or compress code.

IMPORTANT:

If the example uses text values such as country names, city names,
student names, or messages, represent those values using valid
string literals for the selected language.

For example, in Java:

capitalCities.put("England", "London");

NOT:

capitalCities.put("England", London);

Do not translate natural-language values into bare identifiers.
"""

        result = self._generate_structured(
            prompt=prompt,
            schema=_LearningContentSchema,
            api_key=api_key,
        )

        self._validate_learning_content(result)

        return result

    # ========================================================================
    # Quiz
    # ========================================================================

    def generate_quiz(
        self,
        context: str,
        topic_hint: str,
        num_questions: int,
        difficulty: str,
        *,
        api_key: str,
    ) -> _QuizSchema:
        """Generate a multiple-choice quiz."""

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
- Do not contradict the provided study material.
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_QuizSchema,
            api_key=api_key,
        )

    # ========================================================================
    # Doubts
    # ========================================================================

    def answer_doubt(
        self,
        question: str,
        context: str = "",
        history: list[dict] | None = None,
        *,
        api_key: str,
    ) -> str:
        """Answer a student question."""

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

        return self._generate_text(
            prompt,
            api_key=api_key,
        )

    # ========================================================================
    # Explanation
    # ========================================================================

    def generate_explanation(
        self,
        topic: str,
        context: str = "",
        *,
        api_key: str,
    ) -> str:
        """Generate a concise explanation."""

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

        return self._generate_text(
            prompt,
            api_key=api_key,
        )

    # ========================================================================
    # Practice
    # ========================================================================

    def generate_practice(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        context: str = "",
        *,
        api_key: str,
    ) -> _QuizSchema:
        """Generate targeted practice."""

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
            api_key=api_key,
        )

    # ========================================================================
    # Resume extraction
    # ========================================================================

    def extract_resume(
        self,
        text: str,
        *,
        api_key: str,
    ) -> _ResumeExtractionSchema:
        """Extract structured information from resume text."""

        prompt = f"""
You are BodhaQ, an expert technical recruiter and resume analyzer.

Extract the following information from the provided resume text:

1. Technical Skills
   Programming languages, frameworks, tools, databases, cloud,
   DevOps tools, libraries, and other technical skills.

2. Projects
   Extract project title, description, and technologies used.

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
            api_key=api_key,
        )

    # ========================================================================
    # Resume quizzes
    # ========================================================================

    def generate_resume_quiz(
        self,
        item_type: str,
        item_data: dict,
        num_questions: int,
        difficulty: str,
        *,
        api_key: str,
    ) -> _QuizSchema:
        """Generate a quiz for a resume item."""

        if item_type == "skill":

            prompt = f"""
You are BodhaQ, an expert technical interviewer.

Generate {num_questions} multiple-choice questions for:

Skill:
{item_data["name"]}

Difficulty:
{difficulty}

Focus on:

Easy:
- Fundamentals
- Definitions
- Basic concepts
- Differences

Medium:
- Practical understanding
- Implementation decisions
- Common scenarios
- Debugging

Hard:
- Internals
- Trade-offs
- Optimization
- Interview-level understanding

Rules:

- Exactly {num_questions} questions.
- Exactly 4 options per question.
- Options use A., B., C., D.
- correct_answer is A, B, C, or D.
- Provide a clear explanation.
- IDs sequential from 1.
"""

        elif item_type == "project":

            technologies = ", ".join(
                item_data.get("technologies", [])
            )

            prompt = f"""
You are BodhaQ, an expert technical interviewer.

Generate {num_questions} multiple-choice interview questions
based on this project.

Project:
{item_data["name"]}

Description:
{item_data["description"]}

Technologies:
{technologies}

Difficulty:
{difficulty}

Do NOT ask generic questions unrelated to the project.

Focus on:

- Architecture
- Technology choices
- Data flow
- APIs
- Database design
- Security
- Performance
- Scalability
- Deployment
- Debugging
- Trade-offs

Do not invent technologies not mentioned in the project.

Rules:

- Exactly {num_questions} questions.
- Exactly 4 options.
- Options use A., B., C., D.
- correct_answer is A, B, C, or D.
- Provide explanations.
- IDs sequential from 1.
"""

        elif item_type == "certification":

            prompt = f"""
You are BodhaQ, an expert technical interviewer.

Generate {num_questions} multiple-choice questions based on:

Certification:
{item_data["name"]}

Difficulty:
{difficulty}

Focus on concepts relevant to the certification domain.

Rules:

- Exactly {num_questions} questions.
- Exactly 4 options.
- Options use A., B., C., D.
- correct_answer is A, B, C, or D.
- Provide explanations.
- IDs sequential from 1.
"""

        else:
            raise ValueError(
                f"Unknown item type: {item_type}"
            )

        return self._generate_structured(
            prompt=prompt,
            schema=_QuizSchema,
            api_key=api_key,
        )

    # ========================================================================
    # CODING — PROBLEM GENERATION
    # ========================================================================

    def generate_coding_problem(
        self,
        title: str,
        statement: str,
        constraints: str,
        sample: str,
        previous_errors: list[str] | None = None,
        *,
        api_key: str,
    ) -> _CodingProblemSchema:
        """
        Generate a complete executable coding problem.

        The generated package is validated before it is accepted.

        The internal response contains hidden tests. These are never
        returned directly to the frontend.
        """

        title = (title or "").strip()
        statement = (statement or "").strip()
        constraints = (constraints or "").strip()
        sample = (sample or "").strip()

        previous_errors = previous_errors or []

        previous_error_block = ""

        if previous_errors:
            # Only send the most recent validation failures to avoid
            # unnecessarily growing the prompt.
            recent_errors = previous_errors[-6:]

            previous_error_block = """
IMPORTANT — PREVIOUS GENERATION ATTEMPTS FAILED.

You MUST correct these specific validation failures:

""" + "\n".join(
                f"- {error}"
                for error in recent_errors
            ) + """

Do not repeat these mistakes in the new response.
"""

        prompt = f"""
You are BodhaQ, an expert competitive-programming problem designer.

Generate a COMPLETE and EXECUTABLE coding problem.

USER INPUT
==========

Problem name:
{title}

Problem statement:
{statement}

Constraints:
{constraints}

Sample test case:
{sample}

{previous_error_block}

INPUT INTERPRETATION
====================

The user may provide:

1. only a problem name
2. only a problem statement
3. both
4. optional constraints
5. optional sample test case

If only a name is provided, design the complete problem yourself.

If a statement is provided, preserve its intended meaning.

If constraints or sample information are missing, generate reasonable
ones.

Do NOT silently contradict information provided by the user.

If you introduce assumptions because information was missing, clearly
mention those assumptions in the problem statement or constraints.

Do not add the phrase "AI-generated assumption" to every individual
constraint line. Keep the actual constraints clean.

PROBLEM PACKAGE
===============

Generate ALL of these:

- title
- statement
- input_format
- output_format
- constraints
- examples
- difficulty
- topics
- starter_code_java
- starter_code_python
- public_tests
- hidden_tests

Everything MUST describe the exact same problem.

The following must agree with each other:

statement
input format
output format
constraints
examples
public tests
hidden tests
Java solution
Python solution

Do not create contradictory information.

EXAMPLES
========

Generate at least 2 examples.

Every example MUST contain:

input
output
explanation

CRITICAL:

The output field MAY be empty or may contain your best expected output.

IMPORTANT:
- The backend will independently execute the generated reference solutions
  and compute/verify expected outputs before the problem is published.
- Therefore, do not rely on the generated output field being authoritative.
- If you provide an output, it must be a concrete expected result.
- Never use prose placeholders such as "N/A", "unknown", "TODO", or
  "depends" as a claimed expected result.

PUBLIC TESTS
============

Generate at least 2 public tests.

Every public test MUST contain:

input
output

The output may be empty because the backend will independently compute
and verify the expected result before publication.

PUBLIC TESTS MUST:

- follow the input format
- follow the stated constraints
- satisfy all stated assumptions
- have deterministic outputs
- have correct expected outputs
- represent meaningful cases

Include useful cases such as:

- normal case
- edge case
- boundary case

when applicable.

HIDDEN TESTS
============

Generate at least 3 hidden tests.

Every hidden test MUST contain:

input
output

The output may be empty because the backend will independently compute
and verify the expected result before publication.

Hidden tests MUST:

- follow the input format
- follow the stated constraints
- satisfy all assumptions
- have deterministic outputs
- have correct expected outputs

Include additional edge/boundary cases where appropriate.

IMPORTANT:

Hidden tests are INTERNAL ONLY.

They must never be exposed in the public API response.

EXPECTED OUTPUT RULE
====================

The backend is responsible for authoritative expected outputs.

For EVERY example, public test, and hidden test:

1. Read the input.
2. Provide a valid test input that follows the problem contract.
3. If you provide an expected output, make it concrete and deterministic.
4. The backend will execute the generated Java and Python reference
   solutions against the test input.
5. The backend will use the verified execution result as the authoritative
   expected output before publishing the problem.

NEVER use prose placeholders as claimed expected outputs.

The generated Java and Python solutions MUST produce the same output for
every generated test after backend verification.

INPUT FORMAT
============

The starter code must read input exactly according to input_format.

Do not generate tests using a different format.

For example, if the input format says:

First line contains N.
Second line contains N integers.

then tests must actually use:

N
a1 a2 ... aN

Do not mix incompatible formats.

JAVA REQUIREMENTS
=================

Generate COMPLETE executable Java source code.

The Java code MUST contain exactly:

public class Main

and:

public static void main(String[] args)

It must:

- read stdin
- parse the stated input format
- solve the problem
- print the required output

Do NOT return only:

class Solution

Do NOT return only a method.

Do NOT use external libraries.

Do NOT read files.

Do NOT use network access.

Do NOT include Markdown fences.

The code must be executable directly by javac/java.

PYTHON REQUIREMENTS
===================

Generate COMPLETE executable Python source code.

The Python code MUST contain:

if __name__ == "__main__":

It must:

- read stdin
- parse the stated input format
- solve the problem
- print the required output

Do NOT return only a function.

Do NOT use external packages.

Do NOT read files.

Do NOT use network access.

Do NOT include Markdown fences.

The code must be executable directly by Python.

CODE FORMATTING
===============

Code fields MUST contain actual newline characters.

Do not minify code.

Do not put an entire program on one line.

Preserve indentation.

TEST DETERMINISM
================

The problem MUST have deterministic output.

Avoid problems where output ordering is unspecified unless the output
contract explicitly defines a canonical ordering.

If the output contains collections, define their ordering clearly.

Do not create ambiguous problems.

QUALITY REQUIREMENT
===================

Before returning the JSON, mentally verify:

- Every example has non-empty input.
- Every example has non-empty output.
- Every public test has non-empty input.
- Every public test has non-empty output.
- Every hidden test has non-empty input.
- Every hidden test has non-empty output.
- Every test follows constraints.
- Every test follows the input format.
- Java and Python solve the same problem.
- The expected outputs are logically correct.
- The starter code is executable.
- The problem statement is internally consistent.

Return ONLY the structured JSON matching the requested schema.
"""

        result = self._generate_structured(
            prompt=prompt,
            schema=_CodingProblemSchema,
            api_key=api_key,
        )

        self._validate_coding_problem(result)

        return result

    # ========================================================================
    # CODING — ANALYZE
    # ========================================================================

    def analyze_code(
        self,
        problem_statement: str,
        sample: str,
        constraints: str,
        code: str,
        language: str,
        *,
        api_key: str,
    ) -> _CodeAnalyzeSchema:

        problem_text = (
            problem_statement
            if problem_statement
            else (
                "No specific problem provided. "
                "Analyze the code's general purpose and correctness."
            )
        )

        prompt = f"""
You are BodhaQ, an expert coding instructor.

Analyze the user's {language} code.

Problem:
{problem_text}

Constraints:
{constraints}

Sample:
{sample}

User's Code:
{code}

If the code is correct:

- explain the algorithm
- explain the flow
- explain important variables
- give time complexity
- give space complexity

If the code is incorrect:

- explain what is wrong
- explain why it is wrong
- provide a failing scenario
- explain what the user should reconsider

Do NOT output replacement code.
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_CodeAnalyzeSchema,
            api_key=api_key,
        )

    # ========================================================================
    # CODING — IMPROVE
    # ========================================================================

    def improve_code(
        self,
        problem_statement: str,
        sample: str,
        constraints: str,
        code: str,
        language: str,
        *,
        api_key: str,
    ) -> _CodeImproveSchema:

        problem_text = (
            problem_statement
            if problem_statement
            else (
                "No specific problem provided. "
                "Optimize the code in general."
            )
        )

        prompt = f"""
You are BodhaQ, an expert coding instructor.

Improve the user's {language} code.

Problem:
{problem_text}

Constraints:
{constraints}

Sample:
{sample}

User's Code:
{code}

If the code is incorrect:

- explain the problem
- explain the fix
- generate corrected code

If the code is correct but suboptimal:

- explain the optimization
- provide current complexity
- provide possible improved complexity
- generate optimized code

If the code is already appropriate:

- say that the current approach is appropriate
- provide its complexity
- return the same code

Do not claim an approach is optimal unless the analysis supports it.
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_CodeImproveSchema,
            api_key=api_key,
        )

    # ========================================================================
    # CODING — TEST CASE GENERATION
    # ========================================================================

    def generate_test_cases(
        self,
        problem_statement: str,
        sample: str,
        constraints: str,
        code: str,
        language: str,
        *,
        api_key: str,
    ) -> _TestCaseSchema:

        problem_text = (
            problem_statement
            if problem_statement
            else (
                "No specific problem provided. "
                "Infer the intent from the code and generate valid tests."
            )
        )

        prompt = f"""
You are BodhaQ, an expert coding instructor.

Generate comprehensive test cases.

Problem:
{problem_text}

Constraints:
{constraints}

Sample:
{sample}

Current User Code:
{code}

Language:
{language}

Generate public and hidden tests.

Include where applicable:

- basic cases
- edge cases
- boundary cases
- duplicate values
- minimum values
- maximum values
- special cases
- stress cases

CRITICAL:

Every generated test MUST have:

- non-empty input
- non-empty expected output

Never use:

""

" "

null

"N/A"

"unknown"

"TODO"

"..."

as an expected output.

Every test must follow the problem's input format and constraints.

Expected outputs must be logically correct.

Hidden tests remain backend-only.
"""

        return self._generate_structured(
            prompt=prompt,
            schema=_TestCaseSchema,
            api_key=api_key,
        )

    # ========================================================================
    # LEARNING VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_learning_content(
        content: _LearningContentSchema,
    ) -> None:
        """Validate generated learning content."""

        if not content.topic.strip():
            raise GeminiInvalidResponseError(
                "AI service returned an invalid learning topic."
            )

        if not content.definition.strip():
            raise GeminiInvalidResponseError(
                "AI service returned an empty definition."
            )

        if not content.key_concepts:
            raise GeminiInvalidResponseError(
                "AI service returned no key concepts."
            )

        if not content.important_points:
            raise GeminiInvalidResponseError(
                "AI service returned no important points."
            )

        code = content.example.code.strip()

        if not code:
            raise GeminiInvalidResponseError(
                "AI service returned an empty code example."
            )

        if code.startswith("```") or code.endswith("```"):
            raise GeminiInvalidResponseError(
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
            raise GeminiInvalidResponseError(
                "AI service returned a code example containing "
                "unsupported placeholders."
            )

        if not content.example.explanation.strip():
            raise GeminiInvalidResponseError(
                "AI service returned an empty code explanation."
            )

        if "\n" not in code and len(code) > 60:
            raise GeminiInvalidResponseError(
                "AI service returned code as a single line."
            )

    # ========================================================================
    # CODING VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_coding_problem(
        problem: _CodingProblemSchema,
    ) -> None:
        """
        Deterministic validation of an AI-generated coding problem.

        This is deliberately strict.

        We do not want to publish a coding problem until its structure,
        tests, and executable starter code meet the minimum contract.
        """

        # --------------------------------------------------------------------
        # Basic problem fields
        # --------------------------------------------------------------------

        if not problem.title.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has an empty title."
            )

        if not problem.statement.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has an empty statement."
            )

        if not problem.input_format.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has no input format."
            )

        if not problem.output_format.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has no output format."
            )

        if not problem.constraints.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has no constraints."
            )

        # --------------------------------------------------------------------
        # Examples
        # --------------------------------------------------------------------

        if len(problem.examples) < 2:
            raise GeminiInvalidResponseError(
                "AI-generated coding problem must contain at least 2 examples."
            )

        for index, example in enumerate(
            problem.examples,
            start=1,
        ):
            if not example.input.strip():
                raise GeminiInvalidResponseError(
                    f"Example {index} has empty input."
                )

            if not example.explanation.strip():
                raise GeminiInvalidResponseError(
                    f"Example {index} has empty explanation."
                )

            # Expected outputs are verified and populated by the backend
            # execution pipeline. Do not trust Gemini's output field as the
            # authoritative judge result.

        # --------------------------------------------------------------------
        # Starter code
        # --------------------------------------------------------------------

        if not problem.starter_code_java.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has empty Java starter code."
            )

        if not problem.starter_code_python.strip():
            raise GeminiInvalidResponseError(
                "AI-generated coding problem has empty Python starter code."
            )

        java_code = problem.starter_code_java

        if "public class Main" not in java_code:
            raise GeminiInvalidResponseError(
                "Generated Java starter code must contain "
                "'public class Main'."
            )

        if "public static void main(String[] args)" not in java_code:
            raise GeminiInvalidResponseError(
                "Generated Java starter code must contain "
                "'public static void main(String[] args)'."
            )

        python_code = problem.starter_code_python

        python_main_guard_pattern = re.compile(
            r"""
            ^\s*
            if\s+
            __name__\s*==\s*
            (?:"__main__"|'__main__')
            \s*:
            """,
            re.MULTILINE | re.VERBOSE,
        )

        if not python_main_guard_pattern.search(python_code):
            raise GeminiInvalidResponseError(
                'Generated Python starter code must contain a valid '
                'if __name__ == "__main__": entry point.'
            )

        # --------------------------------------------------------------------
        # Public tests
        # --------------------------------------------------------------------

        if len(problem.public_tests) < 2:
            raise GeminiInvalidResponseError(
                "AI-generated coding problem must contain at least "
                "2 public tests."
            )

        for index, test in enumerate(
            problem.public_tests,
            start=1,
        ):
            if not test.input.strip():
                raise GeminiInvalidResponseError(
                    f"Public test {index} has empty input."
                )

            # Expected output is intentionally not validated here.
            # The coding generation pipeline computes and verifies it by
            # executing the generated reference solutions.

        # --------------------------------------------------------------------
        # Hidden tests
        # --------------------------------------------------------------------

        if len(problem.hidden_tests) < 3:
            raise GeminiInvalidResponseError(
                "AI-generated coding problem must contain at least "
                "3 hidden tests."
            )

        for index, test in enumerate(
            problem.hidden_tests,
            start=1,
        ):
            if not test.input.strip():
                raise GeminiInvalidResponseError(
                    f"Hidden test {index} has empty input."
                )

            # Expected output is intentionally not validated here.
            # Hidden expected outputs are computed and verified by the
            # backend and never exposed to the frontend.

    # ========================================================================
    # Placeholder detection
    # ========================================================================

    @staticmethod
    def _looks_like_placeholder(
        value: str,
    ) -> bool:
        """
        Detect obviously unusable AI placeholder outputs.

        This does NOT attempt to determine whether an answer is
        mathematically correct. That is handled by execution validation.
        """

        normalized = value.strip().lower()

        if not normalized:
            return True

        placeholders = {
            "n/a",
            "na",
            "unknown",
            "todo",
            "tbd",
            "depends",
            "not applicable",
            "not available",
            "...",
            "<output>",
            "<expected_output>",
            "<expected output>",
            "your output",
            "expected output",
        }

        return normalized in placeholders

    # ========================================================================
    # Gemini error classification
    # ========================================================================

    @staticmethod
    def _is_authentication_error(
        exc: Exception,
    ) -> bool:
        """Detect Gemini authentication failures."""

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
    def _is_quota_error(
        exc: Exception,
    ) -> bool:
        """Detect Gemini quota/rate-limit failures."""

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
    def _is_transient_error(
        exc: Exception,
    ) -> bool:
        """Determine whether a Gemini error is likely temporary."""

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

    # ========================================================================
    # Structured response helpers
    # ========================================================================

    @staticmethod
    def _extract_response_text(
        response: object,
    ) -> str:
        """Safely extract textual content from a Gemini response."""

        try:
            text = getattr(
                response,
                "text",
                None,
            )

            if text:
                return str(text).strip()

        except Exception:
            pass

        try:
            candidates = getattr(
                response,
                "candidates",
                None,
            ) or []

            parts: list[str] = []

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None,
                )

                if content is None:
                    continue

                candidate_parts = getattr(
                    content,
                    "parts",
                    None,
                ) or []

                for part in candidate_parts:

                    part_text = getattr(
                        part,
                        "text",
                        None,
                    )

                    if part_text:
                        parts.append(
                            str(part_text)
                        )

            return "\n".join(parts).strip()

        except Exception:
            return ""

    @staticmethod
    def _strip_json_fences(
        text: str,
    ) -> str:
        """Remove Markdown JSON fences if Gemini adds them."""

        cleaned = text.strip()

        if cleaned.startswith("```"):

            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    @classmethod
    def _parse_structured_text(
        cls,
        text: str,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        """Parse JSON text and validate it with Pydantic."""

        if not text:
            raise GeminiInvalidResponseError(
                "AI service returned no usable structured content."
            )

        cleaned = cls._strip_json_fences(text)

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError(
                "AI service returned invalid structured JSON."
            ) from exc

        if not isinstance(data, dict):
            raise GeminiInvalidResponseError(
                "AI service returned structured JSON in an unexpected format."
            )

        try:
            return schema.model_validate(data)

        except ValidationError as exc:

            logger.warning(
                "[Gemini] Structured response failed Pydantic validation: %s",
                exc,
            )

            raise GeminiInvalidResponseError(
                "AI service returned structured data that does not "
                "match the required schema."
            ) from exc

    # ========================================================================
    # Structured generation
    # ========================================================================

    def _generate_structured(
        self,
        prompt: str,
        schema: type[StructuredModel],
        api_key: str,
    ) -> StructuredModel:
        """
        Generate structured JSON content.

        Preferred:

            response.parsed

        Fallback:

            response.text -> JSON -> Pydantic
        """

        client = self._get_client(api_key)

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

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                # ------------------------------------------------------------
                # Preferred parsed response
                # ------------------------------------------------------------

                parsed = getattr(
                    response,
                    "parsed",
                    None,
                )

                if parsed is not None:

                    if isinstance(parsed, schema):
                        return parsed

                    try:
                        return schema.model_validate(
                            parsed
                        )

                    except ValidationError as exc:

                        raise GeminiInvalidResponseError(
                            "AI service returned structured data "
                            "that does not match the required schema."
                        ) from exc

                # ------------------------------------------------------------
                # Fallback text parsing
                # ------------------------------------------------------------

                response_text = self._extract_response_text(
                    response
                )

                if response_text:

                    return self._parse_structured_text(
                        response_text,
                        schema,
                    )

                raise GeminiInvalidResponseError(
                    "AI service returned no usable structured content."
                )

            except Exception as exc:

                last_exception = exc

                logger.warning(
                    "[Gemini] Structured request failed "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    type(exc).__name__,
                )

                if self._is_authentication_error(exc):

                    raise GeminiAuthenticationError(
                        "Gemini API authentication failed. "
                        "Please verify the supplied Gemini API key."
                    ) from exc

                if self._is_quota_error(exc):

                    if attempt == MAX_RETRIES - 1:

                        raise GeminiQuotaError(
                            "Gemini API rate limit or quota exceeded. "
                            "Please wait and try again."
                        ) from exc

                if isinstance(
                    exc,
                    GeminiInvalidResponseError,
                ):
                    raise exc

                if not self._is_transient_error(exc):

                    raise RuntimeError(
                        "AI service request failed during structured "
                        f"generation: {exc}"
                    ) from exc

                if attempt == MAX_RETRIES - 1:
                    break

                delay = INITIAL_RETRY_DELAY * (
                    2 ** attempt
                )

                logger.info(
                    "[Gemini] Retrying structured request in %.1fs...",
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            "AI service request failed during structured generation "
            f"after {MAX_RETRIES} attempts."
        ) from last_exception

    # ========================================================================
    # Text generation
    # ========================================================================

    def _generate_text(
        self,
        prompt: str,
        api_key: str,
    ) -> str:
        """Generate a plain-text Gemini response."""

        client = self._get_client(api_key)

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

                response_text = self._extract_response_text(
                    response
                )

                if not response_text:
                    raise RuntimeError(
                        "AI service returned an empty response."
                    )

                return response_text

            except Exception as exc:

                last_exception = exc

                logger.warning(
                    "[Gemini] Text request failed "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    type(exc).__name__,
                )

                if self._is_authentication_error(exc):

                    raise GeminiAuthenticationError(
                        "Gemini API authentication failed. "
                        "Please verify the supplied Gemini API key."
                    ) from exc

                if self._is_quota_error(exc):

                    if attempt == MAX_RETRIES - 1:

                        raise GeminiQuotaError(
                            "Gemini API rate limit or quota exceeded. "
                            "Please wait and try again."
                        ) from exc

                if not self._is_transient_error(exc):

                    raise RuntimeError(
                        "AI service request failed during text "
                        f"generation: {exc}"
                    ) from exc

                if attempt == MAX_RETRIES - 1:
                    break

                delay = INITIAL_RETRY_DELAY * (
                    2 ** attempt
                )

                logger.info(
                    "[Gemini] Retrying text request in %.1fs...",
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            "AI service request failed during text generation "
            f"after {MAX_RETRIES} attempts."
        ) from last_exception


# ============================================================================
# Module-level singleton
# ============================================================================

gemini_service = GeminiService()