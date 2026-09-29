"""
BodhaQ Deep Automated Test Suite

Run from:

    E:\\projects\\BodhaQ\\backend

Recommended:

    venv\\Scripts\\activate
    python tests\\deep_test.py

Optional environment variables:

    BODHAQ_BASE_URL=http://127.0.0.1:8000

    BODHAQ_TEST_DOCUMENT=E:\\path\\to\\test.pdf

    BODHAQ_TEST_RESUME=E:\\path\\to\\resume.pdf
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = os.getenv(
    "BODHAQ_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

TIMEOUT = 30
AI_TIMEOUT = 120

TEST_DOCUMENT = os.getenv(
    "BODHAQ_TEST_DOCUMENT"
)

TEST_RESUME = os.getenv(
    "BODHAQ_TEST_RESUME"
)


# ============================================================================
# TEST RUNNER
# ============================================================================


class TestRunner:

    def __init__(self) -> None:

        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.warnings = 0

        self.start = time.perf_counter()

    def section(
        self,
        title: str,
    ) -> None:

        print(
            "\n"
            + "=" * 72
        )

        print(title)

        print(
            "=" * 72
        )

    def passed_test(
        self,
        name: str,
        detail: str = "",
    ) -> None:

        self.passed += 1

        print(
            f"[PASS] {name}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

    def failed_test(
        self,
        name: str,
        detail: str = "",
    ) -> None:

        self.failed += 1

        print(
            f"[FAIL] {name}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

    def skipped_test(
        self,
        name: str,
        detail: str = "",
    ) -> None:

        self.skipped += 1

        print(
            f"[SKIP] {name}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

    def warning(
        self,
        name: str,
        detail: str = "",
    ) -> None:

        self.warnings += 1

        print(
            f"[WARN] {name}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

    def check(
        self,
        condition: bool,
        name: str,
        detail: str = "",
    ) -> None:

        if condition:

            self.passed_test(
                name,
                detail,
            )

        else:

            self.failed_test(
                name,
                detail,
            )

    def summary(self) -> int:

        elapsed = (
            time.perf_counter()
            - self.start
        )

        self.section(
            "FINAL SUMMARY"
        )

        print(
            f"Passed  : {self.passed}"
        )

        print(
            f"Failed  : {self.failed}"
        )

        print(
            f"Skipped : {self.skipped}"
        )

        print(
            f"Warnings: {self.warnings}"
        )

        print(
            f"Time    : {elapsed:.2f}s"
        )

        print()

        if self.failed:

            print(
                "RESULT: FAILED"
            )

            return 1

        print(
            "RESULT: PASSED"
        )

        return 0


runner = TestRunner()


# ============================================================================
# HTTP HELPERS
# ============================================================================


def get(
    path: str,
    **kwargs: Any,
) -> requests.Response:

    return requests.get(
        BASE_URL + path,
        timeout=kwargs.pop(
            "timeout",
            TIMEOUT,
        ),
        **kwargs,
    )


def post(
    path: str,
    **kwargs: Any,
) -> requests.Response:

    return requests.post(
        BASE_URL + path,
        timeout=kwargs.pop(
            "timeout",
            TIMEOUT,
        ),
        **kwargs,
    )


def delete(
    path: str,
    **kwargs: Any,
) -> requests.Response:

    return requests.delete(
        BASE_URL + path,
        timeout=kwargs.pop(
            "timeout",
            TIMEOUT,
        ),
        **kwargs,
    )


def json_response(
    response: requests.Response,
) -> Any:

    try:

        return response.json()

    except Exception:

        return None


def response_detail(
    response: requests.Response,
) -> str:

    data = json_response(
        response
    )

    if isinstance(
        data,
        dict,
    ):

        detail = (
            data.get("detail")
            or data.get("message")
            or data.get("error")
        )

        if isinstance(
            detail,
            dict,
        ):

            return str(
                detail.get("message")
                or detail.get("error")
                or detail
            )

        if detail:

            return str(
                detail
            )

    return (
        response.text[:500]
        .replace(
            "\n",
            " ",
        )
    )


def assert_no_traceback(
    response: requests.Response,
    name: str,
) -> None:

    body = response.text.lower()

    runner.check(
        "traceback" not in body
        and 'file "' not in body,
        name,
        "No traceback/internal file path exposed",
    )


# ============================================================================
# 1. SERVER / OPENAPI
# ============================================================================


def test_server() -> dict | None:

    runner.section(
        "1. SERVER / OPENAPI"
    )

    try:

        response = get(
            "/openapi.json"
        )

    except Exception as exc:

        runner.failed_test(
            "Backend reachable",
            str(exc),
        )

        return None

    runner.check(
        response.status_code == 200,
        "OpenAPI endpoint",
        f"HTTP {response.status_code}",
    )

    if response.status_code != 200:

        return None

    spec = json_response(
        response
    )

    if not isinstance(
        spec,
        dict,
    ):

        runner.failed_test(
            "OpenAPI JSON",
            "Invalid JSON",
        )

        return None

    runner.passed_test(
        "OpenAPI JSON parsed"
    )

    routes = [
        ("GET", "/"),
        ("GET", "/health"),

        ("POST", "/api/learning/topic"),

        ("POST", "/api/documents/upload"),
        ("GET", "/api/documents"),
        ("DELETE", "/api/documents/{document_id}"),

        ("POST", "/api/quiz/generate"),
        ("POST", "/api/quiz/submit"),
        ("GET", "/api/quiz/gaps"),
        ("POST", "/api/quiz/practice"),

        ("POST", "/api/doubts"),

        ("POST", "/api/settings/test-ai"),

        ("POST", "/api/resume/upload"),
        ("GET", "/api/resume/progress"),
        ("POST", "/api/resume/quiz/generate"),
        ("POST", "/api/resume/quiz/submit"),

        ("POST", "/api/coding/generate-problem"),
        ("POST", "/api/coding/analyze"),
        ("POST", "/api/coding/improve"),
        ("POST", "/api/coding/testcases"),
        ("POST", "/api/coding/submit"),
    ]

    paths = spec.get(
        "paths",
        {},
    )

    for method, path in routes:

        exists = (
            path in paths
            and method.lower()
            in paths[path]
        )

        runner.check(
            exists,
            f"Route registered: {method} {path}",
        )

    return spec


# ============================================================================
# 2. HEALTH
# ============================================================================


def test_health() -> None:

    runner.section(
        "2. HEALTH"
    )

    try:

        response = get(
            "/health"
        )

    except Exception as exc:

        runner.failed_test(
            "Health request",
            str(exc),
        )

        return

    runner.check(
        response.status_code == 200,
        "Health status",
        f"HTTP {response.status_code}",
    )

    data = json_response(
        response
    )

    runner.check(
        isinstance(
            data,
            dict,
        ),
        "Health response JSON",
    )

    if isinstance(
        data,
        dict,
    ):

        runner.check(
            data.get("status") == "ok",
            "Health status is ok",
        )


# ============================================================================
# 3. LEARNING / GEMINI
# ============================================================================


def test_learning() -> None:

    runner.section(
        "3. LEARNING / GEMINI"
    )

    payload = {
        "topic": "HashMap in Java"
    }

    try:

        response = post(
            "/api/learning/topic",
            json=payload,
            timeout=AI_TIMEOUT,
        )

    except Exception as exc:

        runner.failed_test(
            "Learning request",
            str(exc),
        )

        return

    if response.status_code in {
        429,
        503,
    }:

        runner.warning(
            "Learning generation",
            f"AI unavailable: HTTP {response.status_code}",
        )

        return

    runner.check(
        response.status_code == 200,
        "Learning generation",
        (
            f"HTTP {response.status_code}: "
            f"{response_detail(response)}"
        ),
    )

    if response.status_code != 200:

        assert_no_traceback(
            response,
            "Learning error has no traceback",
        )

        return

    data = json_response(
        response
    )

    if not isinstance(
        data,
        dict,
    ):

        runner.failed_test(
            "Learning response structure",
            "Response is not an object",
        )

        return

    expected = [
        "topic",
        "definition",
        "key_concepts",
        "example",
        "important_points",
        "resources",
    ]

    for key in expected:

        runner.check(
            key in data,
            f"Learning response contains {key}",
        )

    runner.check(
        isinstance(
            data.get("key_concepts"),
            list,
        ),
        "Learning key_concepts is a list",
    )

    runner.check(
        isinstance(
            data.get("important_points"),
            list,
        ),
        "Learning important_points is a list",
    )

    runner.check(
        isinstance(
            data.get("example"),
            dict,
        ),
        "Learning example is an object",
    )

    runner.check(
        isinstance(
            data.get("resources"),
            list,
        ),
        "Learning resources is a list",
    )


# ============================================================================
# 4. SETTINGS / AI CONNECTION
# ============================================================================


def test_settings() -> None:

    runner.section(
        "4. SETTINGS / AI CONFIGURATION"
    )

    try:

        response = post(
            "/api/settings/test-ai",
            json={},
            timeout=AI_TIMEOUT,
        )

    except Exception as exc:

        runner.failed_test(
            "AI settings test request",
            str(exc),
        )

        return

    if response.status_code in {
        429,
        503,
    }:

        runner.warning(
            "Backend Gemini connection",
            f"AI unavailable: HTTP {response.status_code}",
        )

        return

    runner.check(
        response.status_code == 200,
        "AI settings endpoint",
        (
            f"HTTP {response.status_code}: "
            f"{response_detail(response)}"
        ),
    )

    data = json_response(
        response
    )

    if isinstance(
        data,
        dict,
    ):

        runner.check(
            "valid" in data,
            "AI settings response contains valid",
        )

        runner.check(
            "message" in data,
            "AI settings response contains message",
        )


# ============================================================================
# 5. QUIZ
# ============================================================================


def test_quiz() -> None:

    runner.section(
        "5. QUIZ ENGINE"
    )

    payload = {
        "source_type": "topic",
        "source_id": "Java HashMap",
        "number_of_questions": 5,
        "difficulty": "medium",
    }

    try:

        response = post(
            "/api/quiz/generate",
            json=payload,
            timeout=AI_TIMEOUT,
        )

    except Exception as exc:

        runner.failed_test(
            "Quiz generation request",
            str(exc),
        )

        return

    if response.status_code in {
        429,
        503,
    }:

        runner.warning(
            "Quiz generation",
            f"AI unavailable: HTTP {response.status_code}",
        )

        return

    runner.check(
        response.status_code == 200,
        "Quiz generation",
        (
            f"HTTP {response.status_code}: "
            f"{response_detail(response)}"
        ),
    )

    if response.status_code != 200:

        return

    data = json_response(
        response
    )

    if not isinstance(
        data,
        dict,
    ):

        return

    quiz_id = data.get(
        "quiz_id"
    )

    questions = data.get(
        "questions"
    )

    runner.check(
        bool(quiz_id),
        "Quiz contains quiz_id",
    )

    runner.check(
        isinstance(
            questions,
            list,
        )
        and len(questions) == 5,
        "Quiz contains exactly 5 questions",
    )

    if not isinstance(
        questions,
        list,
    ):

        return

    forbidden = {
        "answer",
        "correct_answer",
        "correctAnswer",
        "correct_values",
        "correctValues",
        "solution",
    }

    leaked: list[str] = []

    for index, question in enumerate(
        questions
    ):

        if not isinstance(
            question,
            dict,
        ):

            runner.failed_test(
                f"Question {index + 1} structure",
            )

            continue

        for key in forbidden:

            if key in question:

                leaked.append(
                    f"q{index + 1}.{key}"
                )

        runner.check(
            "question" in question,
            f"Question {index + 1} has text",
        )

        runner.check(
            isinstance(
                question.get(
                    "options"
                ),
                list,
            ),
            f"Question {index + 1} has options",
        )

        runner.check(
            "topic" in question,
            f"Question {index + 1} has topic",
        )

    runner.check(
        not leaked,
        "Quiz response does not expose answers",
        str(leaked),
    )

    # Invalid quiz submission.

    try:

        response = post(
            "/api/quiz/submit",
            json={
                "quiz_id": "invalid-quiz-id",
                "answers": {},
            },
        )

        runner.check(
            response.status_code in {
                400,
                404,
            },
            "Invalid quiz submission rejected",
            f"HTTP {response.status_code}",
        )

        assert_no_traceback(
            response,
            "Invalid quiz submission has no traceback",
        )

    except Exception as exc:

        runner.failed_test(
            "Invalid quiz submission",
            str(exc),
        )


# ============================================================================
# 6. CODING — AI LEARN CODE
# ============================================================================


def test_coding_generation() -> dict | None:

    runner.section(
        "6. CODING — AI LEARN CODE"
    )

    # Current CodingProblemGenerateRequest contract.
    payload = {
        "title": "Two Sum",
        "statement": "",
        "constraints": "",
        "sample": "",
    }

    try:

        response = post(
            "/api/coding/generate-problem",
            json=payload,
            timeout=AI_TIMEOUT,
        )

    except Exception as exc:

        runner.failed_test(
            "Coding generation request",
            str(exc),
        )

        return None

    if response.status_code in {
        429,
        503,
    }:

        runner.warning(
            "Coding generation",
            f"AI unavailable: HTTP {response.status_code}",
        )

        return None

    runner.check(
        response.status_code == 200,
        "AI Learn Code generation",
        (
            f"HTTP {response.status_code}: "
            f"{response_detail(response)}"
        ),
    )

    if response.status_code != 200:

        return None

    data = json_response(
        response
    )

    if not isinstance(
        data,
        dict,
    ):

        runner.failed_test(
            "Coding generation response",
            "Invalid JSON object",
        )

        return None

    required = [
        "problem_id",
        "title",
        "statement",
        "input_format",
        "output_format",
        "constraints",
        "examples",
        "difficulty",
        "topics",
        "starter_code_java",
        "starter_code_python",
        "public_tests",
        "hidden_test_count",
    ]

    for key in required:

        runner.check(
            key in data,
            f"Coding response contains {key}",
        )

    runner.check(
        isinstance(
            data.get("problem_id"),
            str,
        )
        and bool(
            data.get("problem_id")
        ),
        "Coding problem_id is valid",
    )

    runner.check(
        isinstance(
            data.get("examples"),
            list,
        ),
        "Coding examples is a list",
    )

    runner.check(
        isinstance(
            data.get("public_tests"),
            list,
        ),
        "Coding public_tests is a list",
    )

    runner.check(
        isinstance(
            data.get("hidden_test_count"),
            int,
        )
        and data.get(
            "hidden_test_count"
        ) >= 0,
        "Coding hidden_test_count is valid",
    )

    public_tests = data.get(
        "public_tests"
    )

    runner.check(
        isinstance(
            public_tests,
            list,
        )
        and len(public_tests) > 0,
        "Public tests exist",
    )

    # ------------------------------------------------------------------------
    # Hidden-test privacy
    # ------------------------------------------------------------------------

    forbidden = {
        "hidden_tests",
        "hiddenTests",
        "private_tests",
        "privateTests",
    }

    leaked = [
        key
        for key in forbidden
        if key in data
    ]

    runner.check(
        not leaked,
        "Generation response hides hidden tests",
        str(leaked),
    )

    raw = json.dumps(
        data
    ).lower()

    leak_tokens = [
        '"hidden_tests"',
        '"hiddentests"',
        '"private_tests"',
        '"privatetests"',
    ]

    leaks = [
        token
        for token in leak_tokens
        if token in raw
    ]

    runner.check(
        not leaks,
        "No hidden-test fields leak recursively",
        str(leaks),
    )

    return data


# ============================================================================
# 7. CODING — AI ACTIONS
# ============================================================================


def test_coding_ai(
    problem: dict | None,
) -> None:

    runner.section(
        "7. CODING — AI ANALYZE / IMPROVE / TEST CASES"
    )

    if not problem:

        runner.skipped_test(
            "Coding AI actions",
            "No generated problem",
        )

        return

    problem_id = problem.get(
        "problem_id"
    )

    statement = str(
        problem.get(
            "statement",
            "",
        )
    ).strip()

    constraints = str(
        problem.get(
            "constraints",
            "",
        )
    ).strip()

    examples = problem.get(
        "examples"
    )

    sample_test_case = ""

    if (
        isinstance(
            examples,
            list,
        )
        and examples
        and isinstance(
            examples[0],
            dict,
        )
    ):

        example = examples[0]

        sample_input = str(
            example.get(
                "input",
                "",
            )
        )

        sample_output = str(
            example.get(
                "output",
                "",
            )
        )

        sample_test_case = (
            f"Input:\n"
            f"{sample_input}\n\n"
            f"Output:\n"
            f"{sample_output}"
        ).strip()

    if not sample_test_case:

        runner.warning(
            "Coding AI actions",
            "Generated problem contains no usable visible example",
        )

        return

    java_code = str(
        problem.get(
            "starter_code_java",
            "",
        )
    )

    base = {
        "problem_statement": statement,
        "sample_test_case": sample_test_case,
        "constraints": constraints,
        "code": java_code,
        "language": "java",
    }

    # ------------------------------------------------------------------------
    # ANALYZE
    # ------------------------------------------------------------------------

    try:

        response = post(
            "/api/coding/analyze",
            json=base,
            timeout=AI_TIMEOUT,
        )

        if response.status_code in {
            429,
            503,
        }:

            runner.warning(
                "AI Analyze",
                f"AI unavailable: HTTP {response.status_code}",
            )

        else:

            runner.check(
                response.status_code == 200,
                "AI Analyze",
                response_detail(response),
            )

            if response.status_code != 200:

                assert_no_traceback(
                    response,
                    "AI Analyze error has no traceback",
                )

    except Exception as exc:

        runner.failed_test(
            "AI Analyze request",
            str(exc),
        )

    # ------------------------------------------------------------------------
    # IMPROVE
    # ------------------------------------------------------------------------

    try:

        response = post(
            "/api/coding/improve",
            json=base,
            timeout=AI_TIMEOUT,
        )

        if response.status_code in {
            429,
            503,
        }:

            runner.warning(
                "AI Improve",
                f"AI unavailable: HTTP {response.status_code}",
            )

        else:

            runner.check(
                response.status_code == 200,
                "AI Improve",
                response_detail(response),
            )

            if response.status_code != 200:

                assert_no_traceback(
                    response,
                    "AI Improve error has no traceback",
                )

    except Exception as exc:

        runner.failed_test(
            "AI Improve request",
            str(exc),
        )

    # ------------------------------------------------------------------------
    # CREATE TEST CASES
    # ------------------------------------------------------------------------

    testcase_payload = {
        "problem_id": problem_id,
        **base,
    }

    try:

        response = post(
            "/api/coding/testcases",
            json=testcase_payload,
            timeout=AI_TIMEOUT,
        )

        if response.status_code in {
            429,
            503,
        }:

            runner.warning(
                "Create Test Cases",
                f"AI unavailable: HTTP {response.status_code}",
            )

        else:

            runner.check(
                response.status_code == 200,
                "Create Test Cases",
                response_detail(response),
            )

            data = json_response(
                response
            )

            if isinstance(
                data,
                dict,
            ):

                runner.check(
                    isinstance(
                        data.get(
                            "public_tests"
                        ),
                        list,
                    ),
                    "Create Test Cases returns public tests",
                )

                runner.check(
                    "hidden_tests"
                    not in data
                    and "private_tests"
                    not in data,
                    "Create Test Cases hides private tests",
                )

                raw = json.dumps(
                    data
                ).lower()

                runner.check(
                    '"hidden_tests"'
                    not in raw
                    and '"hiddentests"'
                    not in raw
                    and '"private_tests"'
                    not in raw
                    and '"privatetests"'
                    not in raw,
                    "Create Test Cases has no recursive hidden-test leak",
                )

    except Exception as exc:

        runner.failed_test(
            "Create Test Cases request",
            str(exc),
        )


# ============================================================================
# 8. CODING — SUBMIT / PRIVACY
# ============================================================================


def test_coding_submit(
    problem: dict | None,
) -> None:

    runner.section(
        "8. CODING — SUBMIT / JUDGE"
    )

    if not problem:

        runner.skipped_test(
            "Coding submit",
            "No generated problem",
        )

        return

    problem_id = problem.get(
        "problem_id"
    )

    # ------------------------------------------------------------------------
    # Invalid problem
    # ------------------------------------------------------------------------

    invalid_payload = {
        "problem_id":
            "00000000-0000-0000-0000-000000000000",

        "language": "java",

        "code": (
            "public class Main { "
            "public static void main(String[] args) {} "
            "}"
        ),

        "test_suite": "all",
    }

    try:

        response = post(
            "/api/coding/submit",
            json=invalid_payload,
        )

        runner.check(
            response.status_code in {
                400,
                404,
            },
            "Invalid coding problem rejected",
            f"HTTP {response.status_code}",
        )

        assert_no_traceback(
            response,
            "Invalid coding submit has no traceback",
        )

    except Exception as exc:

        runner.failed_test(
            "Invalid coding submit",
            str(exc),
        )

    # ------------------------------------------------------------------------
    # Hidden privacy
    # ------------------------------------------------------------------------

    java_code = str(
        problem.get(
            "starter_code_java",
            "",
        )
    )

    payload = {
        "problem_id": problem_id,
        "language": "java",
        "code": java_code,
        "test_suite": "all",
    }

    try:

        response = post(
            "/api/coding/submit",
            json=payload,
            timeout=AI_TIMEOUT,
        )

        raw = response.text.lower()

        forbidden = [
            '"hidden_tests"',
            '"hiddentests"',
            '"private_tests"',
            '"privatetests"',
            "hidden_test_input",
            "hidden_test_output",
        ]

        leaks = [
            value
            for value in forbidden
            if value in raw
        ]

        runner.check(
            not leaks,
            "Submit response hides hidden test data",
            str(leaks),
        )

        data = json_response(
            response
        )

        if isinstance(
            data,
            dict,
        ):

            if data.get(
                "hidden_test_failed"
            ) is True:

                for key in [
                    "input",
                    "expected",
                    "actual",
                    "expected_output",
                    "actual_output",
                ]:

                    runner.check(
                        key not in data,
                        (
                            "Hidden failure does not expose "
                            f"{key}"
                        ),
                    )

    except Exception as exc:

        runner.failed_test(
            "Coding privacy test",
            str(exc),
        )


# ============================================================================
# 9. DOCKER
# ============================================================================


def run_command(
    command: list[str],
    name: str,
    timeout: int = 30,
) -> bool:

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    except FileNotFoundError:

        runner.skipped_test(
            name,
            f"Command not found: {command[0]}",
        )

        return False

    except subprocess.TimeoutExpired:

        runner.failed_test(
            name,
            "Timed out",
        )

        return False

    if result.returncode == 0:

        runner.passed_test(
            name,
            result.stdout.strip()[:200],
        )

        return True

    runner.failed_test(
        name,
        (
            result.stderr
            or result.stdout
        ).strip()[:500],
    )

    return False


def test_docker() -> None:

    runner.section(
        "9. DOCKER / EXECUTION"
    )

    run_command(
        [
            "docker",
            "--version",
        ],
        "Docker client available",
    )

    daemon = run_command(
        [
            "docker",
            "info",
        ],
        "Docker daemon available",
        timeout=30,
    )

    if not daemon:

        runner.warning(
            "Actual code execution",
            "Docker daemon unavailable",
        )

        return

    run_command(
        [
            "docker",
            "run",
            "--rm",
            "alpine",
            "echo",
            "BodhaQ Docker test",
        ],
        "Docker container execution",
        timeout=60,
    )

    # ------------------------------------------------------------------------
    # Network isolation
    # ------------------------------------------------------------------------

    try:

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "alpine",
                "sh",
                "-c",
                (
                    "wget -q -T 3 -O - "
                    "https://example.com"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        runner.check(
            "Example Domain"
            not in result.stdout,
            "Docker network isolation sanity check",
        )

    except subprocess.TimeoutExpired:

        runner.failed_test(
            "Docker network isolation",
            "Timed out",
        )

    except Exception as exc:

        runner.failed_test(
            "Docker network isolation",
            str(exc),
        )


# ============================================================================
# 10. RESUME
# ============================================================================


def test_resume() -> None:

    runner.section(
        "10. RESUME PREP"
    )

    try:

        response = get(
            "/api/resume/progress"
        )

    except Exception as exc:

        runner.failed_test(
            "Resume progress",
            str(exc),
        )

        return

    runner.check(
        response.status_code == 200,
        "Resume progress endpoint",
        f"HTTP {response.status_code}",
    )

    if response.status_code != 200:

        return

    data = json_response(
        response
    )

    runner.check(
        isinstance(
            data,
            dict,
        ),
        "Resume progress response JSON",
    )

    if not TEST_RESUME:

        runner.skipped_test(
            "Resume upload",
            "Set BODHAQ_TEST_RESUME",
        )

        return

    path = Path(
        TEST_RESUME
    )

    if not path.exists():

        runner.failed_test(
            "Resume fixture exists",
            str(path),
        )

        return

    mime_types = {
        ".pdf":
            "application/pdf",

        ".docx":
            (
                "application/"
                "vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
    }

    mime = mime_types.get(
        path.suffix.lower()
    )

    if mime is None:

        runner.failed_test(
            "Resume fixture format",
            "Expected PDF or DOCX",
        )

        return

    try:

        with path.open(
            "rb"
        ) as file:

            response = post(
                "/api/resume/upload",
                files={
                    "file": (
                        path.name,
                        file,
                        mime,
                    )
                },
                timeout=AI_TIMEOUT,
            )

        if response.status_code in {
            429,
            503,
        }:

            runner.warning(
                "Resume extraction",
                (
                    f"AI unavailable: "
                    f"HTTP {response.status_code}"
                ),
            )

        else:

            runner.check(
                response.status_code == 200,
                "Resume upload/extraction",
                response_detail(response),
            )

            if response.status_code != 200:

                assert_no_traceback(
                    response,
                    "Resume error has no traceback",
                )

    except Exception as exc:

        runner.failed_test(
            "Resume upload",
            str(exc),
        )


# ============================================================================
# 11. DOCUMENT / RAG
# ============================================================================


def test_rag() -> None:

    runner.section(
        "11. DOCUMENT / RAG"
    )

    if not TEST_DOCUMENT:

        runner.skipped_test(
            "Document ingestion",
            "Set BODHAQ_TEST_DOCUMENT",
        )

        return

    path = Path(
        TEST_DOCUMENT
    )

    if not path.exists():

        runner.failed_test(
            "Document fixture exists",
            str(path),
        )

        return

    mime_types = {
        ".pdf":
            "application/pdf",

        ".pptx":
            (
                "application/"
                "vnd.openxmlformats-officedocument."
                "presentationml.presentation"
            ),

        ".docx":
            (
                "application/"
                "vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
    }

    mime = mime_types.get(
        path.suffix.lower(),
        "application/octet-stream",
    )

    document_id: str | None = None

    # ------------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------------

    try:

        with path.open(
            "rb"
        ) as file:

            response = post(
                "/api/documents/upload",
                files={
                    "file": (
                        path.name,
                        file,
                        mime,
                    )
                },
                timeout=AI_TIMEOUT,
            )

        if response.status_code in {
            429,
            503,
        }:

            runner.warning(
                "Document ingestion",
                (
                    f"AI/embedding service unavailable: "
                    f"HTTP {response.status_code}"
                ),
            )

            return

        runner.check(
            response.status_code in {
                200,
                201,
            },
            "Document ingestion",
            response_detail(response),
        )

        if response.status_code not in {
            200,
            201,
        }:

            assert_no_traceback(
                response,
                "Document ingestion error has no traceback",
            )

            return

        data = json_response(
            response
        )

        if isinstance(
            data,
            dict,
        ):

            document_id = data.get(
                "document_id"
            )

            runner.check(
                bool(document_id),
                "Document upload returns document_id",
            )

            runner.check(
                isinstance(
                    data.get(
                        "chunk_count"
                    ),
                    int,
                )
                and data.get(
                    "chunk_count",
                    0,
                ) > 0,
                "Document upload created chunks",
            )

    except Exception as exc:

        runner.failed_test(
            "Document ingestion",
            str(exc),
        )

        return

    if not document_id:

        return

    # ------------------------------------------------------------------------
    # List documents
    # ------------------------------------------------------------------------

    try:

        response = get(
            "/api/documents"
        )

        runner.check(
            response.status_code == 200,
            "Document list endpoint",
            f"HTTP {response.status_code}",
        )

        data = json_response(
            response
        )

        if isinstance(
            data,
            dict,
        ):

            documents = data.get(
                "documents"
            )

            runner.check(
                isinstance(
                    documents,
                    list,
                ),
                "Document list contains documents",
            )

            if isinstance(
                documents,
                list,
            ):

                found = any(
                    isinstance(
                        item,
                        dict,
                    )
                    and item.get(
                        "document_id"
                    ) == document_id
                    for item in documents
                )

                runner.check(
                    found,
                    "Uploaded document appears in document list",
                )

    except Exception as exc:

        runner.failed_test(
            "Document list",
            str(exc),
        )

    # ------------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------------

    try:

        response = delete(
            f"/api/documents/{document_id}"
        )

        runner.check(
            response.status_code in {
                200,
                204,
            },
            "Document deletion",
            f"HTTP {response.status_code}",
        )

        if response.status_code not in {
            200,
            204,
        }:

            assert_no_traceback(
                response,
                "Document deletion error has no traceback",
            )

    except Exception as exc:

        runner.failed_test(
            "Document deletion",
            str(exc),
        )


# ============================================================================
# 12. SOURCE / SECURITY AUDIT
# ============================================================================


def test_source_security() -> None:

    runner.section(
        "12. SOURCE / SECURITY AUDIT"
    )

    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    execution_service = (
        root
        / "backend"
        / "app"
        / "services"
        / "code_execution_service.py"
    )

    if execution_service.exists():

        text = execution_service.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        runner.check(
            "eval(" not in text
            and "exec(" not in text,
            "No eval()/exec() in execution service",
        )

        runner.check(
            "docker" in text.lower(),
            "Execution service references Docker",
        )

    else:

        runner.warning(
            "Execution service audit",
            "File not found",
        )

    gitignore = (
        root
        / ".gitignore"
    )

    if gitignore.exists():

        text = gitignore.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        runner.check(
            ".env" in text,
            "Root .gitignore contains .env",
        )


# ============================================================================
# 13. NEGATIVE API TESTS
# ============================================================================


def test_negative_requests() -> None:

    runner.section(
        "13. NEGATIVE / ERROR HANDLING"
    )

    tests = [
        (
            "/api/learning/topic",
            {},
        ),

        (
            "/api/quiz/generate",
            {},
        ),

        (
            "/api/coding/generate-problem",
            {},
        ),

        (
            "/api/settings/test-ai",
            {
                "api_key": "this-field-should-not-be-used",
            },
        ),
    ]

    for path, payload in tests:

        try:

            response = post(
                path,
                json=payload,
            )

        except Exception as exc:

            runner.failed_test(
                f"Negative request {path}",
                str(exc),
            )

            continue

        # Settings intentionally accepts an empty body because the
        # backend owns the Gemini configuration. Extra fields are ignored
        # by default by Pydantic, so this test mainly verifies that the
        # endpoint remains reachable.
        if path == "/api/settings/test-ai":

            runner.check(
                response.status_code == 200,
                "Settings endpoint remains backend-managed",
                f"HTTP {response.status_code}",
            )

            continue

        runner.check(
            response.status_code
            in {
                400,
                422,
                500,
                503,
            },
            f"{path} rejects malformed input",
            f"HTTP {response.status_code}",
        )

        assert_no_traceback(
            response,
            f"{path} does not expose traceback",
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    print(
        "=" * 72
    )

    print(
        "BODHAQ DEEP AUTOMATED TEST SUITE"
    )

    print(
        "=" * 72
    )

    print(
        f"Base URL: {BASE_URL}"
    )

    spec = test_server()

    if spec is None:

        return runner.summary()

    test_health()

    test_learning()

    test_settings()

    test_quiz()

    coding_problem = (
        test_coding_generation()
    )

    test_coding_ai(
        coding_problem
    )

    test_coding_submit(
        coding_problem
    )

    test_docker()

    test_resume()

    test_rag()

    test_source_security()

    test_negative_requests()

    return runner.summary()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )