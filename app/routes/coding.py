"""
Coding routes.

Endpoints:

    POST /api/coding/execute
        Execute code in the isolated execution environment.

    POST /api/coding/generate-problem
        Generate and independently validate an AI coding problem.

    POST /api/coding/analyze
        Analyze the user's code.

    POST /api/coding/improve
        Suggest improvements for the user's code.

    POST /api/coding/testcases
        Generate additional public and hidden test cases and independently
        verify their expected outputs.

    POST /api/coding/submit
        Submit code against samples, public tests, or all tests.

Important security rules:

    - Coding problems are isolated by BodhaQ session.
    - Hidden test inputs/outputs never reach the frontend.
    - Gemini-provided expected outputs are NOT trusted.
    - Backend reference solutions calculate authoritative outputs.
    - Java and Python reference solutions must agree.
    - A generated problem is stored only after verification succeeds.
    - Gemini API keys are request-scoped and are never logged or persisted.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    status,
)

from app.dependencies.session import get_session_id
from app.models.requests import (
    CodeAnalyzeRequest,
    CodeExecutionRequest,
    CodeImproveRequest,
    CodeSubmitRequest,
    CodingProblemGenerateRequest,
    TestCaseGenerateRequest,
)
from app.models.responses import (
    CodeAnalyzeResponse,
    CodeImproveResponse,
    CodeSubmitResponse,
    CodeExecutionResponse,
    CodingProblemExample,
    CodingProblemGenerateResponse,
    CodingTestCase,
    TestCaseGenerateResponse,
)
from app.services.code_execution_service import code_execution_service
from app.services.gemini_service import gemini_service
from app.services.problem_store import problem_store


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_GENERATION_RETRIES = 3

MAX_GENERATION_ERROR_LENGTH = 500

MAX_PUBLIC_ERROR_LENGTH = 5000


# ============================================================================
# HELPERS
# ============================================================================


def _get_gemini_api_key(
    api_key: str | None,
) -> str:
    """
    Validate the request-scoped Gemini API key.

    The key is intentionally never logged, persisted, or included
    in an error response.
    """

    if not isinstance(
        api_key,
        str,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Gemini API key is required.",
                "code": "GEMINI_API_KEY_REQUIRED",
            },
        )

    normalized = api_key.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Gemini API key is required.",
                "code": "GEMINI_API_KEY_REQUIRED",
            },
        )

    return normalized


def _normalize_output(
    value: str | None,
) -> str:
    """
    Normalize program output for deterministic comparison.

    We intentionally ignore:
        - leading/trailing whitespace
        - Windows vs Unix line endings

    Internal whitespace and output ordering are not modified because the
    problem's output contract must define the expected representation.
    """

    if value is None:
        return ""

    return (
        str(value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _safe_public_text(
    value: str | None,
    max_length: int = MAX_PUBLIC_ERROR_LENGTH,
) -> str:
    """
    Safely bound execution output before returning it to the frontend.
    """

    if not isinstance(
        value,
        str,
    ):
        return ""

    if len(value) <= max_length:
        return value

    return (
        value[:max_length].rstrip()
        + "\n[Output truncated]"
    )


def _validate_generated_test_input(
    test_input: Any,
    test_type: str,
) -> str | None:
    """
    Validate generated test input.

    Returns:
        None when valid.
        Error message when invalid.
    """

    if not isinstance(
        test_input,
        str,
    ):
        return (
            f"{test_type} has invalid input."
        )

    if not test_input.strip():
        return (
            f"{test_type} has empty input."
        )

    return None


def _build_problem_tests(
    result: Any,
) -> list[tuple[str, Any]]:
    """
    Build the complete internal test contract.

    Order:
        examples
        public tests
        hidden tests

    Hidden tests remain entirely backend-side.
    """

    tests: list[
        tuple[str, Any]
    ] = []

    for example in result.examples:
        tests.append(
            (
                "example",
                example,
            )
        )

    for test in result.public_tests:
        tests.append(
            (
                "public",
                test,
            )
        )

    for test in result.hidden_tests:
        tests.append(
            (
                "hidden",
                test,
            )
        )

    return tests


def _compute_reference_outputs(
    tests: list[tuple[str, Any]],
    java_code: str,
    python_code: str,
) -> tuple[
    bool,
    dict[int, str],
    str | None,
]:
    """
    Execute the generated Java and Python reference solutions.

    The backend computes the authoritative expected output.

    Returns:
        (
            is_valid,
            outputs_by_test_index,
            internal_error_message,
        )

    IMPORTANT:
        Hidden test inputs/outputs are never included in the returned
        error message.
    """

    if not isinstance(
        java_code,
        str,
    ) or not java_code.strip():
        return (
            False,
            {},
            "Generated Java reference solution is empty.",
        )

    if not isinstance(
        python_code,
        str,
    ) or not python_code.strip():
        return (
            False,
            {},
            "Generated Python reference solution is empty.",
        )

    outputs: dict[int, str] = {}

    for index, (
        test_type,
        test,
    ) in enumerate(tests):
        test_input = getattr(
            test,
            "input",
            None,
        )

        input_error = _validate_generated_test_input(
            test_input,
            test_type,
        )

        if input_error:
            return (
                False,
                {},
                input_error,
            )

        # --------------------------------------------------------------------
        # JAVA REFERENCE
        # --------------------------------------------------------------------

        java_request = CodeExecutionRequest(
            language="java",
            code=java_code,
            stdin=test_input,
        )

        java_result = (
            code_execution_service.execute_code(
                java_request
            )
        )

        if java_result.status != "success":
            return (
                False,
                {},
                (
                    "Generated Java reference solution "
                    f"failed on {test_type}."
                ),
            )

        java_output = _normalize_output(
            java_result.stdout
        )

        # --------------------------------------------------------------------
        # PYTHON REFERENCE
        # --------------------------------------------------------------------

        python_request = CodeExecutionRequest(
            language="python",
            code=python_code,
            stdin=test_input,
        )

        python_result = (
            code_execution_service.execute_code(
                python_request
            )
        )

        if python_result.status != "success":
            return (
                False,
                {},
                (
                    "Generated Python reference solution "
                    f"failed on {test_type}."
                ),
            )

        python_output = _normalize_output(
            python_result.stdout
        )

        # --------------------------------------------------------------------
        # CROSS-LANGUAGE CONSISTENCY
        # --------------------------------------------------------------------

        if java_output != python_output:
            return (
                False,
                {},
                (
                    "Generated Java and Python reference "
                    f"solutions disagree on {test_type}."
                ),
            )

        outputs[index] = java_output

    return (
        True,
        outputs,
        None,
    )


def _apply_reference_outputs(
    result: Any,
    outputs: dict[int, str],
) -> None:
    """
    Replace Gemini-provided expected outputs with backend-computed outputs.

    This function mutates the generated Pydantic objects before storage.
    """

    index = 0

    for example in result.examples:
        example.output = outputs[index]
        index += 1

    for test in result.public_tests:
        test.output = outputs[index]
        index += 1

    for test in result.hidden_tests:
        test.output = outputs[index]
        index += 1


def _verify_additional_tests(
    problem: Any,
    public_tests: list[Any],
    hidden_tests: list[Any],
) -> tuple[
    bool,
    str | None,
]:
    """
    Verify additional generated tests against the stored reference
    Java/Python solutions.

    Expected outputs are calculated by the backend.

    Returns:
        (
            is_valid,
            internal_error_message,
        )
    """

    java_code = getattr(
        problem,
        "starter_code_java",
        "",
    )

    python_code = getattr(
        problem,
        "starter_code_python",
        "",
    )

    tests: list[
        tuple[str, Any]
    ] = []

    for test in public_tests:
        tests.append(
            (
                "public test",
                test,
            )
        )

    for test in hidden_tests:
        tests.append(
            (
                "hidden test",
                test,
            )
        )

    if not tests:
        return (
            False,
            "No generated test cases were returned.",
        )

    is_valid, outputs, error = (
        _compute_reference_outputs(
            tests=tests,
            java_code=java_code,
            python_code=python_code,
        )
    )

    if not is_valid:
        return (
            False,
            error,
        )

    index = 0

    for test in public_tests:
        test.output = outputs[index]
        index += 1

    for test in hidden_tests:
        test.output = outputs[index]
        index += 1

    return (
        True,
        None,
    )


# ============================================================================
# CODE EXECUTION
# ============================================================================


@router.post(
    "/execute",
    response_model=CodeExecutionResponse,
)
async def execute_code(
    request: CodeExecutionRequest,
):
    """
    Execute code in the isolated execution environment.

    This endpoint does not require a Gemini API key or a persisted
    coding-problem session.
    """

    try:
        return code_execution_service.execute_code(
            request
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "[Code Execution] Execution failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Code execution failed.",
                "code": "CODE_EXECUTION_FAILED",
            },
        )


# ============================================================================
# AI LEARN CODE — GENERATE PROBLEM
# ============================================================================


@router.post(
    "/generate-problem",
    response_model=CodingProblemGenerateResponse,
)
async def generate_problem(
    request: CodingProblemGenerateRequest,
    session_id: str = Depends(get_session_id),
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Generate and independently validate a coding problem.

    Gemini API access is performed using the request-scoped
    X-Gemini-API-Key header.

    The generated problem is stored under the authenticated
    anonymous BodhaQ session.

    Quality gate:

        Gemini
            ↓
        Problem + reference solutions + test inputs
            ↓
        Structural validation
            ↓
        Java reference execution
            ↓
        Python reference execution
            ↓
        Java == Python?
            ↓
        Backend computes authoritative outputs
            ↓
        Save verified problem under session
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    title = (
        request.title.strip()
        if isinstance(
            request.title,
            str,
        )
        else ""
    )

    statement = (
        request.statement.strip()
        if isinstance(
            request.statement,
            str,
        )
        else ""
    )

    constraints = (
        request.constraints.strip()
        if isinstance(
            request.constraints,
            str,
        )
        else ""
    )

    sample = (
        request.sample.strip()
        if isinstance(
            request.sample,
            str,
        )
        else ""
    )

    if not title and not statement:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    "Provide either a problem title "
                    "or a problem statement."
                ),
                "code": "INVALID_CODING_PROBLEM_REQUEST",
            },
        )

    previous_errors: list[str] = []

    for attempt in range(
        1,
        MAX_GENERATION_RETRIES + 1,
    ):
        try:
            logger.info(
                "[Coding Generation] Attempt %s/%s",
                attempt,
                MAX_GENERATION_RETRIES,
            )

            result = (
                gemini_service.generate_coding_problem(
                    title=title,
                    statement=statement,
                    constraints=constraints,
                    sample=sample,
                    previous_errors=(
                        previous_errors[-6:]
                        if previous_errors
                        else None
                    ),
                    api_key=api_key,
                )
            )

            all_tests = _build_problem_tests(
                result
            )

            if not all_tests:
                error_msg = (
                    "Generated problem contains no executable tests."
                )

                logger.warning(
                    "[Coding Validation] %s",
                    error_msg,
                )

                previous_errors.append(
                    error_msg
                )

                continue

            invalid_test = None

            for test_type, test in all_tests:
                test_input = getattr(
                    test,
                    "input",
                    None,
                )

                invalid_test = (
                    _validate_generated_test_input(
                        test_input,
                        test_type,
                    )
                )

                if invalid_test:
                    break

            if invalid_test:
                logger.warning(
                    "[Coding Validation] %s",
                    invalid_test,
                )

                previous_errors.append(
                    invalid_test
                )

                continue

            is_valid, outputs, error = (
                _compute_reference_outputs(
                    tests=all_tests,
                    java_code=result.starter_code_java,
                    python_code=result.starter_code_python,
                )
            )

            if not is_valid:
                error_msg = (
                    error
                    or "Generated reference solutions failed validation."
                )

                logger.warning(
                    "[Coding Validation] %s",
                    error_msg,
                )

                previous_errors.append(
                    error_msg[:MAX_GENERATION_ERROR_LENGTH]
                )

                continue

            _apply_reference_outputs(
                result,
                outputs,
            )

            problem_id = (
                problem_store.save_problem(
                    session_id,
                    result,
                )
            )

            logger.info(
                "[Coding Generation] Problem verified successfully."
            )

            return CodingProblemGenerateResponse(
                problem_id=problem_id,
                title=result.title,
                statement=result.statement,
                input_format=result.input_format,
                output_format=result.output_format,
                constraints=result.constraints,
                examples=[
                    CodingProblemExample(
                        input=example.input,
                        output=example.output,
                        explanation=example.explanation,
                    )
                    for example in result.examples
                ],
                difficulty=result.difficulty,
                topics=result.topics,
                starter_code_java=result.starter_code_java,
                starter_code_python=result.starter_code_python,
                public_tests=[
                    CodingTestCase(
                        input=test.input,
                        output=test.output,
                    )
                    for test in result.public_tests
                ],
                hidden_test_count=len(
                    result.hidden_tests
                ),
            )

        except HTTPException:
            raise

        except ValueError as exc:
            logger.warning(
                "[Coding Generation] Attempt %s validation failed: %s",
                attempt,
                str(exc)[:MAX_GENERATION_ERROR_LENGTH],
            )

            previous_errors.append(
                str(exc)[:MAX_GENERATION_ERROR_LENGTH]
            )

        except Exception as exc:
            logger.exception(
                "[Coding Generation] Attempt %s failed.",
                attempt,
            )

            previous_errors.append(
                f"Generation error: {type(exc).__name__}"
            )

    logger.error(
        "[Coding Generation] Failed after %s attempts.",
        MAX_GENERATION_RETRIES,
    )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "error": (
                "AI could not generate a consistent coding "
                "problem right now. Please try again."
            ),
            "code": "CODING_PROBLEM_GENERATION_FAILED",
        },
    )


# ============================================================================
# AI ANALYZE
# ============================================================================


@router.post(
    "/analyze",
    response_model=CodeAnalyzeResponse,
)
async def analyze_code(
    request: CodeAnalyzeRequest,
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Analyze the user's code.

    This does not execute or modify the code.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        result = gemini_service.analyze_code(
            problem_statement=request.problem_statement,
            sample=request.sample_test_case,
            constraints=request.constraints,
            code=request.code,
            language=request.language,
            api_key=api_key,
        )

        return CodeAnalyzeResponse(
            explanation=result.explanation,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_CODE_ANALYSIS_REQUEST",
            },
        ) from exc

    except Exception:
        logger.exception(
            "[Coding Analyze] Failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Code analysis service is unavailable.",
                "code": "CODE_ANALYSIS_FAILED",
            },
        )


# ============================================================================
# AI IMPROVE
# ============================================================================


@router.post(
    "/improve",
    response_model=CodeImproveResponse,
)
async def improve_code(
    request: CodeImproveRequest,
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Analyze and improve the user's code.

    The frontend decides whether to replace the current editor contents.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        result = gemini_service.improve_code(
            problem_statement=request.problem_statement,
            sample=request.sample_test_case,
            constraints=request.constraints,
            code=request.code,
            language=request.language,
            api_key=api_key,
        )

        return CodeImproveResponse(
            explanation=result.explanation,
            current_complexity=result.current_complexity,
            possible_complexity=result.possible_complexity,
            optimized_code=result.optimized_code,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_CODE_IMPROVEMENT_REQUEST",
            },
        ) from exc

    except Exception:
        logger.exception(
            "[Coding Improve] Failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Code improvement service is unavailable.",
                "code": "CODE_IMPROVEMENT_FAILED",
            },
        )


# ============================================================================
# GENERATE TEST CASES
# ============================================================================


@router.post(
    "/testcases",
    response_model=TestCaseGenerateResponse,
)
async def generate_testcases(
    request: TestCaseGenerateRequest,
    session_id: str = Depends(get_session_id),
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
):
    """
    Generate additional tests for an existing coding problem.

    The stored problem is authoritative. Client-supplied problem
    statement/constraints are not trusted for the verification step.

    The problem must belong to the current BodhaQ session.

    Generated expected outputs are independently calculated using
    the stored Java/Python reference solutions.

    Only public tests and hidden-test count are returned.
    """

    api_key = _get_gemini_api_key(
        x_gemini_api_key
    )

    try:
        problem = problem_store.get_problem(
            session_id,
            request.problem_id,
        )

        if problem is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Problem not found or expired.",
                    "code": "CODING_PROBLEM_NOT_FOUND",
                },
            )

        problem_statement = getattr(
            problem,
            "statement",
            "",
        )

        constraints = getattr(
            problem,
            "constraints",
            "",
        )

        sample_test_case = (
            request.sample_test_case.strip()
            if isinstance(
                request.sample_test_case,
                str,
            )
            else ""
        )

        if not sample_test_case:
            sample_test_case = (
                problem.examples[0].input
                if getattr(
                    problem,
                    "examples",
                    None,
                )
                else ""
            )

        result = gemini_service.generate_test_cases(
            problem_statement=problem_statement,
            sample=sample_test_case,
            constraints=constraints,
            code=request.code,
            language=request.language,
            api_key=api_key,
        )

        public_tests = list(
            result.public_tests or []
        )

        hidden_tests = list(
            result.hidden_tests or []
        )

        if not public_tests and not hidden_tests:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "AI generated no test cases.",
                    "code": "NO_TEST_CASES_GENERATED",
                },
            )

        is_valid, error = _verify_additional_tests(
            problem=problem,
            public_tests=public_tests,
            hidden_tests=hidden_tests,
        )

        if not is_valid:
            logger.warning(
                "[Coding Test Cases] Generated tests failed verification: %s",
                error or "unknown error",
            )

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": (
                        "Generated test cases could not be "
                        "independently verified."
                    ),
                    "code": "TEST_CASE_VERIFICATION_FAILED",
                },
            )

        problem_store.append_tests(
            session_id,
            request.problem_id,
            public_tests,
            hidden_tests,
        )

        return TestCaseGenerateResponse(
            public_tests=[
                CodingTestCase(
                    input=test.input,
                    output=test.output,
                )
                for test in public_tests
            ],
            hidden_test_count=len(
                hidden_tests
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_TEST_CASE_REQUEST",
            },
        ) from exc

    except Exception:
        logger.exception(
            "[Coding Test Cases] Generation failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Test case generation service is unavailable.",
                "code": "TEST_CASE_GENERATION_FAILED",
            },
        )


# ============================================================================
# SUBMIT CODE
# ============================================================================


@router.post(
    "/submit",
    response_model=CodeSubmitResponse,
)
async def submit_code(
    request: CodeSubmitRequest,
    session_id: str = Depends(get_session_id),
):
    """
    Submit code against samples, public tests, or all tests.

    The referenced coding problem must belong to the current
    BodhaQ session.

    This endpoint does not require a Gemini API key.

    Hidden test inputs and outputs are never returned.
    """

    try:
        problem = problem_store.get_problem(
            session_id,
            request.problem_id,
        )

        if problem is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Problem not found or expired.",
                    "code": "CODING_PROBLEM_NOT_FOUND",
                },
            )

        samples = [
            CodingTestCase(
                input=example.input,
                output=example.output,
            )
            for example in problem.examples
        ]

        all_tests: list[
            tuple[str, list[Any]]
        ] = []

        if request.test_suite in {
            "samples",
            "all",
        }:
            all_tests.append(
                (
                    "sample",
                    samples,
                )
            )

        if request.test_suite in {
            "public",
            "all",
        }:
            all_tests.append(
                (
                    "public",
                    problem.public_tests,
                )
            )

        if request.test_suite == "all":
            all_tests.append(
                (
                    "hidden",
                    problem.hidden_tests,
                )
            )

        total_samples = (
            len(samples)
            if request.test_suite in {
                "samples",
                "all",
            }
            else 0
        )

        total_public = (
            len(problem.public_tests)
            if request.test_suite in {
                "public",
                "all",
            }
            else 0
        )

        total_hidden = (
            len(problem.hidden_tests)
            if request.test_suite == "all"
            else 0
        )

        total = (
            total_samples
            + total_public
            + total_hidden
        )

        passed = 0
        passed_samples = 0
        passed_public = 0
        passed_hidden = 0

        if total == 0:
            return CodeSubmitResponse(
                status="No Tests",
                details="No tests available for this problem.",
                passed_tests=0,
                total_tests=0,
                passed_samples=0,
                total_samples=0,
                passed_public=0,
                total_public=0,
                passed_hidden=0,
                total_hidden=0,
            )

        for test_type, test_list in all_tests:
            for test in test_list:
                is_hidden = (
                    test_type == "hidden"
                )

                execution_request = CodeExecutionRequest(
                    language=request.language,
                    code=request.code,
                    stdin=test.input,
                )

                execution_result = (
                    code_execution_service.execute_code(
                        execution_request
                    )
                )

                # ------------------------------------------------------------
                # EXECUTION SERVICE UNAVAILABLE
                # ------------------------------------------------------------

                if (
                    execution_result.status
                    == "execution_service_unavailable"
                ):
                    return CodeSubmitResponse(
                        status="System Error",
                        details=(
                            "Code execution environment "
                            "is unavailable."
                        ),
                        passed_tests=passed,
                        total_tests=total,
                        passed_samples=passed_samples,
                        total_samples=total_samples,
                        passed_public=passed_public,
                        total_public=total_public,
                        passed_hidden=passed_hidden,
                        total_hidden=total_hidden,
                    )

                # ------------------------------------------------------------
                # EXECUTION ERROR
                # ------------------------------------------------------------

                if execution_result.status != "success":
                    status_map = {
                        "compilation_error": "Compilation Error",
                        "runtime_error": "Runtime Error",
                        "timeout": "Time Limit Exceeded",
                        "memory_limit": "Memory Limit Exceeded",
                        "output_limit": "Output Limit Exceeded",
                        "execution_error": "Execution Error",
                        "unsupported_language": "Unsupported Language",
                    }

                    readable_status = status_map.get(
                        execution_result.status,
                        "Runtime Error",
                    )

                    if is_hidden:
                        return CodeSubmitResponse(
                            status=readable_status,
                            details="Hidden test failed.",
                            passed_tests=passed,
                            total_tests=total,
                            passed_samples=passed_samples,
                            total_samples=total_samples,
                            passed_public=passed_public,
                            total_public=total_public,
                            passed_hidden=passed_hidden,
                            total_hidden=total_hidden,
                            hidden_test_failed=True,
                            failed_test_type="hidden",
                        )

                    return CodeSubmitResponse(
                        status=readable_status,
                        details=_safe_public_text(
                            execution_result.stderr
                            or "Code execution failed."
                        ),
                        passed_tests=passed,
                        total_tests=total,
                        passed_samples=passed_samples,
                        total_samples=total_samples,
                        passed_public=passed_public,
                        total_public=total_public,
                        passed_hidden=passed_hidden,
                        total_hidden=total_hidden,
                        failed_test_type=test_type,
                        failed_test_input=test.input,
                        failed_test_expected=test.output,
                        failed_test_actual=None,
                    )

                # ------------------------------------------------------------
                # OUTPUT COMPARISON
                # ------------------------------------------------------------

                actual = _normalize_output(
                    execution_result.stdout
                )

                expected = _normalize_output(
                    test.output
                )

                if actual != expected:
                    if is_hidden:
                        return CodeSubmitResponse(
                            status="Wrong Answer",
                            details="Hidden test failed.",
                            passed_tests=passed,
                            total_tests=total,
                            passed_samples=passed_samples,
                            total_samples=total_samples,
                            passed_public=passed_public,
                            total_public=total_public,
                            passed_hidden=passed_hidden,
                            total_hidden=total_hidden,
                            hidden_test_failed=True,
                            failed_test_type="hidden",
                        )

                    return CodeSubmitResponse(
                        status="Wrong Answer",
                        details=(
                            "Output did not match expected output."
                        ),
                        passed_tests=passed,
                        total_tests=total,
                        passed_samples=passed_samples,
                        total_samples=total_samples,
                        passed_public=passed_public,
                        total_public=total_public,
                        passed_hidden=passed_hidden,
                        total_hidden=total_hidden,
                        failed_test_type=test_type,
                        failed_test_input=test.input,
                        failed_test_expected=_safe_public_text(
                            expected,
                            max_length=20000,
                        ),
                        failed_test_actual=_safe_public_text(
                            actual,
                            max_length=20000,
                        ),
                    )

                # ------------------------------------------------------------
                # TEST PASSED
                # ------------------------------------------------------------

                passed += 1

                if test_type == "sample":
                    passed_samples += 1

                elif test_type == "public":
                    passed_public += 1

                elif test_type == "hidden":
                    passed_hidden += 1

        return CodeSubmitResponse(
            status=(
                "Accepted"
                if request.test_suite == "all"
                else "Tests Passed"
            ),
            details="All tests passed.",
            passed_tests=passed,
            total_tests=total,
            passed_samples=passed_samples,
            total_samples=total_samples,
            passed_public=passed_public,
            total_public=total_public,
            passed_hidden=passed_hidden,
            total_hidden=total_hidden,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "code": "INVALID_CODE_SUBMISSION_REQUEST",
            },
        ) from exc

    except Exception:
        logger.exception(
            "[Coding Submit] Submission failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Code submission service is unavailable.",
                "code": "CODE_SUBMISSION_FAILED",
            },
        )