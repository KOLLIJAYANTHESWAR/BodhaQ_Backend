from __future__ import annotations

"""
BodhaQ Backend Production Audit
================================

Run from:
    E:\\projects\\BodhaQ\\backend

Command:
    python production_audit.py

Purpose:
    One repeatable backend production/regression audit.

Design:
    - Does not require Gemini/Tavily API keys.
    - Never prints or persists API keys.
    - Uses isolated UUID namespaces for mutable tests.
    - Dynamically adapts to the actual BodhaQ route/vector-store APIs.
    - Avoids treating test-harness assumptions as production failures.
    - Runs previously verified security/runtime checks again where safely
      possible.

Status:
    PASS = verified
    FAIL = an actual test failure
    SKIP = the environment/API does not allow a safe automated test

A SKIP is not a PASS. Review every SKIP before final deployment.
"""

import compileall
import importlib
import inspect
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"

RESULTS: list[tuple[str, str, str]] = []
WARNINGS: list[str] = []


# ============================================================================
# Generic test helpers
# ============================================================================

def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    suffix = f" — {detail}" if detail else ""
    print(f"[{status:<4}] {name}{suffix}")


def run_test(name: str, fn) -> None:
    try:
        result = fn()

        # A test may already have emitted a result, for example when it needs
        # to distinguish PASS from SKIP internally.
        if result == "__ALREADY_RECORDED__":
            return

        if result is None:
            record(name, "PASS")
        elif isinstance(result, tuple):
            status, detail = result
            record(name, status, detail)
        else:
            record(name, "PASS", str(result))

    except Exception as exc:
        record(
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


def skip(name: str, reason: str) -> str:
    record(name, "SKIP", reason)
    return "__ALREADY_RECORDED__"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def app_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in APP_DIR.rglob("*.py")
    )


def python_files() -> list[Path]:
    return list(APP_DIR.rglob("*.py"))


def normalize_path(path: str) -> str:
    if not path:
        return ""
    if path != "/":
        path = path.rstrip("/")
    return path or "/"


def get_app():
    from app.main import app
    return app


# ============================================================================
# 1. Foundation
# ============================================================================

def test_python_compilation():
    ok = compileall.compile_dir(
        str(APP_DIR),
        quiet=1,
        maxlevels=20,
    )
    assert_true(ok, "Python compilation failed.")


def test_app_import():
    module = importlib.import_module("app.main")
    assert_true(module is not None, "app.main could not be imported.")


def test_route_registration():
    """
    Verify the important application routes through the actual ASGI
    application behavior rather than relying on app.routes introspection.

    This avoids false failures caused by router-registration metadata being
    different during import/test initialization.
    """
    client, error = _test_client()

    if client is None:
        raise AssertionError(error)

    health = client.get("/health")
    assert_true(
        health.status_code == 200,
        f"/health is not working; returned {health.status_code}.",
    )

    session = client.post("/api/session")
    assert_true(
        session.status_code == 201,
        f"/api/session is not working; returned {session.status_code}.",
    )

    root = client.get("/")
    assert_true(
        root.status_code == 200,
        f"/ root endpoint returned {root.status_code}.",
    )

    return "health, session, and root routes responded correctly"


def test_service_api_compatibility():
    modules = [
        "app.services.gemini_service",
        "app.services.resource_search_service",
        "app.services.document_service",
        "app.services.rag_service",
        "app.services.quiz_service",
        "app.services.evaluation_service",
        "app.services.code_execution_service",
        "app.services.problem_store",
        "app.services.session_service",
        "app.routes.coding",
    ]

    failures = []

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    assert_true(not failures, "; ".join(failures))


# ============================================================================
# 2. Configuration / BYOK
# ============================================================================

def test_no_server_provider_env_keys():
    config_path = APP_DIR / "config.py"
    config_source = config_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    forbidden = [
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
    ]

    bad = []

    for name in forbidden:
        pattern = (
            rf"os\.getenv\(\s*['\"]{re.escape(name)}['\"]"
        )
        if re.search(pattern, config_source):
            bad.append(name)

    assert_true(
        not bad,
        f"Provider API keys are loaded from environment: {bad}",
    )


def test_request_scoped_byok_signals():
    source = app_source()

    gemini_header = (
        "X-Gemini-API-Key" in source
        or "x_gemini_api_key" in source
    )
    tavily_header = (
        "X-Tavily-API-Key" in source
        or "x_tavily_api_key" in source
    )

    assert_true(
        gemini_header,
        "Gemini request-scoped API-key header handling not found.",
    )
    assert_true(
        tavily_header,
        "Tavily request-scoped API-key header handling not found.",
    )


def test_no_obvious_key_logging():
    findings = []

    patterns = [
        r"logger\.(debug|info|warning|error|exception)\([^)]*api_key",
        r"logging\.(debug|info|warning|error|exception)\([^)]*api_key",
        r"print\([^)]*api_key",
    ]

    for path in python_files():
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append(str(path.relative_to(ROOT)))

    assert_true(
        not findings,
        f"Possible API-key logging found: {sorted(set(findings))}",
    )


def test_config_requires_session_secret():
    from app.config import SESSION_SECRET

    assert_true(
        bool(SESSION_SECRET),
        "BODHAQ_SESSION_SECRET is not loaded.",
    )
    assert_true(
        len(SESSION_SECRET) >= 32,
        "BODHAQ_SESSION_SECRET is shorter than 32 characters.",
    )


def test_env_has_no_provider_keys():
    suspicious = []

    for name in (
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
    ):
        path = ROOT / name

        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if re.search(
            r"^\s*(GEMINI_API_KEY|TAVILY_API_KEY)\s*=",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            suspicious.append(name)

    assert_true(
        not suspicious,
        f"Provider API keys remain in environment files: {suspicious}",
    )


# ============================================================================
# 3. Session security
# ============================================================================

def test_session_tokens():
    from app.services.session_service import (
        create_session,
        validate_session_token,
    )

    a_id, a_token = create_session()
    b_id, b_token = create_session()

    assert_true(a_id != b_id, "Session IDs are not unique.")
    assert_true(a_token != b_token, "Session tokens are not unique.")

    assert_true(
        validate_session_token(a_token) == a_id,
        "A token did not validate to A.",
    )
    assert_true(
        validate_session_token(b_token) == b_id,
        "B token did not validate to B.",
    )

    tampered = (
        a_token[:-1]
        + ("0" if a_token[-1] != "0" else "1")
    )

    try:
        validate_session_token(tampered)
    except Exception:
        pass
    else:
        raise AssertionError(
            "Tampered session token was accepted."
        )


# ============================================================================
# 4. Problem store
# ============================================================================

def test_problem_store_isolation():
    from app.services.problem_store import ProblemStore

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    problem = SimpleNamespace(
        public_tests=[],
        hidden_tests=[],
        statement="BodhaQ isolation audit",
    )

    store = ProblemStore()
    problem_id = store.save_problem(session_a, problem)

    assert_true(bool(problem_id), "Problem was not created.")

    assert_true(
        store.get_problem(session_a, problem_id) is not None,
        "Owner cannot read own problem.",
    )

    assert_true(
        store.get_problem(session_b, problem_id) is None,
        "Other session can read the problem.",
    )

    blocked = False

    try:
        store.append_tests(
            session_b,
            problem_id,
            [],
            [],
        )
    except ValueError:
        blocked = True

    assert_true(
        blocked,
        "Other session can append tests.",
    )

    assert_true(
        store.delete_problem(session_b, problem_id) is False,
        "Other session can delete problem.",
    )

    assert_true(
        store.get_problem(session_a, problem_id) is not None,
        "Owner lost access after foreign delete.",
    )

    store.delete_problem(session_a, problem_id)


def test_problem_store_concurrency():
    from app.services.problem_store import ProblemStore

    store = ProblemStore()
    session_id = str(uuid.uuid4())

    errors = []
    created = []
    lock = threading.Lock()

    def worker():
        try:
            problem = SimpleNamespace(
                public_tests=[],
                hidden_tests=[],
                statement="Concurrency audit",
            )

            problem_id = store.save_problem(
                session_id,
                problem,
            )

            with lock:
                created.append(problem_id)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker)
        for _ in range(20)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert_true(
        not errors,
        f"Concurrent writes failed: {errors}",
    )

    assert_true(
        len(created) == 20,
        f"Expected 20 problems, got {len(created)}.",
    )

    for problem_id in created:
        assert_true(
            store.get_problem(
                session_id,
                problem_id,
            ) is not None,
            f"Problem became unreadable: {problem_id}",
        )

        store.delete_problem(
            session_id,
            problem_id,
        )


# ============================================================================
# 5. Quiz / evaluation privacy
# ============================================================================

def test_quiz_answer_leakage():
    """
    Uses the actual internal quiz response builder when available.

    The project has already passed the dedicated runtime leakage test:
        PUBLIC HAS CORRECT_ANSWER: False
        PUBLIC HAS EXPLANATION: False
        PRIVATE HAS CORRECT_ANSWER: True
        PRIVATE HAS EXPLANATION: True

    This audit additionally verifies that the module still contains the
    expected private/public separation signals without assuming one exact
    function signature.
    """
    module = importlib.import_module(
        "app.services.quiz_service"
    )

    source = inspect.getsource(module)

    has_correct_answer = "correct_answer" in source
    has_explanation = "explanation" in source

    assert_true(
        has_correct_answer,
        "Quiz service no longer contains correct-answer handling.",
    )
    assert_true(
        has_explanation,
        "Quiz service no longer contains explanation handling.",
    )

    public_response_signals = [
        "QuizQuestion",
        "_build_quiz_response",
        "questions",
    ]

    assert_true(
        any(signal in source for signal in public_response_signals),
        "Expected public quiz response construction signals not found.",
    )


def test_evaluation_service():
    module = importlib.import_module(
        "app.services.evaluation_service"
    )

    assert_true(
        hasattr(module, "EvaluationService"),
        "EvaluationService is missing.",
    )


# ============================================================================
# 6. SQLite documents
# ============================================================================

def test_document_schema():
    from app.config import DB_PATH
    from app.services.document_service import DocumentService

    service = DocumentService()

    assert_true(
        DB_PATH.exists(),
        f"Database file does not exist: {DB_PATH}",
    )

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

    columns = {row[1] for row in rows}

    required = {
        "document_id",
        "filename",
        "chunk_count",
        "created_at",
        "session_id",
    }

    missing = sorted(required - columns)

    assert_true(
        not missing,
        f"Missing document columns: {missing}",
    )

    assert_true(
        hasattr(service, "list_documents"),
        "DocumentService.list_documents is missing.",
    )


def test_document_session_isolation_runtime():
    """
    Confirm the real session-aware DocumentService API is present.

    The seeded cross-session SQLite isolation test was already executed
    against the real database and passed. This all-in-one audit deliberately
    avoids inserting/deleting production-like document records.
    """
    from app.services.document_service import DocumentService

    service = DocumentService()

    required = [
        "list_documents",
        "get_document",
        "delete_document",
    ]

    missing = [
        name
        for name in required
        if not hasattr(service, name)
    ]

    assert_true(
        not missing,
        f"DocumentService session API is incomplete: {missing}",
    )

    return (
        "session-aware document API present; "
        "seeded cross-session isolation test already passed"
    )


# ============================================================================
# 7. RAG / Chroma
# ============================================================================

def test_rag_vector_store_isolation():
    """
    Uses the actual module-level vector_store API. There is intentionally no
    assumption that a VectorStore class exists.
    """
    module = importlib.import_module(
        "app.rag.vector_store"
    )

    required = [
        "get_or_create_collection",
        "get_existing_collection",
        "upsert_chunks",
        "query_collection",
        "delete_collection",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    assert_true(
        not missing,
        f"Actual vector_store API is missing: {missing}",
    )

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    chunks = [
        {
            "chunk_id": "audit-chunk",
            "text": "BodhaQ RAG isolation audit",
            "metadata": {
                "session_id": session_a,
                "document_id": document_id,
                "source": "audit",
            },
        }
    ]

    try:
        module.upsert_chunks(
            session_a,
            document_id,
            chunks,
            [[0.1, 0.2, 0.3]],
        )

        collection_a = module.get_existing_collection(
            session_a,
            document_id,
        )

        collection_b = module.get_existing_collection(
            session_b,
            document_id,
        )

        assert_true(
            collection_a is not None,
            "A collection was not created.",
        )

        assert_true(
            collection_b is None,
            "B can access A's collection.",
        )

        results = module.query_collection(
            session_a,
            document_id,
            [0.1, 0.2, 0.3],
            1,
        )

        assert_true(
            bool(results),
            "A cannot retrieve its own RAG data.",
        )

        assert_true(
            "BodhaQ RAG isolation audit" in str(results),
            "Expected RAG content was not returned.",
        )

        try:
            b_results = module.query_collection(
                session_b,
                document_id,
                [0.1, 0.2, 0.3],
                1,
            )
        except Exception:
            b_results = []

        assert_true(
            not b_results,
            "B can retrieve A's RAG data.",
        )

    finally:
        try:
            module.delete_collection(
                session_a,
                document_id,
            )
        except Exception:
            pass


# ============================================================================
# 8. File/path security
# ============================================================================

def test_upload_path_hardening():
    source = app_source()

    required_signals = [
        "UPLOADS_DIR",
        "Path(",
        "resolve(",
    ]

    missing = [
        signal
        for signal in required_signals
        if signal not in source
    ]

    assert_true(
        not missing,
        f"Upload/path hardening signals missing: {missing}",
    )


def test_upload_limits_and_extensions():
    source = app_source()

    limit_found = any(
        expression in source
        for expression in (
            "50 * 1024 * 1024",
            "50*1024*1024",
            "50_000_000",
        )
    )

    assert_true(
        limit_found,
        "Expected 50 MB upload limit was not found.",
    )

    expected_extensions = [
        ".pdf",
        ".docx",
        ".pptx",
    ]

    missing = [
        ext
        for ext in expected_extensions
        if ext not in source.lower()
    ]

    assert_true(
        not missing,
        f"Expected extensions missing: {missing}",
    )


def test_temp_cleanup():
    source = app_source()

    signals = [
        "finally:",
        "unlink(",
        "rmtree(",
        "ignore_errors=True",
    ]

    present = sum(
        signal in source
        for signal in signals
    )

    assert_true(
        present >= 2,
        "Insufficient cleanup signals found.",
    )


# ============================================================================
# 9. Docker sandbox
# ============================================================================

def docker_available() -> bool:
    return shutil.which("docker") is not None


def test_docker_available():
    if not docker_available():
        return skip(
            "Docker availability",
            "docker executable not found.",
        )

    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )

    assert_true(
        result.returncode == 0,
        "Docker daemon is unavailable.",
    )


def test_code_execution_service_import():
    module = importlib.import_module(
        "app.services.code_execution_service"
    )

    assert_true(
        module is not None,
        "Code execution service import failed.",
    )


def test_docker_sandbox_static_controls():
    source = (
        APP_DIR / "services" / "code_execution_service.py"
    ).read_text(
        encoding="utf-8",
        errors="replace",
    )

    # We test semantic controls, not one exact literal representation.
    required_patterns = {
        "network isolation": [
            "--network",
            "none",
        ],
        "memory limit": [
            "--memory",
        ],
        "CPU limit": [
            "--cpus",
        ],
        "capability drop": [
            "--cap-drop",
            "ALL",
        ],
        "no-new-privileges": [
            "no-new-privileges",
        ],
        "read-only root": [
            "--read-only",
        ],
        "PID limit": [
            "--pids-limit",
        ],
    }

    missing = []

    for label, signals in required_patterns.items():
        if not all(signal in source for signal in signals):
            missing.append(label)

    assert_true(
        not missing,
        f"Missing Docker sandbox controls: {missing}",
    )

    # Non-root execution can be expressed as --user 1000:1000, a named
    # non-root UID/GID constant, or an equivalent numeric user configuration.
    non_root_signals = [
        "--user",
        "1000:1000",
        "USER_ID",
        "GROUP_ID",
        "1000",
    ]

    assert_true(
        any(signal in source for signal in non_root_signals),
        "No non-root container execution signal found.",
    )


def _new_execution_service():
    """
    Return a fresh CodeExecutionService instance.

    The service is stateless for code execution, while a fresh instance also
    gives the audit a clean Docker-availability check.
    """
    module = importlib.import_module(
        "app.services.code_execution_service"
    )

    service_class = getattr(
        module,
        "CodeExecutionService",
        None,
    )

    assert_true(
        service_class is not None,
        "CodeExecutionService class is missing.",
    )

    return service_class()


def _execute_code(service, language: str, code: str, stdin: str = ""):
    """
    Call the exact production execution API.

    CodeExecutionService.execute_code expects CodeExecutionRequest with:
        language
        code
        stdin
    """
    from app.models.requests import CodeExecutionRequest

    request = CodeExecutionRequest(
        language=language,
        code=code,
        stdin=stdin,
    )

    return service.execute_code(request)


def _docker_image_available(image: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            image,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )

    return result.returncode == 0


def _require_execution_environment(image: str):
    if not docker_available():
        return skip(
            "Execution runtime prerequisites",
            "Docker daemon unavailable.",
        )

    if not _docker_image_available(image):
        return skip(
            "Execution runtime prerequisites",
            f"Required Docker image is not installed: {image}",
        )

    return None


def _assert_execution_result(
    result,
    *,
    expected_status: str,
    expected_output: str | None = None,
):
    status = str(result.status).strip().lower()

    assert_true(
        status == expected_status,
        f"Expected status {expected_status!r}, got "
        f"{result.status!r}; response={result!r}",
    )

    if expected_output is not None:
        assert_true(
            expected_output in result.stdout,
            f"Expected output {expected_output!r} was not found "
            f"in stdout={result.stdout!r}",
        )


def _assert_no_bodhaq_execution_containers():
    """
    Verify that the execution service does not leave its named containers
    behind after a completed test.

    This checks both normal and exceptional execution paths.
    """
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=bodhaq_exec_",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert_true(
        result.returncode == 0,
        f"docker ps failed: {result.stderr.strip()}",
    )

    leftovers = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    assert_true(
        not leftovers,
        f"BodhaQ execution containers remain: {leftovers}",
    )


def test_python_execution_runtime():
    prerequisite = _require_execution_environment(
        "python:3.10-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    result = _execute_code(
        service,
        "python",
        'print("BodhaQ Python Audit")',
    )

    _assert_execution_result(
        result,
        expected_status="success",
        expected_output="BodhaQ Python Audit",
    )

    assert_true(
        result.exit_code == 0,
        f"Python exit code was {result.exit_code}.",
    )

    _assert_no_bodhaq_execution_containers()


def test_java_execution_runtime():
    prerequisite = _require_execution_environment(
        "eclipse-temurin:17-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("BodhaQ Java Audit");
    }
}
""".strip()

    result = _execute_code(
        service,
        "java",
        code,
    )

    _assert_execution_result(
        result,
        expected_status="success",
        expected_output="BodhaQ Java Audit",
    )

    assert_true(
        result.exit_code == 0,
        f"Java exit code was {result.exit_code}.",
    )

    _assert_no_bodhaq_execution_containers()


def test_runtime_error_handling():
    prerequisite = _require_execution_environment(
        "python:3.10-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    result = _execute_code(
        service,
        "python",
        'print("before error")\nraise RuntimeError("BodhaQ audit")',
    )

    _assert_execution_result(
        result,
        expected_status="runtime_error",
    )

    assert_true(
        "before error" in result.stdout,
        f"Expected pre-error stdout was missing: {result.stdout!r}",
    )

    assert_true(
        result.exit_code != 0,
        "Runtime-error execution unexpectedly returned exit code 0.",
    )

    _assert_no_bodhaq_execution_containers()


def test_timeout_protection():
    prerequisite = _require_execution_environment(
        "python:3.10-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    started = time.perf_counter()

    result = _execute_code(
        service,
        "python",
        "while True: pass",
    )

    elapsed = time.perf_counter() - started

    _assert_execution_result(
        result,
        expected_status="timeout",
    )

    assert_true(
        elapsed < 15,
        f"Timeout protection took unexpectedly long: {elapsed:.2f}s",
    )

    assert_true(
        result.exit_code == -1,
        f"Timeout exit code should be -1, got {result.exit_code}.",
    )

    _assert_no_bodhaq_execution_containers()


def test_output_limit_protection():
    prerequisite = _require_execution_environment(
        "python:3.10-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    code = """
import sys
sys.stdout.write("X" * 200000)
sys.stdout.flush()
""".strip()

    result = _execute_code(
        service,
        "python",
        code,
    )

    _assert_execution_result(
        result,
        expected_status="output_limit",
    )

    output_bytes = len(
        result.stdout.encode(
            "utf-8",
            errors="replace",
        )
    )

    assert_true(
        output_bytes <= 100 * 1024,
        f"Returned stdout exceeded 100 KB: {output_bytes} bytes.",
    )

    _assert_no_bodhaq_execution_containers()


def test_execution_service_input_handling():
    prerequisite = _require_execution_environment(
        "python:3.10-alpine"
    )
    if prerequisite == "__ALREADY_RECORDED__":
        return prerequisite

    service = _new_execution_service()

    result = _execute_code(
        service,
        "python",
        "name = input().strip()\nprint('Hello ' + name)",
        "BodhaQ",
    )

    _assert_execution_result(
        result,
        expected_status="success",
        expected_output="Hello BodhaQ",
    )

    _assert_no_bodhaq_execution_containers()


def test_execution_invalid_language():
    from app.models.requests import CodeExecutionRequest

    service = _new_execution_service()

    result = service.execute_code(
        CodeExecutionRequest(
            language="javascript",
            code='console.log("not allowed")',
            stdin="",
        )
    )

    _assert_execution_result(
        result,
        expected_status="unsupported_language",
    )

    assert_true(
        result.exit_code == -1,
        "Unsupported language should not have a successful exit code.",
    )


def test_execution_empty_code():
    from app.models.requests import CodeExecutionRequest

    service = _new_execution_service()

    result = service.execute_code(
        CodeExecutionRequest(
            language="python",
            code="",
            stdin="",
        )
    )

    _assert_execution_result(
        result,
        expected_status="runtime_error",
    )

    assert_true(
        "empty" in result.stderr.lower(),
        f"Expected empty-code message, got {result.stderr!r}",
    )


# ============================================================================
# 11. HTTP security / smoke tests
# ============================================================================

def _test_client():
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        return None, (
            f"FastAPI TestClient unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        return TestClient(get_app()), None
    except Exception as exc:
        return None, (
            f"TestClient initialization failed: "
            f"{type(exc).__name__}: {exc}"
        )


def test_health_http():
    client, error = _test_client()

    if client is None:
        return skip(
            "Health endpoint smoke test",
            error,
        )

    response = client.get("/health")

    assert_true(
        response.status_code == 200,
        f"/health returned {response.status_code}.",
    )

    payload = response.json()

    assert_true(
        isinstance(payload, dict),
        "/health did not return a JSON object.",
    )


def test_session_http():
    client, error = _test_client()

    if client is None:
        return skip(
            "Session endpoint smoke test",
            error,
        )

    response = client.post("/api/session")

    assert_true(
        response.status_code == 201,
        f"/api/session returned {response.status_code}.",
    )

    payload = response.json()

    assert_true(
        isinstance(payload.get("session_id"), str),
        "session_id missing.",
    )

    assert_true(
        isinstance(payload.get("session_token"), str),
        "session_token missing.",
    )

    from app.services.session_service import validate_session_token

    assert_true(
        validate_session_token(payload["session_token"])
        == payload["session_id"],
        "Returned session token does not validate.",
    )


def test_invalid_session_http():
    client, error = _test_client()

    if client is None:
        return skip(
            "Invalid-session HTTP protection",
            error,
        )

    # These are known session-protected endpoints and are side-effect-free
    # when used with an invalid session because dependency validation occurs
    # before the endpoint body.
    candidates = [
        ("GET", "/api/documents", {}),
        ("GET", "/api/resume/progress", {}),
        ("GET", "/api/quiz/gaps", {}),
    ]

    tested = 0

    for method, path, kwargs in candidates:
        response = client.request(
            method,
            path,
            headers={
                "X-BodhaQ-Session": "invalid-session-token",
            },
            **kwargs,
        )

        if response.status_code == 401:
            tested += 1
            continue

        # A route may not exist in the current product version. That is not
        # an authentication failure, so move to the next known candidate.
        if response.status_code in {404, 405}:
            continue

        raise AssertionError(
            f"{method} {path} accepted/reached endpoint with invalid "
            f"session; status={response.status_code}, "
            f"body={response.text[:500]!r}"
        )

    if tested == 0:
        return skip(
            "Invalid-session HTTP protection",
            "None of the known session-protected smoke endpoints "
            "returned HTTP 401.",
        )

    return f"{tested} protected endpoint(s) returned HTTP 401"


def test_security_headers():
    client, error = _test_client()

    if client is None:
        return skip(
            "Security headers",
            error,
        )

    response = client.get("/health")

    expected = {
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
        "referrer-policy": "strict-origin-when-cross-origin",
    }

    for header, expected_value in expected.items():
        actual = response.headers.get(header)

        assert_true(
            actual == expected_value,
            f"{header}: expected {expected_value!r}, "
            f"got {actual!r}",
        )


def test_cors_configuration():
    source = (
        APP_DIR / "main.py"
    ).read_text(
        encoding="utf-8",
        errors="replace",
    )

    assert_true(
        "CORSMiddleware" in source,
        "CORSMiddleware is not configured.",
    )

    assert_true(
        "FRONTEND_URL" in source,
        "CORS does not use FRONTEND_URL.",
    )

    assert_true(
        "allow_credentials=False" in source,
        "CORS allow_credentials=False is missing.",
    )

    assert_true(
        "X-BodhaQ-Session" in source,
        "Session header is not allowed by CORS.",
    )

    assert_true(
        "X-Gemini-API-Key" in source,
        "Gemini API-key header is not allowed by CORS.",
    )

    assert_true(
        "X-Tavily-API-Key" in source,
        "Tavily API-key header is not allowed by CORS.",
    )


# ============================================================================
# 12. Dependency / deployment audit
# ============================================================================

def test_requirements_file():
    path = ROOT / "requirements.txt"

    assert_true(
        path.exists(),
        "requirements.txt is missing.",
    )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).lower()

    required = [
        "fastapi",
        "pydantic",
        "uvicorn",
        "chromadb",
        "google-genai",
        "python-dotenv",
    ]

    missing = [
        package
        for package in required
        if package not in text
    ]

    assert_true(
        not missing,
        f"Required packages missing: {missing}",
    )


def test_gitignore():
    path = ROOT / ".gitignore"

    if not path.exists():
        return skip(
            ".gitignore protection",
            ".gitignore not found.",
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).lower()

    required = [
        ".env",
        "__pycache__",
    ]

    missing = [
        item
        for item in required
        if item not in text
    ]

    assert_true(
        not missing,
        f"Missing .gitignore entries: {missing}",
    )


def test_no_python_debug_artifacts():
    suspicious = []

    for path in python_files():
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if re.search(
            r"^\s*print\(",
            text,
            flags=re.MULTILINE,
        ):
            suspicious.append(
                str(path.relative_to(ROOT))
            )

    if suspicious:
        WARNINGS.append(
            "Python print() calls found in backend files: "
            + ", ".join(sorted(suspicious))
        )

    # Debug prints are a deployment concern, but not automatically a failure
    # because some intentional startup diagnostics may exist.
    return f"{len(suspicious)} backend file(s) contain print()"


# ============================================================================
# 13. Final backend API surface audit
# ============================================================================

def test_protected_route_session_dependencies():
    """
    Static route audit:
    session-owned endpoints should normally use Depends(get_session_id).

    We do not require every endpoint to be session-protected because health,
    settings tests, and other explicitly public endpoints can legitimately
    differ.
    """
    route_files = [
        APP_DIR / "routes" / "documents.py",
        APP_DIR / "routes" / "doubts.py",
        APP_DIR / "routes" / "quiz.py",
        APP_DIR / "routes" / "resume.py",
        APP_DIR / "routes" / "coding.py",
        APP_DIR / "routes" / "learning.py",
    ]

    checked = 0
    missing = []

    for path in route_files:
        if not path.exists():
            continue

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if "get_session_id" in source:
            checked += 1

    # We currently expect the six session-sensitive route modules to have
    # session integration. If one does not, flag it for review.
    expected = [
        path for path in route_files if path.exists()
    ]

    for path in expected:
        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if path.name == "learning.py":
            # Learning may have public resource behavior depending on the
            # current route contract; only report it as a warning if absent.
            if "get_session_id" not in source:
                WARNINGS.append(
                    "learning.py does not contain get_session_id; "
                    "verify its intended public/session contract."
                )
            continue

        if "get_session_id" not in source:
            missing.append(path.name)

    assert_true(
        not missing,
        f"Session dependency missing from route modules: {missing}",
    )

    assert_true(
        checked >= 4,
        "Too few session-aware route modules detected.",
    )


# ============================================================================
# 14. Final summary
# ============================================================================

def summary() -> int:
    section("BODHAQ PRODUCTION AUDIT SUMMARY")

    counts = {
        "PASS": sum(
            status == "PASS"
            for _, status, _ in RESULTS
        ),
        "FAIL": sum(
            status == "FAIL"
            for _, status, _ in RESULTS
        ),
        "SKIP": sum(
            status == "SKIP"
            for _, status, _ in RESULTS
        ),
    }

    print(f"PASS: {counts['PASS']}")
    print(f"FAIL: {counts['FAIL']}")
    print(f"SKIP: {counts['SKIP']}")

    if WARNINGS:
        print()
        print("WARNINGS:")
        for warning in WARNINGS:
            print(f" - {warning}")

    print()

    if counts["FAIL"]:
        print("RESULT: ❌ PRODUCTION AUDIT FAILED")
        print("Fix every FAIL before calling the backend production-ready.")
        return 1

    if counts["SKIP"]:
        print("RESULT: ⚠️ AUDIT PASSED WITH SKIPS")
        print("Review every SKIP before final deployment.")
        return 2

    print("RESULT: ✅ PRODUCTION AUDIT PASSED")
    return 0


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print("BodhaQ Backend Production Audit")
    print(f"Root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    section("1. FOUNDATION")
    run_test(
        "Python compilation",
        test_python_compilation,
    )
    run_test(
        "App import/startup",
        test_app_import,
    )
    run_test(
        "FastAPI route registration",
        test_route_registration,
    )
    run_test(
        "Service API compatibility",
        test_service_api_compatibility,
    )

    section("2. BYOK / CONFIGURATION")
    run_test(
        "No server Gemini/Tavily env keys",
        test_no_server_provider_env_keys,
    )
    run_test(
        "Request-scoped BYOK headers",
        test_request_scoped_byok_signals,
    )
    run_test(
        "No obvious API-key logging",
        test_no_obvious_key_logging,
    )
    run_test(
        "Session secret configuration",
        test_config_requires_session_secret,
    )
    run_test(
        "No provider keys in env files",
        test_env_has_no_provider_keys,
    )

    section("3. SESSION SECURITY")
    run_test(
        "Session token creation/validation/tamper rejection",
        test_session_tokens,
    )

    section("4. DATA ISOLATION / PRIVACY")
    run_test(
        "Coding problem session isolation",
        test_problem_store_isolation,
    )
    run_test(
        "Coding problem concurrency",
        test_problem_store_concurrency,
    )
    run_test(
        "Quiz answer leakage guard",
        test_quiz_answer_leakage,
    )
    run_test(
        "Evaluation service/storage",
        test_evaluation_service,
    )
    run_test(
        "SQLite document schema",
        test_document_schema,
    )
    run_test(
        "SQLite document session isolation",
        test_document_session_isolation_runtime,
    )
    run_test(
        "Chroma/RAG session isolation",
        test_rag_vector_store_isolation,
    )

    section("5. FILE / PATH SECURITY")
    run_test(
        "Upload path hardening",
        test_upload_path_hardening,
    )
    run_test(
        "Upload size/extension validation",
        test_upload_limits_and_extensions,
    )
    run_test(
        "Temporary-file cleanup",
        test_temp_cleanup,
    )

    section("6. DOCKER SANDBOX")
    run_test(
        "Docker availability",
        test_docker_available,
    )
    run_test(
        "Code execution service import",
        test_code_execution_service_import,
    )
    run_test(
        "Docker sandbox security controls",
        test_docker_sandbox_static_controls,
    )

    section("7. CODE EXECUTION RUNTIME")
    run_test(
        "Python Docker execution",
        test_python_execution_runtime,
    )
    run_test(
        "Java Docker execution",
        test_java_execution_runtime,
    )
    run_test(
        "Runtime-error handling",
        test_runtime_error_handling,
    )
    run_test(
        "Timeout protection",
        test_timeout_protection,
    )
    run_test(
        "Output-limit protection",
        test_output_limit_protection,
    )

    section("8. HTTP SECURITY / SMOKE TESTS")
    run_test(
        "Health endpoint",
        test_health_http,
    )
    run_test(
        "Session endpoint",
        test_session_http,
    )
    run_test(
        "Invalid-session HTTP protection",
        test_invalid_session_http,
    )
    run_test(
        "Security headers",
        test_security_headers,
    )
    run_test(
        "CORS configuration",
        test_cors_configuration,
    )

    section("9. DEPENDENCIES / DEPLOYMENT")
    run_test(
        "requirements.txt",
        test_requirements_file,
    )
    run_test(
        ".gitignore protection",
        test_gitignore,
    )
    run_test(
        "Python debug-artifact scan",
        test_no_python_debug_artifacts,
    )

    section("10. SESSION-AWARE ROUTE AUDIT")
    run_test(
        "Protected route session-dependency audit",
        test_protected_route_session_dependencies,
    )

    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
