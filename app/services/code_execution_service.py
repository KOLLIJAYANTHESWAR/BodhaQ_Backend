"""
BodhaQ Code Execution Service.

Runs user/generated Java and Python code inside isolated Docker containers.

Security goals:
- No host-side code execution.
- No network access.
- Memory limit.
- CPU limit.
- PID limit.
- Capability dropping.
- No privilege escalation.
- Non-root container execution.
- Read-only container root filesystem during execution.
- Temporary workspace only.
- Execution timeout.
- Compilation timeout.
- Output-size limit for stdout and stderr.
- Automatic container cleanup.
- Automatic temporary workspace cleanup.
- Hidden test inputs/outputs are never exposed by this service.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from app.models.requests import CodeExecutionRequest
from app.models.responses import CodeExecutionResponse


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

COMPILATION_TIMEOUT_SECONDS = 10
EXECUTION_TIMEOUT_SECONDS = 5

MEMORY_LIMIT = "128m"
CPU_LIMIT = "0.5"
PID_LIMIT = "64"

MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB

JAVA_IMAGE = "eclipse-temurin:17-alpine"
PYTHON_IMAGE = "python:3.10-alpine"

SUPPORTED_LANGUAGES = {"java", "python"}

# Non-root UID/GID used inside execution containers.
CONTAINER_UID = "1000"
CONTAINER_GID = "1000"

# Maximum chunk read from Docker stdout/stderr at a time.
OUTPUT_READ_CHUNK_SIZE = 16 * 1024

# Small polling interval used while collecting subprocess output.
PROCESS_POLL_INTERVAL_SECONDS = 0.05

# Give reader threads a short period to finish after the Docker process exits.
READER_JOIN_TIMEOUT_SECONDS = 1.0


# ============================================================================
# INTERNAL PROCESS OUTPUT TYPES
# ============================================================================


class _ProcessOutput:
    """
    Internal result returned by the bounded subprocess collector.
    """

    def __init__(
        self,
        *,
        stdout: bytes,
        stderr: bytes,
        returncode: int,
        timed_out: bool,
        output_limit_hit: bool,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out
        self.output_limit_hit = output_limit_hit


# ============================================================================
# SERVICE
# ============================================================================


class CodeExecutionService:
    """
    Executes Java and Python programs inside Docker.

    The service is intentionally synchronous because it is currently called
    directly from FastAPI routes and the coding validation pipeline.
    """

    def __init__(self) -> None:
        self.docker_available = self._check_docker_available()

        if self.docker_available:
            logger.info(
                "[CodeExecution] Docker is available."
            )
        else:
            logger.warning(
                "[CodeExecution] Docker is unavailable. "
                "Code execution will return execution_service_unavailable."
            )

    # ========================================================================
    # DOCKER AVAILABILITY
    # ========================================================================

    def _check_docker_available(self) -> bool:
        """
        Check whether Docker is installed and the daemon is responsive.
        """

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3,
            )

            return result.returncode == 0

        except FileNotFoundError:
            logger.warning(
                "[CodeExecution] Docker executable was not found."
            )
            return False

        except subprocess.TimeoutExpired:
            logger.warning(
                "[CodeExecution] Docker availability check timed out."
            )
            return False

        except Exception:
            logger.exception(
                "[CodeExecution] Docker availability check failed."
            )
            return False

    def _ensure_docker_available(self) -> bool:
        """
        Re-check Docker when the initial availability check failed.

        This allows Docker to be started after FastAPI has already started.
        """

        if self.docker_available:
            return True

        self.docker_available = self._check_docker_available()

        return self.docker_available

    # ========================================================================
    # RESPONSE HELPERS
    # ========================================================================

    @staticmethod
    def _response(
        *,
        status: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = -1,
        execution_time_ms: int = 0,
    ) -> CodeExecutionResponse:
        """
        Centralized response construction.
        """

        return CodeExecutionResponse(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=execution_time_ms,
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_language(language: str) -> str:
        """
        Normalize supported language names.
        """

        return (language or "").strip().lower()

    @staticmethod
    def _normalize_output(output: str) -> str:
        """
        Normalize line endings and trailing whitespace.

        This helper is intentionally conservative.

        It does NOT:
        - sort output
        - remove arbitrary internal whitespace
        - modify tokens
        - parse problem-specific structures

        Problem-specific output normalization belongs to the judge layer.
        """

        return (
            (output or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    # ========================================================================
    # OUTPUT LIMIT
    # ========================================================================

    @staticmethod
    def _output_exceeds_limit(output: str) -> bool:
        """
        Check output size in UTF-8 bytes.
        """

        if not output:
            return False

        return len(
            output.encode(
                "utf-8",
                errors="replace",
            )
        ) > MAX_OUTPUT_BYTES

    @staticmethod
    def _truncate_output(output: str) -> str:
        """
        Safely truncate output to MAX_OUTPUT_BYTES without splitting
        an invalid UTF-8 sequence.
        """

        if not output:
            return ""

        encoded = output.encode(
            "utf-8",
            errors="replace",
        )

        if len(encoded) <= MAX_OUTPUT_BYTES:
            return output

        return encoded[:MAX_OUTPUT_BYTES].decode(
            "utf-8",
            errors="replace",
        )

    # ========================================================================
    # DOCKER IMAGE CHECK
    # ========================================================================

    @staticmethod
    def _image_available(image: str) -> bool:
        """
        Check whether a Docker image already exists locally.

        This does not pull the image automatically.
        """

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image,
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

            return result.returncode == 0

        except Exception:
            logger.exception(
                "[CodeExecution] Docker image check failed."
            )
            return False

    # ========================================================================
    # DOCKER COMMAND BUILDERS
    # ========================================================================

    @staticmethod
    def _base_docker_command(
        container_name: str,
        temp_dir: str,
    ) -> list[str]:
        """
        Build the common Docker isolation configuration.

        Runtime workspace is mounted read-only.

        The container also receives:
        - no network
        - no Linux capabilities
        - no privilege escalation
        - non-root execution
        - read-only root filesystem
        - temporary writable /tmp
        - memory/CPU/PID limits
        """

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            MEMORY_LIMIT,
            "--cpus",
            CPU_LIMIT,
            "--pids-limit",
            PID_LIMIT,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{CONTAINER_UID}:{CONTAINER_GID}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec",
            "-v",
            f"{temp_dir}:/workspace:ro",
            "-w",
            "/workspace",
        ]

    @staticmethod
    def _base_compile_docker_command(
        container_name: str,
        temp_dir: str,
    ) -> list[str]:
        """
        Build the Docker command used for Java compilation.

        The workspace must remain writable during compilation because javac
        generates .class files there.
        """

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            MEMORY_LIMIT,
            "--cpus",
            CPU_LIMIT,
            "--pids-limit",
            PID_LIMIT,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{CONTAINER_UID}:{CONTAINER_GID}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec",
            "-v",
            f"{temp_dir}:/workspace:rw",
            "-w",
            "/workspace",
        ]

    # ========================================================================
    # CONTAINER CLEANUP
    # ========================================================================

    @staticmethod
    def _force_remove_container(
        container_name: str,
    ) -> None:
        """
        Force-remove a container if it still exists.

        Cleanup failures are logged but never replace the original execution
        result.
        """

        try:
            subprocess.run(
                [
                    "docker",
                    "rm",
                    "-f",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

        except Exception:
            logger.debug(
                "[CodeExecution] Container cleanup failed for %s.",
                container_name,
            )

    # ========================================================================
    # BOUNDED SUBPROCESS OUTPUT
    # ========================================================================

    @staticmethod
    def _reader_thread(
        stream,
        stream_name: str,
        output_queue: queue.Queue[tuple[str, bytes | None]],
    ) -> None:
        """
        Read subprocess output incrementally.

        A dedicated reader thread is used because Windows does not support
        selector-based polling of subprocess pipes in the same way as POSIX.

        The reader never stores the entire process output in memory.
        """

        try:
            while True:
                data = stream.read(
                    OUTPUT_READ_CHUNK_SIZE
                )

                if not data:
                    output_queue.put(
                        (
                            stream_name,
                            None,
                        )
                    )
                    return

                output_queue.put(
                    (
                        stream_name,
                        data,
                    )
                )

        except (OSError, ValueError):
            output_queue.put(
                (
                    stream_name,
                    None,
                )
            )

    def _terminate_process_and_container(
        self,
        process: subprocess.Popen,
        container_name: str,
    ) -> None:
        """
        Stop both the Docker CLI process and the execution container.

        Container removal is attempted first so that a Docker CLI process
        blocked on container execution can terminate promptly.
        """

        self._force_remove_container(
            container_name
        )

        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass

        try:
            process.wait(
                timeout=2
            )
        except Exception:
            pass

    def _run_process_bounded(
        self,
        command: list[str],
        *,
        timeout_seconds: float,
        container_name: str,
    ) -> _ProcessOutput:
        """
        Run a subprocess while enforcing bounded stdout/stderr collection.

        Output is read incrementally in fixed-size chunks. Each stream is
        capped independently at MAX_OUTPUT_BYTES.

        Once either stream exceeds the limit, the Docker container is
        terminated and the result is marked as output_limit.
        """

        process: subprocess.Popen[bytes] | None = None

        output_queue: queue.Queue[
            tuple[str, bytes | None]
        ] = queue.Queue()

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        stdout_size = 0
        stderr_size = 0

        stdout_closed = False
        stderr_closed = False

        output_limit_hit = False
        timed_out = False

        start = time.perf_counter()

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )

            if process.stdout is None or process.stderr is None:
                raise RuntimeError(
                    "Execution process pipes were not created."
                )

            stdout_thread = threading.Thread(
                target=self._reader_thread,
                args=(
                    process.stdout,
                    "stdout",
                    output_queue,
                ),
                daemon=True,
            )

            stderr_thread = threading.Thread(
                target=self._reader_thread,
                args=(
                    process.stderr,
                    "stderr",
                    output_queue,
                ),
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()

            while True:
                elapsed = (
                    time.perf_counter()
                    - start
                )

                if elapsed >= timeout_seconds:
                    timed_out = True
                    self._terminate_process_and_container(
                        process,
                        container_name,
                    )
                    break

                try:
                    stream_name, data = output_queue.get(
                        timeout=PROCESS_POLL_INTERVAL_SECONDS
                    )

                except queue.Empty:
                    if (
                        process.poll() is not None
                        and stdout_closed
                        and stderr_closed
                    ):
                        break

                    continue

                if data is None:
                    if stream_name == "stdout":
                        stdout_closed = True
                    else:
                        stderr_closed = True
                elif stream_name == "stdout":
                    stdout_size += len(data)

                    if stdout_size > MAX_OUTPUT_BYTES:
                        output_limit_hit = True

                        self._terminate_process_and_container(
                            process,
                            container_name,
                        )
                        break

                    stdout_chunks.append(data)

                else:
                    stderr_size += len(data)

                    if stderr_size > MAX_OUTPUT_BYTES:
                        output_limit_hit = True

                        self._terminate_process_and_container(
                            process,
                            container_name,
                        )
                        break

                if (
                    process.poll() is not None
                    and stdout_closed
                    and stderr_closed
                ):
                    break

            # --------------------------------------------------------------
            # Ensure the Docker CLI process is fully reaped.
            # --------------------------------------------------------------

            if process.poll() is None:
                try:
                    process.wait(
                        timeout=2
                    )
                except subprocess.TimeoutExpired:
                    self._terminate_process_and_container(
                        process,
                        container_name,
                    )

            # --------------------------------------------------------------
            # Give reader threads a short opportunity to finish.
            # --------------------------------------------------------------

            stdout_thread.join(
                timeout=READER_JOIN_TIMEOUT_SECONDS
            )

            stderr_thread.join(
                timeout=READER_JOIN_TIMEOUT_SECONDS
            )

            # --------------------------------------------------------------
            # Drain already-buffered queue data without blocking.
            #
            # The data has already been bounded before being stored.
            # --------------------------------------------------------------

            while True:
                try:
                    stream_name, data = (
                        output_queue.get_nowait()
                    )
                except queue.Empty:
                    break

                if data is None:
                    continue

                if stream_name == "stdout":
                    remaining = (
                        MAX_OUTPUT_BYTES
                        - stdout_size
                    )

                    if remaining <= 0:
                        output_limit_hit = True
                        continue

                    if len(data) > remaining:
                        stdout_chunks.append(
                            data[:remaining]
                        )
                        stdout_size += remaining
                        output_limit_hit = True
                    else:
                        stdout_chunks.append(data)
                        stdout_size += len(data)

                else:
                    remaining = (
                        MAX_OUTPUT_BYTES
                        - stderr_size
                    )

                    if remaining <= 0:
                        output_limit_hit = True
                        continue

                    if len(data) > remaining:
                        stderr_chunks.append(
                            data[:remaining]
                        )
                        stderr_size += remaining
                        output_limit_hit = True
                    else:
                        stderr_chunks.append(data)
                        stderr_size += len(data)

            return _ProcessOutput(
                stdout=b"".join(stdout_chunks),
                stderr=b"".join(stderr_chunks),
                returncode=(
                    process.returncode
                    if process.returncode is not None
                    else -1
                ),
                timed_out=timed_out,
                output_limit_hit=output_limit_hit,
            )

        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        self._terminate_process_and_container(
                            process,
                            container_name,
                        )
                except Exception:
                    pass

    # ========================================================================
    # JAVA COMPILATION
    # ========================================================================

    def _compile_java(
        self,
        temp_dir: str,
        compile_container_name: str,
    ) -> CodeExecutionResponse | None:
        """
        Compile Main.java inside Docker.

        Returns:
            None when compilation succeeds.
            CodeExecutionResponse when compilation fails.
        """

        compile_cmd = self._base_compile_docker_command(
            container_name=compile_container_name,
            temp_dir=temp_dir,
        )

        compile_cmd.extend(
            [
                JAVA_IMAGE,
                "javac",
                "Main.java",
            ]
        )

        start = time.perf_counter()

        try:
            result = self._run_process_bounded(
                compile_cmd,
                timeout_seconds=COMPILATION_TIMEOUT_SECONDS,
                container_name=compile_container_name,
            )

        except FileNotFoundError:
            elapsed_ms = int(
                (time.perf_counter() - start) * 1000
            )

            self.docker_available = False

            return self._response(
                status="execution_service_unavailable",
                stdout="",
                stderr="Execution service is unavailable.",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        except Exception:
            elapsed_ms = int(
                (time.perf_counter() - start) * 1000
            )

            logger.exception(
                "[CodeExecution] Java compilation failed unexpectedly."
            )

            return self._response(
                status="compilation_error",
                stdout="",
                stderr="Compilation service failed.",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = int(
            (time.perf_counter() - start) * 1000
        )

        stdout_data = self._decode_output(
            result.stdout
        )

        stderr_data = self._decode_output(
            result.stderr
        )

        if result.timed_out:
            logger.warning(
                "[CodeExecution] Java compilation timed out after %d ms.",
                elapsed_ms,
            )

            return self._response(
                status="compilation_error",
                stdout=stdout_data,
                stderr="Compilation timed out.",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        if result.output_limit_hit:
            return self._response(
                status="output_limit",
                stdout=stdout_data,
                stderr=stderr_data,
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
            )

        if result.returncode != 0:
            stderr = (
                stderr_data.strip()
                or stdout_data.strip()
                or "Compilation Error"
            )

            return self._response(
                status="compilation_error",
                stdout="",
                stderr=stderr,
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
            )

        logger.info(
            "[CodeExecution] Java compilation completed in %d ms.",
            elapsed_ms,
        )

        return None

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def _run_container(
        self,
        command: list[str],
        container_name: str,
    ) -> CodeExecutionResponse:
        """
        Execute a Docker container with timeout and bounded output.
        """

        start = time.perf_counter()

        try:
            result = self._run_process_bounded(
                command,
                timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
                container_name=container_name,
            )

        except FileNotFoundError:
            self.docker_available = False

            elapsed_ms = int(
                (time.perf_counter() - start) * 1000
            )

            return self._response(
                status="execution_service_unavailable",
                stdout="",
                stderr="Execution service is unavailable.",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        except Exception:
            logger.exception(
                "[CodeExecution] Docker execution failed unexpectedly."
            )

            elapsed_ms = int(
                (time.perf_counter() - start) * 1000
            )

            return self._response(
                status="runtime_error",
                stdout="",
                stderr="Execution service failed.",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = int(
            (time.perf_counter() - start) * 1000
        )

        stdout_data = self._decode_output(
            result.stdout
        )

        stderr_data = self._decode_output(
            result.stderr
        )

        if result.timed_out:
            return self._response(
                status="timeout",
                stdout=stdout_data,
                stderr="Time Limit Exceeded",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )

        if result.output_limit_hit:
            return self._response(
                status="output_limit",
                stdout=stdout_data,
                stderr=stderr_data,
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
            )

        if result.returncode == 137:
            return self._response(
                status="runtime_error",
                stdout=stdout_data,
                stderr="Memory Limit Exceeded (OOM Killed)",
                exit_code=137,
                execution_time_ms=elapsed_ms,
            )

        if result.returncode != 0:
            return self._response(
                status="runtime_error",
                stdout=stdout_data,
                stderr=(
                    stderr_data.strip()
                    or "Runtime Error"
                ),
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
            )

        logger.info(
            "[CodeExecution] Container completed in %d ms.",
            elapsed_ms,
        )

        return self._response(
            status="success",
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=0,
            execution_time_ms=elapsed_ms,
        )

    @staticmethod
    def _decode_output(
        data: bytes,
    ) -> str:
        """
        Decode bounded process output safely.
        """

        return data.decode(
            "utf-8",
            errors="replace",
        )

    # ========================================================================
    # WORKSPACE PREPARATION
    # ========================================================================

    @staticmethod
    def _prepare_workspace(
        temp_dir: str,
        *,
        compilation: bool,
    ) -> None:
        """
        Prepare temporary workspace permissions for the non-root container.

        The workspace is created by the FastAPI process and therefore may not
        be writable by UID 1000 inside Docker. The directory is intentionally
        temporary and isolated, so it can safely be made accessible to the
        dedicated execution UID.

        Runtime execution mounts the workspace read-only.
        Java compilation mounts it read-write.
        """

        path = Path(temp_dir)

        try:
            os.chmod(
                path,
                0o777,
            )

            for child in path.iterdir():
                os.chmod(
                    child,
                    0o666 if compilation else 0o444,
                )

        except OSError as exc:
            logger.exception(
                "[CodeExecution] Failed to prepare execution workspace."
            )
            raise RuntimeError(
                "Unable to prepare execution workspace."
            ) from exc

    # ========================================================================
    # PUBLIC EXECUTION API
    # ========================================================================

    def execute_code(
        self,
        request: CodeExecutionRequest,
    ) -> CodeExecutionResponse:
        """
        Execute Java or Python code inside an isolated Docker container.

        This method is used by:

        - Coding IDE "Run"
        - Coding problem validation
        - Coding submission
        - AI-generated problem verification
        """

        language = self._normalize_language(
            request.language
        )

        # --------------------------------------------------------------------
        # LANGUAGE VALIDATION
        # --------------------------------------------------------------------

        if language not in SUPPORTED_LANGUAGES:
            return self._response(
                status="unsupported_language",
                stdout="",
                stderr=(
                    f"Language '{request.language}' "
                    f"is not supported."
                ),
                exit_code=-1,
                execution_time_ms=0,
            )

        # --------------------------------------------------------------------
        # CODE VALIDATION
        # --------------------------------------------------------------------

        code = request.code or ""

        if not code.strip():
            return self._response(
                status="runtime_error",
                stdout="",
                stderr="Source code cannot be empty.",
                exit_code=-1,
                execution_time_ms=0,
            )

        # --------------------------------------------------------------------
        # DOCKER VALIDATION
        # --------------------------------------------------------------------

        if not self._ensure_docker_available():
            return self._response(
                status="execution_service_unavailable",
                stdout="",
                stderr="",
                exit_code=-1,
                execution_time_ms=0,
            )

        # --------------------------------------------------------------------
        # IMAGE VALIDATION
        # --------------------------------------------------------------------

        required_image = (
            JAVA_IMAGE
            if language == "java"
            else PYTHON_IMAGE
        )

        if not self._image_available(
            required_image
        ):
            logger.warning(
                "[CodeExecution] Required Docker image is not available "
                "locally: %s",
                required_image,
            )

            return self._response(
                status="execution_service_unavailable",
                stdout="",
                stderr=(
                    "Required execution environment is unavailable. "
                    "Please ensure the configured Docker image is installed."
                ),
                exit_code=-1,
                execution_time_ms=0,
            )

        # --------------------------------------------------------------------
        # TEMPORARY WORKSPACE
        # --------------------------------------------------------------------

        temp_dir = tempfile.mkdtemp(
            prefix="bodhaq_exec_"
        )

        execution_id = uuid.uuid4().hex[:12]

        container_name = (
            f"bodhaq_exec_{execution_id}"
        )

        compile_container_name = (
            f"{container_name}_compile"
        )

        try:
            # ---------------------------------------------------------------
            # INPUT FILE
            # ---------------------------------------------------------------

            input_file = (
                Path(temp_dir)
                / "input.txt"
            )

            input_file.write_text(
                request.stdin or "",
                encoding="utf-8",
            )

            # ---------------------------------------------------------------
            # JAVA
            # ---------------------------------------------------------------

            if language == "java":

                source_file = (
                    Path(temp_dir)
                    / "Main.java"
                )

                source_file.write_text(
                    code,
                    encoding="utf-8",
                )

                self._prepare_workspace(
                    temp_dir,
                    compilation=True,
                )

                # -----------------------------------------------------------
                # COMPILE
                # -----------------------------------------------------------

                compilation_result = (
                    self._compile_java(
                        temp_dir=temp_dir,
                        compile_container_name=(
                            compile_container_name
                        ),
                    )
                )

                if compilation_result is not None:
                    return compilation_result

                # -----------------------------------------------------------
                # PREPARE READ-ONLY RUNTIME WORKSPACE
                # -----------------------------------------------------------

                self._prepare_workspace(
                    temp_dir,
                    compilation=False,
                )

                # -----------------------------------------------------------
                # EXECUTE
                # -----------------------------------------------------------

                run_cmd = self._base_docker_command(
                    container_name=container_name,
                    temp_dir=temp_dir,
                )

                run_cmd.extend(
                    [
                        JAVA_IMAGE,
                        "sh",
                        "-c",
                        "exec java Main < input.txt",
                    ]
                )

                return self._run_container(
                    command=run_cmd,
                    container_name=container_name,
                )

            # ---------------------------------------------------------------
            # PYTHON
            # ---------------------------------------------------------------

            if language == "python":

                source_file = (
                    Path(temp_dir)
                    / "script.py"
                )

                source_file.write_text(
                    code,
                    encoding="utf-8",
                )

                self._prepare_workspace(
                    temp_dir,
                    compilation=False,
                )

                run_cmd = self._base_docker_command(
                    container_name=container_name,
                    temp_dir=temp_dir,
                )

                run_cmd.extend(
                    [
                        PYTHON_IMAGE,
                        "sh",
                        "-c",
                        "exec python -B script.py < input.txt",
                    ]
                )

                return self._run_container(
                    command=run_cmd,
                    container_name=container_name,
                )

            # This should never happen because of validation above.
            return self._response(
                status="unsupported_language",
                stdout="",
                stderr="Unsupported language.",
                exit_code=-1,
                execution_time_ms=0,
            )

        except OSError:
            logger.exception(
                "[CodeExecution] File-system error during execution."
            )

            return self._response(
                status="runtime_error",
                stdout="",
                stderr="Execution workspace error.",
                exit_code=-1,
                execution_time_ms=0,
            )

        except RuntimeError:
            logger.exception(
                "[CodeExecution] Execution workspace setup failed."
            )

            return self._response(
                status="runtime_error",
                stdout="",
                stderr="Execution workspace could not be prepared.",
                exit_code=-1,
                execution_time_ms=0,
            )

        except Exception:
            logger.exception(
                "[CodeExecution] Unexpected execution-service error."
            )

            return self._response(
                status="runtime_error",
                stdout="",
                stderr="Execution service error.",
                exit_code=-1,
                execution_time_ms=0,
            )

        finally:
            # ---------------------------------------------------------------
            # CONTAINER CLEANUP
            # ---------------------------------------------------------------

            self._force_remove_container(
                container_name
            )

            self._force_remove_container(
                compile_container_name
            )

            # ---------------------------------------------------------------
            # TEMP DIRECTORY CLEANUP
            # ---------------------------------------------------------------

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ============================================================================
# SINGLETON
# ============================================================================

code_execution_service = CodeExecutionService()