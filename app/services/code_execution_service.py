import logging
import subprocess
import tempfile
import os
import shutil
from typing import Tuple

from app.models.requests import CodeExecutionRequest
from app.models.responses import CodeExecutionResponse

logger = logging.getLogger(__name__)

# Configurable resource limits for isolated execution (conceptual, for when Docker is available)
EXECUTION_TIMEOUT_SECONDS = 5
MEMORY_LIMIT = "128m"
CPU_LIMIT = "0.5"
MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB

class CodeExecutionService:
    def __init__(self):
        self.docker_available = self._check_docker_available()

    def _check_docker_available(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            # We use a short timeout to check if the daemon is responsive
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Docker is not available: {e}")
            return False

    def execute_code(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """
        Main entrypoint for code execution.
        """
        # Validate language
        if request.language not in ["java", "python"]:
            return CodeExecutionResponse(
                status="unsupported_language",
                stdout="",
                stderr=f"Language '{request.language}' is not supported.",
                exit_code=-1,
                execution_time_ms=0
            )

        # Enforce execution environment availability
        if not self.docker_available:
            return CodeExecutionResponse(
                status="execution_service_unavailable",
                stdout="",
                stderr="Isolated execution requires Docker to be configured and running. The environment is currently unsafe/unavailable.",
                exit_code=-1,
                execution_time_ms=0
            )

        # This block would contain the actual execution logic if Docker was available.
        # Since we enforce isolation and Docker is the chosen mechanism, we would:
        # 1. Create a temporary directory.
        # 2. Write source code and stdin to files.
        # 3. Run docker container with strict limits (memory, cpu, network none).
        # 4. Capture output and enforce MAX_OUTPUT_BYTES.
        # 5. Handle timeouts gracefully.
        # 6. Cleanup temporary directory.
        
        # As per the requirements, we do NOT fall back to raw subprocess on the host.
        return CodeExecutionResponse(
            status="execution_service_unavailable",
            stdout="",
            stderr="Execution service interface is implemented, but Docker is required for secure execution.",
            exit_code=-1,
            execution_time_ms=0
        )

code_execution_service = CodeExecutionService()
