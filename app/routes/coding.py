from fastapi import APIRouter, HTTPException

from app.models.requests import CodeExecutionRequest
from app.models.responses import CodeExecutionResponse
from app.services.code_execution_service import code_execution_service

router = APIRouter()

@router.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Executes the given code in an isolated environment and returns the result.
    """
    try:
        response = code_execution_service.execute_code(request)
        return response
    except Exception as e:
        # We catch unexpected errors and return a clean EXECUTION_SERVICE_UNAVAILABLE 
        # or CODE_EXECUTION_FAILED instead of internal tracebacks
        raise HTTPException(
            status_code=500,
            detail="CODE_EXECUTION_FAILED"
        )
