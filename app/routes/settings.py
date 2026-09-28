"""
Settings routes.

POST /api/settings/test-ai
    Validate a user-provided Gemini API key.
    The key is used for the request and is NEVER persisted.
"""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel


router = APIRouter(
    prefix="/api/settings",
    tags=["Settings"],
)


class TestAIRequest(BaseModel):
    api_key: str


class TestAIResponse(BaseModel):
    valid: bool
    message: str


@router.post(
    "/test-ai",
    response_model=TestAIResponse,
    status_code=status.HTTP_200_OK,
)
async def test_ai_connection(
    request_body: TestAIRequest,
    request: Request,
) -> TestAIResponse:
    """
    Test a user-provided Gemini API key.

    The key is used for a lightweight API call and is NEVER:
    - stored in the database
    - logged
    - returned in a response
    - placed in ChromaDB metadata

    Returns a controlled success/failure indicator.
    """
    api_key = request_body.api_key.strip()

    if not api_key:
        return TestAIResponse(
            valid=False,
            message="API key is required.",
        )

    try:
        from google import genai

        # Create a one-shot client using the user-provided key.
        # This client is scoped to this request only.
        client = genai.Client(api_key=api_key)

        # Send a minimal prompt to verify the key is valid.
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents="Reply with only: ok",
        )

        if response and response.text:
            return TestAIResponse(
                valid=True,
                message="Gemini API connection successful.",
            )

        return TestAIResponse(
            valid=False,
            message="Gemini returned an unexpected response.",
        )

    except Exception as exc:
        error_str = str(exc).lower()

        # Map to controlled error codes — never expose raw exception text
        if "api_key" in error_str or "api key" in error_str or "invalid" in error_str:
            return TestAIResponse(
                valid=False,
                message="The Gemini API key could not be verified. Check the key and try again.",
            )

        if "quota" in error_str or "rate" in error_str:
            return TestAIResponse(
                valid=False,
                message="Gemini API rate limit reached. Please wait and try again.",
            )

        if "unavailable" in error_str or "503" in error_str:
            return TestAIResponse(
                valid=False,
                message="Gemini is currently unavailable. Please try again later.",
            )

        # Generic — do not expose internal details
        return TestAIResponse(
            valid=False,
            message="Could not connect to Gemini. Check your API key and try again.",
        )
