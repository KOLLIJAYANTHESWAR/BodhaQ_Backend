"""
Settings routes.

POST /api/settings/test-ai
    Test a user-provided Gemini API key.

POST /api/settings/test-tavily
    Test a user-provided Tavily API key.

Security:
    - API keys are supplied per request.
    - API keys are never persisted.
    - API keys are never logged.
    - API keys are never returned in responses.
    - No backend-owned Gemini/Tavily API keys are required.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    status,
)
from pydantic import BaseModel

from app.services.gemini_service import (
    GeminiAuthenticationError,
    GeminiQuotaError,
    gemini_service,
)
from app.services.resource_search_service import (
    resource_search_service,
)


logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/settings",
    tags=["Settings"],
)


# ============================================================================
# RESPONSE MODEL
# ============================================================================


class TestAIResponse(BaseModel):
    valid: bool
    message: str


# ============================================================================
# HELPERS
# ============================================================================


def _get_api_key(
    api_key: str | None,
    *,
    provider: str,
) -> str:
    """
    Validate a request-scoped provider API key.

    The key is intentionally never logged, persisted, or returned.
    """

    if not isinstance(
        api_key,
        str,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": (
                    f"{provider} API key is required."
                ),
                "code": (
                    f"{provider.upper()}_API_KEY_REQUIRED"
                ),
            },
        )

    normalized = api_key.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": (
                    f"{provider} API key is required."
                ),
                "code": (
                    f"{provider.upper()}_API_KEY_REQUIRED"
                ),
            },
        )

    return normalized


# ============================================================================
# TEST GEMINI CONNECTION
# ============================================================================


@router.post(
    "/test-ai",
    response_model=TestAIResponse,
    status_code=status.HTTP_200_OK,
)
def test_ai_connection(
    x_gemini_api_key: str | None = Header(
        default=None,
        alias="X-Gemini-API-Key",
    ),
) -> TestAIResponse:
    """
    Test the Gemini API key supplied by the current browser session.

    The API key:
        - comes from the request header
        - is never returned
        - is never logged
        - is never stored in the database
        - is never stored in application files
        - is never placed in ChromaDB metadata

    The same GeminiService used by the application is tested so that
    this endpoint verifies the actual AI configuration.
    """

    api_key = _get_api_key(
        x_gemini_api_key,
        provider="Gemini",
    )

    try:
        result = gemini_service.answer_doubt(
            question="Reply with only: ok",
            history=[],
            api_key=api_key,
        )

        if result and result.strip():
            return TestAIResponse(
                valid=True,
                message=(
                    "Gemini API connection successful."
                ),
            )

        return TestAIResponse(
            valid=False,
            message=(
                "Gemini returned an empty response."
            ),
        )

    except GeminiAuthenticationError as exc:
        logger.error(
            "[Settings] Gemini authentication test failed: %s",
            type(exc).__name__,
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Gemini authentication failed. "
                "Please check your API key."
            ),
        )

    except GeminiQuotaError:
        logger.warning(
            "[Settings] Gemini quota/rate limit reached."
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Gemini API rate limit reached. "
                "Please wait and try again later."
            ),
        )

    except ValueError as exc:
        logger.warning(
            "[Settings] Invalid Gemini test request: %s",
            type(exc).__name__,
        )

        return TestAIResponse(
            valid=False,
            message=(
                "The Gemini API key could not "
                "be validated."
            ),
        )

    except RuntimeError:
        logger.exception(
            "[Settings] Gemini service test failed."
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Gemini is currently unavailable. "
                "Please try again later."
            ),
        )

    except Exception:
        logger.exception(
            "[Settings] Unexpected Gemini test failure."
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Could not connect to Gemini. "
                "Please check the API key and try again."
            ),
        )


# ============================================================================
# TEST TAVILY CONNECTION
# ============================================================================


@router.post(
    "/test-tavily",
    response_model=TestAIResponse,
    status_code=status.HTTP_200_OK,
)
def test_tavily_connection(
    x_tavily_api_key: str | None = Header(
        default=None,
        alias="X-Tavily-API-Key",
    ),
) -> TestAIResponse:
    """
    Test the Tavily API key supplied by the current browser session.

    The API key:
        - comes from the request header
        - is never returned
        - is never logged
        - is never stored in the database
        - is never stored in application files
    """

    api_key = _get_api_key(
        x_tavily_api_key,
        provider="Tavily",
    )

    try:
        result = resource_search_service.test_connection(
            api_key=api_key,
        )

        if result:
            return TestAIResponse(
                valid=True,
                message=(
                    "Tavily API connection successful."
                ),
            )

        return TestAIResponse(
            valid=False,
            message=(
                "Tavily returned an unsuccessful response."
            ),
        )

    except ValueError as exc:
        logger.warning(
            "[Settings] Invalid Tavily test request: %s",
            type(exc).__name__,
        )

        return TestAIResponse(
            valid=False,
            message=(
                "The Tavily API key could not "
                "be validated."
            ),
        )

    except RuntimeError:
        logger.exception(
            "[Settings] Tavily service test failed."
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Tavily is currently unavailable. "
                "Please try again later."
            ),
        )

    except Exception:
        logger.exception(
            "[Settings] Unexpected Tavily test failure."
        )

        return TestAIResponse(
            valid=False,
            message=(
                "Could not connect to Tavily. "
                "Please check the API key and try again."
            ),
        )