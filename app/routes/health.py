"""Health check route."""

from fastapi import APIRouter


router = APIRouter()


@router.get(
    "/health",
    tags=["Health"],
)
def health_check():
    """Basic liveness check."""

    return {
        "status": "ok",
        "service": "BodhaQ API",
    }