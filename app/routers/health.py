from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import HealthResponse
from app.services.embedder import Embedder, get_embedder

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> HealthResponse:
    """Liveness check. Also tells whether the stylist is configured and CLIP is warm."""
    return HealthResponse(
        status="ok",
        version=settings.version,
        stylist_enabled=settings.anthropic_api_key is not None,
        clip_loaded=embedder.loaded,
    )
