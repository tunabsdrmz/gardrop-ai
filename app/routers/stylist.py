import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.schemas import StylistRequest, StylistResponse
from app.services.stylist import get_anthropic_client, run_stylist

router = APIRouter(tags=["stylist"])
logger = logging.getLogger(__name__)


@router.post("/stylist", response_model=StylistResponse)
def stylist(
    request: StylistRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[Any, Depends(get_anthropic_client)],
) -> StylistResponse:
    """Chat with the AI stylist. It reads the closet and weather you send and suggests outfits."""
    if client is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stylist is not configured")
    try:
        return run_stylist(request, client, settings.stylist_model)
    except Exception as error:
        logger.exception("Stylist request failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="The stylist is unavailable"
        ) from error
