import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import StylistRequest, StylistResponse
from app.services.stylist import Stylist, get_stylist

router = APIRouter(tags=["stylist"])
logger = logging.getLogger(__name__)


@router.post("/stylist", response_model=StylistResponse)
def stylist(
    request: StylistRequest,
    stylist: Annotated[Stylist | None, Depends(get_stylist)],
) -> StylistResponse:
    """Chat with the AI stylist. It reads the closet and weather you send and suggests outfits."""
    if stylist is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stylist is not configured: add the API key for STYLIST_PROVIDER",
        )
    try:
        return stylist.run(request)
    except Exception as error:
        logger.exception("Stylist request failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="The stylist is unavailable"
        ) from error
