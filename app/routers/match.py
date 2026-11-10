from fastapi import APIRouter

from app.schemas import MatchRequest, MatchResponse
from app.services import matching

router = APIRouter(tags=["matching"])


@router.post("/match", response_model=MatchResponse)
def match_item(request: MatchRequest) -> MatchResponse:
    """'Will this go with my closet?' Scores a new piece against the user's wardrobe."""
    return matching.match(request)
