import io
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas import TagResponse
from app.services import tagging
from app.services.embedder import Embedder, get_embedder

router = APIRouter(tags=["tagging"])


# A plain `def` (not `async def`): FastAPI runs it in a worker thread,
# so the heavy CLIP work doesn't block other requests.
@router.post("/tag", response_model=TagResponse)
def tag_photo(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> TagResponse:
    """Detects category, pattern, season, formality and main colors of a clothing photo."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload an image file")

    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is larger than {settings.max_upload_mb} MB",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Could not read the image"
        ) from error

    return tagging.tag(image, embedder)
