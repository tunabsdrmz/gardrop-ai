from fastapi import Depends, FastAPI

from app.config import get_settings
from app.routers import health, match, stylist, tag
from app.security import require_api_key

settings = get_settings()

app = FastAPI(
    title="Gardrop AI",
    version=settings.version,
    description="Clothing tagging, closet matching and an AI stylist for the Gardrop iOS app.",
)

# /health stays public so uptime checks work; everything else may need an API key.
app.include_router(health.router)
protected = [Depends(require_api_key)]
app.include_router(tag.router, dependencies=protected)
app.include_router(match.router, dependencies=protected)
app.include_router(stylist.router, dependencies=protected)
