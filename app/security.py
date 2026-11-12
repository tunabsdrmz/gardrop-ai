import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Blocks requests without the right X-API-Key, when a key is configured.

    The stylist calls a paid API, so a public deployment should always set one.
    """
    expected = settings.service_api_key
    # An empty line like `SERVICE_API_KEY=` in .env arrives as "", so treat it as "not set".
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
