from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "gardrop-ai"
    version: str = "0.1.0"

    # CLIP model used by /tag. Downloaded on first use (about 600 MB).
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # Claude, used by /stylist. Without a key the stylist endpoint returns 503.
    anthropic_api_key: str | None = None
    stylist_model: str = "claude-haiku-5-5"

    # If set, every request except /health must send it in the X-API-Key header.
    service_api_key: str | None = None

    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
