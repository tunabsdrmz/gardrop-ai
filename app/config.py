from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

StylistProvider = Literal["ollama", "groq", "gemini", "anthropic"]

# Sensible default model for each provider. Override with STYLIST_MODEL.
DEFAULT_STYLIST_MODELS: dict[str, str] = {
    "ollama": "qwen2.5:7b",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-5-5",
}


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "gardrop-ai"
    version: str = "0.1.0"

    # CLIP model used by /tag. Downloaded on first use (about 600 MB).
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # Which LLM powers /stylist. Ollama runs locally and needs no key.
    stylist_provider: StylistProvider = "ollama"
    stylist_model: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None

    # If set, every request except /health must send it in the X-API-Key header.
    service_api_key: str | None = None

    max_upload_mb: int = 10

    @property
    def resolved_stylist_model(self) -> str:
        return self.stylist_model or DEFAULT_STYLIST_MODELS[self.stylist_provider]

    @property
    def stylist_configured(self) -> bool:
        """Ollama needs no key; the hosted providers do."""
        keys = {
            "ollama": "local",
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
        }
        return bool(keys[self.stylist_provider])


@lru_cache
def get_settings() -> Settings:
    return Settings()
