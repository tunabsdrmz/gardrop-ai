import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings, get_settings
from app.main import app
from app.services.embedder import get_embedder
from app.services.stylist import get_stylist


class FakeEmbedder:
    """Stands in for CLIP. Prompts containing a target word point the same way as the image."""

    loaded = True

    def __init__(self, targets: set[str]):
        self.targets = targets

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        def vector(text: str) -> np.ndarray:
            if any(word in text for word in self.targets):
                return np.array([1.0, 0.0, 0.0, 0.0])
            return np.array([0.0, 1.0, 0.0, 0.0])

        return np.stack([vector(t) for t in texts])


class FakeAnthropic:
    """Replays scripted responses instead of calling the real API."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Copy the list: the stylist keeps appending to it after this call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def reply(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeOpenAI:
    """Same idea for OpenAI-compatible APIs (Ollama, Groq, Gemini)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def chat_reply(content=None, tool_calls=()):
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(id, name, arguments):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=arguments))


def png_bytes(color=(52, 101, 164), size=(64, 64), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder({"jeans", "solid", "winter", "casual everyday"})


@pytest.fixture
def client(fake_embedder):
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_settings] = lambda: Settings()
    app.dependency_overrides[get_stylist] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()
