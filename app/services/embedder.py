"""CLIP turns images and sentences into vectors in the same space.

If an image's vector points in the same direction as the vector of
"a photo of jeans", the image probably shows jeans. That is all /tag needs.
"""

import threading
from functools import lru_cache
from typing import Protocol

import numpy as np
from PIL import Image

from app.config import get_settings


class Embedder(Protocol):
    """Anything that can embed images and texts. Tests use a fake one."""

    @property
    def loaded(self) -> bool: ...

    def embed_image(self, image: Image.Image) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class ClipEmbedder:
    """Real CLIP model via open_clip. Loads lazily on first use, then stays in memory."""

    def __init__(self, model_name: str, pretrained: str) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._text_cache: dict[str, np.ndarray] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        # Imported here so the app (and the tests) start without loading PyTorch.
        with self._lock:
            if self._model is not None:
                return
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model.eval()
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._preprocess = preprocess
            self._model = model

    def embed_image(self, image: Image.Image) -> np.ndarray:
        import torch

        self._load()
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return _normalize(features[0].numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            with torch.no_grad():
                features = self._model.encode_text(self._tokenizer(missing))
            for text, vector in zip(missing, features.numpy(), strict=True):
                self._text_cache[text] = _normalize(vector)
        return np.stack([self._text_cache[t] for t in texts])


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return ClipEmbedder(settings.clip_model, settings.clip_pretrained)
