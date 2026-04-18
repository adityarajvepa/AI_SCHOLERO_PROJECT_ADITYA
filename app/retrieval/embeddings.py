"""Local sentence-transformers embedding wrapper."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings


class EmbeddingModel:
    """Lazy-loaded embedding model with deterministic encode."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    def _ensure_model(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                self._model = SentenceTransformer(self.model_name)
            return self._model

    @property
    def dimension(self) -> int:
        m = self._ensure_model()
        return int(m.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 matrix (n, dim) L2-normalized rows for stable cosine search."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        m = self._ensure_model()
        emb = m.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)


_embed_singleton: EmbeddingModel | None = None
_embed_lock = threading.Lock()


def get_embedding_model(settings: Settings | None = None) -> EmbeddingModel:
    global _embed_singleton
    if _embed_singleton is None:
        with _embed_lock:
            if _embed_singleton is None:
                _embed_singleton = EmbeddingModel(settings)
    return _embed_singleton


def reset_embedding_model_for_tests() -> None:
    global _embed_singleton
    _embed_singleton = None
