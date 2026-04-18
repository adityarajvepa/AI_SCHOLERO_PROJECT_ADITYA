"""FastAPI dependency providers."""

from app.config import Settings, get_settings
from app.retrieval.embeddings import EmbeddingModel, get_embedding_model
from app.storage.repository import CourseRepository, get_repository


def settings_dep() -> Settings:
    return get_settings()


def repository_dep() -> CourseRepository:
    return get_repository()


def embedder_dep() -> EmbeddingModel:
    return get_embedding_model()
