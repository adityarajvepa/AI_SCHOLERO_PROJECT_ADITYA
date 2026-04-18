"""Pytest fixtures."""

import pytest

from app.retrieval.embeddings import reset_embedding_model_for_tests
from app.storage.repository import reset_repository_for_tests


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    reset_repository_for_tests()
    reset_embedding_model_for_tests()
    yield
    reset_repository_for_tests()
    reset_embedding_model_for_tests()
