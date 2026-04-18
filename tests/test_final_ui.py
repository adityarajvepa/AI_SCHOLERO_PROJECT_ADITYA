"""Tests for the local dashboard and course list helper."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import repository_dep, settings_dep
from app.config import Settings
from app.main import create_app
from app.storage.repository import CourseRepository, reset_repository_for_tests


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCHOLERA_DATA_DIR", str(tmp_path))
    reset_repository_for_tests()
    settings = Settings()
    repo = CourseRepository(settings)
    app = create_app()
    app.dependency_overrides[repository_dep] = lambda: repo
    app.dependency_overrides[settings_dep] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_courses_empty(client: TestClient) -> None:
    r = client.get("/courses")
    assert r.status_code == 200
    assert r.json() == {"course_ids": []}


def test_list_courses_nonempty(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /courses reflects course directories on disk that contain chunks.json."""
    monkeypatch.setenv("SCHOLERA_DATA_DIR", str(tmp_path))
    reset_repository_for_tests()
    course_dir = tmp_path / "courses" / "demo101"
    course_dir.mkdir(parents=True)
    (course_dir / "chunks.json").write_text("[]", encoding="utf-8")
    np.save(course_dir / "embeddings.npy", np.zeros((0, 384), dtype=np.float32))

    settings = Settings()
    repo = CourseRepository(settings)
    app = create_app()
    app.dependency_overrides[repository_dep] = lambda: repo
    app.dependency_overrides[settings_dep] = lambda: settings
    try:
        c = TestClient(app)
        r = c.get("/courses")
        assert r.status_code == 200
        assert "demo101" in r.json()["course_ids"]
    finally:
        app.dependency_overrides.clear()


def test_final_ui_returns_html(client: TestClient) -> None:
    r = client.get("/finalUI")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    assert "Scholera" in body
    assert "/courses" in body or "fetch(" in body


def test_root_links_to_final_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("dashboard") == "/finalUI"
