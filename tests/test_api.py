"""API smoke tests."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import repository_dep, settings_dep
from app.config import Settings
from app.main import create_app
from app.storage.repository import CourseRepository


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCHOLERA_DATA_DIR", str(tmp_path))
    settings = Settings()
    repo = CourseRepository(settings)
    app = create_app()
    app.dependency_overrides[repository_dep] = lambda: repo
    app.dependency_overrides[settings_dep] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "Scholera AI Tutor"
    assert body["docs"] == "/docs"


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats_empty_course(client: TestClient) -> None:
    r = client.get("/courses/demo/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["course_id"] == "demo"
    assert body["num_chunks"] == 0


def test_ask_missing_course(client: TestClient) -> None:
    r = client.post("/courses/demo/ask", json={"question": "What is X?"})
    assert r.status_code == 404
