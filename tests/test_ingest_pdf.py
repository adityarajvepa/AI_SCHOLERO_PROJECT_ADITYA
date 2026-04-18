"""End-to-end ingest test with a tiny PDF (no real embedding model)."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import embedder_dep, repository_dep, settings_dep
from app.config import Settings
from app.main import create_app
from app.storage.repository import CourseRepository


class _FakeEmbedder:
    dimension = 8

    def encode(self, texts: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.ones((len(texts), self.dimension), dtype=np.float32)


@pytest.fixture()
def ingest_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCHOLERA_DATA_DIR", str(tmp_path))
    settings = Settings()
    repo = CourseRepository(settings)
    app = create_app()
    app.dependency_overrides[repository_dep] = lambda: repo
    app.dependency_overrides[settings_dep] = lambda: settings
    app.dependency_overrides[embedder_dep] = lambda: _FakeEmbedder()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _tiny_pdf_bytes(tmp_dir: Path) -> bytes:
    path = tmp_dir / "_gen.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Scholera test: gradients and backpropagation basics.")
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def test_ingest_and_stats(ingest_client: TestClient, tmp_path: Path) -> None:
    files = {"files": ("unit1_notes.pdf", _tiny_pdf_bytes(tmp_path), "application/pdf")}
    r = ingest_client.post("/courses/ml101/ingest", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["course_id"] == "ml101"
    assert body["documents_ingested"] == 1
    assert body["total_chunks_created"] >= 1

    s = ingest_client.get("/courses/ml101/stats")
    assert s.status_code == 200
    stats = s.json()
    assert stats["num_chunks"] >= 1
    assert any("unit1_notes.pdf" in f for f in stats["indexed_files"])

    ask = ingest_client.post(
        "/courses/ml101/ask",
        json={"question": "What is covered about gradients?", "top_k": 3},
    )
    assert ask.status_code == 200
    payload = ask.json()
    assert "answer" in payload
    assert "citations" in payload
    assert payload["debug"]["retrieved_chunks"]
