"""Local disk persistence for chunks, embeddings, and ingestion summaries."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.schemas import DocumentChunk, IngestionSummary
from app.utils.logging import get_logger

logger = get_logger(__name__)

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
SUMMARY_FILE = "last_ingestion_summary.json"


class CourseRepository:
    """Per-course JSON + numpy persistence with in-memory cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = RLock()
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._summaries: dict[str, IngestionSummary] = {}
        self._load_all_from_disk()

    def _course_dir(self, course_id: str) -> Path:
        safe = course_id.replace("/", "_").replace("..", "_")
        return self._settings.courses_dir / safe

    def _load_all_from_disk(self) -> None:
        base = self._settings.courses_dir
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            return
        for p in base.iterdir():
            if not p.is_dir():
                continue
            cid = p.name
            try:
                self._load_course(cid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed loading course %s: %s", cid, exc)

    def _load_course(self, course_id: str) -> None:
        d = self._course_dir(course_id)
        cf = d / CHUNKS_FILE
        ef = d / EMBEDDINGS_FILE
        if not cf.exists():
            return
        raw = json.loads(cf.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.model_validate(x) for x in raw]
        emb = np.load(ef) if ef.exists() else np.zeros((0, 384), dtype=np.float32)
        if emb.shape[0] != len(chunks):
            logger.warning(
                "Embedding row mismatch for %s: %s vs %s chunks",
                course_id,
                emb.shape[0],
                len(chunks),
            )
        self._chunks[course_id] = chunks
        self._embeddings[course_id] = emb.astype(np.float32, copy=False)
        sf = d / SUMMARY_FILE
        if sf.exists():
            self._summaries[course_id] = IngestionSummary.model_validate(
                json.loads(sf.read_text(encoding="utf-8"))
            )

    def _save_course(self, course_id: str) -> None:
        d = self._course_dir(course_id)
        d.mkdir(parents=True, exist_ok=True)
        chunks = self._chunks.get(course_id, [])
        emb = self._embeddings.get(course_id, np.zeros((0, 384), dtype=np.float32))
        (d / CHUNKS_FILE).write_text(
            json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(d / EMBEDDINGS_FILE, emb)
        if course_id in self._summaries:
            (d / SUMMARY_FILE).write_text(
                self._summaries[course_id].model_dump_json(indent=2),
                encoding="utf-8",
            )

    def list_course_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._chunks.keys())

    def get_chunks(self, course_id: str) -> list[DocumentChunk]:
        with self._lock:
            return list(self._chunks.get(course_id, []))

    def get_embeddings(self, course_id: str) -> np.ndarray:
        with self._lock:
            return self._embeddings.get(course_id, np.zeros((0, 384), dtype=np.float32)).copy()

    def get_summary(self, course_id: str) -> IngestionSummary | None:
        with self._lock:
            return self._summaries.get(course_id)

    def delete_lecture_chunks(self, course_id: str, lecture_id: str) -> int:
        """Remove all chunks for a lecture_id. Returns number removed."""
        with self._lock:
            chunks = self._chunks.get(course_id, [])
            if not chunks:
                return 0
            keep: list[DocumentChunk] = []
            keep_idx: list[int] = []
            removed = 0
            for i, c in enumerate(chunks):
                if c.lecture_id == lecture_id:
                    removed += 1
                else:
                    keep.append(c)
                    keep_idx.append(i)
            if removed == 0:
                return 0
            emb = self._embeddings.get(course_id)
            dim = 384
            if emb is not None and emb.ndim == 2 and emb.shape[1] > 0:
                dim = int(emb.shape[1])
            if emb is not None and emb.shape[0] == len(chunks):
                self._embeddings[course_id] = emb[np.array(keep_idx, dtype=np.int64)]
            else:
                self._embeddings[course_id] = np.zeros((0, dim), dtype=np.float32)
            self._chunks[course_id] = keep
            self._save_course(course_id)
            return removed

    def append_chunks(
        self,
        course_id: str,
        new_chunks: list[DocumentChunk],
        new_embeddings: np.ndarray,
        summary: IngestionSummary,
    ) -> None:
        with self._lock:
            existing = self._chunks.get(course_id, [])
            existing_emb = self._embeddings.get(
                course_id, np.zeros((0, new_embeddings.shape[1]), dtype=np.float32)
            )
            if existing_emb.shape[0] != len(existing):
                existing_emb = np.zeros((len(existing), new_embeddings.shape[1]), dtype=np.float32)
            merged_chunks = existing + new_chunks
            merged_emb = (
                np.vstack([existing_emb, new_embeddings])
                if len(existing)
                else new_embeddings.astype(np.float32, copy=False)
            )
            self._chunks[course_id] = merged_chunks
            self._embeddings[course_id] = merged_emb.astype(np.float32, copy=False)
            self._summaries[course_id] = summary
            self._save_course(course_id)

    def replace_course_vectors(
        self,
        course_id: str,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        summary: IngestionSummary,
    ) -> None:
        """Full replace (used if we need consistency repair)."""
        with self._lock:
            self._chunks[course_id] = chunks
            self._embeddings[course_id] = embeddings.astype(np.float32, copy=False)
            self._summaries[course_id] = summary
            self._save_course(course_id)

    def stats_payload(self, course_id: str) -> dict[str, Any]:
        chunks = self.get_chunks(course_id)
        files = sorted({c.source_file for c in chunks})
        lectures = sorted({c.lecture_id for c in chunks})
        return {
            "num_documents": len(files),
            "num_chunks": len(chunks),
            "lectures": lectures,
            "indexed_files": files,
        }


_singleton: CourseRepository | None = None


def get_repository(settings: Settings | None = None) -> CourseRepository:
    global _singleton
    if _singleton is None:
        _singleton = CourseRepository(settings)
    return _singleton


def reset_repository_for_tests() -> None:
    global _singleton
    _singleton = None
