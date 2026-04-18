"""Orchestrates file parsing, heuristics, chunking, and embedding for ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import Settings
from app.ingestion.chunker import chunk_unit_text
from app.ingestion.detectors import detect_formula_heavy, detect_sparse_or_image_heavy
from app.ingestion.loaders import TextUnit, load_pdf_units, load_pptx_units
from app.schemas import DocumentChunk, IngestionSummary
from app.retrieval.embeddings import EmbeddingModel
from app.storage.repository import CourseRepository
from app.utils.ids import make_chunk_id, slugify_filename_stem
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _detect_file_type(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".pptx"):
        return "pptx"
    return None


def _load_units(path: Path, kind: str) -> list[TextUnit]:
    if kind == "pdf":
        return load_pdf_units(path)
    if kind == "pptx":
        return load_pptx_units(path)
    return []


def ingest_files(
    course_id: str,
    files: list[tuple[str, bytes]],
    repo: CourseRepository,
    embedder: EmbeddingModel,
    settings: Settings,
) -> IngestionSummary:
    """
    Ingest uploaded files into the course index.

    Re-ingesting the same filename (lecture_id) replaces prior chunks for that lecture.
    """
    warnings: list[str] = []
    documents_ingested = 0
    total_units = 0
    total_chunks = 0
    formula_units = 0
    sparse_units = 0

    all_new_chunks: list[DocumentChunk] = []
    texts_for_embed: list[str] = []

    for original_name, content in files:
        safe_name = Path(original_name).name
        kind = _detect_file_type(safe_name)
        if kind is None:
            warnings.append(f"Skipped unsupported file: {safe_name}")
            continue
        path = Path(safe_name)
        lecture_id = slugify_filename_stem(path.stem)
        removed = repo.delete_lecture_chunks(course_id, lecture_id)
        if removed:
            logger.info(
                "Replaced %s existing chunks for course=%s lecture=%s",
                removed,
                course_id,
                lecture_id,
            )

        tmp_path = settings.scholera_data_dir / "_tmp_uploads" / course_id / safe_name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(content)
        units: list[TextUnit]
        try:
            units = _load_units(tmp_path, kind)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to parse {safe_name}: {exc}")
            units = []
        finally:
            tmp_path.unlink(missing_ok=True)

        if not units:
            continue

        documents_ingested += 1
        for unit in units:
            total_units += 1
            unit_text = unit.text or ""
            has_formula = detect_formula_heavy(unit_text)
            is_sparse = detect_sparse_or_image_heavy(unit_text)
            if has_formula:
                formula_units += 1
            if is_sparse:
                sparse_units += 1

            piece_texts = chunk_unit_text(unit_text)
            if not piece_texts and unit_text.strip():
                piece_texts = [unit_text.strip()]

            for idx, chunk_text in enumerate(piece_texts):
                if not chunk_text.strip():
                    continue
                chunk_id = make_chunk_id(
                    course_id,
                    lecture_id,
                    unit.unit_type,
                    unit.unit_number,
                    idx,
                )
                est_tokens = max(1, len(chunk_text) // 4)
                ch = DocumentChunk(
                    chunk_id=chunk_id,
                    course_id=course_id,
                    lecture_id=lecture_id,
                    source_file=path.name,
                    unit_type=unit.unit_type,
                    unit_number=unit.unit_number,
                    chunk_index=idx,
                    text=chunk_text,
                    has_formula=has_formula,
                    is_sparse_or_image_heavy=is_sparse,
                    tokens_estimate=est_tokens,
                )
                all_new_chunks.append(ch)
                texts_for_embed.append(chunk_text)
                total_chunks += 1

    if not all_new_chunks:
        return IngestionSummary(
            course_id=course_id,
            documents_ingested=0,
            total_units_processed=total_units,
            total_chunks_created=0,
            formula_heavy_units=formula_units,
            sparse_or_image_heavy_units=sparse_units,
            warnings=warnings or ["No chunks produced; check file types and content."],
        )

    embeddings = embedder.encode(texts_for_embed)
    summary = IngestionSummary(
        course_id=course_id,
        documents_ingested=documents_ingested,
        total_units_processed=total_units,
        total_chunks_created=total_chunks,
        formula_heavy_units=formula_units,
        sparse_or_image_heavy_units=sparse_units,
        warnings=warnings,
    )
    repo.append_chunks(course_id, all_new_chunks, embeddings.astype(np.float32), summary)
    return summary
