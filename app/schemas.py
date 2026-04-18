"""Pydantic request/response and domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A retrievable text segment derived from course materials."""

    chunk_id: str
    course_id: str
    lecture_id: str
    source_file: str
    unit_type: str
    unit_number: int
    chunk_index: int
    text: str
    has_formula: bool = False
    is_sparse_or_image_heavy: bool = False
    tokens_estimate: int | None = None


class IngestionSummary(BaseModel):
    """Returned after a successful ingest operation."""

    course_id: str
    documents_ingested: int
    total_units_processed: int
    total_chunks_created: int
    formula_heavy_units: int
    sparse_or_image_heavy_units: int
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """Reference to an original course unit."""

    source_file: str
    unit_type: str
    unit_number: int


class AskRequest(BaseModel):
    """Student question for the tutor."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class DebugRetrievedChunk(BaseModel):
    """One row in retrieval debug output."""

    chunk_id: str
    source_file: str
    unit_type: str
    unit_number: int
    hybrid_score: float
    dense_score: float
    bm25_score: float
    preview: str


class AskDebug(BaseModel):
    """Structured retrieval diagnostics."""

    retrieved_chunks: list[DebugRetrievedChunk]


class AskResponse(BaseModel):
    """Tutor answer with citations and optional debug info."""

    answer: str
    citations: list[Citation]
    debug: AskDebug


class CourseStats(BaseModel):
    """Aggregate corpus and ingestion metadata for a course."""

    course_id: str
    num_documents: int
    num_chunks: int
    lectures: list[str]
    indexed_files: list[str]
    recent_ingestion: IngestionSummary | None = None


class CourseListResponse(BaseModel):
    """All course identifiers that currently have indexed chunks on disk."""

    course_ids: list[str]


QuizDifficulty = Literal["easy", "medium", "hard"]


class QuizRequest(BaseModel):
    """Generate a grounded quiz from already-indexed course chunks."""

    num_questions: int = Field(default=5, ge=1, le=20)
    topic: str | None = Field(default=None, max_length=500)
    difficulty: QuizDifficulty = "medium"
    top_k: int = Field(default=12, ge=3, le=30)


class QuizCitationItem(BaseModel):
    """One source line tied to retrieved material (page or slide, not both required in input)."""

    source_file: str
    page: int | None = None
    slide: int | None = None


class QuizItem(BaseModel):
    """Single quiz question with answer key and grounded citations."""

    question: str
    answer_key: str
    citations: list[QuizCitationItem] = Field(default_factory=list)


class QuizResponse(BaseModel):
    """Structured quiz output; citations must match retrieval context only."""

    course_id: str
    topic: str | None = None
    difficulty: str
    quiz: list[QuizItem] = Field(default_factory=list)
    message: str | None = Field(
        default=None,
        description="Optional note (e.g. fewer questions than requested, or configuration hint).",
    )
