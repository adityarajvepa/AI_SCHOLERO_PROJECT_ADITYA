"""Course document ingestion endpoint."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import embedder_dep, repository_dep, settings_dep
from app.config import Settings
from app.ingestion.pipeline import ingest_files
from app.retrieval.embeddings import EmbeddingModel
from app.schemas import IngestionSummary
from app.storage.repository import CourseRepository

router = APIRouter(tags=["ingest"])


@router.post("/courses/{course_id}/ingest", response_model=IngestionSummary)
async def ingest_course(
    course_id: str,
    files: list[UploadFile] = File(...),
    repo: CourseRepository = Depends(repository_dep),
    embedder: EmbeddingModel = Depends(embedder_dep),
    settings: Settings = Depends(settings_dep),
) -> IngestionSummary:
    if not course_id.strip():
        raise HTTPException(status_code=400, detail="course_id must be non-empty")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    pairs: list[tuple[str, bytes]] = []
    for uf in files:
        if not uf.filename:
            continue
        data = await uf.read()
        pairs.append((uf.filename, data))
    if not pairs:
        raise HTTPException(status_code=400, detail="No usable filenames in upload")
    return ingest_files(course_id.strip(), pairs, repo, embedder, settings)
