"""AI Tutor question endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import embedder_dep, repository_dep, settings_dep
from app.config import Settings
from app.generation.tutor import answer_question
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.hybrid import hybrid_search
from app.retrieval.query_plan import (
    cross_lecture_pool_size,
    diversify_by_source_file,
    is_cross_lecture_question,
)
from app.schemas import AskRequest, AskResponse
from app.storage.repository import CourseRepository

router = APIRouter(tags=["tutor"])


@router.post("/courses/{course_id}/ask", response_model=AskResponse)
def ask_course(
    course_id: str,
    body: AskRequest,
    repo: CourseRepository = Depends(repository_dep),
    embedder: EmbeddingModel = Depends(embedder_dep),
    settings: Settings = Depends(settings_dep),
) -> AskResponse:
    if not course_id.strip():
        raise HTTPException(status_code=400, detail="course_id must be non-empty")
    chunks = repo.get_chunks(course_id.strip())
    matrix = repo.get_embeddings(course_id.strip())
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed material for this course. Ingest documents first.",
        )
    if matrix.shape[0] != len(chunks):
        raise HTTPException(
            status_code=500,
            detail="Index is inconsistent (embeddings vs chunks). Re-ingest materials.",
        )
    q_emb = embedder.encode([body.question])[0]
    w_d, w_b = settings.hybrid_weights
    cross = is_cross_lecture_question(body.question)
    pool_k = cross_lecture_pool_size(body.top_k) if cross else body.top_k
    rows = hybrid_search(
        chunks,
        matrix,
        q_emb,
        body.question,
        w_d,
        w_b,
        pool_k,
    )
    if cross:
        rows = diversify_by_source_file(rows, body.top_k)
    else:
        rows = rows[: body.top_k]
    return answer_question(
        body.question,
        rows,
        settings,
        query_embedding=q_emb,
        embedder=embedder,
        is_cross_lecture=cross,
    )
