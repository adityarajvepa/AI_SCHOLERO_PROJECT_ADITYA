"""Grounded quiz generation over indexed course materials."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import embedder_dep, repository_dep, settings_dep
from app.config import Settings
from app.generation.quiz import generate_quiz_response, retrieve_quiz_context
from app.retrieval.embeddings import EmbeddingModel
from app.schemas import QuizRequest, QuizResponse
from app.storage.repository import CourseRepository

router = APIRouter(tags=["quiz"])

_TOPIC_WEAK_THRESHOLD = 0.06


@router.post("/courses/{course_id}/quiz", response_model=QuizResponse)
def generate_course_quiz(
    course_id: str,
    body: QuizRequest,
    repo: CourseRepository = Depends(repository_dep),
    embedder: EmbeddingModel = Depends(embedder_dep),
    settings: Settings = Depends(settings_dep),
) -> QuizResponse:
    cid = course_id.strip()
    if not cid:
        raise HTTPException(status_code=400, detail="course_id must be non-empty")
    chunks = repo.get_chunks(cid)
    matrix = repo.get_embeddings(cid)
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

    rows = retrieve_quiz_context(
        chunks,
        matrix,
        embedder,
        settings,
        topic=body.topic,
        top_k=body.top_k,
    )
    if body.topic and body.topic.strip() and rows and float(rows[0].hybrid_score) < _TOPIC_WEAK_THRESHOLD:
        return QuizResponse(
            course_id=cid,
            topic=body.topic.strip(),
            difficulty=body.difficulty,
            quiz=[],
            message=(
                "Retrieval found very weak matches for this topic. "
                "Try different keywords, raise top_k, or omit topic for diversified course coverage."
            ),
        )

    return generate_quiz_response(
        cid,
        rows,
        settings,
        topic=body.topic.strip() if body.topic else None,
        difficulty=body.difficulty,
        num_questions=body.num_questions,
    )
