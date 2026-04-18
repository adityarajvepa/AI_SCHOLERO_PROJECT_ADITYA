"""Course corpus statistics."""

from fastapi import APIRouter, Depends

from app.api.deps import repository_dep
from app.schemas import CourseStats
from app.storage.repository import CourseRepository

router = APIRouter(tags=["stats"])


@router.get("/courses/{course_id}/stats", response_model=CourseStats)
def course_stats(
    course_id: str,
    repo: CourseRepository = Depends(repository_dep),
) -> CourseStats:
    cid = course_id.strip()
    payload = repo.stats_payload(cid)
    return CourseStats(
        course_id=cid,
        num_documents=payload["num_documents"],
        num_chunks=payload["num_chunks"],
        lectures=payload["lectures"],
        indexed_files=payload["indexed_files"],
        recent_ingestion=repo.get_summary(cid),
    )
