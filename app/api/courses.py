"""Course inventory (list) endpoint for dashboards and tooling."""

from fastapi import APIRouter, Depends

from app.api.deps import repository_dep
from app.schemas import CourseListResponse
from app.storage.repository import CourseRepository

router = APIRouter(tags=["courses"])


@router.get("/courses", response_model=CourseListResponse)
def list_courses(repo: CourseRepository = Depends(repository_dep)) -> CourseListResponse:
    """Return every ``course_id`` that has a loaded index (at least ``chunks.json`` present)."""
    return CourseListResponse(course_ids=repo.list_course_ids())
