"""FastAPI entrypoint for Scholera AI Tutor."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import courses, final_ui, health, ingest, quiz, stats, tutor
from app.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Scholera AI Tutor",
        version="1.0.0",
        description="Grounded course Q&A with hybrid retrieval over local PDF/PPTX materials.",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Avoid a bare 404 when opening the server URL in a browser."""
        return {
            "service": "Scholera AI Tutor",
            "docs": "/docs",
            "health": "/health",
            "dashboard": "/finalUI",
        }

    application.include_router(final_ui.router)
    application.include_router(health.router)
    application.include_router(courses.router)
    application.include_router(ingest.router)
    application.include_router(tutor.router)
    application.include_router(quiz.router)
    application.include_router(stats.router)
    return application


app = create_app()
