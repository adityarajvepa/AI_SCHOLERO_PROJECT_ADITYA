"""Local demo dashboard HTML."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_DASHBOARD_HTML = _UI_DIR / "final_dashboard.html"


@router.get("/finalUI", response_class=HTMLResponse)
def final_ui_dashboard() -> HTMLResponse:
    """Single-page dashboard: ingest, stats, and ask (uses existing JSON APIs)."""
    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
