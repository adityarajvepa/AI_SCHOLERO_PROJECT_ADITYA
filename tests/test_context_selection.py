"""Context re-ranking after diversified retrieval."""

from app.generation.context_select import select_synthesis_rows
from app.retrieval.hybrid import RetrievedRow
from app.schemas import DocumentChunk


def _r(
    cid: str,
    text: str,
    hybrid: float,
    *,
    source: str,
    unit: int,
) -> RetrievedRow:
    ch = DocumentChunk(
        chunk_id=cid,
        course_id="c",
        lecture_id="lec",
        source_file=source,
        unit_type="page",
        unit_number=unit,
        chunk_index=0,
        text=text,
    )
    return RetrievedRow(
        chunk=ch,
        hybrid_score=hybrid,
        dense_score=hybrid,
        bm25_score=0.5,
    )


def test_select_prefers_deprecation_slide_over_toc() -> None:
    diversified = [
        _r("c1", "CSS separates presentation from document structure.", 0.9, source="HTMLStyleSheets.pdf", unit=3),
        _r(
            "h2",
            "Table of contents and course schedule for the HTML module.",
            0.58,
            source="HTML.pdf",
            unit=2,
        ),
        _r(
            "h27",
            "Deprecated presentation attributes on BODY include bgcolor; use style sheets instead "
            "of mixing structure with presentation.",
            0.52,
            source="HTML.pdf",
            unit=27,
        ),
    ]
    q = "Why were style sheets introduced if HTML already had presentation-related features?"
    picked = select_synthesis_rows(diversified, q, is_cross_lecture=True, max_chunks=3)
    units = [r.chunk.unit_number for r in picked]
    assert 27 in units
    assert 2 not in units[:2]
