"""Cross-lecture detection and source diversification."""

from app.retrieval.hybrid import RetrievedRow
from app.retrieval.query_plan import (
    cross_lecture_pool_size,
    diversify_by_source_file,
    is_cross_lecture_question,
)
from app.schemas import DocumentChunk


def _mk_row(
    cid: str,
    source: str,
    text: str,
    hybrid: float,
) -> RetrievedRow:
    ch = DocumentChunk(
        chunk_id=cid,
        course_id="c",
        lecture_id="lec",
        source_file=source,
        unit_type="page",
        unit_number=1,
        chunk_index=0,
        text=text,
    )
    return RetrievedRow(
        chunk=ch,
        hybrid_score=hybrid,
        dense_score=hybrid,
        bm25_score=0.5,
    )


def test_cross_question_markers() -> None:
    q = "Why were style sheets introduced if HTML already had presentation-related features?"
    assert is_cross_lecture_question(q) is True
    assert is_cross_lecture_question("List the syntax for a for-loop in JavaScript.") is False


def test_cross_pool_size() -> None:
    assert cross_lecture_pool_size(8) == 16
    assert cross_lecture_pool_size(4) == 15


def test_diversify_round_robin_includes_second_source() -> None:
    """Simulate CSS-heavy hybrid ordering; diversification should still surface HTML.pdf."""
    css = [
        _mk_row(f"c{i}", "HTMLStyleSheets.pdf", f"Separate CSS concerns from markup chunk {i}.", 0.92 - i * 0.01)
        for i in range(6)
    ]
    html = [
        _mk_row("h1", "HTML.pdf", "BODY bgcolor text and link attributes were presentational.", 0.44),
        _mk_row("h2", "HTML.pdf", "Old HTML mixed structure with inline presentation.", 0.42),
    ]
    candidates = css + html
    out = diversify_by_source_file(candidates, final_k=6)
    files = {r.chunk.source_file for r in out}
    assert "HTML.pdf" in files
    assert "HTMLStyleSheets.pdf" in files
    assert len(out) == 6


def test_single_file_corpus_no_crash() -> None:
    rows = [_mk_row(f"x{i}", "only.pdf", f"chunk {i}", 0.9 - i * 0.01) for i in range(5)]
    out = diversify_by_source_file(rows, 3)
    assert len(out) == 3
    assert all(r.chunk.source_file == "only.pdf" for r in out)
