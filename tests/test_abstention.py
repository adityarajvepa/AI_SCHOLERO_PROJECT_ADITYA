"""Keyword-coverage abstention vs normal HTML answers."""

from app.generation.tutor import answer_question
from app.retrieval.hybrid import RetrievedRow
from app.schemas import DocumentChunk


def _row(
    cid: str,
    text: str,
    hybrid: float,
    *,
    source: str = "notes.pdf",
    unit: int = 1,
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


def test_react_hooks_question_abstains_on_generic_js_hits() -> None:
    rows = [
        _row(
            "j1",
            "JavaScript functions use var and let declarations and blocks scope rules.",
            0.41,
            source="javascript_basics.pdf",
        ),
        _row(
            "j2",
            "A for loop iterates with an index variable and uses the document object.",
            0.38,
            source="javascript_basics.pdf",
        ),
        _row(
            "j3",
            "Callbacks and event listeners attach behavior to browser events.",
            0.35,
            source="javascript_basics.pdf",
        ),
    ]
    q = "How do React hooks interact with the virtual DOM in modern apps?"
    resp = answer_question(q, rows, is_cross_lecture=False)
    assert "do not appear to substantively" in resp.answer
    assert "no clear extractable sentences" not in resp.answer.lower()
    assert resp.debug.retrieved_chunks
    assert resp.citations == []
    assert "Reviewed sources" in resp.answer


def test_html_presentation_question_not_abstained() -> None:
    rows = [
        _row(
            "h1",
            "Deprecated presentation attributes on BODY include bgcolor, text, and link.",
            0.52,
            source="HTML.pdf",
        ),
        _row(
            "h2",
            "Presentation was mixed into markup before style sheets separated concerns.",
            0.48,
            source="HTML.pdf",
        ),
    ]
    q = "What are deprecated HTML presentation attributes like bgcolor on the body element?"
    resp = answer_question(q, rows, is_cross_lecture=False)
    assert "do not appear to substantively" not in resp.answer
    assert len(resp.answer) > 40


def test_css_vs_html_filenames_both_represented_after_diversify() -> None:
    """Regression for comparison questions: both HTMLStyleSheets.pdf and HTML.pdf in context."""
    from app.retrieval.query_plan import diversify_by_source_file

    def mk(cid: str, text: str, hybrid: float, source: str) -> RetrievedRow:
        return _row(cid, text, hybrid, source=source)

    css = [
        mk(f"c{i}", "High CSS chunk " + str(i), 0.91 - i * 0.01, "HTMLStyleSheets.pdf")
        for i in range(8)
    ]
    html = [mk("h1", "HTML body bgcolor presentation attributes.", 0.46, "HTML.pdf")]
    out = diversify_by_source_file(css + html, final_k=6)
    names = {r.chunk.source_file for r in out}
    assert "HTMLStyleSheets.pdf" in names
    assert "HTML.pdf" in names


def test_cross_lecture_style_question_still_answers() -> None:
    rows = [
        _row(
            "s3",
            "Style sheets separate presentation from markup content using CSS rules.",
            0.56,
            source="HTMLStyleSheets.pdf",
            unit=3,
        ),
        _row(
            "c1",
            "Style sheets were introduced so presentation rules stay separate from HTML structure "
            "and related presentation features.",
            0.55,
            source="HTMLStyleSheets.pdf",
            unit=10,
        ),
        _row(
            "c2",
            "CSS keeps styling concerns out of the content markup for maintainability.",
            0.52,
            source="HTMLStyleSheets.pdf",
            unit=11,
        ),
        _row(
            "h_intro",
            "Table of contents and course schedule for the HTML module.",
            0.54,
            source="HTML.pdf",
            unit=2,
        ),
        _row(
            "h27",
            "Deprecated presentation attributes on BODY include bgcolor and text; use style sheets "
            "instead of mixing structure with inline presentation.",
            0.5,
            source="HTML.pdf",
            unit=27,
        ),
        _row(
            "h1",
            "HTML already offered presentational attributes such as BODY bgcolor and text.",
            0.48,
            source="HTML.pdf",
            unit=5,
        ),
    ]
    q = "Why were style sheets introduced if HTML already had presentation-related features?"
    resp = answer_question(q, rows, is_cross_lecture=True)
    assert "do not appear to substantively" not in resp.answer
    assert "no clear extractable sentences" not in resp.answer.lower()
    assert len(resp.answer) > 40
    assert resp.citations, "comparison answers must return non-empty citations"
    assert any(c.source_file == "HTML.pdf" and c.unit_number == 27 for c in resp.citations), (
        "Contrast-aware context selection should surface the deprecated styling slide (p.27)."
    )
    assert any(
        c.source_file == "HTMLStyleSheets.pdf" and c.unit_number == 3 for c in resp.citations
    )
    if "Sources:" in resp.answer:
        tail = resp.answer.split("Sources:", 1)[-1].strip()
        assert tail and len(tail) > 5, "Sources section should list files or units after the header"
        assert not resp.answer.rstrip().endswith("Sources:")
    assert not any(c.unit_number == 2 for c in resp.citations)
