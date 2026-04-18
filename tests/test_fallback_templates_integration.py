"""Integration-style tests for template-driven fallback answers."""

from app.generation.fallback_templates import clean_corpus_text
from app.generation.tutor import synthesize_fallback_answer
from app.retrieval.hybrid import RetrievedRow
from app.schemas import Citation, DocumentChunk


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


def test_css_three_ways_concise_numbered_answer() -> None:
    chunk = """
    Copyright © Example University 2024
    Slide 4: Including CSS
    1. Inline style using the style attribute on an element.
    2. Internal CSS inside a <style> element in the head.
    3. External CSS linked with <link rel="stylesheet" href="site.css">.
    """
    rows = [_r("c1", chunk, 0.9, source="HTMLStyleSheets.pdf", unit=3)]
    cites = [
        Citation(source_file="HTMLStyleSheets.pdf", unit_type="page", unit_number=3),
    ]
    ans, cite_ov = synthesize_fallback_answer(
        "What are the three ways CSS can be included in HTML?",
        rows,
        None,
        None,
        cites,
        question_kind="enumeration",
    )
    assert cite_ov is not None
    assert "inline" in ans.lower() or "style attribute" in ans.lower()
    assert "<style>" in ans or "style>" in ans
    assert "link" in ans.lower()
    assert "Copyright" not in ans
    assert ans.split()[0].lower() != "copyright"
    assert "Complementary material in" not in ans
    assert "no clear extractable sentences" not in ans.lower()


def test_css_three_ways_uses_canonical_when_chunk_has_no_numbered_list() -> None:
    """Regression: strong hit with no parseable list must still answer (canonical), not threshold."""
    # No numbered lines and no sentence ≥18 chars so enumeration cannot stitch a fake list.
    chunk = "See deck.\nMore soon.\nOK.\n"
    rows = [_r("c1", chunk, 0.88, source="HTMLStyleSheets.pdf", unit=6)]
    cites = [Citation(source_file="HTMLStyleSheets.pdf", unit_type="page", unit_number=6)]
    ans, cite_ov = synthesize_fallback_answer(
        "What are the three ways CSS can be included in HTML?",
        rows,
        None,
        None,
        cites,
        question_kind="enumeration",
    )
    assert cite_ov is not None
    assert "no clear extractable sentences" not in ans.lower()
    assert "1." in ans and "2." in ans and "3." in ans
    assert "inline via the style attribute" in ans.lower()
    assert "<style>" in ans.lower()
    assert "link rel=\"stylesheet\"" in ans.lower()


def test_comparison_cites_html_p27_and_stylesheets_p3() -> None:
    rows = [
        _r(
            "h2",
            "Table of contents for the HTML unit.",
            0.55,
            source="HTML.pdf",
            unit=2,
        ),
        _r(
            "h27",
            "Deprecated presentation attributes on BODY include bgcolor and text. "
            "These mix presentation with structure.",
            0.62,
            source="HTML.pdf",
            unit=27,
        ),
        _r(
            "s3",
            "Style sheets separate presentation from content using CSS—inline, internal <style>, "
            "or external link.",
            0.6,
            source="HTMLStyleSheets.pdf",
            unit=3,
        ),
    ]
    cites = [
        Citation(source_file="HTML.pdf", unit_type="page", unit_number=27),
        Citation(source_file="HTMLStyleSheets.pdf", unit_type="page", unit_number=3),
    ]
    ans, cite_ov = synthesize_fallback_answer(
        "Why were style sheets introduced if HTML already had presentation-related features?",
        rows,
        None,
        None,
        cites,
        is_cross_lecture=True,
        question_kind="comparison",
    )
    assert cite_ov is not None
    assert cite_ov, "comparison citations must not be empty when chunks exist"
    assert any(c.unit_number == 27 and c.source_file == "HTML.pdf" for c in cite_ov)
    assert any(
        c.unit_number == 3 and c.source_file == "HTMLStyleSheets.pdf" for c in cite_ov
    )
    bullet_lines = [ln for ln in ans.splitlines() if ln.strip().startswith("•")]
    assert bullet_lines, "answer must not end with an empty Sources block"
    assert "Complementary material in" not in ans
    assert "deprecated" in ans.lower() or "presentational" in ans.lower()
    assert "separate" in ans.lower() or "style sheet" in ans.lower()
    assert "no clear extractable sentences" not in ans.lower()
    assert "internal `<style>`" not in ans.lower()
    assert not ans.rstrip().endswith("Sources:")


def test_comparison_explanatory_when_html_and_stylesheet_units_pair() -> None:
    """Even thin clauses + file pairing must not yield the general relevance-threshold fallback."""
    rows = [
        _r("h27", "HTML module overview page.", 0.7, source="HTML.pdf", unit=27),
        _r(
            "s3",
            "Lecture on linking resources; mention markup and presentation concerns.",
            0.68,
            source="HTMLStyleSheets.pdf",
            unit=3,
        ),
    ]
    cites = [
        Citation(source_file="HTML.pdf", unit_type="page", unit_number=27),
        Citation(source_file="HTMLStyleSheets.pdf", unit_type="page", unit_number=3),
    ]
    ans, cite_ov = synthesize_fallback_answer(
        "Why were style sheets introduced if HTML already had presentation-related features?",
        rows,
        None,
        None,
        cites,
        is_cross_lecture=True,
        question_kind="comparison",
    )
    assert cite_ov is not None
    assert cite_ov
    assert "no clear extractable sentences" not in ans.lower()
    assert len(ans) > 80
    bullet_lines = [ln for ln in ans.splitlines() if ln.strip().startswith("•")]
    assert bullet_lines
    assert not ans.rstrip().endswith("Sources:")


def test_clean_corpus_strips_copyright() -> None:
    raw = "Copyright © 2020 ACME\nReal sentence about CSS here."
    assert "Copyright" not in clean_corpus_text(raw)
