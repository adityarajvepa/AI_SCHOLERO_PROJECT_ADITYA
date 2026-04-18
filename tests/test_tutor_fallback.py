"""Tutor fallback synthesis and citation trimming."""

import numpy as np

from app.generation.tutor import MAX_RESPONSE_CITATIONS, _top_citations, synthesize_fallback_answer
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


def test_top_citations_respects_limit_and_order() -> None:
    rows = [
        _row("a", "one", 0.9, source="a.pdf", unit=1),
        _row("b", "two", 0.8, source="a.pdf", unit=2),
        _row("c", "three", 0.7, source="b.pdf", unit=1),
        _row("d", "four", 0.6, source="b.pdf", unit=2),
        _row("e", "five", 0.5, source="c.pdf", unit=9),
    ]
    cites = _top_citations(rows, limit=3)
    assert len(cites) == 3
    assert cites[0].unit_number == 1 and cites[0].source_file == "a.pdf"


def test_fallback_no_long_verbatim_chunk_dump() -> None:
    long_noise = "word " * 200
    rows = [
        _row(
            "x1",
            "URLs identify resources on the web. HTTP is how browsers request pages. "
            + long_noise,
            0.95,
            source="internet.pdf",
            unit=84,
        ),
        _row(
            "x2",
            "HTML structures the document the browser shows. Links use anchor tags.",
            0.9,
            source="html.pdf",
            unit=48,
        ),
    ]
    cites = _top_citations(rows, MAX_RESPONSE_CITATIONS)
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    class FakeEmb:
        def encode(self, texts: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
            # Prefer sentences containing "URL" / "HTTP" / "HTML" for cosine with q=[1,0,0]
            out = []
            for t in texts:
                tl = t.lower()
                v = [0.05, 0.02, 0.02]
                if "url" in tl:
                    v[0] += 1.0
                if "http" in tl:
                    v[0] += 0.5
                if "html" in tl:
                    v[0] += 0.3
                out.append(v)
            arr = np.array(out, dtype=np.float32)
            return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    ans, cite_ov = synthesize_fallback_answer(
        "What are URLs HTTP and HTML?",
        rows,
        q,
        FakeEmb(),
        cites,
        question_kind="general",
    )
    assert "Here is what the retrieved course excerpts support" not in ans
    assert "Sources:" in ans
    assert long_noise.strip() not in ans
    assert "Local tutor mode" not in ans
    assert "Complementary material in" not in ans
    assert cite_ov is None
    assert len(cites) <= MAX_RESPONSE_CITATIONS
