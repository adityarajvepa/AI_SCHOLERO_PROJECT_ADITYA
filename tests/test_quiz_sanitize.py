"""Quiz citation validation (no live Gemini)."""

from app.generation.quiz import _sanitize_quiz_items
from app.retrieval.hybrid import RetrievedRow
from app.schemas import DocumentChunk


def _row(cid: str, text: str, source: str = "Lecture-A.pdf", unit: int = 3) -> RetrievedRow:
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
        hybrid_score=0.5,
        dense_score=0.5,
        bm25_score=0.3,
    )


def test_sanitize_drops_fake_citations() -> None:
    rows = [_row("a", "Alpha content about trees.", "Trees.pdf", 1)]
    raw = [
        {
            "question": "What is discussed in the excerpt?",
            "answer_key": "Trees and forests.",
            "citations": [{"source_file": "Trees.pdf", "page": 1}],
        },
        {
            "question": "Invented question without valid source?",
            "answer_key": "Nothing.",
            "citations": [{"source_file": "Other.pdf", "page": 99}],
        },
    ]
    items, _ = _sanitize_quiz_items(raw, rows, max_questions=5)
    assert len(items) == 1
    assert "Trees" in items[0].citations[0].source_file


def test_sanitize_accepts_json_float_page_numbers() -> None:
    """LLMs often emit JSON numbers as floats; citations must still validate."""
    rows = [_row("a", "Content here.", "Notes.pdf", 4)]
    raw = [
        {
            "question": "What appears in the notes on page four?",
            "answer_key": "This content.",
            "citations": [{"source_file": "Notes.pdf", "page": 4.0}],
        }
    ]
    items, _ = _sanitize_quiz_items(raw, rows, max_questions=3)
    assert len(items) == 1
    assert items[0].citations[0].page == 4


def test_sanitize_accepts_slide_unit() -> None:
    ch = DocumentChunk(
        chunk_id="s1",
        course_id="c",
        lecture_id="lec",
        source_file="Deck.pptx",
        unit_type="slide",
        unit_number=7,
        chunk_index=0,
        text="Slide seven discusses metrics.",
    )
    r = RetrievedRow(chunk=ch, hybrid_score=0.4, dense_score=0.4, bm25_score=0.2)
    raw = [
        {
            "question": "What is on slide 7?",
            "answer_key": "Metrics.",
            "citations": [{"source_file": "Deck.pptx", "slide": 7}],
        }
    ]
    items, _ = _sanitize_quiz_items(raw, [r], max_questions=3)
    assert len(items) == 1
    assert items[0].citations[0].slide == 7
