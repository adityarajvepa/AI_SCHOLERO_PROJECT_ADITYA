"""Tests for page/slide-aware chunking."""

from app.ingestion.chunker import chunk_unit_text
from app.utils.text import word_count


def test_short_unit_single_chunk() -> None:
    text = "Intro paragraph only."
    chunks = chunk_unit_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Intro paragraph only."


def test_long_unit_splits() -> None:
    para = "word " * 400
    text = "\n\n".join([para, para])
    chunks = chunk_unit_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert 40 <= word_count(c) <= 800
