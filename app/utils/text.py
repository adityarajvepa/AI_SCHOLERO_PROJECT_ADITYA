"""Text utilities for chunking and previews."""

import re
from collections.abc import Iterable


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def char_count_non_ws(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def truncate_preview(text: str, max_chars: int = 240) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_words(text: str, max_words: int, overlap_words: int) -> Iterable[str]:
    """Yield overlapping windows by word count."""
    words = text.split()
    if not words:
        return
    n = len(words)
    step = max(1, max_words - overlap_words)
    start = 0
    while start < n:
        end = min(n, start + max_words)
        yield " ".join(words[start:end])
        if end == n:
            break
        start += step
