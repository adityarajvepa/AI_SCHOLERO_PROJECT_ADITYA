"""Page/slide-aware chunking with paragraph-first splitting."""

from __future__ import annotations

from app.utils.text import chunk_words, split_paragraphs, word_count


# Target ~300–700 words; overlap ~50–100 words
TARGET_MIN = 300
TARGET_MAX = 700
OVERLAP_MIN = 50
OVERLAP_MAX = 100


def _pick_overlap(target: int) -> int:
    if target <= 350:
        return OVERLAP_MIN
    return min(OVERLAP_MAX, max(OVERLAP_MIN, target // 6))


def chunk_unit_text(unit_text: str) -> list[str]:
    """
    Split a single page/slide's text into one or more chunks.

    Short units stay as one chunk. Longer text splits on paragraphs first,
    then falls back to overlapping word windows.
    """
    text = " ".join(unit_text.split())
    if not text:
        return []
    wc = word_count(text)
    if wc <= TARGET_MAX:
        return [text]

    paras = split_paragraphs(unit_text.replace("\r\n", "\n"))
    if len(paras) <= 1:
        return list(
            chunk_words(text, max_words=TARGET_MAX, overlap_words=_pick_overlap(TARGET_MAX))
        )

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush_buf() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_words = 0

    for p in paras:
        pw = word_count(p)
        if pw >= TARGET_MAX:
            flush_buf()
            chunks.extend(
                chunk_words(
                    " ".join(p.split()),
                    max_words=TARGET_MAX,
                    overlap_words=_pick_overlap(TARGET_MAX),
                )
            )
            continue
        if buf_words + pw > TARGET_MAX and buf_words >= TARGET_MIN:
            flush_buf()
        buf.append(p)
        buf_words += pw
        if buf_words >= TARGET_MAX:
            flush_buf()

    flush_buf()

    merged: list[str] = []
    for c in chunks:
        c2 = c.strip()
        if c2:
            merged.append(c2)
    if not merged:
        return [text]
    return merged
