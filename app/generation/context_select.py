"""Re-rank diversified retrieval rows for synthesis (contrast-aware, deterministic)."""

from __future__ import annotations

import re
from typing import Sequence

from app.retrieval.hybrid import RetrievedRow
from app.retrieval.query_plan import _keyword_in_corpus, extract_focus_keywords

# Terms/phrases that usually signal substantive contrast / migration / separation-of-concerns
# evidence (vs generic intro or TOC slides).
_CONTRAST_SIGNALS: tuple[str, ...] = (
    "deprecated",
    "deprecation",
    "presentational",
    "presentation attribute",
    "presentation attributes",
    "style sheet",
    "style sheets",
    "stylesheet",
    "separate",
    "separation",
    "separated",
    "instead of",
    "rather than",
    "replaced",
    "replacement",
    "inline style",
    "bgcolor",
    "fgcolor",
    "mixing",
    "mix markup",
    "content and presentation",
    "markup and style",
    "using style",
    "without mixing",
    "old html",
    "legacy",
)

_INTRO_WEAK_PATTERNS: tuple[str, ...] = (
    "table of contents",
    "outline",
    "course schedule",
    "week at a glance",
    "learning objectives",
    "welcome to",
    "about this course",
)


def _contrast_signal_score(text_lower: str) -> float:
    s = 0.0
    for phrase in _CONTRAST_SIGNALS:
        if phrase in text_lower:
            s += 3.0
    return s


def _intro_penalty(text_lower: str) -> float:
    """Down-rank generic intro / TOC-style chunks when stronger evidence exists elsewhere."""
    pen = 0.0
    for pat in _INTRO_WEAK_PATTERNS:
        if pat in text_lower:
            pen += 1.2
    if text_lower.count("\n") > 10 and len(text_lower) < 500:
        pen += 0.6
    return pen


def context_evidence_score(
    row: RetrievedRow,
    question_lower: str,
    focus_terms: Sequence[str],
    *,
    is_cross_lecture: bool,
) -> float:
    """
    Score a chunk for how useful it is as *evidence* for this question.

    Hybrid rank is a weak tie-breaker; overlap with focus terms and contrast signals dominates.
    """
    t = row.chunk.text.lower()
    s = 0.0
    for kw in focus_terms:
        if _keyword_in_corpus(kw, t):
            s += 1.6 if len(kw) >= 6 else 1.1
    if is_cross_lecture:
        s += _contrast_signal_score(t)
        s -= _intro_penalty(t)
    s += float(row.hybrid_score) * 0.35
    return s


def contrast_bonus_for_sentence(sentence_lower: str, *, is_cross_lecture: bool) -> float:
    """Small additive for sentence ranking when contrast language is present."""
    if not is_cross_lecture:
        return 0.0
    b = 0.0
    for phrase in _CONTRAST_SIGNALS:
        if phrase in sentence_lower:
            b += 0.07
    return min(b, 0.35)


def select_synthesis_rows(
    diversified_rows: list[RetrievedRow],
    question: str,
    *,
    is_cross_lecture: bool,
    max_chunks: int,
) -> list[RetrievedRow]:
    """
    Pick which diversified hits feed synthesis and citations (debug stays on full list).

    Rows are re-ordered by evidence score so strong contrast / deprecation passages surface
    even when diversification placed weaker same-file chunks earlier.
    """
    if not diversified_rows:
        return []
    q = question.lower()
    terms = extract_focus_keywords(question, max_terms=14)
    scored = [
        (context_evidence_score(r, q, terms, is_cross_lecture=is_cross_lecture), r)
        for r in diversified_rows
    ]
    scored.sort(key=lambda x: (-x[0], -x[1].hybrid_score, x[1].chunk.chunk_id))
    return [r for _, r in scored[:max_chunks]]
