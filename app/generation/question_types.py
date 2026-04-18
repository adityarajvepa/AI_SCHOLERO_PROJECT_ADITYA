"""Lightweight question-type classification for fallback answer templates."""

from __future__ import annotations

import re
from typing import Literal

from app.retrieval.query_plan import is_cross_lecture_question

QuestionKind = Literal["definitional", "enumeration", "comparison", "explanatory", "general"]

_ENUM_PATTERNS = re.compile(
    r"what are (the )?(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r".{0,140}?\b(ways|methods|means|types|steps|kinds|approaches|options)\b",
    re.IGNORECASE | re.DOTALL,
)

_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def expected_enumeration_count(question: str) -> int | None:
    """Return N when the question asks for N ways/types/steps, else None."""
    m = _ENUM_PATTERNS.search(question)
    if not m:
        return None
    raw = m.group("n").lower()
    if raw.isdigit():
        return int(raw)
    return _WORD_TO_INT.get(raw)


def classify_question(question: str, *, is_cross_from_api: bool = False) -> QuestionKind:
    """
    Classify for template routing. Enumeration is checked before broad ``what are``.
    """
    ql = question.lower().strip()
    if is_cross_from_api or is_cross_lecture_question(question):
        return "comparison"
    if _ENUM_PATTERNS.search(question):
        return "enumeration"
    if any(ql.startswith(p) for p in ("what is ", "define ", "describe ")):
        return "definitional"
    if ql.startswith("what are ") and expected_enumeration_count(question) is None:
        return "definitional"
    if ql.startswith("explain ") and len(ql) < 120:
        return "definitional"
    if "what is " in ql and len(ql) < 100 and expected_enumeration_count(question) is None:
        return "definitional"
    if any(
        ql.startswith(p)
        for p in (
            "how does ",
            "how do ",
            "how did ",
            "how can ",
            "how should ",
            "how is ",
            "how are ",
        )
    ):
        return "explanatory"
    if re.match(r"^why\b", ql) and not is_cross_lecture_question(question):
        return "explanatory"
    return "general"


def is_definitional_kind(kind: QuestionKind) -> bool:
    return kind == "definitional"
