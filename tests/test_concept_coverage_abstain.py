"""Regression: unsupported definitional / explanation questions must abstain without citations."""

from __future__ import annotations

import pytest

from app.generation.tutor import answer_question
from app.retrieval.concept_coverage import extract_core_concepts
from app.retrieval.hybrid import RetrievedRow
from app.schemas import DocumentChunk


def _row(
    cid: str,
    text: str,
    hybrid: float,
    *,
    source: str = "lecture_notes.pdf",
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
        bm25_score=0.55,
    )


_ML_DRIFT = (
    "When interpreting results on multi-label benchmarks, consider calibration plots and "
    "whether reported gains are statistically significant across random seeds."
)


@pytest.mark.parametrize(
    "question",
    [
        "What is BERT?",
        "Explain transformers and self-attention.",
        "What is UMAP?",
        "What is XGBoost?",
    ],
)
def test_unsupported_concept_questions_abstain(question: str) -> None:
    rows = [
        _row("r1", _ML_DRIFT, 0.52, source="ml_eval.pdf", unit=3),
        _row("r2", _ML_DRIFT + " Focus on precision-recall curves for imbalanced labels.", 0.48, unit=4),
    ]
    resp = answer_question(question, rows, is_cross_lecture=False)
    assert "do not appear to substantively" in resp.answer
    assert resp.citations == []
    assert "Reviewed sources" in resp.answer
    assert "this refers to" not in resp.answer.lower()


@pytest.mark.parametrize(
    "q, expected_substrings",
    [
        ("Explain transformers and self-attention.", ["transformers", "self-attention"]),
        ("What is UMAP?", ["umap"]),
        ("What is XGBoost?", ["xgboost"]),
        ("What is BERT?", ["bert"]),
    ],
)
def test_extract_core_concepts_structure(q: str, expected_substrings: list[str]) -> None:
    got = [s.lower() for s in extract_core_concepts(q)]
    for sub in expected_substrings:
        assert any(sub in g.replace(" ", "") or sub in g.replace("-", "") for g in got), (
            f"missing {sub!r} in {got!r}"
        )
