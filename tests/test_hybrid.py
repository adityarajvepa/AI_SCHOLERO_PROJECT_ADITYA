"""Hybrid retrieval fusion tests."""

import numpy as np

from app.retrieval.hybrid import hybrid_search, minmax_norm
from app.schemas import DocumentChunk


def _chunk(cid: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        course_id="c1",
        lecture_id="lec",
        source_file="f.pdf",
        unit_type="page",
        unit_number=1,
        chunk_index=0,
        text=text,
    )


def test_minmax_norm_range() -> None:
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    y = minmax_norm(x)
    assert float(y.min()) == 0.0
    assert float(y.max()) == 1.0


def test_hybrid_search_ordering_and_dedupe() -> None:
    chunks = [
        _chunk("a", "gradient descent optimization"),
        _chunk("b", "cooking recipes for pasta"),
        _chunk("c", "backpropagation gradients"),
    ]
    # Artificial embeddings: first and third align with "gradient" direction
    m = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rows = hybrid_search(chunks, m, q, "gradient backpropagation", 0.65, 0.35, top_k=2)
    top_ids = [r.chunk.chunk_id for r in rows][:2]
    assert set(top_ids) == {"a", "c"}
    assert all(r.chunk.chunk_id != "b" for r in rows)
