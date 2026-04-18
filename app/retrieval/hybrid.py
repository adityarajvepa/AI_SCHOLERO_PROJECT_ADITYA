"""Hybrid dense + BM25 retrieval with normalization and fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.retrieval.bm25_index import bm25_scores, build_bm25, tokenize
from app.retrieval.vector_index import dense_similarities
from app.schemas import DocumentChunk


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] per query; flat maps to ones."""
    if scores.size == 0:
        return scores.astype(np.float32, copy=False)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores, dtype=np.float32)
    out = (scores - lo) / (hi - lo)
    return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class RetrievedRow:
    chunk: DocumentChunk
    hybrid_score: float
    dense_score: float
    bm25_score: float


def hybrid_search(
    chunks: list[DocumentChunk],
    embedding_matrix: np.ndarray,
    query_embedding: np.ndarray,
    query_text: str,
    weight_dense: float,
    weight_bm25: float,
    top_k: int,
) -> list[RetrievedRow]:
    """
    Run dense cosine retrieval and BM25, min-max normalize, fuse, dedupe, top-k.

    Deterministic ordering: hybrid desc, dense desc, chunk_id asc.
    """
    if not chunks or embedding_matrix.shape[0] != len(chunks):
        return []
    dense = dense_similarities(query_embedding, embedding_matrix)
    dn = minmax_norm(dense)
    raw_bm25 = np.zeros_like(dense, dtype=np.float32)
    bn = np.zeros_like(dense, dtype=np.float32)
    if tokenize(query_text):
        bm25_index = build_bm25([c.text for c in chunks])
        raw_bm25 = bm25_scores(bm25_index, query_text)
        if raw_bm25.shape[0] != len(chunks):
            raw_bm25 = np.zeros((len(chunks),), dtype=np.float32)
        bn = minmax_norm(raw_bm25)
        hybrid = weight_dense * dn + weight_bm25 * bn
    else:
        hybrid = dn

    idxs = list(range(len(chunks)))
    idxs.sort(
        key=lambda i: (
            -float(hybrid[i]),
            -float(dense[i]),
            -float(raw_bm25[i]),
            chunks[i].chunk_id,
        )
    )

    seen: set[str] = set()
    out: list[RetrievedRow] = []
    for i in idxs:
        cid = chunks[i].chunk_id
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            RetrievedRow(
                chunk=chunks[i],
                hybrid_score=float(hybrid[i]),
                dense_score=float(dn[i]),
                bm25_score=float(bn[i]),
            )
        )
        if len(out) >= top_k:
            break
    return out
