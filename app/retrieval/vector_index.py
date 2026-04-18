"""Cosine similarity dense retrieval over a numpy embedding matrix."""

from __future__ import annotations

import numpy as np


def dense_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity for L2-normalized query (d,) and row-normalized matrix (n, d).

    Returns shape (n,) float32.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query_vec.astype(np.float32, copy=False).reshape(-1)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = matrix.astype(np.float32, copy=False)
    mn = mn / (np.linalg.norm(mn, axis=1, keepdims=True) + 1e-12)
    return (mn @ qn).astype(np.float32, copy=False)
