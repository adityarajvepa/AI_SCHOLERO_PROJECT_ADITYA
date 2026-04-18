"""BM25 lexical index built from tokenized chunk texts."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def build_bm25(corpus: Sequence[str]) -> BM25Okapi:
    tokenized = [tokenize(t) if t.strip() else ["<empty>"] for t in corpus]
    return BM25Okapi(tokenized)


def bm25_scores(index: BM25Okapi, query: str) -> np.ndarray:
    n = len(index.doc_len)
    q = tokenize(query)
    if not q:
        return np.zeros((n,), dtype=np.float32)
    scores = index.get_scores(q)
    return np.asarray(scores, dtype=np.float32)
