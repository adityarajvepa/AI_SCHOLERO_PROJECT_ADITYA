"""Heuristics for cross-lecture retrieval, diversification, and topic abstention."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from app.retrieval.hybrid import RetrievedRow

# Comparative / cross-lecture phrasing (substring match on lowercased question).
_CROSS_LECTURE_MARKERS: tuple[str, ...] = (
    " why ",
    "why ",
    " why?",
    "compare",
    "comparison",
    "compared",
    "difference",
    "differences",
    "different",
    "relate",
    "related",
    "relates",
    "connect",
    "connection",
    "versus",
    " vs ",
    " vs.",
    "instead of",
    "introduced if",
    "replaced",
    "contrast",
    "relationship",
    " both ",
    "how does ",
    "how do ",
    "how did ",
    "compared to",
    "similarities",
    "interaction between",
    "relationship between",
    " tie ",
    " ties ",
)

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else for to of in on at by as is are was were be been being
    has have had do does did will would could should may might must can this that these those
    it its we you they what which who whom whose how when where why than into from with about
    into over under again further once here there all any both each few more most other some
    such no nor not only own same so than too very just also only even ever still just
    """.split()
)

# Very common in intro programming / web slides; not used to require literal hits for abstention.
_GENERIC_TOPIC_TERMS: frozenset[str] = frozenset(
    """
    javascript ecma js code program programming example examples lecture lesson chapter section
    slide slides page pages figure diagram introduction overview basics basic using uses use
    called following shows show see note notes learning course material materials topic topics
    question questions answer answers web browser internet file files document documents
    """.split()
)

CROSS_LECTURE_POOL_MIN = 15
CROSS_LECTURE_POOL_MAX = 20


def is_cross_lecture_question(question: str) -> bool:
    """True when the phrasing suggests comparing, relating, or contrasting ideas across topics."""
    q = f" {question.lower().strip()} "
    return any(m in q for m in _CROSS_LECTURE_MARKERS)


def cross_lecture_pool_size(top_k: int) -> int:
    """Larger candidate pool before diversification."""
    return min(CROSS_LECTURE_POOL_MAX, max(CROSS_LECTURE_POOL_MIN, top_k * 2))


def diversify_by_source_file(
    candidates: list[RetrievedRow],
    final_k: int,
) -> list[RetrievedRow]:
    """
    Round-robin across source_file buckets so multiple lectures appear in the final context
    when the expanded candidate pool contains them, without discarding hybrid ordering
    within each file.

    Each sweep takes at most one new chunk per file (in file order by best hybrid score),
    repeating until ``final_k`` rows or no file can contribute. Remaining slots are filled
    from the global hybrid-ordered candidate list.
    """
    if final_k <= 0 or not candidates:
        return []
    if len(candidates) <= final_k:
        return candidates[:final_k]

    by_file: dict[str, list[RetrievedRow]] = defaultdict(list)
    for r in candidates:
        by_file[r.chunk.source_file].append(r)
    for f in by_file:
        by_file[f].sort(key=lambda x: (-x.hybrid_score, -x.dense_score, x.chunk.chunk_id))

    files = sorted(by_file.keys(), key=lambda f: (-by_file[f][0].hybrid_score, f))
    files = [f for f in files if by_file[f]]
    if not files:
        return []
    ptrs = {f: 0 for f in files}
    selected: list[RetrievedRow] = []
    seen: set[str] = set()
    turn = 0

    while len(selected) < final_k:
        progressed = False
        for _ in range(len(files)):
            f = files[turn % len(files)]
            turn += 1
            lst = by_file[f]
            while ptrs[f] < len(lst):
                cand = lst[ptrs[f]]
                ptrs[f] += 1
                if cand.chunk.chunk_id not in seen:
                    seen.add(cand.chunk.chunk_id)
                    selected.append(cand)
                    progressed = True
                    break
            if len(selected) >= final_k:
                break
        if not progressed:
            break

    if len(selected) < final_k:
        for r in candidates:
            if len(selected) >= final_k:
                break
            if r.chunk.chunk_id in seen:
                continue
            seen.add(r.chunk.chunk_id)
            selected.append(r)
    return selected[:final_k]


def extract_focus_keywords(question: str, max_terms: int = 12) -> list[str]:
    """Alphanumeric tokens minus stopwords and ultra-generic course vocabulary."""
    raw = re.findall(r"[a-z0-9]+", question.lower())
    out: list[str] = []
    for t in raw:
        if len(t) < 3 or t in _STOPWORDS or t in _GENERIC_TOPIC_TERMS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _keyword_in_corpus(kw: str, corpus_lower: str) -> bool:
    """
    Whole-word match for very short tokens to avoid spurious hits (e.g. ``dom`` in ``domain``).

    Longer tokens use substring match so terms like ``style`` still match ``stylesheet``.
    """
    if len(kw) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", corpus_lower) is not None
    return kw in corpus_lower


def should_abstain_for_keyword_gap(
    question: str,
    rows: Sequence[RetrievedRow],
) -> tuple[bool, str]:
    """
    Abstain when focus terms from the question are largely missing from top hits
    (broad-but-off-topic retrieval), using simple deterministic rules.
    """
    if not rows:
        return False, ""
    kws = extract_focus_keywords(question)
    if len(kws) < 2:
        return False, ""

    hits = 0
    for kw in kws:
        if any(_keyword_in_corpus(kw, r.chunk.text.lower()) for r in rows[:8]):
            hits += 1
    ratio = hits / len(kws)
    top_h = float(rows[0].hybrid_score)

    if hits == 0:
        return (
            True,
            f"None of the focus terms ({', '.join(kws[:6])}) appear in the top retrieved passages.",
        )
    if len(kws) >= 4 and ratio < 0.34:
        return (
            True,
            f"Only {hits}/{len(kws)} focus terms match the retrieved text (ratio {ratio:.2f}), "
            "so the hit looks broad rather than specific to this question.",
        )
    if len(kws) >= 3 and ratio < 0.45 and top_h < 0.36:
        return (
            True,
            f"Keyword overlap is {hits}/{len(kws)} with a modest top hybrid score ({top_h:.2f}), "
            "which is treated as insufficient grounding.",
        )
    if len(kws) == 2 and ratio < 0.5 and top_h < 0.33:
        return (
            True,
            f"Keyword overlap is {hits}/{len(kws)} with top hybrid {top_h:.2f}, "
            "below the confidence threshold for a narrow question.",
        )
    return False, ""


def abstention_summary_files(rows: Sequence[RetrievedRow], limit: int = 4) -> str:
    """Short hint for abstention body: which filenames were closest."""
    stems: list[str] = []
    seen: set[str] = set()
    for r in rows[:6]:
        stem = Path(r.chunk.source_file).stem
        if stem not in seen:
            seen.add(stem)
            stems.append(r.chunk.source_file)
        if len(stems) >= limit:
            break
    return ", ".join(stems)
