"""Core-concept extraction and evidence coverage for strict abstention."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from app.generation.question_types import QuestionKind
from app.retrieval.hybrid import RetrievedRow
from app.retrieval.query_plan import _keyword_in_corpus, extract_focus_keywords

if TYPE_CHECKING:
    from app.retrieval.embeddings import EmbeddingModel

# Question prefixes to strip before phrase / segment extraction.
_LEADER = re.compile(
    r"^\s*(?:"
    r"what\s+is|what\s+are|who\s+is|who\s+are|"
    r"define|describe|explain|outline|sketch|discuss|"
    r"tell\s+me\s+about|give\s+me|list|name|state|summarize|summarise|"
    r"how\s+(?:does|do|is|are|can|should|will|would|might)\s+"
    r")\s+",
    re.IGNORECASE,
)

# Single-token concepts this generic are ignored for long questions (avoid acronym-only false passes).
_REALLY_GENERIC_TOKENS = frozenset(
    "html css js api http https url uri dom tcp ip pdf png csv json xml sql svg git".split()
)

# Segments starting with these are usually procedural, not entity names.
_BAD_SEGMENT_START = re.compile(r"^(?:why|when|where|whether|which)\b", re.IGNORECASE)

_CONCEPT_STOP = frozenset(
    """
    the this that these those your our their its some any each every few more most other
    such both between into onto from with without about over under again further once here
    there then than too very just also only even still much many lot ways means methods
    steps types kinds approaches options things stuff ideas points parts aspects examples
    lecture chapter section course material materials topic topics question questions
    answer answers slide slides page pages figure diagram introduction overview basics basic
    following called shows show see note notes learning using uses use web internet file
    documents related difference differences different similar same compare compared comparison
    versus contrast relationship relationships interaction interactions both various several
    """.split()
)

_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
# Mixed-case technical tokens (e.g. XGBoost, UMAP as typed in prose).
_MIXED_TECH = re.compile(r"\b(?:[A-Z]{2,}[a-z][a-z0-9]*|[a-z]+[A-Z][A-Za-z0-9]*)\b")


def _normalize_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _dedupe_preserve_order(phrases: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        k = _normalize_phrase(p)
        if len(k) < 2 or k in seen:
            continue
        seen.add(k)
        out.append(p.strip())
    return out


def extract_core_concepts(question: str, *, max_concepts: int = 8) -> list[str]:
    """
    Pull short entity / technique phrases the answer must be grounded in.

    Uses structure (leaders, ``and``/comma splits, acronyms, mixed-case tokens) — no
    hardcoded syllabus terms.
    """
    raw = question.strip()
    if not raw:
        return []
    concepts: list[str] = []

    for m in _ACRONYM.finditer(raw):
        tok = m.group(0)
        if tok.isdigit():
            continue
        if len(tok) <= 8 and tok.isalnum():
            concepts.append(tok.lower())

    for m in _MIXED_TECH.finditer(raw):
        concepts.append(m.group(0))

    remainder = _LEADER.sub("", raw).strip()
    remainder_stripped = remainder.rstrip("?.!").strip()
    remainder = remainder_stripped
    if remainder and not _BAD_SEGMENT_START.match(remainder):
        parts = re.split(r"\s*,\s*|\s*;\s*|\s+\band\s+", remainder, flags=re.IGNORECASE)
        for part in parts:
            p = part.strip().strip("'\"").strip()
            if not p or len(p) > 72:
                continue
            if _BAD_SEGMENT_START.match(p):
                continue
            wc = len(p.split())
            if wc > 6:
                continue
            words = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", p)
            if not words:
                continue
            lw = [w.lower() for w in words]
            if all(w in _CONCEPT_STOP for w in lw):
                continue
            concepts.append(" ".join(words))

    out = _dedupe_preserve_order(concepts)
    # Final filter: drop ultra-generic tokens and noise.
    filtered: list[str] = []
    for c in out:
        k = _normalize_phrase(c)
        parts = k.replace("-", " ").split()
        if all(p in _CONCEPT_STOP or len(p) < 2 for p in parts):
            continue
        if len(k) < 2:
            continue
        filtered.append(c)
        if len(filtered) >= max_concepts:
            break

    if (
        len(filtered) == 1
        and _normalize_phrase(filtered[0]) in _REALLY_GENERIC_TOKENS
        and len(remainder_stripped.split()) >= 10
    ):
        return []

    return filtered


def _concept_tokens(concept: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", concept.lower()) if len(t) >= 3]


def _flex_phrase_pattern(norm_phrase: str) -> re.Pattern[str]:
    """Whole-phrase match allowing hyphen/space variants between words."""
    parts = [p for p in re.split(r"[\s/-]+", norm_phrase.strip()) if p]
    if not parts:
        return re.compile(r"$^")
    if len(parts) == 1:
        return re.compile(rf"(?<![a-z0-9]){re.escape(parts[0])}(?![a-z0-9])", re.IGNORECASE)
    sep = r"[\s\-_/]+"
    body = sep.join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _lexical_concept_in_text(concept: str, text_lower: str) -> bool:
    """Strong lexical: phrase or whole-word hits with light plural/stem variants."""
    c = concept.strip()
    if not c:
        return False
    variants = {_normalize_phrase(c)}
    nk = _normalize_phrase(c).replace(" ", "")
    if nk:
        variants.add(nk)
    if c.lower().endswith("s") and len(c) > 4:
        variants.add(_normalize_phrase(c[:-1]))

    for v in variants:
        if not v:
            continue
        if " " in v or "-" in concept or "/" in concept:
            if _flex_phrase_pattern(v).search(text_lower):
                return True
        else:
            if _keyword_in_corpus(v, text_lower):
                return True
    parts = _concept_tokens(concept)
    if len(parts) >= 2:
        pat = _flex_phrase_pattern(" ".join(parts))
        if pat.search(text_lower):
            return True
    return False


def _weak_subtoken_evidence(concept: str, text_lower: str) -> bool:
    """At least one non-trivial token from the concept appears as a whole word."""
    for t in _concept_tokens(concept):
        if len(t) < 4:
            continue
        if _keyword_in_corpus(t, text_lower):
            return True
    return False


def _max_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float((a @ b) / (na * nb))


def _semantic_plus_lexical(
    concept: str,
    chunk_text: str,
    embedder: EmbeddingModel,
    *,
    sim_threshold: float = 0.52,
) -> bool:
    """Semantic similarity only counts with supporting lexical evidence (anti-drift)."""
    tl = chunk_text.lower()
    if not _weak_subtoken_evidence(concept, tl):
        return False
    snippet = chunk_text.strip()
    if len(snippet) > 1500:
        snippet = snippet[:1500]
    try:
        vecs = embedder.encode([concept, snippet])
        sim = _max_cosine_sim(vecs[0], vecs[1])
    except Exception:
        return False
    return sim >= sim_threshold


def _concept_supported_in_top_rows(
    concept: str,
    rows: Sequence[RetrievedRow],
    *,
    top_n: int = 8,
    query_embedding: np.ndarray | None,
    embedder: EmbeddingModel | None,
    kind: QuestionKind,
    allow_semantic: bool,
) -> bool:
    slice_rows = list(rows[:top_n])
    for r in slice_rows:
        tl = r.chunk.text.lower()
        if _lexical_concept_in_text(concept, tl):
            return True
    if (
        allow_semantic
        and embedder is not None
        and query_embedding is not None
        and kind == "general"
    ):
        for r in slice_rows:
            if _semantic_plus_lexical(concept, r.chunk.text, embedder):
                return True
    return False


def should_abstain_insufficient_concept_evidence(
    question: str,
    rows: Sequence[RetrievedRow],
    *,
    kind: QuestionKind,
    query_embedding: np.ndarray | None = None,
    embedder: Any | None = None,
) -> tuple[bool, str]:
    """
    Abstain when extracted core concepts are not evidenced in top retrieval.

    Enumeration questions skip this gate (list-style prompts rarely name stable entities).
    """
    if not rows or kind in ("enumeration", "comparison"):
        return False, ""

    concepts = extract_core_concepts(question)
    if not concepts and kind in ("definitional", "explanatory", "general"):
        alt = [
            k
            for k in extract_focus_keywords(question, max_terms=12)
            if len(k) >= 4 and k.lower() not in _REALLY_GENERIC_TOKENS
        ][:4]
        if len(alt) >= 2:
            concepts = alt[:3]
    if not concepts:
        return False, ""

    # Definitional / explanatory: lexical grounding only (semantic-only is too drift-prone).
    allow_semantic = kind not in ("definitional", "explanatory")

    missing: list[str] = []
    for c in concepts:
        if not _concept_supported_in_top_rows(
            c,
            rows,
            query_embedding=query_embedding,
            embedder=embedder,
            kind=kind,
            allow_semantic=allow_semantic,
        ):
            missing.append(c)

    if not missing:
        return False, ""

    shown = ", ".join(missing[:5])
    extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
    reason = (
        f"Core concept(s) requested by the question are not evidenced in the top retrieved "
        f"passages (missing support for: {shown}{extra}). "
        "The retriever surfaced broadly related material, not text that substantively defines or "
        "explains those concepts."
    )
    return True, reason
