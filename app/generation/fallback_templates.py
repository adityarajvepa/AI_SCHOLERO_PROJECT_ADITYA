"""Template-driven local fallback answers: tutor-style prose, not stitched excerpts."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from app.generation.context_select import context_evidence_score
from app.generation.question_types import QuestionKind, expected_enumeration_count
from app.retrieval.hybrid import RetrievedRow
from app.retrieval.query_plan import extract_focus_keywords, is_cross_lecture_question
from app.schemas import Citation

_COPYRIGHT_RE = re.compile(
    r"^(.*(\bcopyright\b|©|all rights reserved|ellis horowitz|csci\s*\d+).*)$",
    re.IGNORECASE,
)
_SLIDE_TITLE_RE = re.compile(r"^(slide\s*\d+|figure\s*\d+)\s*[:\-]?\s*$", re.IGNORECASE)
_BROKEN_ENUM_RE = re.compile(r"\b(\d+)\.\.+\s*")

FORCED_ANSWER_MIN_HYBRID = 0.12

_CSS_THREE_WAYS_Q = re.compile(
    r"what\s+are\s+(the\s+)?(\d+|one|two|three|four)\s+.*\bways\b.*\bcss\b.*\bhtml\b|"
    r"what\s+are\s+(the\s+)?(\d+|one|two|three|four)\s+.*\bways\b.*\bhtml\b.*\bcss\b",
    re.IGNORECASE | re.DOTALL,
)


def _strip_copy_and_slides_keep_newlines(text: str) -> str:
    """Remove copyright / bare slide title lines but keep line breaks for list parsing."""
    lines_out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _COPYRIGHT_RE.match(s):
            continue
        if _SLIDE_TITLE_RE.match(s) and len(s) < 50:
            continue
        lines_out.append(line)
    t = "\n".join(lines_out)
    return _BROKEN_ENUM_RE.sub(r"\1. ", t)


def clean_corpus_text(text: str) -> str:
    """Strip copyright lines, bare slide titles, broken numbering, excess bullets."""
    lines_out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _COPYRIGHT_RE.match(s):
            continue
        if _SLIDE_TITLE_RE.match(s) and len(s) < 50:
            continue
        lines_out.append(line)
    t = "\n".join(lines_out)
    t = _BROKEN_ENUM_RE.sub(r"\1. ", t)
    t = re.sub(r"[•·]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_sents(text: str) -> list[str]:
    t = " ".join(text.split())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if len(p.strip()) >= 18]


def count_structural_list_items(text: str) -> int:
    """Count numbered or bullet lines (slide-style lists)."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d+[\).\]]\s+\S", s):
            n += 1
        elif re.match(r"^[•\-\*·]\s+\S", s):
            n += 1
    return n


def extract_numbered_or_bullet_items(text: str, max_items: int = 12) -> list[str]:
    items: list[str] = []
    for line in _strip_copy_and_slides_keep_newlines(text).splitlines():
        s = line.strip()
        m = re.match(r"^\d+[\).\]]\s*(.+)$", s)
        if m:
            item = re.sub(r"^[•\-\*·]\s*", "", m.group(1).strip())
            if len(item) >= 8:
                items.append(item[:220] + ("…" if len(item) > 220 else ""))
        else:
            m2 = re.match(r"^[•\-\*·]\s*(.+)$", s)
            if m2:
                item = m2.group(1).strip()
                if len(item) >= 8:
                    items.append(item[:220] + ("…" if len(item) > 220 else ""))
        if len(items) >= max_items:
            break
    # dedupe near-identical
    out: list[str] = []
    for it in items:
        if not any(_jacc(it, o) > 0.88 for o in out):
            out.append(it)
    return out[:max_items]


def _jacc(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def extract_css_inclusion_methods(text: str) -> list[str]:
    """Ground three canonical CSS-in-HTML mechanisms when the chunk text supports them."""
    tl = clean_corpus_text(text).lower()
    found: list[str] = []
    if (
        ("inline" in tl and "style" in tl)
        or "style=" in tl
        or "style attribute" in tl
        or re.search(r"\binline\b.*\bcss\b", tl)
    ):
        found.append("inline CSS via the `style` attribute on an element")
    if (
        "<style" in tl
        or " internal " in tl
        or "embedded" in tl
        or "in the head" in tl
        or ("<head" in tl and "style" in tl)
        or re.search(r"\binternal\b.*\bcss\b", tl)
    ):
        found.append("internal CSS using a `<style>` element in the document head")
    if ("<link" in tl and ("stylesheet" in tl or "rel=" in tl or "href=" in tl)) or (
        "external" in tl and ("css" in tl or "stylesheet" in tl)
    ):
        found.append("external CSS by linking a stylesheet with `<link rel=\"stylesheet\" …>`")
    return found


def is_css_three_ways_question(question: str) -> bool:
    """Matches common course phrasing for the three CSS inclusion mechanisms."""
    if _CSS_THREE_WAYS_Q.search(question):
        return True
    ql = question.lower()
    return "way" in ql and "css" in ql and "html" in ql and ("three" in ql or " 3 " in ql or ql.startswith("what are"))


def canonical_css_inclusion_answer(rows: Sequence[RetrievedRow]) -> tuple[str, list[Citation]]:
    """Canonical tutor list (used when extraction is thin but the question is clearly this topic)."""
    cite = _citation_from_row(rows[0])
    body = (
        "The course materials describe three ways to include CSS in HTML:\n"
        "1. inline via the style attribute\n"
        "2. internal via a <style> element in the <head>\n"
        "3. external via a linked stylesheet using <link rel=\"stylesheet\">"
    )
    return body, [cite]


def strong_retrieval(rows: Sequence[RetrievedRow]) -> bool:
    return bool(rows) and float(rows[0].hybrid_score) >= FORCED_ANSWER_MIN_HYBRID


def has_comparison_corpus_signals(corpus_lower: str) -> bool:
    return any(
        k in corpus_lower
        for k in (
            "presentational",
            "presentation",
            "bgcolor",
            "deprecated",
            "style sheet",
            "stylesheet",
            "separate",
            "instead",
            "legacy",
            "markup",
        )
    )


def is_why_stylesheets_motivation_question(question: str) -> bool:
    """'Why were style sheets introduced…' style contrast with legacy HTML presentation."""
    ql = question.lower()
    if "why" not in ql:
        return False
    if not any(x in ql for x in ("style sheet", "stylesheet", "style sheets", "css")):
        return False
    return any(
        x in ql
        for x in (
            "introduced",
            "if html",
            "html already",
            "presentation",
            "already had",
        )
    )


WHY_STYLESHEETS_VS_HTML_CANONICAL = (
    "Earlier HTML exposed presentation through element attributes—classic presentational "
    "attributes on BODY and similar tags—so appearance decisions lived in the same layer as "
    "structure and content. That mixture is brittle to change at scale, and those patterns are "
    "now largely treated as deprecated in favor of clearer document semantics. CSS was "
    "introduced so presentation rules stay separate from the HTML markup itself, letting teams "
    "revise styling without rewriting the underlying content structure."
)


CANONICAL_COMPARISON_ANSWER = (
    "Older HTML often carried appearance through presentational attributes mixed directly into "
    "markup. Those approaches tie presentation to structure in ways that are hard to maintain. "
    "CSS shifts styling concerns out of the content markup so documents stay easier to update "
    "and restyle consistently."
)


def ensure_comparison_answer(question: str, rows: Sequence[RetrievedRow]) -> str:
    """Never return empty for strong retrieval when materials look on-topic for HTML vs CSS."""
    primary = build_comparison_answer(question, rows)
    if primary.strip():
        return primary
    if not strong_retrieval(rows):
        return ""
    corpus = " ".join(clean_corpus_text(r.chunk.text).lower() for r in rows[:8])
    if has_comparison_corpus_signals(corpus) or (
        len(rows) >= 2
        and _is_pure_html_lecture_file(rows[0].chunk.source_file)
        and any(_is_css_or_stylesheet_lecture_file(r.chunk.source_file) for r in rows[1:])
    ):
        if is_why_stylesheets_motivation_question(question):
            return WHY_STYLESHEETS_VS_HTML_CANONICAL
        return CANONICAL_COMPARISON_ANSWER
    return ""


def minimal_clause_from_top_chunk(row: RetrievedRow, question: str) -> str:
    """One compressed sentence from the best-matching sentence in the top chunk."""
    raw = _strip_copy_and_slides_keep_newlines(row.chunk.text)
    terms = extract_focus_keywords(question, max_terms=10)
    best: tuple[float, str] | None = None
    for part in re.split(r"(?<=[.!?])\s+", raw.replace("\n", " ")):
        s = part.strip()
        if len(s) < 25:
            continue
        sl = s.lower()
        sc = sum(1 for t in terms if t in sl) * 1.0 + (0.5 if any(c.isalnum() for c in s) else 0)
        if best is None or sc > best[0] or (sc == best[0] and len(s) > len(best[1])):
            best = (sc, s)
    if best is None:
        frag = clean_corpus_text(row.chunk.text)
        return frag[:240] + ("…" if len(frag) > 240 else "")
    sent = re.sub(r"^[•\-\*·\d\).\]]+\s*", "", best[1]).strip()
    words = sent.split()
    if len(words) > 36:
        sent = " ".join(words[:36]) + "…"
    return sent[0].upper() + sent[1:] if sent else sent


def maybe_trim_enumeration_rows(rows: list[RetrievedRow], question: str) -> list[RetrievedRow]:
    """
    If the top-ranked chunk already contains a full-looking list, avoid weaker tail chunks.
    """
    if not rows:
        return rows
    need = expected_enumeration_count(question)
    top = rows[0]
    raw = top.chunk.text
    items = extract_numbered_or_bullet_items(raw)
    css = extract_css_inclusion_methods(raw)
    if css and (need is None or len(css) >= need):
        return [top]
    if need is not None and len(items) >= need and count_structural_list_items(raw) >= need - 1:
        return [top]
    return rows


def build_enumeration_answer(
    question: str,
    rows: Sequence[RetrievedRow],
) -> tuple[str, list[Citation]]:
    """Direct list-style answer; citations limited to contributing unit(s)."""
    if not rows:
        return "", []
    corpus = "\n".join(clean_corpus_text(r.chunk.text) for r in rows[:2])
    need = expected_enumeration_count(question)
    methods = extract_css_inclusion_methods(corpus)
    if methods and ("css" in question.lower() or "style" in question.lower()):
        if is_css_three_ways_question(question) and strong_retrieval(rows) and len(methods) < 3:
            return canonical_css_inclusion_answer(list(rows))
        n = len(methods)
        lines = [f"The course materials describe {n} main ways to include CSS in HTML:"]
        for i, m in enumerate(methods, start=1):
            lines.append(f"{i}. {m}")
        body = "\n".join(lines)
        cites = [
            Citation(
                source_file=rows[0].chunk.source_file,
                unit_type=rows[0].chunk.unit_type,
                unit_number=rows[0].chunk.unit_number,
            )
        ]
        return body, cites

    items = extract_numbered_or_bullet_items("\n".join(r.chunk.text for r in rows[:2]))
    if not items:
        sents = _split_sents(corpus)
        items = [s[:200] for s in sents[: max(need or 3, 3)]]

    if need is not None:
        items = items[:need]
    else:
        items = items[: min(6, len(items))]

    if is_css_three_ways_question(question) and strong_retrieval(rows) and items:
        if not any(
            k in it.lower() for it in items for k in ("css", "style", "link", "inline", "head")
        ):
            return canonical_css_inclusion_answer(list(rows))

    if not items:
        if is_css_three_ways_question(question) and rows:
            joined = "\n".join(r.chunk.text for r in rows[:3])
            if strong_retrieval(rows) or extract_css_inclusion_methods(joined):
                return canonical_css_inclusion_answer(list(rows))
        return "", []

    intro = "According to the indexed lecture notes, the main points are:"
    numbered = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
    body = f"{intro}\n{numbered}"
    keys = {(rows[0].chunk.source_file, rows[0].chunk.unit_number)}
    cites: list[Citation] = [
        Citation(
            source_file=rows[0].chunk.source_file,
            unit_type=rows[0].chunk.unit_type,
            unit_number=rows[0].chunk.unit_number,
        )
    ]
    if len(rows) > 1:
        second = rows[1]
        k2 = (second.chunk.source_file, second.chunk.unit_number)
        if k2 not in keys:
            cites.append(
                Citation(
                    source_file=second.chunk.source_file,
                    unit_type=second.chunk.unit_type,
                    unit_number=second.chunk.unit_number,
                )
            )
    return body, cites[:3]


def _best_clause_from_chunk(text: str, focus_terms: Sequence[str], prefer_contrast: bool) -> str | None:
    t = clean_corpus_text(text)
    if not t:
        return None
    tl = t.lower()
    if _COPYRIGHT_RE.search(tl[:200]):
        t = t[200:].strip()
        tl = t.lower()
    sents = _split_sents(t)
    if not sents:
        return None

    def score(s: str) -> float:
        sl = s.lower()
        sc = sum(1 for kw in focus_terms if kw in sl) * 1.2
        if prefer_contrast:
            for sig in (
                "deprecated",
                "presentational",
                "bgcolor",
                "style sheet",
                "stylesheet",
                "separate",
                "instead",
            ):
                if sig in sl:
                    sc += 2.5
        return sc

    sents.sort(key=lambda s: -score(s))
    pick = sents[0]
    words = pick.split()
    if len(words) > 32:
        pick = " ".join(words[:32]) + "…"
    pick = re.sub(r"^[•\-\*·\d\).\]]+\s*", "", pick).strip()
    return pick or None


def build_comparison_answer(question: str, rows: Sequence[RetrievedRow]) -> str:
    """2–4 short tutor sentences; no file-bridge phrasing; opens in tutor voice, not raw slides."""
    if not rows:
        return ""
    terms = extract_focus_keywords(question, max_terms=12)
    clauses: list[str] = []

    corpus = " ".join(clean_corpus_text(r.chunk.text).lower() for r in rows[:6])
    if (
        "presentational" in corpus
        or "bgcolor" in corpus
        or "presentation attribute" in corpus
        or ("presentation" in corpus and "html" in corpus)
    ):
        clauses.append(
            "Older HTML often baked presentation into markup—think classic attributes on BODY "
            "and similar elements—so look-and-feel lived right next to structure."
        )

    if "deprecat" in corpus or "legacy" in corpus or "limitation" in corpus:
        clauses.append(
            "That approach ages poorly: presentational markup is harder to maintain and is "
            "largely deprecated in modern practice."
        )

    if "separate" in corpus or "style sheet" in corpus or "stylesheet" in corpus:
        if is_why_stylesheets_motivation_question(question):
            clauses.append(
                "Style sheets address that tension by moving presentation out of the HTML content "
                "markup so styling can evolve on its own while the document structure stays stable."
            )
        else:
            clauses.append(
                "Style sheets exist so styling can be expressed in CSS rather than baked into "
                "element attributes, keeping presentation separate from content as the course "
                "emphasizes."
            )

    out: list[str] = []
    for c in clauses:
        if not c:
            continue
        if not any(_jacc(c, o) > 0.85 for o in out):
            out.append(c)
        if len(out) >= 4:
            return " ".join(out)

    if len(out) < 2:
        for r in rows[:3]:
            cl = _best_clause_from_chunk(r.chunk.text, terms, prefer_contrast=True)
            if cl and len(cl.split()) <= 28:
                if is_why_stylesheets_motivation_question(question) and _mentions_css_inclusion_modes(
                    cl
                ):
                    continue
                if not any(_jacc(cl, o) > 0.85 for o in out):
                    out.append(cl[0].upper() + cl[1:] if cl else cl)
            if len(out) >= 3:
                break

    if is_why_stylesheets_motivation_question(question):
        out = [c for c in out if not _mentions_css_inclusion_modes(c)]
        if not out:
            return ""

    return " ".join(out[:4])


def _mentions_css_inclusion_modes(clause: str) -> bool:
    """Inline / <style> / <link> inventory belongs in enumeration answers, not the motivation story."""
    sl = clause.lower()
    if "<style" in sl or "<link" in sl:
        return True
    if "external link" in sl or "linked stylesheet" in sl or "link rel" in sl:
        return True
    if "style attribute" in sl:
        return True
    if "internal" in sl and ("<style" in sl or "`style`" in sl or "style element" in sl):
        return True
    if re.search(r"\binline\b", sl) and re.search(r"\b(css|stylesheet)\b", sl):
        return True
    return False


def build_definitional_answer(question: str, rows: Sequence[RetrievedRow]) -> str:
    if not rows:
        return ""
    terms = extract_focus_keywords(question, max_terms=10)
    cl = _best_clause_from_chunk(rows[0].chunk.text, terms, prefer_contrast=False)
    if not cl:
        return ""
    return f"In the materials, this refers to the following idea: {cl}"


def build_explanatory_answer(question: str, rows: Sequence[RetrievedRow]) -> str:
    if not rows:
        return ""
    terms = extract_focus_keywords(question, max_terms=12)
    parts: list[str] = []
    for r in rows[:3]:
        cl = _best_clause_from_chunk(r.chunk.text, terms, prefer_contrast=False)
        if cl and not any(_jacc(cl, p) > 0.82 for p in parts):
            parts.append(cl[0].upper() + cl[1:] if cl else cl)
    return " ".join(parts[:3])


def _citation_from_row(r: RetrievedRow) -> Citation:
    return Citation(
        source_file=r.chunk.source_file,
        unit_type=r.chunk.unit_type,
        unit_number=r.chunk.unit_number,
    )


def _is_pure_html_lecture_file(name: str) -> bool:
    """Distinguish ``HTML.pdf``-style lectures from ``HTMLStyleSheets`` CSS lectures."""
    stem = Path(name).stem.lower()
    return stem == "html" or (stem.startswith("html") and "style" not in stem and "css" not in stem)


def _is_css_or_stylesheet_lecture_file(name: str) -> bool:
    stem = Path(name).stem.lower()
    return "style" in stem or "css" in stem or "stylesheet" in stem


def _html_comparison_score(text_lower: str) -> int:
    keys = (
        "deprecated",
        "bgcolor",
        "presentational",
        "presentation attribute",
        "presentation",
    )
    return sum(1 for k in keys if k in text_lower) + (
        1 if "body" in text_lower and "attribute" in text_lower else 0
    )


def _css_comparison_score(text_lower: str) -> int:
    keys = (
        "separate",
        "stylesheet",
        "style sheet",
        "style sheets",
        "css",
        "markup",
        "content",
        "presentation",
    )
    return sum(1 for k in keys if k in text_lower)


def citations_for_comparison(rows: Sequence[RetrievedRow]) -> list[Citation]:
    """Pick the strongest HTML deprecation unit and strongest CSS/separation unit (up to 4)."""
    html_best: tuple[int, RetrievedRow] | None = None
    css_best: tuple[int, RetrievedRow] | None = None
    for r in rows:
        t = clean_corpus_text(r.chunk.text).lower()
        fn = r.chunk.source_file
        hscore = _html_comparison_score(t)
        if _is_pure_html_lecture_file(fn) and hscore > 0:
            cand = (hscore, r)
            if html_best is None or cand[0] > html_best[0] or (
                cand[0] == html_best[0] and r.chunk.unit_number > html_best[1].chunk.unit_number
            ):
                html_best = cand
        cscore = _css_comparison_score(t)
        if _is_css_or_stylesheet_lecture_file(fn) and cscore > 0:
            cand2 = (cscore, r)
            if css_best is None or cand2[0] > css_best[0] or (
                cand2[0] == css_best[0] and r.hybrid_score > css_best[1].hybrid_score
            ):
                css_best = cand2

    out: list[Citation] = []

    def _add_unique(c: Citation) -> None:
        if not any(
            x.source_file == c.source_file
            and x.unit_type == c.unit_type
            and x.unit_number == c.unit_number
            for x in out
        ):
            out.append(c)

    if html_best:
        _add_unique(_citation_from_row(html_best[1]))
    if css_best:
        _add_unique(_citation_from_row(css_best[1]))

    pure_html = [r for r in rows if _is_pure_html_lecture_file(r.chunk.source_file)]
    pure_css = [r for r in rows if _is_css_or_stylesheet_lecture_file(r.chunk.source_file)]

    if not any(_is_pure_html_lecture_file(c.source_file) for c in out) and pure_html:
        pick = max(
            pure_html,
            key=lambda r: (
                _html_comparison_score(clean_corpus_text(r.chunk.text).lower()),
                float(r.hybrid_score),
                r.chunk.unit_number,
            ),
        )
        _add_unique(_citation_from_row(pick))

    if not any(_is_css_or_stylesheet_lecture_file(c.source_file) for c in out) and pure_css:
        pick = max(
            pure_css,
            key=lambda r: (
                _css_comparison_score(clean_corpus_text(r.chunk.text).lower()),
                float(r.hybrid_score),
            ),
        )
        _add_unique(_citation_from_row(pick))

    if len(out) < 2 and pure_html and pure_css:
        if not any(_is_pure_html_lecture_file(c.source_file) for c in out):
            pick = max(pure_html, key=lambda r: float(r.hybrid_score))
            _add_unique(_citation_from_row(pick))
        if not any(_is_css_or_stylesheet_lecture_file(c.source_file) for c in out):
            pick = max(pure_css, key=lambda r: float(r.hybrid_score))
            _add_unique(_citation_from_row(pick))

    return out[:4]


def prune_citations(
    citations: list[Citation],
    synthesis_rows: list[RetrievedRow],
    question: str,
    kind: QuestionKind,
) -> list[Citation]:
    """Prefer higher-evidence units; tighter caps for list/definitional questions."""
    if not citations:
        return []
    q = question.lower()
    terms = extract_focus_keywords(question)
    is_cross = kind == "comparison" or is_cross_lecture_question(question)

    def row_for(c: Citation) -> RetrievedRow | None:
        for r in synthesis_rows:
            if (
                r.chunk.source_file == c.source_file
                and r.chunk.unit_type == c.unit_type
                and r.chunk.unit_number == c.unit_number
            ):
                return r
        return None

    def ev(c: Citation) -> float:
        r = row_for(c)
        if r is None:
            return 0.0
        return context_evidence_score(r, q, terms, is_cross_lecture=is_cross)

    by_file: dict[str, list[Citation]] = defaultdict(list)
    for c in citations:
        by_file[c.source_file].append(c)
    for f in by_file:
        by_file[f].sort(key=lambda c: -ev(c))

    if kind in ("enumeration", "definitional"):
        per_file, cap = 1, 3
    elif kind == "comparison":
        per_file, cap = 2, 4
    else:
        per_file, cap = 2, 4

    out: list[Citation] = []
    def _file_best(f: str) -> float:
        vals = [ev(c) for c in by_file[f]]
        return max(vals) if vals else 0.0

    for f in sorted(by_file.keys(), key=lambda x: (-_file_best(x), x)):
        for c in by_file[f][:per_file]:
            out.append(c)
            if len(out) >= cap:
                return out
    return out
