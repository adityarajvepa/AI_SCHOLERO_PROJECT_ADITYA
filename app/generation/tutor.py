"""Grounded tutor responses: Gemini via Google ADK or deterministic fallback."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.generation.context_select import contrast_bonus_for_sentence, select_synthesis_rows
from app.generation.fallback_templates import (
    build_definitional_answer,
    build_enumeration_answer,
    build_explanatory_answer,
    clean_corpus_text,
    citations_for_comparison,
    ensure_comparison_answer,
    maybe_trim_enumeration_rows,
    minimal_clause_from_top_chunk,
    prune_citations,
    strong_retrieval,
)
from app.generation.question_types import QuestionKind, classify_question, is_definitional_kind
from app.retrieval.hybrid import RetrievedRow
from app.retrieval.concept_coverage import should_abstain_insufficient_concept_evidence
from app.retrieval.query_plan import should_abstain_for_keyword_gap
from app.schemas import AskDebug, AskResponse, Citation, DebugRetrievedChunk
from app.utils.logging import get_logger
from app.utils.text import truncate_preview

if TYPE_CHECKING:
    from app.retrieval.embeddings import EmbeddingModel

logger = get_logger(__name__)

SUPPORT_THRESHOLD = 0.12
MAX_RESPONSE_CITATIONS = 4
CHUNKS_FOR_SYNTHESIS = 5
CHUNKS_FOR_SYNTHESIS_CROSS = 7
MAX_SENTENCES_PER_CHUNK = 2
MAX_SYNTHESIS_CHARS = 900
MIN_SENTENCE_CHARS = 18
DEFINITIONAL_MAX_CHARS = 520

ADK_TUTOR_INSTRUCTION = """You are Scholera, a course AI tutor.
Each user message includes QUESTION and COURSE_CONTEXT (numbered excerpts from uploaded course materials).

Rules:
- Answer ONLY when the excerpts directly support the asked concepts (names, techniques, definitions). If the
  passages are only loosely related to the question topic, say the materials do not clearly cover it—do not
  force-fit unrelated slides.
- Never reinterpret unrelated passages as definitions of the user's terms, and do not invent bridging lines
  like "in the materials this refers to…" unless the cited text is explicitly about that concept.
- Ground every factual claim in COURSE_CONTEXT; do not contradict those excerpts.
- You may use the google_search tool only for supporting facts that are clearly separate from the course
  index; never use web text to override missing course evidence.
- If evidence is weak or off-topic, state that plainly.
- When you do answer from COURSE_CONTEXT, end with a short "Sources:" line listing source_file and
  page/slide numbers you used.
- Be concise and precise."""


def _top_citations(rows: Sequence[RetrievedRow], limit: int = MAX_RESPONSE_CITATIONS) -> list[Citation]:
    """Most relevant sources first (hybrid order), deduped, capped for a clean demo."""
    seen: set[tuple[str, str, int]] = set()
    out: list[Citation] = []
    for r in rows:
        key = (r.chunk.source_file, r.chunk.unit_type, r.chunk.unit_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Citation(
                source_file=r.chunk.source_file,
                unit_type=r.chunk.unit_type,
                unit_number=r.chunk.unit_number,
            )
        )
        if len(out) >= limit:
            break
    return out


def _rows_to_debug(rows: Sequence[RetrievedRow]) -> AskDebug:
    debug_chunks = [
        DebugRetrievedChunk(
            chunk_id=r.chunk.chunk_id,
            source_file=r.chunk.source_file,
            unit_type=r.chunk.unit_type,
            unit_number=r.chunk.unit_number,
            hybrid_score=r.hybrid_score,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
            preview=truncate_preview(r.chunk.text, 280),
        )
        for r in rows
    ]
    return AskDebug(retrieved_chunks=debug_chunks)


def _strip_slide_markers(s: str) -> str:
    """Remove common slide/bullet prefixes so lines read as prose."""
    x = s.strip()
    x = re.sub(r"^[\s•\-\*·]+", "", x)
    x = re.sub(r"^\d+[\).\]]\s*", "", x)
    return x.strip()


def _simplify_heavy_bullet_sentence(s: str) -> str:
    """If a sentence is mostly bullet fragments, keep the strongest clause."""
    if s.count("•") < 2 and s.count("\n") < 3:
        return _strip_slide_markers(s)
    parts = [p.strip() for p in re.split(r"[•\n]", s) if len(p.strip()) >= MIN_SENTENCE_CHARS]
    if not parts:
        return _strip_slide_markers(s)
    return _strip_slide_markers(max(parts, key=len))


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split for extractive synthesis."""
    t = " ".join(text.split())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if len(s) >= MIN_SENTENCE_CHARS:
            out.append(s)
    return out


def _lexical_overlap_score(question: str, sentence: str) -> float:
    qw = set(re.findall(r"[a-z0-9]+", question.lower()))
    sw = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    if not qw:
        return 0.0
    return len(qw & sw) / max(len(qw), 1)


def _cosine_scores_matrix(
    sentence_embs: np.ndarray, query_vec: np.ndarray
) -> np.ndarray:
    q = query_vec.astype(np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-12)
    m = sentence_embs.astype(np.float32)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return (m @ q).astype(np.float32)


def _jaccard_words(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def _too_similar_to_any(candidate: str, chosen: Sequence[str]) -> bool:
    c = candidate.lower()
    for prev in chosen:
        p = prev.lower()
        if c in p or p in c:
            return True
        if _jaccard_words(candidate, prev) > 0.82:
            return True
    return False


def _format_reviewed_sources(rows: Sequence[RetrievedRow], limit: int = 4) -> str:
    """Human-readable closest hits for abstention (not evidentiary citations)."""
    lines: list[str] = []
    for i, r in enumerate(rows[:limit], start=1):
        unit = "p." if r.chunk.unit_type == "page" else "slide"
        prev = truncate_preview(r.chunk.text, 140)
        lines.append(
            f"{i}. {r.chunk.source_file}, {unit} {r.chunk.unit_number} "
            f"(hybrid {r.hybrid_score:.3f}) — {prev}"
        )
    return "\n".join(lines)


def build_abstention_answer(question: str, rows: Sequence[RetrievedRow], reason: str) -> str:
    """Grounded abstention: explain the gap and list closest retrieval without implying support."""
    q_short = truncate_preview(question, 220)
    reviewed = _format_reviewed_sources(rows)
    return (
        f"The indexed course materials do not appear to substantively answer this question: "
        f"{q_short}\n\n{reason}\n\n"
        f"Reviewed sources (closest retrieval — not used as supporting evidence):\n{reviewed}\n\n"
        "See debug.retrieved_chunks in the API response for full excerpts."
    )


def _format_sources_lines(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""
    lines: list[str] = ["Sources:"]
    for c in citations:
        unit = "p." if c.unit_type == "page" else "slide"
        lines.append(f"• {c.source_file}, {unit} {c.unit_number}")
    return "\n".join(lines)


def _append_sources_block(body: str, citations: Sequence[Citation]) -> str:
    """Append a Sources block only when there is at least one citation."""
    src = _format_sources_lines(citations)
    out = body.rstrip()
    if not src:
        return out
    return f"{out}\n\n{src}"


def synthesize_fallback_answer(
    question: str,
    rows: Sequence[RetrievedRow],
    query_embedding: np.ndarray | None,
    embedder: EmbeddingModel | None,
    footer_citations: list[Citation],
    *,
    is_cross_lecture: bool = False,
    question_kind: QuestionKind = "general",
) -> tuple[str, list[Citation] | None]:
    """
    Tutor-style grounded answer. Returns (answer text, optional citation override).

    When the second value is not ``None``, callers should prefer it over precomputed citations.
    """
    if not rows:
        msg = (
            "I could not find relevant material in the uploaded course documents for this "
            f"question: {question!r}. Try uploading more lectures or rephrasing."
        )
        return _append_sources_block(msg, footer_citations), None

    if question_kind == "enumeration":
        body, cites = build_enumeration_answer(question, rows)
        if body.strip():
            return _append_sources_block(body, cites), cites
        if rows and strong_retrieval(rows):
            frag = minimal_clause_from_top_chunk(rows[0], question)
            if frag.strip():
                return _append_sources_block(frag, footer_citations), None
        weak = (
            "The indexed slides do not list clear extractable points for this question. "
            "Use the Sources list and debug.retrieved_chunks to review excerpts."
        )
        return _append_sources_block(weak, footer_citations), None

    if question_kind == "comparison":
        body = ensure_comparison_answer(question, rows)
        if not body.strip() and rows and strong_retrieval(rows):
            body = minimal_clause_from_top_chunk(rows[0], question)
        if body.strip():
            cites = citations_for_comparison(rows)
            return _append_sources_block(body, cites), cites
        weak = (
            "The retrieved materials touch related themes, but a focused comparison could not "
            "be grounded from extractable lines. Use Sources and debug.retrieved_chunks."
        )
        return _append_sources_block(weak, footer_citations), None

    if question_kind == "definitional":
        body = build_definitional_answer(question, rows)
        if not body.strip() and rows and strong_retrieval(rows):
            cl = minimal_clause_from_top_chunk(rows[0], question)
            if cl.strip():
                body = f"In the materials, this refers to the following idea: {cl}"
        if body.strip():
            return _append_sources_block(body, footer_citations), None
        weak = (
            "A concise definition could not be extracted from the top hits. "
            "Use Sources and debug.retrieved_chunks to verify wording in the materials."
        )
        return _append_sources_block(weak, footer_citations), None

    if question_kind == "explanatory":
        body = build_explanatory_answer(question, rows)
        if not body.strip() and rows and strong_retrieval(rows):
            body = minimal_clause_from_top_chunk(rows[0], question)
        if body.strip():
            return _append_sources_block(body, footer_citations), None
        weak = (
            "An explanatory summary could not be extracted cleanly from the retrieved excerpts. "
            "Use Sources and debug.retrieved_chunks to read the underlying pages."
        )
        return _append_sources_block(weak, footer_citations), None

    # --- general: light extractive compression, no file-bridge phrasing ---
    use_dense = embedder is not None and query_embedding is not None
    chosen_texts: list[str] = []
    chosen_pairs: list[tuple[str, str]] = []
    rows_mine = list(rows)
    max_total_sents = 4
    max_per_chunk = MAX_SENTENCES_PER_CHUNK

    for r in rows_mine:
        if len(chosen_pairs) >= max_total_sents:
            break
        sents = _split_sentences(clean_corpus_text(r.chunk.text))
        if not sents:
            frag = clean_corpus_text(r.chunk.text).strip()
            if len(frag) >= MIN_SENTENCE_CHARS:
                sents = [frag[:400] + ("…" if len(frag) > 400 else "")]
            else:
                continue
        if use_dense:
            emb = embedder.encode(sents)  # type: ignore[union-attr]
            base = _cosine_scores_matrix(emb, query_embedding)  # type: ignore[arg-type]
            bonus = np.array(
                [
                    contrast_bonus_for_sentence(s.lower(), is_cross_lecture=is_cross_lecture)
                    for s in sents
                ],
                dtype=np.float32,
            )
            scores = base + bonus
        else:
            scores = np.array(
                [
                    _lexical_overlap_score(question, s)
                    + contrast_bonus_for_sentence(s.lower(), is_cross_lecture=is_cross_lecture)
                    for s in sents
                ],
                dtype=np.float32,
            )
        order = np.argsort(-scores, kind="stable")
        picked = 0
        for idx in order:
            if picked >= max_per_chunk or len(chosen_pairs) >= max_total_sents:
                break
            raw = sents[int(idx)]
            s = _simplify_heavy_bullet_sentence(raw)
            if len(s) < MIN_SENTENCE_CHARS:
                continue
            if use_dense and float(scores[int(idx)]) < 0.08:
                continue
            if not use_dense and float(scores[int(idx)]) < 0.04:
                continue
            if _too_similar_to_any(s, chosen_texts):
                continue
            chosen_texts.append(s)
            chosen_pairs.append((s, r.chunk.source_file))
            picked += 1

    if not chosen_pairs:
        body = (
            "The retrieved materials look related, but no clear extractable sentences "
            "passed the relevance threshold. Use the Sources list below and the "
            "debug.retrieved_chunks field in the API response to inspect raw excerpts."
        )
        return _append_sources_block(body, footer_citations), None

    body = ". ".join(s for s, _ in chosen_pairs)
    if body and body[-1] not in ".!?":
        body += "."
    cap = DEFINITIONAL_MAX_CHARS if is_definitional_kind(question_kind) else MAX_SYNTHESIS_CHARS
    if len(body) > cap:
        body = body[: cap - 1].rsplit(" ", 1)[0] + "…"
    return _append_sources_block(body, footer_citations), None


def _weak_evidence(rows: Sequence[RetrievedRow]) -> bool:
    if not rows:
        return True
    return float(rows[0].hybrid_score) < SUPPORT_THRESHOLD


@lru_cache(maxsize=16)
def _scholera_root_agent(model: str):
    from google.adk.agents.llm_agent import Agent
    from google.adk.tools import google_search

    return Agent(
        model=model,
        name="root_agent",
        description="A helpful assistant for user questions.",
        instruction=ADK_TUTOR_INSTRUCTION,
        tools=[google_search],
    )


def _content_to_text(content: object | None) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    out: list[str] = []
    for part in content.parts:  # type: ignore[union-attr]
        if part.text and not getattr(part, "thought", False):
            out.append(part.text)
    return "\n".join(out).strip()


async def _run_gemini_adk_chat(user_prompt: str, model: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.adk.utils.context_utils import Aclosing
    from google.genai import types as genai_types

    agent = _scholera_root_agent(model)
    async with InMemoryRunner(agent=agent, app_name="scholera") as runner:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="scholera",
            session_id=str(uuid.uuid4()),
        )
        new_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_prompt)],
        )
        last_text = ""
        async with Aclosing(
            runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=new_message,
            )
        ) as agen:
            async for event in agen:
                if event.content:
                    t = _content_to_text(event.content)
                    if t:
                        last_text = t
        return last_text


def _call_gemini_adk(
    settings: Settings,
    user_prompt: str,
    timeout_s: float = 120.0,
) -> str | None:
    if not settings.google_api_key:
        return None
    prev = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    try:

        async def _with_timeout() -> str:
            return await asyncio.wait_for(
                _run_gemini_adk_chat(user_prompt, settings.llm_model),
                timeout=timeout_s,
            )

        text = asyncio.run(_with_timeout())
        return text.strip() or None
    except asyncio.TimeoutError:
        logger.warning("Gemini ADK call timed out after %s s; using fallback.", timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini ADK call failed; using fallback. Reason: %s", exc)
        return None
    finally:
        if prev is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = prev


def build_context_block(rows: Sequence[RetrievedRow]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(rows, start=1):
        header = (
            f"[{i}] source_file={r.chunk.source_file} "
            f"unit={r.chunk.unit_type} {r.chunk.unit_number}"
        )
        blocks.append(header + "\n" + r.chunk.text.strip())
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    rows: list[RetrievedRow],
    settings: Settings | None = None,
    *,
    query_embedding: np.ndarray | None = None,
    embedder: EmbeddingModel | None = None,
    is_cross_lecture: bool = False,
) -> AskResponse:
    """Produce grounded answer using Gemini (ADK) if configured, else deterministic fallback."""
    cfg = settings or get_settings()
    debug = _rows_to_debug(rows)

    if not rows:
        return AskResponse(
            answer="No course chunks were available to retrieve for this question.",
            citations=[],
            debug=debug,
        )

    kind = classify_question(question, is_cross_from_api=is_cross_lecture)

    abstain_concept, reason_concept = should_abstain_insufficient_concept_evidence(
        question,
        rows,
        kind=kind,
        query_embedding=query_embedding,
        embedder=embedder,
    )
    if abstain_concept:
        return AskResponse(
            answer=build_abstention_answer(question, rows, reason_concept),
            citations=[],
            debug=debug,
        )

    abstain, abstain_reason = should_abstain_for_keyword_gap(question, rows)
    if abstain:
        return AskResponse(
            answer=build_abstention_answer(question, rows, abstain_reason),
            citations=[],
            debug=debug,
        )
    synth_limit = CHUNKS_FOR_SYNTHESIS_CROSS if is_cross_lecture else CHUNKS_FOR_SYNTHESIS
    synthesis_rows = select_synthesis_rows(
        rows,
        question,
        is_cross_lecture=is_cross_lecture,
        max_chunks=synth_limit,
    )
    if kind == "enumeration":
        synthesis_rows = maybe_trim_enumeration_rows(synthesis_rows, question)

    cite_cap = 3 if kind in ("enumeration", "definitional") else MAX_RESPONSE_CITATIONS
    base_cites = _top_citations(synthesis_rows, cite_cap)
    citations = prune_citations(list(base_cites), synthesis_rows, question, kind)

    if _weak_evidence(rows):
        caveat = (
            f"The match to your question looks weak (top hybrid score "
            f"{rows[0].hybrid_score:.3f} < {SUPPORT_THRESHOLD}). "
            "The summary below is tentative—verify in the cited pages.\n\n"
        )
        body, cite_override = synthesize_fallback_answer(
            question,
            synthesis_rows,
            query_embedding,
            embedder,
            citations,
            is_cross_lecture=is_cross_lecture,
            question_kind=kind,
        )
        final_cites = (
            prune_citations(list(cite_override), synthesis_rows, question, kind)
            if cite_override
            else citations
        )
        return AskResponse(answer=caveat + body, citations=final_cites, debug=debug)

    context = build_context_block(synthesis_rows)
    user_prompt = (
        f"QUESTION:\n{question}\n\nCOURSE_CONTEXT:\n{context}\n\n"
        "Answer only if the excerpts directly support the asked concepts. If the context is tangential or "
        "does not define/explain them, say the materials do not clearly cover the topic—do not reinterpret "
        "unrelated text as the answer."
    )
    llm_text = _call_gemini_adk(cfg, user_prompt)
    if llm_text:
        return AskResponse(answer=llm_text, citations=citations, debug=debug)
    answer, cite_override = synthesize_fallback_answer(
        question,
        synthesis_rows,
        query_embedding,
        embedder,
        citations,
        is_cross_lecture=is_cross_lecture,
        question_kind=kind,
    )
    final_cites = (
        prune_citations(list(cite_override), synthesis_rows, question, kind)
        if cite_override
        else citations
    )
    return AskResponse(answer=answer, citations=final_cites, debug=debug)
