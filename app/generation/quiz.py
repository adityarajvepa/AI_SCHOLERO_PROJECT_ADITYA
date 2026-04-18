"""Grounded quiz generation: retrieval + Gemini (ADK) over existing course chunks only."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from app.config import Settings
from app.retrieval.hybrid import RetrievedRow, hybrid_search
from app.retrieval.query_plan import diversify_by_source_file
from app.schemas import QuizCitationItem, QuizItem, QuizResponse
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.retrieval.embeddings import EmbeddingModel

logger = get_logger(__name__)


def _json_int(v: Any) -> int | None:
    """Coerce JSON numbers (int/float) or numeric strings to int for page/slide validation."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v == v:  # NaN
            return None
        r = int(round(v))
        if abs(v - r) < 1e-6:
            return r
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


QUIZ_AGENT_INSTRUCTION = """You are Scholera's quiz generator.
You receive ONLY numbered COURSE_EXCERPTS from a professor's uploaded materials.
You must output ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{
  "quiz": [
    {
      "question": "...",
      "answer_key": "...",
      "citations": [
        {"source_file": "exact filename from excerpt header", "page": 12},
        {"source_file": "...", "slide": 5}
      ]
    }
  ]
}
Rules:
- Use "page" with an integer for unit_type page, or "slide" with an integer for unit_type slide — never invent numbers.
- Every citation must correspond to an excerpt you actually used; source_file strings must match exactly
  (including extension) one of the excerpt headers.
- Generate at most the requested number of questions; if evidence is thin, output fewer questions rather than guessing.
- Questions must be answerable directly from the excerpts; no outside knowledge, no web search.
- Match difficulty: easy = recall/definitions; medium = understanding/compare within excerpts; hard = synthesis or
  multi-excerpt reasoning only when the excerpts support it.
- answer_key should be concise (phrase or short paragraph).
"""


@lru_cache(maxsize=8)
def _quiz_agent(model: str):
    from google.adk.agents.llm_agent import Agent

    return Agent(
        model=model,
        name="quiz_agent",
        description="Generates grounded quizzes from course excerpts.",
        instruction=QUIZ_AGENT_INSTRUCTION,
        tools=[],
    )


def _content_to_text(content: object | None) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    out: list[str] = []
    for part in content.parts:  # type: ignore[union-attr]
        if part.text and not getattr(part, "thought", False):
            out.append(part.text)
    return "\n".join(out).strip()


async def _run_quiz_adk(user_prompt: str, model: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.adk.utils.context_utils import Aclosing
    from google.genai import types as genai_types

    agent = _quiz_agent(model)
    async with InMemoryRunner(agent=agent, app_name="scholera_quiz") as runner:
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


def _call_quiz_llm(
    settings: Settings, user_prompt: str, *, timeout_s: float = 180.0
) -> tuple[str | None, str | None]:
    """
    Returns (response_text, None) on success, or (None, user_facing_error) on failure.
    """
    if not settings.google_api_key:
        return None, None
    prev = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    try:

        async def _with_timeout() -> str:
            return await asyncio.wait_for(
                _run_quiz_adk(user_prompt, settings.llm_model),
                timeout=timeout_s,
            )

        text = asyncio.run(_with_timeout())
        t = text.strip() or None
        return t, None
    except asyncio.TimeoutError:
        logger.warning("Quiz ADK call timed out after %s s", timeout_s)
        return None, (
            f"Quiz generation timed out after {int(timeout_s)}s. "
            "Try fewer questions, a smaller top_k, or retry."
        )
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        logger.warning("Quiz ADK call failed: %s", detail[:800])
        dlow = detail.lower()
        if "429" in detail or "resource_exhausted" in dlow or "quota" in dlow:
            return None, (
                "Gemini API quota or rate limit was hit (HTTP 429). "
                "The free tier allows a limited number of requests per day per model (e.g. gemini-2.5-flash). "
                "Wait and retry, reduce other Gemini usage today, switch LLM_MODEL to another model with quota, "
                "or enable billing for higher limits. "
                "See https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        return None, "Quiz generation failed. Check server logs for the underlying error."
    finally:
        if prev is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = prev


def build_quiz_excerpt_block(rows: Sequence[RetrievedRow]) -> str:
    """Numbered excerpts with chunk_id for model grounding (ids used only in prompt, not required in JSON out)."""
    blocks: list[str] = []
    for i, r in enumerate(rows, start=1):
        header = (
            f"[{i}] chunk_id={r.chunk.chunk_id} source_file={r.chunk.source_file} "
            f"unit={r.chunk.unit_type} {r.chunk.unit_number}"
        )
        blocks.append(header + "\n" + r.chunk.text.strip())
    return "\n\n".join(blocks)


def retrieve_quiz_context(
    chunks: list,
    matrix: np.ndarray,
    embedder: EmbeddingModel,
    settings: Settings,
    *,
    topic: str | None,
    top_k: int,
) -> list[RetrievedRow]:
    """Hybrid retrieval; diversified across files when no topic is given."""
    w_d, w_b = settings.hybrid_weights
    if topic and topic.strip():
        q_text = topic.strip()
        q_emb = embedder.encode([q_text])[0]
        pool = max(top_k, min(len(chunks), top_k * 2))
        rows = hybrid_search(chunks, matrix, q_emb, q_text, w_d, w_b, pool)
        return rows[:top_k]
    # Broad coverage: synthetic query + diversification across lectures/files.
    q_text = "key concepts definitions examples methods covered in the course materials"
    q_emb = embedder.encode([q_text])[0]
    pool = max(15, min(len(chunks), top_k * 3))
    rows = hybrid_search(chunks, matrix, q_emb, q_text, w_d, w_b, pool)
    return diversify_by_source_file(rows, top_k)


def _norm_src(name: str) -> str:
    return name.replace("\\", "/").strip()


def _src_matches(cited: str, allowed: str) -> bool:
    a, b = _norm_src(cited), _norm_src(allowed)
    if a == b:
        return True
    return Path(a).name == Path(b).name


def _allowed_citation_keys(rows: Sequence[RetrievedRow]) -> set[tuple[str, str, int]]:
    return {(r.chunk.source_file, r.chunk.unit_type, int(r.chunk.unit_number)) for r in rows}


def _canonical_source_file(rows: Sequence[RetrievedRow], cited: str) -> str | None:
    for r in rows:
        if _src_matches(cited, r.chunk.source_file):
            return r.chunk.source_file
    return None


def _validate_citation(
    cite: dict[str, Any],
    allowed: set[tuple[str, str, int]],
    rows: Sequence[RetrievedRow],
) -> QuizCitationItem | None:
    sf_raw = cite.get("source_file")
    if not isinstance(sf_raw, str) or not sf_raw.strip():
        return None
    canon = _canonical_source_file(rows, sf_raw.strip())
    if not canon:
        return None
    page = _json_int(cite.get("page"))
    slide = _json_int(cite.get("slide"))
    ut: str | None = None
    un: int | None = None
    if page is not None:
        ut, un = "page", page
    elif slide is not None:
        ut, un = "slide", slide
    else:
        return None
    key = (canon, ut, un)
    if key not in allowed:
        return None
    if ut == "page":
        return QuizCitationItem(source_file=canon, page=un, slide=None)
    return QuizCitationItem(source_file=canon, page=None, slide=un)


def _parse_quiz_json(raw: str) -> list[dict[str, Any]]:
    t = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", t)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    quiz = data.get("quiz")
    if not isinstance(quiz, list):
        return []
    return [x for x in quiz if isinstance(x, dict)]


def _sanitize_quiz_items(
    raw_items: list[dict[str, Any]],
    rows: Sequence[RetrievedRow],
    max_questions: int,
) -> tuple[list[QuizItem], str | None]:
    allowed = _allowed_citation_keys(rows)
    out: list[QuizItem] = []
    for item in raw_items:
        if len(out) >= max_questions:
            break
        q = item.get("question")
        ak = item.get("answer_key")
        if not isinstance(q, str) or not isinstance(ak, str):
            continue
        q, ak = q.strip(), ak.strip()
        if len(q) < 8 or len(ak) < 2:
            continue
        cites_raw = item.get("citations")
        cites_out: list[QuizCitationItem] = []
        if isinstance(cites_raw, list):
            for c in cites_raw:
                if isinstance(c, dict):
                    vc = _validate_citation(c, allowed, rows)
                    if vc:
                        cites_out.append(vc)
        if not cites_out:
            continue
        out.append(QuizItem(question=q, answer_key=ak, citations=cites_out))
    note = None
    if len(out) < max_questions and raw_items:
        note = (
            f"Returned {len(out)} grounded question(s) (fewer than {max_questions} requested) "
            "after validating citations against retrieved excerpts."
        )
    return out, note


def generate_quiz_response(
    course_id: str,
    rows: list[RetrievedRow],
    settings: Settings,
    *,
    topic: str | None,
    difficulty: str,
    num_questions: int,
) -> QuizResponse:
    """Call Gemini with retrieval-backed excerpts; validate JSON and citations."""
    if not rows:
        return QuizResponse(
            course_id=course_id,
            topic=topic,
            difficulty=difficulty,
            quiz=[],
            message="No retrieved excerpts available for this course or topic.",
        )
    if not settings.google_api_key:
        return QuizResponse(
            course_id=course_id,
            topic=topic,
            difficulty=difficulty,
            quiz=[],
            message="Set GOOGLE_API_KEY in the environment to generate quizzes with Gemini.",
        )

    excerpts = build_quiz_excerpt_block(rows)
    user_prompt = (
        f"REQUESTED_QUESTIONS: {num_questions}\n"
        f"DIFFICULTY: {difficulty}\n"
        f"TOPIC_FOCUS: {topic.strip() if topic else 'broad coverage across the excerpts below'}\n\n"
        f"COURSE_EXCERPTS:\n{excerpts}\n\n"
        "Return JSON only as specified in your instructions. "
        "If the excerpts cannot support a full quiz, return fewer items in the quiz array."
    )
    raw, llm_err = _call_quiz_llm(settings, user_prompt)
    if not raw:
        return QuizResponse(
            course_id=course_id,
            topic=topic,
            difficulty=difficulty,
            quiz=[],
            message=llm_err
            or "Quiz generation failed or timed out. Try fewer questions, a narrower topic, or retry.",
        )
    parsed = _parse_quiz_json(raw)
    items, note = _sanitize_quiz_items(parsed, rows, num_questions)
    msg = note
    if not items:
        msg = (
            (msg + " " if msg else "")
            + "No valid grounded questions were produced. Try a different topic or increase top_k."
        ).strip()
    top_h = float(rows[0].hybrid_score) if rows else 0.0
    if items and top_h < 0.08 and not (topic and topic.strip()):
        msg = ((msg + " — ") if msg else "") + "Retrieval confidence was low; verify questions against the cited units."

    return QuizResponse(
        course_id=course_id,
        topic=topic,
        difficulty=difficulty,
        quiz=items,
        message=msg,
    )
