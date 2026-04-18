"""Heuristics for formula-heavy and sparse/image-heavy units."""

from __future__ import annotations

import re

from app.utils.text import char_count_non_ws, word_count


_MATH_SYMBOL_RE = re.compile(
    r"[∑∫√≤≥≠±×÷∂∇α-ωΑ-Ω∞→←⇒⇔]"
)
_LATEXISH_RE = re.compile(
    r"(\\frac|\\sum|\\int|\\sqrt|\\cdot|\\times|\\\(|\\\)|\\\[|\\\]|"
    r"\\mathrm|\\mathbf|\\mathbb|\\mathcal|\\text|\\left|\\right|"
    r"\^\{|_\{|\\begin|\\end)"
)
_EQUATIONISH_RE = re.compile(
    r"(=){2,}|\\quad|\\qquad|\$\$|\$[^\$]+\$"
)


def detect_formula_heavy(text: str) -> bool:
    """True if text likely contains substantial mathematical notation."""
    if not text.strip():
        return False
    wc = word_count(text)
    sym = len(_MATH_SYMBOL_RE.findall(text))
    lx = len(_LATEXISH_RE.findall(text))
    eq = len(_EQUATIONISH_RE.findall(text))
    score = sym * 2 + lx * 3 + eq * 2
    ratio = score / max(1, wc)
    return score >= 8 or ratio >= 0.35


def detect_sparse_or_image_heavy(text: str) -> bool:
    """True if extracted text is unusually thin (possible diagrams/images)."""
    t = text.strip()
    if not t:
        return True
    wc = word_count(t)
    cc = char_count_non_ws(t)
    if wc < 25 and cc < 180:
        return True
    if wc < 40 and cc / max(1, wc) < 4.5:
        return True
    return False
