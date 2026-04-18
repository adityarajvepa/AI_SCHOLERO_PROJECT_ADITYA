"""Tests for heuristic detectors."""

from app.ingestion.detectors import detect_formula_heavy, detect_sparse_or_image_heavy


def test_formula_heavy_detects_latexish() -> None:
    t = r"The loss $\mathcal{L} = \sum_i \|y_i - \hat y_i\|^2$ uses \frac{a}{b} and \int_0^1 x dx"
    assert detect_formula_heavy(t) is True


def test_formula_light_text() -> None:
    t = "This lecture introduces gradients and optimization without heavy notation."
    assert detect_formula_heavy(t) is False


def test_sparse_slide() -> None:
    assert detect_sparse_or_image_heavy("") is True
    assert detect_sparse_or_image_heavy("ok") is True
    t = " ".join(["paragraph"] * 80)
    assert detect_sparse_or_image_heavy(t) is False
