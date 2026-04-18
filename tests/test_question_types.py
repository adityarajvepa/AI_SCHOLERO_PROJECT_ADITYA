"""Question classification for fallback templates."""

from app.generation.question_types import classify_question, expected_enumeration_count


def test_three_ways_css_is_enumeration_not_definitional() -> None:
    q = "What are the three ways CSS can be included in HTML?"
    assert expected_enumeration_count(q) == 3
    assert classify_question(q, is_cross_from_api=False) == "enumeration"


def test_style_sheets_vs_html_is_comparison() -> None:
    q = "Why were style sheets introduced if HTML already had presentation-related features?"
    assert classify_question(q, is_cross_from_api=True) == "comparison"
