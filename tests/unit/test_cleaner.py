"""
tests/unit/test_cleaner.py — TextCleaner Unit Tests
"""

import pytest
from services.ingestion.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestTextCleaner:

    def test_empty_string_returns_empty(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean("   \n\n  ") == ""

    def test_fixes_hyphenation(self, cleaner):
        text = "computa-\ntional methods are impor-\ntant."
        result = cleaner.clean(text)
        assert "computational" in result
        assert "important" in result
        assert "-\n" not in result

    def test_removes_page_numbers(self, cleaner):
        text = "Some content here.\nPage 5\nMore content."
        result = cleaner.clean(text)
        assert "Page 5" not in result
        assert "Some content here." in result

    def test_normalizes_unicode_dashes(self, cleaner):
        text = "The algorithm\u2014developed in 2020\u2014works well."
        result = cleaner.clean(text)
        assert "\u2014" not in result
        assert "-" in result

    def test_normalizes_curly_quotes(self, cleaner):
        text = "\u201cThis is quoted\u201d and \u2018this too\u2019."
        result = cleaner.clean(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"' in result

    def test_collapses_multiple_blank_lines(self, cleaner):
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        result = cleaner.clean(text)
        # Should have max 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_collapses_multiple_spaces(self, cleaner):
        text = "This   has     many   spaces."
        result = cleaner.clean(text)
        assert "  " not in result

    def test_preserves_meaningful_content(self, cleaner):
        text = "The algorithm achieves 95.3% accuracy on the test dataset."
        result = cleaner.clean(text)
        assert "95.3%" in result
        assert "algorithm" in result

    def test_non_breaking_space_replaced(self, cleaner):
        text = "Hello\u00a0World"
        result = cleaner.clean(text)
        assert "\u00a0" not in result
        assert "Hello World" in result


class TestTextCleanerCitation:
    """Tests for citation-specific cleaning."""

    def test_page_of_pattern_removed(self, cleaner):
        text = "Some text.\n- 5 -\nMore text."
        result = cleaner.clean(text)
        assert "- 5 -" not in result

    def test_page_number_pattern_removed(self, cleaner):
        text = "Content.\nPage 12 of 50\nMore content."
        result = cleaner.clean(text)
        assert "Page 12 of 50" not in result
