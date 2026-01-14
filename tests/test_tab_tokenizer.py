"""Tests for tab sheet tokenizer."""

import pytest

from chord_parser.tab_parser.tokenizer import tokenize_line


class TestTokenizeLineBasic:
    """Basic tokenization tests."""

    def test_simple_tokens(self) -> None:
        """Test basic two-token line."""
        tokens = tokenize_line("Gm     C")
        assert len(tokens) == 2
        assert tokens[0].text == "Gm"
        assert tokens[1].text == "C"

    def test_token_spans(self) -> None:
        """Test that token spans are correct."""
        tokens = tokenize_line("Gm     C")
        assert tokens[0].start == 0
        assert tokens[0].end == 2
        assert tokens[1].start == 7
        assert tokens[1].end == 8

    def test_hello_world(self) -> None:
        """Test lyric-like content."""
        tokens = tokenize_line("Hello  world")
        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[0].start == 0
        assert tokens[0].end == 5
        assert tokens[1].text == "world"
        assert tokens[1].start == 7
        assert tokens[1].end == 12


class TestTokenizeLineWhitespace:
    """Whitespace handling tests."""

    def test_leading_spaces(self) -> None:
        """Test line with leading spaces."""
        tokens = tokenize_line("   Hello")
        assert len(tokens) == 1
        assert tokens[0].text == "Hello"
        assert tokens[0].start == 3
        assert tokens[0].end == 8

    def test_trailing_spaces(self) -> None:
        """Test line with trailing spaces."""
        tokens = tokenize_line("Hello   ")
        assert len(tokens) == 1
        assert tokens[0].text == "Hello"

    def test_empty_line(self) -> None:
        """Test empty line returns no tokens."""
        tokens = tokenize_line("")
        assert len(tokens) == 0

    def test_whitespace_only(self) -> None:
        """Test whitespace-only line returns no tokens."""
        tokens = tokenize_line("     ")
        assert len(tokens) == 0


class TestTokenizeLineKind:
    """Test that tokens are created with kind='other'."""

    def test_kind_is_other(self) -> None:
        """All tokens should initially have kind='other'."""
        tokens = tokenize_line("Gm C Hello")
        for token in tokens:
            assert token.kind == "other"


class TestTokenizeLinePunctuation:
    """Punctuation handling tests."""

    def test_punctuation_attached(self) -> None:
        """Punctuation stays attached to word."""
        tokens = tokenize_line("Hello, world!")
        assert len(tokens) == 2
        assert tokens[0].text == "Hello,"
        assert tokens[1].text == "world!"

    def test_apostrophe(self) -> None:
        """Contractions stay together."""
        tokens = tokenize_line("you're turning")
        assert len(tokens) == 2
        assert tokens[0].text == "you're"


class TestTokenizeLineSpanAccuracy:
    """Verify span calculations are accurate."""

    @pytest.mark.parametrize(
        ("line", "expected_spans"),
        [
            ("A", [(0, 1)]),
            ("AB", [(0, 2)]),
            (" A", [(1, 2)]),
            ("A B", [(0, 1), (2, 3)]),
            ("  A  B  ", [(2, 3), (5, 6)]),
            ("Gm7 C/E", [(0, 3), (4, 7)]),
        ],
    )
    def test_span_calculations(self, line: str, expected_spans: list[tuple[int, int]]) -> None:
        """Test various span calculations."""
        tokens = tokenize_line(line)
        actual_spans = [(t.start, t.end) for t in tokens]
        assert actual_spans == expected_spans
