"""Tests for tab sheet chord detection and line classification."""

import pytest

from chord_parser.tab_parser.chord_detector import (
    classify_line,
    classify_token,
    classify_tokens,
    is_chord,
    parse_chord,
)
from chord_parser.tab_parser.models import Token


class TestIsChord:
    """Test chord detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "C",
            "G",
            "Am",
            "Dm",
            "Em",
            "Gm",
            "Bb",
            "F#",
            "Gm7",
            "Cmaj7",
            "Dm7",
            "Bdim",
            "Faug",
            "Dsus4",
            "Asus2",
            "C/E",
            "G/B",
            "F#m",
            "Bbm7",
            "Abm7",
            "Am7",
            "Gm7",
        ],
    )
    def test_valid_chords(self, text: str) -> None:
        """Test that valid chords are recognized."""
        assert is_chord(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Upside",
            "down",
            "you're",
            "turning",
            "me",
            "round",
            "Hello",
            "world",
            "the",
            "and",
            "I",
            "said",
            "love",
            "boy",
        ],
    )
    def test_non_chords(self, text: str) -> None:
        """Test that non-chords are rejected."""
        assert is_chord(text) is False

    def test_empty_string(self) -> None:
        """Test that empty string is not a chord."""
        assert is_chord("") is False

    def test_long_string(self) -> None:
        """Test that very long strings are rejected quickly."""
        assert is_chord("A" * 20) is False


class TestParseChord:
    """Test chord parsing to Chord objects."""

    def test_parse_simple_chord(self) -> None:
        """Test parsing a simple chord."""
        chord = parse_chord("Gm7")
        assert chord is not None
        assert chord.root == "G"
        assert chord.quality == "min7"

    def test_parse_major_chord(self) -> None:
        """Test parsing a major chord."""
        chord = parse_chord("C")
        assert chord is not None
        assert chord.root == "C"
        assert chord.quality == "maj"

    def test_parse_slash_chord(self) -> None:
        """Test parsing a slash chord."""
        chord = parse_chord("C/E")
        assert chord is not None
        assert chord.root == "C"
        assert chord.bass == "E"

    def test_parse_invalid_returns_none(self) -> None:
        """Test that invalid chord returns None."""
        chord = parse_chord("Hello")
        assert chord is None


class TestClassifyToken:
    """Test single token classification."""

    def test_classify_chord_token(self) -> None:
        """Test classifying a chord token."""
        token = Token(text="Gm7", start=0, end=3, kind="other")
        classified = classify_token(token)
        assert classified.kind == "chord"
        assert classified.chord is not None
        assert classified.chord.root == "G"

    def test_classify_word_token(self) -> None:
        """Test classifying a word token."""
        token = Token(text="Hello", start=0, end=5, kind="other")
        classified = classify_token(token)
        assert classified.kind == "word"
        assert classified.chord is None

    def test_classify_punct_token(self) -> None:
        """Test classifying a punctuation token."""
        token = Token(text="...", start=0, end=3, kind="other")
        classified = classify_token(token)
        assert classified.kind == "punct"


class TestClassifyTokens:
    """Test batch token classification."""

    def test_classify_mixed_tokens(self) -> None:
        """Test classifying a list of mixed tokens."""
        tokens = [
            Token(text="Gm", start=0, end=2, kind="other"),
            Token(text="C", start=5, end=6, kind="other"),
            Token(text="Hello", start=0, end=5, kind="other"),
        ]
        classified = classify_tokens(tokens)
        assert classified[0].kind == "chord"
        assert classified[1].kind == "chord"
        assert classified[2].kind == "word"


class TestClassifyLine:
    """Test line classification."""

    def test_empty_line(self) -> None:
        """Test empty line classification."""
        assert classify_line("") == "empty"
        assert classify_line("   ") == "empty"

    def test_section_header(self) -> None:
        """Test section header classification."""
        assert classify_line("[Verse 1]") == "section_header"
        assert classify_line("  [Chorus]  ") == "section_header"
        assert classify_line("[Intro]") == "section_header"

    def test_comment_line(self) -> None:
        """Test comment line classification."""
        assert classify_line("(loop and fade)") == "comment"
        assert classify_line("  (repeat 2x)  ") == "comment"

    def test_chord_line(self) -> None:
        """Test chord line classification."""
        assert classify_line("Gm C F Gm") == "chord"
        assert classify_line("Am  G  F  C") == "chord"
        assert classify_line("Bb Bdim C  F Gm") == "chord"

    def test_lyric_line(self) -> None:
        """Test lyric line classification."""
        assert classify_line("Hello world") == "lyric"
        assert classify_line("Upside down you're turning me") == "lyric"
        assert classify_line("I said, hello") == "lyric"

    def test_mixed_line_is_lyric(self) -> None:
        """Test that lines with both chords and words classify as lyric."""
        # Lines like "Gm Hello C world" are classified as lyric
        # because they have word tokens
        assert classify_line("Hello Gm world") == "lyric"
