"""Tests for tab sheet parser."""

from pathlib import Path

import pytest

from chord_parser.tab_parser import (
    Block,
    ChordLine,
    CommentLine,
    EmptyLine,
    LyricLine,
    parse,
)
from chord_parser.tab_parser.attachment import attach_chords_to_words, compute_overlap
from chord_parser.tab_parser.models import Token

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


class TestComputeOverlap:
    """Test overlap computation."""

    def test_full_overlap(self) -> None:
        """Test when chord is fully within word span."""
        chord = Token(text="G", start=0, end=1, kind="chord")
        word = Token(text="Hello", start=0, end=5, kind="word")
        assert compute_overlap(chord, word) == 1

    def test_partial_overlap(self) -> None:
        """Test partial overlap."""
        chord = Token(text="Gm", start=4, end=6, kind="chord")
        word = Token(text="Hello", start=0, end=5, kind="word")
        assert compute_overlap(chord, word) == 1

    def test_no_overlap(self) -> None:
        """Test no overlap."""
        chord = Token(text="Gm", start=10, end=12, kind="chord")
        word = Token(text="Hello", start=0, end=5, kind="word")
        assert compute_overlap(chord, word) == 0


class TestAttachChordsToWords:
    """Test chord-to-word attachment."""

    def test_simple_attachment(self) -> None:
        """Test simple chord-word attachment."""
        chords = [
            Token(text="Gm", start=0, end=2, kind="chord"),
            Token(text="C", start=7, end=8, kind="chord"),
        ]
        lyrics = [
            Token(text="Hello", start=0, end=5, kind="word"),
            Token(text="world", start=7, end=12, kind="word"),
        ]
        result = attach_chords_to_words(chords, lyrics)

        assert len(result.chord_to_word) == 2
        assert result.chord_to_word[0].word is not None
        assert result.chord_to_word[0].word.text == "Hello"
        assert result.chord_to_word[1].word is not None
        assert result.chord_to_word[1].word.text == "world"

    def test_word_to_chords_mapping(self) -> None:
        """Test reverse mapping."""
        chords = [Token(text="Gm", start=0, end=2, kind="chord")]
        lyrics = [Token(text="Hello", start=0, end=5, kind="word")]
        result = attach_chords_to_words(chords, lyrics)

        assert len(result.word_to_chords) == 1
        assert result.word_to_chords[0].word.text == "Hello"
        assert len(result.word_to_chords[0].chords) == 1
        assert result.word_to_chords[0].chords[0].text == "Gm"


class TestParseBasic:
    """Basic parser tests."""

    def test_empty_input(self) -> None:
        """Test parsing empty input."""
        sheet = parse("")
        assert len(sheet.sections) == 1
        assert sheet.sections[0].name == "Song"

    def test_single_section_header(self) -> None:
        """Test parsing with section header."""
        sheet = parse("[Verse 1]\nHello world")
        assert len(sheet.sections) == 1
        assert sheet.sections[0].name == "Verse 1"


class TestParseBlocks:
    """Test block (chord+lyric pair) parsing."""

    def test_simple_block(self) -> None:
        """Test parsing a simple chord+lyric block."""
        text = "Gm     C\nHello  world"
        sheet = parse(text)

        assert len(sheet.sections) == 1
        items = sheet.sections[0].items
        assert len(items) == 1
        assert isinstance(items[0], Block)

        block = items[0]
        assert len(block.chord_tokens) == 2
        assert len(block.attachments.chord_to_word) == 2

    def test_block_attachments(self) -> None:
        """Test that block has correct attachments."""
        text = "Gm     C\nHello  world"
        sheet = parse(text)

        block = sheet.sections[0].items[0]
        assert isinstance(block, Block)

        attachments = block.attachments
        assert attachments.chord_to_word[0].word is not None
        assert attachments.chord_to_word[0].word.text == "Hello"
        assert attachments.chord_to_word[1].word is not None
        assert attachments.chord_to_word[1].word.text == "world"


class TestParseLineTypes:
    """Test parsing different line types."""

    def test_chord_only_line(self) -> None:
        """Test standalone chord line."""
        text = "[Intro]\nAm G F C"
        sheet = parse(text)

        items = sheet.sections[0].items
        # First item should be the chord line (no lyrics follow)
        assert isinstance(items[0], ChordLine)
        assert len(items[0].tokens) == 4

    def test_lyric_only_line(self) -> None:
        """Test standalone lyric line."""
        text = "Hello world"
        sheet = parse(text)

        items = sheet.sections[0].items
        assert isinstance(items[0], LyricLine)

    def test_empty_lines(self) -> None:
        """Test empty lines are preserved."""
        text = "Hello\n\nWorld"
        sheet = parse(text)

        items = sheet.sections[0].items
        assert len(items) == 3
        assert isinstance(items[1], EmptyLine)

    def test_comment_line(self) -> None:
        """Test comment lines."""
        text = "(loop and fade)"
        sheet = parse(text)

        items = sheet.sections[0].items
        assert isinstance(items[0], CommentLine)
        assert items[0].raw == "(loop and fade)"


class TestParseMultipleSections:
    """Test parsing multiple sections."""

    def test_multiple_sections(self) -> None:
        """Test parsing multiple section headers."""
        text = "[Verse 1]\nLine 1\n\n[Chorus]\nLine 2"
        sheet = parse(text)

        assert len(sheet.sections) == 2
        assert sheet.sections[0].name == "Verse 1"
        assert sheet.sections[1].name == "Chorus"


class TestParseFixtures:
    """Test parsing fixture files."""

    @pytest.fixture
    def simple_pair(self) -> str:
        """Load simple_pair.txt fixture."""
        path = TESTDATA_DIR / "simple_pair.txt"
        return path.read_text()

    @pytest.fixture
    def upside_down(self) -> str:
        """Load upside_down.txt fixture."""
        path = TESTDATA_DIR / "upside_down.txt"
        return path.read_text()

    @pytest.fixture
    def multiple_sections(self) -> str:
        """Load multiple_sections.txt fixture."""
        path = TESTDATA_DIR / "multiple_sections.txt"
        return path.read_text()

    @pytest.fixture
    def intro_chords_only(self) -> str:
        """Load intro_chords_only.txt fixture."""
        path = TESTDATA_DIR / "intro_chords_only.txt"
        return path.read_text()

    def test_simple_pair_fixture(self, simple_pair: str) -> None:
        """Test parsing simple_pair.txt."""
        sheet = parse(simple_pair)

        assert len(sheet.sections) == 1
        items = sheet.sections[0].items
        # Should have one block
        blocks = [i for i in items if isinstance(i, Block)]
        assert len(blocks) == 1

        block = blocks[0]
        assert block.attachments.chord_to_word[0].word is not None
        assert block.attachments.chord_to_word[0].word.text == "Hello"

    def test_upside_down_fixture(self, upside_down: str) -> None:
        """Test parsing upside_down.txt."""
        sheet = parse(upside_down)

        # Should have multiple sections
        assert len(sheet.sections) >= 3
        section_names = [s.name for s in sheet.sections]
        assert "Intro" in section_names
        assert "Verse 1" in section_names
        assert "Chorus" in section_names

    def test_upside_down_has_blocks(self, upside_down: str) -> None:
        """Test that upside_down has chord+lyric blocks."""
        sheet = parse(upside_down)

        all_items = []
        for section in sheet.sections:
            all_items.extend(section.items)

        blocks = [i for i in all_items if isinstance(i, Block)]
        assert len(blocks) > 0

    def test_multiple_sections_fixture(self, multiple_sections: str) -> None:
        """Test parsing multiple_sections.txt."""
        sheet = parse(multiple_sections)

        section_names = [s.name for s in sheet.sections]
        assert "Verse 1" in section_names
        assert "Chorus" in section_names
        assert "Verse 2" in section_names
        assert "Bridge" in section_names

    def test_intro_chords_only_fixture(self, intro_chords_only: str) -> None:
        """Test parsing intro_chords_only.txt."""
        sheet = parse(intro_chords_only)

        # Should have Intro and Outro sections
        section_names = [s.name for s in sheet.sections]
        assert "Intro" in section_names
        assert "Outro" in section_names

        # Intro section should have chord-only lines
        intro = next(s for s in sheet.sections if s.name == "Intro")
        chord_lines = [i for i in intro.items if isinstance(i, ChordLine)]
        assert len(chord_lines) >= 1


class TestParseChordIntegration:
    """Test that chords are properly integrated with Chord model."""

    def test_chord_tokens_have_chord_objects(self) -> None:
        """Test that chord tokens have parsed Chord objects."""
        text = "Gm7    C\nHello  world"
        sheet = parse(text)

        block = sheet.sections[0].items[0]
        assert isinstance(block, Block)

        # Find chord tokens
        chord_tokens = [t for t in block.chord_tokens if t.kind == "chord"]
        assert len(chord_tokens) == 2

        # Check first chord
        gm7 = chord_tokens[0]
        assert gm7.chord is not None
        assert gm7.chord.root == "G"
        assert gm7.chord.quality == "min7"

        # Check second chord
        c = chord_tokens[1]
        assert c.chord is not None
        assert c.chord.root == "C"
        assert c.chord.quality == "maj"
