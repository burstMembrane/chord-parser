"""Tests for chord alignment module."""

from pathlib import Path

from chord_parser import tab_parser
from chord_parser.alignment import (
    align_chords,
    chord_distance_exact,
    chord_distance_flexible,
    chord_similarity,
    extract_tab_chords,
    load_chordino_json,
)

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
UPSIDEDOWN_DIR = Path(__file__).parent.parent / "upsidedown"


def test_load_chordino_json():
    """Test loading Chordino annotations from JSON."""
    chordino_path = UPSIDEDOWN_DIR / "1285_chordino.json"
    if not chordino_path.exists():
        return  # Skip if test data not available

    timed_chords = load_chordino_json(chordino_path)

    assert len(timed_chords) > 0
    # First chord should be "N" (no chord)
    assert timed_chords[0].label == "N"
    assert timed_chords[0].chord is None

    # Second chord is Adim7/G
    assert timed_chords[1].label == "Adim7/G"
    assert timed_chords[1].chord is not None
    assert timed_chords[1].chord.root == "A"

    # Check timing
    assert timed_chords[0].start == 0.0


def test_extract_tab_chords():
    """Test extracting chords from a tab sheet."""
    tab_path = TESTDATA_DIR / "diana-ross-upside-down.md"
    if not tab_path.exists():
        return

    content = tab_path.read_text()
    # Extract just the tab content (between ``` markers)
    lines = content.split("\n")
    in_code_block = False
    tab_lines = []
    for line in lines:
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if in_code_block:
            tab_lines.append(line)

    tab_content = "\n".join(tab_lines)
    sheet = tab_parser.parse(tab_content)
    tab_chords = extract_tab_chords(sheet)

    assert len(tab_chords) > 0
    # First chord should be Gm from intro
    assert tab_chords[0].label == "Gm"
    assert tab_chords[0].chord.root == "G"
    assert tab_chords[0].chord.quality == "min"
    assert tab_chords[0].section == "Intro"


def test_align_simple():
    """Test basic alignment with simple sequences."""
    from chord_parser.alignment.models import TabChord, TimedChord
    from chord_parser.models import Chord

    # Create simple test data
    tab_chords = [
        TabChord(Chord("G", "min"), "Gm", "Test", 0),
        TabChord(Chord("C", "maj"), "C", "Test", 1),
        TabChord(Chord("F", "maj"), "F", "Test", 2),
    ]

    timed_chords = [
        TimedChord(Chord("G", "min"), "Gm", 0.0, 1.0),
        TimedChord(Chord("C", "maj"), "C", 1.0, 2.0),
        TimedChord(Chord("F", "maj"), "F", 2.0, 3.0),
    ]

    result = align_chords(tab_chords, timed_chords)

    # Perfect alignment should have 0 distance
    assert result.total_distance == 0.0
    assert len(result.alignments) == 3

    # Check alignment order
    assert result.alignments[0].tab_chord.label == "Gm"
    assert result.alignments[0].timed_chord.start == 0.0
    assert result.alignments[1].tab_chord.label == "C"
    assert result.alignments[2].tab_chord.label == "F"


def test_align_with_insertions():
    """Test alignment when Chordino has extra chords."""
    from chord_parser.alignment.models import TabChord, TimedChord
    from chord_parser.models import Chord

    # Tab has fewer chords
    tab_chords = [
        TabChord(Chord("G", "min"), "Gm", "Test", 0),
        TabChord(Chord("F", "maj"), "F", "Test", 1),
    ]

    # Chordino detected an extra C in between
    timed_chords = [
        TimedChord(Chord("G", "min"), "Gm", 0.0, 1.0),
        TimedChord(Chord("C", "maj"), "C", 1.0, 2.0),  # Extra
        TimedChord(Chord("F", "maj"), "F", 2.0, 3.0),
    ]

    result = align_chords(tab_chords, timed_chords)

    # Should still align Gm->Gm and F->F
    assert len(result.alignments) == 2
    assert result.alignments[0].tab_chord.label == "Gm"
    assert result.alignments[1].tab_chord.label == "F"


def test_align_upside_down():
    """Test alignment with actual Upside Down data."""
    chordino_path = UPSIDEDOWN_DIR / "1285_chordino.json"
    tab_path = TESTDATA_DIR / "diana-ross-upside-down.md"

    if not chordino_path.exists() or not tab_path.exists():
        return

    # Load Chordino annotations
    timed_chords = load_chordino_json(chordino_path)

    # Parse tab
    content = tab_path.read_text()
    lines = content.split("\n")
    in_code_block = False
    tab_lines = []
    for line in lines:
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if in_code_block:
            tab_lines.append(line)

    tab_content = "\n".join(tab_lines)
    sheet = tab_parser.parse(tab_content)
    tab_chords = extract_tab_chords(sheet)

    # Run alignment with larger lookahead for longer song
    result = align_chords(tab_chords, timed_chords, lookahead=10)

    # Tab has repeating sections (verses, choruses) but audio only has
    # one temporal stream. The greedy aligner will exhaust timed chords
    # before matching all tab chords. This is expected behavior.
    # What matters is that the matches we do get are good quality.
    assert len(result.alignments) > 30  # Should get many matches
    assert result.normalized_distance < 0.3  # Quality should be high

    # Verify most matches are exact or close matches
    exact_matches = sum(1 for a in result.alignments if a.distance == 0.0)
    close_matches = sum(1 for a in result.alignments if a.distance <= 0.5)
    assert exact_matches > 20  # Many exact matches
    assert close_matches == len(result.alignments)  # All are close


def test_chord_distance_flexible():
    """Test flexible distance function with various chord pairs."""
    import pytest

    from chord_parser.models import Chord

    # Exact match
    assert chord_distance_flexible(Chord("G", "min"), Chord("G", "min")) == 0.0
    assert chord_distance_flexible(Chord("C", "maj"), Chord("C", "maj")) == 0.0

    # Same root, different quality -> 0.2
    assert chord_distance_flexible(Chord("B", "dim"), Chord("B", "maj")) == pytest.approx(0.2)
    assert chord_distance_flexible(Chord("G", "min"), Chord("G", "min7")) == pytest.approx(0.2)
    assert chord_distance_flexible(Chord("Ab", "min7"), Chord("Ab", "min")) == pytest.approx(0.2)

    # Bass note match -> 0.5
    # Adim7/G should match Gm (G in bass matches G root)
    assert chord_distance_flexible(Chord("A", "dim7", "G"), Chord("G", "min")) == pytest.approx(0.5)
    assert chord_distance_flexible(Chord("G", "min"), Chord("A", "dim7", "G")) == pytest.approx(0.5)

    # No match -> 1.0
    assert chord_distance_flexible(Chord("G", "min"), Chord("F", "maj")) == 1.0
    assert chord_distance_flexible(Chord("Bb", "maj"), Chord("G", "min")) == 1.0

    # None handling
    assert chord_distance_flexible(None, None) == 0.0
    assert chord_distance_flexible(Chord("G", "min"), None) == 1.0
    assert chord_distance_flexible(None, Chord("G", "min")) == 1.0


def test_chord_distance_exact():
    """Test exact distance function."""
    from chord_parser.models import Chord

    # Exact match
    assert chord_distance_exact(Chord("G", "min"), Chord("G", "min")) == 0.0

    # Same root, different quality -> 1.0 (exact mode)
    assert chord_distance_exact(Chord("G", "min"), Chord("G", "min7")) == 1.0

    # Different root -> 1.0
    assert chord_distance_exact(Chord("G", "min"), Chord("F", "maj")) == 1.0


def test_chord_similarity():
    """Test chord similarity function."""
    from chord_parser.models import Chord

    # Exact match -> 1.0
    assert chord_similarity(Chord("G", "min"), Chord("G", "min")) == 1.0
    assert chord_similarity(Chord("C", "maj"), Chord("C", "maj")) == 1.0

    # Same root, different quality -> 0.8
    assert chord_similarity(Chord("B", "dim"), Chord("B", "maj")) == 0.8
    assert chord_similarity(Chord("G", "min"), Chord("G", "min7")) == 0.8
    assert chord_similarity(Chord("Ab", "min7"), Chord("Ab", "min")) == 0.8

    # Bass note matches root -> 0.5
    assert chord_similarity(Chord("A", "dim7", "G"), Chord("G", "min")) == 0.5
    assert chord_similarity(Chord("G", "min"), Chord("A", "dim7", "G")) == 0.5

    # No match -> 0.0
    assert chord_similarity(Chord("G", "min"), Chord("F", "maj")) == 0.0
    assert chord_similarity(Chord("Bb", "maj"), Chord("G", "min")) == 0.0

    # None handling
    assert chord_similarity(Chord("G", "min"), None) == 0.0
    assert chord_similarity(None, Chord("G", "min")) == 0.0


def test_align_with_quality_variations():
    """Test alignment with chord quality variations."""
    import pytest

    from chord_parser.alignment.models import TabChord, TimedChord
    from chord_parser.models import Chord

    # Tab has Bdim, Chordino detected B (same root, different quality)
    tab_chords = [
        TabChord(Chord("G", "min"), "Gm", "Test", 0),
        TabChord(Chord("B", "dim"), "Bdim", "Test", 1),
        TabChord(Chord("C", "maj"), "C", "Test", 2),
    ]

    timed_chords = [
        TimedChord(Chord("G", "min"), "Gm", 0.0, 1.0),
        TimedChord(Chord("B", "maj"), "B", 1.0, 2.0),  # Different quality
        TimedChord(Chord("C", "maj"), "C", 2.0, 3.0),
    ]

    # Default alignment uses chord_similarity which allows quality variations
    result = align_chords(tab_chords, timed_chords)

    # All 3 should match (Gm-Gm exact, Bdim-B same root, C-C exact)
    assert len(result.alignments) == 3

    # Bdim vs B should have distance 0.2 (1.0 - 0.8 similarity)
    assert result.alignments[1].distance == pytest.approx(0.2)

    # Count matches with distance <= 0.5 (similarity >= 0.5)
    close_matches = sum(1 for a in result.alignments if a.distance <= 0.5)
    assert close_matches == 3
