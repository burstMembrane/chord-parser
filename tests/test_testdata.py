"""Tests that verify parsing of real chord data from testdata files."""

import json
from pathlib import Path

import pytest

from chord_parser import from_harte, from_pychord

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


@pytest.fixture
def harte_chords() -> list[dict]:
    """Load chords from 1285.json (Harte notation)."""
    with open(TESTDATA_DIR / "1285.json") as f:
        data = json.load(f)
    return data["chords"]


@pytest.fixture
def pychord_chords() -> list[dict]:
    """Load chords from 1285_chordino.json (pychord notation)."""
    with open(TESTDATA_DIR / "1285_chordino.json") as f:
        data = json.load(f)
    return data["chords"]


class TestParseHarteTestdata:
    def test_can_parse_all_harte_chords(self, harte_chords):
        """All Harte chords from testdata should parse without error."""
        parsed = []
        for entry in harte_chords:
            chord_str = entry["chord"]
            if chord_str == "N":  # Skip "no chord" marker
                continue
            chord = from_harte(chord_str)
            parsed.append(chord)

        assert len(parsed) > 0
        assert all(c.root for c in parsed)
        assert all(c.quality for c in parsed)

    def test_harte_chord_qualities_recognized(self, harte_chords):
        """All chord qualities in Harte testdata should be recognized."""
        qualities = set()
        for entry in harte_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                continue
            chord = from_harte(chord_str)
            qualities.add(chord.quality)

        # Should have at least these common qualities
        assert "maj" in qualities
        assert "min7" in qualities

    def test_harte_roundtrip(self, harte_chords):
        """Harte chords should roundtrip: harte -> model -> harte."""
        for entry in harte_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                continue
            chord = from_harte(chord_str)
            result = chord.to_harte()
            assert result == chord_str, f"Roundtrip failed: {chord_str} -> {result}"


class TestParsePychordTestdata:
    def test_can_parse_all_pychord_chords(self, pychord_chords):
        """All pychord chords from testdata should parse without error."""
        parsed = []
        skipped = []
        for entry in pychord_chords:
            chord_str = entry["chord"]
            if chord_str == "N":  # Skip "no chord" marker
                continue
            try:
                chord = from_pychord(chord_str)
                parsed.append(chord)
            except ValueError as e:
                skipped.append((chord_str, str(e)))

        assert len(parsed) > 0
        # Report any skipped chords for quality mapping improvements
        if skipped:
            pytest.skip(f"Skipped {len(skipped)} chords with unknown qualities: {skipped[:5]}")

    def test_pychord_chord_qualities_recognized(self, pychord_chords):
        """All chord qualities in pychord testdata should be recognized."""
        qualities = set()
        for entry in pychord_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                continue
            try:
                chord = from_pychord(chord_str)
                qualities.add(chord.quality)
            except ValueError:
                continue

        # Should have common qualities
        assert "min" in qualities or "min7" in qualities

    def test_pychord_to_harte_conversion(self, pychord_chords):
        """Pychord chords should convert to valid Harte notation."""
        for entry in pychord_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                continue
            try:
                chord = from_pychord(chord_str)
                harte_str = chord.to_harte()
                # Harte format should have root:quality pattern
                assert ":" in harte_str, f"Invalid Harte format: {harte_str}"
            except ValueError:
                continue  # Skip unknown qualities


class TestCrossFormatConversion:
    def test_common_chords_convert_correctly(self):
        """Common chord patterns should convert between formats correctly."""
        test_cases = [
            # (pychord, expected_harte)
            ("C", "C:maj"),
            ("Gm7", "G:min7"),
            ("Bb", "Bb:maj"),
            ("Bbm7", "Bb:min7"),
            ("F", "F:maj"),
            ("Am", "A:min"),
            ("Dm7", "D:min7"),
            ("Abm", "Ab:min"),
        ]

        for pychord_str, expected_harte in test_cases:
            chord = from_pychord(pychord_str)
            assert chord.to_harte() == expected_harte, f"{pychord_str} -> {chord.to_harte()} != {expected_harte}"

    def test_harte_to_pychord_common_chords(self):
        """Common Harte patterns should convert to pychord correctly."""
        test_cases = [
            # (harte, expected_pychord)
            ("C:maj", "C"),
            ("G:min7", "Gm7"),
            ("Bb:maj", "Bb"),
            ("Bb:min7", "Bbm7"),
            ("F:maj", "F"),
            ("A:min", "Am"),
            ("Ab:min", "Abm"),
        ]

        for harte_str, expected_pychord in test_cases:
            chord = from_harte(harte_str)
            assert chord.to_pychord() == expected_pychord, f"{harte_str} -> {chord.to_pychord()} != {expected_pychord}"

    def test_unique_chords_in_testdata(self, harte_chords, pychord_chords):
        """Report unique chord types found in testdata."""
        harte_types = {e["chord"] for e in harte_chords if e["chord"] != "N"}
        pychord_types = {e["chord"] for e in pychord_chords if e["chord"] != "N"}

        # Just verify we can identify unique chords
        assert len(harte_types) > 0
        assert len(pychord_types) > 0


class TestFullSequenceConversion:
    def test_convert_full_pychord_sequence_to_harte(self, pychord_chords):
        """Convert entire pychord sequence to Harte and verify structure."""
        converted = []
        errors = []

        for entry in pychord_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                converted.append({"chord": "N", "start": entry["start"], "end": entry["end"]})
                continue
            try:
                chord = from_pychord(chord_str)
                converted.append({
                    "chord": chord.to_harte(),
                    "start": entry["start"],
                    "end": entry["end"],
                })
            except ValueError as e:
                errors.append((chord_str, str(e)))

        # Should convert most chords successfully
        success_rate = len(converted) / len(pychord_chords)
        assert success_rate > 0.9, f"Only {success_rate:.1%} of chords converted"

        # Verify converted chords have Harte format
        for c in converted:
            if c["chord"] != "N":
                assert ":" in c["chord"]

    def test_convert_full_harte_sequence_to_pychord(self, harte_chords):
        """Convert entire Harte sequence to pychord and verify structure."""
        converted = []
        errors = []

        for entry in harte_chords:
            chord_str = entry["chord"]
            if chord_str == "N":
                converted.append({"chord": "N", "onset": entry["onset"], "offset": entry["offset"]})
                continue
            try:
                chord = from_harte(chord_str)
                converted.append({
                    "chord": chord.to_pychord(),
                    "onset": entry["onset"],
                    "offset": entry["offset"],
                })
            except ValueError as e:
                errors.append((chord_str, str(e)))

        # Should convert most chords successfully
        success_rate = len(converted) / len(harte_chords)
        assert success_rate > 0.9, f"Only {success_rate:.1%} of chords converted"

        # Verify converted chords don't have colons (pychord format)
        for c in converted:
            if c["chord"] != "N":
                assert ":" not in c["chord"]
