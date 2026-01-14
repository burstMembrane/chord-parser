"""Tests that verify parsing of the generated chord catalogue."""

import json
from collections import Counter
from pathlib import Path

import pytest
from pychord import Chord as PyChord

from chord_parser import from_pychord
from chord_parser.converter import PYCHORD_TO_HARTE_QUALITY

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
CATALOGUE_PATH = TESTDATA_DIR / "chord_catalogue_core.json"


@pytest.fixture
def catalogue_chords() -> list[str]:
    """Load chord symbols from the generated catalogue."""
    with open(CATALOGUE_PATH) as f:
        data = json.load(f)
    return [entry["symbol"] for entry in data["chords"]]


class TestPychordParsing:
    """Test that pychord can parse the generated chord symbols."""

    def test_catalogue_exists(self):
        """Verify the catalogue file exists."""
        assert CATALOGUE_PATH.exists(), f"Run scripts/generate_chord_catalogue.py first"

    def test_pychord_can_parse_all_chords(self, catalogue_chords):
        """All generated chords should be parseable by pychord."""
        unparseable = []
        for symbol in catalogue_chords:
            if symbol == "N":
                continue
            try:
                PyChord(symbol)
            except ValueError as e:
                unparseable.append((symbol, str(e)))

        if unparseable:
            details = "\n".join(f"  {s}: {e}" for s, e in unparseable[:20])
            pytest.fail(f"pychord could not parse {len(unparseable)} chords:\n{details}")


class TestConverterCoverage:
    """Test converter coverage against the chord catalogue."""

    def test_quality_mapping_coverage(self, catalogue_chords):
        """Report which pychord qualities are missing from our mapping."""
        missing_qualities: Counter[str] = Counter()
        parsed_count = 0
        failed_examples: dict[str, str] = {}

        for symbol in catalogue_chords:
            if symbol == "N":
                continue
            try:
                pc = PyChord(symbol)
                quality = str(pc.quality)
                if quality not in PYCHORD_TO_HARTE_QUALITY:
                    missing_qualities[quality] += 1
                    if quality not in failed_examples:
                        failed_examples[quality] = symbol
                else:
                    parsed_count += 1
            except ValueError:
                continue

        total = len(catalogue_chords) - 1  # exclude N
        coverage = parsed_count / total if total > 0 else 0

        if missing_qualities:
            missing_info = "\n".join(
                f"  '{q}': count={c}, example={failed_examples[q]}" for q, c in missing_qualities.most_common(20)
            )
            pytest.fail(
                f"Missing {len(missing_qualities)} quality mappings ({coverage:.1%} coverage):\n{missing_info}\n\n"
                f"Add these to PYCHORD_TO_HARTE_QUALITY in converter.py"
            )

    def test_full_conversion_coverage(self, catalogue_chords):
        """All parseable chords should convert to Harte notation."""
        converted = 0
        failed: list[tuple[str, str]] = []

        for symbol in catalogue_chords:
            if symbol == "N":
                continue
            try:
                chord = from_pychord(symbol)
                harte = chord.to_harte()
                assert ":" in harte
                converted += 1
            except ValueError as e:
                failed.append((symbol, str(e)))

        total = len(catalogue_chords) - 1
        coverage = converted / total if total > 0 else 0

        # We want at least 80% coverage
        assert coverage >= 0.8, f"Only {coverage:.1%} of chords converted successfully. Failed examples: {failed[:10]}"


class TestRoundTripConversion:
    """Test round-trip conversion for supported chords."""

    def test_roundtrip_preserves_root_and_quality(self, catalogue_chords):
        """Round-trip conversion should preserve root and quality."""
        mismatches = []

        for symbol in catalogue_chords:
            if symbol == "N":
                continue
            try:
                chord = from_pychord(symbol)
                pychord_out = chord.to_pychord()
                chord2 = from_pychord(pychord_out)

                if chord.root != chord2.root or chord.quality != chord2.quality:
                    mismatches.append((symbol, chord, chord2))
            except ValueError:
                continue

        if mismatches:
            examples = "\n".join(f"  {s}: {c1} -> {c2}" for s, c1, c2 in mismatches[:10])
            pytest.fail(f"{len(mismatches)} round-trip mismatches:\n{examples}")


class TestSpecificChordTypes:
    """Test specific chord type categories."""

    @pytest.mark.parametrize("root", ["C", "G", "F", "Bb", "F#"])
    def test_major_triads(self, root):
        """Major triads should convert correctly."""
        chord = from_pychord(root)
        assert chord.root == root
        assert chord.quality == "maj"
        assert chord.to_harte() == f"{root}:maj"

    @pytest.mark.parametrize("root", ["C", "G", "F", "Bb", "F#"])
    def test_minor_triads(self, root):
        """Minor triads should convert correctly."""
        chord = from_pychord(f"{root}m")
        assert chord.root == root
        assert chord.quality == "min"
        assert chord.to_harte() == f"{root}:min"

    @pytest.mark.parametrize("root", ["C", "G", "F", "Bb", "F#"])
    def test_dominant_seventh(self, root):
        """Dominant 7th chords should convert correctly."""
        chord = from_pychord(f"{root}7")
        assert chord.root == root
        assert chord.quality == "7"
        assert chord.to_harte() == f"{root}:7"

    @pytest.mark.parametrize("root", ["C", "G", "F", "Bb", "F#"])
    def test_minor_seventh(self, root):
        """Minor 7th chords should convert correctly."""
        chord = from_pychord(f"{root}m7")
        assert chord.root == root
        assert chord.quality == "min7"
        assert chord.to_harte() == f"{root}:min7"

    @pytest.mark.parametrize("root", ["C", "G", "F", "Bb", "F#"])
    def test_major_seventh(self, root):
        """Major 7th chords should convert correctly."""
        chord = from_pychord(f"{root}maj7")
        assert chord.root == root
        assert chord.quality == "maj7"
        assert chord.to_harte() == f"{root}:maj7"

    @pytest.mark.parametrize("root", ["C", "G", "F"])
    def test_diminished(self, root):
        """Diminished chords should convert correctly."""
        chord = from_pychord(f"{root}dim")
        assert chord.root == root
        assert chord.quality == "dim"

    @pytest.mark.parametrize("root", ["C", "G", "F"])
    def test_augmented(self, root):
        """Augmented chords should convert correctly."""
        chord = from_pychord(f"{root}aug")
        assert chord.root == root
        assert chord.quality == "aug"
