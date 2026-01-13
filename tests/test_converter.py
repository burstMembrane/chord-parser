import pytest

from chord_parser import (
    Chord,
    from_harte,
    from_pychord,
    harte_quality_to_pychord,
    harte_to_pychord,
    pychord_quality_to_harte,
    pychord_to_harte,
)


class TestQualityMapping:
    def test_pychord_major_to_harte(self):
        assert pychord_quality_to_harte("") == "maj"

    def test_pychord_minor_to_harte(self):
        assert pychord_quality_to_harte("m") == "min"

    def test_pychord_minor7_to_harte(self):
        assert pychord_quality_to_harte("m7") == "min7"

    def test_pychord_maj7_to_harte(self):
        assert pychord_quality_to_harte("maj7") == "maj7"

    def test_pychord_dom7_to_harte(self):
        assert pychord_quality_to_harte("7") == "7"

    def test_pychord_dim_to_harte(self):
        assert pychord_quality_to_harte("dim") == "dim"

    def test_pychord_dim7_to_harte(self):
        assert pychord_quality_to_harte("dim7") == "dim7"

    def test_pychord_hdim_to_harte(self):
        assert pychord_quality_to_harte("m7-5") == "hdim7"

    def test_harte_maj_to_pychord(self):
        assert harte_quality_to_pychord("maj") == ""

    def test_harte_min_to_pychord(self):
        assert harte_quality_to_pychord("min") == "m"

    def test_harte_min7_to_pychord(self):
        assert harte_quality_to_pychord("min7") == "m7"

    def test_unknown_pychord_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown pychord quality"):
            pychord_quality_to_harte("unknown_quality")

    def test_unknown_harte_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown Harte quality"):
            harte_quality_to_pychord("unknown_quality")


class TestFromPychord:
    def test_simple_major(self):
        chord = from_pychord("C")
        assert chord.root == "C"
        assert chord.quality == "maj"
        assert chord.bass is None

    def test_minor_seventh(self):
        chord = from_pychord("Gm7")
        assert chord.root == "G"
        assert chord.quality == "min7"
        assert chord.bass is None

    def test_flat_root(self):
        chord = from_pychord("Bbm7")
        assert chord.root == "Bb"
        assert chord.quality == "min7"

    def test_sharp_root(self):
        chord = from_pychord("F#dim7")
        assert chord.root == "F#"
        assert chord.quality == "dim7"

    def test_slash_chord(self):
        chord = from_pychord("C/E")
        assert chord.root == "C"
        assert chord.quality == "maj"
        assert chord.bass == "E"


class TestFromHarte:
    def test_simple_major(self):
        chord = from_harte("C:maj")
        assert chord.root == "C"
        assert chord.quality == "maj"
        assert chord.bass is None

    def test_minor_seventh(self):
        chord = from_harte("G:min7")
        assert chord.root == "G"
        assert chord.quality == "min7"
        assert chord.bass is None

    def test_flat_root(self):
        chord = from_harte("Bb:min7")
        assert chord.root == "Bb"
        assert chord.quality == "min7"


class TestChordModel:
    def test_to_harte(self):
        chord = Chord(root="G", quality="min7")
        assert chord.to_harte() == "G:min7"

    def test_to_harte_with_bass(self):
        chord = Chord(root="C", quality="maj", bass="E")
        assert chord.to_harte() == "C:maj/E"

    def test_to_pychord(self):
        chord = Chord(root="G", quality="min7")
        assert chord.to_pychord() == "Gm7"

    def test_to_pychord_major(self):
        chord = Chord(root="C", quality="maj")
        assert chord.to_pychord() == "C"

    def test_to_pychord_with_bass(self):
        chord = Chord(root="C", quality="maj", bass="E")
        assert chord.to_pychord() == "C/E"

    def test_str_returns_harte(self):
        chord = Chord(root="G", quality="min7")
        assert str(chord) == "G:min7"

    def test_chord_is_immutable(self):
        chord = Chord(root="C", quality="maj")
        with pytest.raises(AttributeError):
            chord.root = "D"  # type: ignore[misc]


class TestConversionFunctions:
    def test_pychord_to_harte_major(self):
        assert pychord_to_harte("C") == "C:maj"

    def test_pychord_to_harte_minor7(self):
        assert pychord_to_harte("Gm7") == "G:min7"

    def test_pychord_to_harte_flat(self):
        assert pychord_to_harte("Bbm") == "Bb:min"

    def test_harte_to_pychord_major(self):
        assert harte_to_pychord("C:maj") == "C"

    def test_harte_to_pychord_minor7(self):
        assert harte_to_pychord("G:min7") == "Gm7"

    def test_harte_to_pychord_flat(self):
        assert harte_to_pychord("Bb:min") == "Bbm"


class TestRoundTrip:
    @pytest.mark.parametrize(
        "pychord_str",
        ["C", "Gm7", "Bbm", "F#dim7", "Am", "Dmaj7", "E7"],
    )
    def test_pychord_roundtrip(self, pychord_str):
        chord = from_pychord(pychord_str)
        result = chord.to_pychord()
        # Parse again to compare
        chord2 = from_pychord(result)
        assert chord.root == chord2.root
        assert chord.quality == chord2.quality

    @pytest.mark.parametrize(
        "harte_str",
        ["C:maj", "G:min7", "Bb:min", "F#:dim7", "A:min", "D:maj7", "E:7"],
    )
    def test_harte_roundtrip(self, harte_str):
        chord = from_harte(harte_str)
        result = chord.to_harte()
        assert result == harte_str
