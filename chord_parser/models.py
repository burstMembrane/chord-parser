"""Unified chord data models for chord-parser.

This module provides a common representation for chords that can be
converted to/from both pychord and Harte notation formats.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chord:
    """Unified chord representation.

    Parameters
    ----------
    root : str
        The root note of the chord (e.g., "C", "F#", "Bb").
    quality : str
        The chord quality in Harte notation (e.g., "maj", "min7", "dim").
    bass : str | None
        The bass note if different from root (for slash chords).

    Examples
    --------
    >>> chord = Chord(root="G", quality="min7")
    >>> chord.to_harte()
    'G:min7'
    >>> chord.to_pychord()
    'Gm7'
    """

    root: str
    quality: str
    bass: str | None = None

    def to_harte(self) -> str:
        """Convert to Harte notation string.

        Returns
        -------
        str
            Chord in Harte notation (e.g., "G:min7", "C:maj/E").
        """
        result = f"{self.root}:{self.quality}"
        if self.bass:
            result = f"{result}/{self.bass}"
        return result

    def to_pychord(self) -> str:
        """Convert to pychord notation string.

        Returns
        -------
        str
            Chord in pychord notation (e.g., "Gm7", "C/E").
        """
        from chord_parser.converter import harte_quality_to_pychord

        pychord_quality = harte_quality_to_pychord(self.quality)
        result = f"{self.root}{pychord_quality}"
        if self.bass:
            result = f"{result}/{self.bass}"
        return result

    def __str__(self) -> str:
        """Return Harte notation as default string representation."""
        return self.to_harte()
