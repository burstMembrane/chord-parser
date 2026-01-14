"""Chord alignment module for matching tab chords to timed audio annotations.

This module provides functionality to align chords parsed from tab sheets
with timed chord annotations from audio analysis tools like Chordino,
Essentia, and madmom.
"""

from chord_parser.alignment.aligner import (
    align_chords,
    chord_distance_exact,
    chord_distance_flexible,
    chord_distance_pitchclass,
    chord_to_pitch_class_set,
)
from chord_parser.alignment.chordino import load_chordino_json, parse_chordino_data
from chord_parser.alignment.extractor import extract_tab_chords
from chord_parser.alignment.models import (
    AlignedChord,
    AlignmentResult,
    TabChord,
    TimedChord,
)

__all__ = [
    "AlignedChord",
    "AlignmentResult",
    "TabChord",
    "TimedChord",
    "align_chords",
    "chord_distance_exact",
    "chord_distance_flexible",
    "chord_distance_pitchclass",
    "chord_to_pitch_class_set",
    "extract_chords_essentia",
    "extract_chords_madmom",
    "extract_tab_chords",
    "load_chordino_json",
    "parse_chordino_data",
]


def __getattr__(name: str):
    """Lazy import for optional audio extractors.

    These require heavy dependencies (essentia, madmom) that are optional.
    """
    if name == "extract_chords_essentia":
        from chord_parser.alignment.essentia_extractor import extract_chords_essentia

        return extract_chords_essentia
    if name == "extract_chords_madmom":
        from chord_parser.alignment.madmom_extractor import extract_chords_madmom

        return extract_chords_madmom
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
