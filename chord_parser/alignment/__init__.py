"""Chord alignment module for matching tab chords to timed audio annotations.

This module provides functionality to align chords parsed from tab sheets
with timed chord annotations from audio analysis tools like Chordino.
"""

from chord_parser.alignment.aligner import (
    DistanceFn,
    align_chords,
    align_with_transposition,
    chord_distance_exact,
    chord_distance_flexible,
    chord_similarity,
    chord_similarity_weighted,
    compute_alignment_metrics,
)
from chord_parser.alignment.api import align_tab_text_with_smams, align_tab_with_smams
from chord_parser.alignment.chordino import load_chordino_json, parse_chordino_data
from chord_parser.alignment.evaluation import (
    EvaluationResult,
    compare_chord_sequences,
    evaluate_against_ground_truth,
)
from chord_parser.alignment.extractor import extract_tab_chords
from chord_parser.alignment.models import (
    AlignedChord,
    AlignmentResult,
    TabChord,
    TimedChord,
)
from chord_parser.alignment.smams import (
    parse_chordino_from_smams,
    parse_chordino_from_smams_dict,
)

__all__ = [
    "AlignedChord",
    "AlignmentResult",
    "DistanceFn",
    "EvaluationResult",
    "TabChord",
    "TimedChord",
    "align_chords",
    "align_tab_text_with_smams",
    "align_tab_with_smams",
    "align_with_transposition",
    "chord_distance_exact",
    "chord_distance_flexible",
    "chord_similarity",
    "chord_similarity_weighted",
    "compare_chord_sequences",
    "compute_alignment_metrics",
    "evaluate_against_ground_truth",
    "extract_tab_chords",
    "load_chordino_json",
    "parse_chordino_data",
    "parse_chordino_from_smams",
    "parse_chordino_from_smams_dict",
]
