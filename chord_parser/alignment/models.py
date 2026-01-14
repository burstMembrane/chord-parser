"""Data models for chord alignment.

This module defines the core data structures for timed chord annotations
and alignment results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chord_parser.models import Chord


@dataclass(frozen=True)
class TimedChord:
    """A chord with timing information.

    Parameters
    ----------
    chord : Chord | None
        The parsed chord, or None for "no chord" (N).
    label : str
        Original chord label.
    start : float
        Start time in seconds.
    end : float | None
        End time in seconds, or None if last chord.
    """

    chord: Chord | None
    label: str
    start: float
    end: float | None


@dataclass(frozen=True)
class TabChord:
    """A chord from a tab sheet with context.

    Parameters
    ----------
    chord : Chord
        The parsed chord.
    label : str
        Original chord text from the tab.
    section : str
        Section name (e.g., "Verse 1", "Chorus").
    index : int
        Position in the flattened chord sequence.
    """

    chord: Chord
    label: str
    section: str
    index: int


@dataclass(frozen=True)
class AlignedChord:
    """Result of aligning a tab chord with a timed chord.

    Parameters
    ----------
    tab_chord : TabChord
        The chord from the tab.
    timed_chord : TimedChord
        The matched timed chord from Chordino.
    distance : float
        The alignment distance/cost.
    """

    tab_chord: TabChord
    timed_chord: TimedChord
    distance: float

    def get_start(self) -> float:
        """Get start time from the timed chord."""
        return self.timed_chord.start

    def get_end(self) -> float | None:
        """Get end time from the timed chord."""
        return self.timed_chord.end


@dataclass(frozen=True)
class AlignmentResult:
    """Result of DTW alignment between tab and timed chords.

    Parameters
    ----------
    alignments : tuple[AlignedChord, ...]
        All aligned chord pairs.
    tab_chords : tuple[TabChord, ...]
        Original tab chord sequence.
    timed_chords : tuple[TimedChord, ...]
        Original timed chord sequence.
    total_distance : float
        Total DTW alignment distance.
    normalized_distance : float
        Distance normalized by alignment path length.
    """

    alignments: tuple[AlignedChord, ...]
    tab_chords: tuple[TabChord, ...]
    timed_chords: tuple[TimedChord, ...]
    total_distance: float
    normalized_distance: float
