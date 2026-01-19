"""Type adapters between chord_parser and DECIBEL types.

This module provides conversion functions to bridge chord_parser's data models
with DECIBEL's internal representations for HMM-based alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from decibel.music_objects.chord import Chord as DECIBELChord
from decibel.music_objects.chord_alphabet import ChordAlphabet

if TYPE_CHECKING:
    from chord_parser.alignment.models import TabChord, TimedChord
    from chord_parser.models import Chord


def chord_to_harte_string(chord: Chord | None) -> str:
    """Convert a chord_parser Chord to a Harte notation string.

    Parameters
    ----------
    chord : Chord | None
        The chord to convert, or None for no chord.

    Returns
    -------
    str
        Harte notation string (e.g., "C:maj", "G:min7"), or "N" for no chord.
    """
    if chord is None:
        return "N"
    return chord.to_harte()


def chord_to_decibel(chord: Chord | None) -> DECIBELChord | None:
    """Convert a chord_parser Chord to a DECIBEL Chord.

    Parameters
    ----------
    chord : Chord | None
        The chord to convert, or None for no chord.

    Returns
    -------
    DECIBELChord | None
        DECIBEL Chord object, or None for no chord.
    """
    if chord is None:
        return None
    harte_str = chord.to_harte()
    return DECIBELChord.from_harte_chord_string(harte_str)


def tabchord_to_chord_id(tab_chord: TabChord, alphabet: ChordAlphabet) -> int:
    """Convert a TabChord to a DECIBEL chord alphabet index.

    Parameters
    ----------
    tab_chord : TabChord
        The tab chord to convert.
    alphabet : ChordAlphabet
        The chord alphabet to index into.

    Returns
    -------
    int
        Index of the chord in the alphabet.
    """
    decibel_chord = chord_to_decibel(tab_chord.chord)
    return int(alphabet.get_index_of_chord_in_alphabet(decibel_chord))


def tabchords_to_alignment_arrays(
    tab_chords: list[TabChord],
    alphabet: ChordAlphabet,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of TabChords to DECIBEL alignment arrays.

    Extracts chord IDs and line boundary information needed for the
    jump alignment algorithm.

    Parameters
    ----------
    tab_chords : list[TabChord]
        List of tab chords with section context.
    alphabet : ChordAlphabet
        The chord alphabet to index into.

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray, np.ndarray]
        - nr_of_chords: Number of chords in the tab
        - chord_ids: Array of chord indices in the alphabet
        - is_first_in_line: Binary array indicating first chord in each section
        - is_last_in_line: Binary array indicating last chord in each section
    """
    nr_of_chords = len(tab_chords)
    if nr_of_chords == 0:
        return 0, np.array([]), np.array([]), np.array([])

    # Extract chord IDs
    chord_ids = np.zeros(nr_of_chords, dtype=int)
    for i, tc in enumerate(tab_chords):
        chord_ids[i] = tabchord_to_chord_id(tc, alphabet)

    # Determine line boundaries based on section changes
    is_first_in_line = np.zeros(nr_of_chords, dtype=int)
    is_last_in_line = np.zeros(nr_of_chords, dtype=int)

    is_first_in_line[0] = 1
    is_last_in_line[-1] = 1

    for i in range(1, nr_of_chords):
        if tab_chords[i].section != tab_chords[i - 1].section:
            is_first_in_line[i] = 1
            is_last_in_line[i - 1] = 1

    return nr_of_chords, chord_ids, is_first_in_line, is_last_in_line


def timedchord_to_chord_id(timed_chord: TimedChord, alphabet: ChordAlphabet) -> int:
    """Convert a TimedChord to a DECIBEL chord alphabet index.

    Parameters
    ----------
    timed_chord : TimedChord
        The timed chord to convert.
    alphabet : ChordAlphabet
        The chord alphabet to index into.

    Returns
    -------
    int
        Index of the chord in the alphabet.
    """
    decibel_chord = chord_to_decibel(timed_chord.chord)
    return int(alphabet.get_index_of_chord_in_alphabet(decibel_chord))
